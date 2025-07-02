import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.collators import VQACollator
from data.custom_dataset import CustomDataset, CustomMultiImageDataset
from data.processors import get_image_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def init_dist():
    """Initialize distributed training (CUDA only)."""
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(dist.get_rank())
        return True
    return False

def destroy_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(o):
    if not is_dist():
        return o
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all

def wrap_model(model):
    if is_dist():
        return DistributedDataParallel(model, device_ids=[dist.get_rank()])
    return model

def get_run_name(train_cfg, vlm_cfg):
    dataset_name = Path(train_cfg.custom_dataset_path).stem
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    device_info = f"{get_world_size()}x{get_device().type.upper()}"
    date = time.strftime("%m%d-%H%M%S")
    
    return f"nanoVLM_custom_{dataset_name}_{device_info}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"

def get_dataloaders(train_cfg, vlm_cfg):
    """Create dataloaders for custom dataset."""
    # Create processors
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    
    # Load custom dataset
    DatasetClass = CustomMultiImageDataset if train_cfg.multi_image else CustomDataset
    
    full_dataset = DatasetClass(
        json_path=train_cfg.custom_dataset_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=vlm_cfg.mp_image_token_length,
        image_root_dir=train_cfg.image_root_dir,
        max_length=vlm_cfg.lm_max_length
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * train_cfg.val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    # Create collator
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    
    g = torch.Generator()
    g.manual_seed(0)
    
    # Determine number of workers based on device
    device = get_device()
    num_workers = 0 if device.type == "mps" else 8  # MPS has issues with multiprocessing
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=vqa_collator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=vqa_collator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    return train_loader, val_loader

def get_lr(it, max_lr, max_steps):
    """Cosine learning rate schedule with warmup."""
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(train_cfg, vlm_cfg):
    device = get_device()
    
    if device.type == "mps":
        # Enable CPU fallback for MPS operations not yet implemented
        torch.mps.set_per_process_memory_fraction(0.0)  # Reset memory
        
    print(f"Using device: {device}")
    print(f"Device type: {device.type}")
    
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    
    run_name = get_run_name(train_cfg, vlm_cfg)
    total_dataset_size = len(train_loader.dataset)
    
    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            entity=train_cfg.wandb_entity,
            project="nanoVLM-custom",
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg),
                "device": device.type
            },
            name=run_name,
        )
    
    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    if is_master():
        print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Training summary: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")
        print(f"Validation summary: {len(val_loader.dataset)} samples, {len(val_loader)} batches/epoch")
    
    # Move model to device
    model.to(device)
    
    # Compile model if requested (skip for MPS as it may cause issues)
    if train_cfg.compile and device.type != "mps":
        model = torch.compile(model)
    
    # Wrap model for DDP if using distributed training
    model = wrap_model(model)
    
    # Define optimizer
    param_groups = [
        {'params': list(model.module.MP.parameters() if is_dist() else model.MP.parameters()), 
         'lr': train_cfg.lr_mp},
        {'params': list(model.module.decoder.parameters() if is_dist() else model.decoder.parameters()) + 
                   list(model.module.vision_encoder.parameters() if is_dist() else model.vision_encoder.parameters()), 
         'lr': train_cfg.lr_backbones}
    ]
    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    epoch = 0
    
    # Determine autocast settings based on device
    if device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.bfloat16
    
    # Create autocast context
    autocast_context = torch.autocast(device_type=device.type, dtype=autocast_dtype)
    
    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device) 
            attention_mask = batch["attention_mask"].to(device)
            
            # Use DDP no_sync context if applicable
            if (is_dist() and train_cfg.gradient_accumulation_steps > 1 and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()
            
            # Forward pass with autocast
            with autocast_context:
                with context:
                    _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
            
            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps
            
            loss.backward()
            
            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)
                
                # Update learning rates
                adj_lr_mp = get_lr(global_step, train_cfg.lr_mp, train_cfg.max_training_steps)
                adj_lr_backbones = get_lr(global_step, train_cfg.lr_backbones, train_cfg.max_training_steps)
                optimizer.param_groups[0]['lr'] = adj_lr_mp
                optimizer.param_groups[1]['lr'] = adj_lr_backbones
                
                optimizer.step()
                optimizer.zero_grad()
            
            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss
            
            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens
            
            # Evaluation
            if train_cfg.eval_interval > 0 and global_step % train_cfg.eval_interval == 0 and is_update_step:
                model.eval()
                
                # Clear cache if using CUDA/MPS
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                
                with torch.no_grad():
                    total_val_loss = 0
                    for val_batch in val_loader:
                        val_images = val_batch["images"]
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        
                        with autocast_context:
                            _, val_loss = model(val_input_ids, val_images, 
                                              attention_mask=val_attention_mask, 
                                              targets=val_labels)
                        
                        total_val_loss += val_loss.item()
                    
                    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    
                    if is_dist():
                        avg_val_loss = mean(dist_gather(avg_val_loss))
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            save_model = model.module if is_dist() else model
                            save_path = os.path.join(vlm_cfg.vlm_checkpoint_path, run_name)
                            save_model.save_pretrained(save_directory=save_path)
                            print(f"Saved best model to {save_path}")
                    
                    if is_master():
                        print(f"Step: {global_step}, Train Loss: {batch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                        if train_cfg.log_wandb:
                            wandb.log({
                                "train_loss": batch_loss,
                                "val_loss": avg_val_loss,
                                "lr_mp": adj_lr_mp,
                                "lr_backbones": adj_lr_backbones
                            }, step=global_step)
                
                model.train()
            
            if is_update_step:
                global_step += 1
                if global_step >= train_cfg.max_training_steps:
                    break
        
        avg_train_loss = total_train_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        epoch_tokens_per_second = total_tokens_processed / epoch_duration
        
        if is_master():
            print(f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, "
                  f"Avg Train Loss: {avg_train_loss:.4f}, Time: {epoch_duration:.2f}s, "
                  f"Tokens/s: {epoch_tokens_per_second:.2f}")
    
    # Training complete
    if is_master():
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {os.path.join(vlm_cfg.vlm_checkpoint_path, run_name)}")
        
        if train_cfg.log_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train NanoVLM on custom dataset")
    
    # Dataset arguments
    parser.add_argument('--custom_dataset_path', type=str, required=True,
                        help='Path to JSON file containing custom dataset')
    parser.add_argument('--image_root_dir', type=str, default=None,
                        help='Root directory for images (if paths in JSON are relative)')
    parser.add_argument('--multi_image', action='store_true',
                        help='Use multi-image dataset format')
    
    # Training arguments
    parser.add_argument('--lr_mp', type=float, default=0.00512,
                        help='Learning rate for the modality projector')
    parser.add_argument('--lr_backbones', type=float, default=5e-5,
                        help='Learning rate for the vision and language backbones')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max_training_steps', type=int, default=1000,
                        help='Maximum number of training steps')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval (0 to disable)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Model arguments
    parser.add_argument('--vlm_checkpoint_path', type=str, default='checkpoints',
                        help='Path to save/load VLM checkpoints')
    parser.add_argument('--resume_from_vlm_checkpoint', action='store_true',
                        help='Resume training from existing VLM checkpoint')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (not recommended for MPS)')
    
    # Logging arguments
    parser.add_argument('--log_wandb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    
    args = parser.parse_args()
    
    # Create configs
    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()
    
    # Update configs with arguments
    train_cfg.custom_dataset_path = args.custom_dataset_path
    train_cfg.image_root_dir = args.image_root_dir
    train_cfg.multi_image = args.multi_image
    train_cfg.lr_mp = args.lr_mp
    train_cfg.lr_backbones = args.lr_backbones
    train_cfg.batch_size = args.batch_size
    train_cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    train_cfg.max_training_steps = args.max_training_steps
    train_cfg.eval_interval = args.eval_interval
    train_cfg.val_ratio = args.val_ratio
    train_cfg.max_grad_norm = args.max_grad_norm
    train_cfg.log_wandb = args.log_wandb
    train_cfg.compile = args.compile
    train_cfg.resume_from_vlm_checkpoint = args.resume_from_vlm_checkpoint
    
    if args.vlm_checkpoint_path:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.wandb_entity:
        train_cfg.wandb_entity = args.wandb_entity
    
    # Initialize distributed training if available
    if torch.cuda.is_available():
        init_dist()
    
    if is_master():
        print("--- Custom Training Configuration ---")
        print(f"Dataset: {train_cfg.custom_dataset_path}")
        print(f"Image root: {train_cfg.image_root_dir}")
        print(f"Multi-image: {train_cfg.multi_image}")
        print(f"Device: {get_device()}")
        print(f"Batch size: {train_cfg.batch_size}")
        print(f"Gradient accumulation: {train_cfg.gradient_accumulation_steps}")
        print(f"Effective batch size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps * get_world_size()}")
        print(f"Max steps: {train_cfg.max_training_steps}")
        print("---")
    
    # Run training
    train(train_cfg, vlm_cfg)
    
    # Cleanup distributed training
    destroy_dist()

if __name__ == "__main__":
    main()