#!/usr/bin/env python3
"""
Enhanced NanoVLM training on Modal.com with advanced features:
- Checkpoint recovery & resumable training
- Cost tracking & budget limits
- Monitoring & alerts
- Dynamic batch sizing
- Performance optimization
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch

import modal

# Modal app configuration
app = modal.App("nanovlm-enhanced-training")

# Enhanced Modal image with optimizations
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("modal/requirements.txt")
    .apt_install("git", "wget", "curl", "htop")
    .run_commands("pip install --upgrade pip")
    # Install additional packages for monitoring
    .pip_install("psutil", "gpustat")
    # Set environment variables for optimization
    .env({"PYTHONPATH": "/root/nanovlm", "TOKENIZERS_PARALLELISM": "false"})
    # Copy the entire nanoVLM codebase - MUST BE LAST
    .add_local_dir(".", "/root/nanovlm")
)

# Enhanced persistent volume
volume = modal.Volume.from_name("nanovlm-enhanced-data", create_if_missing=True)


def get_secrets():
    """Get available secrets for API keys"""
    secrets = []
    try:
        secrets.append(modal.Secret.from_name("wandb-secret"))
    except:
        pass
    try:
        secrets.append(modal.Secret.from_name("huggingface-secret"))
    except:
        pass
    return secrets


secrets = get_secrets()


def find_optimal_batch_size(model, sample_batch, device, max_batch_size=32):
    """Automatically find the largest batch size that fits in memory"""
    print("🔍 Finding optimal batch size...")

    batch_size = max_batch_size
    while batch_size > 0:
        try:
            # Test forward pass
            test_images = sample_batch["images"][:batch_size]
            test_input_ids = sample_batch["input_ids"][:batch_size].to(device)
            test_attention_mask = sample_batch["attention_mask"][:batch_size].to(device)
            test_labels = sample_batch["labels"][:batch_size].to(device)

            with torch.cuda.amp.autocast():
                _ = model(
                    test_input_ids,
                    test_images,
                    attention_mask=test_attention_mask,
                    targets=test_labels,
                )

            torch.cuda.empty_cache()
            print(f"✅ Optimal batch size found: {batch_size}")
            return batch_size

        except torch.cuda.OutOfMemoryError:
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            print(f"⚠️ Batch size {batch_size * 2} too large, trying {batch_size}")

    print("❌ Could not find suitable batch size, using 1")
    return 1


def save_checkpoint(
    model, optimizer, global_step, checkpoint_dir: Path, metadata: Dict[str, Any]
):
    """Save training checkpoint with metadata"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}.pt"

    # Unwrap model if using DDP
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
        "timestamp": time.time(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3):
    """Remove old checkpoints to save storage space"""
    if not checkpoint_dir.exists():
        return

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*.pt"), key=lambda x: int(x.stem.split("-")[1])
    )

    if len(checkpoints) > keep_last_n:
        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint.unlink()
            print(f"🗑️ Removed old checkpoint: {checkpoint}")


def load_latest_checkpoint(checkpoint_dir: Path):
    """Load the most recent checkpoint"""
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*.pt"), key=lambda x: int(x.stem.split("-")[1])
    )

    if not checkpoints:
        return None

    latest_checkpoint = checkpoints[-1]
    print(f"🔄 Loading checkpoint: {latest_checkpoint}")
    return torch.load(latest_checkpoint, map_location="cpu")


class CostTracker:
    """Track training costs in real-time"""

    def __init__(self, gpu_type: str = "A100-40GB", use_spot: bool = False):
        # Approximate pricing (USD per hour)
        self.pricing = {
            "A100-40GB": 0.75 if use_spot else 2.49,
            "A100-80GB": 1.20 if use_spot else 3.99,
            "H100": 2.00 if use_spot else 6.49,
        }
        self.gpu_cost_per_hour = self.pricing.get(gpu_type, 2.49)
        self.start_time = time.time()
        self.use_spot = use_spot

    def get_current_cost(self) -> float:
        """Get current estimated cost"""
        hours = (time.time() - self.start_time) / 3600
        return hours * self.gpu_cost_per_hour

    def get_cost_per_step(self, current_step: int) -> float:
        """Get cost per training step"""
        if current_step == 0:
            return 0
        return self.get_current_cost() / current_step

    def estimate_total_cost(self, current_step: int, max_steps: int) -> float:
        """Estimate total training cost"""
        if current_step == 0:
            return 0
        cost_per_step = self.get_cost_per_step(current_step)
        return cost_per_step * max_steps


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    secrets=secrets,
    timeout=3600 * 12,  # 12 hour timeout
    memory=64 * 1024,  # 64GB RAM
    min_containers=1,  # Keep one instance warm
)
def enhanced_train(
    # Dataset parameters
    dataset_type: str = "mixed",
    dataset_limit: int = 10000,
    dataset_split: str = "train",
    # Training parameters
    batch_size: Optional[int] = None,  # Auto-detect if None
    gradient_accumulation_steps: int = 4,
    max_training_steps: int = 2000,
    eval_interval: int = 200,
    lr_mp: float = 0.00512,
    lr_backbones: float = 5e-5,
    # Enhanced features
    resume_from_checkpoint: Optional[str] = None,
    checkpoint_interval: int = 500,
    keep_last_n_checkpoints: int = 3,
    budget_limit: Optional[float] = None,  # USD
    use_spot_pricing: bool = True,
    auto_batch_size: bool = True,
    enable_compile: bool = True,
    # Output parameters
    output_dir: str = "/data/checkpoints",
    wandb_entity: Optional[str] = None,
    wandb_project: str = "nanovlm-enhanced",
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    # Alert parameters
    alert_on_collapse: bool = True,
    pause_on_collapse: bool = False,
):
    """Enhanced training with all improvements"""

    try:
        # Set up environment
        sys.path.insert(0, "/root/nanovlm")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("🚀 Enhanced NanoVLM Training Starting")
        print(
            f"💻 Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
        )
        print(f"📊 Dataset: {dataset_type} ({dataset_limit} samples)")
        print(
            f"💰 Budget limit: ${budget_limit}"
            if budget_limit
            else "💰 No budget limit"
        )

        # Initialize cost tracker
        cost_tracker = CostTracker(use_spot=use_spot_pricing)

        # Import required modules
        from models.config import VLMConfig, TrainConfig
        from models.vision_language_model import VisionLanguageModel
        from data.custom_dataset import CustomDataset, CustomMultiImageDataset
        from data.processors import get_image_processor, get_tokenizer
        from data.collators import VQACollator
        from torch.utils.data import DataLoader
        from utils.training_logger import detect_model_collapse
        import wandb

        # Setup directories
        output_path = Path(output_dir)
        checkpoint_dir = output_path / "checkpoints"
        output_path.mkdir(parents=True, exist_ok=True)

        # Build dataset (reusing existing function from modal_app_fixed.py)
        dataset_path = build_public_dataset_on_modal(
            dataset_type=dataset_type, limit=dataset_limit, split=dataset_split
        )
        dataset_dir = Path(dataset_path).parent

        # Initialize configs
        vlm_cfg = VLMConfig()
        train_cfg = TrainConfig(
            lr_mp=lr_mp,
            lr_backbones=lr_backbones,
            batch_size=batch_size or 8,  # Default if not auto-detected
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_training_steps=max_training_steps,
            eval_interval=eval_interval,
            compile=enable_compile,
        )

        # Initialize wandb with enhanced metadata
        run_metadata = {
            "platform": "modal.com",
            "enhanced_features": True,
            "auto_batch_size": auto_batch_size,
            "budget_limit": budget_limit,
            "use_spot_pricing": use_spot_pricing,
            "dataset_type": dataset_type,
            "dataset_limit": dataset_limit,
        }

        wandb_run = None
        if wandb_entity:
            wandb_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                config={**vlm_cfg.__dict__, **train_cfg.__dict__, **run_metadata},
                name=f"enhanced-{dataset_type}-{int(time.time())}",
            )

        # Create processors and dataset
        image_processor = get_image_processor(vlm_cfg.vit_img_size)
        tokenizer = get_tokenizer(
            vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
        )

        DatasetClass = (
            CustomMultiImageDataset if "multi" in dataset_type else CustomDataset
        )
        dataset = DatasetClass(
            json_path=dataset_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            mp_image_token_length=vlm_cfg.mp_image_token_length,
            image_root_dir=str(dataset_dir),
            max_length=vlm_cfg.lm_max_length,
        )

        print(f"📚 Dataset loaded: {len(dataset)} samples")

        # Initialize model
        model = VisionLanguageModel(vlm_cfg, load_backbone=True).to(device)

        # Auto-detect optimal batch size if requested
        if auto_batch_size and batch_size is None:
            collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
            sample_loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
            sample_batch = next(iter(sample_loader))

            optimal_batch_size = find_optimal_batch_size(model, sample_batch, device)
            train_cfg.batch_size = optimal_batch_size
            print(f"🎯 Using auto-detected batch size: {optimal_batch_size}")

        # Create final dataloader
        collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # Setup optimizer
        param_groups = [
            {"params": list(model.MP.parameters()), "lr": lr_mp},
            {
                "params": list(model.decoder.parameters())
                + list(model.vision_encoder.parameters()),
                "lr": lr_backbones,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups)

        # Compile model if requested
        if enable_compile:
            try:
                model = torch.compile(model)
                print("✅ Model compiled for optimization")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")

        # Load checkpoint if available
        start_step = 0
        if resume_from_checkpoint == "auto":
            checkpoint = load_latest_checkpoint(checkpoint_dir)
        elif resume_from_checkpoint:
            checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        else:
            checkpoint = None

        if checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["global_step"]
            print(f"✅ Resumed from step {start_step}")

        # Training loop with enhanced monitoring
        model.train()
        global_step = start_step
        best_val_loss = float("inf")
        last_collapse_check = 0

        print(f"🏃 Starting training from step {start_step}")

        for epoch in range(100):  # Large number, will break on max_steps
            for batch_idx, batch in enumerate(dataloader):
                if global_step >= max_training_steps:
                    break

                # Check budget limit
                current_cost = cost_tracker.get_current_cost()
                if budget_limit and current_cost >= budget_limit:
                    print(f"💰 Budget limit reached: ${current_cost:.2f}")

                    # Save final checkpoint
                    save_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        checkpoint_dir,
                        {
                            "reason": "budget_limit_reached",
                            "cost": current_cost,
                            "steps_completed": global_step,
                        },
                    )

                    if wandb_run:
                        wandb.log(
                            {
                                "training/budget_limit_reached": True,
                                "training/final_cost": current_cost,
                            }
                        )
                        wandb.finish()

                    return {
                        "status": "budget_limit_reached",
                        "cost": current_cost,
                        "steps_completed": global_step,
                    }

                # Prepare batch
                images = batch["images"]
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass
                with torch.cuda.amp.autocast():
                    logits, loss = model(
                        input_ids, images, attention_mask=attention_mask, targets=labels
                    )

                # Backward pass
                loss.backward()

                # Update every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Enhanced logging with cost tracking
                    if wandb_run and global_step % 10 == 0:
                        cost_per_step = cost_tracker.get_cost_per_step(global_step)
                        estimated_total = cost_tracker.estimate_total_cost(
                            global_step, max_training_steps
                        )

                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/step": global_step,
                                "cost/current_usd": current_cost,
                                "cost/per_step_usd": cost_per_step,
                                "cost/estimated_total_usd": estimated_total,
                                "system/gpu_memory_gb": torch.cuda.memory_allocated()
                                / 1e9,
                            },
                            step=global_step,
                        )

                    # Checkpoint saving
                    if global_step % checkpoint_interval == 0:
                        checkpoint_metadata = {
                            "cost": current_cost,
                            "steps_per_hour": global_step
                            / ((time.time() - cost_tracker.start_time) / 3600),
                            "loss": loss.item(),
                        }
                        save_checkpoint(
                            model,
                            optimizer,
                            global_step,
                            checkpoint_dir,
                            checkpoint_metadata,
                        )
                        cleanup_old_checkpoints(checkpoint_dir, keep_last_n_checkpoints)

                    # Collapse detection with alerting
                    if alert_on_collapse and global_step - last_collapse_check >= 100:
                        try:
                            # Create a small validation set for testing
                            val_samples = [
                                dataset[i] for i in range(min(5, len(dataset)))
                            ]

                            collapse_stats = detect_model_collapse(
                                model,
                                val_samples,
                                tokenizer,
                                image_processor,
                                device,
                                global_step,
                            )

                            if collapse_stats.get("collapsed", False):
                                alert_msg = f"⚠️ MODEL COLLAPSE DETECTED at step {global_step}: {collapse_stats.get('reason', 'unknown')}"
                                print(alert_msg)
                                print(
                                    f"Most common token: {collapse_stats.get('most_common_token', 'unknown')}"
                                )

                                if wandb_run:
                                    wandb.log(
                                        {
                                            "alert/model_collapsed": True,
                                            "alert/collapse_reason": collapse_stats.get(
                                                "reason", "unknown"
                                            ),
                                            "alert/step": global_step,
                                        }
                                    )

                                if pause_on_collapse:
                                    print(
                                        "🛑 Training paused due to collapse detection"
                                    )
                                    save_checkpoint(
                                        model,
                                        optimizer,
                                        global_step,
                                        checkpoint_dir,
                                        {
                                            "reason": "model_collapse",
                                            "collapse_stats": collapse_stats,
                                        },
                                    )

                                    if wandb_run:
                                        wandb.finish()

                                    return {
                                        "status": "paused_collapse",
                                        "reason": collapse_stats.get("reason"),
                                        "step": global_step,
                                    }

                            last_collapse_check = global_step

                        except Exception as e:
                            print(f"Error in collapse detection: {e}")

                    # Progress updates
                    if global_step % 100 == 0:
                        progress = (global_step / max_training_steps) * 100
                        eta_hours = (
                            (max_training_steps - global_step)
                            * cost_per_step
                            / cost_tracker.gpu_cost_per_hour
                        )

                        print(
                            f"📊 Step {global_step}/{max_training_steps} ({progress:.1f}%) | "
                            f"Loss: {loss.item():.4f} | "
                            f"Cost: ${current_cost:.2f} | "
                            f"ETA: {eta_hours:.1f}h"
                        )

                if global_step >= max_training_steps:
                    break

            if global_step >= max_training_steps:
                break

        # Final save
        final_cost = cost_tracker.get_current_cost()
        save_checkpoint(
            model,
            optimizer,
            global_step,
            checkpoint_dir,
            {
                "reason": "training_complete",
                "final_cost": final_cost,
                "total_steps": global_step,
            },
        )

        # Push to hub if requested
        if push_to_hub and hub_model_id:
            try:
                model.save_pretrained(output_path / "final_model")
                print(f"🤗 Model saved locally: {output_path / 'final_model'}")
                # Note: Actual hub push would require additional setup
            except Exception as e:
                print(f"⚠️ Hub push failed: {e}")

        if wandb_run:
            wandb.log(
                {
                    "training/completed": True,
                    "training/final_cost": final_cost,
                    "training/total_steps": global_step,
                }
            )
            wandb.finish()

        print(f"✅ Training completed! Total cost: ${final_cost:.2f}")

        return {
            "status": "completed",
            "total_cost": final_cost,
            "steps_completed": global_step,
            "model_path": str(output_path / "final_model"),
        }

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()

        if wandb_run:
            wandb.log({"error": str(e)})
            wandb.finish()

        return {"status": "failed", "error": str(e)}


def build_public_dataset_on_modal(dataset_type: str, limit: int, split: str = "train"):
    """
    Build public dataset directly on Modal infrastructure
    """
    from pathlib import Path

    print(f"🔄 Loading {dataset_type} dataset...")

    # Create dataset directory
    dataset_dir = Path("/data/datasets")
    dataset_dir.mkdir(exist_ok=True)

    image_dir = dataset_dir / "images"
    image_dir.mkdir(exist_ok=True)

    if dataset_type == "mixed":
        return build_mixed_dataset_on_modal(limit, dataset_dir)
    elif dataset_type == "coco":
        return build_coco_dataset_on_modal(limit, split, dataset_dir)
    elif dataset_type == "vqav2":
        return build_vqav2_dataset_on_modal(limit, dataset_dir)
    elif dataset_type == "llava":
        return build_llava_dataset_on_modal(limit, dataset_dir)
    else:
        # Create a simple test dataset for testing
        dataset_path = dataset_dir / f"{dataset_type}_{limit}_{split}.json"
        if not dataset_path.exists():
            test_data = [
                {
                    "image_path": "test_image.jpg",
                    "conversations": [
                        {"role": "user", "content": "What do you see?"},
                        {"role": "assistant", "content": "I see a test image."},
                    ],
                }
            ] * min(
                limit, 10
            )  # Small test dataset

            with open(dataset_path, "w") as f:
                json.dump(test_data, f)

        return str(dataset_path)


def build_coco_dataset_on_modal(limit: int, split: str, dataset_dir):
    """Build COCO dataset on Modal"""
    from datasets import load_dataset
    from tqdm import tqdm
    import json
    from pathlib import Path

    # Convert to Path object if needed
    dataset_dir = Path(dataset_dir)

    print(f"Loading COCO Captions {split} dataset...")
    dataset = load_dataset("HuggingFaceM4/COCO", split=split, trust_remote_code=True)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    converted_data = []
    image_dir = dataset_dir / "images"

    print("Converting COCO to NanoVLM format...")
    for idx, item in enumerate(tqdm(dataset)):
        try:
            captions = item["sentences"]["raw"]
            if not captions:
                continue

            caption = captions[0]
            image_filename = f"coco_{split}_{item['cocoid']}.jpg"
            image_path = image_dir / image_filename

            # Save image
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path, "JPEG", quality=90)

            converted_item = {
                "image_path": str(image_path.relative_to(dataset_dir)),
                "conversations": [
                    {"role": "user", "content": "Describe this image."},
                    {"role": "assistant", "content": caption},
                ],
            }
            converted_data.append(converted_item)

        except Exception as e:
            print(f"Error processing COCO item {idx}: {e}")
            continue

    # Save dataset
    output_path = dataset_dir / "coco_dataset.json"
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)

    print(f"COCO dataset saved: {len(converted_data)} samples")
    return str(output_path)


def build_vqav2_dataset_on_modal(limit: int, dataset_dir):
    """Build VQAv2 dataset on Modal"""
    from datasets import load_dataset
    from tqdm import tqdm
    import json
    from pathlib import Path

    # Convert to Path object if needed
    dataset_dir = Path(dataset_dir)

    print("Loading VQAv2 dataset...")
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", trust_remote_code=True)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    converted_data = []
    image_dir = dataset_dir / "images"

    print("Converting VQAv2 to NanoVLM format...")
    for idx, item in enumerate(tqdm(dataset)):
        try:
            if not item["answers"]:
                continue

            answers = item["answers"]["answer"]
            answer = max(set(answers), key=answers.count)

            image_filename = f"vqa_{item['image_id']}.jpg"
            image_path = image_dir / image_filename

            # Save image
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path, "JPEG", quality=90)

            converted_item = {
                "image_path": str(image_path.relative_to(dataset_dir)),
                "conversations": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": answer},
                ],
            }
            converted_data.append(converted_item)

        except Exception as e:
            print(f"Error processing VQA item {idx}: {e}")
            continue

    # Save dataset
    output_path = dataset_dir / "vqav2_dataset.json"
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)

    print(f"VQAv2 dataset saved: {len(converted_data)} samples")
    return str(output_path)


def build_mixed_dataset_on_modal(limit: int, dataset_dir):
    """Build mixed COCO + VQAv2 dataset on Modal"""
    import json
    import random
    from pathlib import Path

    # Convert to Path object if needed
    dataset_dir = Path(dataset_dir)

    print("Building mixed dataset (COCO + VQAv2)...")

    # Build individual datasets
    coco_limit = limit // 2
    vqa_limit = limit // 2

    coco_path = build_coco_dataset_on_modal(coco_limit, "train", dataset_dir)
    vqa_path = build_vqav2_dataset_on_modal(vqa_limit, dataset_dir)

    # Combine datasets
    combined_data = []

    # Load COCO data
    with open(coco_path, "r") as f:
        coco_data = json.load(f)
        combined_data.extend(coco_data)

    # Load VQA data
    with open(vqa_path, "r") as f:
        vqa_data = json.load(f)
        combined_data.extend(vqa_data)

    # Shuffle combined data
    random.shuffle(combined_data)

    # Save mixed dataset
    output_path = dataset_dir / "mixed_dataset.json"
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Mixed dataset saved: {len(combined_data)} samples")
    return str(output_path)


def build_llava_dataset_on_modal(limit: int, dataset_dir):
    """Build LLaVA-style instruction dataset on Modal"""
    from datasets import load_dataset
    from tqdm import tqdm
    import json
    from pathlib import Path

    # Convert to Path object if needed
    dataset_dir = Path(dataset_dir)

    print("Building LLaVA-style instruction dataset...")

    # Use VQA as base and convert to instruction format
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", trust_remote_code=True)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    converted_data = []
    image_dir = dataset_dir / "images"

    print("Converting to LLaVA instruction format...")
    for idx, item in enumerate(tqdm(dataset)):
        try:
            if not item["answers"]:
                continue

            answers = item["answers"]["answer"]
            answer = max(set(answers), key=answers.count)

            image_filename = f"llava_{item['image_id']}.jpg"
            image_path = image_dir / image_filename

            # Save image
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path, "JPEG", quality=90)

            # Create instruction-following format
            instruction_prompt = f"Please examine the image carefully and answer the following question: {item['question']}"

            converted_item = {
                "image_path": str(image_path.relative_to(dataset_dir)),
                "conversations": [
                    {"role": "user", "content": instruction_prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
            converted_data.append(converted_item)

        except Exception as e:
            print(f"Error processing LLaVA item {idx}: {e}")
            continue

    # Save dataset
    output_path = dataset_dir / "llava_dataset.json"
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)

    print(f"LLaVA dataset saved: {len(converted_data)} samples")
    return str(output_path)


@app.local_entrypoint()
def main(
    dataset_type: str = "mixed",
    dataset_limit: int = 1000,
    max_training_steps: int = 500,
    budget_limit: float = 5.0,
    wandb_entity: str = None,
    auto_batch_size: bool = True,
    test_run: bool = True,
):
    """Local entrypoint to run enhanced training"""

    print("🚀 Launching Enhanced NanoVLM Training on Modal")

    if test_run:
        print("🧪 Running in test mode with small dataset")
        dataset_limit = min(dataset_limit, 100)
        max_training_steps = min(max_training_steps, 50)
        budget_limit = min(budget_limit, 2.0)

    # Launch the enhanced training job
    result = enhanced_train.remote(
        dataset_type=dataset_type,
        dataset_limit=dataset_limit,
        max_training_steps=max_training_steps,
        budget_limit=budget_limit,
        wandb_entity=wandb_entity,
        auto_batch_size=auto_batch_size,
        alert_on_collapse=True,
        checkpoint_interval=25,  # More frequent for testing
        use_spot_pricing=True,
        enable_compile=False,  # Disable for faster startup in test
    )

    print(f"📋 Training Result: {result}")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Parse basic arguments manually
        dataset_limit = 1000
        max_training_steps = 500
        budget_limit = 5.0
        test_run = False

        for arg in sys.argv[1:]:
            if arg.startswith("--dataset_limit="):
                dataset_limit = int(arg.split("=")[1])
            elif arg.startswith("--max_training_steps="):
                max_training_steps = int(arg.split("=")[1])
            elif arg.startswith("--budget_limit="):
                budget_limit = float(arg.split("=")[1])
            elif arg == "--test_run":
                test_run = True

        main(
            dataset_limit=dataset_limit,
            max_training_steps=max_training_steps,
            budget_limit=budget_limit,
            test_run=test_run,
        )
    else:
        main()
