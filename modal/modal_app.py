#!/usr/bin/env python3
"""
NanoVLM training on Modal.com

This script sets up and runs NanoVLM training on Modal's cloud infrastructure.
"""

import os
import sys
import time
from typing import Optional

import modal

# Modal app configuration
app = modal.App("nanovlm-training")

# Create Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("modal/requirements.txt")
    .apt_install("git", "wget", "curl")
    .run_commands("pip install --upgrade pip")
    # Copy the entire nanoVLM codebase into the image
    .add_local_dir(".", "/root/nanovlm")
    # Copy test images if they exist
    .add_local_dir("test_images", "/root/nanovlm/test_images")
)

# Create persistent volume for datasets and checkpoints
volume = modal.Volume.from_name("nanovlm-data", create_if_missing=True)


# Secrets for API keys (optional - will be used if they exist)
def get_secrets():
    secrets = []
    try:
        secrets.append(modal.Secret.from_name("wandb-secret"))
    except:
        pass  # Secret doesn't exist, that's OK
    try:
        secrets.append(modal.Secret.from_name("huggingface-secret"))
    except:
        pass  # Secret doesn't exist, that's OK
    return secrets


secrets = get_secrets()


@app.function(
    image=image,
    gpu="A100-40GB",  # Single A100 GPU
    volumes={"/data": volume},
    secrets=secrets,
    timeout=3600 * 4,  # 4 hour timeout
    memory=32768,  # 32GB RAM
)
def train_nanovlm(
    custom_dataset_path: str = None,
    dataset_content: str = None,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    max_training_steps: int = 2000,
    eval_interval: int = 200,
    lr_mp: float = 0.00512,
    lr_backbones: float = 5e-5,
    output_dir: str = "/data/checkpoints",
    wandb_entity: Optional[str] = None,
    wandb_project: str = "nanovlm-modal",
    multi_image: bool = False,
    compile_model: bool = False,
    image_root_dir: Optional[str] = None,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_private: bool = False,
):
    """
    Train NanoVLM on Modal with custom dataset
    """
    # Set up Python path to use the copied codebase
    sys.path.insert(0, "/root/nanovlm")

    # Now import the training modules
    from models.config import VLMConfig, TrainConfig
    from models.vision_language_model import VisionLanguageModel
    from data.custom_dataset import CustomDataset, CustomMultiImageDataset
    from data.processors import get_image_processor, get_tokenizer
    from data.collators import VQACollator
    from torch.utils.data import DataLoader
    import torch
    import wandb
    from contextlib import nullcontext

    print("🚀 Starting NanoVLM training on Modal.com")
    print(
        f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Dataset: {custom_dataset_path}")
    print(
        f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})"
    )
    print(f"Training steps: {max_training_steps}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize configs
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig(
        lr_mp=lr_mp,
        lr_backbones=lr_backbones,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_training_steps=max_training_steps,
        eval_interval=eval_interval,
        compile=compile_model,
    )

    # Store additional config for our use
    train_cfg.custom_dataset_path = custom_dataset_path
    train_cfg.multi_image = multi_image
    train_cfg.image_root_dir = image_root_dir

    # Initialize W&B
    if wandb_entity:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config={
                "VLMConfig": vlm_cfg.__dict__,
                "TrainConfig": train_cfg.__dict__,
                "platform": "modal.com",
                "custom_dataset_path": custom_dataset_path,
                "multi_image": multi_image,
            },
            name=f"nanovlm-modal-{int(time.time())}",
        )

    # Create processors
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    # Load dataset
    if dataset_content:
        # Create dataset from content
        print("Loading dataset from provided content")
        import tempfile

        # Write dataset content to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(dataset_content)
            temp_dataset_path = f.name

        custom_dataset_path = temp_dataset_path
    else:
        print(f"Loading dataset from {custom_dataset_path}")

    DatasetClass = CustomMultiImageDataset if multi_image else CustomDataset

    dataset = DatasetClass(
        json_path=custom_dataset_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mp_image_token_length=vlm_cfg.mp_image_token_length,
        image_root_dir=image_root_dir,
        max_length=vlm_cfg.lm_max_length,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Create dataloader
    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize model
    print("Initializing NanoVLM model...")
    model = VisionLanguageModel(
        vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights
    )
    model = model.to(device)

    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Setup optimizer
    mp_params = list(model.MP.parameters())
    backbone_params = list(model.vision_encoder.parameters()) + list(
        model.decoder.parameters()
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": mp_params, "lr": lr_mp},
            {"params": backbone_params, "lr": lr_backbones},
        ]
    )

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0

    print("Starting training...")

    for epoch in range(100):  # Large number, will break based on max_training_steps
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_training_steps:
                break

            # Move batch to device
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            with nullcontext():
                logits, loss = model(
                    input_ids=input_ids,
                    images=images,
                    attention_mask=attention_mask,
                    targets=labels,
                )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = total_loss / gradient_accumulation_steps

                # Log metrics
                if wandb_entity:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/step": global_step,
                            "train/epoch": epoch,
                        }
                    )

                if global_step % 50 == 0:
                    print(
                        f"Step {global_step}/{max_training_steps}, Loss: {avg_loss:.4f}"
                    )

                # Save checkpoint
                if global_step % eval_interval == 0:
                    checkpoint_path = f"{output_dir}/nanovlm_step_{global_step}"
                    print(f"Saving checkpoint to {checkpoint_path}")
                    model.save_pretrained(checkpoint_path)

                total_loss = 0.0

                if global_step >= max_training_steps:
                    break

        if global_step >= max_training_steps:
            break

    # Save final model
    final_checkpoint_path = f"{output_dir}/nanovlm_final"
    print(f"Saving final model to {final_checkpoint_path}")
    model.save_pretrained(final_checkpoint_path)

    # Push to HuggingFace Hub if requested
    if push_to_hub and hub_model_id:
        print(f"🤗 Pushing model to HuggingFace Hub: {hub_model_id}")
        try:
            model.push_to_hub(hub_model_id, private=hub_private)
            print(
                f"✅ Model successfully pushed to https://huggingface.co/{hub_model_id}"
            )
        except Exception as e:
            print(f"❌ Failed to push to HuggingFace Hub: {e}")
            print("   Model is still saved locally and can be uploaded manually")

    if wandb_entity:
        wandb.finish()

    print("✅ Training completed successfully!")
    return final_checkpoint_path


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def upload_dataset(dataset_path: str, target_path: str = "/data/datasets"):
    """
    Upload a custom dataset to Modal volume
    """
    import shutil

    print(f"Uploading dataset from {dataset_path} to {target_path}")

    # Create target directory
    os.makedirs(target_path, exist_ok=True)

    # Copy dataset file
    if os.path.isfile(dataset_path):
        filename = os.path.basename(dataset_path)
        target_file = os.path.join(target_path, filename)
        shutil.copy2(dataset_path, target_file)
        print(f"Dataset uploaded to {target_file}")
        return target_file
    else:
        raise ValueError(f"Dataset file not found: {dataset_path}")


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def list_checkpoints():
    """
    List available model checkpoints
    """
    checkpoint_dir = "/data/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            d
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        print("Available checkpoints:")
        for checkpoint in sorted(checkpoints):
            print(f"  - {checkpoint}")
        return checkpoints
    else:
        print("No checkpoints found")
        return []


if __name__ == "__main__":
    # Example usage
    print(
        "NanoVLM Modal App - Use modal run modal_app.py::train_nanovlm to start training"
    )
