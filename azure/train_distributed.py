#!/usr/bin/env python3
"""
Distributed training wrapper for nanoVLM on Azure ML
Handles multi-GPU setup and Azure-specific configurations
"""

import os
import sys
import torch
import torch.distributed as dist
import argparse
from pathlib import Path

# Add parent directory to path to import nanoVLM modules
sys.path.append(str(Path(__file__).parent.parent))


def setup_azure_environment():
    """Setup Azure ML specific environment variables"""

    # Azure ML sets these environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Set PyTorch distributed environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Azure ML specific paths
    output_dir = os.environ.get("AZUREML_MODEL_DIR", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "output_dir": output_dir,
    }


def setup_distributed():
    """Initialize PyTorch distributed training"""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        # Set device
        torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Distributed nanoVLM training on Azure ML"
    )

    # Training arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="COCO",
        choices=["COCO", "VQAv2", "MIXED", "THE_CAULDRON", "custom"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--custom_dataset_path", type=str, help="Path to custom dataset JSON"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_training_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_projector", type=float, default=0.00512)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    # Model arguments
    parser.add_argument(
        "--vision_model_id", type=str, default="google/siglip-base-patch16-224"
    )
    parser.add_argument(
        "--language_model_id", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    parser.add_argument("--pixel_shuffle_factor", type=int, default=2)

    # Azure ML arguments
    parser.add_argument(
        "--output_dir", type=str, help="Output directory (overrides AZUREML_MODEL_DIR)"
    )
    parser.add_argument("--wandb_project", type=str, default="nanovlm-azure")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity/username")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push model to HuggingFace Hub"
    )
    parser.add_argument("--hub_model_id", type=str, help="HuggingFace model ID")

    args = parser.parse_args()

    # Setup Azure environment
    azure_config = setup_azure_environment()

    # Override output directory if specified
    if args.output_dir:
        azure_config["output_dir"] = args.output_dir

    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # Only print on main process
    if rank == 0:
        print(f"Starting distributed training with {world_size} GPUs")
        print(f"Output directory: {azure_config['output_dir']}")

    # Import and run training
    if args.dataset == "custom" and args.custom_dataset_path:
        from train_custom import main as train_main

        # Prepare arguments for train_custom.py
        train_args = [
            "--custom_dataset_path",
            args.custom_dataset_path,
            "--batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--max_training_steps",
            str(args.max_training_steps),
            "--eval_interval",
            str(args.eval_interval),
            "--output_dir",
            azure_config["output_dir"],
        ]

        if args.compile:
            train_args.append("--compile")

        if args.wandb_entity:
            train_args.extend(["--log_wandb", "--wandb_entity", args.wandb_entity])
            train_args.extend(["--wandb_project", args.wandb_project])

        if args.push_to_hub and args.hub_model_id:
            train_args.extend(["--push_to_hub", "--hub_model_id", args.hub_model_id])

        # Modify sys.argv for train_custom.py
        sys.argv = ["train_custom.py"] + train_args

    else:
        from train import main as train_main

        # Prepare arguments for train.py
        train_args = [
            "--dataset",
            args.dataset,
            "--batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--max_training_steps",
            str(args.max_training_steps),
            "--eval_interval",
            str(args.eval_interval),
            "--lr_backbone",
            str(args.lr_backbone),
            "--lr_projector",
            str(args.lr_projector),
            "--warmup_ratio",
            str(args.warmup_ratio),
            "--output_dir",
            azure_config["output_dir"],
        ]

        if args.compile:
            train_args.append("--compile")

        if args.wandb_entity:
            train_args.extend(["--log_wandb", "--wandb_entity", args.wandb_entity])
            train_args.extend(["--wandb_project", args.wandb_project])

        if args.push_to_hub and args.hub_model_id:
            train_args.extend(["--push_to_hub", "--hub_model_id", args.hub_model_id])

        # Modify sys.argv for train.py
        sys.argv = ["train.py"] + train_args

    try:
        # Run training
        train_main()
    finally:
        # Cleanup distributed training
        cleanup_distributed()

    if rank == 0:
        print(f"Training completed! Model saved to: {azure_config['output_dir']}")


if __name__ == "__main__":
    main()
