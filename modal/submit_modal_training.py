#!/usr/bin/env python3
"""
Submit NanoVLM training job to Modal.com

This script provides a convenient interface to submit training jobs to Modal
with various configuration options.
"""

import argparse
import json
import os
import sys


def validate_dataset(dataset_path: str) -> bool:
    """Validate that the dataset file exists and has correct format"""
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("❌ Dataset must be a JSON list")
            return False

        if len(data) == 0:
            print("❌ Dataset is empty")
            return False

        # Check first item structure
        first_item = data[0]
        required_keys = ["conversations"]

        if "image_path" in first_item:
            print("✅ Single-image dataset format detected")
        elif "image_paths" in first_item:
            print("✅ Multi-image dataset format detected")
        else:
            print("❌ Dataset must have either 'image_path' or 'image_paths' field")
            return False

        if "conversations" not in first_item:
            print("❌ Dataset must have 'conversations' field")
            return False

        print(f"✅ Dataset validated: {len(data)} samples")
        return True

    except json.JSONDecodeError:
        print("❌ Invalid JSON format")
        return False
    except Exception as e:
        print(f"❌ Dataset validation error: {e}")
        return False


def submit_dataset_build_and_training_job(args):
    """Submit integrated dataset building + training job to Modal"""

    # Validate HuggingFace options
    if args.push_to_hub and not args.hub_model_id:
        print("❌ --hub_model_id is required when using --push_to_hub")
        print("   Example: --hub_model_id pgryko/my-nanovlm-model")
        return False

    # Check if Modal is available
    try:
        import modal
    except ImportError:
        print("❌ Modal not installed. Install with: uv add modal")
        return False

    # Import the Modal app
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from modal_app import app, build_dataset_and_train

    print("🚀 Submitting integrated dataset building + training job to Modal.com")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Dataset limit: {args.dataset_limit}")
    print(
        f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})"
    )
    print(f"Training Steps: {args.max_training_steps}")
    print(
        f"Expected Duration: ~{(args.dataset_limit // 1000 * 2) + (args.max_training_steps // 100)} minutes"
    )

    # Prepare training arguments
    training_kwargs = {
        "dataset_type": args.dataset_type,
        "dataset_limit": args.dataset_limit,
        "dataset_split": args.dataset_split,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_training_steps": args.max_training_steps,
        "eval_interval": args.eval_interval,
        "lr_mp": args.lr_mp,
        "lr_backbones": args.lr_backbones,
        "wandb_project": args.wandb_project,
        "multi_image": args.multi_image,
        "compile_model": args.compile,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "hub_private": args.hub_private,
    }

    if args.wandb_entity:
        training_kwargs["wandb_entity"] = args.wandb_entity

    # Submit integrated job
    try:
        print("🎯 Starting dataset building + training on Modal...")
        with app.run():
            result = build_dataset_and_train.remote(**training_kwargs)
            print(f"✅ Training completed! Model saved to: {result}")
            return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def submit_training_job(args):
    """Submit training job to Modal with existing dataset"""

    # Validate dataset
    if not validate_dataset(args.custom_dataset_path):
        return False

    # Validate HuggingFace options
    if args.push_to_hub and not args.hub_model_id:
        print("❌ --hub_model_id is required when using --push_to_hub")
        print("   Example: --hub_model_id pgryko/my-nanovlm-model")
        return False

    # Check if Modal is available
    try:
        import modal
    except ImportError:
        print("❌ Modal not installed. Install with: uv add modal")
        return False

    # Import the Modal app
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from modal_app import app, train_nanovlm

    print("🚀 Submitting NanoVLM training job to Modal.com")
    print(f"Dataset: {args.custom_dataset_path}")
    print(
        f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})"
    )
    print(f"Training Steps: {args.max_training_steps}")
    print(f"Expected Duration: ~{args.max_training_steps // 100} minutes")

    # For now, we'll pass the dataset content directly to avoid upload issues
    # In a production setup, you'd want to upload large datasets to Modal volumes
    dataset_path = args.custom_dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False

    # Read dataset content to pass to Modal
    with open(dataset_path, "r") as f:
        dataset_content = f.read()

    print("📤 Preparing dataset for Modal training...")

    # Prepare training arguments
    training_kwargs = {
        "dataset_content": dataset_content,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_training_steps": args.max_training_steps,
        "eval_interval": args.eval_interval,
        "lr_mp": args.lr_mp,
        "lr_backbones": args.lr_backbones,
        "wandb_project": args.wandb_project,
        "multi_image": args.multi_image,
        "compile_model": args.compile,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "hub_private": args.hub_private,
    }

    if args.wandb_entity:
        training_kwargs["wandb_entity"] = args.wandb_entity

    if args.image_root_dir:
        training_kwargs["image_root_dir"] = args.image_root_dir

    # Submit training job
    try:
        print("🎯 Starting training on Modal...")
        with app.run():
            result = train_nanovlm.remote(**training_kwargs)
            print(f"✅ Training completed! Model saved to: {result}")
            return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit NanoVLM training job to Modal.com",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--build_dataset",
        action="store_true",
        help="Build dataset on Modal (instead of using existing dataset)",
    )

    # Dataset building arguments (for --build_dataset mode)
    parser.add_argument(
        "--dataset_type",
        choices=["mixed", "coco", "vqav2", "llava"],
        default="mixed",
        help="Type of public dataset to build (only used with --build_dataset)",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=10000,
        help="Number of samples to include in dataset (only used with --build_dataset)",
    )
    parser.add_argument(
        "--dataset_split",
        default="train",
        help="Dataset split to use (only used with --build_dataset)",
    )

    # Existing dataset arguments (for regular mode)
    parser.add_argument(
        "--custom_dataset_path",
        type=str,
        help="Path to JSON file containing custom dataset (required if not using --build_dataset)",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        help="Root directory for images (if paths in JSON are relative)",
    )
    parser.add_argument(
        "--multi_image", action="store_true", help="Use multi-image dataset format"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_training_steps",
        type=int,
        default=2000,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=200,
        help="Evaluation and checkpoint saving interval",
    )
    parser.add_argument(
        "--lr_mp",
        type=float,
        default=0.00512,
        help="Learning rate for the modality projector",
    )
    parser.add_argument(
        "--lr_backbones",
        type=float,
        default=5e-5,
        help="Learning rate for vision and language backbones",
    )

    # Model options
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for model optimization",
    )

    # HuggingFace options
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HuggingFace model ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--hub_private", action="store_true", help="Make HuggingFace repository private"
    )

    # Logging options
    parser.add_argument(
        "--wandb_entity", type=str, help="Weights & Biases entity (username or team)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nanovlm-modal",
        help="Weights & Biases project name",
    )

    # Utility commands
    parser.add_argument(
        "--list_checkpoints",
        action="store_true",
        help="List available model checkpoints",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_checkpoints:
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from modal_app import app, list_checkpoints

            with app.run():
                list_checkpoints.remote()
        except Exception as e:
            print(f"❌ Failed to list checkpoints: {e}")
        return

    # Validate arguments based on mode
    if args.build_dataset:
        print("🏗️  Using integrated dataset building mode")
        success = submit_dataset_build_and_training_job(args)
    else:
        if not args.custom_dataset_path:
            print("❌ --custom_dataset_path is required when not using --build_dataset")
            print(
                "   Use --build_dataset to build dataset on Modal, or provide --custom_dataset_path"
            )
            sys.exit(1)
        print("📁 Using existing dataset mode")
        success = submit_training_job(args)

    if not success:
        sys.exit(1)

    print("\n🎉 Training job submitted successfully!")
    print("\n📊 Monitor your training:")
    if args.wandb_entity:
        print(f"   W&B: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
    print("   Modal: https://modal.com/apps")

    print("\n📁 Access your trained model:")
    print("   Run: python modal/submit_modal_training.py --list_checkpoints")


if __name__ == "__main__":
    main()
