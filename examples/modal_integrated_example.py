#!/usr/bin/env python3
"""
Example: Integrated dataset building + training on Modal.com

This example shows how to build a dataset and train NanoVLM in a single Modal job,
exactly as requested by the user.
"""

import subprocess
import sys
import os


def run_integrated_training():
    """
    Run the integrated dataset building + training pipeline on Modal.com

    This replaces the need to run:
    1. python examples/simple_public_datasets.py --dataset mixed --output datasets/mixed_train.json --limit 10000
    2. python modal/submit_modal_training.py --custom_dataset_path datasets/mixed_train.json ...

    Instead, everything happens in a single Modal job.
    """

    print("🚀 Starting integrated dataset building + training on Modal.com")
    print("=" * 60)

    # Configuration matching the user's request
    config = {
        "dataset_type": "mixed",  # COCO + VQAv2 combination
        "dataset_limit": 10000,  # Same as --limit 10000
        "batch_size": 8,
        "max_training_steps": 3000,  # Adjusted for 10K dataset
        "wandb_entity": "piotr-gryko-devalogic",
        "hub_model_id": "pgryko/nanovlm-COCO-VQAv2",
        "compile": True,  # Enable torch.compile for faster training
        "push_to_hub": True,
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Build the command
    cmd = [
        "uv",
        "run",
        "python",
        "modal/submit_modal_training.py",
        "--build_dataset",
        "--dataset_type",
        config["dataset_type"],
        "--dataset_limit",
        str(config["dataset_limit"]),
        "--batch_size",
        str(config["batch_size"]),
        "--max_training_steps",
        str(config["max_training_steps"]),
        "--wandb_entity",
        config["wandb_entity"],
        "--hub_model_id",
        config["hub_model_id"],
    ]

    if config["compile"]:
        cmd.append("--compile")

    if config["push_to_hub"]:
        cmd.append("--push_to_hub")

    print("Executing command:")
    print(" ".join(cmd))
    print()

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False


def show_monitoring_info():
    """Show information about monitoring the training"""
    print("\n📊 Monitor your training:")
    print("   W&B: https://wandb.ai/piotr-gryko-devalogic/nanovlm-modal")
    print("   Modal: https://modal.com/apps")
    print("\n🤗 Your model will be available at:")
    print("   https://huggingface.co/pgryko/nanovlm-COCO-VQAv2")
    print("\n💡 Tips:")
    print("   - Training will take approximately 45-60 minutes")
    print("   - Dataset building adds ~10-15 minutes")
    print("   - Total cost estimate: $6-8 on Modal A100")


def main():
    """Main function"""
    print("NanoVLM Integrated Training Example")
    print("This example demonstrates the integrated approach requested:")
    print("- Build mixed dataset (COCO + VQAv2) on Modal")
    print("- Train NanoVLM with the built dataset")
    print("- Push trained model to HuggingFace Hub")
    print()

    # Check prerequisites
    try:
        import modal

        print("✅ Modal is installed")
    except ImportError:
        print("❌ Modal not installed. Run: uv add modal")
        return False

    # Check Modal authentication
    try:
        result = subprocess.run(
            ["modal", "token", "list"], capture_output=True, check=True
        )
        print("✅ Modal authentication verified")
    except subprocess.CalledProcessError:
        print("❌ Modal not authenticated. Run: modal setup")
        return False
    except FileNotFoundError:
        print("❌ Modal CLI not found. Run: uv add modal")
        return False

    print()

    # Run the integrated training
    success = run_integrated_training()

    if success:
        show_monitoring_info()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
