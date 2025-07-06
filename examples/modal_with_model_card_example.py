#!/usr/bin/env python3
"""
Example: Training NanoVLM on Modal with automatic model card generation

This example shows how the enhanced Modal setup automatically generates
and pushes comprehensive model cards to HuggingFace Hub.
"""

import subprocess
import sys
import os


def main():
    """
    Demonstrate the enhanced Modal training with automatic model card generation
    """

    print("🚀 NanoVLM Training with Automatic Model Card Generation")
    print("=" * 60)
    print()

    print("This example will:")
    print("✅ Build a mixed dataset (COCO + VQAv2) on Modal")
    print("✅ Train NanoVLM with the dataset")
    print("✅ Generate a comprehensive model card with:")
    print("   - Training configuration details")
    print("   - Dataset information and statistics")
    print("   - Model architecture specifications")
    print("   - Usage examples and code snippets")
    print("   - Performance expectations")
    print("   - Reproducibility instructions")
    print("   - Links to W&B monitoring")
    print("✅ Push model + model card to HuggingFace Hub")
    print()

    # Configuration
    config = {
        "dataset_type": "mixed",
        "dataset_limit": 8000,  # Smaller for demo
        "batch_size": 8,
        "max_training_steps": 2000,  # Shorter training for demo
        "wandb_entity": "piotr-gryko-devalogic",
        "hub_model_id": "pgryko/nanovlm-demo-with-card",
        "compile": True,
        "push_to_hub": True,
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("Expected model card features:")
    print("📋 Comprehensive metadata (license, tags, datasets)")
    print("🏗️  Architecture details (222M params, SigLIP + SmolLM2)")
    print("📊 Training configuration (batch size, learning rates, etc.)")
    print("📈 Dataset information (type, size, description)")
    print("💻 Usage examples (Python code, installation)")
    print("🔄 Reproducibility instructions")
    print("📱 Interactive widgets for testing")
    print("⚠️  Limitations and ethical considerations")
    print("📚 Citations and acknowledgments")
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

    print("Command to execute:")
    print(" ".join(cmd))
    print()

    # Ask for confirmation
    response = input("Do you want to start the training? (y/N): ").strip().lower()
    if response != "y":
        print("Training cancelled.")
        return False

    print("🎯 Starting training with automatic model card generation...")

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print()
        print("✅ Training completed successfully!")
        print()
        print("🤗 Your model with comprehensive model card is available at:")
        print(f"   https://huggingface.co/{config['hub_model_id']}")
        print()
        print("📊 Training metrics available at:")
        print(f"   https://wandb.ai/{config['wandb_entity']}/nanovlm-modal")
        print()
        print("🔍 The model card includes:")
        print("   - Complete training configuration")
        print("   - Dataset statistics and information")
        print("   - Usage examples and code")
        print("   - Performance expectations")
        print("   - Reproducibility instructions")
        print("   - Interactive testing widgets")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
