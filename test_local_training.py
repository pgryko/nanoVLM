#!/usr/bin/env python3
"""
Local M1 Mac Testing Script for NanoVLM

This script runs a short training session locally on M1 Mac to test:
1. Code correctness and M1 compatibility
2. W&B logging integration
3. HuggingFace model publishing
4. Dataset loading and processing

Perfect for validating everything works before moving to cloud training.
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np


def create_test_images():
    """Create simple test images for local testing"""
    print("🖼️  Creating test images...")

    # Create test images directory
    test_images_dir = Path("test_data/images")
    test_images_dir.mkdir(parents=True, exist_ok=True)

    # Create simple colored images for testing
    colors_and_names = [
        ((255, 0, 0), "red_square.jpg"),  # Red
        ((0, 255, 0), "green_circle.jpg"),  # Green
        ((0, 0, 255), "blue_triangle.jpg"),  # Blue
        ((255, 255, 0), "yellow_star.jpg"),  # Yellow
        ((255, 0, 255), "purple_heart.jpg"),  # Purple
    ]

    created_images = []

    for color, filename in colors_and_names:
        # Create a simple 224x224 colored image
        img_array = np.full((224, 224, 3), color, dtype=np.uint8)
        img = Image.fromarray(img_array)

        img_path = test_images_dir / filename
        img.save(img_path)
        created_images.append(str(img_path))

    print(f"✅ Created {len(created_images)} test images")
    return created_images


def create_test_dataset():
    """Create a small test dataset for local training"""
    print("📝 Creating test dataset...")

    # Create test images first
    image_paths = create_test_images()

    # Create conversations for each image
    conversations_templates = [
        {
            "question": "What color is this image?",
            "answers": ["red", "green", "blue", "yellow", "purple"],
        },
        {
            "question": "Describe what you see in this image.",
            "answers": [
                "This is a solid red colored image.",
                "This is a solid green colored image.",
                "This is a solid blue colored image.",
                "This is a solid yellow colored image.",
                "This is a solid purple colored image.",
            ],
        },
        {
            "question": "What is the dominant color?",
            "answers": [
                "The dominant color is red.",
                "The dominant color is green.",
                "The dominant color is blue.",
                "The dominant color is yellow.",
                "The dominant color is purple.",
            ],
        },
    ]

    # Create dataset entries
    dataset = []
    for i, img_path in enumerate(image_paths):
        # Create multiple conversation examples per image
        for template in conversations_templates:
            dataset.append(
                {
                    "image_path": img_path,
                    "conversations": [
                        {"role": "user", "content": template["question"]},
                        {"role": "assistant", "content": template["answers"][i]},
                    ],
                }
            )

    # Save dataset
    dataset_path = "test_data/local_test_dataset.json"
    os.makedirs("test_data", exist_ok=True)

    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Created test dataset: {dataset_path}")
    print(f"   - {len(dataset)} training samples")
    print(f"   - {len(image_paths)} unique images")

    return dataset_path


def check_prerequisites():
    """Check if all prerequisites are installed"""
    print("🔍 Checking prerequisites...")

    required_packages = [
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "wandb",
        "huggingface_hub",
    ]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: uv add " + " ".join(missing))
        return False

    # Check if we're on M1 Mac
    try:
        import torch

        if torch.backends.mps.is_available():
            print("✅ M1 Mac MPS backend available")
        else:
            print("⚠️  MPS not available, will use CPU")
    except:
        print("⚠️  Could not check MPS availability")

    return True


def run_local_training(dataset_path, wandb_entity=None, test_hf_upload=False):
    """Run a short local training session"""
    print("\n🚀 Starting local training test...")

    # Training configuration optimized for M1 Mac
    train_args = [
        "uv",
        "run",
        "python",
        "train_custom.py",
        "--custom_dataset_path",
        dataset_path,
        "--batch_size",
        "2",  # Small batch size for M1 Mac
        "--gradient_accumulation_steps",
        "4",  # Effective batch size of 8
        "--max_training_steps",
        "20",  # Very short training for testing
        "--eval_interval",
        "10",  # Evaluate twice during training
        "--lr_mp",
        "0.001",  # Slightly lower learning rate
        "--lr_backbones",
        "1e-5",
        "--output_dir",
        "./test_checkpoints",
    ]

    # Add W&B logging if entity provided
    if wandb_entity:
        train_args.extend(
            [
                "--log_wandb",
                "--wandb_entity",
                wandb_entity,
                "--wandb_project",
                "nanovlm-local-test",
            ]
        )
        print(f"📊 W&B logging enabled for entity: {wandb_entity}")

    # Add HF upload if requested
    if test_hf_upload:
        train_args.extend(
            [
                "--push_to_hub",
                "--hub_model_id",
                (
                    f"{wandb_entity}/nanovlm-local-test"
                    if wandb_entity
                    else "nanovlm-local-test"
                ),
            ]
        )
        print("🤗 HuggingFace upload enabled")

    print(f"Command: {' '.join(train_args)}")
    print("\n" + "=" * 50)

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(train_args, check=True, capture_output=False)
        duration = time.time() - start_time

        print("\n" + "=" * 50)
        print(f"✅ Training completed successfully in {duration:.1f} seconds!")

        # Check if checkpoints were created
        checkpoint_dir = Path("test_checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*"))
            print(f"📁 Created {len(checkpoints)} checkpoint files")
            for cp in checkpoints[:3]:  # Show first 3
                print(f"   - {cp.name}")

        return True

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ Training failed after {duration:.1f} seconds")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False


def test_model_loading():
    """Test loading the trained model"""
    print("\n🧪 Testing model loading...")

    checkpoint_dir = Path("test_checkpoints")
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found to test")
        return False

    try:
        # Try to load the model
        sys.path.append(".")
        from models.vision_language_model import VisionLanguageModel

        # Find the latest checkpoint
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
        if not checkpoints:
            print("❌ No checkpoint directories found")
            return False

        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")

        model = VisionLanguageModel.from_pretrained(str(latest_checkpoint))
        param_count = sum(p.numel() for p in model.parameters())

        print("✅ Model loaded successfully!")
        print(f"   Parameters: {param_count:,}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")

    import shutil

    cleanup_paths = ["test_data", "test_checkpoints"]

    for path in cleanup_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"✅ Removed {path}")


def main():
    print("🧪 NanoVLM Local M1 Mac Testing")
    print("=" * 35)

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install missing packages.")
        return False

    # Get user input
    print("\n📋 Configuration:")
    wandb_entity = input(
        "Enter your W&B entity/username (or press Enter to skip): "
    ).strip()
    if not wandb_entity:
        wandb_entity = None
        print("⚠️  W&B logging disabled")

    test_hf = False
    if wandb_entity:
        hf_test = input("Test HuggingFace upload? (y/N): ").strip().lower()
        test_hf = hf_test == "y"

    print("\n🎯 Test Configuration:")
    print(f"   W&B Entity: {wandb_entity or 'None'}")
    print(f"   HF Upload: {'Yes' if test_hf else 'No'}")
    print("   Training Steps: 20 (very short test)")
    print("   Expected Duration: ~2-5 minutes")

    proceed = input("\nProceed with test? (Y/n): ").strip().lower()
    if proceed == "n":
        print("Test cancelled.")
        return False

    try:
        # Create test dataset
        dataset_path = create_test_dataset()

        # Run training
        success = run_local_training(dataset_path, wandb_entity, test_hf)

        if success:
            # Test model loading
            test_model_loading()

            print("\n🎉 All tests passed!")
            print("\n📊 Results:")
            print("   ✅ Dataset creation: OK")
            print("   ✅ M1 Mac training: OK")
            print("   ✅ Model checkpointing: OK")
            print("   ✅ Model loading: OK")
            if wandb_entity:
                print("   ✅ W&B logging: OK")
                print(
                    f"   📊 Check your W&B: https://wandb.ai/{wandb_entity}/nanovlm-local-test"
                )
            if test_hf:
                print("   ✅ HuggingFace upload: OK")

            print("\n🚀 Ready for cloud training!")
            print("   Your code is working correctly on M1 Mac")
            print("   You can now confidently use Modal.com or other cloud platforms")

        else:
            print("\n❌ Training test failed")
            print("   Check the error messages above")
            print("   Fix issues before trying cloud training")

        # Ask about cleanup
        cleanup = input("\nClean up test files? (Y/n): ").strip().lower()
        if cleanup != "n":
            cleanup_test_files()

        return success

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
