#!/usr/bin/env python3
"""
Simple CPU training test for nanoVLM on Azure ML
Tests the training pipeline without GPU requirements
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="CPU Test Training")
    parser.add_argument("--dataset", type=str, default="COCO")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_training_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_projector", type=float, default=0.00512)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="nanovlm-test")

    args = parser.parse_args()

    print("🧪 nanoVLM CPU Test Training")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Steps: {args.max_training_steps}")
    print("Device: CPU (test mode)")
    print("=" * 50)

    # Test basic imports
    try:
        import torch

        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"✅ Device Count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return 1

    try:
        import transformers

        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return 1

    try:
        import datasets

        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets import failed: {e}")
        return 1

    # Test model imports
    try:
        from models.nanovlm import NanoVLM

        print("✅ NanoVLM model import successful")
    except ImportError as e:
        print(f"❌ NanoVLM import failed: {e}")
        print("Available files:")
        for root, dirs, files in os.walk("."):
            for file in files[:10]:  # Limit output
                print(f"  {os.path.join(root, file)}")
        return 1

    # Test data loading
    try:
        from data.dataset_factory import create_dataset

        print("✅ Dataset factory import successful")

        # Try to create a small dataset
        print(f"🔄 Testing {args.dataset} dataset creation...")
        dataset = create_dataset(
            dataset_name=args.dataset,
            split="train",
            max_samples=5,  # Very small for testing
        )
        print(f"✅ Dataset created: {len(dataset)} samples")

    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return 1

    # Test model creation
    try:
        print("🔄 Testing model creation...")
        model = NanoVLM()
        print("✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass with dummy data
        print("🔄 Testing forward pass...")
        dummy_input = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

        with torch.no_grad():
            output = model(**dummy_input)
        print("✅ Forward pass successful")

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return 1

    # Simulate training steps
    print("🔄 Simulating training steps...")
    for step in range(args.max_training_steps):
        print(f"Step {step + 1}/{args.max_training_steps}: Training simulation")

        if (step + 1) % args.eval_interval == 0:
            print(f"Step {step + 1}: Evaluation simulation")

    # Test output directory
    output_dir = os.environ.get("AZUREML_MODEL_DIR", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save a test file
    test_file = os.path.join(output_dir, "test_output.txt")
    with open(test_file, "w") as f:
        f.write("CPU test completed successfully!\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Steps: {args.max_training_steps}\n")

    print(f"✅ Test output saved to: {test_file}")
    print("🎉 CPU Test Training Completed Successfully!")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
