#!/usr/bin/env python3
"""
Example: Training NanoVLM on Modal.com with a custom dataset

This example demonstrates how to:
1. Create a simple custom dataset
2. Submit a training job to Modal.com
3. Monitor the training progress

Prerequisites:
- Modal account and CLI setup (modal setup)
- W&B account (optional but recommended)
- Custom dataset in JSON format
"""

import json
import os


def create_example_dataset(
    output_path: str = "datasets/modal_example.json", num_samples: int = 100
):
    """
    Create a simple example dataset for demonstration

    In practice, you would replace this with your actual dataset creation logic.
    """
    print(f"Creating example dataset with {num_samples} samples...")

    # Create datasets directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Example dataset - in practice, you'd load your real images and conversations
    example_data = []

    for i in range(num_samples):
        # This is just example data - replace with your actual image paths and conversations
        sample = {
            "image_path": f"path/to/your/image_{i:04d}.jpg",  # Replace with actual image paths
            "conversations": [
                {
                    "role": "user",
                    "content": f"What do you see in this image? (Sample {i+1})",
                },
                {
                    "role": "assistant",
                    "content": f"This is a sample response for image {i+1}. In a real dataset, this would be a meaningful description of the image content.",
                },
            ],
        }
        example_data.append(sample)

    # Save dataset
    with open(output_path, "w") as f:
        json.dump(example_data, f, indent=2)

    print(f"✅ Example dataset created: {output_path}")
    print(
        "📝 Note: This is just example data. Replace with your actual images and conversations."
    )
    return output_path


def submit_modal_training_example():
    """
    Example of how to submit a training job to Modal.com
    """

    # Check if Modal is installed
    try:
        import modal

        print("✅ Modal is available")
    except ImportError:
        print("❌ Modal not installed. Install with: pip install modal")
        print("   Then run: modal setup")
        return False

    # Create example dataset (replace this with your actual dataset)
    dataset_path = create_example_dataset()

    print("\n🚀 Example Modal.com Training Submission")
    print("=" * 50)

    # Example training command
    training_command = f"""
python modal/submit_modal_training.py \\
  --custom_dataset_path {dataset_path} \\
  --batch_size 4 \\
  --gradient_accumulation_steps 8 \\
  --max_training_steps 500 \\
  --eval_interval 100 \\
  --lr_mp 0.00512 \\
  --lr_backbones 5e-5 \\
  --wandb_entity YOUR_WANDB_USERNAME \\
  --wandb_project nanovlm-modal-example
"""

    print("To submit this training job, run:")
    print(training_command)

    print("\n📋 Training Configuration:")
    print(f"  Dataset: {dataset_path}")
    print("  Batch Size: 4 (effective: 32 with gradient accumulation)")
    print("  Training Steps: 500")
    print("  Expected Duration: ~5-10 minutes")
    print("  Expected Cost: ~$0.50-1.00")

    print("\n🔧 Before running, make sure to:")
    print("  1. Replace example dataset with your real data")
    print("  2. Update image paths in the JSON file")
    print("  3. Set up Modal secrets for W&B (optional)")
    print("  4. Replace YOUR_WANDB_USERNAME with your actual username")

    return True


def create_multi_image_example():
    """
    Create an example multi-image dataset
    """
    output_path = "datasets/modal_multi_image_example.json"
    print("Creating multi-image example dataset...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    example_data = []
    for i in range(20):  # Smaller dataset for multi-image example
        sample = {
            "image_paths": [
                f"path/to/your/image_{i:04d}_a.jpg",
                f"path/to/your/image_{i:04d}_b.jpg",
            ],
            "conversations": [
                {
                    "role": "user",
                    "content": f"Compare these two images. What are the differences? (Sample {i+1})",
                },
                {
                    "role": "assistant",
                    "content": f"Comparing the two images in sample {i+1}: The first image shows... while the second image shows... The main differences are...",
                },
            ],
        }
        example_data.append(sample)

    with open(output_path, "w") as f:
        json.dump(example_data, f, indent=2)

    print(f"✅ Multi-image example dataset created: {output_path}")

    # Show training command for multi-image
    training_command = f"""
python modal/submit_modal_training.py \\
  --custom_dataset_path {output_path} \\
  --multi_image \\
  --batch_size 2 \\
  --gradient_accumulation_steps 16 \\
  --max_training_steps 300 \\
  --eval_interval 50 \\
  --wandb_entity YOUR_WANDB_USERNAME \\
  --wandb_project nanovlm-modal-multi-image
"""

    print("\nFor multi-image training, use:")
    print(training_command)

    return output_path


def main():
    print("🎯 NanoVLM Modal.com Training Example")
    print("=" * 40)

    # Check if we're in the right directory
    if not os.path.exists("modal/submit_modal_training.py"):
        print("❌ Please run this script from the nanoVLM root directory")
        print("   Current directory should contain the 'modal/' folder")
        return

    print("\n1️⃣ Single Image Dataset Example")
    submit_modal_training_example()

    print("\n" + "=" * 50)
    print("2️⃣ Multi-Image Dataset Example")
    create_multi_image_example()

    print("\n" + "=" * 50)
    print("🎉 Examples created successfully!")

    print("\n📚 Next Steps:")
    print("1. Replace example datasets with your real data")
    print("2. Update image paths to point to actual images")
    print("3. Set up Modal account: modal setup")
    print("4. Create W&B account and set up secrets in Modal")
    print("5. Run the training commands shown above")

    print("\n🔗 Useful Resources:")
    print("- Modal Setup: https://modal.com/docs/guide/setup")
    print("- Modal Secrets: https://modal.com/docs/guide/secrets")
    print("- W&B Setup: https://wandb.ai/authorize")
    print("- NanoVLM Docs: modal/README.md")


if __name__ == "__main__":
    main()
