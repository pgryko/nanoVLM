#!/usr/bin/env python3
"""
Complete example: Train NanoVLM with W&B logging and upload to HuggingFace.

This script shows the full pipeline:
1. Prepare dataset
2. Train with W&B monitoring
3. Upload to HuggingFace Hub
"""

import os
import json
import subprocess
from pathlib import Path


def create_example_dataset():
    """Create a small example dataset for demonstration."""

    # Create example data
    dataset = [
        {
            "image_path": "example_images/cat.jpg",
            "conversations": [
                {"role": "user", "content": "What animal is in this image?"},
                {
                    "role": "assistant",
                    "content": "This image shows a domestic cat. The cat appears to be resting comfortably and has orange tabby markings.",
                },
            ],
        },
        {
            "image_path": "example_images/dog.jpg",
            "conversations": [
                {"role": "user", "content": "Describe what you see."},
                {
                    "role": "assistant",
                    "content": "I see a friendly-looking golden retriever dog sitting on grass. The dog has a happy expression with its tongue out.",
                },
                {"role": "user", "content": "What color is the dog?"},
                {
                    "role": "assistant",
                    "content": "The dog has a golden or light brown colored coat, which is characteristic of golden retriever breeds.",
                },
            ],
        },
        {
            "image_path": "example_images/landscape.jpg",
            "conversations": [
                {"role": "user", "content": "What type of scene is this?"},
                {
                    "role": "assistant",
                    "content": "This is a beautiful mountain landscape scene. I can see snow-capped peaks in the distance with a serene lake in the foreground, surrounded by evergreen trees.",
                },
            ],
        },
    ]

    # Save dataset
    os.makedirs("example_data", exist_ok=True)
    with open("example_data/dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("✓ Created example dataset at example_data/dataset.json")
    print("  Note: You'll need to add actual images to example_images/ directory")


def main():
    print("=== NanoVLM Full Training Pipeline Example ===\n")

    # Check credentials
    print("1. Checking credentials...")

    # Check W&B
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        print("⚠️  WANDB_API_KEY not found. Please set it:")
        print("   export WANDB_API_KEY='your_api_key'")
        print("   Get your key from: https://wandb.ai/authorize")
    else:
        print("✓ W&B API key found")

    # Check HF
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not found. Please set it:")
        print("   export HF_TOKEN='your_token'")
        print("   Get your token from: https://huggingface.co/settings/tokens")
    else:
        print("✓ HuggingFace token found")

    # Create example dataset
    print("\n2. Creating example dataset...")
    create_example_dataset()

    # Show training command
    print("\n3. Training command:")

    wandb_entity = input("Enter your W&B username/entity: ").strip()
    hf_username = input("Enter your HuggingFace username: ").strip()

    train_cmd = f"""python train_custom.py \\
  --custom_dataset_path ./example_data/dataset.json \\
  --image_root_dir ./example_images \\
  --batch_size 4 \\
  --gradient_accumulation_steps 4 \\
  --max_training_steps 500 \\
  --eval_interval 50 \\
  --log_wandb \\
  --wandb_entity {wandb_entity} \\
  --vlm_checkpoint_path ./checkpoints"""

    print(f"\n{train_cmd}")

    run_training = input("\nRun training now? (y/n): ").strip().lower() == "y"

    if run_training:
        print("\n4. Starting training...")
        result = subprocess.run(train_cmd, shell=True)

        if result.returncode != 0:
            print("❌ Training failed!")
            return

    # Show upload command
    print("\n5. Upload to HuggingFace Hub command:")

    # Find the latest checkpoint
    checkpoints = list(Path("./checkpoints").glob("nanoVLM_custom_*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Found checkpoint: {latest_checkpoint}")

        upload_cmd = f"""python upload_to_hf.py \\
  --checkpoint_path {latest_checkpoint} \\
  --repo_name {hf_username}/nanovlm-custom-example \\
  --commit_message "Upload custom trained NanoVLM model" """

        print(f"\n{upload_cmd}")

        run_upload = input("\nUpload model now? (y/n): ").strip().lower() == "y"

        if run_upload:
            print("\n6. Uploading to HuggingFace Hub...")
            result = subprocess.run(upload_cmd, shell=True)

            if result.returncode == 0:
                print("\n✅ Success! Your model is available at:")
                print(f"   https://huggingface.co/{hf_username}/nanovlm-custom-example")

    # Show how to use the model
    print("\n7. Using your uploaded model:")
    print(
        f"""
from transformers import AutoModel
from PIL import Image

# Load model
model = AutoModel.from_pretrained("{hf_username}/nanovlm-custom-example", trust_remote_code=True)

# Use for inference
image = Image.open("test.jpg")
response = model.generate(image, "What's in this image?")
print(response)
"""
    )

    print("\n=== Pipeline Complete! ===")
    print("\nNext steps:")
    print("1. Add more training data to improve model quality")
    print("2. Experiment with hyperparameters")
    print("3. Try different base models (vision/language)")
    print("4. Share your model with the community!")


if __name__ == "__main__":
    main()
