#!/usr/bin/env python3
"""
Create a synthetic dataset that doesn't require external image files

This creates a dataset that generates images programmatically,
avoiding the need to upload image files to Modal.
"""

import json
import os


def create_synthetic_dataset():
    """Create a synthetic dataset with programmatically generated images"""
    print("📝 Creating synthetic dataset...")

    # Create dataset entries that will generate images on-the-fly
    dataset = []

    colors = [
        {"name": "red", "rgb": [255, 100, 100]},
        {"name": "green", "rgb": [100, 255, 100]},
        {"name": "blue", "rgb": [100, 100, 255]},
        {"name": "yellow", "rgb": [255, 255, 100]},
        {"name": "purple", "rgb": [255, 100, 255]},
    ]

    for i, color in enumerate(colors):
        # Create multiple conversation examples per color
        conversations = [
            {
                "conversations": [
                    {"role": "user", "content": "What color is this image?"},
                    {"role": "assistant", "content": f"This image is {color['name']}."},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "Describe what you see."},
                    {
                        "role": "assistant",
                        "content": f"I see a {color['name']} colored square.",
                    },
                ]
            },
        ]

        # Add each conversation as a separate dataset entry
        for conv in conversations:
            dataset.append(
                {
                    "image_path": f"synthetic://{color['name']}_square_{color['rgb'][0]}_{color['rgb'][1]}_{color['rgb'][2]}",
                    "conversations": conv["conversations"],
                }
            )

    # Save dataset
    dataset_path = "datasets/synthetic_test_dataset.json"
    os.makedirs("datasets", exist_ok=True)

    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Created synthetic dataset: {dataset_path}")
    print(f"   - {len(dataset)} training samples")
    print("   - Uses synthetic:// protocol for on-the-fly image generation")

    return dataset_path


def main():
    """Create the synthetic dataset"""
    print("🎯 Creating Synthetic Test Dataset for Modal.com")
    print("=" * 48)

    try:
        dataset_path = create_synthetic_dataset()

        print("\n🎉 Synthetic dataset created successfully!")

        print("\n📊 Dataset Details:")
        print(f"   Dataset file: {dataset_path}")
        print("   Image generation: On-the-fly synthetic images")
        print("   Format: Single-image conversations")

        print("\n🚀 Ready for Modal.com Training!")
        print("This dataset will work with a modified custom_dataset.py that")
        print("generates images programmatically when it sees 'synthetic://' paths.")

        return True

    except Exception as e:
        print(f"❌ Failed to create synthetic dataset: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
