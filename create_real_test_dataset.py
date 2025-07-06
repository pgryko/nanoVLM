#!/usr/bin/env python3
"""
Create a real test dataset with actual images for Modal.com training

This script creates a small dataset with real images that can be used
to test the Modal.com training pipeline end-to-end.
"""

import json
import os
from PIL import Image, ImageDraw, ImageFont


def create_test_images():
    """Create simple test images with different colors and text"""
    print("🖼️  Creating test images...")

    # Create test images directory
    os.makedirs("test_images", exist_ok=True)

    # Define colors and descriptions
    test_data = [
        {
            "color": (255, 100, 100),
            "name": "red",
            "description": "a red colored square",
        },
        {
            "color": (100, 255, 100),
            "name": "green",
            "description": "a green colored square",
        },
        {
            "color": (100, 100, 255),
            "name": "blue",
            "description": "a blue colored square",
        },
        {
            "color": (255, 255, 100),
            "name": "yellow",
            "description": "a yellow colored square",
        },
        {
            "color": (255, 100, 255),
            "name": "purple",
            "description": "a purple colored square",
        },
    ]

    created_images = []

    for i, data in enumerate(test_data):
        # Create a 224x224 image with the specified color
        img = Image.new("RGB", (224, 224), data["color"])

        # Add some text to make it more interesting
        draw = ImageDraw.Draw(img)
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None

        # Draw the color name on the image
        text = data["name"].upper()
        if font:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center the text
            x = (224 - text_width) // 2
            y = (224 - text_height) // 2

            # Draw text with contrasting color
            text_color = (0, 0, 0) if sum(data["color"]) > 400 else (255, 255, 255)
            draw.text((x, y), text, fill=text_color, font=font)

        # Save the image
        img_path = f"test_images/{data['name']}_square.jpg"
        img.save(img_path)
        created_images.append((img_path, data))

    print(f"✅ Created {len(created_images)} test images")
    return created_images


def create_test_dataset():
    """Create a test dataset with real images"""
    print("📝 Creating test dataset...")

    # Create test images
    image_data = create_test_images()

    # Create dataset entries
    dataset = []

    for img_path, data in image_data:
        # Create multiple conversation examples per image
        conversations = [
            {
                "conversations": [
                    {"role": "user", "content": "What color is this image?"},
                    {"role": "assistant", "content": f"This image is {data['name']}."},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "Describe what you see."},
                    {"role": "assistant", "content": f"I see {data['description']}."},
                ]
            },
            {
                "conversations": [
                    {
                        "role": "user",
                        "content": "What is the dominant color in this image?",
                    },
                    {
                        "role": "assistant",
                        "content": f"The dominant color is {data['name']}.",
                    },
                ]
            },
        ]

        # Add each conversation as a separate dataset entry
        for conv in conversations:
            dataset.append(
                {"image_path": img_path, "conversations": conv["conversations"]}
            )

    # Save dataset
    dataset_path = "datasets/real_test_dataset.json"
    os.makedirs("datasets", exist_ok=True)

    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Created test dataset: {dataset_path}")
    print(f"   - {len(dataset)} training samples")
    print(f"   - {len(image_data)} unique images")

    return dataset_path


def main():
    """Create the test dataset and provide usage instructions"""
    print("🎯 Creating Real Test Dataset for Modal.com")
    print("=" * 45)

    try:
        dataset_path = create_test_dataset()

        print("\n🎉 Test dataset created successfully!")

        print("\n📊 Dataset Details:")
        print(f"   Dataset file: {dataset_path}")
        print("   Images directory: test_images/")
        print("   Format: Single-image conversations")

        print("\n🚀 Ready for Modal.com Training!")
        print("Run this command to test:")
        print(
            f"""
uv run python modal/submit_modal_training.py \\
  --custom_dataset_path {dataset_path} \\
  --batch_size 2 \\
  --max_training_steps 50 \\
  --eval_interval 25 \\
  --wandb_entity pgryko
"""
        )

        print("\n💰 Expected cost: ~$0.25-0.50 (very short test)")
        print("⏱️  Expected duration: ~2-3 minutes")

        print("\n📋 What this test will validate:")
        print("   ✅ Modal.com training pipeline")
        print("   ✅ Dataset loading with real images")
        print("   ✅ Model training on A100 GPU")
        print("   ✅ W&B logging (if configured)")
        print("   ✅ Model checkpointing")

        return True

    except Exception as e:
        print(f"❌ Failed to create test dataset: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
