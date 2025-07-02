#!/usr/bin/env python3
"""
Example script to prepare custom datasets for NanoVLM training.

This script shows how to create the JSON format expected by the custom dataset classes.
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional
import random


def create_simple_dataset(
    image_dir: str,
    output_json: str,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp'],
    default_prompt: str = "Describe this image in detail.",
    default_response: str = "This image needs to be annotated."
):
    """
    Create a simple dataset JSON from a directory of images.
    This creates a template that you can then manually edit to add proper descriptions.
    
    Args:
        image_dir: Directory containing images
        output_json: Path to save the JSON file
        image_extensions: List of image file extensions to include
        default_prompt: Default user prompt for each image
        default_response: Default assistant response (placeholder)
    """
    image_dir = Path(image_dir)
    data = []
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.rglob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Create dataset entries
    for img_path in sorted(image_files):
        relative_path = img_path.relative_to(image_dir)
        
        entry = {
            "image_path": str(relative_path),
            "conversations": [
                {"role": "user", "content": default_prompt},
                {"role": "assistant", "content": default_response}
            ]
        }
        data.append(entry)
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created dataset template with {len(data)} entries at {output_json}")
    print("Please edit the JSON file to add proper image descriptions!")


def create_vqa_dataset(
    annotations_file: str,
    image_dir: str,
    output_json: str,
    image_key: str = "image_path",
    question_key: str = "question",
    answer_key: str = "answer"
):
    """
    Create a dataset from existing VQA-style annotations.
    
    Args:
        annotations_file: JSON file with VQA annotations
        image_dir: Directory containing images
        output_json: Path to save the formatted JSON
        image_key: Key in annotations for image filename
        question_key: Key in annotations for question
        answer_key: Key in annotations for answer
    """
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    data = []
    image_dir = Path(image_dir)
    
    for ann in annotations:
        if image_key not in ann or question_key not in ann or answer_key not in ann:
            continue
            
        entry = {
            "image_path": str(image_dir / ann[image_key]),
            "conversations": [
                {"role": "user", "content": ann[question_key]},
                {"role": "assistant", "content": ann[answer_key]}
            ]
        }
        data.append(entry)
    
    # Save formatted dataset
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created VQA dataset with {len(data)} entries at {output_json}")


def create_multi_turn_example():
    """Create an example of multi-turn conversation dataset."""
    example = [
        {
            "image_path": "images/cat_on_mat.jpg",
            "conversations": [
                {"role": "user", "content": "What do you see in this image?"},
                {"role": "assistant", "content": "I see a gray tabby cat sitting on a blue yoga mat in what appears to be a living room."},
                {"role": "user", "content": "What is the cat doing?"},
                {"role": "assistant", "content": "The cat is sitting upright on the mat, looking directly at the camera with an alert expression."},
                {"role": "user", "content": "Can you describe the surroundings?"},
                {"role": "assistant", "content": "The room has hardwood floors, and I can see part of a couch in the background. There's natural lighting coming from the left side of the image."}
            ]
        },
        {
            "image_path": "images/sunset_beach.jpg",
            "conversations": [
                {"role": "user", "content": "Describe this scene."},
                {"role": "assistant", "content": "This is a beautiful sunset scene at a beach. The sun is low on the horizon, creating vibrant orange and pink colors across the sky."},
                {"role": "user", "content": "Are there any people or objects visible?"},
                {"role": "assistant", "content": "Yes, I can see silhouettes of a few people walking along the shoreline, and there are some seagulls flying in the distance."}
            ]
        }
    ]
    
    return example


def create_multi_image_example():
    """Create an example of multi-image dataset."""
    example = [
        {
            "image_paths": ["images/before_renovation.jpg", "images/after_renovation.jpg"],
            "conversations": [
                {"role": "user", "content": "Compare these before and after renovation photos. What are the main changes?"},
                {"role": "assistant", "content": "The renovation has transformed the space significantly. The main changes include: new hardwood flooring replacing the old carpet, fresh white paint on all walls instead of the previous beige, modern recessed lighting replacing the single ceiling fixture, and updated baseboards and trim work. The room appears much brighter and more spacious after the renovation."}
            ]
        },
        {
            "image_paths": ["images/product_front.jpg", "images/product_back.jpg", "images/product_side.jpg"],
            "conversations": [
                {"role": "user", "content": "Analyze these product images and provide a detailed description."},
                {"role": "assistant", "content": "These images show a wireless bluetooth speaker from multiple angles. The front view reveals a sleek mesh grille with the brand logo centered. The back shows the control panel with power button, volume controls, and connectivity ports including USB-C charging and AUX input. The side view demonstrates the compact cylindrical design with rubber feet for stability. The speaker appears to be approximately 6 inches tall with a matte black finish."}
            ]
        }
    ]
    
    return example


def validate_dataset(json_path: str, image_root: Optional[str] = None):
    """
    Validate a dataset JSON file and check if all images exist.
    
    Args:
        json_path: Path to the dataset JSON
        image_root: Root directory for images (if paths are relative)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Validating dataset with {len(data)} entries...")
    
    missing_images = []
    invalid_entries = []
    
    for idx, entry in enumerate(data):
        # Check format
        if 'conversations' not in entry:
            invalid_entries.append(idx)
            continue
            
        # Check images
        if 'image_path' in entry:
            image_paths = [entry['image_path']]
        elif 'image_paths' in entry:
            image_paths = entry['image_paths']
        else:
            invalid_entries.append(idx)
            continue
        
        # Verify images exist
        for img_path in image_paths:
            if image_root:
                full_path = Path(image_root) / img_path
            else:
                full_path = Path(img_path)
                
            if not full_path.exists():
                missing_images.append(str(full_path))
    
    # Report results
    print("Validation complete:")
    print(f"  - Valid entries: {len(data) - len(invalid_entries)}")
    print(f"  - Invalid entries: {len(invalid_entries)}")
    print(f"  - Missing images: {len(missing_images)}")
    
    if invalid_entries:
        print(f"\nInvalid entry indices: {invalid_entries[:10]}{'...' if len(invalid_entries) > 10 else ''}")
    
    if missing_images:
        print("\nMissing images (first 10):")
        for img in missing_images[:10]:
            print(f"  - {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    return len(invalid_entries) == 0 and len(missing_images) == 0


def split_dataset(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split a dataset into train and validation sets.
    
    Args:
        input_json: Input dataset JSON
        train_json: Output path for training set
        val_json: Output path for validation set
        val_ratio: Ratio of data to use for validation
        seed: Random seed for splitting
    """
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(data)
    
    # Split
    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]
    
    # Save splits
    with open(train_json, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_json, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print("Split dataset into:")
    print(f"  - Training: {len(train_data)} samples ({train_json})")
    print(f"  - Validation: {len(val_data)} samples ({val_json})")


def main():
    parser = argparse.ArgumentParser(description="Prepare custom datasets for NanoVLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create simple dataset
    simple_parser = subparsers.add_parser('create-simple', help='Create simple dataset from image directory')
    simple_parser.add_argument('--image_dir', required=True, help='Directory containing images')
    simple_parser.add_argument('--output', required=True, help='Output JSON file')
    simple_parser.add_argument('--prompt', default="Describe this image in detail.", help='Default prompt')
    
    # Create from VQA annotations
    vqa_parser = subparsers.add_parser('create-vqa', help='Create dataset from VQA annotations')
    vqa_parser.add_argument('--annotations', required=True, help='VQA annotations JSON file')
    vqa_parser.add_argument('--image_dir', required=True, help='Directory containing images')
    vqa_parser.add_argument('--output', required=True, help='Output JSON file')
    
    # Create examples
    example_parser = subparsers.add_parser('create-examples', help='Create example dataset files')
    example_parser.add_argument('--output_dir', default='.', help='Directory to save examples')
    
    # Validate dataset
    validate_parser = subparsers.add_parser('validate', help='Validate dataset JSON')
    validate_parser.add_argument('--dataset', required=True, help='Dataset JSON file')
    validate_parser.add_argument('--image_root', help='Root directory for images')
    
    # Split dataset
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('--input', required=True, help='Input dataset JSON')
    split_parser.add_argument('--train_output', required=True, help='Training set output')
    split_parser.add_argument('--val_output', required=True, help='Validation set output')
    split_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    
    args = parser.parse_args()
    
    if args.command == 'create-simple':
        create_simple_dataset(args.image_dir, args.output, default_prompt=args.prompt)
        
    elif args.command == 'create-vqa':
        create_vqa_dataset(args.annotations, args.image_dir, args.output)
        
    elif args.command == 'create-examples':
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save multi-turn example
        with open(output_dir / 'example_multiturn.json', 'w') as f:
            json.dump(create_multi_turn_example(), f, indent=2)
        print(f"Created multi-turn example at {output_dir / 'example_multiturn.json'}")
        
        # Save multi-image example
        with open(output_dir / 'example_multiimage.json', 'w') as f:
            json.dump(create_multi_image_example(), f, indent=2)
        print(f"Created multi-image example at {output_dir / 'example_multiimage.json'}")
        
    elif args.command == 'validate':
        if validate_dataset(args.dataset, args.image_root):
            print("\nDataset is valid!")
        else:
            print("\nDataset has issues that need to be fixed.")
            
    elif args.command == 'split':
        split_dataset(args.input, args.train_output, args.val_output, args.val_ratio)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()