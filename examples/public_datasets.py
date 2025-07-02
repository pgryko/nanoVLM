#!/usr/bin/env python3
"""
Examples of using public datasets for NanoVLM fine-tuning.

This script shows how to convert popular public datasets into NanoVLM format.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse


def convert_llava_instruct(output_path: str, subset: str = "150k", limit: int = None):
    """
    Convert LLaVA Instruct dataset to NanoVLM format.
    
    Args:
        output_path: Path to save converted dataset
        subset: Dataset subset ("150k" or full)
        limit: Limit number of samples (None for all)
    """
    print(f"Loading LLaVA-Instruct-{subset} dataset...")
    
    # Load dataset
    if subset == "150k":
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
    else:
        dataset = load_dataset("liuhaotian/LLaVA-Instruct")
    
    train_data = dataset['train']
    if limit:
        train_data = train_data.select(range(min(limit, len(train_data))))
    
    converted_data = []
    
    print("Converting to NanoVLM format...")
    for item in tqdm(train_data):
        # LLaVA format has conversations already
        conversations = []
        
        for conv in item['conversations']:
            role = "user" if conv['from'] == 'human' else "assistant"
            content = conv['value']
            
            # Remove image tokens from user messages (we'll add them back)
            if role == "user":
                content = content.replace('<image>', '').strip()
            
            conversations.append({
                "role": role,
                "content": content
            })
        
        # Skip if no conversations
        if not conversations:
            continue
            
        converted_item = {
            "image_path": item['image'],  # Will need to download images separately
            "conversations": conversations
        }
        converted_data.append(converted_item)
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_path}")
    print("Note: You'll need to download the images separately using the image URLs")


def convert_coco_captions(output_path: str, split: str = "train", limit: int = None):
    """
    Convert COCO Captions to simple image description format.
    
    Args:
        output_path: Path to save converted dataset
        split: Dataset split ("train", "validation")
        limit: Limit number of samples
    """
    print(f"Loading COCO Captions {split} dataset...")
    
    dataset = load_dataset("HuggingFaceM4/COCO", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    converted_data = []
    
    print("Converting to NanoVLM format...")
    for item in tqdm(dataset):
        # Use the first caption as the description
        caption = item['sentences']['raw'][0] if item['sentences']['raw'] else "No description available"
        
        converted_item = {
            "image_path": f"coco_{split}_{item['cocoid']}.jpg",  # You'll need to save images with this naming
            "conversations": [
                {"role": "user", "content": "Describe this image."},
                {"role": "assistant", "content": caption}
            ]
        }
        converted_data.append(converted_item)
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_path}")


def convert_vqav2(output_path: str, limit: int = None):
    """
    Convert VQAv2 dataset to NanoVLM format.
    """
    print("Loading VQAv2 dataset...")
    
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    converted_data = []
    
    print("Converting to NanoVLM format...")
    for item in tqdm(dataset):
        # Get the most common answer
        if item['answers']:
            answer = max(set(item['answers']['answer']), key=item['answers']['answer'].count)
        else:
            continue
            
        converted_item = {
            "image_path": f"vqa_{item['image_id']}.jpg",
            "conversations": [
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": answer}
            ]
        }
        converted_data.append(converted_item)
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_path}")


def convert_vizwiz(output_path: str, split: str = "train", limit: int = None):
    """
    Convert VizWiz dataset (accessibility-focused VQA).
    """
    print(f"Loading VizWiz {split} dataset...")
    
    dataset = load_dataset("HuggingFaceM4/VizWiz", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    converted_data = []
    
    print("Converting to NanoVLM format...")
    for item in tqdm(dataset):
        # Skip if no valid answers
        if not item['answers'] or not any(ans['answer_confidence'] == 'yes' for ans in item['answers']):
            continue
            
        # Get confident answers
        confident_answers = [ans['answer'] for ans in item['answers'] if ans['answer_confidence'] == 'yes']
        if not confident_answers:
            continue
            
        answer = max(set(confident_answers), key=confident_answers.count)
        
        converted_item = {
            "image_path": f"vizwiz_{item['image_id']}.jpg",
            "conversations": [
                {"role": "user", "content": item['question']},
                {"role": "assistant", "content": answer}
            ]
        }
        converted_data.append(converted_item)
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_path}")


def convert_science_qa(output_path: str, split: str = "train", limit: int = None):
    """
    Convert ScienceQA dataset to NanoVLM format.
    """
    print(f"Loading ScienceQA {split} dataset...")
    
    dataset = load_dataset("HuggingFaceM4/ScienceQA", split=split)
    # Filter for items with images
    dataset = dataset.filter(lambda x: x['image'] is not None)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    converted_data = []
    
    print("Converting to NanoVLM format...")
    for idx, item in enumerate(tqdm(dataset)):
        # Create question with choices
        question = item['question']
        choices = item['choices']
        
        if choices:
            question += "\nChoices:\n"
            for i, choice in enumerate(choices):
                question += f"{chr(65+i)}. {choice}\n"
            question += "Answer with the letter only."
        
        # Get answer
        answer_idx = item['answer']
        if answer_idx < len(choices):
            answer = chr(65 + answer_idx)  # Convert to letter
        else:
            continue
            
        converted_item = {
            "image_path": f"scienceqa_{idx}.jpg",  # Will need to save images
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        converted_data.append(converted_item)
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} samples to {output_path}")


def download_images_from_urls(dataset_path: str, image_dir: str, url_key: str = "image_path"):
    """
    Download images from URLs in the dataset.
    
    Args:
        dataset_path: Path to the JSON dataset
        image_dir: Directory to save images
        url_key: Key in dataset containing image URLs
    """
    import requests
    from PIL import Image
    import io
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"Downloading {len(data)} images...")
    
    for idx, item in enumerate(tqdm(data)):
        if url_key not in item:
            continue
            
        url = item[url_key]
        if not url.startswith('http'):
            continue
            
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save image
            img = Image.open(io.BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            filename = f"image_{idx:06d}.jpg"
            img.save(os.path.join(image_dir, filename))
            
            # Update dataset with local path
            item[url_key] = filename
            
        except Exception as e:
            print(f"Failed to download image {idx}: {e}")
            # Remove failed items
            continue
    
    # Save updated dataset
    with open(dataset_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Downloaded images to {image_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert public datasets to NanoVLM format")
    parser.add_argument('--dataset', choices=['llava', 'coco', 'vqav2', 'vizwiz', 'scienceqa'], 
                        required=True, help='Dataset to convert')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--limit', type=int, help='Limit number of samples')
    parser.add_argument('--split', default='train', help='Dataset split to use')
    parser.add_argument('--subset', help='Dataset subset (for LLaVA: "150k")')
    parser.add_argument('--download_images', action='store_true', 
                        help='Download images from URLs (where applicable)')
    parser.add_argument('--image_dir', help='Directory to save downloaded images')
    
    args = parser.parse_args()
    
    if args.dataset == 'llava':
        subset = args.subset or "150k"
        convert_llava_instruct(args.output, subset, args.limit)
        
    elif args.dataset == 'coco':
        convert_coco_captions(args.output, args.split, args.limit)
        
    elif args.dataset == 'vqav2':
        convert_vqav2(args.output, args.limit)
        
    elif args.dataset == 'vizwiz':
        convert_vizwiz(args.output, args.split, args.limit)
        
    elif args.dataset == 'scienceqa':
        convert_science_qa(args.output, args.split, args.limit)
    
    # Download images if requested
    if args.download_images and args.image_dir:
        download_images_from_urls(args.output, args.image_dir)
    
    print("\n✅ Dataset conversion complete!")
    print(f"📁 Dataset saved to: {args.output}")
    if args.download_images and args.image_dir:
        print(f"🖼️  Images saved to: {args.image_dir}")
    
    print("\n📝 To train with this dataset:")
    print("python train_custom.py \\")
    print(f"  --custom_dataset_path {args.output} \\")
    if args.image_dir:
        print(f"  --image_root_dir {args.image_dir} \\")
    print("  --batch_size 8 \\")
    print("  --max_training_steps 2000")


if __name__ == "__main__":
    main()