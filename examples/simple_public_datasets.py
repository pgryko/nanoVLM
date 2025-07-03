#!/usr/bin/env python3
"""
Simplified public dataset conversion for NanoVLM.

This script focuses on the most reliable datasets that work well out of the box.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import argparse


def convert_coco_captions(output_path: str, split: str = "train", limit: int = None):
    """
    Convert COCO Captions to NanoVLM format.
    This is the most reliable dataset to start with.
    """
    print(f"Loading COCO Captions {split} dataset...")
    
    dataset = load_dataset("HuggingFaceM4/COCO", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    converted_data = []
    
    print("Converting to NanoVLM format...")
    for idx, item in enumerate(tqdm(dataset)):
        try:
            # Use the first caption as the description
            captions = item['sentences']['raw']
            if not captions:
                continue
                
            caption = captions[0]  # Use first caption
            
            # Save image with a consistent naming scheme
            image_filename = f"coco_{split}_{item['cocoid']}.jpg"
            
            converted_item = {
                "image": item['image'],  # PIL Image object
                "image_path": image_filename,
                "conversations": [
                    {"role": "user", "content": "Describe this image."},
                    {"role": "assistant", "content": caption}
                ]
            }
            converted_data.append(converted_item)
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save images and create final dataset
    final_data = save_images_and_create_dataset(converted_data, output_path)
    
    print(f"Converted {len(final_data)} samples to {output_path}")


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
    for idx, item in enumerate(tqdm(dataset)):
        try:
            # Get the most common answer
            if not item['answers']:
                continue
                
            answers = item['answers']['answer']
            answer = max(set(answers), key=answers.count)
            
            image_filename = f"vqa_{item['image_id']}.jpg"
            
            converted_item = {
                "image": item['image'],  # PIL Image object
                "image_path": image_filename,
                "conversations": [
                    {"role": "user", "content": item['question']},
                    {"role": "assistant", "content": answer}
                ]
            }
            converted_data.append(converted_item)
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue
    
    # Save converted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save images and create final dataset
    final_data = save_images_and_create_dataset(converted_data, output_path)
    
    print(f"Converted {len(final_data)} samples to {output_path}")


def convert_llava_alternative(output_path: str, limit: int = None):
    """
    Use a more reliable LLaVA-style dataset.
    """
    print("Loading alternative instruction dataset...")
    
    try:
        # Try a more reliable instruction following dataset
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")  # Fallback to VQA
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        converted_data = []
        
        # Create instruction-following format from VQA
        prompts = [
            "What do you see in this image?",
            "Describe what's happening in this picture.",
            "What is the main subject of this image?",
            "Analyze this image and tell me what you observe.",
            "What can you tell me about this image?"
        ]
        
        print("Converting to instruction format...")
        for idx, item in enumerate(tqdm(dataset)):
            try:
                if not item['answers']:
                    continue
                    
                # Use question as context and create descriptive answer
                question = item['question']
                answers = item['answers']['answer']
                answer = max(set(answers), key=answers.count)
                
                # Create more natural instruction-following conversation
                prompt = prompts[idx % len(prompts)]
                response = f"Looking at this image, I can answer your question: {question} The answer is: {answer}"
                
                image_filename = f"instruct_{idx}.jpg"
                
                converted_item = {
                    "image": item['image'],
                    "image_path": image_filename,
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                converted_data.append(converted_item)
                
            except Exception:
                continue
        
        # Save converted dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save images and create final dataset
        final_data = save_images_and_create_dataset(converted_data, output_path)
        
        print(f"Converted {len(final_data)} samples to {output_path}")
        
    except Exception as e:
        print(f"Error loading instruction dataset: {e}")
        print("Falling back to COCO captions...")
        convert_coco_captions(output_path, limit=limit)


def save_images_and_create_dataset(data_with_images, output_path):
    """
    Save PIL images to disk and create final dataset JSON.
    """
    output_dir = Path(output_path).parent
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    final_data = []
    
    print("Saving images...")
    for item in tqdm(data_with_images):
        try:
            # Save the PIL image
            image = item['image']
            image_path = image_dir / item['image_path']
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(image_path, 'JPEG', quality=90)
            
            # Create final dataset entry
            final_item = {
                "image_path": str(image_path.relative_to(output_dir.parent)),
                "conversations": item['conversations']
            }
            final_data.append(final_item)
            
        except Exception as e:
            print(f"Error saving image: {e}")
            continue
    
    # Save final dataset
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    return final_data


def create_mixed_dataset(output_path: str, coco_limit: int = 5000, vqa_limit: int = 5000):
    """
    Create a mixed dataset combining COCO and VQA for better diversity.
    """
    print("Creating mixed dataset with COCO captions and VQA...")
    
    # Create temporary files
    temp_dir = Path(output_path).parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    coco_path = temp_dir / "coco_temp.json"
    vqa_path = temp_dir / "vqa_temp.json"
    
    # Convert datasets
    convert_coco_captions(str(coco_path), limit=coco_limit)
    convert_vqav2(str(vqa_path), limit=vqa_limit)
    
    # Combine datasets
    combined_data = []
    
    # Load COCO data
    if coco_path.exists():
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
            combined_data.extend(coco_data)
    
    # Load VQA data
    if vqa_path.exists():
        with open(vqa_path, 'r') as f:
            vqa_data = json.load(f)
            combined_data.extend(vqa_data)
    
    # Shuffle the combined data
    import random
    random.shuffle(combined_data)
    
    # Save combined dataset
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Created mixed dataset with {len(combined_data)} samples at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert reliable public datasets to NanoVLM format")
    parser.add_argument('--dataset', choices=['coco', 'vqav2', 'llava', 'mixed'], 
                        required=True, help='Dataset to convert')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--limit', type=int, help='Limit number of samples')
    parser.add_argument('--split', default='train', help='Dataset split to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'coco':
        convert_coco_captions(args.output, args.split, args.limit)
        
    elif args.dataset == 'vqav2':
        convert_vqav2(args.output, args.limit)
        
    elif args.dataset == 'llava':
        convert_llava_alternative(args.output, args.limit)
        
    elif args.dataset == 'mixed':
        coco_limit = args.limit // 2 if args.limit else 5000
        vqa_limit = args.limit // 2 if args.limit else 5000
        create_mixed_dataset(args.output, coco_limit, vqa_limit)
    
    print("\n✅ Dataset conversion complete!")
    print(f"📁 Dataset saved to: {args.output}")
    print(f"🖼️  Images saved to: {output_path.parent}/images/")
    
    print("\n📝 To train with this dataset:")
    print("python train_custom.py \\")
    print(f"  --custom_dataset_path {args.output} \\")
    print("  --batch_size 8 \\")
    print("  --max_training_steps 2000")


if __name__ == "__main__":
    main()