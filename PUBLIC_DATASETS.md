# Public Datasets for NanoVLM Fine-tuning

This guide covers popular public datasets you can use to fine-tune NanoVLM, with conversion scripts and training examples.

## Quick Start

1. **Choose a dataset** from the options below
2. **Convert to NanoVLM format** using provided scripts
3. **Train** using the custom training script

## Available Datasets

### 1. LLaVA-Instruct (Recommended for General Use)

**Description**: High-quality instruction-following conversations with images
- **Size**: 150K samples (150k subset) or 665K (full)
- **Type**: Instruction following, conversation
- **Quality**: High (human-curated)
- **Use case**: General vision-language understanding

```bash
# Convert LLaVA-Instruct-150K
python examples/public_datasets.py \
  --dataset llava \
  --output datasets/llava_150k.json \
  --subset 150k \
  --limit 10000  # Optional: limit for testing

# Train on LLaVA
python train_custom.py \
  --custom_dataset_path datasets/llava_150k.json \
  --batch_size 8 \
  --max_training_steps 5000 \
  --log_wandb
```

### 2. COCO Captions (Good for Image Description)

**Description**: Natural image descriptions
- **Size**: 118K training images, 5K validation
- **Type**: Image captioning
- **Quality**: High (human-annotated)
- **Use case**: Learning to describe images

```bash
# Convert COCO Captions
python examples/public_datasets.py \
  --dataset coco \
  --output datasets/coco_captions.json \
  --split train \
  --limit 20000

# Train on COCO
python train_custom.py \
  --custom_dataset_path datasets/coco_captions.json \
  --batch_size 16 \
  --max_training_steps 3000
```

### 3. VQAv2 (Best for Question Answering)

**Description**: Visual question answering dataset
- **Size**: 200K+ training samples
- **Type**: Question answering
- **Quality**: High (balanced answers)
- **Use case**: Answering questions about images

```bash
# Convert VQAv2
python examples/public_datasets.py \
  --dataset vqav2 \
  --output datasets/vqav2.json \
  --limit 50000

# Train on VQAv2
python train_custom.py \
  --custom_dataset_path datasets/vqav2.json \
  --batch_size 12 \
  --max_training_steps 4000
```

### 4. VizWiz (Accessibility-Focused)

**Description**: Real-world images from visually impaired users
- **Size**: 20K+ training samples
- **Type**: Accessibility VQA
- **Quality**: High (real user questions)
- **Use case**: Accessibility applications

```bash
# Convert VizWiz
python examples/public_datasets.py \
  --dataset vizwiz \
  --output datasets/vizwiz.json \
  --split train

# Train on VizWiz
python train_custom.py \
  --custom_dataset_path datasets/vizwiz.json \
  --batch_size 8 \
  --max_training_steps 2000
```

### 5. ScienceQA (Educational Content)

**Description**: Science questions with diagrams
- **Size**: 12K+ samples with images
- **Type**: Multiple choice science questions
- **Quality**: High (educational)
- **Use case**: Educational/scientific applications

```bash
# Convert ScienceQA
python examples/public_datasets.py \
  --dataset scienceqa \
  --output datasets/scienceqa.json \
  --split train

# Train on ScienceQA
python train_custom.py \
  --custom_dataset_path datasets/scienceqa.json \
  --batch_size 6 \
  --max_training_steps 1500
```

## Dataset Combinations

### Multi-Task Training (Recommended)

Combine multiple datasets for better generalization:

```bash
# Convert multiple datasets
python examples/public_datasets.py --dataset coco --output datasets/coco.json --limit 10000
python examples/public_datasets.py --dataset vqav2 --output datasets/vqav2.json --limit 10000
python examples/public_datasets.py --dataset vizwiz --output datasets/vizwiz.json

# Combine datasets
python -c "
import json
datasets = []
for name in ['coco', 'vqav2', 'vizwiz']:
    with open(f'datasets/{name}.json') as f:
        datasets.extend(json.load(f))
        
with open('datasets/combined.json', 'w') as f:
    json.dump(datasets, f, indent=2)
print(f'Combined dataset: {len(datasets)} samples')
"

# Train on combined dataset
python train_custom.py \
  --custom_dataset_path datasets/combined.json \
  --batch_size 8 \
  --max_training_steps 8000 \
  --eval_interval 200
```

## Dataset Sizes and Training Times

| Dataset | Samples | Est. Training Time (M1) | Est. Training Time (GPU) | Memory Usage |
|---------|---------|-------------------------|--------------------------|--------------|
| COCO (20K) | 20,000 | 4-6 hours | 1-2 hours | 4-8GB |
| VQAv2 (50K) | 50,000 | 10-15 hours | 3-5 hours | 6-12GB |
| LLaVA (10K) | 10,000 | 3-4 hours | 1 hour | 4-8GB |
| VizWiz (full) | 20,000 | 4-6 hours | 1-2 hours | 4-8GB |
| ScienceQA (full) | 12,000 | 3-4 hours | 45 minutes | 3-6GB |

## Training Recommendations by Use Case

### General Purpose Model
```bash
# Use LLaVA-Instruct for best overall performance
python examples/public_datasets.py --dataset llava --output llava.json --limit 20000
python train_custom.py --custom_dataset_path llava.json --batch_size 8 --max_training_steps 5000
```

### Image Captioning Specialist
```bash
# Use COCO Captions
python examples/public_datasets.py --dataset coco --output coco.json --limit 30000
python train_custom.py --custom_dataset_path coco.json --batch_size 12 --max_training_steps 4000
```

### Question Answering Specialist
```bash
# Use VQAv2
python examples/public_datasets.py --dataset vqav2 --output vqav2.json --limit 40000
python train_custom.py --custom_dataset_path vqav2.json --batch_size 10 --max_training_steps 6000
```

### Educational/Science Model
```bash
# Use ScienceQA
python examples/public_datasets.py --dataset scienceqa --output science.json
python train_custom.py --custom_dataset_path science.json --batch_size 8 --max_training_steps 2000
```

## Tips for Success

### 1. Start Small
- Begin with 5K-10K samples to test your setup
- Increase dataset size once you're satisfied with the pipeline

### 2. Monitor Training
- Always use validation splits
- Watch for overfitting (validation loss increasing)
- Use W&B for better monitoring

### 3. Quality over Quantity
- LLaVA-150K (smaller, higher quality) often outperforms larger, noisier datasets
- Consider filtering/cleaning datasets

### 4. Hardware Considerations

**M1 Mac:**
- Batch size 4-8
- Gradient accumulation 4-8 steps
- Expect 2-5x slower than GPU

**NVIDIA GPU:**
- Batch size 12-32 (depending on GPU memory)
- Can use torch.compile for speedup
- Multi-GPU training supported

## Example: Complete Training Pipeline

```bash
#!/bin/bash
# Complete example: Train on LLaVA-Instruct with W&B and HF upload

# 1. Convert dataset
python examples/public_datasets.py \
  --dataset llava \
  --output datasets/llava_training.json \
  --limit 15000

# 2. Train with monitoring
python train_custom.py \
  --custom_dataset_path datasets/llava_training.json \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_training_steps 3000 \
  --eval_interval 150 \
  --log_wandb \
  --wandb_entity your_username

# 3. Upload to HuggingFace
python upload_to_hf.py \
  --checkpoint_path ./checkpoints/nanoVLM_custom_* \
  --repo_name your-username/nanovlm-llava-finetuned

echo "Training complete! Model available at: https://huggingface.co/your-username/nanovlm-llava-finetuned"
```

## Dataset-Specific Notes

### LLaVA-Instruct
- **Best for**: General instruction following
- **Note**: Images referenced by URLs (need to download separately for some versions)
- **Conversations**: Multi-turn, high quality

### COCO Captions  
- **Best for**: Image description tasks
- **Note**: Simple single-turn conversations
- **Images**: Widely available, good quality

### VQAv2
- **Best for**: Factual question answering
- **Note**: Balanced dataset (each question has multiple annotators)
- **Answers**: Usually short (1-3 words)

### VizWiz
- **Best for**: Real-world, accessibility applications
- **Note**: Images often blurry/challenging (real user photos)
- **Questions**: Natural, unfiltered user questions

### ScienceQA
- **Best for**: Educational, STEM applications
- **Note**: Multiple choice format
- **Images**: Diagrams, charts, scientific illustrations

Choose the dataset that best matches your intended use case, or combine multiple datasets for a more robust model!