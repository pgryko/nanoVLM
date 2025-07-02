# Quick Start: Fine-tuning NanoVLM on Public Datasets

This guide provides the fastest way to start fine-tuning NanoVLM on high-quality public datasets.

## 🚀 Quick Start Commands

### Option 1: LLaVA-Instruct (Recommended for Beginners)

```bash
# 1. Convert dataset (10K samples for testing)
python examples/public_datasets.py \
  --dataset llava \
  --output llava_150k.json \
  --limit 10000

# 2. Train on M1 Mac
python train_custom.py \
  --custom_dataset_path llava_150k.json \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --log_wandb \
  --wandb_entity your_username

# 3. Train on NVIDIA GPU
python train_custom.py \
  --custom_dataset_path llava_150k.json \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --compile \
  --log_wandb \
  --wandb_entity your_username
```

### Option 2: COCO Captions (Image Description)

```bash
# 1. Convert dataset
python examples/public_datasets.py \
  --dataset coco \
  --output coco_captions.json \
  --limit 20000

# 2. Train
python train_custom.py \
  --custom_dataset_path coco_captions.json \
  --batch_size 8 \
  --max_training_steps 3000 \
  --log_wandb
```

### Option 3: VQAv2 (Question Answering)

```bash
# 1. Convert dataset
python examples/public_datasets.py \
  --dataset vqav2 \
  --output vqav2.json \
  --limit 15000

# 2. Train
python train_custom.py \
  --custom_dataset_path vqav2.json \
  --batch_size 8 \
  --max_training_steps 4000 \
  --log_wandb
```

### Option 4: Multi-Task Training (Best Performance)

```bash
# 1. Convert multiple datasets
python examples/public_datasets.py --dataset llava --output llava.json --limit 8000
python examples/public_datasets.py --dataset coco --output coco.json --limit 8000  
python examples/public_datasets.py --dataset vqav2 --output vqav2.json --limit 8000

# 2. Combine datasets
python -c "
import json
combined = []
for filename in ['llava.json', 'coco.json', 'vqav2.json']:
    with open(filename) as f:
        combined.extend(json.load(f))
with open('combined_dataset.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Combined dataset: {len(combined)} samples')
"

# 3. Train on combined dataset
python train_custom.py \
  --custom_dataset_path combined_dataset.json \
  --batch_size 8 \
  --max_training_steps 5000 \
  --eval_interval 200 \
  --log_wandb \
  --wandb_entity your_username
```

## 📊 Dataset Overview

| Dataset | Samples | Type | Best For | Training Time (M1) | Training Time (GPU) |
|---------|---------|------|----------|-------------------|-------------------|
| **LLaVA-Instruct** | 150K | Instruction following | General purpose | 3-4 hours (10K) | 1 hour (10K) |
| **COCO Captions** | 118K | Image description | Captioning tasks | 4-6 hours (20K) | 1-2 hours (20K) |
| **VQAv2** | 200K+ | Question answering | Factual QA | 3-5 hours (15K) | 1-1.5 hours (15K) |
| **VizWiz** | 20K | Accessibility VQA | Real-world apps | 2-3 hours (full) | 45 min (full) |
| **ScienceQA** | 12K | Educational QA | STEM applications | 2-3 hours (full) | 30-45 min (full) |

## 🎯 Recommended Workflows

### For Beginners
```bash
# Start with small LLaVA subset
python examples/public_datasets.py --dataset llava --output test.json --limit 1000
python train_custom.py --custom_dataset_path test.json --batch_size 4 --max_training_steps 200
```

### For Production Use
```bash
# Use larger, combined dataset
# ... multi-task training commands above ...
```

### With W&B Monitoring and HF Upload
```bash
# Set up credentials
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"

# Train with full monitoring
python train_custom.py \
  --custom_dataset_path combined_dataset.json \
  --batch_size 8 \
  --max_training_steps 5000 \
  --log_wandb \
  --wandb_entity your_username

# Upload to HuggingFace
python upload_to_hf.py \
  --checkpoint_path ./checkpoints/nanoVLM_custom_* \
  --repo_name your-username/nanovlm-public-finetuned
```

## 💡 Tips for Success

### Hardware-Specific Settings

**M1 Mac (MPS):**
```bash
--batch_size 4
--gradient_accumulation_steps 8
# Don't use --compile flag
```

**NVIDIA GPU:**
```bash
--batch_size 16
--gradient_accumulation_steps 2
--compile  # 20% speedup
```

**Multi-GPU (NVIDIA only):**
```bash
torchrun --nproc_per_node=4 train_custom.py \
  --custom_dataset_path dataset.json \
  --batch_size 8 \
  --gradient_accumulation_steps 1
```

### Dataset Size Guidelines

- **Testing/Development**: 1K-5K samples
- **Proof of Concept**: 5K-15K samples  
- **Production**: 15K+ samples
- **Research**: 50K+ samples

### Training Steps Guidelines

- **Small datasets (1K-5K)**: 500-1000 steps
- **Medium datasets (5K-15K)**: 1000-3000 steps
- **Large datasets (15K+)**: 3000-8000 steps

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch size and increase accumulation
--batch_size 2 --gradient_accumulation_steps 16
```

### Slow Training on M1
```bash
# This is expected - M1 is 3-5x slower than modern GPUs
# Consider using cloud GPUs for large datasets
```

### Dataset Download Issues
```bash
# Some datasets need manual image downloads
# The conversion script will show URLs that need downloading
```

## 🎉 Complete Example: 0 to Trained Model

```bash
#!/bin/bash
# Complete pipeline from scratch

# 1. Setup (one time)
pip install wandb huggingface_hub
wandb login
huggingface-cli login

# 2. Prepare dataset
python examples/public_datasets.py \
  --dataset llava \
  --output my_dataset.json \
  --limit 10000

# 3. Train with monitoring
python train_custom.py \
  --custom_dataset_path my_dataset.json \
  --batch_size 8 \
  --max_training_steps 2000 \
  --log_wandb \
  --wandb_entity your_username

# 4. Upload to HuggingFace
python upload_to_hf.py \
  --checkpoint_path ./checkpoints/nanoVLM_custom_* \
  --repo_name your-username/my-first-vlm

echo "🎉 Success! Your model is at: https://huggingface.co/your-username/my-first-vlm"
```

This should get you from zero to a trained, publicly available vision-language model in under 2 hours (depending on your hardware)!

## 📚 Next Steps

After training your first model:

1. **Evaluate**: Test on various images to see performance
2. **Iterate**: Try different datasets or combine multiple
3. **Share**: Upload to HuggingFace for others to use
4. **Scale**: Use larger datasets for better performance
5. **Specialize**: Focus on specific domains (medical, scientific, etc.)

## 🆘 Getting Help

If you run into issues:

1. Check the [troubleshooting section](./CUSTOM_TRAINING.md#troubleshooting) in the main guide
2. Validate your dataset: `python prepare_custom_dataset.py validate --dataset your_dataset.json`
3. Start with smaller datasets to test your setup
4. Monitor training with W&B to spot issues early

Happy training! 🚀