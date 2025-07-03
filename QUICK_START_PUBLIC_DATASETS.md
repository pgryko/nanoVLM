# Quick Start: Fine-tuning NanoVLM on Public Datasets

This guide provides the fastest way to start fine-tuning NanoVLM on high-quality public datasets.

## 🚀 Quick Start Commands

### ⚡ **RECOMMENDED: Mixed Dataset (Most Reliable)**

```bash
# 1. Make sure you're in your virtual environment
source .venv/bin/activate

# 2. Convert dataset (combines COCO + VQAv2 - most reliable!)
python examples/simple_public_datasets.py \
  --dataset mixed \
  --output datasets/mixed_train.json \
  --limit 8000

# 3. Train on M1 Mac
python train_custom.py \
  --custom_dataset_path datasets/mixed_train.json \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --log_wandb \
  --wandb_entity your_username

# 4. Train on NVIDIA GPU
python train_custom.py \
  --custom_dataset_path datasets/mixed_train.json \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --compile \
  --log_wandb \
  --wandb_entity your_username
```

### Option 1: COCO Captions (Most Reliable for Beginners)

```bash
# 1. Convert dataset (reliable image descriptions)
python examples/simple_public_datasets.py \
  --dataset coco \
  --output datasets/coco_train.json \
  --limit 5000

# 2. Train
python train_custom.py \
  --custom_dataset_path datasets/coco_train.json \
  --batch_size 8 \
  --max_training_steps 2000 \
  --log_wandb
```

### Option 2: VQAv2 Only (Question Answering)

```bash
# 1. Convert dataset
python examples/simple_public_datasets.py \
  --dataset vqav2 \
  --output datasets/vqav2_train.json \
  --limit 6000

# 2. Train
python train_custom.py \
  --custom_dataset_path datasets/vqav2_train.json \
  --batch_size 8 \
  --max_training_steps 3000 \
  --log_wandb
```

### Option 3: Advanced (LLaVA-style) - If You Want Instructions

```bash
# 1. Convert to instruction format (uses VQA as base)
python examples/simple_public_datasets.py \
  --dataset llava \
  --output datasets/instruct_train.json \
  --limit 5000

# 2. Train
python train_custom.py \
  --custom_dataset_path datasets/instruct_train.json \
  --batch_size 8 \
  --max_training_steps 3000 \
  --log_wandb
```

## ⚠️ **Important Notes**

### Use the Reliable Script
Always use `examples/simple_public_datasets.py` instead of `examples/public_datasets.py` - it's more reliable and handles image saving automatically.

### Virtual Environment
Make sure you're in your virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

## 📊 Dataset Overview (Updated with Reliable Options)

| Dataset | Samples | Type | Reliability | Best For | Training Time (M1) | Training Time (GPU) |
|---------|---------|------|-------------|----------|-------------------|-------------------|
| **Mixed (COCO+VQA)** | Combined | Captions + QA | ⭐⭐⭐⭐⭐ | **General purpose (RECOMMENDED)** | 2-3 hours (8K) | 45-60 min (8K) |
| **COCO Captions** | 118K | Image description | ⭐⭐⭐⭐⭐ | Learning to describe images | 1-2 hours (5K) | 30-45 min (5K) |
| **VQAv2** | 200K+ | Question answering | ⭐⭐⭐⭐⭐ | Factual QA | 2-3 hours (6K) | 45-60 min (6K) |
| **Instruction (VQA-based)** | Converted | Instruction following | ⭐⭐⭐⭐ | Instruction following | 2-3 hours (5K) | 45-60 min (5K) |
| ~~LLaVA-Instruct~~ | ~~150K~~ | ~~Instructions~~ | ⭐⭐ | ~~Has loading issues~~ | ~~N/A~~ | ~~N/A~~ |

## 🎯 Recommended Workflows

### For Beginners (Start Here!)
```bash
# Activate environment
source .venv/bin/activate

# Start with reliable COCO dataset
python examples/simple_public_datasets.py --dataset coco --output test.json --limit 1000
python train_custom.py --custom_dataset_path test.json --batch_size 4 --max_training_steps 200
```

### For Production Use (Recommended)
```bash
# Use mixed dataset for best results
python examples/simple_public_datasets.py --dataset mixed --output production.json --limit 10000
python train_custom.py --custom_dataset_path production.json --batch_size 8 --max_training_steps 3000
```

### With W&B Monitoring and HF Upload
```bash
# Set up credentials
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"

# Create mixed dataset
python examples/simple_public_datasets.py \
  --dataset mixed \
  --output datasets/full_training.json \
  --limit 8000

# Train with full monitoring
python train_custom.py \
  --custom_dataset_path datasets/full_training.json \
  --batch_size 8 \
  --max_training_steps 3000 \
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

## 🎉 Complete Example: 0 to Trained Model (Updated)

```bash
#!/bin/bash
# Complete pipeline from scratch - UPDATED FOR RELIABILITY

# 1. Setup (one time)
source .venv/bin/activate  # Make sure you're in the right environment
pip install wandb huggingface_hub
wandb login
huggingface-cli login

# 2. Prepare dataset (using reliable script)
python examples/simple_public_datasets.py \
  --dataset mixed \
  --output datasets/my_dataset.json \
  --limit 8000

# 3. Train with monitoring
python train_custom.py \
  --custom_dataset_path datasets/my_dataset.json \
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

## 🚨 **Key Changes from Previous Instructions**

1. **Use `simple_public_datasets.py`** instead of `public_datasets.py`
2. **Always activate your virtual environment first**
3. **Use `mixed` dataset for best reliability**
4. **Images are automatically saved locally** (no manual downloads needed)

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