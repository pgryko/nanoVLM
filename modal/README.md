# Training NanoVLM on Modal.com

This directory contains scripts and configurations for training NanoVLM on Modal.com, a serverless cloud platform optimized for AI workloads.

## 🚀 Why Modal.com?

- **Cost-effective**: Pay only for compute time used, no idle costs
- **Fast startup**: Containers start in seconds, not minutes
- **GPU access**: Easy access to A100, H100, and other high-end GPUs
- **Serverless**: No infrastructure management required
- **Great for experimentation**: Perfect for iterative model development

## Prerequisites

### 1. Automated Setup (Recommended)

```bash
# Run the setup helper - it will guide you through everything
python modal/setup_modal.py
```

This will:
- Install Modal if needed
- Guide you through authentication
- Help set up W&B and HuggingFace secrets
- Run validation tests

### 2. Manual Setup

If you prefer manual setup:

```bash
# Install Modal
uv add modal

# Authenticate
modal setup
```

### 3. Set up Secrets

Create secrets in Modal dashboard for API keys:

**Weights & Biases Secret** (optional but recommended):
- Name: `wandb-secret`
- Key: `WANDB_API_KEY`
- Value: Your W&B API key from https://wandb.ai/authorize

**HuggingFace Secret** (optional):
- Name: `huggingface-secret`
- Key: `HF_TOKEN`
- Value: Your HF token from https://huggingface.co/settings/tokens

### 4. Validate Setup

```bash
# Test your Modal setup
uv run python modal/test_modal_setup.py
```

## 🎯 Quick Start

### 1. Prepare Your Dataset

Create a JSON file with your custom dataset:

```json
[
    {
        "image_path": "path/to/image1.jpg",
        "conversations": [
            {"role": "user", "content": "What do you see in this image?"},
            {"role": "assistant", "content": "I see a cat sitting on a mat."}
        ]
    },
    {
        "image_path": "path/to/image2.jpg", 
        "conversations": [
            {"role": "user", "content": "Describe this scene."},
            {"role": "assistant", "content": "This is a beautiful sunset over the ocean."}
        ]
    }
]
```

### 2. Submit Training Job

```bash
cd modal

# Basic training
python submit_modal_training.py \
  --custom_dataset_path ../datasets/my_dataset.json \
  --batch_size 8 \
  --max_training_steps 2000 \
  --wandb_entity your_username

# Advanced training with custom settings
python submit_modal_training.py \
  --custom_dataset_path ../datasets/my_dataset.json \
  --image_root_dir ../datasets/images \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_training_steps 5000 \
  --lr_mp 0.01 \
  --lr_backbones 1e-4 \
  --compile \
  --wandb_entity your_username \
  --wandb_project my-nanovlm-project
```

### 3. Monitor Training

- **Modal Dashboard**: https://modal.com/apps
- **Weights & Biases**: https://wandb.ai/your_username/nanovlm-modal

### 4. Access Trained Models

```bash
# List available checkpoints
python submit_modal_training.py --list_checkpoints
```

## 📊 Training Configuration

### Recommended Settings by Use Case

**🧪 Quick Experimentation (5-10 minutes)**
```bash
python submit_modal_training.py \
  --custom_dataset_path dataset.json \
  --batch_size 4 \
  --max_training_steps 500 \
  --eval_interval 100
```

**🎯 Standard Training (30-60 minutes)**
```bash
python submit_modal_training.py \
  --custom_dataset_path dataset.json \
  --batch_size 8 \
  --max_training_steps 2000 \
  --eval_interval 200 \
  --compile
```

**🚀 Production Training (2-4 hours)**
```bash
python submit_modal_training.py \
  --custom_dataset_path dataset.json \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --max_training_steps 10000 \
  --eval_interval 500 \
  --compile
```

### GPU and Cost Estimates

| Configuration | GPU | Duration | Estimated Cost |
|---------------|-----|----------|----------------|
| Quick Experiment | A100 | 5-10 min | $0.50-1.00 |
| Standard Training | A100 | 30-60 min | $3.00-6.00 |
| Production Training | A100 | 2-4 hours | $12.00-24.00 |

*Costs are approximate and based on Modal's A100 pricing*

## 🛠️ Advanced Usage

### Multi-Image Training

For datasets with multiple images per example:

```json
[
    {
        "image_paths": ["image1.jpg", "image2.jpg"],
        "conversations": [
            {"role": "user", "content": "Compare these two images."},
            {"role": "assistant", "content": "The first shows... while the second..."}
        ]
    }
]
```

```bash
python submit_modal_training.py \
  --custom_dataset_path multi_image_dataset.json \
  --multi_image \
  --batch_size 4
```

### Custom Image Root Directory

If your JSON contains relative paths:

```bash
python submit_modal_training.py \
  --custom_dataset_path dataset.json \
  --image_root_dir /path/to/images
```

### Model Compilation

Enable PyTorch compilation for faster training:

```bash
python submit_modal_training.py \
  --custom_dataset_path dataset.json \
  --compile
```

## 🔧 Troubleshooting

### Common Issues

**1. Dataset Upload Fails**
- Ensure your dataset JSON is valid
- Check that image paths are correct
- Verify file permissions

**2. Out of Memory Errors**
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Disable `--compile` if enabled

**3. Training Stalls**
- Check Modal dashboard for logs
- Verify W&B logging is working
- Ensure dataset is not corrupted

**4. Slow Training**
- Enable `--compile` for faster training
- Increase batch size if memory allows
- Check if images are too large

### Getting Help

1. Check Modal logs in the dashboard
2. Monitor W&B for training metrics
3. Use `--list_checkpoints` to verify model saving
4. Check the Modal community forum

## 📈 Best Practices

1. **Start Small**: Begin with a small dataset and few steps
2. **Monitor Closely**: Use W&B to track training progress
3. **Save Frequently**: Use reasonable `--eval_interval` values
4. **Optimize Gradually**: Start with default settings, then tune
5. **Use Compilation**: Enable `--compile` for production runs

## 🔗 Useful Links

- [Modal Documentation](https://modal.com/docs)
- [NanoVLM Repository](https://github.com/huggingface/nanoVLM)
- [Weights & Biases](https://wandb.ai)
- [HuggingFace Hub](https://huggingface.co)
