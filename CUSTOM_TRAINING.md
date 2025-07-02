# Training NanoVLM on Custom Datasets

This guide explains how to train NanoVLM on your own image-text datasets, with support for both Apple Silicon (M1/M2) and NVIDIA GPUs.

## Quick Start

1. **Prepare your dataset** in JSON format
2. **Run training** with the custom training script
3. **Use your model** for inference

## Dataset Format

### Single Image Format

Create a JSON file with the following structure:

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

### Multi-turn Conversations

```json
{
  "image_path": "path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "What's in this image?"},
    {"role": "assistant", "content": "I see a dog playing in a park."},
    {"role": "user", "content": "What color is the dog?"},
    {"role": "assistant", "content": "The dog is golden brown."},
    {"role": "user", "content": "What is it doing?"},
    {"role": "assistant", "content": "The dog is catching a frisbee mid-air."}
  ]
}
```

### Multiple Images Format

For training on multiple images per example:

```json
[
  {
    "image_paths": ["before.jpg", "after.jpg"],
    "conversations": [
      {"role": "user", "content": "Compare these two images."},
      {"role": "assistant", "content": "The first image shows the room before renovation..."}
    ]
  }
]
```

## Preparing Your Dataset

Use the provided helper script to prepare datasets:

### Create a template from image directory:
```bash
python prepare_custom_dataset.py create-simple \
  --image_dir ./my_images \
  --output my_dataset.json \
  --prompt "Describe this image"
```

### Create example datasets:
```bash
python prepare_custom_dataset.py create-examples \
  --output_dir ./examples
```

### Validate your dataset:
```bash
python prepare_custom_dataset.py validate \
  --dataset my_dataset.json \
  --image_root ./my_images
```

### Split into train/validation:
```bash
python prepare_custom_dataset.py split \
  --input my_dataset.json \
  --train_output train.json \
  --val_output val.json \
  --val_ratio 0.1
```

## Training

### Basic Training (M1 Mac)

```bash
python train_custom.py \
  --custom_dataset_path ./my_dataset.json \
  --image_root_dir ./my_images \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_training_steps 1000 \
  --eval_interval 100
```

### Training with NVIDIA GPU

```bash
python train_custom.py \
  --custom_dataset_path ./my_dataset.json \
  --image_root_dir ./my_images \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_training_steps 5000 \
  --eval_interval 200 \
  --compile
```

### Multi-GPU Training (NVIDIA only)

```bash
torchrun --nproc_per_node=4 train_custom.py \
  --custom_dataset_path ./my_dataset.json \
  --image_root_dir ./my_images \
  --batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_training_steps 5000
```

### Resume Training

```bash
python train_custom.py \
  --custom_dataset_path ./my_dataset.json \
  --image_root_dir ./my_images \
  --resume_from_vlm_checkpoint \
  --vlm_checkpoint_path ./checkpoints/my_previous_run
```

## Device-Specific Considerations

### Apple Silicon (M1/M2)

- **Batch size**: Start with 4-8 due to memory constraints
- **Gradient accumulation**: Use 4-8 steps for larger effective batch sizes
- **Compilation**: Skip `--compile` flag (not well supported on MPS)
- **Workers**: The script automatically uses 0 workers for MPS
- **Mixed precision**: Uses float16 automatically

### NVIDIA GPUs

- **Batch size**: Can use larger sizes (16-32) depending on GPU memory
- **Compilation**: Use `--compile` for ~20% speedup
- **Multi-GPU**: Supports distributed training with torchrun
- **Mixed precision**: Uses bfloat16 if available, otherwise float16

### CPU Training

- Not recommended for full training
- Useful for testing dataset loading
- Will automatically use CPU if no GPU available

## Training Parameters

| Parameter | Description | M1 Recommended | GPU Recommended |
|-----------|-------------|----------------|-----------------|
| `--batch_size` | Batch size per device | 4-8 | 16-32 |
| `--gradient_accumulation_steps` | Gradient accumulation | 4-8 | 1-4 |
| `--lr_mp` | Learning rate for modality projector | 0.00512 | 0.00512 |
| `--lr_backbones` | Learning rate for backbones | 5e-5 | 5e-5 |
| `--max_training_steps` | Total training steps | 1000-2000 | 5000-10000 |
| `--eval_interval` | Steps between evaluations | 100 | 200-500 |
| `--max_grad_norm` | Gradient clipping | 1.0 | 1.0 |

## Monitoring Training

### With Weights & Biases

1. **Setup W&B** (first time only):
```bash
pip install wandb
wandb login  # Enter your API key
```

2. **Train with W&B logging**:
```bash
python train_custom.py \
  --custom_dataset_path ./my_dataset.json \
  --log_wandb \
  --wandb_entity your_wandb_username
```

The W&B dashboard will show:
- Real-time loss curves
- Learning rate schedules
- Training/validation metrics
- System metrics (GPU usage, etc.)
- Model checkpoints

### Console Output

The script prints:
- Training loss
- Validation loss
- Learning rates
- Tokens per second
- Best model checkpoint location

## Uploading to Hugging Face Hub

### Method 1: Upload After Training

```bash
# 1. Login to Hugging Face (first time only)
huggingface-cli login

# 2. Upload your trained model
python upload_to_hf.py \
  --checkpoint_path ./checkpoints/nanoVLM_custom_* \
  --repo_name your-username/my-custom-vlm \
  --private  # Optional: make it private
```

### Method 2: Direct Upload from Training Script

Add to your `models/config.py` or modify in training:
```python
vlm_cfg.hf_repo_name = "your-username/my-custom-vlm"
```

The model will automatically upload after training completes.

### Method 3: Manual Upload

```python
from models.vision_language_model import VisionLanguageModel

# Load your model
model = VisionLanguageModel.from_pretrained("./checkpoints/your_model")

# Push to hub
model.push_to_hub("your-username/my-custom-vlm", private=False)
```

### Using Your Uploaded Model

Once uploaded, anyone can use your model:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("your-username/my-custom-vlm", trust_remote_code=True)
```

## Using Your Trained Model

### For inference:

```python
from models.vision_language_model import VisionLanguageModel

# Load your trained model
model = VisionLanguageModel.from_pretrained("./checkpoints/your_run_name")

# Use for generation
python generate.py \
  --vlm_checkpoint_path ./checkpoints/your_run_name \
  --image_path test.jpg \
  --prompt "What's in this image?"
```

### Push to Hugging Face Hub:

```python
model = VisionLanguageModel.from_pretrained("./checkpoints/your_run_name")
model.push_to_hub("username/my-custom-vlm")
```

## Tips for Best Results

1. **Dataset Quality**: Higher quality annotations lead to better models
2. **Dataset Size**: Aim for at least 1000+ examples for meaningful fine-tuning
3. **Learning Rates**: Start with defaults, reduce if training is unstable
4. **Validation**: Always use a validation set to monitor overfitting
5. **Early Stopping**: Stop training when validation loss stops improving

## Troubleshooting

### Out of Memory (M1)
- Reduce `batch_size` to 2-4
- Increase `gradient_accumulation_steps`
- Reduce `max_sample_length` in code

### Slow Training (M1)
- This is expected; M1 is ~5-10x slower than modern GPUs
- Consider using cloud GPUs for large datasets

### CUDA Out of Memory
- Reduce batch size
- Enable gradient checkpointing (modify code)
- Use smaller image size (modify config)

### Dataset Loading Issues
- Ensure all image paths are correct
- Use `prepare_custom_dataset.py validate` to check
- Check image formats (JPG, PNG, WebP supported)

## Example: Fine-tuning on Product Images

```bash
# 1. Prepare dataset
python prepare_custom_dataset.py create-simple \
  --image_dir ./product_images \
  --output products.json \
  --prompt "Describe this product in detail"

# 2. Edit products.json with actual descriptions

# 3. Validate
python prepare_custom_dataset.py validate \
  --dataset products.json \
  --image_root ./product_images

# 4. Train
python train_custom.py \
  --custom_dataset_path products.json \
  --image_root_dir ./product_images \
  --batch_size 8 \
  --max_training_steps 2000 \
  --log_wandb

# 5. Test
python generate.py \
  --vlm_checkpoint_path ./checkpoints/nanoVLM_custom_products_* \
  --image_path ./test_product.jpg \
  --prompt "What product is this and what are its key features?"
```

## Advanced Usage

### Custom Data Preprocessing

Modify `data/custom_dataset.py` to add:
- Custom image augmentations
- Different text preprocessing
- Special tokens or formats

### Multi-Image Training

```bash
python train_custom.py \
  --custom_dataset_path multi_image_dataset.json \
  --multi_image \
  --max_images_per_example 4 \
  --max_images_per_knapsack 18
```

### Different Model Sizes

Edit the config in the training script or create a custom config:
- Larger vision encoder: `vit_model_type = 'google/siglip2-large-patch16-384'`
- Larger language model: `lm_model_type = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'`