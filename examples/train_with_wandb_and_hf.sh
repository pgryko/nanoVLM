#!/bin/bash
# Example script showing how to train with W&B logging and upload to Hugging Face

# Set your credentials (or use environment variables)
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"

# Train with W&B logging
python train_custom.py \
  --custom_dataset_path ./my_dataset.json \
  --image_root_dir ./my_images \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --log_wandb \
  --wandb_entity "your_wandb_username" \
  --vlm_checkpoint_path "./checkpoints"

# The model will be saved to ./checkpoints/nanoVLM_custom_[timestamp]
# You can find the exact path in the training output