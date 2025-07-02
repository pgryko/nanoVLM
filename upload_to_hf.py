#!/usr/bin/env python3
"""
Upload a trained NanoVLM model to Hugging Face Hub.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig


def upload_model_to_hub(
    checkpoint_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload NanoVLM model",
    create_model_card: bool = True
):
    """
    Upload a trained NanoVLM model to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to the model checkpoint directory
        repo_name: Name of the HF repository (e.g., "username/model-name")
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        create_model_card: Whether to create a model card
    """
    # Validate checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model to verify it works
    print(f"Loading model from {checkpoint_path}...")
    model = VisionLanguageModel.from_pretrained(str(checkpoint_path))
    config = model.config
    
    # Create repository
    api = HfApi()
    print(f"Creating repository {repo_name}...")
    
    try:
        create_repo(repo_id=repo_name, private=private, exist_ok=True)
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Create model card if requested
    if create_model_card:
        model_card_path = checkpoint_path / "README.md"
        if not model_card_path.exists():
            print("Creating model card...")
            create_model_card_content(model_card_path, repo_name, config)
    
    # Upload the model
    print(f"Uploading model to {repo_name}...")
    try:
        # Method 1: Use the model's built-in push_to_hub
        model.push_to_hub(repo_name, private=private, commit_message=commit_message)
        print(f"Successfully uploaded model to https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error with push_to_hub, trying alternative method: {e}")
        
        # Method 2: Upload folder directly
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"Successfully uploaded model to https://huggingface.co/{repo_name}")


def create_model_card_content(save_path: Path, repo_name: str, config: VLMConfig):
    """Create a model card for the uploaded model."""
    
    # Calculate model size
    param_count_m = 222  # Default, you can calculate from config
    
    content = f"""---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- vision-language-model
- image-to-text
- visual-question-answering
- custom-trained
- nanoVLM
model-index:
- name: {repo_name.split('/')[-1]}
  results: []
---

# {repo_name.split('/')[-1]}

This is a custom-trained NanoVLM model fine-tuned on a custom dataset.

## Model Details

- **Model Type**: Vision-Language Model
- **Base Architecture**: NanoVLM
- **Vision Encoder**: {config.vit_model_type}
- **Language Model**: {config.lm_model_type}
- **Parameters**: ~{param_count_m}M
- **Image Size**: {config.vit_img_size}x{config.vit_img_size}
- **Max Sequence Length**: {config.lm_max_length}

## Usage

### Installation

```bash
pip install torch torchvision pillow transformers
```

### Inference

```python
from models.vision_language_model import VisionLanguageModel
from data.processors import get_image_processor, get_tokenizer
from PIL import Image
import torch

# Load model
model = VisionLanguageModel.from_pretrained("{repo_name}")
model.eval()

# Load processors
image_processor = get_image_processor(model.config.vit_img_size)
tokenizer = get_tokenizer(
    model.config.lm_tokenizer,
    model.config.vlm_extra_tokens,
    model.config.lm_chat_template
)

# Prepare input
image = Image.open("your_image.jpg")
processed_image = image_processor(image)

# Create prompt
messages = [
    {{"role": "user", "content": tokenizer.image_token * model.config.mp_image_token_length + "Describe this image."}}
]

# Tokenize
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        images=[[processed_image]],
        max_length=512,
        do_sample=True,
        temperature=0.7
    )

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

This model was trained using the NanoVLM framework with custom data.

### Training Hyperparameters

- Learning Rate (Projector): {config.lr_mp if hasattr(config, 'lr_mp') else '0.00512'}
- Learning Rate (Backbones): {config.lr_backbones if hasattr(config, 'lr_backbones') else '5e-5'}
- Batch Size: [Varies by hardware]
- Training Steps: [Custom]

## Limitations

- This model is trained on custom data and may have biases present in the training dataset
- Performance on out-of-domain images may vary
- The model has a maximum sequence length of {config.lm_max_length} tokens

## Citation

If you use this model, please cite the original NanoVLM repository:

```bibtex
@misc{{nanovlm2024,
  author = {{Hugging Face}},
  title = {{NanoVLM: A Tiny Vision-Language Model}},
  year = {{2024}},
  publisher = {{GitHub}},
  url = {{https://github.com/huggingface/nanoVLM}}
}}
```

## License

This model is released under the Apache 2.0 license.
"""
    
    with open(save_path, 'w') as f:
        f.write(content)
    
    print(f"Created model card at {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Upload NanoVLM model to Hugging Face Hub")
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--repo_name', type=str, required=True,
                        help='HF repository name (e.g., username/model-name)')
    parser.add_argument('--private', action='store_true',
                        help='Create a private repository')
    parser.add_argument('--commit_message', type=str, 
                        default='Upload custom-trained NanoVLM model',
                        help='Commit message for the upload')
    parser.add_argument('--no_model_card', action='store_true',
                        help='Skip creating model card')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token (or use HF_TOKEN env var)')
    
    args = parser.parse_args()
    
    # Set token if provided
    if args.token:
        os.environ['HF_TOKEN'] = args.token
    
    # Check if user is logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception:
        print("Please login to Hugging Face Hub first:")
        print("  huggingface-cli login")
        print("Or provide a token with --token or HF_TOKEN environment variable")
        return
    
    # Upload model
    upload_model_to_hub(
        checkpoint_path=args.checkpoint_path,
        repo_name=args.repo_name,
        private=args.private,
        commit_message=args.commit_message,
        create_model_card=not args.no_model_card
    )


if __name__ == "__main__":
    main()