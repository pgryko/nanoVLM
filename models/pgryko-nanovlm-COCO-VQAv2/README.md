---
license: apache-2.0
base_model: lusxvr/nanoVLM-222M
tags:
- vision-language-model
- multimodal
- pytorch
- nanovlm
- modal-trained
- image-captioning
- visual-question-answering
datasets:
- HuggingFaceM4/COCO
- HuggingFaceM4/VQAv2
language:
- en
pipeline_tag: image-to-text
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  text: "What do you see in this image?"
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg
  text: "Describe what's happening in this picture."
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg
  text: "What animals can you see in this image?"
---

# nanovlm-COCO-VQAv2

This is a fine-tuned **NanoVLM** (Nano Vision-Language Model) trained on **Mixed (COCO Captions + VQAv2)** using Modal.com's cloud infrastructure.

## Model Details

- **Base Model**: [lusxvr/nanoVLM-222M](https://huggingface.co/lusxvr/nanoVLM-222M)
- **Model Size**: 222M parameters
- **Architecture**: Vision Transformer (SigLIP) + Small Language Model (SmolLM2)
- **Training Platform**: Modal.com (A100 GPU)
- **Training Date**: 2025-07-06

### Architecture Components

- **Vision Encoder**: SigLIP-B/16-224 (85M parameters)
- **Language Model**: SmolLM2-135M
- **Modality Projection**: Pixel shuffle projection layer
- **Total Parameters**: ~222M

## Training Details

### Dataset
- **Type**: Mixed (COCO Captions + VQAv2)
- **Description**: A balanced combination of COCO image captions and VQAv2 question-answering pairs
- **Size**: 5,000 samples
- **Multi-image Support**: No

### Training Configuration
- **Batch Size**: 8 (effective: 32)
- **Training Steps**: 500
- **Learning Rate (MP)**: 0.00512
- **Learning Rate (Backbones)**: 5e-05
- **Model Compilation**: Enabled
- **Gradient Accumulation**: 4 steps

### Model Configuration
- **Vision Model**: google/siglip2-base-patch16-256
- **Language Model**: HuggingFaceTB/SmolLM2-360M-Instruct
- **Image Size**: 256x256
- **Max Sequence Length**: 1024
- **Image Token Length**: 64

## Usage

### Quick Start

```python
from models.vision_language_model import VisionLanguageModel
from PIL import Image
import requests

# Load the model
model = VisionLanguageModel.from_pretrained("pgryko/nanovlm-COCO-VQAv2")

# Load an image
url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Generate a response
response = model.generate(
    image=image,
    prompt="What do you see in this image?",
    max_length=50
)
print(response)
```

## Training Infrastructure

This model was trained using Modal.com's serverless GPU infrastructure:

- **GPU**: NVIDIA A100-40GB
- **Training Time**: ~60-75 minutes (including dataset preparation)
- **Cost**: ~$6-8 USD
- **Platform**: Modal.com serverless compute

### Reproducibility

To reproduce this training:

```bash
# Using the integrated Modal approach
python modal/submit_modal_training.py \
  --build_dataset \
  --dataset_type mixed \
  --dataset_limit 5000 \
  --batch_size 8 \
  --max_training_steps 500 \
  --compile \
  --push_to_hub \
  --hub_model_id your-username/your-model-name
```

## Monitoring

Training metrics and logs are available on Weights & Biases:
- **Project**: [piotr-gryko-devalogic/nanovlm-modal](https://wandb.ai/piotr-gryko-devalogic/nanovlm-modal)


## Limitations

- **Context Length**: Limited to 1024 tokens
- **Image Resolution**: Fixed at 256x256 pixels
- **Language**: Primarily English
- **Domain**: General vision-language tasks (performance may vary on specialized domains)

## Ethical Considerations

This model inherits potential biases from its training datasets (COCO, VQAv2). Users should be aware of potential limitations in:
- Representation of diverse populations
- Cultural and geographic biases
- Object and scene recognition across different contexts

## Citation

```bibtex
@misc{pgryko_nanovlm_COCO_VQAv2,
  title={NanoVLM Fine-tuned on Mixed (COCO Captions + VQAv2)},
  author={Modal.com Training Pipeline},
  year={2024},
  url={https://huggingface.co/pgryko/nanovlm-COCO-VQAv2}
}
```

## Acknowledgments

- **Base Model**: [nanoVLM](https://github.com/huggingface/nanoVLM) by HuggingFace
- **Training Platform**: [Modal.com](https://modal.com) for serverless GPU compute
- **Datasets**: Microsoft COCO and VQAv2 teams
- **Infrastructure**: NVIDIA A100 GPU via Modal.com

---

*This model was trained using an automated pipeline on Modal.com. For questions or issues, please refer to the [nanoVLM repository](https://github.com/huggingface/nanoVLM).*
