# NanoVLM - Production Ready

A 222M parameter Vision-Language Model with complete training infrastructure.

## 🚀 Quick Start

### Local Testing (M1 Mac)
```bash
# Quick validation (30 seconds)
uv run python quick_validation_test.py

# Full local training test (2-5 minutes)  
uv run python test_local_training.py
```

### Modal.com Cloud Training

1. **Setup Modal.com**:
```bash
uv add modal
modal setup
```

2. **Train with your dataset**:
```bash
uv run python modal/submit_modal_training.py \
  --custom_dataset_path your_dataset.json \
  --batch_size 8 \
  --max_training_steps 2000 \
  --eval_interval 200 \
  --wandb_entity your-wandb-username \
  --push_to_hub \
  --hub_model_id your-username/your-model-name
```

## 📊 Features

- ✅ **M1 Mac Compatible** - Local testing and validation
- ✅ **Modal.com Integration** - Serverless GPU training  
- ✅ **W&B Logging** - Complete experiment tracking
- ✅ **HuggingFace Publishing** - Automatic model uploads
- ✅ **Custom Datasets** - JSON format with conversations
- ✅ **Cost Optimized** - Pay only for GPU time used

## 📁 Project Structure

```
├── modal/                  # Modal.com training infrastructure
│   ├── modal_app.py       # Main training app
│   ├── submit_modal_training.py  # Training submission script
│   └── README.md          # Detailed Modal setup guide
├── models/                # Model architecture
├── data/                  # Dataset handling
├── examples/              # Example scripts and datasets
└── README.md             # This file
```

## 💰 Training Costs

| Duration | Steps | Modal Cost |
|----------|-------|------------|
| 5-10 min | 500   | ~$0.50-1.00 |
| 30-60 min | 2000  | ~$3.00-6.00 |
| 2-4 hours | 5000  | ~$12.00-24.00 |

## 📚 Documentation

- **Modal Setup**: `modal/README.md`
- **Dataset Format**: `CUSTOM_TRAINING.md`
- **Examples**: `examples/`

## 🎯 Production Training

For production training, see the complete command in `modal/README.md`.

---

**Ready for production training on Modal.com with W&B logging and HuggingFace publishing!** 🚀
