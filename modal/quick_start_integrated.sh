#!/bin/bash
# Quick start script for integrated dataset building + training on Modal.com

set -e

echo "🚀 NanoVLM Integrated Training on Modal.com"
echo "============================================="

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal not found. Installing..."
    uv add modal
fi

# Check if user is authenticated
if modal token list &> /dev/null; then
    echo "✅ Modal authentication verified"
else
    echo "🔐 Setting up Modal authentication..."
    modal setup
fi

echo ""
echo "🎯 Starting integrated dataset building + training..."
echo ""

# Default configuration for quick start
DATASET_TYPE="mixed"
DATASET_LIMIT=10000
BATCH_SIZE=8
MAX_TRAINING_STEPS=3000
WANDB_ENTITY="piotr-gryko-devalogic"
HUB_MODEL_ID="pgryko/nanovlm-COCO-VQAv2"

echo "Configuration:"
echo "  Dataset: $DATASET_TYPE ($DATASET_LIMIT samples)"
echo "  Batch size: $BATCH_SIZE"
echo "  Training steps: $MAX_TRAINING_STEPS"
echo "  W&B entity: $WANDB_ENTITY"
echo "  Hub model ID: $HUB_MODEL_ID"
echo ""

# Submit the integrated job
uv run python modal/submit_modal_training.py \
  --build_dataset \
  --dataset_type "$DATASET_TYPE" \
  --dataset_limit "$DATASET_LIMIT" \
  --batch_size "$BATCH_SIZE" \
  --max_training_steps "$MAX_TRAINING_STEPS" \
  --compile \
  --wandb_entity "$WANDB_ENTITY" \
  --push_to_hub \
  --hub_model_id "$HUB_MODEL_ID"

echo ""
echo "✅ Training job submitted!"
echo ""
echo "📊 Monitor your training:"
echo "   W&B: https://wandb.ai/$WANDB_ENTITY/nanovlm-modal"
echo "   Modal: https://modal.com/apps"
echo ""
echo "🤗 Your model will be available at:"
echo "   https://huggingface.co/$HUB_MODEL_ID"
echo ""
echo "📝 Features included:"
echo "   ✅ Automatic dataset building on Modal"
echo "   ✅ Comprehensive model card generation"
echo "   ✅ Training configuration documentation"
echo "   ✅ Usage examples and code snippets"
echo "   ✅ Performance metrics and monitoring links"
