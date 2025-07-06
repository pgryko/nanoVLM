#!/bin/bash
# Optimized Modal training for NanoVLM with improved reliability

set -e

echo "🚀 NanoVLM Optimized Training on Modal.com"
echo "==========================================="

# Check authentication
if modal token list &> /dev/null 2>&1; then
    echo "✅ Modal authentication verified"
else
    echo "🔐 Setting up Modal authentication..."
    modal setup
fi

echo ""
echo "🎯 Starting optimized training with improvements:"
echo "   ✅ Extended 6-hour timeout"
echo "   ✅ Checkpoint resumption capability"
echo "   ✅ More frequent progress updates"
echo "   ✅ Enhanced error handling"
echo "   ✅ Automatic volume commits"
echo "   ✅ Detached mode (continues if you disconnect)"
echo ""

# Optimized configuration for reliability
DATASET_TYPE="mixed"
DATASET_LIMIT=10000
BATCH_SIZE=8
MAX_TRAINING_STEPS=500
WANDB_ENTITY="piotr-gryko-devalogic"
HUB_MODEL_ID="pgryko/nanovlm-COCO-VQAv2"

echo "Configuration:"
echo "  Dataset: $DATASET_TYPE ($DATASET_LIMIT samples)"
echo "  Batch size: $BATCH_SIZE (effective: 32 with gradient accumulation)"
echo "  Training steps: $MAX_TRAINING_STEPS"
echo "  Expected duration: ~60-90 minutes"
echo "  W&B entity: $WANDB_ENTITY"
echo "  Hub model ID: $HUB_MODEL_ID"
echo ""

# Submit the optimized training job
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
echo ""
echo "✅ Training job submitted with optimizations!"
echo ""
echo "💡 IMPORTANT: Robust training mode active"
echo "   - Handles disconnections gracefully"
echo "   - Job continues running on Modal even if you disconnect"
echo "   - Better error recovery and reporting"
echo ""
echo "📊 Monitor your training:"
echo "   W&B: https://wandb.ai/$WANDB_ENTITY/nanovlm-modal"
echo "   Modal: https://modal.com/apps"
echo "   Status: uv run modal app list"
echo ""
echo "🤗 Your model will be available at:"
echo "   https://huggingface.co/$HUB_MODEL_ID"
echo ""
echo "🔧 Improvements in this version:"
echo "   - Extended timeout (6 hours vs 4 hours)"
echo "   - Checkpoint resumption if interrupted"
echo "   - Progress updates every 25 steps"
echo "   - Checkpoints saved every 100 steps"
echo "   - Automatic volume commits for persistence"
echo "   - Enhanced error handling and logging"
echo "   - Robust disconnection handling"
echo "   - Signal handling for graceful exits"
