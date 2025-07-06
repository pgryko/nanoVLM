#!/bin/bash

# Quick start script for NanoVLM training on Modal.com
# This script helps you get started with training NanoVLM on Modal

set -e

echo "🚀 NanoVLM Modal.com Quick Start"
echo "================================"

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    uv add modal
fi

# Check if Modal is set up
if ! uv run modal app list &> /dev/null; then
    echo "❌ Modal not authenticated. Please run:"
    echo "   modal setup"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "✅ Modal CLI is ready"

# Check if we're in the right directory
if [ ! -f "modal/submit_modal_training.py" ]; then
    echo "❌ Please run this script from the nanoVLM root directory"
    exit 1
fi

# Create example dataset if it doesn't exist
if [ ! -f "datasets/modal_example.json" ]; then
    echo "📝 Creating example dataset..."
    uv run python examples/train_modal_example.py
fi

# Default values
DATASET_PATH="datasets/modal_example.json"
BATCH_SIZE=4
TRAINING_STEPS=500
WANDB_ENTITY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --training_steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset PATH          Path to dataset JSON file (default: datasets/modal_example.json)"
            echo "  --batch_size SIZE       Training batch size (default: 4)"
            echo "  --training_steps STEPS  Number of training steps (default: 500)"
            echo "  --wandb_entity ENTITY   Weights & Biases entity/username"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Quick test with example dataset"
            echo "  $0 --dataset my_data.json            # Train with custom dataset"
            echo "  $0 --wandb_entity myusername         # Enable W&B logging"
            echo "  $0 --batch_size 8 --training_steps 1000  # Custom training config"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset file not found: $DATASET_PATH"
    echo "Create a dataset or use the example:"
    echo "   python examples/train_modal_example.py"
    exit 1
fi

echo "📊 Training Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Training Steps: $TRAINING_STEPS"
if [ -n "$WANDB_ENTITY" ]; then
    echo "  W&B Entity: $WANDB_ENTITY"
fi

# Estimate cost and duration
ESTIMATED_MINUTES=$((TRAINING_STEPS / 100))
ESTIMATED_COST=$(echo "scale=2; $ESTIMATED_MINUTES * 0.1" | bc -l 2>/dev/null || echo "~\$0.50-2.00")

echo ""
echo "⏱️  Estimated Duration: ~$ESTIMATED_MINUTES minutes"
echo "💰 Estimated Cost: ~\$$ESTIMATED_COST"

echo ""
read -p "🤔 Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Build the training command
TRAIN_CMD="uv run python modal/submit_modal_training.py"
TRAIN_CMD="$TRAIN_CMD --custom_dataset_path $DATASET_PATH"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --max_training_steps $TRAINING_STEPS"
TRAIN_CMD="$TRAIN_CMD --eval_interval $((TRAINING_STEPS / 5))"

if [ -n "$WANDB_ENTITY" ]; then
    TRAIN_CMD="$TRAIN_CMD --wandb_entity $WANDB_ENTITY"
fi

echo "🎯 Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Execute the training
eval $TRAIN_CMD

echo ""
echo "🎉 Training completed!"
echo ""
echo "📊 Monitor your training:"
if [ -n "$WANDB_ENTITY" ]; then
    echo "   W&B: https://wandb.ai/$WANDB_ENTITY/nanovlm-modal"
fi
echo "   Modal: https://modal.com/apps"
echo ""
echo "📁 List your trained models:"
echo "   uv run python modal/submit_modal_training.py --list_checkpoints"
