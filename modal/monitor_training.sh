#!/bin/bash
# Monitor NanoVLM training jobs on Modal.com

echo "📊 NanoVLM Training Monitor"
echo "=========================="

# Check if Modal is available
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: uv add modal"
    exit 1
fi

# Check authentication
if ! modal token list &> /dev/null 2>&1; then
    echo "❌ Not authenticated with Modal. Run: modal setup"
    exit 1
fi

echo "✅ Modal CLI ready"
echo ""

# List running apps
echo "🔍 Checking for running Modal apps..."
modal app list

echo ""
echo "📈 Quick links for monitoring:"
echo "   Modal Dashboard: https://modal.com/apps"
echo "   W&B Dashboard: https://wandb.ai/piotr-gryko-devalogic/nanovlm-modal"
echo "   HuggingFace: https://huggingface.co/pgryko/nanovlm-COCO-VQAv2"
echo ""

# Ask if user wants to see logs
read -p "Do you want to see logs for a specific app? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Enter the app ID from the list above:"
    read -r app_id
    if [ -n "$app_id" ]; then
        echo "📋 Showing logs for app: $app_id"
        modal app logs "$app_id"
    fi
fi

echo ""
echo "💡 Useful Modal commands:"
echo "   modal app list                    # List all apps"
echo "   modal app logs <app-id>          # Show logs for specific app"
echo "   modal app stop <app-id>          # Stop a running app"
echo "   modal volume list                # List volumes"
echo "   modal volume ls nanovlm-data     # List files in training volume"
