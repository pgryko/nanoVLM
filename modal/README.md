# Training nanoVLM on Modal.com

Modal provides serverless GPU compute with no quotas or restrictions. You only pay for what you use!

## Prerequisites

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Create Modal account and authenticate**:
   ```bash
   modal setup
   ```
   This will open a browser to authenticate.

3. **Set up secrets** (optional but recommended):
   
   Go to https://modal.com/secrets and create:
   
   - **huggingface** secret with `HF_TOKEN` = your HuggingFace write token
   - **wandb** secret with `WANDB_API_KEY` = your Weights & Biases API key

## Quick Start

### Basic Training (T4 GPU - Cheapest)
```bash
modal run modal/train_modal.py
```

### Training with Different GPUs
```bash
# T4 GPU (~$0.59/hour) - Good for testing
modal run modal/train_modal.py --gpu-type t4

# A10G GPU (~$1.10/hour) - 2x faster than T4  
modal run modal/train_modal.py --gpu-type a10g

# A100 40GB (~$3.70/hour) - Fastest, for production
modal run modal/train_modal.py --gpu-type a100
```

### Full Example with All Options
```bash
modal run modal/train_modal.py \
  --dataset COCO \
  --batch-size 16 \
  --max-training-steps 5000 \
  --gpu-type a10g \
  --wandb-entity your-username \
  --hub-model-id "your-username/nanovlm-modal"
```

## Cost Estimates

| GPU Type | Cost/Hour | Steps/Hour | 5000 Steps Cost | 5000 Steps Time |
|----------|-----------|------------|-----------------|-----------------|
| T4       | $0.59     | ~200       | ~$15            | ~25 hours       |
| A10G     | $1.10     | ~400       | ~$14            | ~12.5 hours     |
| A100     | $3.70     | ~800       | ~$23            | ~6.25 hours     |

## Monitoring

1. **Modal Dashboard**: https://modal.com/apps
   - See running jobs, logs, and GPU usage

2. **Weights & Biases**: If configured, view training metrics at wandb.ai

3. **Logs**: Modal streams logs in real-time to your terminal

## Advanced Usage

### Custom Dataset
```python
# Modify train_modal.py to add custom dataset support
@app.function(...)
def train_nanovlm_custom(
    custom_dataset_path: str,
    ...
):
    # Upload your dataset to Modal volume first
    # Then use train_custom.py instead of train.py
```

### Multi-GPU Training
```python
# Modify GPU configuration for multiple GPUs
gpu=modal.gpu.A100(count=2, size=40),  # 2x A100 GPUs
```

### Persistent Storage
```python
# Add a volume for model checkpoints
volume = modal.Volume.from_name("nanovlm-checkpoints")

@app.function(
    volumes={"/checkpoints": volume},
    ...
)
```

## Advantages over Azure

1. ✅ **No quotas** - GPUs always available
2. ✅ **No setup** - Just run the script
3. ✅ **Pay per second** - Only charged while training
4. ✅ **Easy scaling** - Switch GPUs with one parameter
5. ✅ **Automatic cleanup** - Resources released when done

## Tips

- Start with T4 for testing, then switch to A10G/A100 for full training
- Modal automatically handles retries if spot instances are preempted
- Use `modal app logs` to view logs from past runs
- Set up secrets to avoid putting tokens in code

## Troubleshooting

1. **"Modal not authenticated"**: Run `modal setup`
2. **"Secret not found"**: Create secrets at https://modal.com/secrets
3. **"GPU unavailable"**: Very rare, try a different GPU type
4. **"Out of memory"**: Reduce batch_size or use larger GPU