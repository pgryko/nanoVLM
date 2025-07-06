# Training nanoVLM on Azure ML

This directory contains scripts and configurations for training nanoVLM on Azure Machine Learning.

## Prerequisites

1. **Azure Account**: You need an Azure subscription with access to GPU compute
2. **Azure ML Workspace**: Create an Azure ML workspace in your subscription
3. **Azure CLI**: Install and authenticate with `az login`
4. **Python packages**: Install Azure ML SDK
   ```bash
   pip install azure-ai-ml azure-identity
   ```
5. **Environment variables**: Set these for authentication
   ```bash
   export AZURE_SUBSCRIPTION_ID="your-subscription-id"
   export AZURE_RESOURCE_GROUP="your-resource-group"
   export AZURE_ML_WORKSPACE="your-workspace-name"
   
   # Optional: for logging and model upload
   export WANDB_API_KEY="your-wandb-key"
   export HF_TOKEN="your-huggingface-token"
   ```

## Quick Start

### 1. Setup Azure ML Resources

First, create the compute cluster and environment:

```bash
cd azure
python azure_ml_setup.py
```

This will:
- List available GPU compute sizes
- Create a GPU compute cluster (default: 1x A100 40GB)
- Create a Docker environment with all dependencies

### 2. Submit Training Job

#### Train on public datasets (COCO, VQAv2, etc.)

```bash
python submit_training_job.py \
  --subscription_id $AZURE_SUBSCRIPTION_ID \
  --resource_group $AZURE_RESOURCE_GROUP \
  --workspace_name $AZURE_ML_WORKSPACE \
  --dataset COCO \
  --batch_size 16 \
  --max_training_steps 5000 \
  --push_to_hub \
  --hub_model_id "your-username/nanovlm-coco-model" \
  --wandb_entity your-wandb-username
```

#### Train on custom dataset

First, upload your dataset to Azure ML:

```python
from azure_ml_setup import AzureMLSetup

setup = AzureMLSetup(subscription_id, resource_group, workspace_name)
setup.register_dataset("my-dataset", local_path="./my_dataset_folder")
```

Then submit the job:

```bash
python submit_training_job.py \
  --subscription_id $AZURE_SUBSCRIPTION_ID \
  --resource_group $AZURE_RESOURCE_GROUP \
  --workspace_name $AZURE_ML_WORKSPACE \
  --dataset custom \
  --dataset_name my-dataset \
  --batch_size 16 \
  --max_training_steps 5000
```

### 3. Multi-GPU Training

For faster training with multiple GPUs:

```bash
# 4 GPUs on single node (if using 4x A100 instance)
python submit_training_job.py \
  --subscription_id $AZURE_SUBSCRIPTION_ID \
  --resource_group $AZURE_RESOURCE_GROUP \
  --workspace_name $AZURE_ML_WORKSPACE \
  --dataset THE_CAULDRON \
  --num_gpus 4 \
  --gpus_per_node 4 \
  --batch_size 8 \
  --gradient_accumulation_steps 1
```

## Compute Size Recommendations

Based on your needs and budget:

| VM Size | GPUs | VRAM | Cost/hour | Use Case |
|---------|------|------|-----------|----------|
| Standard_NC6s_v3 | 1x V100 | 16GB | ~$0.90 | Testing, small datasets |
| Standard_NC12s_v3 | 2x V100 | 16GB each | ~$1.80 | Medium datasets |
| Standard_NC24ads_A100_v4 | 1x A100 | 40GB | ~$3.67 | Full training (recommended) |
| Standard_NC96ads_A100_v4 | 4x A100 | 40GB each | ~$14.69 | Fastest training |

## Monitoring and Outputs

1. **Azure ML Studio**: Click the portal URL printed after job submission
2. **Weights & Biases**: If configured, view real-time metrics at wandb.ai
3. **Logs**: Available in Azure ML Studio under "Outputs + logs"
4. **Model**: Saved to Azure ML outputs, can be registered as a model

## Cost Optimization Tips

1. **Use spot instances**: Add `--use_spot` flag (not implemented in current script)
2. **Auto-scaling**: Cluster scales down to 0 when idle
3. **Gradient accumulation**: Use larger effective batch size without more GPUs
4. **Compile mode**: Add `--compile` for ~20% speedup on NVIDIA GPUs

## Troubleshooting

1. **Authentication errors**: Ensure you're logged in with `az login`
2. **Compute creation fails**: Check quota limits in your subscription
3. **Out of memory**: Reduce batch size or use gradient accumulation
4. **Slow training**: Enable compile mode or use more GPUs

## Advanced Configuration

### Custom Docker Image

Modify `Dockerfile` to add dependencies, then rebuild:

```bash
python azure_ml_setup.py --rebuild_environment
```

### Different Model Configurations

```bash
python submit_training_job.py \
  ... \
  --vision_model_id "google/siglip-large-patch16-384" \
  --language_model_id "meta-llama/Llama-3.2-1B-Instruct" \
  --pixel_shuffle_factor 4
```

### Push to HuggingFace Hub

```bash
python submit_training_job.py \
  ... \
  --push_to_hub \
  --hub_model_id "your-username/nanovlm-custom"
```