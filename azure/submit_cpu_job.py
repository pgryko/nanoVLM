#!/usr/bin/env python3
"""
Submit nanoVLM training job using CPU (no GPU quota required)
Much slower but works within student subscription limits
"""

import os
import argparse
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential


def submit_cpu_job(args):
    """Submit a CPU training job using Azure Container Instances"""

    # Initialize ML client
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    # Prepare command - reduce batch size for CPU
    command_str = "python azure/train_distributed.py"

    # CPU-optimized training arguments
    training_args = {
        "dataset": args.dataset,
        "batch_size": 1,  # Very small batch for CPU
        "gradient_accumulation_steps": 16,  # Simulate larger batch
        "max_training_steps": min(args.max_training_steps, 500),  # Shorter for CPU
        "eval_interval": 50,  # More frequent eval
        "lr_backbone": 1e-4,  # Higher LR for faster convergence
        "lr_projector": 0.01,
        "compile": False,  # Don't compile on CPU
    }

    if args.wandb_entity:
        training_args["wandb_entity"] = args.wandb_entity
        training_args["wandb_project"] = "nanovlm-cpu"

    if args.push_to_hub and args.hub_model_id:
        training_args["push_to_hub"] = True
        training_args["hub_model_id"] = args.hub_model_id

    for key, value in training_args.items():
        if isinstance(value, bool) and value:
            command_str += f" --{key}"
        elif not isinstance(value, bool):
            command_str += f" --{key} {value}"

    # Create job with CPU-only compute
    job = command(
        code="../",
        command=command_str,
        environment="nanovlm-env:2.0",
        experiment_name="nanovlm-cpu",
        display_name=f"nanovlm-{args.dataset}-cpu",
        # CPU instance (no GPU quota needed)
        instance_type="Standard_E4s_v3",  # 4 vCPUs, 32GB RAM
        instance_count=1,
        environment_variables={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
    )

    print("Submitting CPU training job...")
    print("- Instance type: Standard_E4s_v3 (4 vCPUs, 32GB RAM)")
    print("- No GPU quota required")
    print("- Cost: ~$0.20/hour")
    print("- Expected time: 12-24 hours (much slower than GPU)")
    print("- Batch size reduced to 1 for CPU training")

    try:
        submitted_job = ml_client.jobs.create_or_update(job)
        print(f"\n✓ Submitted job: {submitted_job.name}")
        print(f"Portal URL: {submitted_job.studio_url}")
        return submitted_job
    except Exception as e:
        print(f"\nError: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Submit CPU-only nanoVLM training")

    # Azure ML configuration
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)

    # Training configuration (CPU optimized)
    parser.add_argument(
        "--dataset",
        type=str,
        default="COCO",
        choices=["COCO", "VQAv2", "MIXED", "THE_CAULDRON", "custom"],
    )
    parser.add_argument(
        "--max_training_steps",
        type=int,
        default=500,
        help="Max steps (reduced for CPU training)",
    )

    # Logging and outputs
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)

    args = parser.parse_args()

    print("⚠️  CPU-only training is MUCH slower than GPU training!")
    print("💡 Consider requesting GPU quota for faster training.")
    print("📖 See azure/request_quota.md for instructions.\n")

    job = submit_cpu_job(args)

    print("\nCPU job submitted successfully!")
    print("Note: This will take 12-24 hours to complete.")


if __name__ == "__main__":
    main()
