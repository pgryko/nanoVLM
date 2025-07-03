#!/usr/bin/env python3
"""
Submit nanoVLM training job using Azure ML Serverless Compute
No quota required - pay per use
"""

import os
import argparse
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential


def submit_serverless_job(args):
    """Submit a training job using serverless compute"""

    # Initialize ML client
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    # Prepare command
    command_str = "python azure/train_distributed.py"

    # Add training arguments
    training_args = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_training_steps": args.max_training_steps,
        "eval_interval": args.eval_interval,
        "lr_backbone": args.lr_backbone,
        "lr_projector": args.lr_projector,
        "compile": args.compile,
    }

    if args.wandb_entity:
        training_args["wandb_entity"] = args.wandb_entity
        training_args["wandb_project"] = args.wandb_project

    if args.push_to_hub and args.hub_model_id:
        training_args["push_to_hub"] = True
        training_args["hub_model_id"] = args.hub_model_id

    for key, value in training_args.items():
        if isinstance(value, bool) and value:
            command_str += f" --{key}"
        elif not isinstance(value, bool):
            command_str += f" --{key} {value}"

    # Create job with serverless compute
    job = command(
        code="../",  # Upload entire nanoVLM directory
        command=command_str,
        environment="nanovlm-env:1.0",
        experiment_name="nanovlm-serverless",
        display_name=f"nanovlm-{args.dataset}-serverless",
        # Use serverless compute by specifying instance type
        instance_type="Standard_NC6s_v3",  # V100 16GB GPU
        instance_count=1,
        environment_variables={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
    )

    print("Submitting job with serverless compute...")
    print("- Instance type: Standard_NC6s_v3 (1x V100 16GB)")
    print("- No quota required - billed per minute")
    print("- Approximate cost: $0.015/minute ($0.90/hour)")

    try:
        # Submit job
        submitted_job = ml_client.jobs.create_or_update(job)

        print(f"\n✓ Submitted job: {submitted_job.name}")
        print(f"Portal URL: {submitted_job.studio_url}")

        return submitted_job
    except Exception as e:
        print(f"\nError: {e}")
        print("\nServerless compute might not be available in your region.")
        print("Alternative options:")
        print(
            "1. Request quota increase: https://docs.microsoft.com/azure/machine-learning/how-to-manage-quotas"
        )
        print("2. Try a different region with available quota")
        print("3. Use Azure Container Instances or Azure Batch instead")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Submit nanoVLM training with serverless compute"
    )

    # Azure ML configuration
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)

    # Training configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="COCO",
        choices=["COCO", "VQAv2", "MIXED", "THE_CAULDRON", "custom"],
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_training_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_projector", type=float, default=0.00512)
    parser.add_argument("--compile", action="store_true")

    # Logging and outputs
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str, default="nanovlm-azure")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)

    args = parser.parse_args()

    # Submit job
    job = submit_serverless_job(args)

    print("\nJob submitted successfully!")
    print(
        "Note: Serverless compute automatically scales and you only pay for what you use."
    )


if __name__ == "__main__":
    main()
