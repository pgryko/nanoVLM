#!/usr/bin/env python3
"""
Submit nanoVLM training job using spot instances for maximum cost savings
"""

import os
import argparse
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute


def create_spot_compute_if_needed(ml_client, compute_name="nanovlm-spot-gpu"):
    """Create spot compute cluster if it doesn't exist"""
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"✅ Spot compute '{compute_name}' already exists")
        return compute
    except:
        print(f"Creating spot compute cluster: {compute_name}")

        # Create spot compute with K80 GPU (available in your quota)
        compute_config = AmlCompute(
            name=compute_name,
            size="Standard_NC6",  # K80 GPU - you have quota for this
            min_instances=0,  # Scale to zero when idle
            max_instances=1,  # Single instance
            idle_time_before_scale_down=300,  # 5 minutes
            tier="low_priority",  # Spot pricing
        )

        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        print(f"✅ Created spot compute: {compute_name}")
        return compute


def submit_spot_training_job(args):
    """Submit training job optimized for spot instances"""

    # Initialize ML client
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    # Create or get spot compute
    compute = create_spot_compute_if_needed(ml_client, args.compute_target)

    # Spot-optimized training arguments
    training_args = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_training_steps": args.max_training_steps,
        "eval_interval": args.eval_interval,
        "lr_backbone": args.lr_backbone,
        "lr_projector": args.lr_projector,
        "compile": args.compile,
        # Spot-specific optimizations
        "save_steps": 100,  # Save checkpoints frequently
        "logging_steps": 10,  # Log frequently for monitoring
        "resume_from_checkpoint": True,  # Enable checkpoint resuming
    }

    # Add W&B logging
    if args.wandb_entity:
        training_args["wandb_entity"] = args.wandb_entity
        training_args["wandb_project"] = args.wandb_project

    # Build command
    command_str = "python azure/train_distributed.py"
    for key, value in training_args.items():
        if isinstance(value, bool) and value:
            command_str += f" --{key}"
        elif not isinstance(value, bool):
            command_str += f" --{key} {value}"

    # Create job with spot-optimized settings
    job = command(
        code="./",
        command=command_str,
        environment=f"{args.environment_name}:1.0",
        compute=args.compute_target,
        experiment_name=f"{args.experiment_name}-spot",
        display_name=f"nanovlm-spot-{args.dataset.lower()}",
        environment_variables={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
    )

    # Submit job
    submitted_job = ml_client.jobs.create_or_update(job)

    print("\n🎯 Spot Training Job Submitted!")
    print(f"Job Name: {submitted_job.name}")
    print(f"Portal URL: {submitted_job.studio_url}")
    print(f"Compute: {args.compute_target} (Spot Instance)")
    print("Expected Cost: ~$0.05-0.15/hour (vs $0.50-1.50/hour regular)")

    print("\n💡 Spot Instance Tips:")
    print("- Monitor job progress closely")
    print("- Job may be interrupted - will resume from last checkpoint")
    print("- Use smaller training steps for faster completion")
    print("- Consider running multiple short jobs vs one long job")

    return submitted_job


def main():
    parser = argparse.ArgumentParser(description="Submit spot training job")

    # Azure configuration
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)
    parser.add_argument("--compute_target", default="nanovlm-spot-gpu")
    parser.add_argument("--environment_name", default="nanovlm-env")
    parser.add_argument("--experiment_name", default="nanovlm-spot-training")

    # Training configuration (spot-optimized defaults)
    parser.add_argument("--dataset", default="COCO", choices=["COCO", "VQAv2", "MIXED"])
    parser.add_argument(
        "--batch_size", type=int, default=4
    )  # Smaller for faster completion
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8
    )  # Maintain effective batch size
    parser.add_argument("--max_training_steps", type=int, default=1000)  # Shorter runs
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_projector", type=float, default=0.00512)
    parser.add_argument("--compile", action="store_true")

    # Logging
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", default="nanovlm-spot")

    args = parser.parse_args()

    print("🔥 Spot Instance Training - Maximum Cost Savings!")
    print(f"Dataset: {args.dataset}")
    print(
        f"Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})"
    )
    print(f"Training Steps: {args.max_training_steps}")
    print("Expected Duration: ~10-30 minutes")
    print("Expected Cost: ~$0.10-0.50 total")

    submit_spot_training_job(args)


if __name__ == "__main__":
    main()
