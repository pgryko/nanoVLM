#!/usr/bin/env python3
"""
Submit nanoVLM training job to Azure ML
"""

import os
import argparse
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes


def submit_training_job(
    subscription_id,
    resource_group,
    workspace_name,
    experiment_name="nanovlm-training",
    compute_target="nanovlm-gpu-cluster",
    environment_name="nanovlm-env",
    dataset_name=None,
    training_args=None,
):
    """Submit a training job to Azure ML"""

    # Initialize ML client
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace_name
    )

    # Get environment
    environment = f"{environment_name}:1.0"

    # Prepare inputs
    inputs = {}
    if dataset_name:
        inputs["dataset"] = Input(
            type=AssetTypes.URI_FOLDER, path=f"azureml:{dataset_name}:latest"
        )

    # Prepare command
    command_str = "python azure/train_distributed.py"

    # Add training arguments
    if training_args:
        for key, value in training_args.items():
            if isinstance(value, bool) and value:
                command_str += f" --{key}"
            elif not isinstance(value, bool):
                command_str += f" --{key} {value}"

    # If using custom dataset from inputs
    if dataset_name and training_args.get("dataset") == "custom":
        command_str += " --custom_dataset_path ${{inputs.dataset}}"

    # Create job
    job = command(
        code="./",  # Upload current directory only
        command=command_str,
        environment=environment,
        compute=compute_target,
        experiment_name=experiment_name,
        display_name=f"nanovlm-{training_args.get('dataset', 'custom')}",
        inputs=inputs,
        instance_count=training_args.get("num_gpus", 1),  # Number of nodes
        distribution={
            "type": "PyTorch",
            "process_count_per_instance": training_args.get("gpus_per_node", 1),
        },
        environment_variables={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
    )

    # Submit job
    submitted_job = ml_client.jobs.create_or_update(job)

    print(f"Submitted job: {submitted_job.name}")
    print(f"Portal URL: {submitted_job.studio_url}")

    return submitted_job


def main():
    parser = argparse.ArgumentParser(description="Submit nanoVLM training to Azure ML")

    # Azure ML configuration
    parser.add_argument(
        "--subscription_id", type=str, required=True, help="Azure subscription ID"
    )
    parser.add_argument(
        "--resource_group", type=str, required=True, help="Azure resource group"
    )
    parser.add_argument(
        "--workspace_name", type=str, required=True, help="Azure ML workspace name"
    )
    parser.add_argument("--experiment_name", type=str, default="nanovlm-training")
    parser.add_argument("--compute_target", type=str, default="nanovlm-gpu-cluster")
    parser.add_argument("--environment_name", type=str, default="nanovlm-env")

    # Data configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="COCO",
        choices=["COCO", "VQAv2", "MIXED", "THE_CAULDRON", "custom"],
    )
    parser.add_argument(
        "--dataset_name", type=str, help="Azure ML dataset name (for custom datasets)"
    )

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_training_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--lr_projector", type=float, default=0.00512)
    parser.add_argument("--compile", action="store_true")

    # Distributed training
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Total number of GPUs across all nodes"
    )
    parser.add_argument("--gpus_per_node", type=int, default=1, help="GPUs per node")

    # Logging and outputs
    parser.add_argument("--wandb_entity", type=str, help="W&B entity/username")
    parser.add_argument("--wandb_project", type=str, default="nanovlm-azure")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, help="HuggingFace model ID")

    args = parser.parse_args()

    # Prepare training arguments
    training_args = {
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_training_steps": args.max_training_steps,
        "eval_interval": args.eval_interval,
        "lr_backbone": args.lr_backbone,
        "lr_projector": args.lr_projector,
        "compile": args.compile,
        "num_gpus": args.num_gpus,
        "gpus_per_node": args.gpus_per_node,
    }

    if args.wandb_entity:
        training_args["wandb_entity"] = args.wandb_entity
        training_args["wandb_project"] = args.wandb_project

    if args.push_to_hub and args.hub_model_id:
        training_args["push_to_hub"] = True
        training_args["hub_model_id"] = args.hub_model_id

    # Submit job
    job = submit_training_job(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        experiment_name=args.experiment_name,
        compute_target=args.compute_target,
        environment_name=args.environment_name,
        dataset_name=args.dataset_name,
        training_args=training_args,
    )

    print("\nJob submitted successfully!")
    print(f"Monitor progress at: {job.studio_url}")


if __name__ == "__main__":
    main()
