#!/usr/bin/env python3
"""
Rebuild Azure ML environment with fixed Dockerfile
"""

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext


def rebuild_environment():
    # Get configuration from environment
    subscription_id = os.getenv(
        "AZURE_SUBSCRIPTION_ID", "fb992ba5-7179-418e-8b18-65a7e81d5cc1"
    )
    resource_group = os.getenv("AZURE_RESOURCE_GROUP", "nanovlm-rg")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE", "nanovlm-workspace")

    # Initialize ML client
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace_name
    )

    # Create new environment version
    env = Environment(
        name="nanovlm-env",
        description="nanoVLM training environment with PyTorch and dependencies (fixed)",
        build=BuildContext(path=".", dockerfile_path="./Dockerfile"),
        version="2.0",  # New version
    )

    print("Creating new environment version with fixed Dockerfile...")
    env = ml_client.environments.create_or_update(env)
    print(f"Created environment: {env.name}:{env.version}")
    return env


if __name__ == "__main__":
    rebuild_environment()
