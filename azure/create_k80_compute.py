#!/usr/bin/env python3
"""
Create compute cluster with K80 GPU (Standard_NC6)
This uses your existing quota!
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute


def create_k80_compute():
    # Configuration
    subscription_id = "fb992ba5-7179-418e-8b18-65a7e81d5cc1"
    resource_group = "nanovlm-rg"
    workspace_name = "nanovlm-workspace"

    # Initialize ML client
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace_name
    )

    # Create compute with K80 GPU (you have quota for this!)
    compute_config = AmlCompute(
        name="nanovlm-k80-gpu",
        size="Standard_NC6",  # 1x K80 GPU (12GB), 6 vCPUs
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,
    )

    print("Creating K80 GPU compute cluster...")
    print("- VM Size: Standard_NC6")
    print("- GPU: 1x Tesla K80 (12GB VRAM)")
    print("- vCPUs: 6 (within your quota!)")
    print("- Cost: ~$0.90/hour")
    print("- Note: K80 is older but still capable for training")

    try:
        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        print(f"\n✓ Successfully created compute cluster: {compute.name}")
        print(f"  Provisioning state: {compute.provisioning_state}")
        return compute
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    create_k80_compute()
