#!/usr/bin/env python3
"""
Create Azure ML compute cluster with spot instances for reduced cost
"""

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute


def create_spot_compute():
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

    # Define compute with spot instances
    compute_config = AmlCompute(
        name="nanovlm-spot-gpu",
        size="Standard_NC6s_v3",  # 1x V100 16GB GPU, 6 vCPUs
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,  # 2 minutes
        tier="low_priority",  # This enables spot instances
        enable_node_public_ip=True,
    )

    print("Creating spot instance compute cluster...")
    print("- VM Size: Standard_NC6s_v3 (1x V100 16GB)")
    print("- Spot pricing: Up to 90% discount")
    print("- Note: Spot instances can be evicted when Azure needs capacity")

    try:
        # Create the compute
        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        print(f"✓ Created compute cluster: {compute.name}")
        print(f"  Provisioning state: {compute.provisioning_state}")
        return compute
    except Exception as e:
        print(f"Error creating compute: {e}")

        # Try with an even smaller VM if quota is still an issue
        print("\nTrying smaller VM size...")
        compute_config.size = "Standard_NC4as_T4_v3"  # 1x T4 16GB GPU, 4 vCPUs

        try:
            compute = ml_client.compute.begin_create_or_update(compute_config).result()
            print(f"✓ Created compute cluster with T4 GPU: {compute.name}")
            return compute
        except Exception as e2:
            print(f"Error with smaller VM: {e2}")
            raise


if __name__ == "__main__":
    create_spot_compute()
