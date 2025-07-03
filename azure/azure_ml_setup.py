#!/usr/bin/env python3
"""
Azure ML setup script for nanoVLM training
"""

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data


class AzureMLSetup:
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name

        # Initialize ML client
        self.ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )

    def create_compute_target(
        self,
        compute_name="gpu-cluster",
        vm_size="Standard_NC24ads_A100_v4",
        min_instances=0,
        max_instances=4,
    ):
        """Create GPU compute cluster for training"""

        try:
            # Check if compute already exists
            compute = self.ml_client.compute.get(compute_name)
            print(f"Compute cluster '{compute_name}' already exists.")
        except:
            # Create new compute
            compute = AmlCompute(
                name=compute_name,
                size=vm_size,  # A100 40GB GPU
                min_instances=min_instances,
                max_instances=max_instances,
                idle_time_before_scale_down=120,  # Scale down after 2 minutes
            )
            compute = self.ml_client.compute.begin_create_or_update(compute).result()
            print(f"Created compute cluster: {compute_name}")

        return compute

    def create_environment(
        self, env_name="nanovlm-env", dockerfile_path="./Dockerfile"
    ):
        """Create training environment with all dependencies"""

        env = Environment(
            name=env_name,
            description="nanoVLM training environment with PyTorch and dependencies",
            build=BuildContext(path=".", dockerfile_path=dockerfile_path),
            version="1.0",
        )

        env = self.ml_client.environments.create_or_update(env)
        print(f"Created environment: {env_name}")
        return env

    def register_dataset(self, dataset_name, local_path=None, uri=None):
        """Register dataset in Azure ML"""

        if local_path:
            # Upload local dataset
            data_asset = Data(
                name=dataset_name,
                description="nanoVLM training dataset",
                type=AssetTypes.URI_FOLDER,
                path=local_path,
            )
        elif uri:
            # Use remote dataset
            data_asset = Data(
                name=dataset_name,
                description="nanoVLM training dataset",
                type=AssetTypes.URI_FOLDER,
                path=uri,
            )
        else:
            raise ValueError("Either local_path or uri must be provided")

        data_asset = self.ml_client.data.create_or_update(data_asset)
        print(f"Registered dataset: {dataset_name}")
        return data_asset

    def get_compute_sizes(self, gpu_only=True):
        """List available compute sizes"""

        sizes = self.ml_client.compute.list_sizes()

        gpu_sizes = []
        for size in sizes:
            if gpu_only and size.gpus > 0:
                gpu_sizes.append(
                    {
                        "name": size.name,
                        "gpus": size.gpus,
                        "vcpus": getattr(
                            size, "v_cpus", getattr(size, "v_cp_us", "N/A")
                        ),
                        "memory_gb": size.memory_gb,
                    }
                )

        # Sort by number of GPUs
        gpu_sizes.sort(key=lambda x: x["gpus"], reverse=True)

        return gpu_sizes


def main():
    # Configuration
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE")

    if not all([subscription_id, resource_group, workspace_name]):
        print(
            "Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_ML_WORKSPACE environment variables"
        )
        return

    # Initialize setup
    setup = AzureMLSetup(subscription_id, resource_group, workspace_name)

    # List available GPU compute sizes
    print("\nAvailable GPU compute sizes:")
    gpu_sizes = setup.get_compute_sizes()
    for size in gpu_sizes[:10]:  # Show top 10
        print(
            f"- {size['name']}: {size['gpus']} GPU(s), {size['vcpus']} vCPUs, {size['memory_gb']}GB RAM"
        )

    # Recommended sizes for nanoVLM
    print("\nRecommended sizes for nanoVLM:")
    print("- Standard_NC6s_v3: 1x V100 16GB GPU (~$0.90/hour) - Good for testing")
    print(
        "- Standard_NC12s_v3: 2x V100 16GB GPU (~$1.80/hour) - Good for small datasets"
    )
    print(
        "- Standard_NC24ads_A100_v4: 1x A100 40GB GPU (~$3.67/hour) - Best for full training"
    )
    print(
        "- Standard_NC96ads_A100_v4: 4x A100 40GB GPU (~$14.69/hour) - Fastest training"
    )

    # Create compute cluster (using smaller VM due to quota limits)
    compute = setup.create_compute_target(
        compute_name="nanovlm-gpu-cluster",
        vm_size="Standard_NC6s_v3",  # 1x V100 16GB, 6 vCPUs (fits in 20 vCPU quota)
        min_instances=0,
        max_instances=1,
    )

    # Create environment
    env = setup.create_environment()

    print("\nSetup complete! You can now submit training jobs.")


if __name__ == "__main__":
    main()
