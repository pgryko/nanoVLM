#!/usr/bin/env python3
"""
Check Azure ML setup and create workspace if needed
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Workspace
from azure.core.exceptions import ResourceNotFoundError


def main():
    # Azure configuration
    subscription_id = "fb992ba5-7179-418e-8b18-65a7e81d5cc1"
    resource_group = "nanovlm-rg"
    workspace_name = "nanovlm-workspace"
    location = "eastus"

    print(f"Subscription ID: {subscription_id}")
    print(f"Resource Group: {resource_group}")
    print(f"Workspace Name: {workspace_name}")
    print(f"Location: {location}")
    print("-" * 50)

    try:
        # Initialize ML client
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )

        # Try to get the workspace
        try:
            workspace = ml_client.workspaces.get(workspace_name)
            print(f"✅ Azure ML Workspace '{workspace_name}' already exists!")
            print(f"   Location: {workspace.location}")
            print(f"   Resource Group: {workspace.resource_group}")
        except ResourceNotFoundError:
            print(f"❌ Workspace '{workspace_name}' not found. Creating it...")

            # Create workspace
            workspace = Workspace(
                name=workspace_name,
                location=location,
                resource_group=resource_group,
                description="nanoVLM training workspace",
            )

            # Create the workspace
            ml_client_sub = MLClient(DefaultAzureCredential(), subscription_id)
            workspace = ml_client_sub.workspaces.begin_create_or_update(
                workspace
            ).result()
            print(f"✅ Created workspace: {workspace_name}")

        # Check compute targets
        print("\n🖥️  Checking compute targets...")
        computes = list(ml_client.compute.list())
        if computes:
            print("Existing compute targets:")
            for compute in computes:
                print(
                    f"   - {compute.name} ({compute.type}, {compute.size if hasattr(compute, 'size') else 'N/A'})"
                )
        else:
            print("   No compute targets found.")

        # Check environments
        print("\n🐳 Checking environments...")
        environments = list(ml_client.environments.list())
        nanovlm_envs = [env for env in environments if "nanovlm" in env.name.lower()]
        if nanovlm_envs:
            print("nanoVLM environments found:")
            for env in nanovlm_envs:
                print(f"   - {env.name}:{env.version}")
        else:
            print("   No nanoVLM environments found.")

        print("\n✅ Azure ML setup check complete!")
        print("You can now run training jobs using:")
        print(f"   Subscription ID: {subscription_id}")
        print(f"   Resource Group: {resource_group}")
        print(f"   Workspace: {workspace_name}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
