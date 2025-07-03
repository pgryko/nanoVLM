# Azure ML Account Setup Guide

## Finding Existing Azure Resources

### 1. Get Your Subscription ID

**Option A: Azure Portal**
1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for "Subscriptions" in the top search bar
3. Click on your subscription name
4. Copy the "Subscription ID" (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

**Option B: Azure CLI**
```bash
# Install Azure CLI if not already installed
# macOS: brew install azure-cli
# Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
# Windows: Download from https://aka.ms/installazurecliwindows

# Login to Azure
az login

# List all subscriptions
az account list --output table

# Get current subscription
az account show --query id -o tsv
```

### 2. Find or Create Resource Group

**Find existing resource groups:**
```bash
# List all resource groups
az group list --output table

# Or in Azure Portal:
# Search for "Resource groups" and view the list
```

**Create a new resource group:**
```bash
# Create resource group (choose a region close to you)
az group create --name "nanovlm-rg" --location "eastus"

# Other common locations: westus2, westeurope, southeastasia
# List all locations: az account list-locations -o table
```

### 3. Find or Create ML Workspace

**Check for existing ML workspace:**
```bash
# List ML workspaces in a resource group
az ml workspace list --resource-group "nanovlm-rg" --output table
```

**Create a new ML workspace:**
```bash
# Create ML workspace
az ml workspace create \
  --name "nanovlm-workspace" \
  --resource-group "nanovlm-rg" \
  --location "eastus"
```

## Complete Setup Script

Here's a complete script to set everything up:

```bash
#!/bin/bash

# Set your desired names
RESOURCE_GROUP="nanovlm-rg"
WORKSPACE_NAME="nanovlm-workspace"
LOCATION="eastus"  # Change to your preferred region

# Login to Azure
echo "Logging in to Azure..."
az login

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"

# Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Install ML extension if needed
echo "Installing Azure ML extension..."
az extension add -n ml

# Create ML workspace
echo "Creating ML workspace..."
az ml workspace create \
  --name $WORKSPACE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Export environment variables
echo ""
echo "Setup complete! Add these to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "export AZURE_SUBSCRIPTION_ID=\"$SUBSCRIPTION_ID\""
echo "export AZURE_RESOURCE_GROUP=\"$RESOURCE_GROUP\""
echo "export AZURE_ML_WORKSPACE=\"$WORKSPACE_NAME\""
```

## Quick Setup Commands

If you want to create everything from scratch quickly:

```bash
# 1. Install Azure CLI and ML extension
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az extension add -n ml

# 2. Login
az login

# 3. Create everything with one command
SUBSCRIPTION_ID=$(az account show --query id -o tsv) && \
az group create --name "nanovlm-rg" --location "eastus" && \
az ml workspace create --name "nanovlm-workspace" --resource-group "nanovlm-rg"

# 4. Set environment variables
export AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
export AZURE_RESOURCE_GROUP="nanovlm-rg"
export AZURE_ML_WORKSPACE="nanovlm-workspace"
```

## Verify Setup

Test that everything is working:

```bash
# Check connection to workspace
az ml workspace show \
  --name $AZURE_ML_WORKSPACE \
  --resource-group $AZURE_RESOURCE_GROUP

# Install Python packages
pip install azure-ai-ml azure-identity

# Test Python connection
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

ml_client = MLClient(
    DefaultAzureCredential(),
    os.environ['AZURE_SUBSCRIPTION_ID'],
    os.environ['AZURE_RESOURCE_GROUP'],
    os.environ['AZURE_ML_WORKSPACE']
)
print('Connected to:', ml_client.workspace_name)
"
```

## Cost Considerations

- **Resource Group**: Free (just a container)
- **ML Workspace**: ~$50/month (includes storage and metadata)
- **Compute**: Only charged when running (see VM pricing in README)
- **Storage**: Minimal for models/logs (~$0.02/GB/month)

## Free Trial

If you're new to Azure:
1. Sign up at [azure.microsoft.com/free](https://azure.microsoft.com/free)
2. Get $200 credit for 30 days
3. Many services free for 12 months

## Next Steps

Once you have your subscription ID, resource group, and workspace name:

1. Set the environment variables
2. Run the nanoVLM Azure setup: `python azure/azure_ml_setup.py`
3. Submit your first training job!

## Troubleshooting

**"Subscription not found"**: Make sure you're logged in to the correct Azure account
```bash
az account clear
az login
```

**"Insufficient permissions"**: You need at least "Contributor" role on the subscription or resource group

**"Quota exceeded"**: Check your GPU quotas
```bash
az ml compute list-usage --workspace-name $AZURE_ML_WORKSPACE --resource-group $AZURE_RESOURCE_GROUP
```