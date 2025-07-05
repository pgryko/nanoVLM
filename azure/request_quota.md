# Requesting GPU Quota for Azure Student Subscription

## Quick Steps:

1. **Go to Azure Portal**: [portal.azure.com](https://portal.azure.com)
2. **Search for "Quotas"** in the top search bar
3. **Click "Compute"** → **"Machine Learning"**
4. **Find "Standard NCv3 Family vCPUs"** 
5. **Click "Request quota increase"**
6. **Request**: 6-12 vCPUs (enough for 1-2 V100 GPUs)
7. **Business justification**: "Educational machine learning research project"

## Alternative: Try Different GPU Families

If NCv3 is blocked, try requesting quota for:
- **Standard NC Family** (older K80 GPUs)
- **Standard NCasT4_v3 Family** (T4 GPUs - cheaper)
- **Standard NV Family** (for visualization workloads)

## Approval Time
- Usually 1-3 business days for student subscriptions
- Small requests (6-12 vCPUs) are often approved quickly