# Requesting V100 GPU Quota for Azure Student

## Step-by-Step Instructions:

### 1. Open Azure Portal
```
https://portal.azure.com
```

### 2. Navigate to Quotas
- Search for **"Quotas"** in the top search bar
- Click **"My quotas"**
- Select **"Compute"**

### 3. Filter and Find
- **Region**: East US
- **Provider**: Microsoft.MachineLearningServices
- Search for: **"Standard NCv3 Family"**

### 4. Request Increase
Click the pencil icon ✏️ next to "Standard NCv3 Family" and request:
- **New limit**: 6 vCPUs (enough for 1x V100 GPU)

### 5. Provide Justification
Copy and paste this:
```
I am a student working on an educational machine learning project focused on vision-language models. 
I need access to a single V100 GPU (6 vCPUs) to train a small educational model for my coursework. 
This is for learning purposes and academic research only. The training will run for approximately 
24-48 hours total over the next month.
```

### 6. Additional Details
- **Severity**: Low (non-production)
- **Contact method**: Email
- **Preferred contact language**: English

### 7. Submit and Wait
- Student requests are usually processed within 1-3 business days
- You'll receive an email when approved

## Alternative: Request via Azure CLI

```bash
# Check current quota
az quota show --resource-name standardNCv3Family --scope "/subscriptions/fb992ba5-7179-418e-8b18-65a7e81d5cc1/providers/Microsoft.MachineLearning/locations/eastus"

# Request increase (if CLI method is available)
az quota update --resource-name standardNCv3Family --scope "/subscriptions/fb992ba5-7179-418e-8b18-65a7e81d5cc1/providers/Microsoft.MachineLearning/locations/eastus" --limit-object value=6
```

## What You'll Get:
- **VM**: Standard_NC6s_v3
- **GPU**: 1x Tesla V100 (16GB VRAM)
- **Performance**: ~10x faster than K80
- **Cost**: ~$0.90/hour
- **Good for**: Training nanoVLM in 12-24 hours

## Tips for Approval:
1. ✅ Request only 6 vCPUs (minimum for 1 GPU)
2. ✅ Mention "student" and "educational"
3. ✅ Specify short-term use (not continuous)
4. ✅ Mention it's for coursework/research
5. ❌ Don't request large amounts
6. ❌ Don't mention commercial use