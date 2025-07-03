# Public Dataset Workaround

The LLaVA dataset has some loading issues. Here are working alternatives:

## 🚀 Quick Solution: Use COCO Captions (Recommended)

```bash
# Activate virtual environment first
source .venv/bin/activate  # or however you activate your venv

# Use the reliable COCO dataset instead
python examples/simple_public_datasets.py \
  --dataset coco \
  --output datasets/coco_train.json \
  --limit 5000
```

## 🔧 Alternative Datasets That Work

### 1. COCO Captions (Most Reliable)
```bash
python examples/simple_public_datasets.py \
  --dataset coco \
  --output datasets/coco.json \
  --limit 10000
```

### 2. VQAv2 (Question Answering)
```bash
python examples/simple_public_datasets.py \
  --dataset vqav2 \
  --output datasets/vqav2.json \
  --limit 8000
```

### 3. Mixed Dataset (Best for Training)
```bash
python examples/simple_public_datasets.py \
  --dataset mixed \
  --output datasets/mixed_train.json \
  --limit 6000
```

## 🎯 Complete Training Example

```bash
# 1. Make sure you're in the right environment
source .venv/bin/activate

# 2. Create a good training dataset
python examples/simple_public_datasets.py \
  --dataset mixed \
  --output datasets/training_data.json \
  --limit 8000

# 3. Train the model (M1 Mac)
python train_custom.py \
  --custom_dataset_path datasets/training_data.json \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --log_wandb \
  --wandb_entity your_username

# 4. Train the model (NVIDIA GPU)
python train_custom.py \
  --custom_dataset_path datasets/training_data.json \
  --batch_size 16 \
  --gradient_accumulation_steps 2 \
  --max_training_steps 2000 \
  --eval_interval 100 \
  --compile \
  --log_wandb \
  --wandb_entity your_username
```

## 📊 Dataset Comparison

| Dataset | Samples | Type | Reliability | Best For |
|---------|---------|------|-------------|----------|
| **COCO** | 118K | Captions | ⭐⭐⭐⭐⭐ | Learning to describe images |
| **VQAv2** | 200K+ | Q&A | ⭐⭐⭐⭐⭐ | Question answering |
| **Mixed** | Combined | Both | ⭐⭐⭐⭐⭐ | General purpose (recommended) |
| LLaVA | 150K | Instructions | ⭐⭐ | Advanced (has loading issues) |

## 🛠️ If LLaVA is Required

If you specifically need LLaVA, try this manual approach:

```bash
# Download the JSON directly
curl -o llava_150k.json "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"

# Convert manually
python -c "
import json
with open('llava_150k.json') as f:
    data = json.load(f)

converted = []
for i, item in enumerate(data[:5000]):  # Limit to 5K
    try:
        conversations = []
        for conv in item['conversations']:
            role = 'user' if conv['from'] == 'human' else 'assistant'
            content = conv['value'].replace('<image>', '').strip()
            if content:
                conversations.append({'role': role, 'content': content})
        
        if conversations:
            converted.append({
                'image_path': item['image'],
                'conversations': conversations
            })
    except:
        continue

with open('datasets/llava_manual.json', 'w') as f:
    json.dump(converted, f, indent=2)

print(f'Converted {len(converted)} LLaVA samples')
"
```

## 🎉 Recommended Workflow

1. **Start with COCO** for reliability
2. **Add VQAv2** for question answering
3. **Use mixed dataset** for best results
4. **Train incrementally** (start small, scale up)

```bash
# Complete workflow
source .venv/bin/activate

# Create mixed dataset
python examples/simple_public_datasets.py \
  --dataset mixed \
  --output datasets/final_training.json \
  --limit 10000

# Train
python train_custom.py \
  --custom_dataset_path datasets/final_training.json \
  --batch_size 8 \
  --max_training_steps 3000 \
  --log_wandb

# Upload to HF
python upload_to_hf.py \
  --checkpoint_path ./checkpoints/nanoVLM_custom_* \
  --repo_name your-username/nanovlm-public-trained
```

The key is to use the `simple_public_datasets.py` script which handles image saving automatically and uses more reliable dataset loading methods.