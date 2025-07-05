"""
Train nanoVLM on Modal.com - Working version
"""

import modal
import os
from pathlib import Path

app = modal.App("nanovlm-training")

# Define the container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "transformers==4.40.0",
        "datasets==2.19.0",
        "huggingface-hub==0.22.2",
        "numpy==1.24.3",
        "pillow==10.3.0",
        "wandb==0.17.0",
        "accelerate==0.30.0",
        "sentencepiece==0.2.0",
        "protobuf==4.25.3",
        "scipy==1.13.0",
        "tqdm",
        "safetensors",
    )
    .run_commands("apt-get update && apt-get install -y git")
)

# Path to nanoVLM directory
nanovlm_path = Path(__file__).parent.parent


@app.function(
    image=image,
    gpu="T4",  # Default GPU
    timeout=86400,  # 24 hours
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    mounts=[modal.Mount.from_local_dir(nanovlm_path, remote_path="/workspace/nanovlm")],
)
def train(
    dataset: str = "COCO",
    batch_size: int = 16,
    max_training_steps: int = 5000,
    wandb_entity: str = None,
    hub_model_id: str = None,
):
    import subprocess
    import sys

    os.chdir("/workspace/nanovlm")

    # Check environment
    print("=" * 80)
    print("Environment Check:")
    print(f"Python: {sys.version}")

    # Check GPU
    result = subprocess.run(
        [
            "python",
            "-c",
            "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print("=" * 80)

    # Build command
    cmd = [
        "python",
        "train.py",
        "--dataset",
        dataset,
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        "4",
        "--max_training_steps",
        str(max_training_steps),
        "--eval_interval",
        "400",
        "--output_dir",
        "/tmp/nanovlm_output",
    ]

    # Add optional arguments
    if wandb_entity:
        cmd.extend(["--log_wandb", "--wandb_entity", wandb_entity])

    if hub_model_id:
        cmd.extend(["--push_to_hub", "--hub_model_id", hub_model_id])

    print("\nStarting training...")
    print("Command:", " ".join(cmd))
    print("=" * 80 + "\n")

    # Run training
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise Exception(f"Training failed with return code {result.returncode}")

    print("\n✅ Training completed successfully!")


@app.local_entrypoint()
def main(
    dataset: str = "COCO",
    batch_size: int = 16,
    max_training_steps: int = 100,
    gpu: str = "t4",
    wandb_entity: str = None,
    hub_model_id: str = None,
):
    """Run nanoVLM training on Modal"""

    print("🚀 Starting nanoVLM training on Modal.com")
    print(f"📊 Dataset: {dataset}")
    print(f"🖥️  GPU: {gpu.upper()}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔄 Steps: {max_training_steps}")

    # Configure GPU
    if gpu == "a10g":
        train_func = train.with_options(gpu="A10G")
        cost_per_hour = 1.10
    elif gpu == "a100":
        train_func = train.with_options(gpu=modal.gpu.A100(size=40))
        cost_per_hour = 3.70
    else:  # t4
        train_func = train
        cost_per_hour = 0.59

    print(f"💰 Cost: ~${cost_per_hour}/hour")

    # Run training
    with app.run():
        train_func.remote(
            dataset=dataset,
            batch_size=batch_size,
            max_training_steps=max_training_steps,
            wandb_entity=wandb_entity,
            hub_model_id=hub_model_id,
        )


if __name__ == "__main__":
    import sys

    # Simple argument parsing
    args = {
        "dataset": "COCO",
        "batch_size": 16,
        "max_training_steps": 100,
        "gpu": "t4",
        "wandb_entity": None,
        "hub_model_id": None,
    }

    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--max-training-steps" and i + 1 < len(sys.argv):
            args["max_training_steps"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--wandb-entity" and i + 1 < len(sys.argv):
            args["wandb_entity"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--hub-model-id" and i + 1 < len(sys.argv):
            args["hub_model_id"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--gpu-type" and i + 1 < len(sys.argv):
            args["gpu"] = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    main(**args)
