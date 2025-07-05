"""
Train nanoVLM on Modal.com with GPU support
No quota restrictions - pay per minute of GPU usage
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("nanovlm-training")

# Define the container image with all dependencies
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

# Mount the code directory
nanovlm_path = Path(__file__).parent.parent  # Path to nanoVLM directory

# Define secrets (set these in Modal dashboard)
secrets = [
    modal.Secret.from_name("huggingface"),  # Should contain HF_TOKEN
    modal.Secret.from_name("wandb"),  # Should contain WANDB_API_KEY
]


@app.function(
    image=image,
    gpu=modal.gpu.T4(count=1),  # T4 GPU - cheapest option ($0.59/hour)
    # gpu=modal.gpu.A10G(count=1),  # A10G GPU - faster ($1.10/hour)
    # gpu=modal.gpu.A100(count=1, size=40),  # A100 40GB - fastest ($3.70/hour)
    timeout=86400,  # 24 hours max
    secrets=secrets,
    mounts=[modal.Mount.from_local_dir(nanovlm_path, remote_path="/workspace/nanovlm")],
)
def train_nanovlm(
    dataset: str = "COCO",
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    max_training_steps: int = 5000,
    eval_interval: int = 400,
    lr_backbone: float = 5e-5,
    lr_projector: float = 0.00512,
    warmup_ratio: float = 0.05,
    compile: bool = False,
    log_wandb: bool = True,
    wandb_entity: str = None,
    wandb_project: str = "nanovlm-modal",
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """Train nanoVLM on Modal's cloud GPUs"""

    import sys
    import subprocess

    # Change to workspace directory
    os.chdir("/workspace/nanovlm")

    # Verify environment
    print("Python version:", sys.version)
    print(
        "PyTorch version:",
        subprocess.check_output(
            ["python", "-c", "import torch; print(torch.__version__)"], text=True
        ).strip(),
    )
    print(
        "CUDA available:",
        subprocess.check_output(
            ["python", "-c", "import torch; print(torch.cuda.is_available())"],
            text=True,
        ).strip(),
    )
    print(
        "GPU:",
        subprocess.check_output(
            [
                "python",
                "-c",
                "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')",
            ],
            text=True,
        ).strip(),
    )

    # Build training command
    cmd = [
        "python",
        "train.py",
        "--dataset",
        dataset,
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--max_training_steps",
        str(max_training_steps),
        "--eval_interval",
        str(eval_interval),
        "--lr_backbone",
        str(lr_backbone),
        "--lr_projector",
        str(lr_projector),
        "--warmup_ratio",
        str(warmup_ratio),
        "--output_dir",
        "/tmp/nanovlm_output",
    ]

    if compile:
        cmd.append("--compile")

    if log_wandb and wandb_entity:
        cmd.extend(
            [
                "--log_wandb",
                "--wandb_entity",
                wandb_entity,
                "--wandb_project",
                wandb_project,
            ]
        )

    if push_to_hub and hub_model_id:
        cmd.extend(["--push_to_hub", "--hub_model_id", hub_model_id])

    print("\nStarting training with command:")
    print(" ".join(cmd))
    print("\n" + "=" * 80 + "\n")

    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise Exception(f"Training failed with return code {result.returncode}")

    print("\n" + "=" * 80)
    print("Training completed successfully!")

    # Return output directory for downloading
    return "/tmp/nanovlm_output"


@app.local_entrypoint()
def main(
    dataset: str = "COCO",
    batch_size: int = 16,
    max_training_steps: int = 5000,
    gpu_type: str = "t4",  # t4, a10g, or a100
    wandb_entity: str = None,
    hub_model_id: str = None,
):
    """
    Train nanoVLM on Modal.com

    Examples:
        # Basic training on T4 GPU
        modal run modal/train_modal.py

        # Train with A10G GPU and push to HuggingFace
        modal run modal/train_modal.py --gpu-type a10g --hub-model-id "username/nanovlm-modal"

        # Full training with W&B logging
        modal run modal/train_modal.py --wandb-entity your-username --max-training-steps 10000
    """

    print("🚀 Starting nanoVLM training on Modal.com")
    print(f"📊 Dataset: {dataset}")
    print(f"🖥️  GPU Type: {gpu_type.upper()}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"🔄 Max Steps: {max_training_steps}")

    # GPU pricing info
    if gpu_type == "a10g":
        print("💰 Estimated cost: ~$1.10/hour")
    elif gpu_type == "a100":
        print("💰 Estimated cost: ~$3.70/hour")
    else:  # t4
        print("💰 Estimated cost: ~$0.59/hour")

    # Estimate training time
    if gpu_type == "t4":
        hours = max_training_steps / 200  # ~200 steps/hour on T4
    elif gpu_type == "a10g":
        hours = max_training_steps / 400  # ~400 steps/hour on A10G
    else:  # a100
        hours = max_training_steps / 800  # ~800 steps/hour on A100

    print(f"⏱️  Estimated time: {hours:.1f} hours")
    print(
        f"💵 Estimated total cost: ${hours * (0.59 if gpu_type == 't4' else 1.10 if gpu_type == 'a10g' else 3.70):.2f}"
    )

    print("\n" + "=" * 80 + "\n")

    # Run training
    with app.run():
        output_dir = train_nanovlm.remote(
            dataset=dataset,
            batch_size=batch_size,
            max_training_steps=max_training_steps,
            wandb_entity=wandb_entity,
            push_to_hub=bool(hub_model_id),
            hub_model_id=hub_model_id,
        )

    print("\n✅ Training completed!")
    print(f"📁 Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
