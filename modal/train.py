"""
Train nanoVLM on Modal.com
Usage: modal run modal/train.py --help
"""

import modal
import os
from pathlib import Path

app = modal.App("nanovlm-training")

# Create volume for persistent storage
volume = modal.Volume.from_name("nanovlm-storage", create_if_missing=True)

# Define image
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    [
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
    ]
)


@app.function(
    image=image,
    gpu="T4",
    timeout=86400,
    volumes={"/workspace": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train(
    max_steps: int = 100,
    wandb_entity: str = "",
    hub_model_id: str = "",
    gpu_type: str = "t4",
    code_snapshot: dict = None,
):
    """Train nanoVLM on Modal GPUs"""
    import subprocess

    print("🚀 Starting Modal function...")
    print(f"📍 Working directory: {os.getcwd()}")
    print(
        f"📁 Workspace contents: {os.listdir('/workspace') if os.path.exists('/workspace') else 'Empty'}"
    )

    # Write code files from snapshot
    if code_snapshot:
        print(f"📝 Writing {len(code_snapshot)} code files...")
        os.makedirs("/workspace/nanovlm", exist_ok=True)

        for filepath, content in code_snapshot.items():
            full_path = f"/workspace/nanovlm/{filepath}"
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w") as f:
                f.write(content)

            if filepath.endswith("train.py"):
                print(f"  ✓ Written {filepath} ({len(content)} bytes)")

        print("✅ All code files written successfully")

    # Change to nanovlm directory
    os.chdir("/workspace/nanovlm")
    print(f"📍 Changed to: {os.getcwd()}")
    print(f"📂 Files in directory: {len(os.listdir('.'))}")

    # Check environment
    print("\n🖥️  Environment check:")
    result = subprocess.run(
        [
            "python",
            "-c",
            "import torch; print(f'  • PyTorch: {torch.__version__}'); print(f'  • CUDA available: {torch.cuda.is_available()}'); print(f'  • GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Check if secrets are available
    print("\n🔐 Checking secrets:")
    print(f"  • HF_TOKEN: {'✓ Set' if os.getenv('HF_TOKEN') else '✗ Not set'}")
    print(
        f"  • WANDB_API_KEY: {'✓ Set' if os.getenv('WANDB_API_KEY') else '✗ Not set'}"
    )

    # Just run with minimal arguments for now
    cmd = ["python", "train.py"]

    # Only add the arguments that work
    if not wandb_entity:
        cmd.append("--no_log_wandb")

    print(f"🔧 W&B Entity: {wandb_entity or 'Not set'}")
    print(f"🔧 Hub Model: {hub_model_id or 'Not set'}")

    # Set environment variables for configuration
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    if hub_model_id:
        os.environ["HF_REPO_ID"] = hub_model_id

    print(f"\n🚀 Starting training for {max_steps} steps...")
    print("Command:", " ".join(cmd))
    print("-" * 80)

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Training completed successfully!")
        if hub_model_id:
            print(f"🤗 Model pushed to: https://huggingface.co/{hub_model_id}")
        if wandb_entity:
            print(f"📊 Metrics at: https://wandb.ai/{wandb_entity}/nanovlm-modal")
    else:
        print("\n❌ Training failed!")
        raise Exception("Training failed")


@app.local_entrypoint()
def main(
    max_steps: int = 100,
    wandb_entity: str = "",
    hub_model_id: str = "",
    gpu_type: str = "t4",
):
    """
    Train nanoVLM on Modal.com

    Examples:
        modal run modal/train.py --max-steps 100
        modal run modal/train.py --max-steps 5000 --wandb-entity myusername --gpu-type a10g
    """
    print("🚀 nanoVLM Training on Modal.com")
    print(f"📦 Steps: {max_steps}")
    print(f"🖥️  GPU: {gpu_type.upper()}")

    # Configure GPU and estimate cost
    if gpu_type == "a10g":
        fn = train.with_options(gpu="A10G")
        cost_per_hour = 1.10
        steps_per_hour = 400
    elif gpu_type == "a100":
        fn = train.with_options(gpu=modal.gpu.A100(size=40))
        cost_per_hour = 3.70
        steps_per_hour = 800
    else:  # t4
        fn = train
        cost_per_hour = 0.59
        steps_per_hour = 200

    hours = max_steps / steps_per_hour
    total_cost = hours * cost_per_hour

    print(f"⏱️  Estimated time: {hours:.1f} hours")
    print(f"💰 Estimated cost: ${total_cost:.2f}")

    if wandb_entity:
        print(f"📊 W&B: {wandb_entity}")
    if hub_model_id:
        print(f"🤗 Hub: {hub_model_id}")

    print("\n📦 Collecting nanoVLM code files...")

    # Get nanoVLM directory and collect files
    nanovlm_dir = Path(__file__).parent.parent
    code_snapshot = {}

    # Essential files to copy
    essential_files = [
        "train.py",
        "models/__init__.py",
        "models/config.py",
        "models/vision_transformer.py",
        "models/language_model.py",
        "models/modality_projector.py",
        "models/vision_language_model.py",
        "data/__init__.py",
        "data/dataset.py",
        "data/collators.py",
    ]

    # Also collect any additional Python files we might have missed
    for pattern in ["*.py", "data/*.py", "models/*.py", "eval/*.py"]:
        for file_path in nanovlm_dir.glob(pattern):
            if "__pycache__" not in str(file_path) and ".venv" not in str(file_path):
                relative_path = str(file_path.relative_to(nanovlm_dir))
                if relative_path not in essential_files:
                    essential_files.append(relative_path)

    print(f"📁 Looking in: {nanovlm_dir}")

    for file_path in essential_files:
        full_path = nanovlm_dir / file_path
        if full_path.exists():
            code_snapshot[file_path] = full_path.read_text()
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ Missing: {file_path}")

    print(f"\n📦 Collected {len(code_snapshot)} files")
    print("Starting Modal run...\n")

    # Run the function
    fn.remote(
        max_steps=max_steps,
        wandb_entity=wandb_entity,
        hub_model_id=hub_model_id,
        gpu_type=gpu_type,
        code_snapshot=code_snapshot,
    )


if __name__ == "__main__":
    main()
