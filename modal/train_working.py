"""
Train nanoVLM on Modal - Simplified working version
"""

import modal
import os

app = modal.App("nanovlm")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch",
            "torchvision",
            "transformers",
            "datasets",
            "huggingface-hub",
            "numpy",
            "pillow",
            "wandb",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "scipy",
            "tqdm",
            "safetensors",
        ]
    )
    .run_commands("apt-get update && apt-get install -y git")
)

# Create a volume for code
volume = modal.Volume.from_name("nanovlm-code", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=86400,
    volumes={"/workspace": volume},
)
def train_model(
    code_files: dict,
    dataset: str = "COCO",
    batch_size: int = 16,
    max_steps: int = 100,
    wandb_entity: str = None,
    hub_model_id: str = None,
):
    import subprocess

    # Write code files to workspace
    os.makedirs("/workspace/nanovlm", exist_ok=True)
    os.chdir("/workspace/nanovlm")

    for filepath, content in code_files.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)

    # Set environment variables if provided
    if wandb_entity and os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    if hub_model_id and os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    print("🚀 Starting training...")
    print(f"📊 Dataset: {dataset}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔄 Steps: {max_steps}")

    # Build command
    cmd = [
        "python",
        "train.py",
        "--dataset",
        dataset,
        "--batch_size",
        str(batch_size),
        "--max_training_steps",
        str(max_steps),
        "--output_dir",
        "/tmp/model",
    ]

    if wandb_entity:
        cmd.extend(["--log_wandb", "--wandb_entity", wandb_entity])

    if hub_model_id:
        cmd.extend(["--push_to_hub", "--hub_model_id", hub_model_id])

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Training completed!")
    else:
        print("\n❌ Training failed!")

    return result.returncode


@app.local_entrypoint()
def main():
    import sys
    from pathlib import Path

    # Parse arguments
    max_steps = 100
    wandb_entity = None
    hub_model_id = None
    gpu_type = "t4"

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--max-training-steps" and i + 1 < len(sys.argv):
            max_steps = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--wandb-entity" and i + 1 < len(sys.argv):
            wandb_entity = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--hub-model-id" and i + 1 < len(sys.argv):
            hub_model_id = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--gpu-type" and i + 1 < len(sys.argv):
            gpu_type = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    print("📁 Collecting nanoVLM code files...")

    # Collect all Python files
    nanovlm_dir = Path(__file__).parent.parent
    code_files = {}

    for pattern in ["*.py", "models/*.py", "data/*.py", "eval/*.py"]:
        for file in nanovlm_dir.glob(pattern):
            if "__pycache__" not in str(file) and ".venv" not in str(file):
                relative_path = file.relative_to(nanovlm_dir)
                code_files[str(relative_path)] = file.read_text()

    print(f"📦 Found {len(code_files)} files to upload")

    # Configure GPU
    if gpu_type == "a10g":
        train_func = train_model.with_options(gpu="A10G")
        print("🖥️  Using A10G GPU (~$1.10/hour)")
    elif gpu_type == "a100":
        train_func = train_model.with_options(gpu=modal.gpu.A100(size=40))
        print("🖥️  Using A100 40GB GPU (~$3.70/hour)")
    else:
        train_func = train_model
        print("🖥️  Using T4 GPU (~$0.59/hour)")

    # Run on Modal
    with app.run():
        result = train_func.remote(
            code_files=code_files,
            max_steps=max_steps,
            wandb_entity=wandb_entity,
            hub_model_id=hub_model_id,
        )

    if result == 0:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n❌ Training failed!")


if __name__ == "__main__":
    main()
