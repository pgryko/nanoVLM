"""
Simplified Modal training script - just run: modal run modal/train_simple.py
"""

import modal

app = modal.App("nanovlm-simple")

# Minimal image with just what we need
image = modal.Image.debian_slim(python_version="3.11").pip_install(
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
)


@app.function(
    image=image,
    gpu="T4",  # Cheapest GPU
    timeout=86400,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/workspace")],
)
def train():
    import os
    import subprocess

    os.chdir("/workspace")

    # Simple training command
    subprocess.run(
        [
            "python",
            "train.py",
            "--dataset",
            "COCO",
            "--batch_size",
            "16",
            "--max_training_steps",
            "1000",  # Short test run
            "--output_dir",
            "/tmp/model",
        ]
    )

    print("✅ Training complete!")


@app.local_entrypoint()
def main():
    print("🚀 Starting nanoVLM training on Modal...")
    print("💰 Using T4 GPU (~$0.59/hour)")
    print("⏱️  Training 1000 steps (~5 hours)")

    with app.run():
        train.remote()


if __name__ == "__main__":
    main()
