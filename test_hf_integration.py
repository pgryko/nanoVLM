#!/usr/bin/env python3
"""
Test HuggingFace Integration with Modal.com

This script tests the HuggingFace Hub integration for NanoVLM training on Modal.
"""

import subprocess
import sys
import time


def test_hf_authentication():
    """Test HuggingFace authentication"""
    print("🔍 Testing HuggingFace authentication...")

    try:
        from huggingface_hub import whoami

        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        return user_info["name"]
    except Exception as e:
        print(f"❌ HuggingFace authentication failed: {e}")
        print("   Run: huggingface-cli login")
        return None


def test_modal_hf_secret():
    """Test if HuggingFace secret exists in Modal"""
    print("\n🔐 Checking Modal HuggingFace secret...")

    try:
        result = subprocess.run(
            ["uv", "run", "modal", "secret", "list"], capture_output=True, text=True
        )

        if result.returncode == 0:
            if "huggingface-secret" in result.stdout:
                print("✅ huggingface-secret found in Modal")
                return True
            else:
                print("❌ huggingface-secret not found in Modal")
                return False
        else:
            print(f"❌ Failed to list Modal secrets: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error checking Modal secrets: {e}")
        return False


def run_training_with_hf_push(username):
    """Run a short training with HuggingFace push"""
    print("\n🚀 Testing training with HuggingFace push...")

    # Generate a unique model name
    timestamp = int(time.time())
    model_id = f"{username}/nanovlm-modal-test-{timestamp}"

    print(f"Model ID: {model_id}")
    print("This will:")
    print("  1. Train NanoVLM for 25 steps (~1-2 minutes)")
    print("  2. Push the trained model to HuggingFace Hub")
    print("  3. Create a public repository (you can delete it later)")

    proceed = input("\nProceed with test? (y/N): ").strip().lower()
    if proceed != "y":
        print("Test cancelled.")
        return False

    # Run training with HF push
    cmd = [
        "uv",
        "run",
        "python",
        "modal/submit_modal_training.py",
        "--custom_dataset_path",
        "datasets/synthetic_test_dataset.json",
        "--batch_size",
        "2",
        "--max_training_steps",
        "25",
        "--eval_interval",
        "25",
        "--push_to_hub",
        "--hub_model_id",
        model_id,
        "--wandb_entity",
        "piotr-gryko-devalogic",
    ]

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False, timeout=600)  # 10 minute timeout

        if result.returncode == 0:
            print("✅ Training and HuggingFace push completed successfully!")
            print(f"🤗 Model available at: https://huggingface.co/{model_id}")
            return True
        else:
            print("❌ Training or HuggingFace push failed")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False


def test_model_loading(model_id):
    """Test loading the uploaded model"""
    print("\n🧪 Testing model loading from HuggingFace...")

    try:
        # Test loading the model
        test_code = f"""
import sys
sys.path.append(".")
from models.vision_language_model import VisionLanguageModel

print("Loading model from HuggingFace Hub...")
model = VisionLanguageModel.from_pretrained("{model_id}")
print("✅ Model loaded successfully!")
print(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")
"""

        result = subprocess.run(
            ["uv", "run", "python", "-c", test_code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("✅ Model loading test passed")
            print(result.stdout)
            return True
        else:
            print("❌ Model loading test failed")
            print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        return False


def main():
    """Main test workflow"""
    print("🧪 HuggingFace + Modal.com Integration Test")
    print("=" * 42)

    # Test 1: HF Authentication
    username = test_hf_authentication()
    if not username:
        print("\n❌ HuggingFace authentication required")
        print("   Run: huggingface-cli login")
        return False

    # Test 2: Modal HF Secret
    if not test_modal_hf_secret():
        print("\n❌ Modal HuggingFace secret not configured")
        print("   The secret should already exist from previous setup")
        return False

    # Test 3: Training with HF Push
    print("\n🎯 Ready to test training with HuggingFace push")
    print(f"   Username: {username}")
    print("   This will create a temporary model repository")

    success = run_training_with_hf_push(username)
    if not success:
        return False

    # Test 4: Model Loading (optional)
    timestamp = int(time.time())
    model_id = f"{username}/nanovlm-modal-test-{timestamp}"

    test_loading = input("\nTest loading the uploaded model? (y/N): ").strip().lower()
    if test_loading == "y":
        test_model_loading(model_id)

    # Success!
    print("\n🎉 HuggingFace integration test completed!")
    print("\n📋 Summary:")
    print("   ✅ HuggingFace authentication working")
    print("   ✅ Modal secrets configured")
    print("   ✅ Training with HF push working")
    print(f"   ✅ Model uploaded to: https://huggingface.co/{model_id}")

    print("\n🚀 Ready for production training with HF push:")
    print(
        f"""
uv run python modal/submit_modal_training.py \\
  --custom_dataset_path your_dataset.json \\
  --batch_size 8 \\
  --max_training_steps 2000 \\
  --eval_interval 200 \\
  --wandb_entity piotr-gryko-devalogic \\
  --push_to_hub \\
  --hub_model_id {username}/my-custom-nanovlm
"""
    )

    print("\n🗑️  Clean up:")
    print(
        f"   You can delete the test model at: https://huggingface.co/{model_id}/settings"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
