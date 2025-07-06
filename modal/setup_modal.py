#!/usr/bin/env python3
"""
Modal.com Setup Helper

This script helps you set up Modal.com for NanoVLM training by:
1. Installing Modal if needed
2. Guiding through authentication
3. Setting up secrets for W&B and HuggingFace
4. Running validation tests
"""

import subprocess
import sys


def check_modal_installation():
    """Check if Modal is installed"""
    print("🔍 Checking Modal installation...")

    try:
        import modal

        print("✅ Modal Python package is installed")
    except ImportError:
        print("❌ Modal not installed")
        install = input("Install Modal now? (Y/n): ").strip().lower()
        if install != "n":
            print("Installing Modal...")
            subprocess.run(["uv", "add", "modal"], check=True)
            print("✅ Modal installed")
        else:
            print("❌ Modal installation cancelled")
            return False

    # Check CLI
    try:
        result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal CLI: {result.stdout.strip()}")
            return True
        else:
            print("❌ Modal CLI not working")
            return False
    except FileNotFoundError:
        print("❌ Modal CLI not found")
        return False


def setup_modal_auth():
    """Guide user through Modal authentication"""
    print("\n🔐 Setting up Modal authentication...")

    # Check if already authenticated by trying to list apps
    try:
        result = subprocess.run(
            ["uv", "run", "modal", "app", "list"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ Already authenticated with Modal")
            return True
    except:
        pass

    print("🚀 Starting Modal authentication...")
    print("This will open your browser to authenticate with Modal.com")

    proceed = input("Continue? (Y/n): ").strip().lower()
    if proceed == "n":
        print("❌ Authentication cancelled")
        return False

    try:
        # Run modal setup
        result = subprocess.run(["modal", "setup"], check=False)

        # Give a moment for the token to be written
        import time

        time.sleep(2)

        # Verify authentication by trying to list apps
        verify_result = subprocess.run(
            ["uv", "run", "modal", "app", "list"], capture_output=True, text=True
        )

        if verify_result.returncode == 0:
            print("✅ Modal authentication successful!")
            return True
        else:
            print("⚠️  Authentication verification failed, but setup may have completed")
            print("   Try running: uv run modal app list")
            # Don't fail completely - let user proceed
            return True

    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def setup_wandb_secret():
    """Guide user through W&B secret setup"""
    print("\n📊 Setting up Weights & Biases secret...")

    setup_wandb = input("Set up W&B logging? (Y/n): ").strip().lower()
    if setup_wandb == "n":
        print("⚠️  Skipping W&B setup")
        return True

    print("\n📋 W&B Setup Instructions:")
    print("1. Go to https://wandb.ai/authorize")
    print("2. Copy your API key")
    print("3. Go to https://modal.com/secrets")
    print("4. Create a new secret:")
    print("   - Name: wandb-secret")
    print("   - Key: WANDB_API_KEY")
    print("   - Value: [paste your API key]")

    print("\n🔗 Opening W&B authorize page...")
    try:
        import webbrowser

        webbrowser.open("https://wandb.ai/authorize")
    except:
        print("   Manual URL: https://wandb.ai/authorize")

    input("\nPress Enter after setting up the W&B secret in Modal dashboard...")
    print("✅ W&B secret setup completed")
    return True


def setup_hf_secret():
    """Guide user through HuggingFace secret setup"""
    print("\n🤗 Setting up HuggingFace secret...")

    setup_hf = input("Set up HuggingFace model uploads? (Y/n): ").strip().lower()
    if setup_hf == "n":
        print("⚠️  Skipping HuggingFace setup")
        return True

    print("\n📋 HuggingFace Setup Instructions:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Write' permissions")
    print("3. Copy the token")
    print("4. Go to https://modal.com/secrets")
    print("5. Create a new secret:")
    print("   - Name: huggingface-secret")
    print("   - Key: HF_TOKEN")
    print("   - Value: [paste your token]")

    print("\n🔗 Opening HuggingFace tokens page...")
    try:
        import webbrowser

        webbrowser.open("https://huggingface.co/settings/tokens")
    except:
        print("   Manual URL: https://huggingface.co/settings/tokens")

    input("\nPress Enter after setting up the HuggingFace secret in Modal dashboard...")
    print("✅ HuggingFace secret setup completed")
    return True


def run_validation():
    """Run the Modal setup validation"""
    print("\n🧪 Running validation tests...")

    try:
        # Test Modal CLI availability
        result = subprocess.run(
            ["modal", "--version"], capture_output=True, check=False
        )
        if result.returncode != 0:
            print("❌ Modal CLI not available")
            return False
        print("✅ Modal CLI available")

        # Test Modal authentication
        result = subprocess.run(
            ["modal", "app", "list"], capture_output=True, check=False
        )
        if result.returncode != 0:
            print("❌ Modal authentication failed")
            return False
        print("✅ Modal authentication working")

        # Test Python imports
        try:
            import modal

            print("✅ Modal Python package available")
        except ImportError:
            print("❌ Modal Python package not available")
            return False

        print("✅ All validation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    """Main setup flow"""
    print("🚀 Modal.com Setup for NanoVLM")
    print("=" * 32)
    print("This script will help you set up Modal.com for training NanoVLM")
    print()

    # Step 1: Install Modal
    if not check_modal_installation():
        print("❌ Modal installation failed")
        return False

    # Step 2: Authenticate
    if not setup_modal_auth():
        print("❌ Modal authentication failed")
        return False

    # Step 3: Set up secrets
    setup_wandb_secret()
    setup_hf_secret()

    # Step 4: Run validation
    print("\n" + "=" * 50)
    print("🧪 VALIDATION")
    print("=" * 50)

    if run_validation():
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 You're ready to train on Modal.com!")

        print("\n📋 Quick Start Commands:")
        print("   # Test with example dataset")
        print("   ./modal/quick_start.sh")
        print()
        print("   # Train with your dataset")
        print("   python modal/submit_modal_training.py \\")
        print("     --custom_dataset_path your_dataset.json \\")
        print("     --wandb_entity your_username")

        print("\n📚 Documentation:")
        print("   - Modal guide: modal/README.md")
        print("   - Dataset format: CUSTOM_TRAINING.md")
        print("   - Examples: examples/train_modal_example.py")

        return True
    else:
        print("\n❌ Setup validation failed")
        print("Please check the errors above and try again")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
