#!/usr/bin/env python3
"""
Debug W&B Integration with Modal.com

This script helps debug and set up Weights & Biases logging with Modal.com
for NanoVLM training.
"""

import os
import subprocess
import sys


def check_local_wandb():
    """Check local W&B setup"""
    print("🔍 Checking local W&B setup...")

    try:
        import wandb

        print("✅ W&B package installed")

        # Check if logged in
        try:
            # Simple check - try to initialize a test run
            test_run = wandb.init(project="test", mode="disabled")
            wandb.finish()

            # Check API key
            if wandb.api.api_key:
                print("✅ W&B logged in successfully")
                print(f"   API key found: {wandb.api.api_key[:8]}...")
                return "piotr-gryko", "piotr-gryko"
            else:
                print("❌ No W&B API key found")
                return None, None

        except Exception as e:
            print(f"❌ W&B login issue: {e}")
            print("   Run: wandb login")
            return None, None

    except ImportError:
        print("❌ W&B not installed")
        print("   Install with: uv add wandb")
        return None, None


def get_wandb_api_key():
    """Get W&B API key"""
    print("\n🔑 Getting W&B API key...")

    # Check environment variable
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        print("✅ Found WANDB_API_KEY in environment")
        return api_key

    # Check wandb settings
    try:
        import wandb

        api_key = wandb.api.api_key
        if api_key:
            print("✅ Found API key in W&B settings")
            return api_key
    except:
        pass

    print("❌ No API key found")
    print("   Get your API key from: https://wandb.ai/authorize")
    return None


def test_wandb_locally():
    """Test W&B logging locally"""
    print("\n🧪 Testing W&B logging locally...")

    try:
        import wandb

        # Initialize a test run
        run = wandb.init(
            project="nanovlm-modal-test",
            entity="piotr-gryko",
            name="local-debug-test",
            config={"test": True},
        )

        # Log some test metrics
        for i in range(5):
            wandb.log({"test_metric": i * 0.1, "step": i})

        # Finish the run
        wandb.finish()

        print("✅ Local W&B logging test successful")
        print("   Check: https://wandb.ai/piotr-gryko/nanovlm-modal-test")
        return True

    except Exception as e:
        print(f"❌ Local W&B test failed: {e}")
        return False


def check_modal_secret():
    """Check if W&B secret exists in Modal"""
    print("\n🔐 Checking Modal W&B secret...")

    try:
        # Try to list Modal secrets
        result = subprocess.run(
            ["uv", "run", "modal", "secret", "list"], capture_output=True, text=True
        )

        if result.returncode == 0:
            if "wandb-secret" in result.stdout:
                print("✅ wandb-secret found in Modal")
                return True
            else:
                print("❌ wandb-secret not found in Modal")
                print("   Available secrets:")
                for line in result.stdout.split("\n"):
                    if line.strip():
                        print(f"     {line}")
                return False
        else:
            print(f"❌ Failed to list Modal secrets: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error checking Modal secrets: {e}")
        return False


def create_modal_secret(api_key):
    """Create W&B secret in Modal"""
    print("\n🔧 Creating W&B secret in Modal...")

    if not api_key:
        print("❌ No API key provided")
        return False

    try:
        # Create the secret
        result = subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "secret",
                "create",
                "wandb-secret",
                "WANDB_API_KEY=" + api_key,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ W&B secret created in Modal")
            return True
        else:
            print(f"❌ Failed to create secret: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error creating Modal secret: {e}")
        return False


def test_modal_wandb():
    """Test W&B integration in Modal"""
    print("\n🚀 Testing W&B integration in Modal...")

    # Create a simple Modal test script
    test_script = """
import modal
import os

app = modal.App("wandb-test")

@app.function(
    image=modal.Image.debian_slim().pip_install("wandb"),
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def test_wandb():
    import wandb
    
    print("Testing W&B in Modal...")
    print(f"WANDB_API_KEY present: {'WANDB_API_KEY' in os.environ}")
    
    try:
        run = wandb.init(
            project="nanovlm-modal-test",
            entity="piotr-gryko",
            name="modal-debug-test",
            config={"platform": "modal", "test": True}
        )
        
        # Log test metrics
        for i in range(3):
            wandb.log({"modal_test_metric": i * 0.2, "step": i})
        
        wandb.finish()
        print("✅ W&B logging successful in Modal")
        return True
        
    except Exception as e:
        print(f"❌ W&B logging failed in Modal: {e}")
        return False

if __name__ == "__main__":
    with app.run():
        result = test_wandb.remote()
        print(f"Modal W&B test result: {result}")
"""

    # Write test script
    with open("modal_wandb_test.py", "w") as f:
        f.write(test_script)

    try:
        # Run the test
        result = subprocess.run(
            ["uv", "run", "python", "modal_wandb_test.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if (
            result.returncode == 0
            and "✅ W&B logging successful in Modal" in result.stdout
        ):
            print("✅ Modal W&B integration test passed")
            return True
        else:
            print("❌ Modal W&B integration test failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Modal W&B test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running Modal W&B test: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists("modal_wandb_test.py"):
            os.remove("modal_wandb_test.py")


def main():
    """Main debugging workflow"""
    print("🐛 W&B + Modal.com Debug Tool")
    print("=" * 30)

    # Step 1: Check local W&B
    username, entity = check_local_wandb()
    if not username:
        print("\n❌ Local W&B setup incomplete")
        print("   1. Install W&B: uv add wandb")
        print("   2. Login: wandb login")
        print("   3. Re-run this script")
        return False

    # Step 2: Get API key
    api_key = get_wandb_api_key()
    if not api_key:
        print("\n❌ W&B API key not found")
        print("   1. Go to: https://wandb.ai/authorize")
        print("   2. Copy your API key")
        print("   3. Run: export WANDB_API_KEY=your_key_here")
        print("   4. Re-run this script")
        return False

    # Step 3: Test local W&B
    if not test_wandb_locally():
        print("\n❌ Local W&B test failed")
        return False

    # Step 4: Check Modal secret
    if not check_modal_secret():
        print("\n🔧 Creating Modal W&B secret...")
        if not create_modal_secret(api_key):
            print("\n❌ Failed to create Modal secret")
            print("   Manual setup:")
            print("   1. Go to: https://modal.com/secrets")
            print("   2. Create new secret: wandb-secret")
            print("   3. Add key: WANDB_API_KEY")
            print(f"   4. Add value: {api_key[:8]}...")
            return False

    # Step 5: Test Modal W&B integration
    if not test_modal_wandb():
        print("\n❌ Modal W&B integration failed")
        return False

    # Success!
    print("\n🎉 W&B + Modal.com integration working!")
    print("\n🚀 Ready for NanoVLM training with W&B logging:")
    print(
        f"""
uv run python modal/submit_modal_training.py \\
  --custom_dataset_path datasets/synthetic_test_dataset.json \\
  --batch_size 2 \\
  --max_training_steps 50 \\
  --eval_interval 25 \\
  --wandb_entity {entity or 'piotr-gryko'}
"""
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
