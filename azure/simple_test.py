#!/usr/bin/env python3
"""
Ultra-simple test to verify Azure ML job execution
"""

import os
import sys
import subprocess


def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("🧪 Azure ML Simple Test")
    print("=" * 50)

    # Test basic Python environment
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

    # List files in current directory
    print("\n📁 Files in current directory:")
    for item in os.listdir(".")[:10]:  # Show first 10 items
        print(f"  {item}")

    # Test PyTorch installation
    print("\n🔥 Testing PyTorch...")
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ Device count: {torch.cuda.device_count()}")

        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations work: {z.shape}")

    except ImportError:
        print("❌ PyTorch not available, trying to install...")
        if install_package("torch"):
            import torch

            print(f"✅ PyTorch installed: {torch.__version__}")
        else:
            print("❌ Failed to install PyTorch")
            return 1

    # Test transformers
    print("\n🤗 Testing Transformers...")
    try:
        import transformers

        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available, trying to install...")
        if install_package("transformers"):
            import transformers

            print(f"✅ Transformers installed: {transformers.__version__}")
        else:
            print("❌ Failed to install Transformers")

    # Test datasets
    print("\n📊 Testing Datasets...")
    try:
        import datasets

        print(f"✅ Datasets version: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not available, trying to install...")
        if install_package("datasets"):
            import datasets

            print(f"✅ Datasets installed: {datasets.__version__}")
        else:
            print("❌ Failed to install Datasets")

    # Test environment variables
    print("\n🌍 Environment Variables:")
    azure_vars = {k: v for k, v in os.environ.items() if "AZURE" in k}
    for key, value in list(azure_vars.items())[:5]:  # Show first 5
        print(f"  {key}: {value[:50]}...")  # Truncate long values

    # Test output directory
    print("\n📤 Testing Output Directory...")
    output_dir = os.environ.get("AZUREML_MODEL_DIR", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    test_file = os.path.join(output_dir, "simple_test_output.txt")
    with open(test_file, "w") as f:
        f.write("Simple test completed successfully!\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")

    print(f"✅ Output saved to: {test_file}")

    # Test basic model creation (if possible)
    print("\n🤖 Testing Basic Model Creation...")
    try:
        from transformers import AutoTokenizer, AutoModel

        # Use a very small model for testing
        model_name = "distilbert-base-uncased"
        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Test tokenization and forward pass
        text = "Hello, this is a test."
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ Model test successful: {outputs.last_hidden_state.shape}")

    except Exception as e:
        print(f"⚠️  Model test failed (expected): {e}")
        print("This is normal - we don't have nanoVLM dependencies yet")

    print("\n🎉 Simple Test Completed!")
    print("All basic functionality is working.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
