FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install nanoVLM dependencies
RUN pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    transformers==4.40.0 \
    datasets==2.19.0 \
    huggingface-hub==0.22.2 \
    numpy==1.24.3 \
    pillow==10.3.0 \
    wandb==0.17.0 \
    accelerate==0.30.0 \
    sentencepiece==0.2.0 \
    protobuf==4.25.3 \
    scipy==1.13.0 \
    tqdm \
    safetensors

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Create working directory
WORKDIR /workspace

# Copy nanoVLM code (will be mounted during job submission instead)
# COPY . /workspace/

# Set entrypoint
ENTRYPOINT ["python"]