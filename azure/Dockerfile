FROM mcr.microsoft.com/azureml/pytorch-2.0-ubuntu20.04-py38-cuda11.7-gpu:latest

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# Install nanoVLM dependencies
RUN pip install \
    transformers>=4.53.0 \
    datasets>=3.6.0 \
    huggingface-hub>=0.33.2 \
    numpy>=2.3.1 \
    pillow>=11.3.0 \
    wandb>=0.21.0 \
    accelerate>=1.3.0 \
    sentencepiece>=0.2.0 \
    protobuf>=5.30.0 \
    scipy>=1.15.1 \
    tqdm \
    azure-ai-ml \
    azure-identity

# Install additional tools for evaluation
RUN pip install lmms-eval

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Create working directory
WORKDIR /workspace

# Copy nanoVLM code (will be mounted during job submission instead)
# COPY . /workspace/

# Set entrypoint
ENTRYPOINT ["python"]