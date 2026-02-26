# DegradoMap Reproducibility Container
# Build: docker build -t degradomap .
# Run: docker run --gpus all -it degradomap

FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8
RUN pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
RUN pip3 install torch-geometric==2.4.0

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY configs/ configs/
COPY docs/ docs/

# Create necessary directories
RUN mkdir -p checkpoints results

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python3", "scripts/train.py", "--split", "target_unseen"]
