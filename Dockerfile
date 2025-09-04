# GPU runtime for the HPC
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Base OS deps
RUN apt-get update && apt-get install -y python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN pip3 install --upgrade pip uv pixi

# IMPORTANT: in Linux container, point pip/uv at the CUDA 12.4 wheels for PyTorch
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cu124

WORKDIR /work

# Install deps first for caching
COPY pixi.toml pixi.lock ./
# Solve for linux-64 only (don’t try to solve osx here)
RUN pixi install --platform linux-64

# Now add your code
COPY . /work

# Default health check; override in Slurm with your train command
CMD ["pixi","run","python","-c","import torch; print({'cuda': torch.cuda.is_available(), 'cuda_version': torch.version.cuda})"]
