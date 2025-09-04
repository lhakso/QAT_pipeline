# GPU runtime for HPC
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# OS deps
RUN apt-get update && apt-get install -y python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN python3 -m pip install -U pip uv

# Pull CUDA-enabled PyTorch wheels in this image
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cu124

WORKDIR /work

# Install deps first (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Add code last
COPY . /work

# Health check (override in Slurm when training)
CMD ["uv","run","python","-c","import torch; print({'cuda': torch.cuda.is_available(), 'cuda_version': torch.version.cuda})"]
