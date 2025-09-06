# Start from NVIDIA NGC PyTorch runtime (includes CUDA/cuDNN + PyTorch)
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Minimal extra OS deps (if needed)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy code only; rely on base for Torch/CUDA stack
COPY . /work

# Quick health check: print CUDA info
CMD ["python","-c","import torch; print({'cuda': torch.cuda.is_available(), 'cuda_version': torch.version.cuda})"]
