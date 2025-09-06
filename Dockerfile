# Start from official PyTorch runtime with CUDA/cuDNN preinstalled
FROM pytorch/pytorch:2.8.0-cuda12.4-cudnn9-runtime

# Minimal extra OS deps (if needed)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy code only; rely on base for Torch/CUDA stack
COPY . /work

# Quick health check: print CUDA info
CMD ["python","-c","import torch; print({'cuda': torch.cuda.is_available(), 'cuda_version': torch.version.cuda})"]
