import torch
import platform
dev = "cuda" if torch.cuda.is_available() else \
      "mps" if getattr(torch.backends, "mps",
                       None) and torch.backends.mps.is_available() else "cpu"
print({"device": dev, "torch": torch.__version__,
      "cuda": torch.version.cuda, "python": platform.python_version()})
