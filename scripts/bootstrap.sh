#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Python deps inside the container.
# - Installs uv
# - Creates a virtualenv at ./.venv WITH system site packages (so Torch/CUDA from the base image are reused)
# - Syncs dependencies from uv.lock/pyproject.toml

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

echo "[bootstrap] Using Python: $(python3 -V)"

echo "[bootstrap] Upgrading pip and installing uv"
python3 -m pip install -U pip uv

# Create venv reusing system site-packages so we don't reinstall PyTorch/CUDA
if [ ! -d .venv ]; then
  echo "[bootstrap] Creating virtualenv at .venv (with system site packages)"
  uv venv --system-site-packages .venv
else
  echo "[bootstrap] Existing .venv detected; reusing"
fi

echo "[bootstrap] Syncing dependencies from uv.lock"
uv sync --frozen

echo "[bootstrap] Done. Activate with: source .venv/bin/activate"
echo "[bootstrap] Or run commands with uv, e.g.: uv run pytest -q"

