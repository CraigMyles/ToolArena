#! /bin/bash
set -e

# Install build dependencies including cmake, ninja, and openssl
apt-get update && apt-get install -y cmake ninja-build libssl-dev

# Install runtime dependencies for implementation.py (runs under system Python)
pip install pyyaml h5py

git clone https://github.com/KatherLab/COBRA.git /workspace/COBRA
cd /workspace/COBRA
nvcc --version

# Create virtual environment with seed (includes pip)
pip install uv
uv venv --python=3.11 --seed

# Use uv pip install with explicit python path to install into venv
UV_PIP="uv pip install --python /workspace/COBRA/.venv/bin/python"

# Install base dependencies first
$UV_PIP hatchling editables pyyaml h5py setuptools packaging wheel
$UV_PIP torch==2.4.1 numpy==2.0.0

# Install build dependencies for CUDA extensions
$UV_PIP ninja scikit-build-core

# For packages that need to compile CUDA extensions, we need to use pip with --no-build-isolation
source .venv/bin/activate
pip install --no-binary causal-conv1d --no-build-isolation --no-deps causal-conv1d==1.5.0.post8
pip install --no-binary mamba-ssm --no-build-isolation --no-deps mamba-ssm==2.2.4

# Install COBRA without deps to prevent torch upgrade
pip install -e . --no-build-isolation --no-deps

# Install COBRA's remaining dependencies
pip install einops timm "torchvision==0.19.1" pytorch_lightning openslide-python openslide-bin pandas openpyxl matplotlib scikit-learn "transformers<4.41"

# CRITICAL: Reinstall torch 2.4.1 to ensure ABI matches CUDA extensions
pip install torch==2.4.1 --force-reinstall --no-deps

# Try an older version of triton that might work better
pip install triton==2.1.0

# Verify torch version is correct (non-fatal - just logging)
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || true
python -c "from mamba_ssm import Mamba2; print('mamba_ssm import successful')" || true

ulimit -n 8192