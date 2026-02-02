#! /bin/bash
set -e

git clone https://github.com/qinghezeng/ABRS-P /workspace/ABRS-P
cd /workspace/ABRS-P && git checkout 8ba3a7c

# Pin versions for reproducibility (per CONTRIBUTING.md guidance)
# Versions chosen for Python 3.12 compatibility
pip install \
    pandas==2.1.4 \
    "numpy<2" \
    torch==2.4.1 \
    torchmetrics==0.11.4 \
    scipy==1.11.4 \
    h5py==3.11.0 \
    wandb==0.16.0
