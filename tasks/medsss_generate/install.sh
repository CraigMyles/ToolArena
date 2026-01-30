#! /bin/bash
set -e

git clone https://github.com/pixas/MedSSS /workspace/MedSSS
cd /workspace/MedSSS && git checkout ebbfd02

pip install /workspace/MedSSS
# Download the merged MedSSS_Policy model (includes base model, no separate Llama download needed)
python -c "from huggingface_hub import snapshot_download; snapshot_download('pixas/MedSSS_Policy')"