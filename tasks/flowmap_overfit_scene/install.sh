#! /bin/bash
set -e

git clone https://github.com/dcharatan/flowmap /workspace/FlowMap
cd /workspace/FlowMap && git checkout 578a515

pip install gdown && mkdir -p checkpoints && gdown --fuzzy 'https://drive.google.com/uc?id=1BI9-E6Jy5D1k9VIJ-0Z5Xt_4EIPZYXMR' -O checkpoints/initialization_finetuned.ckpt
pip install -r requirements.txt

# Fix: Clamp color values to [0, 255] before casting to uint8 to avoid overflow errors
sed -i 's/rgb \* 255/np.clip(rgb * 255, 0, 255)/g' /workspace/FlowMap/flowmap/export/colmap.py
