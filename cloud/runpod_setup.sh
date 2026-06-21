#!/bin/bash
# RunPod bootstrap — run INSIDE a fresh pod (PyTorch 2.4+ / CUDA 12.x template,
# 48GB+ GPU). Sets up ai-toolkit, installs deps, pulls the model, trains, and
# leaves the LoRA in /workspace/output for download.
#
# Usage on the pod:
#   cd /workspace
#   # upload iso-stl-dataset.tar.gz + train_qwen_edit_cloud.yaml here first
#   bash runpod_setup.sh
set -e
cd /workspace

echo "=== 1. ai-toolkit ==="
[ -d ai-toolkit ] || git clone --depth 1 https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q --upgrade pip
# torch preinstalled in RunPod PyTorch template; install the rest.
pip install -q -r requirements.txt
# ai-toolkit imports torchaudio (config_modules) — ensure it matches the image's torch
python -c "import torch,sys; v=torch.__version__.split('+')[0]; print(v)" > /tmp/tv 2>/dev/null
TV=$(cat /tmp/tv)
pip install -q "torchaudio==${TV}" || pip install -q torchaudio
python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
cd /workspace

echo "=== 2. dataset ==="
[ -f iso-stl-dataset.tar.gz ] && tar xzf iso-stl-dataset.tar.gz   # -> dataset/{style,raw}
ls dataset/style/*.png | wc -l | xargs echo "pairs:"

echo "=== 3. HF auth (optional, faster dl) ==="
[ -n "$HF_TOKEN" ] && huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

echo "=== 4. train ==="
cd ai-toolkit
# model auto-downloads from HF on first run (~54GB, fast on pod bandwidth)
.venv/bin/python run.py /workspace/train_qwen_edit_cloud.yaml

echo "=== DONE — LoRA in /workspace/output/iso_stl_diorama_v1/ ==="
ls -la /workspace/output/iso_stl_diorama_v1/*.safetensors
echo "Download it with: runpodctl send /workspace/output/iso_stl_diorama_v1/iso_stl_diorama_v1.safetensors"
