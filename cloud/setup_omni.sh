#!/bin/bash
# RunPod bootstrap for OMNI INFILL LoRA training. Upload omni-dataset.tar.gz +
# train_qwen_omni.yaml here first, then: bash setup_omni.sh
set -e
cd /workspace

echo "=== 1. ai-toolkit ==="
[ -d ai-toolkit ] || git clone --depth 1 https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
python -c "import torch; print(torch.__version__.split('+')[0])" > /tmp/tv 2>/dev/null
TV=$(cat /tmp/tv)
pip install -q "torchaudio==${TV}" || pip install -q torchaudio
python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
cd /workspace

echo "=== 2. omni dataset ==="
[ -f omni-dataset.tar.gz ] && tar xzf omni-dataset.tar.gz   # -> dataset/omni_v1/{control,target}
echo "control=$(ls dataset/omni_v1/control/*.png 2>/dev/null|wc -l) target=$(ls dataset/omni_v1/target/*.png 2>/dev/null|wc -l)"

echo "=== 3. HF auth (optional) ==="
[ -n "$HF_TOKEN" ] && huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

echo "=== 4. train omni ==="
cd ai-toolkit
.venv/bin/python run.py /workspace/train_qwen_omni.yaml

echo "=== DONE — omni LoRA in /workspace/output/iso_stl_omni_v1/ ==="
ls -la /workspace/output/iso_stl_omni_v1/*.safetensors 2>/dev/null
