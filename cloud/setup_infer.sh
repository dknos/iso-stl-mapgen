#!/bin/bash
# Pod-side inference bootstrap for Blackwell (B200/B300). Fail-fast GPU sanity
# BEFORE the big model download so a broken kernel image dies in seconds.
set -e
cd /workspace
echo "=== GPU sanity (fail fast before 40GB model pull) ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda);assert torch.cuda.is_available();print('dev',torch.cuda.get_device_name(0));x=torch.randn(2048,2048,device='cuda');y=(x@x).sum().item();print('cuda matmul ok',y)"
echo "=== install inference deps ==="
pip install -q --no-cache-dir "diffusers>=0.35.0" "transformers>=4.46" accelerate peft safetensors sentencepiece pillow 2>&1 | tail -3
python -c "from diffusers import QwenImageEditPlusPipeline; print('QwenImageEditPlusPipeline import OK')"
echo "=== unpack tiles ==="
[ -f heldout.tgz ] && { rm -rf heldout && mkdir heldout && tar xzf heldout.tgz -C heldout --strip-components=1; }
echo "heldout=$(ls heldout/*.png 2>/dev/null | wc -l)"

# SWEEP mode: stylization-strength matrix -> one grid, then stop (skip normal inference)
if [ "${SWEEP:-0}" = "1" ]; then
  echo "=== SWEEP: stylization-strength matrix ==="
  python infer_sweep.py --tiles-dir heldout --lora iso_stl_diorama_v1.safetensors \
    --out-dir sweep_out --grid sweep_grid.png --steps "${INFER_STEPS:-28}" 2>&1 | tail -30
  echo "=== SWEEP DONE: $(ls sweep_out/*.png 2>/dev/null | wc -l) imgs, grid sweep_grid.png ==="
  exit 0
fi
STEPS="${INFER_STEPS:-28}"; LSCALE="${LORA_SCALE:-1.0}"; GUID="${GUIDANCE:-4.0}"
echo "=== inference: final -> styled_2000 (steps=$STEPS scale=$LSCALE guid=$GUID) ==="
python infer_batched.py --grid-dir heldout --out-dir styled_2000 --lora iso_stl_diorama_v1.safetensors --steps "$STEPS" --lora-scale "$LSCALE" --guidance "$GUID" 2>&1 | tail -12
# ckpt-1000 pass only if that lora was uploaded (held-out compare); skipped for single-lora city/test runs
if [ -f iso_stl_diorama_v1_000001000.safetensors ]; then
  echo "=== inference: ckpt1000 -> styled_1000 ==="
  python infer_batched.py --grid-dir heldout --out-dir styled_1000 --lora iso_stl_diorama_v1_000001000.safetensors --steps "$STEPS" 2>&1 | tail -12
fi
echo "styled_2000=$(ls styled_2000/*.png 2>/dev/null|wc -l) styled_1000=$(ls styled_1000/*.png 2>/dev/null|wc -l)"
echo "=== SETUP_INFER DONE ==="
