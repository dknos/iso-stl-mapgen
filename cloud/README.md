# Cloud training (RunPod H100) — iso-stl diorama LoRA

Train the Qwen-Image-Edit-2509 control LoRA on a rented H100 (~15–25 min,
~$1), download the `.safetensors`, run inference locally on the 5080.

## Why cloud
Local 5080 (16GB) can't fit Qwen-Image-Edit training (transformer alone is
~40GB bf16; even uint3-quantized it's ~1GB over, and `expandable_segments` —
the usual fix — crashes on Blackwell/cu130). H100 80GB runs it in full bf16,
no compromise, ~5–15× faster.

## Steps

### 0. Package the dataset (local)
```
bash ~/iso-stl-lora/cloud/package_dataset.sh
```
Produces `cloud/iso-stl-dataset.tar.gz`. (11 pairs now = ok for a validation
run; harvest toward ~40 for the final LoRA — see scripts/harvest.sh.)

### 1. Launch a pod (RunPod)
- GPU: **H100 80GB** (PCIe or SXM). Template: **RunPod PyTorch 2.4** (CUDA 12.x).
- Disk: 100GB+ container/volume (model is ~54GB).
- Start it, open the web terminal (or SSH).

### 2. Upload these to `/workspace` on the pod
- `cloud/iso-stl-dataset.tar.gz`
- `cloud/train_qwen_edit_cloud.yaml`
- `cloud/runpod_setup.sh`

(Use the RunPod file uploader, or `runpodctl send` from local, or `scp`.)

### 3. Run
```
cd /workspace
export HF_TOKEN=...        # optional, faster model download
bash runpod_setup.sh
```
Watch `output/iso_stl_diorama_v1/samples/` — sample images show the style
converging. The model auto-downloads (~54GB) on first run.

### 4. Download the LoRA
```
runpodctl send /workspace/output/iso_stl_diorama_v1/iso_stl_diorama_v1.safetensors
```
Drop it into `~/iso-stl-lora/output/iso_stl_diorama_v1/` locally.

### 5. Inference locally (5080)
```
~/iso-stl-lora/ai-toolkit/.venv/bin/python ~/iso-stl-lora/scripts/infer.py \
  --grid-dir ~/stlradar-3d/tools/iso-stl/out/raw \
  --out-dir  ~/stlradar-3d/tools/iso-stl/out/final_lora
```
Inference is far lighter than training (no gradients/optimizer) and should fit
16GB; if it OOMs, the same H100 pod can batch-infer all tiles in seconds.
Then harmonize + stitch with `tools/iso-stl/`.

## Cost
H100 ~$2–3/hr. A full run (download + 2000 steps) ≈ 45–75 min wall = **~$2–4**.
A 48GB A6000 (~$0.5/hr, set qtype:qfloat8 in the yaml) is the budget option.
