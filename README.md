# iso-stl-mapgen

Tools to generate the **isometric 3D diorama map tiles** behind the corridor traffic map at
[stlcity2000.com/iso-map/corridor-traffic-fp](https://stlcity2000.com/iso-map/corridor-traffic-fp),
and a way to **contribute GPU compute** to extend it.

The map is built by taking ortho aerial captures of St. Louis and restyling each tile
into a low-poly toy-diorama look with a trained **Qwen-Image-Edit LoRA**. Generating a
city is GPU-heavy, so this repo lets anyone with a GPU (local) or a cloud account
(Modal / RunPod / Lambda) style a batch of tiles and send them back.

How to submit tiles is in [CONTRIBUTING.md](CONTRIBUTING.md). What must stay off GitHub is in [SECURITY.md](SECURITY.md).

## Links

| Piece | Where |
| --- | --- |
| Live traffic map | https://stlcity2000.com/iso-map/corridor-traffic-fp |
| Base model | https://huggingface.co/Qwen/Qwen-Image-Edit-2509 |
| GGUF quant (ComfyUI, 16 GB) | https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF (`Qwen-Image-Edit-2509-Q4_K_M.gguf`) |
| Text encoder / VAE / mmproj | [QuantStack README](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF) and [unsloth Qwen2.5-VL-7B GGUF](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF) |
| LoRA for new areas | [iso_stl_omni_v4_3000.safetensors](https://github.com/dknos/iso-stl-mapgen/releases/download/v1.0/iso_stl_omni_v4_3000.safetensors) |
| LoRA for downtown infill | [iso_stl_diorama_v1.safetensors](https://github.com/dknos/iso-stl-mapgen/releases/download/v1.0/iso_stl_diorama_v1.safetensors) |
| Both LoRAs | [v1.0 release](https://github.com/dknos/iso-stl-mapgen/releases/tag/v1.0) |
| Stitcher | [`scripts/stitch.mjs`](scripts/stitch.mjs) |

---

## The contribution model: you-capture / they-style

The raw aerial capture needs a Google 3D-Tiles key and a capture rig — that part stays
with the maintainer. **You only run the style pass**, which needs neither.

1. **Maintainer** publishes a *tile pack*: a folder of raw `tile_C_R.png` (1024×1024) for a
   region that isn't styled yet, plus the LoRA weights (see Releases).
2. **You** run the LoRA style pass on your GPU or cloud account → styled tiles.
3. **You** submit the styled tiles back (see *Submitting* below). The maintainer runs the
   shared finishing/water/stitch pass so the whole map stays consistent.

No Google key, no HuggingFace gating beyond the base model, no private data needed.

---

## Get the weights (GitHub Releases)

The LoRA adapters are ~563 MB each and ship as **Release assets** (not in the git tree):

| Asset | Use |
|-------|-----|
| `iso_stl_omni_v4_3000.safetensors` | **Default.** The generalized "omni" LoRA — best for styling *new* areas. |
| `iso_stl_diorama_v1.safetensors` | Tuned to match the existing downtown St. Louis style (for seamless infill of already-styled regions). |

Download from the [latest release](../../releases/latest) and pass the path with `--lora`.

The **base model** is `Qwen/Qwen-Image-Edit-2509` (~20B). It downloads from HuggingFace on
first run; you may need a HuggingFace account / `HF_TOKEN` to accept its license. The local
ComfyUI path uses a GGUF quant of the same model (see `cloud/start_local_comfy.sh`).

---

## Run the style pass

### Local GPU (simplest, $0)
A single NVIDIA card with enough VRAM (bf16 needs ~24 GB; an 8–16 GB card can use the
quantized ComfyUI path in `cloud/`).

```bash
pip install torch diffusers transformers pillow numpy safetensors
python scripts/infer_batched.py \
  --grid-dir ./raw_tiles \
  --out-dir ./styled_tiles \
  --lora iso_stl_omni_v4_3000.safetensors \
  --steps 20 --lora-scale 1.0
```

### Modal (serverless — new accounts get $30 free credit)
Deploy your **own** endpoint (you never use anyone else's — that's the point):

```bash
pip install modal && modal token new
export ISO_EDIT_TOKEN=$(openssl rand -hex 24)   # keep this private; the server refuses to deploy without it
modal volume create isometric-lora-vol
modal volume put isometric-lora-vol iso_stl_omni_v4_3000.safetensors /loras/iso-stl-omni/iso_stl_omni_v4_3000.safetensors
LORA_MODEL_ID=iso-stl-omni modal deploy cloud/modal_omni_server.py
# Modal prints your endpoint URL. Do not commit it. Use it:
export EP=https://<you>--qwen-edit-stl-omni-imageeditor-edit-b64.modal.run
python scripts/walk_grid_rect.py --cols 6 --rows 6 --tiles ./raw_tiles --endpoint "$EP" --out region.png
```

`walk_grid_rect.py` sends `ISO_EDIT_TOKEN` with each request. The endpoint bills your Modal account, so leave the URL and the token off GitHub, Discord, and the pull request.

`$30` of B200 credit styles a lot of tiles (roughly $0.50–1.50 per region).

### RunPod
```bash
echo "RUNPOD_API_KEY=..." > ~/.iso_runpod_env
python cloud/runpod_infer.py --tiles raw_tiles.tgz --lora iso_stl_omni_v4_3000.safetensors --out ./styled
```

### Lambda / any rented H100
Same as **Local GPU** above — SSH in, `pip install`, run `infer_batched.py`.

---

## The grid-replacement tools

The map is regenerated *in place* tile-by-tile so you never re-render the whole city:

- **`scripts/walk_grid_rect.py`** — seam-free wavefront walk over a `cols × rows` rect. Each
  interior tile is infilled with its already-styled neighbors as context → no seams.
- **`scripts/grid_replace.py`** — replace specific tiles in a finished map (water/farmland/
  barge fixes etc.) with hinted prompts.
- **`scripts/regen_region.py`** / **`scripts/fix_region.py`** — re-style one bad region of a
  finished map (sources the raw geometry into a red box, infills through your endpoint,
  feather-pastes it back). Set `EP` to your Modal endpoint.
- **`scripts/stitch.mjs`** — stitch `tile_C_R.png` files into one preview PNG. Needs `npm install sharp`.
- **`scripts/water_fix.py`**, **`scripts/finishing_pass.py`**, **`scripts/make_dzi.py`**,
  **`scripts/stitch_deploy.py`** — the maintainer's finishing/water/stitch/deep-zoom pass.
  These are not run by pull requests.

Tiles are named `tile_C_R.png` (column, row), 1024×1024 PNG. Output keeps the same names.

---

## Submitting styled tiles

Open a pull request adding your styled `tile_C_R.png` files under `submissions/<region>/`.
Read [CONTRIBUTING.md](CONTRIBUTING.md) first. The maintainer reviews style consistency on
the PR before stitching them into the [live traffic map](https://stlcity2000.com/iso-map/corridor-traffic-fp).
In the PR description, note the source tile-pack and which LoRA / step count you used.
Do not include API keys, the Modal URL, or the `ISO_EDIT_TOKEN`.

---

## Training your own

`configs/iso_qwen_edit_16gb.yaml` (local 16 GB) and `cloud/train_qwen_*.yaml` (cloud) train a
control LoRA on aerial→styled pairs with [ai-toolkit](https://github.com/ostris/ai-toolkit).
The training dataset is not distributed here.

---

See `NOTICE.md` for provenance and terms.
