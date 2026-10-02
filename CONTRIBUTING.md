# Contributing tiles

The live map is the isometric corridor traffic map:

https://stlcity2000.com/iso-map/corridor-traffic-fp

Capture your own raw tiles, style them, and send the styled tiles back in a pull request. You do not deploy the site.

## Weights and tools

| Piece | Where |
| --- | --- |
| Live map | https://stlcity2000.com/iso-map/corridor-traffic-fp |
| Base model | https://huggingface.co/Qwen/Qwen-Image-Edit-2509 |
| GGUF base (16 GB cards) | https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF — file `Qwen-Image-Edit-2509-Q4_K_M.gguf` |
| Text encoder GGUF | https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF |
| VAE and mmproj | same QuantStack repo (`qwen_image_vae.safetensors`, `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf`) |
| LoRA, new areas | https://github.com/dknos/iso-stl-mapgen/releases/download/v1.0/iso_stl_omni_v4_3000.safetensors |
| LoRA, downtown infill | https://github.com/dknos/iso-stl-mapgen/releases/download/v1.0/iso_stl_diorama_v1.safetensors |
| How to capture raw tiles | https://cannoneyed.com/projects/isometric-nyc |
| Isometric NYC map | https://cannoneyed.com/isometric-nyc/ |
| Stitcher | [`scripts/stitch.mjs`](scripts/stitch.mjs) |
| Style pass | [`scripts/infer_batched.py`](scripts/infer_batched.py) |
| Seam-free walk | [`scripts/walk_grid_rect.py`](scripts/walk_grid_rect.py) |

The release page with both LoRAs is https://github.com/dknos/iso-stl-mapgen/releases/tag/v1.0.

Use `iso_stl_omni_v4_3000` for a region that is not styled yet. Use `iso_stl_diorama_v1` only when you are filling a hole in the downtown style.

## Capture, then style

Make the raw `tile_C_R.png` files yourself, 1024×1024, the way the [Isometric NYC writeup](https://cannoneyed.com/projects/isometric-nyc) describes: an orthographic render of Google Maps 3D Tiles. The finished NYC map is at [cannoneyed.com/isometric-nyc](https://cannoneyed.com/isometric-nyc/). Use your own Google key, and do not commit it or the raw captures.

Then style the folder:

```bash
pip install torch diffusers transformers pillow numpy safetensors
python scripts/infer_batched.py \
  --grid-dir ./raw_tiles \
  --out-dir ./styled_tiles \
  --lora iso_stl_omni_v4_3000.safetensors \
  --steps 20 --lora-scale 1.0
```

Preview the block before you submit:

```bash
npm install sharp
node scripts/stitch.mjs --src ./styled_tiles --out region.png
```

`region.png` is a local preview. Do not commit it.

Cloud paths (your own Modal, RunPod, or rented GPU) are in the [README](README.md). A Modal endpoint has to be deployed with your own `ISO_EDIT_TOKEN`. Do not publish that URL or the token.

## Send the tiles back

```bash
mkdir -p submissions/my-region
cp styled_tiles/tile_*.png submissions/my-region/
```

Add `submissions/my-region/NOTES.md` with the column and row range you captured, the LoRA filename, the step count, and the command. Open a pull request. One region per pull request, about 50 tiles or fewer.

The maintainer checks that the style matches, then runs the shared finishing and stitch pass. Nothing in a pull request is deployed automatically.
