# Contributing tiles

The live map is the isometric corridor traffic map:

https://stlcity2000.com/iso-map/corridor-traffic-fp

You style a pack of raw aerial tiles and send the styled tiles back in a pull request. You do not deploy the site, and you do not need the Google 3D Tiles key.

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
| Stitcher | [`scripts/stitch.mjs`](scripts/stitch.mjs) |
| Style pass | [`scripts/infer_batched.py`](scripts/infer_batched.py) |
| Seam-free walk | [`scripts/walk_grid_rect.py`](scripts/walk_grid_rect.py) |

The release page with both LoRAs is https://github.com/dknos/iso-stl-mapgen/releases/tag/v1.0.

Use `iso_stl_omni_v4_3000` for a region that is not styled yet. Use `iso_stl_diorama_v1` only when you are filling a hole in the downtown style.

## Style a tile pack

The maintainer publishes a folder of raw `tile_C_R.png` files (1024×1024). Ask in an issue if you need a pack. Then:

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

Add `submissions/my-region/NOTES.md` with the tile pack name, the LoRA filename, the step count, and the command. Open a pull request. One region per pull request, about 50 tiles or fewer.

The maintainer checks that the style matches, then runs the shared finishing and stitch pass. Nothing in a pull request is deployed automatically.
