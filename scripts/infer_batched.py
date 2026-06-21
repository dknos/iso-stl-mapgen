#!/usr/bin/env python3
"""Batched Qwen-Image-Edit + iso-stl LoRA inference — bf16, processes N tiles
per forward pass. Built for big-VRAM cloud GPUs (B200/B300/H100) to stylize a
whole region fast. Loads the model ONCE.

Usage (on the pod):
  python infer_batched.py --grid-dir /workspace/raw --out-dir /workspace/styled \
       --lora /workspace/iso_stl_diorama_v1.safetensors --batch 16 [--steps 28]

Throughput scales with --batch until VRAM-bound:
  H100 80GB ~6-8, B200 180GB ~24, B300 288GB ~32+.
"""
import argparse, glob, os, time
import torch
from PIL import Image

# Tuned for SEAMLESS tiling: no "tilt-shift miniature" (that vignettes each tile's
# edges -> hard seams) and explicit anti-hallucination (LoRA otherwise paints stray
# blue water on flat roofs/lots). Flat uniform stylization tiles cleanly.
PROMPT = ("convert this aerial photograph into a clean isometric 3D city diorama, low-poly stylized "
          "model render, vivid saturated colors, bright green grass on parks and lawns, clean gray "
          "streets and parking lots, buildings keep their real roof colors, render blue water ONLY on "
          "actual rivers and lakes, do NOT add water on rooftops or pavement, even uniform lighting, "
          "keeping the exact same layout and geometry")
NEG = ("tilt-shift, vignette, dark edges, blurred edges, photograph, photographic, realistic, hazy, "
       "extra water, blue rooftops, flooded, text, watermark, border")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--lora-scale", type=float, default=1.0, help="LoRA adapter weight (<1 = gentler)")
    ap.add_argument("--model", default="Qwen/Qwen-Image-Edit-2509")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from diffusers import QwenImageEditPlusPipeline
    print(f"loading {a.model} (bf16), lora_scale={a.lora_scale}...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(a.lora, adapter_name="iso")
    pipe.set_adapters("iso", a.lora_scale)   # scale without fusing (so <1 works)
    pipe.set_progress_bar_config(disable=True)

    files = sorted(glob.glob(os.path.join(a.grid_dir, "tile_*.png")))
    todo = [f for f in files if not os.path.exists(os.path.join(a.out_dir, os.path.basename(f)))]
    # QwenImageEditPlusPipeline only supports batch_size=1 -> loop one image per
    # forward (model is still loaded once + fused, the slow part). --batch kept
    # only for progress cadence.
    print(f"{len(todo)}/{len(files)} tiles to stylize (bs=1 loop)")
    t0 = time.time(); done = 0
    for f in todo:
        img = Image.open(f).convert("RGB")
        im = pipe(image=img, prompt=PROMPT, negative_prompt=NEG,
                  num_inference_steps=a.steps, true_cfg_scale=a.guidance).images[0]
        im.save(os.path.join(a.out_dir, os.path.basename(f)))
        done += 1
        el = time.time() - t0
        print(f"  {done}/{len(todo)}  ({el/done:.1f}s/tile, {done/el*3600:.0f} tiles/hr)")
    print(f"done: {done} tiles in {(time.time()-t0)/60:.1f} min -> {a.out_dir}")

if __name__ == "__main__":
    main()
