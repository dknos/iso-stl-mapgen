#!/usr/bin/env python3
"""Stylization-strength sweep: load Qwen-Image-Edit + iso-stl LoRA ONCE, run a
matrix of (lora_scale, guidance, prompt) over a few representative tiles, and
build one labeled grid (rows=tiles, cols=settings) so the user dials in how
stylized vs realistic. Pushes past the default look ("too realistic").

Pod usage:
  python infer_sweep.py --tiles-dir sweep_tiles --lora iso_stl_diorama_v1.safetensors \
      --out-dir sweep_out --grid sweep_grid.png
"""
import argparse, glob, os
import torch
from PIL import Image, ImageDraw, ImageFont

BASE = ("convert this aerial photograph into a detailed isometric 3D diorama city render, "
        "tilt-shift miniature, vivid saturated colors, clean crisp buildings, bright green grass, "
        "clean blue water, keeping the exact same layout and geometry")
STRONG = ("Convert this aerial photo into a STYLIZED isometric miniature city DIORAMA - a tilt-shift "
          "scale-model / video-game city render. Toy-like matte surfaces, low-poly clean shapes, vivid "
          "saturated colors, bright green grass, clean blue water. It must look like a MODEL, NOT a real "
          "photograph. Keep the exact same layout and geometry.")
NEG_BASE = "photographic, photo, blurry, hazy, text, watermark, border"
NEG_STRONG = "photograph, photographic, realistic, real aerial photo, satellite imagery, dull, hazy, blurry, text, watermark, border"

# (label, lora_scale, guidance, prompt, negative)
COMBOS = [
    ("base s1.0 g4",      1.0, 4.0, BASE,   NEG_BASE),
    ("s1.4 g4.5",         1.4, 4.5, BASE,   NEG_BASE),
    ("s1.8 g5",           1.8, 5.0, BASE,   NEG_STRONG),
    ("STRONG s1.4 g5",    1.4, 5.0, STRONG, NEG_STRONG),
    ("STRONG s1.8 g6",    1.8, 6.0, STRONG, NEG_STRONG),
    ("STRONG s2.2 g6",    2.2, 6.0, STRONG, NEG_STRONG),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--model", default="Qwen/Qwen-Image-Edit-2509")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from diffusers import QwenImageEditPlusPipeline
    print(f"loading {a.model} (bf16)...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(a.lora, adapter_name="iso")   # NO fuse -> scale per combo
    pipe.set_progress_bar_config(disable=True)

    tiles = sorted(glob.glob(os.path.join(a.tiles_dir, "*.png")))
    print(f"{len(tiles)} tiles x {len(COMBOS)} settings")
    results = {}   # (combo_label, tile_name) -> path
    for label, scale, guid, prompt, neg in COMBOS:
        pipe.set_adapters("iso", scale)
        for f in tiles:
            img = Image.open(f).convert("RGB")
            out = pipe(image=img, prompt=prompt, negative_prompt=neg,
                       num_inference_steps=a.steps, true_cfg_scale=guid).images[0]
            p = os.path.join(a.out_dir, f"{label.replace(' ','_')}__{os.path.basename(f)}")
            out.save(p); results[(label, os.path.basename(f))] = p
            print(f"  {label} | {os.path.basename(f)}")

    # grid: rows = tiles, cols = combos, +header row
    CELL, HDR = 320, 30
    tnames = [os.path.basename(f) for f in tiles]
    W = len(COMBOS) * CELL
    H = HDR + len(tnames) * CELL
    grid = Image.new("RGB", (W, H), (17, 17, 17))
    d = ImageDraw.Draw(grid)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception: font = ImageFont.load_default()
    for ci, (label, *_ ) in enumerate(COMBOS):
        d.text((ci * CELL + 6, 7), label, fill="white", font=font)
    for ri, tn in enumerate(tnames):
        for ci, (label, *_ ) in enumerate(COMBOS):
            im = Image.open(results[(label, tn)]).convert("RGB").resize((CELL, CELL))
            grid.paste(im, (ci * CELL, HDR + ri * CELL))
    grid.save(a.grid)
    print(f"grid -> {a.grid} ({W}x{H})")

if __name__ == "__main__":
    main()
