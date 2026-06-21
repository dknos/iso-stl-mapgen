#!/usr/bin/env python3
"""Local inference: Qwen-Image-Edit-2509 + trained iso-stl diorama LoRA.

Runs entirely on the 5080 (quantized). Stylizes raw ortho tiles into the
diorama look with ZERO quota — the whole point of training locally.

Usage:
  python3 scripts/infer.py --in tile.png --out styled.png
  python3 scripts/infer.py --grid-dir /path/raw --out-dir /path/styled   # batch

Requires the ai-toolkit venv (diffusers + torchao). Run with:
  ~/iso-stl-lora/ai-toolkit/.venv/bin/python scripts/infer.py ...
"""
import argparse, glob, os
from pathlib import Path
import torch
from PIL import Image

LORA = str(Path.home() / "iso-stl-lora" / "output" / "iso_stl_diorama_v1" /
           "iso_stl_diorama_v1.safetensors")
PROMPT = ("convert this aerial photograph into a detailed isometric 3D diorama city render, "
          "tilt-shift miniature, vivid saturated colors, clean crisp buildings, bright green "
          "grass, clean blue water, keeping the exact same layout and geometry")

def load_pipe(lora_path):
    from diffusers import QwenImageEditPlusPipeline  # diffusers >= the ai-toolkit pin
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()           # fit 16GB
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        print(f"loaded LoRA: {lora_path}")
    else:
        print("WARNING: no LoRA found — running base model (no learned style)")
    return pipe

def run(pipe, src, dst, steps, guidance):
    img = Image.open(src).convert("RGB")
    out = pipe(image=img, prompt=PROMPT, negative_prompt="photographic, blurry, text, border",
               num_inference_steps=steps, true_cfg_scale=guidance).images[0]
    out.save(dst)
    print(f"  {os.path.basename(src)} -> {dst}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", dest="out")
    ap.add_argument("--grid-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--lora", default=LORA)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    a = ap.parse_args()
    pipe = load_pipe(a.lora)
    if a.grid_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(a.grid_dir, "tile_*.png"))):
            run(pipe, f, os.path.join(a.out_dir, os.path.basename(f)), a.steps, a.guidance)
    else:
        run(pipe, a.inp, a.out, a.steps, a.guidance)

if __name__ == "__main__":
    main()
