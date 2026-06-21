#!/usr/bin/env python3
"""Regenerate ONE region of the stylized map IN PLACE (cannoneyed 013-style), so we
never re-render the whole thing. Sources the RAW Google capture (which has the real
geometry) into a red box, surrounds it with the CURRENT render for context, infills
through the Modal omni server (so it blends natively), feather-pastes it back.

  python regen_region.py --box X0 Y0 X1 Y1 [--hint "..."] [--render final.png]
     [--tiles city_test] [--out test.png | --apply] [--seed 7] [--win 1024]
Non-destructive by default (writes --out). --apply overwrites the render.
"""
import argparse, base64, io, json, os, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage  # noqa (kept for parity)

EP = os.environ.get("EP")  # your Modal endpoint — deploy cloud/modal_omni_server.py, then: export EP=https://<you>--...modal.run
if not EP:
    raise SystemExit("Set EP to your Modal endpoint URL (deploy cloud/modal_omni_server.py with `modal deploy`, then export EP=...).")
RED = (237, 28, 36)
TRIGGER = "<isometric st louis pixel art>"
NEG = "blurry, hazy, photograph, text, watermark, red border, seam, tunnel, underpass"
Q = 512

def call(img, prompt, seed, steps=16, cfg=2.0):
    b = io.BytesIO(); img.save(b, "PNG")
    body = json.dumps({"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": prompt,
                       "negative_prompt": NEG, "steps": steps, "guidance_scale": 3.0,
                       "true_cfg_scale": cfg, "seed": seed}).encode()
    for att in range(8):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                EP, data=body, headers={"Content-Type": "application/json"}), timeout=600))
            return Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        except Exception as e:
            print(f"  retry {att}: {str(e)[:50]}"); time.sleep(10)
    raise SystemExit("server not responding")

def load_raw(tiles_dir, N=6, px=6144):
    """Stitch the raw Google-capture tiles (tile_c_r.png) into one full-res image."""
    canvas = Image.new("RGB", (px, px)); t = px // N
    for c in range(N):
        for r in range(N):
            p = os.path.join(tiles_dir, f"tile_{c}_{r}.png")
            if os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize((t, t)), (c*t, r*t))
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", nargs=4, type=int, required=True, metavar=("X0","Y0","X1","Y1"))
    ap.add_argument("--render", default="/home/nemoclaw/stlradar-3d/tools/iso-stl/out/walk_v3_6x6_parallel_final.png")
    ap.add_argument("--tiles", default="/home/nemoclaw/stlradar-3d/tools/iso-stl/out/city_test")
    ap.add_argument("--raw", default="", help="pre-built raw canvas image (crop instead of 6x6 load_raw)")
    ap.add_argument("--hint", default="")
    ap.add_argument("--out", default="/home/nemoclaw/stlradar-3d/tools/iso-stl/out/_regen_test.png")
    ap.add_argument("--apply", action="store_true", help="overwrite the render in place")
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--feather", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    render = Image.open(a.render).convert("RGB"); W, H = render.size
    raw = Image.open(a.raw).convert("RGB") if a.raw else load_raw(a.tiles, px=W)
    x0, y0, x1, y1 = a.box
    cx, cy = (x0+x1)//2, (y0+y1)//2
    win = a.win
    wx0 = max(0, min(W-win, cx-win//2)); wy0 = max(0, min(H-win, cy-win//2))
    window = render.crop((wx0, wy0, wx0+win, wy0+win)).copy()   # CURRENT render = context
    bx0, by0, bx1, by1 = x0-wx0, y0-wy0, x1-wx0, y1-wy0          # box in window coords
    # paste RAW geometry into the box region
    window.paste(raw.crop((x0, y0, x1, y1)), (bx0, by0))
    d = ImageDraw.Draw(window)
    for i in range(2): d.rectangle([bx0+i, by0+i, bx1-1-i, by1-1-i], outline=RED)
    print(f"window ({wx0},{wy0}) box=({bx0},{by0})-({bx1},{by1}) {100*(bx1-bx0)*(by1-by0)/(win*win):.0f}% of win")
    prompt = (f"Fill in the red-outlined region with the missing pixels in the {TRIGGER} isometric 3D city "
              "diorama style, removing the red border and exactly following the color, shape and structure of "
              "the surrounding already-generated image so the result is seamless.")
    if a.hint: prompt += " " + a.hint
    res = call(window, prompt, a.seed)
    if res.size != (win, win): res = res.resize((win, win))
    patch = res.crop((bx0, by0, bx1, by1)); pw, ph = patch.size
    mask = Image.new("L", (pw, ph), 255); md = ImageDraw.Draw(mask)
    for k in range(a.feather):
        v = int(255*(k+1)/(a.feather+1)); md.rectangle([k, k, pw-1-k, ph-1-k], outline=v)
    out = render.copy(); out.paste(patch, (x0, y0), mask)
    dst = a.render if a.apply else a.out
    out.save(dst)
    out.crop((max(0,x0-300), max(0,y0-300), min(W,x1+300), min(H,y1+300))).save("/tmp/diag_prev/regen_crop.png")
    print(f"regen -> {dst}  (crop -> /tmp/diag_prev/regen_crop.png)")

if __name__ == "__main__":
    main()
