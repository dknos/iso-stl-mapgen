#!/usr/bin/env python3
"""Targeted OMNI infill of one bad region in a finished map, in place.
Builds a 1024 window centered on the bad bbox (good generated pixels = context),
red-boxes the bad sub-region, runs it through the LoRA server, feather-pastes the
fixed region back. Reusable for whack-a-mole defect cleanup (arch, stray ponds,
glitches) on the 6x6 AND the full 1296 city.

Usage:
  python fix_region.py --img final.png --bbox X0 Y0 X1 Y1 [--out final.png]
     [--hint "the region is a railyard/parking lot, paved gray, not water"]
     [--win 1024] [--feather 12] [--seed 7]
"""
import argparse, base64, io, json, os, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw

EP = os.environ.get("EP")  # your Modal endpoint — deploy cloud/modal_omni_server.py, then: export EP=https://<you>--...modal.run
if not EP:
    raise SystemExit("Set EP to your Modal endpoint URL (deploy cloud/modal_omni_server.py with `modal deploy`, then export EP=...).")
RED = (237, 28, 36)
TRIGGER = "<isometric st louis pixel art>"
NEG = "blurry, hazy, photograph, text, watermark, red border, seam, water"

def call(img, prompt, seed, cfg=2.0):
    b = io.BytesIO(); img.save(b, "PNG")
    body = json.dumps({"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": prompt,
                       "negative_prompt": NEG, "steps": 16, "guidance_scale": 3.0,
                       "true_cfg_scale": cfg, "seed": seed}).encode()
    for att in range(8):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                EP, data=body, headers={"Content-Type": "application/json"}), timeout=600))
            return Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        except Exception as e:
            print(f"  retry {att}: {str(e)[:50]}"); time.sleep(12)
    raise SystemExit("server not responding")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--bbox", nargs=4, type=int, required=True, metavar=("X0","Y0","X1","Y1"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--hint", default="")
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--feather", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--prefill", action="store_true",
                    help="paint the bad bbox with surrounding-ground gray before infill "
                         "(removes the blue/water prior so the model fills fresh)")
    a = ap.parse_args()
    out = a.out or a.img
    im = Image.open(a.img).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = a.bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    win = a.win
    # window clamped to image
    wx0 = max(0, min(W - win, cx - win // 2)); wy0 = max(0, min(H - win, cy - win // 2))
    wx1, wy1 = wx0 + win, wy0 + win
    window = im.crop((wx0, wy0, wx1, wy1)).copy()
    # bad bbox in window coords
    bx0, by0, bx1, by1 = x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0
    area = (bx1 - bx0) * (by1 - by0)
    print(f"window ({wx0},{wy0})-({wx1},{wy1})  redbox=({bx0},{by0})-({bx1},{by1})  "
          f"{100*area/(win*win):.0f}% of window")
    if a.prefill:
        # sample a ground color from a ring just OUTSIDE the bad bbox (skip blue/water),
        # then paint the bbox interior with it so the model has no blue to copy.
        wa = np.asarray(window).astype(np.int16)
        ring = np.zeros((win, win), bool)
        m = 40
        ring[max(0,by0-m):min(win,by1+m), max(0,bx0-m):min(win,bx1+m)] = True
        ring[max(0,by0):min(win,by1), max(0,bx0):min(win,bx1)] = False  # exclude interior
        rr, gg, bb = wa[...,0], wa[...,1], wa[...,2]
        notblue = ~((bb-rr>14)&(bb-gg>8)&(bb>90))
        sel = ring & notblue
        if sel.sum() > 50:
            fill = tuple(int(np.median(wa[sel][:, c])) for c in range(3))
        else:
            fill = (150, 148, 143)
        print(f"  prefill bbox with ground gray {fill}")
        ImageDraw.Draw(window).rectangle([bx0, by0, bx1-1, by1-1], fill=fill)
    d = ImageDraw.Draw(window)
    for i in range(2): d.rectangle([bx0+i, by0+i, bx1-1-i, by1-1-i], outline=RED)
    prompt = (f"Fill in the red-outlined region with the missing pixels in the {TRIGGER} isometric 3D "
              "city diorama style, removing the red border and exactly following the color, shape, and "
              "structure of the surrounding already-generated image so the result is seamless.")
    if a.hint: prompt += " " + a.hint
    print("infill call...")
    res = call(window, prompt, a.seed, a.cfg)
    if res.size != (win, win): res = res.resize((win, win))
    # feather-paste ONLY the red-boxed region back into the full image
    patch = res.crop((bx0, by0, bx1, by1))
    pw, ph = patch.size
    mask = Image.new("L", (pw, ph), 255)
    md = ImageDraw.Draw(mask)
    fe = a.feather
    for k in range(fe):  # linear feather border
        v = int(255 * (k + 1) / (fe + 1))
        md.rectangle([k, k, pw-1-k, ph-1-k], outline=v)
    base = im.copy()
    base.paste(patch, (x0, y0), mask)
    base.save(out)
    print(f"fixed region -> {out}")

if __name__ == "__main__":
    main()
