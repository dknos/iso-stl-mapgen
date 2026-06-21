#!/usr/bin/env python3
"""v5 walk = v4 dataset + checkerboard water aug. THE METHOD NEEDS BOTH HALVES:
training control water was checkered (omni_v5), so at inference we MUST checker the
raw input water too, or the model never sees the 'this is water -> fill flat' trigger
and reverts to v4-style hallucination.

This applies water_checker (delta=10, same as training) to each raw input tile's
flat-water regions BEFORE sending to the v5 endpoint. Per-tile full conversion.

Env:
  EP   = v5 modal endpoint (set after the v5 checkpoint is picked + served)
  Output -> out/v5_river/tile_{wedgecol}_{wedgerow}.png
Usage: EP=https://...modal.run python3 scripts/walk_v5.py [tiles.json]
"""
import json, base64, io, time, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy import ndimage

EP = os.environ.get("EP", "")
TRIGGER = "<isometric st louis pixel art>"
P_FULL = ("Convert this aerial render into the " + TRIGGER + " isometric 3D city diorama style, "
          "clean low-poly model, keeping the exact same layout and geometry.")
DELTA = int(os.environ.get("DELTA", "10"))

# --- checker (identical to scripts/water_checker.py, applied to INFERENCE input) ---
def water_mask(rgb):
    g = rgb.mean(2); k = 9
    mean = ndimage.uniform_filter(g, k); sq = ndimage.uniform_filter(g*g, k)
    std = np.sqrt(np.maximum(sq - mean*mean, 0))
    m = (std < 6.0) & (g < 200)
    lbl, n = ndimage.label(m); out = np.zeros_like(m); th = 0.03*m.size
    for i in range(1, n+1):
        c = lbl == i
        if c.sum() > th: out |= c
    return ndimage.binary_closing(out, iterations=3)

def checker(rgb, delta=DELTA):
    a = np.asarray(rgb.convert("RGB"))
    m = water_mask(a)
    out = a.astype(np.int16).copy()
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pat = (((xx//2)+(yy//2)) % 2)*2 - 1
    for c in range(3):
        ch = out[..., c]; ch[m] = np.clip(ch[m] + pat[m]*delta, 0, 255)
    return Image.fromarray(out.astype(np.uint8)), 100*m.mean()

def call(img, seed):
    b = io.BytesIO(); img.save(b, "PNG")
    body = json.dumps({"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": P_FULL,
                       "steps": 14, "guidance_scale": 3.0, "true_cfg_scale": 1.0, "seed": seed}).encode()
    for att in range(10):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                EP, data=body, headers={"Content-Type": "application/json"}), timeout=600))
            return Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        except Exception as e:
            print(f"  retry {att}: {str(e)[:40]}"); time.sleep(8)
    raise SystemExit("call failed")

def main():
    if not EP:
        raise SystemExit("set EP=<v5 modal endpoint>")
    tiles = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/river_tiles.json"))
    os.makedirs("/home/nemoclaw/iso-stl-lora/out/v5_river", exist_ok=True)
    done = [0]
    def do(i):
        c, r = tiles[i]
        src = f"/home/nemoclaw/iso-stl-lora/dataset/corridor_raw/tile_{c-11}_{r-2}.png"
        if not os.path.exists(src):
            src = f"/home/nemoclaw/iso-stl-lora/dataset/downtown_raw/tile_{c-5}_{r-9}.png"
        if not os.path.exists(src):
            print(f"  no raw for {c},{r}"); return
        raw = Image.open(src).convert("RGB").resize((1024, 1024))
        ck, wp = checker(raw)                       # <-- BOTH HALVES: checker the input water
        out = call(ck, 100 + i*7919)
        out.save(f"/home/nemoclaw/iso-stl-lora/out/v5_river/tile_{c}_{r}.png")
        done[0] += 1
        if done[0] % 10 == 0: print(f"  {done[0]}/{len(tiles)}")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(do, range(len(tiles))))
    print(f"v5 walk done (checkered inputs): {done[0]} tiles")

if __name__ == "__main__":
    main()
