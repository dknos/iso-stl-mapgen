#!/usr/bin/env python3
"""Final-map finishing for the iso-stl diorama:
  - soften_whites: aggressive highlight rolloff (bright roofs -> light gray)
  - strict_water:  keep only the LARGEST connected blue body (the river); recolor
                   stray blue ponds/creeks (hallucinated water in the Arch grounds
                   etc.) to the surrounding grass/ground
  - mute_green:    desaturate + darken neon grass (too much / too bright green)
  - tag:           tiny discrete 'github: @dknos' on a flat gray surface
Usage: python finishing_pass.py --in map.png --out final.png [--knee 185]
       [--no-water] [--no-green] [--text "github: @dknos"]
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

def soften_whites(f, knee=185, ceil=222):
    luma = 0.299*f[...,0] + 0.587*f[...,1] + 0.114*f[...,2]
    m = luma > knee
    if m.any():
        target = knee + (luma - knee) * ((ceil - knee) / (255.0 - knee))
        scale = np.where(luma > 0, target / np.maximum(luma, 1e-3), 1.0)
        for c in range(3): f[...,c] = np.where(m, f[...,c]*scale, f[...,c])
    return f

def strict_water(f, river_min=200000):
    """Keep large connected water (the river, >=river_min px). Recolor every SMALLER
    blue blob (hallucinated ponds in railyards / parking / parks) to its LOCAL
    shoreline ground colour -- grey on pavement, green in parks -- sampled per blob.
    Only blue pixels change (buildings, the Arch, roads stay untouched), and we work
    inside each blob's bounding box so it's fast + can't paint rectangles or the river."""
    H, W = f.shape[:2]
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    blue = (b - r > 14) & (b - g > 8) & (b > 90)
    lbl, n = ndimage.label(blue)
    if n == 0:
        return f
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    slices = ndimage.find_objects(lbl)
    rng = np.random.RandomState(7)
    fixed = 0
    for i in range(1, n + 1):
        sz = sizes[i - 1]
        if sz >= river_min or sz < 40:               # keep river; ignore tiny specks
            continue
        sl = slices[i - 1]
        if sl is None:
            continue
        ys, xs = sl
        y0, y1 = max(0, ys.start - 22), min(H, ys.stop + 22)
        x0, x1 = max(0, xs.start - 22), min(W, xs.stop + 22)
        sub_f = f[y0:y1, x0:x1]
        blob = lbl[y0:y1, x0:x1] == i
        sub_blue = blue[y0:y1, x0:x1]
        shore = ndimage.binary_dilation(blob, iterations=18) & (~blob) & (~sub_blue)
        fill = (np.median(sub_f[shore], axis=0) if shore.sum() >= 30
                else np.array([150, 148, 143], np.float32))
        idx = np.where(blob)
        noise = rng.randint(-5, 6, size=(idx[0].size, 3))
        for c in range(3):
            sub_f[..., c][blob] = np.clip(fill[c] + noise[:, c], 0, 255)
        fixed += 1
    print(f"  deponded {fixed} stray pools (kept river>{river_min}px)")
    return f

def mute_green(f, sat_keep=0.62, darken=0.92):
    """Desaturate + slightly darken neon grass: pixels where green dominates."""
    r, g, b = f[...,0], f[...,1], f[...,2]
    green = (g - r > 18) & (g - b > 18)
    if green.any():
        luma = (0.299*r + 0.587*g + 0.114*b)
        for c in range(3):
            # pull toward luma (desaturate), then darken
            f[...,c] = np.where(green, (luma*(1-sat_keep) + f[...,c]*sat_keep) * darken, f[...,c])
    return f

def find_flat_spot(arr, tw, th):
    """Vectorized: analyze a downscaled copy (fast), scale the chosen point back."""
    H, W = arr.shape[:2]
    ds = max(1, W // 1400)                       # downscale factor
    small = arr[::ds, ::ds].astype(np.float32)
    g = small.mean(2); k = max(3, 11 // ds)
    mean = ndimage.uniform_filter(g, k); sq = ndimage.uniform_filter(g*g, k)
    std = np.sqrt(np.maximum(sq - mean*mean, 0))
    rm = ndimage.uniform_filter(small[...,0], k); gm = ndimage.uniform_filter(small[...,1], k); bm = ndimage.uniform_filter(small[...,2], k)
    gray = (np.maximum(np.maximum(rm, gm), bm) - np.minimum(np.minimum(rm, gm), bm)) < 18
    flat = (std < 6) & (mean > 95) & (mean < 170) & gray
    lbl, n = ndimage.label(flat)
    if n == 0: return (W-tw-60, H-th-60)
    idx = np.arange(1, n+1)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, idx)
    coms = ndimage.center_of_mass(np.ones_like(lbl), lbl, idx)   # (row,col) per label, vectorized
    sh, sw = g.shape; best = None; bestp = -1
    min_area = (tw*th*1.5) / (ds*ds)
    for (cy, cx), sz in zip(coms, sizes):
        if sz < min_area: continue
        fx, fy = cx*ds, cy*ds
        if fx < tw//2+12 or fx > W-tw//2-12 or fy < th//2+12 or fy > H-th//2-12: continue
        periph = (((fx-W/2)/W)**2 + ((fy-H/2)/H)**2)**0.5
        if periph > bestp: bestp = periph; best = (int(fx-tw//2), int(fy-th//2))
    return best or (W-tw-60, H-th-60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="github: @dknos")
    ap.add_argument("--knee", type=int, default=185)
    ap.add_argument("--no-water", action="store_true")
    ap.add_argument("--green", action="store_true", help="desaturate greens (default OFF -- keep vibrant raw greens)")
    a = ap.parse_args()
    im = Image.open(a.inp).convert("RGB")
    f = np.asarray(im).astype(np.float32)
    f = soften_whites(f, a.knee)
    if not a.no_water: f = strict_water(f)
    if a.green: f = mute_green(f)
    arr = np.clip(f, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr); W, H = im.size
    fp = max(11, int(W*0.0042))
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fp)
    except Exception: font = ImageFont.load_default()
    d = ImageDraw.Draw(im); tw = int(d.textlength(a.text, font=font)); th = fp
    x, y = find_flat_spot(arr, tw, th)
    patch = arr[max(0,y):min(H,y+th), max(0,x):min(W,x+tw)]
    base = float(patch.mean()) if patch.size else 150.0
    col = tuple(int(np.clip(base + (-55 if base > 110 else 55), 0, 255)) for _ in range(3))
    d.text((x, y), a.text, fill=col, font=font); im.save(a.out)
    print(f"finished (knee{a.knee}, water={'off' if a.no_water else 'strict'}, "
          f"green={'muted' if a.green else 'kept-vibrant'}, tag@({x},{y})) -> {a.out}")

if __name__ == "__main__":
    main()
