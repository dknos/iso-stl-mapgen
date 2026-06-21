"""Cannoneyed lesson 2: water augmentation. Edit models can't 'understand' flat
water (a flat color reads as noise -> hallucination). Fix: paint a faint 2x2
checkerboard over water regions in the RENDER (control) tiles. That structured
signal teaches the model 'this is water, fill with a pure flat color' instead of
hallucinating. Apply to BOTH training control tiles AND inference inputs.

Water mask = large, low-texture (flat) regions of the render (the river is a big
smooth body; buildings/streets are high-variance). Detected per-tile, no GIS dep.

Usage: python water_checker.py --in-dir dataset/raw --out-dir dataset/raw_ck \
         [--mask-dir dataset/watermask] [--delta 10] [--debug grid.png]
"""
import argparse, glob, os
import numpy as np
from PIL import Image
from scipy import ndimage

def water_mask(rgb):
    """Large smooth low-variance region = water. Returns bool mask."""
    g = rgb.mean(2)
    # local std via box filter on (x^2)-mean^2
    k = 9
    mean = ndimage.uniform_filter(g, k)
    sq = ndimage.uniform_filter(g*g, k)
    std = np.sqrt(np.maximum(sq - mean*mean, 0))
    flat = std < 6.0                                  # smooth
    # water tends to be mid/dark and not pure white pavement; allow brown or blue
    notwhite = g < 200
    m = flat & notwhite
    # keep only large connected components (>3% of image) -> the river, not roofs
    lbl, n = ndimage.label(m)
    out = np.zeros_like(m)
    thresh = 0.03 * m.size
    for i in range(1, n+1):
        comp = lbl == i
        if comp.sum() > thresh:
            out |= comp
    # close small holes
    out = ndimage.binary_closing(out, iterations=3)
    return out

def checker(rgb, mask, delta=10):
    out = rgb.astype(np.int16).copy()
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pat = (((xx // 2) + (yy // 2)) % 2) * 2 - 1        # +/-1 in 2x2 blocks
    for c in range(3):
        ch = out[..., c]
        ch[mask] = np.clip(ch[mask] + pat[mask] * delta, 0, 255)
    return out.astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mask-dir", default="")
    ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--debug", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mask_dir: os.makedirs(a.mask_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.in_dir, "*.png")))
    dbg = []
    for f in files:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        m = water_mask(rgb)
        ck = checker(rgb, m, a.delta)
        Image.fromarray(ck).save(os.path.join(a.out_dir, os.path.basename(f)))
        if a.mask_dir:
            Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(a.mask_dir, os.path.basename(f)))
        pct = 100*m.mean()
        print(f"  {os.path.basename(f)}: water {pct:.0f}%")
        if a.debug and len(dbg) < 4:
            dbg.append((Image.fromarray(rgb).resize((256,256)),
                        Image.fromarray((m*255).astype(np.uint8)).convert("RGB").resize((256,256)),
                        Image.fromarray(ck).resize((256,256))))
    if a.debug and dbg:
        g = Image.new("RGB", (256*3, 256*len(dbg)), (20,20,20))
        for r,(o,mk,c) in enumerate(dbg):
            g.paste(o,(0,r*256)); g.paste(mk,(256,r*256)); g.paste(c,(512,r*256))
        g.save(a.debug); print(f"debug (render|mask|checker) -> {a.debug}")
    print(f"done -> {a.out_dir}")

if __name__ == "__main__":
    main()
