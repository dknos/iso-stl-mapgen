#!/usr/bin/env python3
"""Soft-Blend water normalization (cannoneyed /tasks/018_water_fix method).

Recolors water across stylized tiles to ONE canonical blue using a soft alpha
matte (color-distance -> gray mask -> alpha composite), so anti-aliased edges
at banks/bridges stay clean and EVERY tile's water matches -> seamless map.

Per-tile target auto-detection: water hue drifts tile-to-tile, so unlike the
fixed-sample original we detect each tile's dominant water color (large, smooth,
blue-dominant region) and blend THAT toward the canonical color. --target can
override with an explicit R,G,B.

Usage:
  python water_fix.py --in-dir styled_2000 --out-dir styled_2000_water \
      --new 60,150,220 --softness 60 [--target 75,105,125] [--preview grid.png]
"""
import argparse, glob, os
import numpy as np
from PIL import Image

def parse_rgb(s):
    return tuple(int(x) for x in s.split(","))

def auto_water_target(rgb):
    """Estimate the tile's dominant water color: pixels that are blue-dominant
    (B > R and B > G), reasonably bright, and in a large smooth area. Returns
    the median color of that cluster, or None if no plausible water."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bluish = (b.astype(np.int16) - r > 12) & (b.astype(np.int16) - g > 4) & (b > 70) & (b < 250)
    if bluish.sum() < rgb.shape[0] * rgb.shape[1] * 0.02:   # <2% area -> no real water
        return None, bluish
    sel = rgb[bluish]
    return tuple(np.median(sel, axis=0).astype(int)), bluish

def soft_color_replace(img_float, target, new, softness):
    """cannoneyed soft blend: float math, Euclidean color distance -> soft mask."""
    t = np.array(target, np.float32) / 255.0
    n = np.array(new, np.float32) / 255.0
    softness_scale = max((softness / 255.0) * np.sqrt(3), 1e-5)
    dist = np.linalg.norm(img_float - t, axis=2)
    alpha = np.clip(1.0 - dist / softness_scale, 0.0, 1.0)[..., None]
    out = np.full_like(img_float, n) * alpha + img_float * (1.0 - alpha)
    return out, alpha[..., 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--new", default="60,150,220", help="canonical water RGB (all tiles -> this)")
    ap.add_argument("--target", default="auto", help="'auto' per-tile detect, or R,G,B")
    ap.add_argument("--softness", type=float, default=60.0, help="20-100; low=hard edges, high=tints structures")
    ap.add_argument("--preview", default="", help="optional before/after grid path")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    new = parse_rgb(a.new)
    files = sorted(glob.glob(os.path.join(a.in_dir, "*.png")))
    print(f"water-fix {len(files)} tiles -> canonical {new}, softness {a.softness}")
    prev_rows = []
    for f in files:
        im = Image.open(f).convert("RGB")
        rgb = np.asarray(im)
        imgf = rgb.astype(np.float32) / 255.0
        if a.target == "auto":
            tgt, _ = auto_water_target(rgb)
            if tgt is None:
                Image.fromarray(rgb).save(os.path.join(a.out_dir, os.path.basename(f)))
                print(f"  {os.path.basename(f)}: no water, copied")
                continue
        else:
            tgt = parse_rgb(a.target)
        out, _ = soft_color_replace(imgf, tgt, new, a.softness)
        res = (out * 255).astype(np.uint8)
        Image.fromarray(res).save(os.path.join(a.out_dir, os.path.basename(f)))
        print(f"  {os.path.basename(f)}: target {tuple(tgt)} -> {new}")
        if a.preview and len(prev_rows) < 6:
            prev_rows.append(np.concatenate([rgb, res], axis=1))
    if a.preview and prev_rows:
        h = min(r.shape[0] for r in prev_rows)
        grid = np.concatenate([r[:h] for r in prev_rows], axis=0)
        Image.fromarray(grid).save(a.preview)
        print(f"preview -> {a.preview}")
    print(f"done -> {a.out_dir}")

if __name__ == "__main__":
    main()
