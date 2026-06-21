#!/usr/bin/env python3
"""water_tool — micro-tool for the pathological water case (cannoneyed 014/018 method).

The fine-tuned model can't reliably render water, so DON'T trust the styled output's
water. Instead: classify the TRUE water extent from the RAW aerial (real water = one
big SMOOTH low-texture region), then flood that extent in the styled image with ONE
canonical muddy color (soft-feathered so banks/bridges stay). v3-clean water on a
seamless v4 base.

Usage:
  python water_tool.py --styled walk.png --raw-dir dataset/corridor_raw \
      --cols 17 --rows 26 --out fixed.png [--muddy 104,111,120] [--preview p.jpg]
"""
import argparse, glob, os
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter, label, binary_closing, binary_dilation
Image.MAX_IMAGE_PIXELS = None


def assemble_raw(d, ncols, nrows, tile):
    canvas = np.zeros((nrows*tile, ncols*tile, 3), np.uint8)
    for c in range(ncols):
        for r in range(nrows):
            f = f"{d}/tile_{c}_{r}.png"
            if os.path.exists(f):
                canvas[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = np.asarray(
                    Image.open(f).convert("RGB").resize((tile, tile)))
    return canvas


def water_mask_from_raw(raw, min_frac=0.003):
    """real water = large, smooth, not-green, not-bright connected region(s)."""
    g = raw.mean(2).astype(np.float32)
    m = uniform_filter(g, 25); std = np.sqrt(np.maximum(uniform_filter(g*g, 25)-m*m, 0))
    R, G, B = raw[..., 0].astype(np.int16), raw[..., 1].astype(np.int16), raw[..., 2].astype(np.int16)
    cand = (std < 13) & (~((G-R > 10) & (G-B > 10))) & (g < 165) & (g > 30)
    lab, n = label(cand); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep = np.where(sizes > raw.shape[0]*raw.shape[1]*min_frac)[0]
    mask = np.isin(lab, keep)
    mask = binary_closing(mask, iterations=4)
    mask = binary_dilation(mask, iterations=1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styled", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--muddy", default="104,111,120")
    ap.add_argument("--tile", type=int, default=512, help="raw assembly tile px (lower=faster)")
    ap.add_argument("--feather", type=int, default=4)
    ap.add_argument("--preview", default="")
    a = ap.parse_args()

    styled = Image.open(a.styled).convert("RGB")
    W, H = styled.size
    raw = assemble_raw(a.raw_dir, a.cols, a.rows, a.tile)
    mask = water_mask_from_raw(raw)
    print(f"raw water extent: {100*mask.mean():.1f}% of {a.cols}x{a.rows}")
    # upscale mask to styled res
    mimg = Image.fromarray((mask*255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    fm = np.asarray(mimg.filter(ImageFilter.GaussianBlur(a.feather)), np.float32)/255.0
    muddy = np.array([int(x) for x in a.muddy.split(",")], np.float32)
    st = np.asarray(styled, np.float32)
    # PROTECT STRUCTURES: bridges/barges sit ON the water and are TEXTURED in the styled
    # image -> detect that texture and zero the flood there so we never paint over them.
    sg = st.mean(2)
    sm = uniform_filter(sg, 7); sstd = np.sqrt(np.maximum(uniform_filter(sg*sg, 7)-sm*sm, 0))
    structure = sstd > 11                      # bridge trusses/decks/barges = high local texture
    structure = binary_dilation(structure, iterations=2)
    fm = fm * (1.0 - structure.astype(np.float32))   # carve structures out of the flood
    print(f"protected {100*structure.mean():.1f}% as structures (bridges/barges)")
    # subtle low-freq variation so water isn't dead-flat
    yy = np.linspace(0, 6.28, H)[:, None]; xx = np.linspace(0, 6.28, W)[None, :]
    varn = (np.sin(yy*1.7)+np.cos(xx*1.3))*1.2
    water_col = muddy[None, None, :] + np.stack([varn]*3, -1)
    out = st*(1-fm[..., None]) + water_col*fm[..., None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out)
    print(f"flooded water -> {a.out}")
    if a.preview:
        res = Image.open(a.out)
        bb = res.crop((0, int(H*0.18), W, int(H*0.45)))
        bb.resize((1400, int(1400*bb.size[1]/bb.size[0]))).save(a.preview, quality=92)
        print(f"preview -> {a.preview}")


if __name__ == "__main__":
    main()
