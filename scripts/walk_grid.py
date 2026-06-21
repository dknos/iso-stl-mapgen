#!/usr/bin/env python3
"""Full row-major OMNI quadrant walk over an NxN grid of non-overlapping 1024px
tiles -> one seamless map. Each tile = 2x2 of 512px quadrants; the quadrant grid
is contiguous. Fill modes (all use the (237,28,36) red box the LoRA trained on):
  - seed tile_0_0 full
  - TOP ROW tiles (r=0,c>0): horizontal-half fill (left tile = context)
  - LEFT COL tiles (c=0,r>0): vertical-half fill (top tile = context)
  - INTERIOR tiles: 4 quadrant bottom-right L-fills (top+left+topleft = context)
Walk anchors every quad to the seed via neighbors -> no per-tile drift, no seams.
"""
import argparse, base64, io, json, os, time, urllib.request
from PIL import Image, ImageDraw

RED = (237, 28, 36)
TRIGGER = "<isometric st louis pixel art>"
P_INFILL = ("Fill in the red-outlined region with the missing pixels in the " + TRIGGER +
            " isometric 3D city diorama style, removing the red border and exactly following the color, "
            "shape, and structure of the surrounding already-generated image so the result is seamless.")
P_FULL = ("Convert this aerial render into the " + TRIGGER + " isometric 3D city diorama style, clean "
          "low-poly model, keeping the exact same layout and geometry.")
NEG = "blurry, hazy, photograph, text, watermark, red border, seam"
Q = 512

def call(ep, img, prompt, steps=14, guidance=3.0, true_cfg=1.0, seed=42):  # CFG OFF (no neg prompt) = half compute, same look
    b = io.BytesIO(); img.save(b, "PNG")
    body = json.dumps({"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": prompt,
                       "steps": steps, "guidance_scale": guidance,
                       "true_cfg_scale": true_cfg, "seed": seed}).encode()
    for att in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                ep, data=body, headers={"Content-Type": "application/json"}), timeout=300))
            return Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        except Exception as e:
            print(f"  retry {att}: {str(e)[:60]}"); time.sleep(5)
    raise SystemExit("call failed")

def redbox(win, box):
    d = ImageDraw.Draw(win)
    for i in range(2): d.rectangle([box[0]+i, box[1]+i, box[2]-1-i, box[3]-1-i], outline=RED)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tiles", default=os.path.expanduser("~/stlradar-3d/tools/iso-stl/out/city_test"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    N = a.n
    # render quads rq[(qx,qy)] = 512 render
    rq = {}
    for r in range(N):
        for c in range(N):
            t = Image.open(os.path.join(a.tiles, f"tile_{c}_{r}.png")).convert("RGB").resize((1024, 1024))
            for dy in range(2):
                for dx in range(2):
                    rq[(c*2+dx, r*2+dy)] = t.crop((dx*Q, dy*Q, dx*Q+Q, dy*Q+Q))
    gq = {}; t0 = time.time(); calls = [0]
    def C(*args, **kw):
        calls[0] += 1
        kw.setdefault("seed", 100003 + calls[0] * 7919)  # UNIQUE seed/call -> no repeat-collapse
        print(f"  call {calls[0]} ({time.time()-t0:.0f}s)"); return call(*args, **kw)

    # seed tile_0_0 full
    print("seed tile_0_0...")
    seed = Image.new("RGB", (1024, 1024))
    for dy in range(2):
        for dx in range(2): seed.paste(rq[(dx, dy)], (dx*Q, dy*Q))
    g = C(a.endpoint, seed, P_FULL)
    for dy in range(2):
        for dx in range(2): gq[(dx, dy)] = g.crop((dx*Q, dy*Q, dx*Q+Q, dy*Q+Q))

    def fill_half(ctx_quads, new_quads, axis):
        """axis='h': ctx=left col (2 quads stacked), new=right col -> fill right.
           axis='v': ctx=top row, new=bottom row -> fill bottom."""
        win = Image.new("RGB", (1024, 1024))
        if axis == 'h':
            win.paste(gq[ctx_quads[0]], (0, 0)); win.paste(gq[ctx_quads[1]], (0, Q))
            win.paste(rq[new_quads[0]], (Q, 0)); win.paste(rq[new_quads[1]], (Q, Q))
            redbox(win, (Q, 0, 1024, 1024)); out = C(a.endpoint, win, P_INFILL)
            gq[new_quads[0]] = out.crop((Q, 0, 1024, Q)); gq[new_quads[1]] = out.crop((Q, Q, 1024, 1024))
        else:
            win.paste(gq[ctx_quads[0]], (0, 0)); win.paste(gq[ctx_quads[1]], (Q, 0))
            win.paste(rq[new_quads[0]], (0, Q)); win.paste(rq[new_quads[1]], (Q, Q))
            redbox(win, (0, Q, 1024, 1024)); out = C(a.endpoint, win, P_INFILL)
            gq[new_quads[0]] = out.crop((0, Q, Q, 1024)); gq[new_quads[1]] = out.crop((Q, Q, 1024, 1024))

    def fill_br(qx, qy):
        """bottom-right L-fill: TL,TR,BL generated context, BR new."""
        win = Image.new("RGB", (1024, 1024))
        win.paste(gq[(qx-1, qy-1)], (0, 0)); win.paste(gq[(qx, qy-1)], (Q, 0))
        win.paste(gq[(qx-1, qy)], (0, Q)); win.paste(rq[(qx, qy)], (Q, Q))
        redbox(win, (Q, Q, 1024, 1024)); out = C(a.endpoint, win, P_INFILL)
        gq[(qx, qy)] = out.crop((Q, Q, 1024, 1024))

    # TOP ROW tiles (r=0, c>0): horizontal-half fill, left tile = context
    for c in range(1, N):
        qx = c*2
        fill_half([(qx-1, 0), (qx-1, 1)], [(qx, 0), (qx, 1)], 'h')   # left half of tile
        fill_half([(qx, 0), (qx, 1)], [(qx+1, 0), (qx+1, 1)], 'h')   # right half
    # LEFT COL tiles (c=0, r>0): vertical-half fill, top tile = context
    for r in range(1, N):
        qy = r*2
        fill_half([(0, qy-1), (1, qy-1)], [(0, qy), (1, qy)], 'v')   # top half of tile
        fill_half([(0, qy), (1, qy)], [(0, qy+1), (1, qy+1)], 'v')   # bottom half
    # INTERIOR tiles (c>0,r>0): 4 quadrant BR fills in order
    for r in range(1, N):
        for c in range(1, N):
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                fill_br(c*2+dx, r*2+dy)

    # assemble map
    QN = N*2; mp = Image.new("RGB", (QN*Q, QN*Q))
    for (qx, qy), im in gq.items(): mp.paste(im, (qx*Q, qy*Q))
    mp.save(a.out); mp.resize((1024, 1024)).save(a.out.replace(".png", "_prev.png"))
    print(f"walk {N}x{N} done: {calls[0]} calls, {(time.time()-t0)/60:.1f} min -> {a.out}")

if __name__ == "__main__":
    main()
