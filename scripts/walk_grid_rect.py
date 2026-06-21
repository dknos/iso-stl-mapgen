#!/usr/bin/env python3
"""Rectangular (NC x NR) parallel wavefront walk — same seamless infill DAG as
walk_grid_parallel.py, but width != height so we can walk a corridor sub-rectangle
of a bigger captured grid instead of a square. Tiles must be named tile_{c}_{r}.png
for c in 0..NC-1, r in 0..NR-1 (extract+rename the sub-block first).

Identical anchoring (each tile generated after its top/left/top-left neighbours);
cost = total calls (~4 per interior tile). CFG-off by default (true_cfg=1.0).
"""
import argparse, base64, io, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image, ImageDraw

RED = (237, 28, 36)
TRIGGER = "<isometric st louis pixel art>"
P_INFILL = ("Fill in the red-outlined region with the missing pixels in the " + TRIGGER +
            " isometric 3D city diorama style, removing the red border and exactly following the color, "
            "shape, and structure of the surrounding already-generated image so the result is seamless.")
P_FULL = ("Convert this aerial render into the " + TRIGGER + " isometric 3D city diorama style, clean "
          "low-poly model, keeping the exact same layout and geometry.")
Q = 512


def call(ep, img, prompt, seed, steps=14, guidance=3.0, true_cfg=1.0):  # CFG OFF = half compute, same look
    b = io.BytesIO(); img.save(b, "PNG")
    body = json.dumps({"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": prompt,
                       "steps": steps, "guidance_scale": guidance,
                       "true_cfg_scale": true_cfg, "seed": seed}).encode()
    for att in range(8):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                ep, data=body, headers={"Content-Type": "application/json"}), timeout=600))
            return Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        except Exception as e:
            print(f"    call retry {att}: {str(e)[:50]}"); time.sleep(8)
    raise SystemExit("call failed after retries")


def redbox(win, box):
    d = ImageDraw.Draw(win)
    for i in range(2): d.rectangle([box[0]+i, box[1]+i, box[2]-1-i, box[3]-1-i], outline=RED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    NC, NR = a.cols, a.rows

    rq = {}
    for r in range(NR):
        for c in range(NC):
            t = Image.open(os.path.join(a.tiles, f"tile_{c}_{r}.png")).convert("RGB").resize((1024, 1024))
            for dy in range(2):
                for dx in range(2):
                    rq[(c*2+dx, r*2+dy)] = t.crop((dx*Q, dy*Q, dx*Q+Q, dy*Q+Q))

    gq = {}
    calls = []
    def add(kind, deps, prod, **extra):
        calls.append({"kind": kind, "deps": set(deps), "prod": list(prod), **extra})

    add("seed", [], [(0,0),(1,0),(0,1),(1,1)])
    for c in range(1, NC):
        qx = c*2
        add("half_h", [(qx-1,0),(qx-1,1)], [(qx,0),(qx,1)],   ctx=[(qx-1,0),(qx-1,1)], new=[(qx,0),(qx,1)])
        add("half_h", [(qx,0),(qx,1)],     [(qx+1,0),(qx+1,1)], ctx=[(qx,0),(qx,1)],   new=[(qx+1,0),(qx+1,1)])
    for r in range(1, NR):
        qy = r*2
        add("half_v", [(0,qy-1),(1,qy-1)], [(0,qy),(1,qy)],     ctx=[(0,qy-1),(1,qy-1)], new=[(0,qy),(1,qy)])
        add("half_v", [(0,qy),(1,qy)],     [(0,qy+1),(1,qy+1)], ctx=[(0,qy),(1,qy)],     new=[(0,qy+1),(1,qy+1)])
    for r in range(1, NR):
        for c in range(1, NC):
            for (dx, dy) in [(0,0),(1,0),(0,1),(1,1)]:
                qx, qy = c*2+dx, r*2+dy
                add("br", [(qx-1,qy-1),(qx,qy-1),(qx-1,qy)], [(qx,qy)], q=(qx,qy))

    producer = {}
    for i, cl in enumerate(calls):
        for q in cl["prod"]: producer[q] = i
    deps_calls = [set() for _ in calls]
    dependents = [set() for _ in calls]
    for i, cl in enumerate(calls):
        for q in cl["deps"]:
            deps_calls[i].add(producer[q])
    for i in range(len(calls)):
        for j in deps_calls[i]:
            dependents[j].add(i)
    indeg = [len(deps_calls[i]) for i in range(len(calls))]

    def build(cl):
        win = Image.new("RGB", (1024, 1024))
        if cl["kind"] == "seed":
            for dy in range(2):
                for dx in range(2): win.paste(rq[(dx,dy)], (dx*Q, dy*Q))
            return win, None, P_FULL
        if cl["kind"] == "half_h":
            (c0, c1), (n0, n1) = cl["ctx"], cl["new"]
            win.paste(gq[c0], (0,0)); win.paste(gq[c1], (0,Q))
            win.paste(rq[n0], (Q,0)); win.paste(rq[n1], (Q,Q))
            redbox(win, (Q,0,1024,1024)); return win, [(n0,(Q,0,1024,Q)),(n1,(Q,Q,1024,1024))], P_INFILL
        if cl["kind"] == "half_v":
            (c0, c1), (n0, n1) = cl["ctx"], cl["new"]
            win.paste(gq[c0], (0,0)); win.paste(gq[c1], (Q,0))
            win.paste(rq[n0], (0,Q)); win.paste(rq[n1], (Q,Q))
            redbox(win, (0,Q,1024,1024)); return win, [(n0,(0,Q,Q,1024)),(n1,(Q,Q,1024,1024))], P_INFILL
        qx, qy = cl["q"]
        win.paste(gq[(qx-1,qy-1)], (0,0)); win.paste(gq[(qx,qy-1)], (Q,0))
        win.paste(gq[(qx-1,qy)], (0,Q)); win.paste(rq[(qx,qy)], (Q,Q))
        redbox(win, (Q,Q,1024,1024)); return win, [((qx,qy),(Q,Q,1024,1024))], P_INFILL

    lock = threading.Lock()
    finished = [0]; total = len(calls); t0 = time.time()

    def run_call(i):
        cl = calls[i]
        seed = 100003 + i * 7919
        win, crops, prompt = build(cl)
        out = call(a.endpoint, win, prompt, seed)
        with lock:
            if crops is None:
                for dy in range(2):
                    for dx in range(2): gq[(dx,dy)] = out.crop((dx*Q, dy*Q, dx*Q+Q, dy*Q+Q))
            else:
                for q, box in crops: gq[q] = out.crop(box)
        return i

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_call, i): i for i in range(len(calls)) if indeg[i] == 0}
        while futs:
            done_set, _ = wait(list(futs), return_when=FIRST_COMPLETED)
            for fut in done_set:
                i = futs.pop(fut); fut.result()
                finished[0] += 1
                if finished[0] % 20 == 0 or finished[0] == total:
                    print(f"  {finished[0]}/{total} calls ({time.time()-t0:.0f}s, {a.workers}-way)")
                for j in dependents[i]:
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        futs[ex.submit(run_call, j)] = j

    QNx, QNy = NC*2, NR*2
    mp = Image.new("RGB", (QNx*Q, QNy*Q))
    for (qx, qy), im in gq.items(): mp.paste(im, (qx*Q, qy*Q))
    mp.save(a.out); mp.resize((min(2048, QNx*Q//6), min(2048, QNy*Q//6))).save(a.out.replace(".png", "_prev.png"))
    print(f"rect walk {NC}x{NR}: {total} calls in {(time.time()-t0)/60:.1f} min ({a.workers}-way) -> {a.out}")


if __name__ == "__main__":
    main()
