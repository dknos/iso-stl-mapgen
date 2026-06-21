#!/usr/bin/env python3
"""Walk the river-band tiles with the v3 LoRA (renders clean blue water natively),
per-tile full conversion. Output -> out/v3_river/tile_{wedgecol}_{wedgerow}.png."""
import json, base64, io, time, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

EP = os.environ.get("EP")  # your Modal endpoint — deploy cloud/modal_omni_server.py, then: export EP=https://<you>--...modal.run
if not EP:
    raise SystemExit("Set EP to your Modal endpoint URL (deploy cloud/modal_omni_server.py with `modal deploy`, then export EP=...).")
TRIGGER = "<isometric st louis pixel art>"
P_FULL = ("Convert this aerial render into the " + TRIGGER + " isometric 3D city diorama style, "
          "clean low-poly model, keeping the exact same layout and geometry.")

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

tiles = json.load(open("/tmp/river_tiles.json"))
os.makedirs("/home/nemoclaw/iso-stl-lora/out/v3_river", exist_ok=True)
done = [0]
def do(i):
    c, r = tiles[i]
    src = f"/home/nemoclaw/iso-stl-lora/dataset/corridor_raw/tile_{c-11}_{r-2}.png"
    if not os.path.exists(src):
        src = f"/home/nemoclaw/iso-stl-lora/dataset/downtown_raw/tile_{c-5}_{r-9}.png"
    if not os.path.exists(src):
        print(f"  no raw for {c},{r}"); return
    raw = Image.open(src).convert("RGB").resize((1024, 1024))
    out = call(raw, 100 + i*7919)
    out.save(f"/home/nemoclaw/iso-stl-lora/out/v3_river/tile_{c}_{r}.png")
    done[0] += 1
    if done[0] % 10 == 0: print(f"  {done[0]}/{len(tiles)}")

with ThreadPoolExecutor(max_workers=12) as ex:
    list(ex.map(do, range(len(tiles))))
print(f"v3 river walk done: {done[0]} tiles")
