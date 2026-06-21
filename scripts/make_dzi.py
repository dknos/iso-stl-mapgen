import os, math, sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SRC = "/home/nemoclaw/iso-stl-lora/out/stitched_deploy_river.png"
NAME = "stl_corridor3"
OUT = "/home/nemoclaw/iso-stl-lora/out/dzi_build"
TILE, OVERLAP, FMT, Q = 256, 0, "jpg", 85

im = Image.open(SRC).convert("RGB")
W, H = im.size
maxlevel = math.ceil(math.log2(max(W, H)))
os.makedirs(f"{OUT}/{NAME}_files", exist_ok=True)
with open(f"{OUT}/{NAME}.dzi", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
            f'Format="{FMT}" Overlap="{OVERLAP}" TileSize="{TILE}">'
            f'<Size Width="{W}" Height="{H}"/></Image>')
total = 0
for level in range(maxlevel + 1):
    scale = 2 ** (maxlevel - level)
    lw, lh = max(1, math.ceil(W/scale)), max(1, math.ceil(H/scale))
    lvl = im.resize((lw, lh), Image.LANCZOS) if scale > 1 else im
    d = f"{OUT}/{NAME}_files/{level}"; os.makedirs(d, exist_ok=True)
    cols, rows = math.ceil(lw/TILE), math.ceil(lh/TILE)
    for r in range(rows):
        for c in range(cols):
            box = (c*TILE, r*TILE, min((c+1)*TILE, lw), min((r+1)*TILE, lh))
            lvl.crop(box).save(f"{d}/{c}_{r}.{FMT}", quality=Q)
            total += 1
    print(f"level {level}: {lw}x{lh} -> {cols}x{rows} tiles")
print(f"DZI {NAME} done: {W}x{H}, {maxlevel+1} levels, {total} tiles -> {OUT}")
