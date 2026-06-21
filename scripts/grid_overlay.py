#!/usr/bin/env python3
"""Overlay a labeled grid on the corridor map (stitched_deploy_river = corridor.html map) so the
user names which cells to replace. Cell = 1024px (one nano window). Label = COL_ROW. Each cell ->
canvas box (col*1024, row*1024). Outputs full-res grid PNG + a readable preview."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS=None
H_="/home/nemoclaw/iso-stl-lora/"; CELL=1024
im=Image.open(H_+"out/stitched_deploy_river.png").convert("RGB"); W,H=im.size
d=ImageDraw.Draw(im)
try: font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",110)
except: font=ImageFont.load_default()
cols=(W+CELL-1)//CELL; rows=(H+CELL-1)//CELL
for c in range(cols):
    for r in range(rows):
        x,y=c*CELL,r*CELL
        d.rectangle([x,y,min(x+CELL,W)-1,min(y+CELL,H)-1],outline=(255,255,0),width=5)
        lab=f"{c}_{r}"
        d.rectangle([x+6,y+6,x+6+len(lab)*62,y+128],fill=(0,0,0))
        d.text((x+12,y+8),lab,fill=(255,255,0),font=font)
im.save(H_+"out/grid_overlay.png")
im.resize((W//4,H//4)).save("/mnt/c/Users/rneeb/iso_GRID.jpg",quality=90)
im.crop((0,3500,5500,6800)).save("/mnt/c/Users/rneeb/iso_GRID_downtown.jpg",quality=92)
print(f"grid {cols}x{rows} cells (1024px) -> grid_overlay.png + iso_GRID.jpg / iso_GRID_downtown.jpg")
