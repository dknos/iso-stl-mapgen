#!/usr/bin/env python3
"""Replace specific grid cells with nano-banana. WATER STAYS ACCURATE (muddy, matching the real
river) -- only remove hallucinations (sandbars/islands/sand-under-bridge/fields-as-water). After
each cell, RESTORE the base's bright-neutral structures (Gateway Arch, bridges, buildings) so they
are never deleted (warm tan sandbars are NOT restored). Context window + feather for seamless edges.
Base = stitched_deploy_river (corridor map). Paced (429-safe)."""
import os, sys, tempfile, time
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from vertex_nano import edit
Image.MAX_IMAGE_PIXELS=None
H_="/home/nemoclaw/iso-stl-lora/"; CELL=1024; PAD=220; FEATH=64
P_RIVER=("Edit this isometric low-poly toy-diorama city map. Remove the sandbars, mudflats, tan/brown "
         "bars, exposed-dirt patches and tree-islands sitting in the river, and any sand under the "
         "bridges, replacing them with realistic river water that MATCHES the color of the surrounding "
         "river water (keep the water its natural muddy color, do NOT make it bright blue). Keep the "
         "bridges, the tall silver Gateway Arch monument, the barges, the buildings and all land "
         "EXACTLY as they are. Same isometric style, same layout. No text, no labels, no border.")
P_BARGE=("Edit this isometric low-poly toy-diorama city map. Remove the sandbars and mudflats in the "
         "river, replacing them with realistic water that MATCHES the surrounding river water color "
         "(natural muddy color, not bright blue). Replace the barges with a single small towboat on the "
         "water. Keep the docks, cranes, buildings and roads EXACTLY. Same style. No text/labels/border.")
P_FIELD=("Edit this isometric low-poly toy-diorama city map. This area is FARMLAND, NOT water. Render "
         "it as realistic farm fields and grass that MATCH the neighboring farmland (greens and tans). "
         "Remove any incorrect blue/gray water that should be field. Keep roads, railways and buildings "
         "EXACTLY. Same isometric style. No text, no labels, no border.")
CELLS=[(1,5,P_RIVER),(2,5,P_RIVER),(10,4,P_BARGE),(11,4,P_BARGE),(8,3,P_FIELD),(9,3,P_FIELD),(10,3,P_FIELD)]
base=Image.open(H_+"out/stitched_deploy_river.png").convert("RGB"); W,Hh=base.size
basea=np.asarray(base)
out=base.copy()
ND=f"{H_}out/nano_tiles"; os.makedirs(ND,exist_ok=True)
for (c,r,prompt) in CELLS:
    cx,cy=c*CELL,r*CELL
    wx0=max(0,cx-PAD); wy0=max(0,cy-PAD); wx1=min(W,cx+CELL+PAD); wy1=min(Hh,cy+CELL+PAD)
    win=base.crop((wx0,wy0,wx1,wy1)); ww,wh=win.size
    src=tempfile.mktemp(suffix=".png"); win.save(src); dst=f"{ND}/cellv2_{c}_{r}.png"
    try: ok=edit(src,dst,prompt,"global",None); print(f"  {'ok' if ok else 'noimg'} {c}_{r}",flush=True)
    except BaseException as e: print(f"  fail {c}_{r}: {str(e)[:40]}",flush=True); ok=False
    finally:
        try: os.remove(src)
        except: pass
    time.sleep(6)
    if not ok or not os.path.exists(dst): continue
    nano=Image.open(dst).convert("RGB").resize((ww,wh))
    ox,oy=cx-wx0,cy-wy0; cw=min(CELL,W-cx); chh=min(CELL,Hh-cy)
    cell=np.asarray(nano.crop((ox,oy,ox+cw,oy+chh))).copy()
    # restore base bright-NEUTRAL structures (Arch/bridges/buildings) in this cell; NOT warm sandbars
    bc=basea[cy:cy+chh,cx:cx+cw]
    g=bc.mean(2); rb=bc[...,0].astype(int)-bc[...,2].astype(int)
    keep=(g>140)&(np.abs(rb)<24)
    cell[keep]=bc[keep]
    cellim=Image.fromarray(cell)
    mask=Image.new("L",(cw,chh),255); md=ImageDraw.Draw(mask)
    for k in range(FEATH): md.rectangle([k,k,cw-1-k,chh-1-k],outline=int(255*(k+1)/(FEATH+1)))
    out.paste(cellim,(cx,cy),mask)
out.save(H_+"out/stitched_grid.png")
out.crop((600,4300,3400,6200)).save("/mnt/c/Users/rneeb/iso_GR_downtown.jpg",quality=94)
out.crop((9200,3600,11776,5400)).save("/mnt/c/Users/rneeb/iso_GR_barges.jpg",quality=94)
out.crop((7800,2900,11264,4200)).save("/mnt/c/Users/rneeb/iso_GR_fields.jpg",quality=94)
print("-> stitched_grid.png + iso_GR_downtown / iso_GR_barges / iso_GR_fields .jpg")
