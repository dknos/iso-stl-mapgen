#!/usr/bin/env python3
"""Final finish: tight vivid-blue river recolor (kills seams) + fold in-channel islands to blue,
then RESTORE v4's neutral-bright structures (bridges/highways/buildings) so NO water lands on them.
Sandbars (warm/tan) are NOT restored -> they stay flooded blue. v4-brightness is the clean
structure-vs-water discriminator (v4 water is dark, structures bright). Quarter ops kept light."""
import os, sys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import uniform_filter, binary_dilation, binary_erosion, binary_fill_holes, binary_closing, label
Image.MAX_IMAGE_PIXELS=None
H_="/home/nemoclaw/iso-stl-lora/"
UB=np.array([88,139,179.],np.float32)
SRC=sys.argv[1] if len(sys.argv)>1 else (H_+"out/stitched_riverwalk2.png" if os.path.exists(H_+"out/stitched_riverwalk2.png") else H_+"out/stitched_riverwalk.png")
print("src:",SRC)
base=np.asarray(Image.open(SRC).convert("RGB")).copy()
v4=np.asarray(Image.open(H_+"out/stitched_deploy.png").convert("RGB"))
CH,CW=base.shape[:2]
R=base[...,0].astype(np.int16); G=base[...,1].astype(np.int16); B=base[...,2].astype(np.int16)
gf=base.mean(2).astype(np.float32)
std=np.sqrt(np.maximum(uniform_filter(gf*gf,9)-uniform_filter(gf,9)**2,0)).astype(np.float32); del gf
bluewater=(B>R+18)&(B>G)&(B>115)&(std<12)            # VIVID blue water only (not blue-gray roads)
print(f"vivid blue water {100*bluewater.mean():.2f}%")
# flood the WHOLE river region: close hard to bridge fragmented blue across sandbars, then fill holes
recmask=binary_fill_holes(binary_closing(bluewater,iterations=16))
print(f"river region (sandbars filled) {100*recmask.mean():.2f}%")
del bluewater,std
# recolor (water+islands) -> uniform blue + ripple
yy=np.linspace(0,9,CH).astype(np.float32)[:,None]; xx=np.linspace(0,9,CW).astype(np.float32)[None,:]
rip=(np.sin(yy*1.3)+np.cos(xx*1.1))*5.0
fm=np.asarray(Image.fromarray((recmask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)),np.float32)/255.0; del recmask
for k in range(3):
    base[...,k]=np.clip(base[...,k]*(1-fm)+(UB[k]+rip)*fm,0,255).astype(np.uint8)
del fm,rip
# RESTORE v4 neutral-bright structures (bridges/highways/buildings); NOT warm sandbars
v4R=v4[...,0].astype(np.int16); v4B=v4[...,2].astype(np.int16)
restore=(v4.mean(2)>128)&(np.abs(v4R-v4B)<22)
restore=binary_dilation(restore,iterations=1)
print(f"restored v4 structures {100*restore.mean():.2f}%")
base[restore]=v4[restore]
del v4,restore
res=Image.fromarray(base); del base
g2=ImageEnhance.Color(res).enhance(1.14); g2=ImageEnhance.Brightness(g2).enhance(1.03); g2=ImageEnhance.Contrast(g2).enhance(1.03)
g2.save(H_+"out/stitched_final.png")
g2.crop((2100,4350,3500,5150)).save("/mnt/c/Users/rneeb/iso_CF_bridge.jpg",quality=94)
g2.crop((1400,4400,3600,5900)).save("/mnt/c/Users/rneeb/iso_CF_downtown.jpg",quality=93)
g2.resize((CW//7,CH//7)).save("/mnt/c/Users/rneeb/iso_CF_fullmap.jpg",quality=86)
print("-> iso_CF_bridge / iso_CF_downtown / iso_CF_fullmap .jpg")
