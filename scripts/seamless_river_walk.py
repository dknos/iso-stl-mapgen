#!/usr/bin/env python3
"""Cheap SEAMLESS river-only v3 walk on the v4 map. BFS the 88 river tiles; for each, paste the
RAW geometry into a red box surrounded by the CURRENT working render (v4 + already-done v3-blue
neighbors) and infill with v3 -> renders clean blue water, continuing neighbors -> consistent,
seamless. Only river tiles change; v4 land/roads kept. --apply writes stitched_final.png.
  Set EP to your Modal endpoint (deploy cloud/modal_omni_server.py).  --limit N for a test run."""
import argparse, base64, io, json, os, time, urllib.request, collections
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS=None
H_="/home/nemoclaw/iso-stl-lora/"
EP=os.environ.get("EP")
if not EP: raise SystemExit("Set EP to your Modal endpoint URL (deploy cloud/modal_omni_server.py, then export EP=...).")
RED=(237,28,36); TRIG="<isometric st louis pixel art>"
P=("Fill in the red-outlined region with the missing pixels in the "+TRIG+" isometric 3D city "
   "diorama style, removing the red border and exactly following the color, shape and structure of "
   "the surrounding already-generated image so the result is seamless. Render any river as clean flat blue water.")
NEG="blurry, photograph, text, watermark, red border, seam, sandbar, island, muddy water"
T=512; WIN=1024

def call(img, seed):
    b=io.BytesIO(); img.save(b,"PNG")
    body=json.dumps({"image_b64":base64.b64encode(b.getvalue()).decode(),"prompt":P,
        "negative_prompt":NEG,"steps":16,"guidance_scale":3.0,"true_cfg_scale":2.0,"seed":seed}).encode()
    for att in range(6):
        try:
            r=json.load(urllib.request.urlopen(urllib.request.Request(
                EP,data=body,headers={"Content-Type":"application/json"}),timeout=600))
            return Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        except Exception as e:
            print(f"   retry {att}: {str(e)[:50]}"); time.sleep(8)
    raise SystemExit("call failed")

def bfs_order(tiles):
    rem=set(tiles); order=[]
    nb=lambda c,r:[(c+1,r),(c-1,r),(c,r+1),(c,r-1)]
    while rem:
        s=min(rem); q=collections.deque([s]); seen={s}; order.append(s)
        while q:
            cur=q.popleft()
            for n in nb(*cur):
                if n in rem and n not in seen: seen.add(n); order.append(n); q.append(n)
        rem-=seen
    return order

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--apply",action="store_true")
    ap.add_argument("--start",default=""); ap.add_argument("--skipblue",action="store_true")
    ap.add_argument("--out",default=""); a=ap.parse_args()
    tiles=set(tuple(x) for x in json.load(open(H_+"river_full.json")))
    working=Image.open(a.start or H_+"out/stitched_deploy.png").convert("RGB")
    raw=Image.open(H_+"out/raw_canvas.png").convert("RGB"); W,H=working.size
    order=bfs_order(tiles)
    if a.limit: order=order[:a.limit]
    t0=time.time()
    walked=0
    for i,(c,r) in enumerate(order):
        x0,y0=(c-5)*T,(r-2)*T
        if x0<0 or y0<0 or x0+T>W or y0+T>H: continue
        if a.skipblue:
            tc=np.asarray(working.crop((x0,y0,x0+T,y0+T)),np.int32)
            bl=((tc[...,2]>tc[...,0]+6)&(tc[...,2]>tc[...,1])&(tc[...,2]>110)).mean()
            if bl>0.35: continue        # already blue -> keep, don't re-render
        walked+=1
        cx,cy=x0+T//2,y0+T//2
        wx0=max(0,min(W-WIN,cx-WIN//2)); wy0=max(0,min(H-WIN,cy-WIN//2))
        win=working.crop((wx0,wy0,wx0+WIN,wy0+WIN)).copy()
        bx0,by0=x0-wx0,y0-wy0; bx1,by1=bx0+T,by0+T
        win.paste(raw.crop((x0,y0,x0+T,y0+T)),(bx0,by0))
        d=ImageDraw.Draw(win)
        for k in range(2): d.rectangle([bx0+k,by0+k,bx1-1-k,by1-1-k],outline=RED)
        out=call(win,100003+i*7919)
        if out.size!=(WIN,WIN): out=out.resize((WIN,WIN))
        patch=out.crop((bx0,by0,bx1,by1))
        mask=Image.new("L",(T,T),255); md=ImageDraw.Draw(mask)
        for k in range(20): md.rectangle([k,k,T-1-k,T-1-k],outline=int(255*(k+1)/21))
        working.paste(patch,(x0,y0),mask)
        if (i+1)%10==0: print(f"  {i+1}/{len(order)} ({time.time()-t0:.0f}s)")
    dst=H_+"out/stitched_final.png" if a.apply else (a.out or H_+"out/stitched_riverwalk.png")
    working.save(dst)
    print(f"walked {walked} tiles (skipped {len(order)-walked} already-blue)")
    working.crop((1400,4400,3600,5900)).save("/mnt/c/Users/rneeb/iso_RW_downtown.jpg",quality=93)
    working.crop((3600,3900,5400,5100)).save("/mnt/c/Users/rneeb/iso_RW_edge.jpg",quality=93)
    working.resize((W//7,H//7)).save("/mnt/c/Users/rneeb/iso_RW_fullmap.jpg",quality=86)
    print(f"river walk done: {len(order)} tiles, {(time.time()-t0)/60:.1f} min -> {dst}")

if __name__=="__main__": main()
