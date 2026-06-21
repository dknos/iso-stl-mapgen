#!/usr/bin/env python3
"""Geometric BLUE river flood (CORRECT OSM multipolygon) + v3 video-game grade.

Root-cause fix: the Mississippi is an OSM *relation* (multipolygon). Its member ways are
boundary ARCS, not closed loops. Drawing each arc as its own polygon (ImageDraw closes it
with a straight chord) made garbage triangles over downtown AND never filled the real channel
(arcs aren't a filled area). Fix = chain outer member ways into proper closed rings by shared
endpoints, then fill. Result: true river filled edge-to-edge, no downtown blob.

Then flood v3-blue + sine ripple, fill_holes (any sandbar inside the channel -> blue), carve
thin bridges back so they show. Grade medium for the v3 video-game pop. Saves flood_mask.png."""
import json, urllib.request, urllib.parse, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from scipy.ndimage import uniform_filter, binary_dilation, binary_opening, binary_fill_holes, binary_closing
Image.MAX_IMAGE_PIXELS = None
H_="/home/nemoclaw/iso-stl-lora/"
M = json.load(open(H_+"out/bounds_model.json"))
A = np.array(M["A"]); CLAT, CLNG = M["center"]; mLat, mLng = M["mLat"], M["mLng"]
COL0, ROW0, TILE, S = M["col0"], M["row0"], 1024, 0.5
BLUE = np.array([88,139,179.], np.float32)

def geo_to_canvas(lat, lng):
    dE=(lng-CLNG)*mLng; dN=(lat-CLAT)*mLat
    cr=np.array([dE,dN,1.0])@A
    return ((cr[0]-COL0)*TILE*S,(cr[1]-ROW0)*TILE*S)

def fetch():
    cache=H_+"out/osm_water_cache.json"
    if os.path.exists(cache):
        d=json.load(open(cache)); print(f"OSM from cache ({len(d['elements'])} elements)")
    else:
        bbox=(38.56,-90.33,38.70,-90.12)
        q=(f'[out:json][timeout:90];('
           f'way["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
           f'relation["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););out geom;')
        mirrors=["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://lz4.overpass-api.de/api/interpreter"]
        d=None
        for m in mirrors:
            for att in range(2):
                try:
                    req=urllib.request.Request(m,data=urllib.parse.urlencode({"data":q}).encode(),
                                               headers={"User-Agent":"stl-iso-map/1.0"})
                    d=json.load(urllib.request.urlopen(req,timeout=150)); break
                except Exception as e:
                    print(f"  fetch fail {m.split('//')[1][:20]} att{att}: {str(e)[:40]}"); time.sleep(6)
            if d: break
        if not d: raise SystemExit("overpass unreachable on all mirrors")
        json.dump(d,open(cache,"w")); print(f"OSM fetched + cached ({len(d['elements'])} elements)")
    ways=[]; rel_ways=[]
    for el in d["elements"]:
        if el["type"]=="way" and "geometry" in el:
            ways.append([(p["lat"],p["lon"]) for p in el["geometry"]])
        elif el["type"]=="relation" and "members" in el:
            for mem in el["members"]:
                if mem.get("role") in ("outer","") and "geometry" in mem:
                    rel_ways.append([(p["lat"],p["lon"]) for p in mem["geometry"]])
    print(f"OSM: {len(ways)} closed ways, {len(rel_ways)} relation member-arcs"); return ways, rel_ways

def assemble_rings(arcs, tol=1e-7):
    """Chain boundary arcs into closed rings by matching shared endpoints (OSM multipolygon)."""
    arcs=[list(a) for a in arcs if len(a)>=2]
    rings=[]
    def close(p,q): return abs(p[0]-q[0])<tol and abs(p[1]-q[1])<tol
    while arcs:
        ring=arcs.pop(0); grew=True
        while grew and not close(ring[0],ring[-1]):
            grew=False
            for i,w in enumerate(arcs):
                if close(w[0],ring[-1]):   ring+=w[1:];        arcs.pop(i); grew=True; break
                if close(w[-1],ring[-1]):  ring+=w[-2::-1];    arcs.pop(i); grew=True; break
                if close(w[-1],ring[0]):   ring=w[:-1]+ring;   arcs.pop(i); grew=True; break
                if close(w[0],ring[0]):    ring=w[::-1][:-1]+ring; arcs.pop(i); grew=True; break
        if close(ring[0],ring[-1]) and len(ring)>=4:
            rings.append(ring)                       # only CLOSED rings -> no garbage chords
        else:
            print(f"  dropped unclosed ring ({len(ring)} pts)")
    return rings

def assemble_raw_canvas(W,H):
    T=int(TILE*S); canvas=np.zeros((H,W,3),np.uint8)
    def place(d,nc,nr,ox,oy):
        for c in range(nc):
            for r in range(nr):
                f=f"{d}/tile_{c}_{r}.png"
                if os.path.exists(f):
                    t=np.asarray(Image.open(f).convert("RGB").resize((T,T)))
                    y0,x0=oy+r*T,ox+c*T
                    if y0+T<=H and x0+T<=W: canvas[y0:y0+T,x0:x0+T]=t
    place(H_+"dataset/corridor_raw",17,26,int(6*TILE*S),0)
    place(H_+"dataset/downtown_raw",10,10,0,int(7*TILE*S))
    return canvas

def main():
    styled=Image.open(H_+"out/stitched_deploy.png").convert("RGB"); W,Hh=styled.size
    ways, rel_ways=fetch()
    mimg=Image.new("L",(W,Hh),0); dr=ImageDraw.Draw(mimg)
    for ring in assemble_rings(rel_ways):     # RIVERS ONLY (Mississippi/Missouri relations) -> NO city ponds
        pts=[geo_to_canvas(la,ln) for (la,ln) in ring]
        if len(pts)>=3: dr.polygon(pts,fill=255)
    print(f"rivers-only: {len(ways)} lake/pond ways DROPPED (kills 'water in the city')")
    mask=np.asarray(mimg)>128
    print(f"river mask: {100*mask.mean():.1f}% of canvas")
    river=binary_fill_holes(mask)                                  # enclosed islands -> water
    river=binary_closing(river,iterations=int(os.environ.get("CLOSE","22")))  # swallow mid-channel bars/islands
    # PRESERVE bright structures (bridges / arch / riverfront buildings); flood only DARK pixels
    # (muddy water + tan sandbars + green tree-islands are all darker than bridge decks / arch / buildings).
    sc=np.asarray(styled).astype(np.int16)
    bright=sc.mean(2)
    STH=float(os.environ.get("STRUCT","165"))
    bs=bright>STH
    thinstruct=bs&~binary_opening(bs,iterations=6)   # bridges/arch = THIN bright; large bright blobs
    flood=river&~thinstruct                          # (glint, tile-seam brightening) get flooded -> uniform water
    print(f"flood={100*flood.mean():.2f}%; thin structures (bridges) kept={100*thinstruct.mean():.3f}%")
    Image.fromarray((flood*255).astype(np.uint8)).save(H_+"out/flood_mask.png")
    # apply blue flood + sine ripple
    st=np.asarray(styled,np.float32)
    fm=np.asarray(Image.fromarray((flood*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),np.float32)/255.0
    yy=np.linspace(0,6.28,Hh)[:,None]; xx=np.linspace(0,6.28,W)[None,:]
    varn=(np.sin(yy*1.7)+np.cos(xx*1.3))*2.0
    out=np.empty((Hh,W,3),np.uint8)
    for c in range(3):
        out[...,c]=np.clip(st[...,c]*(1-fm)+(BLUE[c]+varn)*fm,0,255).astype(np.uint8)
    del st
    img=Image.fromarray(out); del out
    img.save(H_+"out/stitched_blueriver_geo.png")
    g=ImageEnhance.Color(img).enhance(1.30); g=ImageEnhance.Brightness(g).enhance(1.06); g=ImageEnhance.Contrast(g).enhance(1.05)
    g.save(H_+"out/stitched_final.png")
    print("saved stitched_final.png (correct multipolygon flood + grade)")
    g.crop((int(W*0.20),int(Hh*0.34),int(W*0.46),int(Hh*0.50))).save("/mnt/c/Users/rneeb/iso_FIX_downtown.jpg",quality=92)
    g.crop((int(W*0.06),int(Hh*0.28),int(W*0.30),int(Hh*0.44))).save("/mnt/c/Users/rneeb/iso_FIX_river.jpg",quality=92)
    g.crop((int(W*0.30),int(Hh*0.18),int(W*0.62),int(Hh*0.36))).save("/mnt/c/Users/rneeb/iso_FIX_riverE.jpg",quality=92)
    g.resize((W//7,Hh//7)).save("/mnt/c/Users/rneeb/iso_FIX_fullmap.jpg",quality=86)
    print("-> iso_FIX_downtown / iso_FIX_river / iso_FIX_riverE / iso_FIX_fullmap .jpg")

if __name__=="__main__":
    main()
