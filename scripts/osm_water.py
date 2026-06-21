#!/usr/bin/env python3
"""osm_water — geometric water mask from REAL OSM river polygons (the durable water fix).

Image-based water detection can't separate hallucinated forest-islands/sandbars from
real vegetation (the cannoneyed water/trees pathology). So instead: pull the actual
Mississippi water polygons from OSM, project them onto the tiles with the CALIBRATED
bounds_model, rasterize -> a mask of where water REALLY is. Flood that extent (kills
every hallucination inside the real river), minus raw-textured bridges (preserved).
"""
import json, urllib.request, urllib.parse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter, binary_dilation, binary_opening
import os
Image.MAX_IMAGE_PIXELS = None

M = json.load(open("/home/nemoclaw/iso-stl-lora/out/bounds_model.json"))
A = np.array(M["A"]); CLAT, CLNG = M["center"]; mLat, mLng = M["mLat"], M["mLng"]
COL0, ROW0, TILE, S = M["col0"], M["row0"], 1024, 0.5

def geo_to_canvas(lat, lng):
    dE = (lng-CLNG)*mLng; dN = (lat-CLAT)*mLat
    cr = np.array([dE, dN, 1.0]) @ A          # [col, row]
    return ((cr[0]-COL0)*TILE*S, (cr[1]-ROW0)*TILE*S)

def fetch_water(bbox):
    # bbox = (s,w,n,e). natural=water (incl. the Mississippi riverbank multipolygons)
    q = (f'[out:json][timeout:60];('
         f'way["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
         f'relation["natural"="water"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});'
         f');out geom;')
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "stl-iso-map/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=90))
    polys = []
    for el in d["elements"]:
        if el["type"] == "way" and "geometry" in el:
            polys.append([(p["lat"], p["lon"]) for p in el["geometry"]])
        elif el["type"] == "relation" and "members" in el:
            for mem in el["members"]:
                if mem.get("role") in ("outer", "") and "geometry" in mem:
                    polys.append([(p["lat"], p["lon"]) for p in mem["geometry"]])
    print(f"OSM water: {len(polys)} polygons")
    return polys

def main():
    styled = Image.open("/home/nemoclaw/iso-stl-lora/out/stitched_deploy.png").convert("RGB")
    W, H = styled.size
    polys = fetch_water((38.56, -90.33, 38.70, -90.12))
    mask_img = Image.new("L", (W, H), 0); d = ImageDraw.Draw(mask_img)
    for poly in polys:
        pts = [geo_to_canvas(lat, lng) for (lat, lng) in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    mask = np.asarray(mask_img) > 128
    print(f"OSM water rasterized: {100*mask.mean():.1f}% of canvas")
    # exclude raw-textured bridges within the river (assemble raw at canvas layout)
    raw = assemble_raw_canvas(W, H)
    rg = raw.mean(2).astype(np.float32)
    rstd = np.sqrt(np.maximum(uniform_filter(rg*rg, 9)-uniform_filter(rg, 9)**2, 0))
    textured = (rstd > 16) & mask                       # bridges + islands (both textured in river)
    blobs = binary_opening(textured, iterations=22)      # opening removes THIN bridges, keeps BLOBS (islands)
    bridge = binary_dilation(textured & ~blobs, iterations=1)  # thin structures only = real bridges
    flood = mask & ~bridge                               # flood water + islands(blobs), keep bridges
    print(f"protected {100*bridge.mean():.2f}% (thin bridges only); islands flooded {100*blobs.mean():.2f}%")
    # diagnostic overlay: OSM mask in cyan over styled
    ov = np.asarray(styled).copy(); ov[mask] = (ov[mask]*0.45 + np.array([0,200,255])*0.55).astype(np.uint8)
    Image.fromarray(ov).crop((0, int(H*0.25), int(W*0.42), int(H*0.45))).resize((1100,540)).save("/mnt/c/Users/rneeb/iso_OSM_overlay.jpg", quality=90)
    # apply flood
    st = np.asarray(styled, np.float32)
    from PIL import ImageFilter
    fm = np.asarray(Image.fromarray((flood*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32)/255.0
    muddy = np.array([104, 111, 120], np.float32)
    yy = np.linspace(0, 6.28, H)[:, None]; xx = np.linspace(0, 6.28, W)[None, :]
    varn = (np.sin(yy*1.7)+np.cos(xx*1.3))*1.2
    out = st*(1-fm[..., None]) + (muddy[None, None, :]+np.stack([varn]*3, -1))*fm[..., None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save("/home/nemoclaw/iso-stl-lora/out/stitched_osm_river.png")
    # verify crops: arch-east forest + a sandbar spot
    res = Image.open("/home/nemoclaw/iso-stl-lora/out/stitched_osm_river.png")
    res.crop((0, int(H*0.25), int(W*0.42), int(H*0.45))).resize((1100, 540)).save("/mnt/c/Users/rneeb/iso_OSM_arch.jpg", quality=92)
    print("OSM-flooded river saved + arch verify crop")

def assemble_raw_canvas(W, H):
    T = int(TILE*S)
    canvas = np.zeros((H, W, 3), np.uint8)
    def place(d, ncols, nrows, ox, oy):
        for c in range(ncols):
            for r in range(nrows):
                f = f"{d}/tile_{c}_{r}.png"
                if os.path.exists(f):
                    t = np.asarray(Image.open(f).convert("RGB").resize((T, T)))
                    y0, x0 = oy+r*T, ox+c*T
                    if y0+T <= H and x0+T <= W:
                        canvas[y0:y0+T, x0:x0+T] = t
    place("/home/nemoclaw/iso-stl-lora/dataset/corridor_raw", 17, 26, int(6*TILE*S), 0)
    place("/home/nemoclaw/iso-stl-lora/dataset/downtown_raw", 10, 10, 0, int(7*TILE*S))
    return canvas

if __name__ == "__main__":
    main()
