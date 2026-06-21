import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter, label, binary_closing
Image.MAX_IMAGE_PIXELS = None

S = 0.11   # working scale for preview
TILE = 1024

def finish(im):
    a = np.asarray(im.convert("RGB"), np.float32)
    # STRONG T190 whites
    T, k = 190., 0.30; L = a.mean(2); over = L > T
    a = a * np.where(over, (T+(L-T)*k)/np.maximum(L, 1), 1.0)[..., None]
    # 018 muddy soft-blend within spatial river mask
    g = a.mean(2); win = max(9, int(31*S))
    m = uniform_filter(g, win); std = np.sqrt(np.maximum(uniform_filter(g*g, win)-m*m, 0))
    R, Gc, B = a[..., 0], a[..., 1], a[..., 2]
    watery = (std < 9) & (g < 150) & (~((Gc-R > 10) & (Gc-B > 10)))
    lab, n = label(watery); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    thr = a.shape[0]*a.shape[1]*0.004
    big = np.where(sizes > thr)[0]; mask = np.isin(lab, big); mask = binary_closing(mask, iterations=2)
    if mask.sum() > 100:
        med = np.median(a[mask], axis=0)
        fm = np.asarray(Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32)/255.0
        strength = 0.75*fm
        a = a*(1-strength[..., None]) + med[None, None, :]*strength[..., None]
    return np.clip(a, 0, 255)

# load + downscale + finish
cor = Image.open("/home/nemoclaw/iso-stl-lora/out/corridor_v4_3000.png")
dtn = Image.open("/home/nemoclaw/iso-stl-lora/out/downtown_v4_3000.png")
cw, ch = int(cor.size[0]*S), int(cor.size[1]*S)
dw, dh = int(dtn.size[0]*S), int(dtn.size[1]*S)
corA = finish(cor.resize((cw, ch)))
dtnA = finish(dtn.resize((dw, dh)))

# canvas = wedge cols 5-27 (23 cols), rows 2-27 (26 rows)
CW = int(23*TILE*S); CH = int(26*TILE*S)
canvas = np.zeros((CH, CW, 3), np.float32)
# corridor = cols 11-27 -> x offset (11-5)=6 cols ; rows 2-27 -> y 0
cx = int(6*TILE*S)
canvas[0:ch, cx:cx+cw] = corA
# downtown = cols 5-14 -> x 0 ; rows 9-18 -> y (9-2)=7 cols
dy = int(7*TILE*S)
# feather downtown top/bottom/right edges into corridor (left edge = map boundary, keep hard)
fade = max(4, int(0.6*TILE*S))
ay = np.ones((dh, dw), np.float32)
for i in range(fade):
    v = i/fade
    ay[i, :] *= v          # top
    ay[dh-1-i, :] *= v     # bottom
    ay[:, dw-1-i] *= v     # right
alpha = ay[..., None]
region = canvas[dy:dy+dh, 0:dw]
canvas[dy:dy+dh, 0:dw] = dtnA*alpha + region*(1-alpha)

out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
out.save("/mnt/c/Users/rneeb/iso_STITCHED.jpg", quality=92)
out.save("/home/nemoclaw/iso-stl-lora/out/stitched_preview.png")
print("stitched", out.size, "-> iso_STITCHED.jpg")
