import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter, label, binary_closing
Image.MAX_IMAGE_PIXELS = None

S = 0.5
TILE = 1024

def finish(im):
    # WHITES ONLY (moderate) -- water is handled in ONE separate pass (river_flood) to
    # avoid double-processing / haze. T200/k0.40 tames blown roofs without flattening.
    a = np.asarray(im.convert("RGB"), np.float32)
    T, k = 200., 0.40; L = a.mean(2); over = L > T
    a = a * np.where(over, (T+(L-T)*k)/np.maximum(L, 1), 1.0)[..., None]
    return np.clip(a, 0, 255)

cor = Image.open("/home/nemoclaw/iso-stl-lora/out/corridor_v4_3000.png")
dtn = Image.open("/home/nemoclaw/iso-stl-lora/out/downtown_v4_3000.png")
cw, ch = int(cor.size[0]*S), int(cor.size[1]*S)
dw, dh = int(dtn.size[0]*S), int(dtn.size[1]*S)
print("finishing corridor..."); corA = finish(cor.resize((cw, ch)))
print("finishing downtown..."); dtnA = finish(dtn.resize((dw, dh)))

CW = int(23*TILE*S); CH = int(26*TILE*S)
canvas = np.zeros((CH, CW, 3), np.float32)
covered = np.zeros((CH, CW), bool)
cx = int(6*TILE*S)
canvas[0:ch, cx:cx+cw] = corA; covered[0:ch, cx:cx+cw] = True
dy = int(7*TILE*S)
fade = max(4, int(0.6*TILE*S))
ay = np.ones((dh, dw), np.float32)
for i in range(fade):
    v = i/fade; ay[i,:]*=v; ay[dh-1-i,:]*=v; ay[:,dw-1-i]*=v
alpha = ay[..., None]
canvas[dy:dy+dh, 0:dw] = dtnA*alpha + canvas[dy:dy+dh, 0:dw]*(1-alpha)
covered[dy:dy+dh, 0:dw] |= (ay > 0.05)

# fill uncovered (black corners) with a muted ground tone (median of covered ground)
ground = np.array([120, 135, 110], np.float32)  # muted green-grey for uncovered corners
canvas[~covered] = ground

out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
out.save("/home/nemoclaw/iso-stl-lora/out/stitched_deploy.png")
print("deploy stitch saved", out.size)
