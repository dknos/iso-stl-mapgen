"""Local ComfyUI shim — exposes the SAME HTTP interface walk_grid_rect.py expects
(POST {image_b64,prompt,steps,guidance_scale,true_cfg_scale,seed} -> {image_b64}),
internally driving a running ComfyUI (Qwen-Image-Edit-2509 GGUF + iso LoRA, 5080+2080).

Run:  python3 comfy_local_shim.py            # listens 127.0.0.1:8191
Then: walk_grid_rect.py --endpoint http://127.0.0.1:8191/edit ...
"""
import json, time, base64, threading, urllib.request, urllib.parse, os, io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY   = os.environ.get("COMFY_URL", "http://127.0.0.1:8190")
INPUT   = os.environ.get("COMFY_INPUT", "/home/nemoclaw/ComfyUI/input")
PORT    = int(os.environ.get("SHIM_PORT", "8191"))
CFG     = float(os.environ.get("SHIM_CFG", "2.5"))     # validated KSampler cfg for this LoRA
VVRAM   = float(os.environ.get("SHIM_VVRAM", "8.0"))   # GB of Q6 DiT offloaded 5080->CPU
UNET    = "Qwen-Image-Edit-2509-Q6_K.gguf"
CLIP    = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
VAE     = "qwen_image_vae.safetensors"
LORA    = os.environ.get("SHIM_LORA", "iso_stl_omni_v4_3000.safetensors")
_ctr = [0]; _lock = threading.Lock()

def build(img_name, prompt, steps, cfg, seed):
    # Q6_K is ~16GB. 5080 has ~12.8GB free after residual, so offload 8GB to CPU.
    # DiT on cuda:0; the 5GB VL encoder lives on the idle 2080 (cuda:1).
    return {
     "1":{"class_type":"UnetLoaderGGUFDisTorch2MultiGPU","inputs":{
          "unet_name":UNET,"compute_device":"cuda:0","virtual_vram_gb":VVRAM,
          "donor_device":"cpu","eject_models":False}},
     "2":{"class_type":"LoraLoaderModelOnly","inputs":{
          "model":["1",0],"lora_name":LORA,"strength_model":1.0}},
     "3":{"class_type":"CLIPLoaderGGUFMultiGPU","inputs":{
          "clip_name":CLIP,"type":"qwen_image","device":"cuda:1"}},
     "4":{"class_type":"VAELoader","inputs":{"vae_name":VAE}},
     "5":{"class_type":"LoadImage","inputs":{"image":img_name}},
     "6":{"class_type":"TextEncodeQwenImageEditPlus","inputs":{
          "clip":["3",0],"vae":["4",0],"image1":["5",0],"prompt":prompt}},
     "7":{"class_type":"ConditioningZeroOut","inputs":{"conditioning":["6",0]}},
     "8":{"class_type":"EmptySD3LatentImage","inputs":{"width":1024,"height":1024,"batch_size":1}},
     "9":{"class_type":"KSampler","inputs":{
          "model":["2",0],"seed":int(seed)%(2**63),"steps":int(steps),"cfg":float(cfg),
          "sampler_name":"euler","scheduler":"simple","positive":["6",0],
          "negative":["7",0],"latent_image":["8",0],"denoise":1.0}},
     "10":{"class_type":"VAEDecode","inputs":{"samples":["9",0],"vae":["4",0]}},
     "11":{"class_type":"SaveImage","inputs":{"images":["10",0],"filename_prefix":img_name[:-4]}},
    }

def run_one(image_b64, prompt, steps, cfg, seed):
    with _lock:
        _ctr[0]+=1; uid=f"_shim_{int(time.time())}_{_ctr[0]}"
    name=uid+".png"
    open(os.path.join(INPUT,name),"wb").write(base64.b64decode(image_b64))
    wf=build(name, prompt, steps, cfg, seed)
    req=urllib.request.Request(COMFY+"/prompt",
        data=json.dumps({"prompt":wf}).encode(),headers={"Content-Type":"application/json"})
    pid=json.load(urllib.request.urlopen(req,timeout=60))["prompt_id"]
    out=None
    for _ in range(900):
        time.sleep(1.5)
        try: h=json.load(urllib.request.urlopen(COMFY+f"/history/{pid}",timeout=20))
        except Exception: continue
        if pid in h:
            st=h[pid].get("status",{})
            if st.get("status_str")=="error":
                raise RuntimeError("comfy exec error: "+json.dumps(st)[:400])
            imgs=h[pid].get("outputs",{}).get("11",{}).get("images",[])
            if imgs: out=imgs[0]; break
    try: os.remove(os.path.join(INPUT,name))
    except OSError: pass
    if not out: raise RuntimeError("no output (timeout)")
    url=COMFY+"/view?"+urllib.parse.urlencode(
        {"filename":out["filename"],"subfolder":out.get("subfolder",""),"type":out.get("type","output")})
    data=urllib.request.urlopen(url,timeout=30).read()
    return base64.b64encode(data).decode()

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        try:
            body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res=run_one(body["image_b64"], body["prompt"],
                        body.get("steps",14), CFG, body.get("seed",0))
            payload=json.dumps({"image_b64":res}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(payload))); self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            msg=json.dumps({"error":str(e)[:300]}).encode()
            self.send_response(500); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(msg))); self.end_headers()
            self.wfile.write(msg)

if __name__=="__main__":
    print(f"shim on 127.0.0.1:{PORT} -> {COMFY}  (cfg={CFG} vvram={VVRAM})  POST /edit")
    ThreadingHTTPServer(("127.0.0.1",PORT),H).serve_forever()
