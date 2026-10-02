"""Modal Qwen-Image-Edit-2509 + iso-stl OMNI LoRA server (cannoneyed inference/
server.py adapted: base swapped to Qwen-Image-Edit-2509 + QwenImageEditPlusPipeline
to MATCH the LoRA's training arch (qwen_image_edit_plus). LoRA in volume.

The HTTP endpoint bills YOUR Modal account. It refuses to deploy unless
ISO_EDIT_TOKEN is set, and every request must send that token back.

Deploy:
  modal volume create isometric-lora-vol            # once
  modal volume put isometric-lora-vol <local.safetensors> /loras/iso-stl-omni/iso_stl_omni_v4_3000.safetensors
  export ISO_EDIT_TOKEN=$(openssl rand -hex 24)
  LORA_MODEL_ID=iso-stl-omni LORA_WEIGHT_NAME=iso_stl_omni_v4_3000.safetensors modal deploy cloud/modal_omni_server.py
"""
import base64, hmac, os, random
from io import BytesIO
import modal
from fastapi import HTTPException
from pydantic import BaseModel

EDIT_TOKEN = os.environ.get("ISO_EDIT_TOKEN", "")
if len(EDIT_TOKEN) < 16:
    raise SystemExit("Refusing to deploy: set ISO_EDIT_TOKEN to at least 16 characters (openssl rand -hex 24).")

DEFAULT_LORA_MODEL_ID = "iso-stl-omni"
LORA_MODEL_ID = os.environ.get("LORA_MODEL_ID", DEFAULT_LORA_MODEL_ID)
LORA_WEIGHT_NAME = os.environ.get("LORA_WEIGHT_NAME", "iso_stl_omni_v4_3000.safetensors")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "diffusers>=0.35.0", "transformers>=4.46",
    "accelerate", "peft", "pillow", "fastapi", "uvicorn", "python-multipart", "sentencepiece",
).env({"HF_HOME": "/data/hf"})  # cache the 20B base on the volume -> no 40GB re-download per cold start

LORA_SHORT_ID = "-".join(LORA_MODEL_ID.split("-")[-2:]) if "-" in LORA_MODEL_ID else LORA_MODEL_ID
app = modal.App(f"qwen-edit-{LORA_SHORT_ID}")
lora_volume = modal.Volume.from_name("isometric-lora-vol", create_if_missing=True)


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    negative_prompt: str | None = None
    true_cfg_scale: float = 2.0
    steps: int = 14
    guidance_scale: float = 3.0
    seed: int | None = None
    token: str | None = None


def require_token(req: EditRequest):
    got = req.token or ""
    if len(got) != len(EDIT_TOKEN) or not hmac.compare_digest(got, EDIT_TOKEN):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.cls(image=image, gpu="B200", volumes={"/data": lora_volume},  # Modal handles Blackwell drivers
         scaledown_window=300, timeout=600,
         max_containers=30,                 # autoscale up to 30 B200s for the parallel wavefront walk
         secrets=[modal.Secret.from_dict({"LORA_MODEL_ID": LORA_MODEL_ID,
                                          "LORA_WEIGHT_NAME": LORA_WEIGHT_NAME,
                                          "ISO_EDIT_TOKEN": EDIT_TOKEN})])
class ImageEditor:
    @modal.enter()
    def setup(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline   # 2509 plus arch (matches LoRA)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch_cache"
        self.lora_model_id = os.environ.get("LORA_MODEL_ID", DEFAULT_LORA_MODEL_ID)
        self.lora_weight_name = os.environ.get("LORA_WEIGHT_NAME") or None
        lora_path = f"/data/loras/{self.lora_model_id}"
        print(f"🚀 Qwen-Image-Edit-2509 + LoRA {self.lora_model_id} ({self.lora_weight_name})")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
        try:
            self.pipe.load_lora_weights(lora_path, adapter_name="isometric",
                                        weight_name=self.lora_weight_name)
            self.pipe.set_adapters(["isometric"], adapter_weights=[1.0])
            print("🔗 LoRA loaded")
        except Exception as e:
            print(f"⚠️ LoRA load failed: {e}")
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        print("✅ ready")

    @modal.fastapi_endpoint(method="POST")
    async def edit_b64(self, req: EditRequest):
        require_token(req)
        from PIL import Image
        img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
        with torch.inference_mode():
            out = self.pipe(prompt=req.prompt, negative_prompt=req.negative_prompt,
                            true_cfg_scale=req.true_cfg_scale, image=img,
                            num_inference_steps=req.steps, guidance_scale=req.guidance_scale,
                            generator=torch.manual_seed(seed)).images[0]
        buf = BytesIO(); out.save(buf, "PNG")
        return {"image_b64": base64.b64encode(buf.getvalue()).decode()}
