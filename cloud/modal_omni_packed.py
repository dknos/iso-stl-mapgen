"""Packed multi-stream Qwen-Image-Edit server: load N full model copies on ONE
GPU and serve N concurrent requests, so a big-VRAM card (B200 192GB / B300 288GB)
isn't wasted running a single 40GB stream. Used to BENCHMARK packing throughput
before committing to the full-city walk.

Deploy:  modal deploy cloud/modal_omni_packed.py
Endpoint: https://<your-modal-username>--qwen-edit-packed-packededitor-edit-b64.modal.run
"""
import asyncio, base64, os, random
from io import BytesIO
import modal
from pydantic import BaseModel

N_COPIES = int(os.environ.get("N_COPIES", "3"))   # full model copies on one GPU
PACK_GPU = os.environ.get("PACK_GPU", "B200")     # B200(192GB)~3-4, B300(288GB)~5-6

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "diffusers>=0.35.0", "transformers>=4.46",
    "accelerate", "peft", "pillow", "fastapi", "uvicorn", "python-multipart", "sentencepiece",
).env({"HF_HOME": "/data/hf"})

app = modal.App("qwen-edit-packed")
lora_volume = modal.Volume.from_name("isometric-lora-vol", create_if_missing=True)


class EditRequest(BaseModel):
    image_b64: str
    prompt: str
    negative_prompt: str | None = None
    true_cfg_scale: float = 2.0
    steps: int = 16
    guidance_scale: float = 3.0
    seed: int | None = None


@app.cls(image=image, gpu=PACK_GPU, volumes={"/data": lora_volume},
         max_containers=1,                      # force ONE GPU -> true packing measurement
         scaledown_window=120, timeout=1800)
@modal.concurrent(max_inputs=N_COPIES)          # one container handles N concurrent inputs
class PackedEditor:
    @modal.enter()
    def setup(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        n = N_COPIES
        print(f"🧰 loading {n} copies of Qwen-Image-Edit-2509 on {PACK_GPU} ...")
        self.pipes = []
        for i in range(n):
            p = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
            p.load_lora_weights("/data/loras/iso-stl-omni", adapter_name="iso",
                                weight_name="iso_stl_omni_v1.safetensors")
            p.set_adapters(["iso"], adapter_weights=[1.0])
            p.to("cuda"); p.set_progress_bar_config(disable=True)
            self.pipes.append(p)
            free, total = torch.cuda.mem_get_info()
            print(f"  copy {i+1}/{n} ready; VRAM free {free/1e9:.0f}/{total/1e9:.0f} GB")
        self.pool = asyncio.Queue()
        for p in self.pipes: self.pool.put_nowait(p)
        print("✅ packed server ready")

    @modal.fastapi_endpoint(method="POST")
    async def edit_b64(self, req: EditRequest):
        import torch
        from PIL import Image
        pipe = await self.pool.get()                 # grab a free copy
        try:
            img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
            seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
            def run():
                with torch.inference_mode():
                    return pipe(prompt=req.prompt, negative_prompt=req.negative_prompt,
                                true_cfg_scale=req.true_cfg_scale, image=img,
                                num_inference_steps=req.steps, guidance_scale=req.guidance_scale,
                                generator=torch.manual_seed(seed)).images[0]
            out = await asyncio.to_thread(run)        # run blocking infer off the event loop
            buf = BytesIO(); out.save(buf, "PNG")
            return {"image_b64": base64.b64encode(buf.getvalue()).decode()}
        finally:
            self.pool.put_nowait(pipe)
