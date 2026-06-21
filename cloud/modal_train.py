"""Train the iso-stl OMNI LoRA on MODAL (RunPod ran out of balance; Modal credits
work + Modal manages drivers = no cu130/cu124 mismatch + no preemption + no SSH).

Runs ai-toolkit (the proven Qwen-Image-Edit-2509 trainer) inside a long Modal
function. Dataset comes from a Modal volume; the trained LoRA is written to the
SAME serving volume the inference server reads, so deploy = just bounce.

Setup (once):
  modal volume put -f iso-stl-omni-ds dataset/omni_v2 /omni_v2
Run (detached, survives disconnect):
  modal run --detach cloud/modal_train.py
"""
import modal

app = modal.App("iso-stl-omni-train")
ds_vol = modal.Volume.from_name("iso-stl-omni-ds", create_if_missing=True)
lora_vol = modal.Volume.from_name("isometric-lora-vol", create_if_missing=True)

# Modal drivers are current -> ai-toolkit's torch installs/runs fine; pin numpy<2
# (scipy needs <1.29) which was the only real dep conflict.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "ffmpeg")  # libGL for opencv
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/ostris/ai-toolkit.git",
        "cd /root/ai-toolkit && pip install -r requirements.txt",
        "pip install torchaudio",          # ai-toolkit imports it; matches the installed torch
        "pip install 'numpy<2' --force-reinstall",
        "python -c \"import torchaudio,torch,numpy,cv2;print('deps ok', torch.__version__, numpy.__version__)\"",
    )
    .env({"HF_HOME": "/data/hf"})
)

TRIGGER = "<isometric st louis pixel art>"
INFILL_PROMPT = ("Fill in the red-outlined region with the missing pixels in the " + TRIGGER +
                 " isometric 3D city diorama style, removing the red border and exactly following the "
                 "color, shape, and structure of the surrounding already-generated image so the result "
                 "is seamless.")

CONFIG = """
job: extension
config:
  name: "iso_stl_omni_v5"
  process:
    - type: 'diffusion_trainer'
      training_folder: "/root/output"
      device: cuda:0
      network: {{ type: "lora", linear: 32, linear_alpha: 32 }}
      save: {{ dtype: float16, save_every: 500, max_step_saves_to_keep: 10 }}
      datasets:
        - folder_path: "/ds/omni_v5/target"
          control_path: [ "/ds/omni_v5/control" ]
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [ 768, 1024 ]
      train:
        batch_size: 1
        cache_text_embeddings: true
        steps: 4000
        gradient_accumulation: 1
        timestep_type: "weighted"
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw"
        lr: 1e-4
        dtype: bf16
      model: {{ name_or_path: "Qwen/Qwen-Image-Edit-2509", arch: "qwen_image_edit_plus", quantize: false }}
      sample:
        sampler: "flowmatch"
        sample_every: 100000
        width: 1024
        height: 1024
        samples: [ {{ prompt: "{prompt}", ctrl_img_1: "{ctrl}" }} ]
        neg: "blurry, hazy, photograph, text, watermark, red border, seam"
        seed: 42
        sample_steps: 20
meta: {{ name: "[name]", version: '1.0' }}
"""


@app.function(image=image, gpu="H100", volumes={"/ds": ds_vol, "/data": lora_vol},
              timeout=4 * 3600)
def train():
    import os, shutil, subprocess
    print("=== sanity ===")
    subprocess.run("python -c \"import torch;print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))\"", shell=True, check=True)
    subprocess.run("python -c \"import numpy;print('numpy',numpy.__version__)\"", shell=True, check=True)
    print(f"=== dataset ===")
    subprocess.run("ls /ds/omni_v5/control | wc -l; ls /ds/omni_v5/target | wc -l", shell=True)
    import glob
    ctrls = sorted(glob.glob("/ds/omni_v5/control/*.png"))
    if not ctrls: raise RuntimeError("no control images in /ds/omni_v5/control")
    ctrl = next((c for c in ctrls if "_00_full" in c), ctrls[0])   # a full-tile control for the sample
    print(f"sample ctrl: {ctrl}")
    open("/root/cfg.yaml", "w").write(CONFIG.format(prompt=INFILL_PROMPT, ctrl=ctrl))
    os.makedirs("/data/hf", exist_ok=True)
    print("=== train ===")
    subprocess.run("cd /root/ai-toolkit && python run.py /root/cfg.yaml", shell=True, check=True)
    # Stage ALL v5 checkpoints (= v4 dataset + checkerboard water aug) to an EVAL dir for
    # held-out review. Do NOT overwrite the live-served LoRA — it stays live until a v5
    # checkpoint is picked + approved via the river-band walk A/B.
    ckpt_dir = "/data/loras/iso-stl-omni-v5-ckpts"; os.makedirs(ckpt_dir, exist_ok=True)
    cks = sorted(glob.glob("/root/output/iso_stl_omni_v5/iso_stl_omni_v5*.safetensors"))
    for c in cks:
        shutil.copy(c, f"{ckpt_dir}/{os.path.basename(c)}")
    lora_vol.commit()
    print(f"=== DONE: {len(cks)} v5 checkpoints staged -> {ckpt_dir} (live v3/v4 untouched) ===")
    print("checkpoints:", [os.path.basename(c) for c in cks])
    return "ok"


@app.local_entrypoint()
def main():
    print(train.remote())
