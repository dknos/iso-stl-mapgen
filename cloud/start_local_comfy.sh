#!/bin/bash
# Start the LOCAL $0 iso-stl walk backend: ComfyUI (Qwen-Image-Edit-2509 GGUF + iso LoRA)
# on the 5080 (Qwen-Image-Edit-2509-Q6_K, +8GB CPU offload) with the VL text encoder on the 2080, plus the shim that
# speaks walk_grid_rect.py's HTTP interface.  Idempotent: skips anything already up.
#
# After this: walk_grid_rect.py --endpoint http://127.0.0.1:8195/edit ...
set -u
COMFY_PORT=8190
SHIM_PORT=8195
cd /home/nemoclaw/ComfyUI || exit 1

is_up(){ ss -tlnp 2>/dev/null | grep -q "127.0.0.1:$1"; }

if is_up "$COMFY_PORT"; then
  echo "ComfyUI already up on :$COMFY_PORT"
else
  echo "starting ComfyUI on :$COMFY_PORT ..."
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup python3 main.py --listen 127.0.0.1 --port "$COMFY_PORT" --preview-method none \
    > /home/nemoclaw/_comfy_server.log 2>&1 &
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$COMFY_PORT/object_info 2>/dev/null)" = "200" ] && break
    sleep 2
  done
  echo "ComfyUI ready"
fi

if is_up "$SHIM_PORT"; then
  echo "shim already up on :$SHIM_PORT"
else
  echo "starting shim on :$SHIM_PORT ..."
  SHIM_PORT="$SHIM_PORT" SHIM_UNET="Qwen-Image-Edit-2509-Q6_K.gguf" SHIM_VVRAM=8 nohup python3 -u \
    /home/nemoclaw/iso-stl-lora/cloud/comfy_local_shim.py > /home/nemoclaw/_shim.log 2>&1 &
  sleep 3
  tail -1 /home/nemoclaw/_shim.log
fi
echo "LOCAL walk backend ready -> http://127.0.0.1:$SHIM_PORT/edit"
