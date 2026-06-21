#!/usr/bin/env python3
"""Headless RunPod B200/B300 INFERENCE launcher. Provisions a Blackwell pod,
uploads LoRA weights + tiles + infer_batched.py + setup_infer.sh, runs inference
(GPU-sanity fail-fast before the model download), downloads styled tiles, prints
the terminate command. Reuse for the city: --in <dir> --out <dir>.

Usage:
  python3 cloud/runpod_infer.py --gpu "NVIDIA B200" \
      --tiles /tmp/heldout.tgz \
      --lora ~/iso-stl-lora/output/iso_stl_diorama_v1/iso_stl_diorama_v1.safetensors \
      --lora2 ~/iso-stl-lora/output/iso_stl_diorama_v1/iso_stl_diorama_v1_000001000.safetensors \
      --out ~/iso-stl-lora/output/iso_stl_diorama_v1
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

# Blackwell-ready image: torch 2.8 / CUDA 12.8 (sm_100 kernels). cu124 images
# lack B200 kernels -> "no kernel image" at runtime.
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
KEY = str(Path.home() / ".ssh" / "id_ed25519")

def api_key():
    for line in open(Path.home() / ".iso_runpod_env"):
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no RUNPOD_API_KEY")

def gql(key, query):
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"content-type": "application/json", "user-agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=40))
    if r.get("errors"):
        raise SystemExit("GraphQL error: " + json.dumps(r["errors"])[:400])
    return r["data"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--disk", type=int, default=140)
    ap.add_argument("--tiles", required=True, help="local .tgz of tiles (extracted to heldout/)")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--lora2", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=28, help="diffusion steps (lower=faster)")
    ap.add_argument("--lora-scale", type=float, default=1.0, help="LoRA adapter weight (<1 = gentler)")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--sweep", action="store_true", help="run stylization-strength sweep -> sweep_grid.png")
    ap.add_argument("--keep", action="store_true", help="do NOT auto-terminate (for chaining)")
    a = ap.parse_args()
    key = api_key()
    HERE = Path(__file__).resolve().parent

    def terminate(pid):
        try:
            gql(key, f'mutation{{podTerminate(input:{{podId:"{pid}"}})}}')
            print(f"[terminate] pod {pid} terminated")
        except Exception as e:
            print(f"[terminate] FAILED {str(e)[:80]} — MANUALLY: python3 cloud/runpod_terminate.py {pid}")

    print(f"creating {a.gpu} pod ({IMAGE})...")
    m = gql(key, f"""mutation{{ podFindAndDeployOnDemand(input:{{
        cloudType: ALL, gpuCount:1, volumeInGb:0, containerDiskInGb:{a.disk},
        minVcpuCount:8, minMemoryInGb:64, gpuTypeId:{json.dumps(a.gpu)},
        name:"iso-stl-infer", imageName:{json.dumps(IMAGE)},
        ports:"22/tcp", startSsh:true, volumeMountPath:"/workspace"
    }}){{ id }} }}""")
    pid = m["podFindAndDeployOnDemand"]["id"]
    print(f"pod {pid} — waiting for SSH...  (terminate: python3 cloud/runpod_terminate.py {pid})")

    try:
        ip = port = None
        for _ in range(90):
            time.sleep(10)
            d = gql(key, f'query{{ pod(input:{{podId:"{pid}"}}){{ runtime{{ ports{{ ip isIpPublic privatePort publicPort type }} }} }} }}')
            rt = (d.get("pod") or {}).get("runtime") or {}
            for p in rt.get("ports") or []:
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    ip, port = p["ip"], p["publicPort"]
            if ip: break
            print("  ...provisioning")
        if not ip:
            raise SystemExit(f"pod {pid} never exposed SSH")
        print(f"SSH: root@{ip}:{port}")

        import paramiko
        from scp import SCPClient
        ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        for _ in range(30):
            try: ssh.connect(ip, port=port, username="root", key_filename=KEY, timeout=15); break
            except Exception as e: print(f"  ssh retry: {str(e)[:50]}"); time.sleep(10)
        else:
            raise SystemExit("SSH failed")

        print("uploading weights + tiles + scripts...")
        uploads = [(a.tiles, "/workspace/heldout.tgz"),
                   (a.lora, "/workspace/iso_stl_diorama_v1.safetensors"),
                   (str(HERE.parent / "scripts" / "infer_batched.py"), "/workspace/infer_batched.py"),
                   (str(HERE.parent / "scripts" / "infer_sweep.py"), "/workspace/infer_sweep.py"),
                   (str(HERE / "setup_infer.sh"), "/workspace/setup_infer.sh")]
        if a.lora2:
            uploads.append((a.lora2, "/workspace/iso_stl_diorama_v1_000001000.safetensors"))
        with SCPClient(ssh.get_transport()) as scp:
            for src, dst in uploads:
                print(f"  {Path(src).name} -> {dst}"); scp.put(src, dst)
        print("uploaded.")

        print("running inference (streaming)...")
        ch = ssh.get_transport().open_session(); ch.get_pty()
        ch.exec_command(f"cd /workspace && SWEEP={1 if a.sweep else 0} INFER_STEPS={a.steps} "
                        f"LORA_SCALE={a.lora_scale} GUIDANCE={a.guidance} bash setup_infer.sh 2>&1")
        while not ch.exit_status_ready():
            if ch.recv_ready():
                sys.stdout.write(ch.recv(4096).decode(errors="ignore")); sys.stdout.flush()
            time.sleep(0.4)
        while ch.recv_ready():
            sys.stdout.write(ch.recv(4096).decode(errors="ignore"))

        out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
        if a.sweep:
            print("\ndownloading sweep grid...")
            with SCPClient(ssh.get_transport()) as scp:
                try: scp.get("/workspace/sweep_grid.png", str(out / "sweep_grid.png"))
                except Exception as e: print(f"  sweep_grid download failed: {str(e)[:60]}")
                try: scp.get("/workspace/sweep_out", str(out), recursive=True)
                except Exception as e: print(f"  sweep_out download failed: {str(e)[:60]}")
            print(f"sweep -> {out}/sweep_grid.png")
        else:
            print("\ndownloading styled tiles...")
            with SCPClient(ssh.get_transport()) as scp:
                for d in ["styled_2000", "styled_1000"]:
                    try:
                        (out / d).mkdir(parents=True, exist_ok=True)
                        scp.get(f"/workspace/{d}", str(out), recursive=True)
                    except Exception as e:
                        print(f"  {d} download failed: {str(e)[:60]}")
            print(f"styled -> {out}")
    finally:
        if a.keep:
            print(f"[keep] pod {pid} LEFT RUNNING — terminate manually: python3 cloud/runpod_terminate.py {pid}")
        else:
            terminate(pid)
            time.sleep(5)
            rem = gql(key, "query{myself{pods{id}}}").get("myself", {}).get("pods", [])
            print(f"[verify] remaining pods: {[p['id'] for p in rem] if rem else 'NONE'}")
    print("POD_ID=" + pid)

if __name__ == "__main__":
    main()
