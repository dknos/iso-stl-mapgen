#!/usr/bin/env python3
"""Headless RunPod H100 launcher (GraphQL ?api_key= auth — the method that
actually authenticates for this key). Creates the pod, SSHes in with the local
key, uploads the dataset+config+setup, runs training, downloads the LoRA.

Prereqs:
  - ~/.iso_runpod_env  with RUNPOD_API_KEY=...   (read/write key)
  - the WSL pubkey (~/.ssh/id_ed25519.pub) added to RunPod > Settings > SSH Keys
  - pip install paramiko scp   (done)

Usage:
  python3 cloud/runpod_launch.py [--gpu "NVIDIA H100 80GB HBM3"] [--no-train]
"""
import json, os, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["iso-stl-dataset.tar.gz", "train_qwen_edit_cloud.yaml", "runpod_setup.sh"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY = str(Path.home() / ".ssh" / "id_ed25519")

def api_key():
    for line in open(Path.home() / ".iso_runpod_env"):
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no RUNPOD_API_KEY in ~/.iso_runpod_env")

def gql(key, query):
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": query}).encode(),
        headers={"content-type": "application/json", "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"})
    r = json.load(urllib.request.urlopen(req, timeout=30))
    if r.get("errors"):
        raise SystemExit("GraphQL error: " + json.dumps(r["errors"])[:300])
    return r["data"]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H200")  # H200 SXM: Hopper (no Blackwell driver pain) + ~1.3x H100
    ap.add_argument("--disk", type=int, default=120)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--setup", default="runpod_setup.sh", help="pod bootstrap script to run")
    ap.add_argument("--files", default=",".join(FILES), help="comma list of files to upload")
    ap.add_argument("--output-name", default="iso_stl_diorama_v1", help="output subdir + .safetensors name")
    ap.add_argument("--keep", action="store_true", help="do NOT auto-terminate the pod after training")
    ap.add_argument("--image", default=IMAGE, help="container image (use cu128 for Blackwell B200/B300)")
    a = ap.parse_args()
    globals()["IMAGE"] = a.image
    key = api_key()
    files_list = [s.strip() for s in a.files.split(",") if s.strip()]
    globals()["FILES"] = files_list
    for f in FILES:
        if not (HERE / f).exists():
            sys.exit(f"missing {f} — run cloud/package_dataset.sh")

    print(f"creating {a.gpu} pod...")
    gpu = json.dumps(a.gpu)
    img = json.dumps(IMAGE)
    m = gql(key, f"""mutation{{ podFindAndDeployOnDemand(input:{{
        cloudType: ALL, gpuCount:1, volumeInGb:0, containerDiskInGb:{a.disk},
        minVcpuCount:8, minMemoryInGb:64, gpuTypeId:{gpu},
        name:"iso-stl-lora", imageName:{img},
        ports:"22/tcp", startSsh:true,
        dockerArgs:"", volumeMountPath:"/workspace"
    }}){{ id }} }}""")
    pid = m["podFindAndDeployOnDemand"]["id"]
    print(f"pod {pid} — waiting for SSH...")

    ip = port = None
    for _ in range(90):
        time.sleep(10)
        d = gql(key, f'query{{ pod(input:{{podId:"{pid}"}}){{ runtime{{ ports{{ ip isIpPublic privatePort publicPort type }} }} }} }}')
        rt = (d.get("pod") or {}).get("runtime") or {}
        for p in rt.get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                ip, port = p["ip"], p["publicPort"]
        if ip:
            break
        print("  ...provisioning")
    if not ip:
        sys.exit(f"pod {pid} never exposed SSH — check runpod.io. Terminate it if unused.")
    print(f"SSH: root@{ip}:{port}")

    import paramiko
    from scp import SCPClient
    ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for _ in range(30):
        try:
            ssh.connect(ip, port=port, username="root", key_filename=KEY, timeout=15); break
        except Exception as e:
            print(f"  ssh retry: {str(e)[:50]}"); time.sleep(10)
    else:
        sys.exit("SSH failed — is the pubkey added to RunPod + private key at ~/.ssh/id_ed25519?")

    print("uploading...")
    with SCPClient(ssh.get_transport()) as scp:
        for f in FILES:
            scp.put(str(HERE / f), f"/workspace/{f}")
    print("uploaded.")
    if a.no_train:
        print(f"pod ready. ssh root@{ip} -p {port}; cd /workspace && bash runpod_setup.sh")
        return

    try:
        print("training (streaming)...")
        ch = ssh.get_transport().open_session(); ch.get_pty()
        ch.exec_command(f"cd /workspace && bash {a.setup} 2>&1")
        while not ch.exit_status_ready():
            if ch.recv_ready():
                sys.stdout.write(ch.recv(4096).decode(errors="ignore")); sys.stdout.flush()
            time.sleep(0.4)
        while ch.recv_ready():
            sys.stdout.write(ch.recv(4096).decode(errors="ignore"))

        print("\ndownloading LoRA (+ all checkpoints)...")
        out = Path.home() / "iso-stl-lora" / "output" / a.output_name; out.mkdir(parents=True, exist_ok=True)
        with SCPClient(ssh.get_transport()) as scp:
            try:
                scp.get(f"/workspace/output/{a.output_name}", str(out.parent), recursive=True)
            except Exception as e:
                print(f"  ckpt dir scp failed ({str(e)[:50]}), trying final only")
                scp.get(f"/workspace/output/{a.output_name}/{a.output_name}.safetensors",
                        str(out / f"{a.output_name}.safetensors"))
        print(f"LoRA -> {out}")
        print("local weights:", [p.name for p in out.glob("*.safetensors")])
    finally:
        if not a.keep:
            try:
                gql(key, f'mutation{{podTerminate(input:{{podId:"{pid}"}})}}')
                print(f"[terminate] pod {pid} terminated")
                time.sleep(5)
                rem = gql(key, "query{myself{pods{id}}}").get("myself", {}).get("pods", [])
                print(f"[verify] remaining pods: {[p['id'] for p in rem] if rem else 'NONE'}")
            except Exception as e:
                print(f"[terminate] FAILED {str(e)[:80]} — MANUAL: python3 cloud/runpod_terminate.py {pid}")
        else:
            print(f"[keep] pod {pid} left running — terminate: python3 cloud/runpod_terminate.py {pid}")

if __name__ == "__main__":
    main()
