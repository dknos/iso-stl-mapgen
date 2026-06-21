#!/usr/bin/env python3
"""Robust RunPod training launcher: starts training DETACHED on the pod (nohup),
disconnects, then POLLS with fresh short SSH connections for the output .safetensors.
Survives SSH-stream drops (the killer for multi-hour trains over a single channel).
Auto-terminates in finally.

Usage:
  python3 cloud/runpod_train_detached.py --gpu "NVIDIA H200" --setup setup_omni_v2.sh \
    --files "omni-dataset-v2.tar.gz,train_qwen_omni_v2.yaml,setup_omni_v2.sh" \
    --output-name iso_stl_omni_v2
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY = str(Path.home() / ".ssh" / "id_ed25519")
SSHO = dict(username="root", key_filename=KEY, timeout=20)

def api_key():
    for l in open(Path.home()/".iso_runpod_env"):
        if l.startswith("RUNPOD_API_KEY="): return l.split("=",1)[1].strip()
    sys.exit("no key")

def gql(key, q):
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": q}).encode(),
        headers={"content-type":"application/json","user-agent":"Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=40))
    if r.get("errors"): raise SystemExit("GraphQL: "+json.dumps(r["errors"])[:300])
    return r["data"]

def ssh_connect(ip, port, tries=6):
    import paramiko
    for _ in range(tries):
        try:
            c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, port=port, **SSHO); return c
        except Exception as e:
            print(f"  ssh retry: {str(e)[:50]}"); time.sleep(10)
    return None

def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="ignore"), e.read().decode(errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="NVIDIA H200")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--disk", type=int, default=140)
    ap.add_argument("--setup", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--max-hours", type=float, default=4.0)
    a = ap.parse_args()
    key = api_key(); HERE = Path(__file__).resolve().parent
    files = [s.strip() for s in a.files.split(",") if s.strip()]
    out_local = Path.home()/"iso-stl-lora"/"output"/a.output_name
    pidfile = Path.home()/"iso-stl-lora"/".train_pod_id"

    def ports_for(pid):
        for _ in range(90):
            d = gql(key, f'query{{ pod(input:{{podId:"{pid}"}}){{ runtime{{ ports{{ ip isIpPublic privatePort publicPort }} }} }} }}')
            for p in ((d.get("pod") or {}).get("runtime") or {}).get("ports") or []:
                if p.get("privatePort")==22 and p.get("isIpPublic"): return p["ip"], p["publicPort"]
            time.sleep(10); print("  ...provisioning")
        return None, None

    # RESUME: if a training pod is recorded + still running, poll it (no new pod)
    pid = None; ip = port = None; resumed = False
    if pidfile.exists():
        rid = pidfile.read_text().strip()
        try:
            d = gql(key, f'query{{ pod(input:{{podId:"{rid}"}}){{ desiredStatus }} }}')
            if (d.get("pod") or {}).get("desiredStatus") == "RUNNING":
                pid = rid; print(f"RESUME training pod {pid}")
                ip, port = ports_for(pid); resumed = bool(ip)
        except Exception: pass
    if not pid:
        print(f"creating {a.gpu} pod...")
        # SECURE cloud: no preemption (community/ALL pods were getting killed mid-train)
        m = gql(key, f"""mutation{{ podFindAndDeployOnDemand(input:{{
            cloudType: SECURE, gpuCount:1, volumeInGb:0, containerDiskInGb:{a.disk},
            minVcpuCount:8, minMemoryInGb:64, gpuTypeId:{json.dumps(a.gpu)},
            name:"iso-stl-train", imageName:{json.dumps(a.image)},
            ports:"22/tcp", startSsh:true, volumeMountPath:"/workspace" }}){{ id }} }}""")
        pid = m["podFindAndDeployOnDemand"]["id"]
        pidfile.write_text(pid)
        print(f"pod {pid} (terminate: python3 cloud/runpod_terminate.py {pid})")
    try:
        if not ip: ip, port = ports_for(pid)
        if not ip: raise SystemExit("no SSH")
        print(f"SSH root@{ip}:{port}")
        from scp import SCPClient
        if not resumed:
            c = ssh_connect(ip, port)
            if not c: raise SystemExit("ssh connect failed")
            print("uploading...")
            with SCPClient(c.get_transport()) as scp:
                for f in files: scp.put(str(HERE/f), f"/workspace/{f}")
            # START DETACHED (nohup). Do NOT read the channel — with the bg process
            # the channel can stay open and block o.read(). Fire and move on.
            ch = c.get_transport().open_session()
            ch.exec_command(f"cd /workspace && nohup bash {a.setup} > /workspace/train.log 2>&1 < /dev/null & disown; echo OK")
            time.sleep(3)
            try: ch.close()
            except Exception: pass
            print("training started detached; polling...")
            c.close()
        else:
            print("resumed; polling existing training...")

        target = f"/workspace/output/{a.output_name}/{a.output_name}.safetensors"
        deadline = time.time() + a.max_hours*3600
        done = False; fatal = False; misses = 0
        while time.time() < deadline:
            time.sleep(a.poll)
            try:
                c = ssh_connect(ip, port, tries=3)
                if not c:
                    # transient unreachable — only give up after it's EXPLICITLY dead twice
                    d = gql(key, f'query{{ pod(input:{{podId:"{pid}"}}){{ desiredStatus }} }}')
                    st = (d.get("pod") or {}).get("desiredStatus")
                    print(f"  poll: unreachable, status={st}")
                    if st in ("EXITED", "TERMINATED"):
                        misses += 1
                        if misses >= 2: print("  -> pod confirmed dead"); break
                    continue
                misses = 0
                ls, _ = run(c, f"ls -la {target} 2>/dev/null", timeout=30)
                tail, _ = run(c, "tr '\\r' '\\n' < /workspace/train.log 2>/dev/null | grep -aE 'iso_stl_omni_v2:|Error running|Failed to import|driver too old' | tail -1", timeout=40)
                try: c.close()
                except Exception: pass
                print(f"  poll {time.strftime('%H:%M')}: {tail.strip()[:70] or 'starting...'}")
                if ls.strip() and "safetensors" in ls:
                    done = True; print("  -> safetensors present"); break
                if "Error running" in tail or "Failed to import" in tail or "driver too old" in tail:
                    fatal = True; print("  -> fatal error in train.log"); break
            except Exception as e:
                # transient poll error MUST NOT kill a healthy training pod — just retry
                print(f"  poll error (continuing, pod left running): {str(e)[:60]}")
                continue
        if done:
            print("downloading LoRA...")
            out_local.mkdir(parents=True, exist_ok=True)
            c = ssh_connect(ip, port)
            with SCPClient(c.get_transport()) as scp:
                scp.get(target, str(out_local/f"{a.output_name}.safetensors"))
            c.close()
            print(f"LoRA -> {out_local}")
        else:
            print("training did NOT produce output within window")
    finally:
        try:
            gql(key, f'mutation{{podTerminate(input:{{podId:"{pid}"}})}}')
            print(f"[terminate] {pid}")
            try: pidfile.unlink()
            except Exception: pass
            time.sleep(5)
            rem = gql(key, "query{myself{pods{id}}}").get("myself",{}).get("pods",[])
            print(f"[verify] remaining: {[p['id'] for p in rem] if rem else 'NONE'}")
        except Exception as e:
            print(f"[terminate] FAILED {str(e)[:80]} — MANUAL: python3 cloud/runpod_terminate.py {pid}")
    sys.exit(0 if (out_local/f"{a.output_name}.safetensors").exists() else 1)

if __name__ == "__main__":
    main()
