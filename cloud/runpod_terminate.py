#!/usr/bin/env python3
"""Terminate a RunPod pod (stop billing). Usage: python3 runpod_terminate.py <podId> [<podId>...]
Or with no args, lists running pods."""
import json, sys, urllib.request
from pathlib import Path
key = [l.split("=", 1)[1].strip() for l in open(Path.home()/".iso_runpod_env") if l.startswith("RUNPOD_API_KEY")][0]
def gql(q):
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={key}",
        data=json.dumps({"query": q}).encode(),
        headers={"content-type": "application/json", "user-agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))
if len(sys.argv) < 2:
    d = gql("query{myself{pods{id name desiredStatus runtime{uptimeInSeconds}}}}")
    for p in (d.get("data", {}).get("myself", {}) or {}).get("pods", []) or []:
        print(p["id"], p.get("name"), p.get("desiredStatus"))
    sys.exit(0)
for pid in sys.argv[1:]:
    r = gql(f'mutation{{podTerminate(input:{{podId:"{pid}"}})}}')
    print(pid, "->", "terminated" if "errors" not in r else r["errors"])
