#!/usr/bin/env python3
"""Reject pull requests that add secrets or images outside submissions/."""
import os
import re
import subprocess
import sys

base = os.environ.get("BASE_SHA", "")
if not base:
    sys.exit("BASE_SHA is not set")

diff = subprocess.check_output(
    ["git", "diff", "--name-status", "--diff-filter=A", base, "HEAD"],
    text=True,
)
added = []
for line in diff.splitlines():
    parts = line.split("\t")
    if parts and parts[0].startswith("A"):
        added.append(parts[-1])

secret_name = re.compile(
    r"(^|/)\.env$|\.pem$|id_rsa|id_ed25519|\.safetensors$|\.token$|credentials\.json$|service-account",
    re.I,
)
image = re.compile(r"\.(png|jpe?g|webp)$", re.I)
key_line = re.compile(
    r"(RUNPOD_API_KEY|HF_TOKEN|CLOUDFLARE_API_TOKEN|ISO_EDIT_TOKEN|AWS_SECRET_ACCESS_KEY|AKIA[0-9A-Z]{16})\s*[:=]\s*\S{12,}"
)
bad = []
for path in added:
    if secret_name.search(path):
        bad.append(f"secret-like file: {path}")
        continue
    if image.search(path) and not path.startswith("submissions/"):
        bad.append(f"image outside submissions/: {path}")
        continue
    if not os.path.isfile(path):
        continue
    size = os.path.getsize(path)
    if size > 25 * 1024 * 1024:
        bad.append(f"file over 25MB: {path}")
        continue
    if image.search(path):
        continue
    text = open(path, errors="ignore").read()
    if "BEGIN OPENSSH PRIVATE KEY" in text or "BEGIN RSA PRIVATE KEY" in text or key_line.search(text):
        bad.append(f"possible secret in {path}")

if bad:
    print("\n".join(bad))
    sys.exit(1)
print(f"ok, {len(added)} added file(s)")
