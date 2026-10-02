# Security

This repository is the public way to generate tiles for the St. Louis isometric map. The live site and deploy keys stay with the maintainer. Capture uses your own Google 3D Tiles key.

## Do not commit

- RunPod, Modal, Hugging Face, Cloudflare, or Google API keys
- `.env`, `~/.iso_runpod_env`, private keys, or service-account JSON
- LoRA or base-model weights (those stay on the GitHub Release and on Hugging Face)
- Your Google 3D Tiles key. Capture raw tiles the way [Isometric NYC](https://cannoneyed.com/projects/isometric-nyc) describes, and leave the key on your machine.

`cloud/runpod_*.py` reads `RUNPOD_API_KEY` from `~/.iso_runpod_env` on your machine. That file is gitignored. `HF_TOKEN` is an environment variable, never a file in this repo.

A pull request that adds a key-shaped file, a private key, or an image outside `submissions/` is rejected by `.github/workflows/contribution-guard.yml`.

## Modal endpoints

`cloud/modal_omni_server.py` and `cloud/modal_omni_packed.py` bill the Modal account that deploys them. They refuse to deploy unless `ISO_EDIT_TOKEN` is set to at least 16 characters, and they reject requests that do not send that token back. Generate it locally:

```bash
export ISO_EDIT_TOKEN=$(openssl rand -hex 24)
```

Keep the token and the endpoint URL private. `scripts/walk_grid_rect.py` attaches the token when that variable is set. The local ComfyUI shim listens on `127.0.0.1` only.

## What contributors cannot do

Direct pushes to `master` are closed. Tile pull requests are reviewed by the maintainer before any stitch onto https://stlcity2000.com/iso-map/corridor-traffic-fp. There is no workflow in this repo that deploys the map.

Maintainer-only scripts such as `scripts/stitch_deploy.py` contain local filesystem paths from the original build machine. They do not contain credentials. Do not point them at a machine that is not yours.

## Reporting

Open a private security advisory on this repository, or email the address on the [dknos](https://github.com/dknos) profile. Do not file a public issue that includes a key or an endpoint URL.
