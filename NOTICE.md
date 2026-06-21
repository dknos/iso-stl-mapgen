# Provenance & terms

The LoRA weights distributed via GitHub Releases were trained to restyle ortho aerial
captures of St. Louis into an isometric low-poly diorama look, to build and extend the
map at [stl-radar.com](https://stl-radar.com). They are shared for research and hobbyist
use.

- **Base model:** [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).
  Respect its model license. The LoRA is an adapter and requires the base model to run.
- **Training inputs:** ortho captures derived from aerial/3D map imagery, stylized with an
  image model. The released weights are adapters, not redistributions of any source imagery.
  If you re-capture source tiles yourself, follow the terms of whatever imagery provider you use.
- **Not affiliated** with Google, Alibaba, or Qwen.

The Python tooling in this repo is provided as-is. Add your preferred OSS license here if you
intend others to reuse the code (MIT/Apache-2.0 are common choices).
