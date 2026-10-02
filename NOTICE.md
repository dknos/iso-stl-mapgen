# Provenance & terms

The LoRA weights distributed via GitHub Releases were trained to restyle ortho aerial
captures of St. Louis into an isometric low-poly diorama look, to build and extend the
map at [stlcity2000.com/iso-map/corridor-traffic-fp](https://stlcity2000.com/iso-map/corridor-traffic-fp).
They are shared for research and hobbyist use.

- **Base model:** [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509).
  Respect its model license. The LoRA is an adapter and requires the base model to run.
- **Training inputs:** ortho captures derived from aerial/3D map imagery, stylized with an
  image model. The released weights are adapters, not redistributions of any source imagery.
  If you re-capture source tiles yourself, follow the terms of whatever imagery provider you use.
- **Not affiliated** with Google, Alibaba, or Qwen.

The tooling in this repo is MIT licensed. See [LICENSE](LICENSE). The Qwen base model
keeps its own license.
