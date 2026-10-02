# Tile submissions

Put styled tiles for one region in a subdirectory here and open a pull request.

```
submissions/<region-name>/tile_0_0.png
submissions/<region-name>/tile_1_0.png
submissions/<region-name>/NOTES.md
```

`NOTES.md` should name the column and row range you captured, the LoRA file, the step count, and the command you ran.

Rules:

- Only `tile_<column>_<row>.png`, 1024×1024, plus a short `NOTES.md`.
- One region per pull request. Keep it under about 50 tiles.
- Do not add LoRA weights, raw aerial packs, mosaics, `.env` files, or API keys.
- The maintainer reviews the style, then runs the shared stitch before anything reaches the live map.

See [CONTRIBUTING.md](../CONTRIBUTING.md).
