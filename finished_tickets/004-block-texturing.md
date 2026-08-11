# 004 - Block textures: name → atlas UV

## Status
Open

## Depends on
003.

## Goal

Each block renders with its own texture, per face, instead of every face
being `stone.png`.

## What we have

`assets/minecraft/` is a full extracted vanilla resource pack — the complete
`textures/block/*.png` set (16×16 each), plus `models/block/*.json` and
`blockstates/*.json`. `src/main.rs` already loads a single texture with
`ImageSampler::nearest()`, which is the right sampler; keep it.

## Scope — do it in two passes

### Pass 1: convention-based atlas (land this first)

- Build a texture atlas at startup from `assets/minecraft/textures/block/`
  (Bevy 0.15: `TextureAtlasBuilder`, or hand-pack into one image). Nearest
  filtering, and **padding/extrusion between tiles** or neighbouring textures
  will bleed at distance.
- Map `BlockId` → per-face atlas indices with a small explicit table plus
  conventions:
  - default: `<name>.png` (e.g. `minecraft:stone` → `stone.png`)
  - `<name>_top` / `<name>_bottom` / `<name>_side` when present
    (`grass_block_top`, `oak_log_top` + `oak_log`, …)
  - hardcode the handful of irregulars that matter visually.
- A loud fallback texture (magenta/black checker) for unmapped names, so gaps
  are obvious rather than silently stone-coloured.
- Mesher (003) writes atlas UVs per face instead of `[0,1]×[0,1]`.

### Pass 2: read the model JSONs (separate, later — split into its own ticket if it grows)

`models/block/*.json` gives real per-face texture assignments via `textures`
+ `elements`, with `parent` inheritance to resolve. That's the correct
general answer and removes the hardcoded table, but it is a meaningful
parser (serde + parent resolution + variable indirection like `#side`) and
should not block Pass 1.

## Out of scope

- Non-cube geometry (stairs, slabs, fences, torches, plants). Render them as
  full cubes for now; real model geometry is ticket 010 territory.
- Biome tinting of grass/foliage/water — the textures are greyscale masks and
  will look washed out until tinted. Also 010.
- Transparency and alpha sorting (glass, leaves) — 009/010.

## Done when

- Grass, dirt, stone, logs, ores and water are visually distinguishable in
  the rendered world, with correct top/side/bottom faces on grass and logs.
- Unmapped blocks render as the fallback checker, and their names are logged
  once each (not per block) so the mapping gaps are enumerable.
- No visible texture bleeding at the far edge of the render distance.
