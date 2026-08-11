# 010 - Visual fidelity: biome tint, real block models, transparency

## Status
Open — optional / later

## Depends on
004, and realistically 009.

## Why this is separate

Everything here makes the world look *right* rather than making it *work*.
None of it blocks exploring a save, and each piece is large enough to be its
own ticket when it's actually picked up. Listed so it isn't confused with
core work — split this file up rather than doing it as one lump.

## Candidate work

### Biome tinting
Grass, foliage and water textures ship as greyscale/neutral masks and are
tinted per biome at render time; without it, grass renders grey-green and
wrong. Needs the `biomes` palette from each section's NBT (sibling of
`block_states`, same palette+packed-data encoding but with a **1-bit**
minimum, not 4 — see ticket 001) plus
`assets/minecraft/textures/colormap/{grass,foliage}.png` indexed by
temperature/downfall.

### Real block models
Parse `assets/minecraft/models/block/*.json` `elements` for non-cube
geometry: stairs, slabs, fences, walls, doors, torches, rails, plants.
Requires blockstate variant resolution
(`assets/minecraft/blockstates/*.json` maps `Properties` → model + rotation).
This is the natural continuation of ticket 004's Pass 2 and is a large piece
of work on its own.

### Transparency
Water, glass, ice, leaves — needs an alpha-blended material, back-to-front
sorting per chunk, and the rule that adjacent same-type transparent blocks
don't emit faces between themselves (or water looks like stacked panes).

### Lighting
Chunk NBT carries `BlockLight`/`SkyLight` byte arrays per section. Baking
them into vertex colours gives Minecraft's characteristic look far more
cheaply than real-time lighting, and replaces the current single
`PointLight`.

### Ambient occlusion
The classic per-vertex block AO (count solid neighbours around each vertex)
is cheap, folds into the mesher's vertex-colour channel alongside lighting,
and does more for perceived quality than anything else on this list.

## Done when
N/A — split into real tickets when picked up.
