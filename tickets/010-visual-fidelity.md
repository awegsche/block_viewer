# 010 - Visual fidelity: block models, transparency, baked light, AO

## Status
Open — optional / later

## Depends on
004, and realistically 009.

## Why this is separate

Everything here makes the world look *right* rather than making it *work*.
None of it blocks exploring a save, and each piece is large enough to be its
own ticket when it's actually picked up. Listed so it isn't confused with
core work — split this file up rather than doing it as one lump.

## Already split out

Biome tinting became real tickets **011–014** (vertex colour channel, biome
decode, biome tint, cutouts + grass overlay); sky and sun became **015–018**.
See `ROADMAP.md`. What's left below is what those didn't take.

## Candidate work

### Real block models
Parse `assets/minecraft/models/block/*.json` `elements` for non-cube
geometry: stairs, slabs, fences, walls, doors, torches, rails, plants.
Requires blockstate variant resolution
(`assets/minecraft/blockstates/*.json` maps `Properties` → model + rotation).
This is the natural continuation of ticket 004's Pass 2 and is a large piece
of work on its own.

Note that `world::decode` currently discards each palette entry's
`Properties` entirely (it only reads `Name`), so variant resolution needs a
decode-layer change before it needs a model parser.

### Transparency
Water, glass and ice need a genuinely alpha-*blended* material, back-to-front
sorting per chunk, and the rule that adjacent same-type transparent blocks
don't emit faces between themselves (or water looks like stacked panes).

Ticket 014 gets alpha *masking* (cutouts) for free with a single material;
blending is the harder problem and is what forces the opaque/transparent
mesh split that 009 also lists. Expect 014's grass overlay quads to move
into that pass when it happens.

Leaves are a half-case: 014 makes their holes see-through but leaves them
occluding neighbours, so you see through a leaf into geometry that was
culled. Matching vanilla's "fancy" leaves means treating leaves as
non-occluding against other leaves — a `mesh::NON_SOLID`-adjacent category
and a real vertex-count increase.

### Lighting (baked)
Chunk NBT carries `BlockLight`/`SkyLight` byte arrays per section (a nibble
per block, 2048 bytes each). Baking them into vertex colours gives
Minecraft's characteristic look far more cheaply than real-time lighting —
and it's what makes caves and building interiors dark, which 015's
directional sun does not.

This composes with, rather than replaces, the sun rig: it multiplies into
the **same** vertex colour channel ticket 011 adds, alongside 013's biome
tint. Read 011's documented composition rule before starting.

### Ambient occlusion
The classic per-vertex block AO (count solid neighbours around each vertex)
is cheap, folds into 011's vertex colour channel alongside tint and baked
light, and does more for perceived quality than anything else on this list.

## Done when
N/A — split into real tickets when picked up.
