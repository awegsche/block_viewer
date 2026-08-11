# 002 - Decode layer: chunk NBT → dense block grids

## Status
Open

## Depends on
001 (a correct bit width, or this ticket's tests bake in the bug).

## Goal

A module in this repo that turns a loaded `mc_anvil::ChunkRegion` into a
dense, render-friendly block representation, decoupled from `rnbt::NbtField`
and fast enough to feed a mesher.

## Why not just call `get_block`

`ChunkRegion::get_block` is a *single-block* API: per call it re-walks the
NBT tree, re-resolves the section, and re-reads the palette. Meshing one
chunk column touches 16×16×384 ≈ 98k blocks; a whole region is ~100M. Doing
that through `get_block` means re-walking the NBT tree 100M times.

Decode each **section** once into a flat array instead. `get_block` stays
useful for one-off lookups (block-under-cursor readout in ticket 007) — this
ticket does not replace it.

## Scope

New module `src/world/` (suggested: `mod.rs`, `block.rs`, `decode.rs`).

- `BlockState { name: String, properties: Option<...> }` — or, better, intern
  names into a `BlockId(u16)` via a global `BlockRegistry`, so the grids are
  `Vec<BlockId>` and comparisons are integer compares. Ticket 004 will want
  that id as its texture-lookup key too.
- `ChunkSection { y: i8, blocks: Box<[BlockId; 4096]> }` with `x + z*16 + y*256`
  indexing (matching the NBT layout: index = `dy*256 + dz*16 + dx`).
- `ChunkColumn { x: i32, z: i32, sections: Vec<ChunkSection> }`.
- `fn decode_chunk(nbt: &NbtField, registry: &mut BlockRegistry) -> Result<ChunkColumn, _>`.

### Decoding rules (verified against the real save `nbt_test`, DataVersion 4438)

- Sections live at `sections` (a compound list); each has a **`Y` byte**
  (signed, e.g. -4..19). Select/place sections by that `Y` field, **not** by
  list position — see ticket 001's note.
- Palette is at `block_states.palette`, a compound list of entries with a
  `Name` string (`"minecraft:deepslate"`) and optional `Properties`.
- **Palette length 1 ⇒ no `block_states.data` field at all**; the whole
  section is that one block. Very common: in the probed chunk, sections
  Y=4..19 are all single-entry `minecraft:air`. Fast-path these — a
  single-entry *air* section should decode to "empty" and cost nothing.
- Bits per index = `max(4, ilog2(palette.len() - 1) + 1)` (ticket 001).
- Indices **do not span longs** (1.16+ format): `64 / bits` indices per long,
  leftover high bits of each long are padding and must be ignored.
- Skip chunks whose `Status` is not `minecraft:full` (partially generated).
- `xPos`/`zPos` are ints on the chunk root giving *world* chunk coords
  (e.g. -32, -32 for region (-1,-1) — region chunk `(cx, cz)` maps to world
  chunk `(region_x*32 + cx, region_z*32 + cz)`).

## Out of scope

- Meshing (003), textures (004), block entities, entities, heightmaps,
  lighting data, biomes.
- Pre-1.18 chunk formats. Target modern saves only, and return a clear error
  on anything older rather than guessing.

## Done when

- `decode_chunk` produces a `ChunkColumn` for a real chunk from the local
  save, and decoding one full region's chunks completes in reasonable time
  (log the timing; it should be well under a second per chunk column).
- Unit tests cover: single-entry air section, a small palette (2-8 entries,
  the ticket-001 case), a >16 entry palette (5-bit), and a section list with
  a gap / unexpected `Y` ordering.
- A temporary startup log prints e.g. the topmost non-air block of a few
  chunk columns, confirming names look plausible (`minecraft:grass_block`
  and friends, not garbage).
