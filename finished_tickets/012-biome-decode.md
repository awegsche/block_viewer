# 012 - Decode: per-section biome grids

## Status
Open

## Depends on
002 (decode layer). Independent of 011.

## Goal

Every decoded `ChunkSection` carries the biome of each of its 4x4x4
sub-cubes, interned into a shared `BiomeRegistry` the same way block names
go through `BlockRegistry`. No rendering changes — this is the data 013
needs, decoded and tested on its own.

## The NBT format

`biomes` is a sibling of `block_states` inside each entry of `sections`.
Same palette + packed-data shape, with **three differences** that will each
silently produce garbage if missed:

1. **The palette is a list of strings, not a list of compounds.**
   `block_states.palette` is `TAG_List<TAG_Compound>` where each entry has a
   `Name` (and optionally `Properties`). `biomes.palette` is
   `TAG_List<TAG_String>` — the biome name *is* the entry. In `rnbt` terms
   that's `NbtList::String(..)`, not `as_compound_list()`.

2. **There is no 4-bit minimum.** Ticket 001 was about `block_states`
   enforcing `max(4)` on the bit width. Biomes do not: the width is
   `((len - 1).ilog2() + 1).max(1)`. A 2-entry biome palette packs at
   **1 bit**, 16 per section is 4 bits, and using `.max(4)` here would
   misread every multi-biome section.

3. **The grid is 4x4x4, not 16x16x16.** 64 entries per section, one per
   4x4x4 block cube, indexed `x + z*4 + y*16` — the same X-fastest, Y-slowest
   order `block_states` uses, just at a quarter resolution per axis.

Unchanged from `block_states`: indices never span a long (leftover high bits
are padding), and a single-entry palette omits `data` entirely.

## Work

### `src/world/biome.rs` (new)

`BiomeId(pub u16)` + `BiomeRegistry`, a near-copy of `src/world/block.rs`.
Reserve `BiomeId(0)` for `"minecraft:plains"` (interned by
`BiomeRegistry::new`, mirroring how `BlockRegistry` reserves air) so every
consumer has a total function — a section with no biome data reads back as
plains rather than forcing an `Option` through the mesher.

Consider factoring the intern/name/len logic the two registries share, but
only if it comes out clean; two 60-line registries are not a problem worth
a generic for.

### `src/world/decode.rs`

- `ChunkSection` gains `biomes: Box<[BiomeId; 64]>`.
- `ChunkSection::biome_at(x, y, z)` taking **block** locals (0..16 each) and
  indexing `(x/4) + (z/4)*4 + (y/4)*16`. Callers work in block coordinates;
  the division belongs here, once, not at every call site.
- `decode_chunk` takes `&mut BiomeRegistry` as a second registry parameter
  and decodes `biomes` per section.
- A section with no `biomes` compound, or an empty palette, fills with
  `BiomeRegistry::PLAINS` and logs once (use the warn-once
  `HashSet<String>` pattern from `atlas::build_block_uv_table`, not a
  per-section `println!`).

### What about skipped sections?

`decode_chunk` currently `continue`s past uniform-air sections and sections
with no `block_states` before storing anything. Keep that: those sections
render nothing, so their biome is never sampled. Say so in a comment —
otherwise it reads like an oversight rather than a decision.

### Callers to update

`decode_chunk`'s signature change reaches:

- `chunk_pipeline::load_and_mesh_chunk` — needs an
  `Arc<Mutex<BiomeRegistry>>` alongside the existing registry. Follow
  exactly what `BlockRegistry` does: it lives in `DecodedWorld`, is cloned
  into the task, and is locked for the duration of decode+mesh. The module
  docs' note about serialising background tasks on the registry lock
  applies unchanged; don't introduce a second lock ordering, take both
  locks in one place at the top of `load_and_mesh_chunk`.
- `DecodedWorld` in `main.rs` — add `biomes: Arc<Mutex<BiomeRegistry>>`.
- `decode.rs`'s own tests, and `mesh.rs`'s test helpers that build
  `ChunkSection` literals.

## Memory

64 x 2 bytes = 128 B per section against 4096 x 2 = 8 KB of blocks. 1.5%.
Not worth a packed representation; note it in the status panel's byte
estimate (`src/ui/status.rs`) or deliberately don't, but don't leave the
estimate silently wrong by more than it already is.

## Nice-to-have, cheap here

The block inspector (`src/ui/block_inspector.rs`) already resolves a block
under the cursor out of `DecodedWorld`. Adding a "Biome: minecraft:plains"
line is a handful of lines once the data exists, and it is the fastest way
to sanity-check that decoding is correct against a save you know. Do it.

## Tests

In `src/world/decode.rs`:

- uniform (single-entry) biome palette, no `data` → all 64 entries set;
- 2-entry palette at **1 bit** — the case a `.max(4)` would break;
- a larger palette (say 5 entries → 3 bits) reading the right entry at a
  known index;
- a section with no `biomes` compound → all plains, no panic;
- `biome_at` maps block locals to the right 4x4x4 cell: `(0,0,0)`,
  `(3,3,3)` and `(4,0,0)` are cells 0, 0 and 1 respectively;
- a chunk from the real save decodes with a plausible biome set (mirror
  `chunk_pipeline`'s `load_and_mesh_chunk_decodes_and_meshes_a_real_chunk`
  convention — resolve the region centre, assert the biome registry is
  non-trivial afterwards).

## Done when

- `cargo test` passes, including the 1-bit palette case.
- The block inspector shows the biome under the cursor.
- Nothing renders differently yet.
