# 022 - Blueprint extraction: the selected volume into a palette + block array

## Status
Open

## Depends on
019 (`SelectionBounds`). Independent of 020/021 — a test can build bounds
directly.

## Goal

Turn a `SelectionBounds` into an in-memory `Blueprint`: a palette of
distinct block states and a dense array of indices into it. Reading only —
nothing about files (023) or dialogs (024) belongs here.

## The one decision that matters: where the blocks come from

**Not `DecodedWorld`.** This is the ticket's central constraint and the
reason it's separate from 023.

`world::BlockRegistry::intern` is called with only the palette entry's
`Name` (`src/world/decode.rs`, the `get_string("Name")` call). The
`Properties` compound is dropped on the floor — the mesher never needed it.
So a blueprint built from `DecodedWorld` would turn every stair, slab, log,
fence, door, and repeater into its default state: geometry preserved,
orientation destroyed. For a "blueprint" that is a silent, total failure.

**Use the raw NBT path instead**, the same one the block inspector (007)
goes through: `SharedRegionCache` → `RegionCache::get_or_load(region_coord)`
→ `ChunkRegion::get_block(local_x, y, local_z)`, which returns the palette
entry `&NbtField` with `Name` *and* `Properties` intact. See
`src/ui/block_inspector.rs::lookup_block` for the coordinate arithmetic
(region width = `REGION_WIDTH_IN_CHUNKS * world::SECTION_SIZE` = 512, with
`div_euclid`/`rem_euclid` for negative coordinates).

This also means **the selection is not limited to loaded chunks** — the
region cache loads from disk on demand, so a box can extend past the render
distance and still export correctly.

## Off the main thread

`ChunkRegion::get_block` walks the chunk's NBT fields by name on every call
(`get_list("sections")`, `get_path(["block_states", "palette"])`, …) —
that's fine for the inspector's one-block-per-frame use, and far too slow to
run a million times inside a frame. Extraction goes on `AsyncComputeTaskPool`
and is polled from a system, exactly like the chunk loads in
`src/chunk_pipeline.rs`:

- `SharedRegionCache` is already `Arc<Mutex<_>>` and already crosses this
  boundary (see `chunk_pipeline`'s "Send boundary" module docs — `NbtField`
  and `ChunkRegion` are plain owned data, no `unsafe impl` needed).
- One `Task<Result<Blueprint, ExtractError>>` at a time, held in a resource;
  the panel's export button is disabled while one is in flight.
- Poll with `block_on(poll_once(&mut task))` in an `Update` system, same as
  `poll_completed_chunk_loads`.

**Hold the cache lock for as short a span as possible, and never across the
whole extraction.** The lock is shared with every streaming chunk-load task;
holding it for a multi-million-block walk would stall terrain streaming
completely. Take it per region (or per chunk column) and release between.

### Iteration order

Iterate **by chunk column, then by block within it**, not naively by
`iter_blocks()` — a Y/Z/X walk across a wide box re-resolves the region and
chunk on nearly every block. Group the selection into the chunk columns it
overlaps, resolve each once, then sample the box's slice of that column.
Write the results into the dense array by 019's documented Y/Z/X index so
the output order is still the one 023 wants.

## The output

```rust
pub struct Blueprint {
    /// Selection size in blocks (X, Y, Z), Minecraft axes.
    pub size: IVec3,
    /// Where in the world this came from — the selection's `min` corner.
    /// Not written to the file (a structure is position-independent), but
    /// worth carrying for the UI and for a filename default (024).
    pub origin: IVec3,
    /// Distinct block states, in first-seen order. Each entry is the raw
    /// palette compound: `Name` plus optional `Properties`.
    pub palette: Vec<BlockState>,
    /// One palette index per block, `size.x * size.y * size.z` of them, in
    /// 019's Y-outer / Z-middle / X-inner order.
    pub blocks: Vec<u16>,
    /// The save's `DataVersion`, copied from a chunk in the selection.
    pub data_version: i32,
}

pub struct BlockState {
    pub name: String,
    /// Sorted by key so two identical states always dedupe to one palette
    /// entry regardless of NBT field order.
    pub properties: Vec<(String, String)>,
}
```

Notes:

- **Palette dedupe key** is `(name, sorted properties)`. NBT compound field
  order is not guaranteed stable, and an unsorted key would emit the same
  stair twice under two orderings — bloating the palette and, worse, making
  the output non-deterministic between runs. A `HashMap<BlockState, u16>`
  with a derived `Hash`/`Eq` over the sorted form does it.
- **`u16` indices** matches `world::BlockId`'s width; a selection with more
  than 65,535 *distinct* block states doesn't exist in practice. Return an
  error rather than truncating if it somehow happens.
- **`data_version`** comes from any chunk NBT inside the selection
  (`ChunkRegion::get_chunk(..).get_int("DataVersion")`). 023 needs it and
  guessing it wrong makes the file load incorrectly (or not at all) in a
  different game version. Fall back to a named `const` with a comment if no
  chunk in the box has one.

## Missing data is normal, not an error

A selection can legitimately cover chunks that were never generated, Y
values above the highest section, or a region file the save doesn't have.
Every one of those is **air**, not a failure — fill and continue. Only
propagate a real `ExtractError` for I/O and malformed-NBT failures, and even
then prefer "this column is air, and here's a count of columns that failed"
over aborting a large extraction because one chunk is corrupt. Report that
count to the UI.

One `mc_anvil` quirk to be aware of while doing this: `ChunkRegion::get_block`
indexes `sections` positionally (`sections.get(section_index)` after adding
`ZERO_OFFSET`), which assumes every chunk's section list starts at the world
bottom and is contiguous. If a save turns up where that doesn't hold, blocks
come back from the wrong height rather than erroring — cross-check a couple
of extracted blocks against the block inspector at known coordinates before
trusting a large export, and if it's genuinely wrong, fix it in `../ranvil`
(sibling crate — see `CLAUDE.md`) rather than working around it here.

## Progress

Large extractions take seconds. The task should publish a coarse progress
figure (columns done / columns total) through an `Arc<AtomicUsize>` for 021's
panel to show. Cheap, and the alternative is a UI that looks frozen.

## Out of scope

- Block entities (chest contents, sign text) and entities. The vanilla
  structure format has slots for both; leave them empty and note it. Chest
  contents in particular are a whole second NBT path (`block_entities` in
  the chunk root) and a natural follow-up ticket.
- Biomes. Also a structure-format field; also empty for now.
- Writing anything to disk (023).

## Tests

- Palette dedupe: two blocks with identical name and properties in a
  different NBT field order collapse to one entry.
- Distinct properties on the same name (two stairs, different `facing`) stay
  two entries — the regression test for the whole reason this doesn't read
  `DecodedWorld`.
- `blocks.len() == volume`, and a known block lands at the index 019's
  documented order predicts.
- Ungenerated chunks / out-of-range Y come back as `minecraft:air` and don't
  error.
- A selection spanning a chunk boundary and a region boundary, at negative
  coordinates (the `div_euclid` path), samples the right blocks.

Build these against a synthetic `NbtField` chunk the way
`src/world/decode.rs`'s tests already do (see its `new_compound` fixtures) —
no real save needed.

## Done when

- Extracting a selection returns a `Blueprint` whose palette contains the
  block names *and properties* visible in the block inspector at those
  coordinates.
- A multi-million-block extraction doesn't freeze the window or stall chunk
  streaming.
- `cargo test` passes.
- `todo.md` gets a manual check: select a small structure containing
  orientation-sensitive blocks (a staircase, a log wall, a door), extract,
  and confirm — via a debug log of the palette for now — that the properties
  match what the block inspector reports for those same blocks. Also
  extract a box straddling the loaded/unloaded boundary and confirm the
  unloaded part comes back as real blocks (loaded from disk), not air.
