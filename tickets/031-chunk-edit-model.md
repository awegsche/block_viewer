# 031 - The chunk edit model (roadmap W4)

## Status
Open

## Depends on
Nothing outstanding. This is the ticket the whole write path was waiting for,
and what it was waiting *on* has landed:

| needs | state |
|---|---|
| `rnbt` mutation API (W1) | done — `get_mut`, `insert`, `remove`, `as_*_mut` |
| region file writer, atomic replace (`ranvil` 009) | done |
| mutable chunk access + dirty tracking (010) | done — `get_chunk_mut`, `is_dirty`, `dirty_chunks` |
| `set_block`/`set_blocks`, palette insert + re-pack (011) | done, batched and all-or-nothing |
| orphaned `block_entities` / tick cleanup (015) | done, **inside `set_blocks`** |
| relight flag (014) | done, **inside `set_blocks`** |
| `Heightmaps` delete *and* recompute (013) | done |
| `session.lock` detection (016) | done — `SaveMeta::is_locked`, `SessionLock::acquire` |

## What this ticket is

The **policy** layer, and only that. `ranvil` owns the Anvil format; `edit` owns
what we are allowed to do to a world. The split is the reason both stay
readable.

Concretely: a new `src/edit/` module (per the roadmap's layout), shared by the
viewer and the citybuilder, that answers "is this edit well-formed, and in what
order do the pieces happen" — and applies it to **one region**. Routing an edit
across several region files, and the `RegionCache` mutation that needs, is W5.
Backup/lock/atomicity is W6. A UI for it is W8.

That boundary is what makes this testable on its own: an `apply` that takes a
`&mut ChunkRegion` can be driven end to end against a temp copy of a real region
file, with no cache, no Bevy app and no window.

## The sequencing is shorter than the roadmap thought

The roadmap's W4 says: *block changes → block-entity cleanup (015) → heightmap
invalidation (013) → relight flag (014) → write (009)*.

Three of those five now happen **inside `ranvil::set_blocks`**: it cleans up
orphaned `block_entities`/`block_ticks`/`fluid_ticks` in its own pass and clears
`isLightOn` per touched chunk, both after the block writes and both only if the
batch was accepted. So what's left for this layer is:

1. **Preflight** — validate the whole edit, mutating nothing.
2. **`set_blocks`** — one call per region, which is already all-or-nothing.
3. **Heightmaps** — per touched chunk, *after* the last block write.

Step 3 is the one with a trap in it, and it's why heightmap policy belongs here
rather than in `ranvil`: it must run once per chunk at the **end of the edit
transaction**, not once per `set_blocks` call. Two edits landing in the same
chunk, each recomputing heightmaps, means the first recompute is thrown away —
harmless — but a *delete* followed by a *recompute* over stale intermediate
state is not. One pass, at the end.

## Design

### 1. The value type, in Minecraft coordinates only

```rust
// src/edit/mod.rs
pub struct BlockEdit { pub at: IVec3, pub state: BlockState }   // blueprint::BlockState
pub struct WorldEdit { edits: Vec<BlockEdit>, data_version: Option<i32> }
```

`at` is **Minecraft world coordinates**, everywhere, with no exceptions. The
`bevy.z = -mc.z` flip lives at the rendering boundary (`world::mesh`,
`selection::gizmo`) and must not leak in here — that flip is exactly how
mirrored buildings happen, and 019 already fixed the rules for `SelectionBounds`
in mc coordinates. Inherit them; don't reinvent them. `selection::{WORLD_MIN_Y,
WORLD_MAX_Y}` are the build limits and are already the numbers 020/021 clamp to.

Two decisions the type makes, so nothing downstream has to:

- **Air is a block.** A `WorldEdit` carrying `minecraft:air` at a position
  writes air there. Whether a building's declared-empty space should be cleared
  or left as terrain is the *blueprint* layer's question (roadmap B2/E4), not
  this one. Filtering here would make "place a building with a hollow interior"
  impossible to express.
- **Last write wins**, and the order is the edit's own. `set_blocks` already
  applies in iteration order within a section, so this is documenting reality
  rather than adding a rule — but a caller stamping two overlapping blueprints
  needs it stated.

### 2. Preflight: every reason to refuse, found before anything is written

```rust
pub enum EditRefusal {
    OutsideBuildLimits { at: IVec3 },
    ChunkNotGenerated { chunk: (i32, i32) },
    StatusNotFull { chunk: (i32, i32), status: String },
    DataVersionMismatch { save: i32, edit: i32 },
    SectionMissing { chunk: (i32, i32), section_y: i32 },
    Empty,
}
```

- **`Status` must be `minecraft:full`.** Writing into a partially generated
  chunk invites the generator to overwrite it later. `world::decode` already
  reads this field and has `DecodeError::NotFullyGenerated`; read it the same
  way rather than inventing a second check.
- **`DataVersion`.** Compare the edit's against the target chunk's and refuse on
  a mismatch. Block names and properties are not stable across versions, and
  `blueprint::extract` already records a `data_version` (with
  `FALLBACK_DATA_VERSION` for an all-air extraction, which should count as "no
  claim" rather than as a mismatch). Refuse by default, with an override the UI
  can offer — the failure mode is a building made of blocks that don't exist.
- **Ungenerated chunks are refused, not generated.** Iteration 1 does not
  generate terrain; the roadmap is explicit.
- **Missing sections**: `ChunkRegion::check_set_block` answers this without
  writing. Note that a generated 1.18+ chunk carries all 24 sections (the probe
  in `ranvil` 013 confirmed Y -4..=19 on the real save), so this is a backstop
  rather than a common path — `ranvil` 012 decided writes never create sections.

Preflight is a separate pass over the whole edit, not a fail-fast inside the
apply, because a half-refused edit is the thing this layer exists to prevent.
`set_blocks` is all-or-nothing per call for the same reason; this extends that
guarantee across the checks `ranvil` can't make.

### 3. Heightmap policy

```rust
pub enum HeightmapPolicy { Delete, Recompute, Leave }
```

**Default `Delete`** (`ChunkRegion::remove_heightmaps`), per the roadmap's "let
the game rebuild them", same reasoning as lighting. `Recompute` exists because
`ranvil` 013 landed the other half; taking it needs a classifier — a per-block-
name `HeightmapClass` table built once against `world::BlockRegistry`, following
the pattern `world::tint::build_block_tint_table` already establishes rather
than inventing a second one. **Don't build that table until the in-game check in
`../todo.md` says it's needed**; the point of having both is that the answer
costs one enum variant.

`Leave` is for a caller that knows its edit can't move a surface, and for tests.

Note the interaction with ticket 030: the citybuilder's render floor reads
`OCEAN_FLOOR`, so under `Delete` an edited chunk temporarily has no floor data
and renders in full. 030 already specifies that fallback; nothing to do here
beyond knowing it's deliberate.

### 4. The dry run is the same code path

```rust
pub struct EditReport {
    pub blocks_written: usize,
    pub chunks: Vec<(i32, i32)>,
    pub regions: Vec<(i32, i32)>,
    pub replaced: Option<Vec<(IVec3, BlockState)>>,   // see 5.
}

pub fn plan(edit: &WorldEdit, region: &ChunkRegion, policy: &EditPolicy)
    -> Result<EditReport, EditRefusal>;
pub fn apply(edit: &WorldEdit, region: &mut ChunkRegion, policy: &EditPolicy)
    -> Result<EditReport, EditRefusal>;
```

`plan` is preflight plus counting; `apply` is `plan` followed by the two
mutating steps. The roadmap files the dry run under W6, but it costs nothing
here and it is the test harness for everything else — every refusal test is a
`plan` call, with no temp files at all.

`apply` deliberately does **not** call `region.save()`. Saving is where backup,
`session.lock` and atomicity live (W6), and a function that both edits and
writes to disk can't be tested without one.

### 5. Capture what was there — the one thing that can't be added later

`EditReport::replaced` is the **as-built baseline** (roadmap I1) and D3's undo
record and E5's terrain restore, and it is knowable only *while the edit is
being applied*. Afterwards the old blocks are gone from the save and
unrecoverable — the roadmap calls this out as the single piece of group I that
can't be retrofitted.

Reading it is one `get_block` per edited position, so it's opt-in
(`EditPolicy::capture_replaced`) rather than always on. **Ship the plumbing now
even though nothing consumes it yet**; the consumer (E4) is several tickets away
and this is the only place the data exists.

### 6. What this hands to W5, and the corruption trap in the handoff

W5 groups edits by region file, applies each once, and gives `RegionCache` a
mutable path. Two things to carry across, written here because they're this
layer's knowledge:

- **A dirty region must not be evicted.** `RegionCache::get_or_load` evicts LRU
  with no notion of unsaved changes; evicting an edited region silently drops
  the edit. `ChunkRegion::is_dirty` exists precisely for this — its doc comment
  in `ranvil` names this cache by name.
- **Region boundaries are a routing problem, not a placement one.** A building
  near a region corner spans up to four files. `chunk_to_region_coord` and
  `local_chunk_index` already do the arithmetic `blueprint::extract` uses for
  reads; the write path uses the same two functions rather than a second copy.

## Explicitly not in this ticket

- **Multi-region routing and `RegionCache` mutation** — W5.
- **Saving, backups, `session.lock`, atomic write** — W6. `apply` mutates the
  in-memory region and stops.
- **Re-meshing edited chunks** — W7. Nothing here touches `DecodedWorld` or the
  chunk pipeline.
- **Any UI** — W8's fill/stamp command is what proves this end to end.
- **Writing block entities.** A blueprint's chests and signs place their blocks;
  the entities are dropped, with a warning. Removing *existing* ones we overwrite
  is already handled inside `set_blocks` — that's corruption avoidance, not a
  feature.
- **Terrain generation.** Ungenerated chunks are refused (item 2).

## Tests

Pure, no files:

- Routing: a world coordinate → `(region, chunk, region-local x/z, world y)`,
  including negative coordinates and both sides of a region boundary. `-1 → r.-1`
  and local 511, not 1 — the `div_euclid` trap.
- Every `EditRefusal` variant, from a synthetic chunk NBT: Y over 319 and under
  -64, `Status = "minecraft:features"`, a `DataVersion` mismatch, an empty edit.
- An all-air extraction's `FALLBACK_DATA_VERSION` does **not** count as a
  mismatch.
- Last-write-wins for two edits at the same position.

Against a temp copy of a real region file (precedent:
`blueprint::structure`'s `writes_a_real_box_from_the_save`, which already
copies out of the real save into `%TEMP%`):

- A small edit applies, and `get_block` reads the new blocks back — the
  round-trip through `save()` + reload, driven by the test rather than by
  `apply`.
- **A refused edit changes nothing**: no block moved, `is_dirty()` is false, and
  the file on disk is byte-identical. This is the single most important test
  here.
- `HeightmapPolicy::Delete` removes the compound from exactly the touched
  chunks; `Recompute` (with a stub classifier) leaves `WORLD_SURFACE` matching
  the new blocks; `Leave` leaves them stale. One chunk each, and check an
  untouched neighbour in the same region is unaffected in all three.
- `isLightOn` is 0 on touched chunks and 1 on their untouched neighbours — the
  cross-crate assertion that `ranvil` 014 is actually wired in, which no test in
  this repo makes yet.
- `capture_replaced` returns the pre-edit states, and re-applying them restores
  the original blocks. That's the undo path exercised before anything depends
  on it.

## Done when

- `plan` and `apply` exist, `apply` is provably all-or-nothing, and a real
  region file survives an edit + save + reload with the right blocks in it.
- Nothing in the viewer or the citybuilder behaves differently — this ticket
  adds a module and calls it from tests only. W8 is the first caller.
- The heightmap policy default is `Delete`, and swapping it is one line.

## Resolution
