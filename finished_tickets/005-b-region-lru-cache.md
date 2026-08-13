# 005-b - Region LRU cache

## Status
Done — implemented in `src/region_cache.rs`.

## Part of
[005 - Stream chunks around the camera](005-chunk-streaming.md).

## Depends on
005-a (chunk coords to translate into region coords), 002 (decode layer).

## Problem

Region files are the I/O unit (32x32 chunks): opening `r.x.z.mca` and
inflating it is per-region work, not per-chunk. Streaming individual chunks
naively would re-open/re-parse the same region file repeatedly unless
decoded regions are cached and shared across every chunk that needs them,
and evicted once nothing does.

## Goal

A `RegionCache` that, given a region coordinate, returns an already-decoded
region (loading + parsing it on first access), with an LRU policy that
evicts and frees the least-recently-used region once the cache exceeds a
size budget.

## Scope

- Chunk-coord -> region-coord conversion (Anvil regions are 32x32 chunks —
  check whether `ranvil` already exposes this before writing it again;
  `ChunkRegion` already carries `region.get_x_coord()`/`get_z_coord()`).
- A `RegionCache` struct with `get_or_load(region_coord) ->
  Result<&ChunkRegion, _>`, backed by an LRU eviction policy (a small
  hand-rolled one over a `HashMap` + access-order tracking is fine; only
  reach for the `lru` crate if that turns out simpler — check `Cargo.toml`
  conventions before adding a new dependency).
- Capacity sized to comfortably cover one render-distance's worth of regions
  (e.g. `ceil(render_distance / 32) + 1` per axis) — pass it in rather than
  hardcoding, so 005-a's `RenderDistance` can size it later.
- Eviction actually drops the region's decoded chunk data (`ChunkRegion`'s
  `chunks: Option<Vec<Option<NbtField>>>`), not just unlinks a pointer to
  it — confirm with a measured RSS drop, not just by reading the `Drop`
  impl.
- This cache is synchronous for now; 005-c is what moves calls into it onto
  a background task.

## Watch out

`ChunkRegion::load_chunks()` parses **all 1024 chunks** of a region in one
call — there's no per-chunk lazy path. Measure how long that takes against
a real region from the loaded save (a `println!` with `Instant::now()` is
enough) and record the number in this ticket's notes when done: it decides
whether 005-c can hand a whole region to one background task or needs to
shard the work. If it's too slow, that's an upstream ticket against
`../ranvil` for a per-chunk load path, not something to solve here.

## Out of scope

Async execution (005-c). Feeding decoded chunks to the mesher or the
streaming system (005-c/005-d).

## Done when

- [x] `get_or_load` returns correct chunk data for a region coordinate
  against the real save directory.
- [x] A capacity-exceeding sequence of accesses evicts the actual
  least-recently-used region, verified by a test or a log.
- [x] The single-region load+decode timing from "Watch out" is measured and
  written down (in this file or the commit message) for 005-c to use.

## Notes

- Measured on the dev machine, real save (`nbt_test`), region `(-1, -1)`:
  **`ChunkRegion::load_chunks()` took ~153ms for 1024 chunk slots**
  (`region_cache::tests::measure_single_region_load_time`, run with
  `cargo test --bin block_viewer region_cache -- --nocapture`). That's a
  chunky unit of work for one background task (005-c) but not pathological
  — no upstream `../ranvil` ticket needed for a per-chunk load path yet.
- No `chunk-coord -> region-coord` helper existed in `ranvil` (only
  filename <-> region-coord parsing) — added `chunk_to_region_coord` in
  `region_cache.rs` rather than upstream, since it's a one-line
  `div_euclid`.
- `RegionCache`, `chunk_to_region_coord`, and `recommended_capacity` aren't
  called anywhere yet (`#![allow(dead_code)]` on the module) — 005-c wires
  the cache into the async pipeline, 005-e wires that into `main.rs`.
