# 005-c - Async load/decode/mesh pipeline

## Status
Implemented in `src/chunk_pipeline.rs`, `cargo build`/`cargo test` clean
(37 tests, including an end-to-end pipeline test against the real save).
Manual runtime verification (watching for stutter in `cargo run`) is still
open — see `005-c-async-pipeline.manual-verification.md`.

## Part of
[005 - Stream chunks around the camera](005-chunk-streaming.md).

## Depends on
005-a (deltas to act on), 005-b (region cache to call from a task).

## Problem

Region read, decode (002), and mesh build (003) all currently happen
synchronously on whichever thread calls them. Doing that per streamed chunk
on the main thread would stall frames every time new terrain comes into
range.

## Goal

Turn a `to_load` chunk coordinate into a spawned mesh entity via
`AsyncComputeTaskPool`, polling completed tasks from a main-thread system
and doing only the mesh upload / entity spawn there.

## Scope

- A resource tracking in-flight `Task<ChunkLoadResult>` per chunk coord, so
  005-a's diff never double-requests a chunk that's already loading.
- The task body: resolve the coord's region via 005-b's cache, decode (002)
  if not already decoded, mesh (003) with whatever neighbour columns are
  already available — seam handling is 005-f's problem, not this one.
- A polling system (`Update`) that checks in-flight tasks with
  `block_on(poll_once(&mut task))`; for each completed one, insert
  `Mesh3d`/`MeshMaterial3d`/`Transform` (mirroring today's `setup()` spawn
  logic) and drop it from the in-flight map.
- A budget on how many completed tasks get uploaded per frame, so a big
  batch landing at once doesn't spike a frame (mirrors the parent ticket's
  "budget spawns per frame").

## Watch out

`mc_anvil`/`rnbt` types need to cross a `Send` boundary into the task
closure. Check whether `ChunkRegion`/`NbtField` are actually `Send` before
assuming the region cache itself can be shared into a task:
- If they are `Send`, decide how the cache is shared across threads (e.g.
  `Arc<Mutex<RegionCache>>`) — mind that this can partially serialize
  loads, which may be fine or may need a cache-per-task-pool-worker
  compromise.
- If they are not `Send`, the simplest fallback is to keep `RegionCache` on
  the main thread only, do the file-read (bytes only, definitely `Send`)
  inside the task, and move decode (002) back to the polling system instead
  of the task — slower per-chunk but avoids fighting the type system. Note
  in the commit message which path was taken and why.

## Out of scope

Unloading (005-d). Deleting the eager main.rs load / startup wiring
(005-e).

## Done when

- [ ] Chunks in `to_load` become spawned entities within a few frames, with
  no measurable stutter watched against the real save in a debug build —
  needs a human at the window; tracked in
  `005-c-async-pipeline.manual-verification.md`, not yet checked off.
- [x] Re-requesting a coord that's already in flight does not spawn a
  duplicate task — guarded by `start_chunk_loads`'s `in_flight` lookup
  (`chunk_pipeline.rs`); not yet covered by an automated test (would need a
  Bevy `App`/ECS harness, not just the plain-function tests this ticket's
  other pieces got away with).

## Notes

- `ChunkRegion`/`NbtField`/`SaveMeta` are all plain owned data (`String`,
  `Vec`, primitives, no interior mutability) — auto-`Send`/`Sync`, no
  `unsafe impl` needed. Took the `Arc<Mutex<RegionCache>>` path from
  "Watch out", not the bytes-only fallback: `SharedRegionCache` wraps one
  `RegionCache` shared across every task-pool worker.
- `BlockRegistry` needed the same treatment for a reason the ticket didn't
  call out: chunk decode interns block names into it, and those
  `BlockId`s have to stay valid globally (shared with `DecodedWorld`,
  which the eagerly-loaded startup region and the camera's raycast both
  already read), not just within one task. `DecodedWorld.registry` is now
  `Arc<Mutex<BlockRegistry>>` instead of a bare `BlockRegistry`.
- A task holds the registry lock across both decode *and* mesh (mesh's face
  culling calls back into the registry via `is_solid`) — this serializes
  background chunk tasks against each other, but never blocks the main
  thread, which is what actually avoids frame stutter. Documented in
  `chunk_pipeline.rs` as a trade-off to revisit (e.g. a per-worker registry
  merged back on poll) only if it shows up in a profile.
- `world::atlas::TextureAtlas` couldn't cross the `Send` boundary as-is
  without dragging its render-side `Image` along, so it grew a
  `uv_index()` method returning `AtlasUvIndex` — the plain-data
  name-to-UV-rect lookup half, shared via `Arc` instead. Without this,
  block names first seen only in a streamed-in (not eagerly-loaded) chunk
  would have permanently rendered with the fallback checker texture, since
  the original design built one `uv_table` from the registry once at
  startup and never revisited it.
- Startup (`main.rs::setup`) now also builds `SharedRegionCache`,
  `SharedAtlasIndex`, and `TerrainMaterial` and inserts them as resources
  so the pipeline has something to load with — this augments today's
  eager single-region load, it doesn't replace it. Deleting that eager
  load in favor of streaming as the sole population path is still 005-e's
  job.
- Manual runtime verification (watching the real save in `cargo run` for
  stutter) is tracked separately in
  `005-c-async-pipeline.manual-verification.md` — not yet checked off by a
  human at the window.
