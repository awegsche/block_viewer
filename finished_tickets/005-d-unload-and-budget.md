# 005-d - Unload path + memory budget

## Status
Implemented in `src/unload.rs` (plus small `pub(crate)` visibility bumps in
`src/chunk_pipeline.rs` and `src/streaming.rs`), `cargo build`/`cargo test`
clean (38 tests). Manual runtime verification (RSS over a flight, no stale
spawns on reversal) is still open — see
`005-d-unload-and-budget.manual-verification.md`.

## Part of
[005 - Stream chunks around the camera](005-chunk-streaming.md).

## Depends on
005-a (the `to_unload` side of the diff), 005-c (something loaded to
unload, and the in-flight task map).

## Problem

Nothing currently despawns a chunk entity or frees its decoded data —
everything loaded stays loaded forever. Once loading is real (005-c), flying
around would only ever grow memory without this half of the pair.

## Goal

Chunks in `to_unload` (005-a) get their entity despawned and their decoded
`ChunkColumn` dropped from the world's column map; regions no longer
referenced by any loaded chunk get evicted from 005-b's cache too.

## Scope

- For every `to_unload` coord each diff tick: despawn its chunk entity,
  remove its `ChunkColumn` from the streaming successor to today's
  `DecodedWorld.columns`, and free its `Mesh` handle via
  `Assets<Mesh>::remove` if nothing else references it.
- Track whether any region in 005-b's cache has zero chunks left
  referencing it — either a per-region refcount, or (simpler, pick this
  unless it doesn't hold up) just size the cache capacity close to render
  distance's region footprint and let plain LRU eviction handle it. Say in
  the commit message which was chosen.
- Cancel any in-flight *load* task (005-c's in-flight map) for a coord that
  leaves render distance again before the task finishes — e.g. the camera
  reverses near the edge — so it doesn't spawn something already out of
  range.
- A per-frame unload budget only if a large camera jump (e.g. a teleport)
  is shown to unload hundreds of chunks in one frame; measure before adding
  one.

## Watch out

The shared `BlockRegistry` (today part of `DecodedWorld`) must keep living
independently of any single column — `BlockId`s have to stay stable and
global even as columns come and go. Don't let column unload logic touch the
registry.

## Out of scope

Region cache internals (owned by 005-b). The diff computation itself
(005-a).

## Done when

- [ ] Flying in one direction for a few minutes against the real save keeps
  RSS roughly flat rather than monotonically growing — the parent ticket's
  memory criterion, verified here specifically. Not yet checked off; see
  `005-d-unload-and-budget.manual-verification.md`.
- [ ] Reversing direction near the loading edge doesn't spawn chunks that
  should already be out of range. Not yet checked off; see
  `005-d-unload-and-budget.manual-verification.md`. (Automated coverage for
  the underlying mechanism —
  `chunk_pipeline::tests::cancel_out_of_range_drops_tasks_outside_the_desired_set`
  — passes.)

## Notes

- For every `to_unload` coordinate, `unload::unload_chunks` despawns the
  entity (via a new `SpawnedChunkEntities` coord -> `Entity` map, populated
  both by `chunk_pipeline::poll_completed_chunk_loads` and by
  `main.rs::setup`'s still-live eager startup spawn), frees its `Mesh` via
  `Assets<Mesh>::remove` (safe unconditionally — chunk meshes are never
  shared across entities), and drops its `ChunkColumn` from
  `DecodedWorld.columns`.
- In-flight loads (005-c) for a coordinate the camera has since left aren't
  visible to `to_unload` at all — `to_unload` is diffed against
  `DecodedWorld.columns`, and an in-flight coordinate isn't in `columns`
  yet. `unload::cancel_out_of_range_in_flight_loads` recomputes the desired
  set itself every frame (reusing `streaming::desired_chunks` and
  `streaming::camera_chunk_coord`, the latter bumped to `pub(crate)`) and
  drops (`InFlightChunkLoads::cancel_out_of_range`) any in-flight task
  outside it — dropping a `bevy_tasks::Task` cancels it per its own doc
  comment, no `.await` needed.
- Both new unload systems are ordered `.before()` `chunk_pipeline`'s
  `poll_completed_chunk_loads`/`start_chunk_loads` (both bumped to
  `pub(crate)` for this) so a coordinate canceled or unloaded this frame
  can't also get polled-complete-and-spawned, or re-requested, the same
  frame.
- Region cache eviction: took the "simpler" option per the ticket's own
  suggestion — no per-region refcount. `RegionCache`'s capacity (005-b's
  `recommended_capacity`) is already sized to one render distance's worth
  of regions, so its existing plain LRU eviction naturally drops regions no
  loaded chunk still needs.
- No per-frame unload budget added — nothing so far shows a large jump
  (only a flying, not teleporting, camera exercises this path yet)
  unloading hundreds of chunks in one frame. Add one if that's measured to
  change.
