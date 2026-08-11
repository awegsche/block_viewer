# 005-d - Unload path + memory budget

## Status
Open

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

- Flying in one direction for a few minutes against the real save keeps RSS
  roughly flat rather than monotonically growing — the parent ticket's
  memory criterion, verified here specifically.
- Reversing direction near the loading edge doesn't spawn chunks that
  should already be out of range.
