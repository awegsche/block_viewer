# 005-f - Re-mesh at the loading frontier

## Status
Open (deferrable — see note at the end)

## Part of
[005 - Stream chunks around the camera](005-chunk-streaming.md).

## Depends on
005-c (the async mesh-build path this reuses).

## Problem

003's mesher bakes "neighbour missing = air" into a chunk's mesh at build
time (documented, tested behaviour: `missing_neighbor_leaves_a_seam`). With
static loading that was a one-off edge case; with streaming it becomes a
seam that continuously trails the camera at the render-distance boundary,
since chunks are now routinely meshed before their neighbours arrive.

## Goal

When a chunk finishes loading, re-mesh any already-loaded neighbour that
was missing it, so the seam between them closes instead of persisting.

## Scope

- On a completed chunk load (005-c), check its four neighbour coords; for
  any that are already loaded, enqueue *them* for a re-mesh (not a
  re-load — the neighbour's decoded `ChunkColumn` hasn't changed, only its
  mesh needs to be rebuilt now that real neighbour data exists).
- Route re-mesh requests through 005-c's existing async mesh-build path;
  replace the entity's `Mesh3d` handle in place rather than
  despawning/respawning it.
- Cap re-mesh uploads per frame the same way 005-c budgets initial loads,
  so a batch of simultaneous neighbour arrivals doesn't reintroduce the
  stutter that budget was added to avoid.

## Watch out

Re-meshing a chunk doesn't itself require re-meshing its other neighbours
(nothing about its own exposed-face set changed for them), but a naive
implementation can still enqueue duplicate re-mesh requests for the same
coord within one tick if two of its neighbours load in the same frame —
dedupe before spawning tasks.

## Out of scope

Nothing beyond what's listed — this is the last slice of 005.

## Done when

- Flying past the loading frontier at normal fly speed doesn't show a
  seam that outlives the one frame a chunk is legitimately still in
  flight.

## Note

This ticket is the parent ticket's explicitly-flagged "or you accept seams
at the loading frontier" alternative. If 005-e's seams look acceptable in
practice at the default render distance, this can be deprioritized or
dropped without blocking 005's own "Done when" — flag that call back to
whoever's reviewing rather than silently skipping it.
