# 034 - Live re-mesh: edits mark chunks dirty

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 234
tests). See the Resolution.

## Part of
Roadmap W7 (`tickets/CITYBUILDER_ROADMAP.md`), the last piece of the write
path's shared infrastructure before W8 (a paint/fill command) can prove it
end to end.

## Depends on
- 031/032 (the edit model and its `EditReport::chunks`)
- 005-f (the re-mesh queue this reuses)

## Problem

`edit::apply`/`apply_routed` mutate a `ChunkRegion` that lives in the same
`RegionCache` the streaming pipeline reads from, so a read *of the region*
sees the edit immediately (route.rs's module docs say as much). But nothing
tells `DecodedWorld` or its meshes that's happened: the decoded `ChunkColumn`
for an edited chunk is a snapshot taken before the edit, and its mesh was
built from that snapshot. Without this ticket, blocks written by an edit
never appear on screen until the chunk happens to unload and re-stream.

## Goal

An edit's `EditReport::chunks` drives two things, reusing 005-f's queue
rather than inventing a second dirty-chunk mechanism:

- **The edited chunks themselves** need a full re-decode (their blocks
  actually changed, unlike 005-f's neighbours) plus a re-mesh.
- **Their loaded neighbours** need only a re-mesh — same as 005-f, because
  the edit may have exposed or hidden faces across the boundary even though
  the neighbour's own blocks didn't move.

## Scope

- A `ChunksEdited` event (`chunk_pipeline`) that any edit-issuing system
  (future W8, city building placement) fires with `EditReport::chunks`.
- A new reload queue (`PendingChunkReloads`/`InFlightChunkReloads`,
  `ChunkReloadBudget`) parallel to 005-f's remesh queue: re-decodes a
  chunk from the shared `RegionCache` (which already holds the mutated
  region — no extra invalidation needed) and re-meshes it, off the main
  thread, reusing `load_and_mesh_chunk` as-is.
- A coordinate not currently in `DecodedWorld` is dropped, not queued —
  the next real load reads the (already-edited) region file and is correct
  for free.
- Both queues cancel out of range the same way loads/remeshes already do
  (`unload.rs`), so an edit to a chunk the camera has since left doesn't
  spawn a ghost entity later.

## Watch out

005-f drops a re-mesh request that arrives while one for the same
coordinate is already in flight — acceptable there because the frontier
self-heals on the next neighbour event. A dropped edit is not
self-healing the same way, so the reload queue defers instead of
dropping: a coordinate already in flight stays in the pending set for the
next frame rather than being discarded.

## Out of scope

Nothing calls `ChunksEdited` yet — that's W8. This ticket is the pipeline
machinery only.

## Done when

`cargo build`/`cargo test` clean, including coverage for: the event
populating the reload queue and the (non-edited) neighbour remesh queue,
cancel-out-of-range on both new resources, and the reload task producing a
mesh that reflects a mutated region rather than the stale decoded column.

## Resolution

Landed in `chunk_pipeline.rs`, alongside 005-f's existing queue rather than
as a separate module — it's the same pipeline reacting to a second kind of
trigger:

- `ChunksEdited(Vec<(i32, i32)>)`, a Bevy `Event` any edit-issuing system
  fires with `EditReport::chunks`. Nothing fires it yet (W8 will).
- `queue_edited_chunk_reloads` drains it each frame: every edited chunk
  into `PendingChunkReloads`, every one of its loaded neighbours *not*
  itself edited into 005-f's `PendingChunkRemeshes` — an edited chunk's
  own reload already re-meshes it against fresh neighbours, so its
  edited neighbours don't also need a plain re-mesh entry.
- `start_chunk_reloads`/`poll_completed_chunk_reloads` are a fourth pair
  alongside the load/re-mesh pairs, reusing `load_and_mesh_chunk` verbatim
  as the task body — a reload *is* a load, decode included, just
  triggered by an edit instead of streaming. The poll side records the
  fresh column into `DecodedWorld` (the point of a reload over a re-mesh)
  and then applies the mesh the same spawn/swap/despawn way a re-mesh
  does; that three-way match was pulled out of
  `poll_completed_chunk_remeshes` into a shared `apply_mesh_update` so the
  two pollers don't carry two copies of it.
- One deliberate divergence from 005-f: `start_chunk_reloads` leaves a
  coordinate already in flight in the pending set (deferred to next
  frame) instead of dropping it, since a second edit to a chunk mid-reload
  has nothing else that will re-trigger it later, unlike the frontier's
  self-healing neighbour arrivals.
- `unload.rs`'s existing cancel-out-of-range system grew the two new
  resources so a coordinate leaving render distance can't leave a ghost
  reload behind, the same reasoning already applied to loads and
  re-meshes.

New tests: `cancel_out_of_range` on both new resources (mirroring the
existing load/re-mesh coverage), and a system-level test driving
`queue_edited_chunk_reloads` through a bare `App` to check the edited/
neighbour split, including that two adjacent edited chunks don't queue a
redundant re-mesh for the boundary between them.
