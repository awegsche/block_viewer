# 005 - Stream chunks around the camera

## Status
Open — split into subtickets, listed below in dependency order. This file
is now an index; implement against the lettered tickets, not this one.

- [x] [005-a - render distance resource + desired/loaded chunk diff](../finished_tickets/005-a-render-distance-and-diff.md)
- [x] [005-b - region LRU cache](../finished_tickets/005-b-region-lru-cache.md)
- [x] [005-c - async load/decode/mesh pipeline](../finished_tickets/005-c-async-pipeline.md)
- [005-d - unload path + memory budget](005-d-unload-and-budget.md)
- [005-e - startup wiring: delete the eager load, plug in streaming](005-e-startup-wiring.md)
- [005-f - re-mesh at the loading frontier](005-f-boundary-reseam.md) (deferrable)

Move this file to `../finished_tickets/` once every subticket above has
moved (005-f may be deliberately dropped per its own note — see that file).

## Depends on
003 (something to spawn), 006 (a camera worth following) in practice.

## Problem

`load_real_save()` loads exactly one region — `save.regions.first_mut()` —
eagerly, on the main thread, before `App::run()` is even called. That means:
the window doesn't appear until parsing finishes, only 1/N of the world is
reachable, and moving the camera past the region edge shows nothing.

## Goal

Chunks load, mesh, and spawn as the camera approaches them, and unload when
it leaves — off the main thread, without frame hitches.

## Scope

- A `render_distance` (in chunks) resource; default something modest (8-12).
- A system that computes the set of chunk coords within `render_distance` of
  the camera, diffs it against currently-spawned chunks, and issues
  load/unload work for the delta only.
- Region files are the I/O unit (32×32 chunks): opening `r.x.z.mca` and
  inflating is per-region work, so cache decoded regions with an LRU keyed by
  region coords, and drop regions no chunk still needs.
- Move region read + decode (002) + mesh build (003) onto
  `AsyncComputeTaskPool`; poll completed tasks in a system and spawn the
  resulting `Mesh3d` entities. Only mesh upload should touch the main thread.
- Budget spawns per frame (e.g. N chunk meshes/frame) so a big move doesn't
  stall.
- Delete the eager pre-`App::run()` load in `main()`; startup should show a
  window immediately and stream in.

## Watch out

- `mc_anvil::ChunkRegion::load_chunks()` parses **all 1024 chunks** of a
  region in one call and stores them; there is no per-chunk lazy path
  (`get_chunk_or_load` loads the whole region too). That's a chunky unit of
  work to hand a task — measure it. If it's too slow to be a single task,
  that's an upstream ticket in `../ranvil` for a per-chunk load path.
- Unloading must free the decoded NBT too, not just despawn the entity —
  `ChunkRegion` holds `Vec<Option<NbtField>>` for 1024 chunks and that is not
  small.
- Chunk-boundary face culling (003) needs neighbours; a chunk meshed before
  its neighbour arrives will need re-meshing, or you accept seams at the
  loading frontier.

## Out of scope

- Multiple dimensions (nether/end), multiple saves at once (007).
- Frustum culling and draw-call reduction (009).

## Done when

- The window opens immediately; terrain streams in visibly.
- Flying in one direction keeps loading new terrain and unloading terrain
  behind, with steady memory (watch RSS over a few minutes of flight).
- No frame stutter attributable to chunk loading at the default render
  distance.
