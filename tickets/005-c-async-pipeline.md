# 005-c - Async load/decode/mesh pipeline

## Status
Open

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

- Chunks in `to_load` become spawned entities within a few frames, with no
  measurable stutter watched against the real save in a debug build.
- Re-requesting a coord that's already in flight does not spawn a duplicate
  task.
