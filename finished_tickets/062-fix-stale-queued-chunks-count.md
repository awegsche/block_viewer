# 062 - fix stale "queued chunks" count / pending.to_load never draining

## Report

User: "I tested the block_viewer app: when loading a new world, not all
chunks seem to get loaded. I see queued chunks: 850 and the world only
displays some chunks."

## Root cause

`chunk_pipeline::start_chunk_loads` iterated `&pending.to_load` (a shared
borrow) and, unlike its siblings `start_chunk_remeshes`/`start_chunk_reloads`,
never removed a coordinate from the list once it dispatched a task for it
(or found it already loaded/in flight). Every coordinate diffed into
`PendingChunkWork::to_load` therefore sat there forever, unchanged, until
the next camera chunk-boundary crossing recomputed the whole list from
scratch.

The status panel's "Queued chunks" counter
(`pending.to_load.len() + in_flight_loads.len()`, `viewer/ui/status.rs`) was
built on the assumption that `to_load` shrinks as work is dispatched, the
same way `PendingChunkRemeshes`/`PendingChunkReloads` do. Since it didn't,
the counter stayed pinned at the render distance's full chunk count (e.g.
850 for a ~14-chunk render distance) for as long as the camera stood still —
even while loading was actually happening correctly in the background
(loads are all dispatched into `AsyncComputeTaskPool` in the same frame;
`start_chunk_loads` already skips a coordinate that's in flight or already
loaded, so no duplicate work was ever spawned).

Net effect: the counter gave no real signal of progress, so a large render
distance's slow-but-working streaming (decode+mesh for every chunk is
serialized behind one shared `Mutex<BlockRegistry>`/`Mutex<BiomeRegistry>` —
see `chunk_pipeline`'s module docs) reads as permanently stuck rather than
"still catching up."

## Fix

`start_chunk_loads` now takes `ResMut<PendingChunkWork>` and drains
`pending.to_load` via `std::mem::take` each call, mirroring
`start_chunk_remeshes`'s `pending.0.drain()`. Every coordinate gets resolved
in the same pass it's read: spawned into `in_flight`, or dropped because
it's already loaded/already in flight. Nothing is left behind to be
re-scanned (and mis-displayed) on later frames.

## Verification

- `cargo check` — clean.
- `cargo test --lib chunk_pipeline streaming unload` — all pass except the
  pre-existing `load_and_mesh_chunk_decodes_and_meshes_a_real_chunk`, which
  depends on the real Minecraft saves directory on this machine having a
  fully-generated chunk at a region centre and fails the same way on `main`
  before this change (confirmed via `git stash`) — unrelated to this fix.
- Manual/visual verification (does the "Queued chunks" counter now visibly
  count down, and do all chunks in view eventually render, for a freshly
  loaded large world) is noted in `todo.md` — needs a human watching the
  window, per this repo's convention.
