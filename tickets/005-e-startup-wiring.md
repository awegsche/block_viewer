# 005-e - Startup wiring: delete the eager load, plug in streaming

## Status
Open

## Part of
[005 - Stream chunks around the camera](005-chunk-streaming.md).

## Depends on
005-a, 005-b, 005-c, 005-d — needs the whole pipeline to swap in for the
current eager path.

## Problem

`main.rs` still eagerly loads and decodes one whole region on the main
thread before `App::run()` is even called (`load_real_save` /
`decode_region`). The window doesn't appear until that finishes, only 1/N
of the world is reachable, and `setup()`'s spawn loop assumes every column
it will ever need is already decoded.

## Goal

Delete the eager pre-`App::run()` load and the one-shot `setup()` spawn
loop; wire the streaming pipeline (005-a through 005-d) in as a plugin so
the window opens immediately and terrain streams in around wherever the
camera starts.

## Scope

- Remove `load_chunks()`/`decode_region()` from `main()`'s pre-`App::run()`
  path. `get_saves()`/picking which save to open stays synchronous and
  eager — it's cheap metadata, not chunk data.
- Replace `setup()`'s per-column spawn loop with the streaming plugin's
  systems (005-a through 005-d wired together). Keep in `setup()` whatever
  still makes sense to do once at startup: building the texture atlas +
  material (004), spawning the camera and light.
- `spawn_point` currently reads `decoded_world.columns` synchronously at
  startup — nothing is decoded yet at that point anymore. Pick a startup
  placement: a fixed sensible default height (or the save's spawn point
  from `level.dat`, if `ranvil`/`rnbt` expose it) with the camera free-flying
  while terrain streams in underneath. Prefer this over deferring camera
  spawn until the first chunk loads — simpler, and matches "window opens
  immediately."
- Wire `camera.rs`'s far-plane/fog distance to 005-a's real `RenderDistance`
  resource, replacing `PLACEHOLDER_RENDER_DISTANCE_CHUNKS`.

## Watch out

`world::atlas::build_block_uv_table` currently reads
`decoded_world.registry`, which only has entries for block names seen
during the eager decode. With streaming, block names get interned as new
chunks decode over the app's lifetime — check whether the UV table can
still be built once at startup (e.g. from every name the resource-pack's
texture directory offers, not just names seen so far) or needs to grow as
new names are interned while streaming. A save whose starting view is
missing block types present elsewhere is exactly the case that would
surface a bug here — don't skip testing it.

## Out of scope

007's save picker UI, 008's error surfacing — both are separate tickets
that also touch `main.rs`/startup; don't preempt their scope here.

## Done when

- `cargo run` shows a window immediately — no multi-second blank/frozen
  startup.
- Terrain visibly streams in around the camera.
- Flying around behaves per the parent ticket's "Done when" (steady memory,
  no stutter attributable to chunk loading).
