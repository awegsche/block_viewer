# 005-e - Startup wiring: delete the eager load, plug in streaming

## Status
Implemented in `src/main.rs` (+ `src/camera.rs` for the render-distance
wiring), `cargo build`/`cargo test` clean (38 tests). Manual runtime
verification (window opens immediately, terrain visibly streams in, no
stutter) is still open — see `005-e-startup-wiring.manual-verification.md`.

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

- [ ] `cargo run` shows a window immediately — no multi-second blank/frozen
  startup. Not yet checked off; see
  `005-e-startup-wiring.manual-verification.md`.
- [ ] Terrain visibly streams in around the camera. Not yet checked off; see
  `005-e-startup-wiring.manual-verification.md`.
- [ ] Flying around behaves per the parent ticket's "Done when" (steady
  memory, no stutter attributable to chunk loading). Not yet checked off;
  overlaps 005-c/005-d's own still-open manual verification, since this
  ticket is what finally makes streaming the *only* path chunks spawn
  through.

## Notes

- `main.rs::load_real_save` no longer calls `ChunkRegion::load_chunks()` —
  it's metadata only (`get_saves` + `SaveMeta -> Save`, no file I/O beyond
  reading the saves directory listing), same as the ticket's scope note.
  `main.rs::decode_region` (the eager NBT-to-`ChunkColumn` decode) and
  `setup()`'s per-column spawn loop are both deleted outright; `main()` now
  inserts an empty `DecodedWorld` and `setup()` only builds the atlas,
  material, camera, and light before handing off to the streaming plugins
  (005-a through 005-d), already wired into `App::new()` from 005-c/005-d.
- Startup placement: took the "fixed sensible default height" option, not
  a `level.dat` spawn point — `ranvil`/`rnbt` don't expose one (checked
  `save.rs`/`lib.rs`; only `SaveMeta`/`Save`/region enumeration, no
  player/level NBT reading). To avoid the camera free-flying over empty
  space until it happens to wander into a region, X/Z is the horizontal
  centroid of the save's *region* footprint (`SaveMeta.regions`, cheap
  metadata, no chunk I/O) snapped to a real region like the old
  column-centroid code snapped to a real column; Y is a fixed 100 blocks.
  `world::ChunkColumn::topmost_non_air`, which the old height-aware
  `spawn_point` used, has no caller left outside its own test — kept
  (`#[allow(dead_code)]`) for ticket 007's planned block-under-cursor
  readout rather than deleted, per the codebase's existing convention for
  API surface with no caller yet.
- `camera::far_plane_distance`/`camera::atmosphere_fog` now take a
  `render_distance_chunks: u32` parameter instead of reading the deleted
  `PLACEHOLDER_RENDER_DISTANCE_CHUNKS` constant; `setup()` passes
  `RenderDistance`'s value (available at `Startup` time — `streaming`'s
  plugin `init_resource`s it during `Plugin::build`, which runs before
  `Startup` systems).
- "Watch out"'s UV-table-growth question turned out to already be resolved
  by 005-c's own design, not something this ticket needed to add: main.rs's
  atlas-registry lock + `build_block_uv_table` call was only ever there to
  feed the eager spawn loop this ticket deletes.
  `chunk_pipeline::load_and_mesh_chunk` already rebuilds a fresh
  `build_block_uv_table` from the *current* registry state inside every
  chunk's background task, after that chunk's own decode has interned
  whatever new names it introduced — so a save whose starting view is
  missing block types seen only in a chunk that streams in later already
  resolves correctly, with no separate startup UV table needed at all.
  `setup()` now only builds the atlas's `AtlasUvIndex` (the name -> rect
  mapping, independent of any registry) once, and shares it via
  `SharedAtlasIndex`.
