# 005-a - Render distance resource + desired/loaded chunk diff

## Status
Open

## Part of
[005 - Stream chunks around the camera](005-chunk-streaming.md), split because that
ticket is too big to land in one piece. This slice is pure logic: no I/O, no
async, nothing spawned yet.

## Depends on
003 (something to eventually spawn), 006 (a camera position to measure from)
— both already done.

## Problem

There is currently no notion of "which chunks should be loaded right now."
`camera.rs` has a `PLACEHOLDER_RENDER_DISTANCE_CHUNKS` constant explicitly
documented as a stand-in until this ticket lands, and nothing computes a
desired chunk set from the camera's position at all.

## Goal

A `RenderDistance` resource and a pure diff function that, given the
camera's current chunk coordinate and the set of currently-loaded chunk
coordinates, produces the set of chunks to load and the set to unload.

## Scope

- `RenderDistance(u32)` resource (chunks), default 8-12, matching the range
  the parent ticket and `camera.rs`'s placeholder already document.
- A pure function `desired_chunks(center: (i32, i32), radius: u32) ->
  HashSet<(i32, i32)>` — a square radius is fine; circular is not worth the
  complexity yet.
- A pure function `diff_chunks(desired: &HashSet<(i32, i32)>, loaded:
  &HashSet<(i32, i32)>) -> (to_load: Vec<(i32, i32)>, to_unload: Vec<(i32,
  i32)>)`.
- A Bevy system that reads the `CameraRig`/`Transform` each frame (or only
  when the camera crosses a chunk boundary, if recomputing every frame shows
  up in a profile), converts to a chunk coordinate, and republishes the
  current delta somewhere later tickets can consume it — e.g. a
  `PendingChunkWork { to_load: Vec<(i32,i32)>, to_unload: Vec<(i32,i32)> }`
  resource. This ticket does not act on the delta.

## Watch out

- Don't build an incremental/spatial structure for this yet — a render
  distance of 12 is at most ~625 chunks; a `HashSet` diff over that every
  frame is cheap. Optimize only if it shows up in a profile.
- Decide and document whether the diff recomputes every frame or only on
  chunk-boundary crossings — either is fine, but 005-c/005-d need to know
  which so they don't do redundant work.

## Out of scope

Actually loading, decoding, meshing, spawning, or unloading anything
(005-b onward). Wiring `RenderDistance` into `camera.rs`'s far-plane/fog
distance (005-e).

## Done when

- Unit tests cover `desired_chunks` (correct radius, correct count) and
  `diff_chunks` (chunks entering/leaving the set land in the right bucket,
  unchanged chunks land in neither).
- Running the app and moving the camera visibly changes the computed delta
  (a debug `println!`/log is enough — no UI needed yet).
