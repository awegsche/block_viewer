# 122 - Chunk streaming: hide the frontier in the fog, load a disc, keep more

## Status
Done — `cargo test --lib`: 1129 passed. Manual check pending, see `todo.md`.

## Problem

"I fly around and chunks vanish and new ones have to be redrawn."

Three things conspire, all in `streaming`/`camera`:

1. **The fog doesn't cover the loading edge.** `desired_chunks` loads a
   *square* of radius `RenderDistance` chunks, so terrain ends `rd * 16`
   blocks out in the cardinal directions — but `camera::fog_falloff` only
   goes opaque at the far plane, `rd * 16 * sqrt(2) + 32`, the square's
   *diagonal* corner. At the viewer's rd=10 the frontier sits at 160 blocks
   under ~5% fog; at the citybuilder's rd=16 it's at 256 blocks under ~13%.
   Every load and unload happens in plain view, and the annulus between
   `rd * 16` and the fog end is visible but never loaded.
2. **Retention is short.** `ChunkRetention::max_lingering = 256` is about
   7-10 chunk crossings of trail (a crossing sheds a whole 25-37 column
   edge), so flying ~150 blocks away and back reloads the lot.
3. **Loads dispatch in `HashSet` order** — the chunk straight ahead is no
   more likely to finish first than a corner.

## Change

- `RenderDistance` now means *what you can see*: fog goes opaque at
  `rd * 16` blocks, far plane sits one chunk past that (instead of the
  circumscribed radius). Anything drawn beyond the fog end was pure fog
  colour anyway, so the tighter far plane only culls invisible geometry.
- New `ChunkPreload` resource (default 2): chunks load out to
  `rd + preload`, so the frontier is always inside opaque fog. Retention's
  margin/grace sit past the *load* radius, as before.
- `desired_chunks` builds a **disc** (Euclidean), not a square. A disc of
  radius `rd + 2` covers every point within the fog end with margin to
  spare, and drops the square's corners — which were always fully fogged.
  Net chunk count at the defaults ends up about where it was
  (rd=10: 441 -> ~450; rd=16: 1089 -> ~1020), but now everything loaded is
  either visible or the hidden frontier, and everything visible is loaded.
- `ChunkRetention` defaults: `max_lingering` 256 -> 1024, `grace` 30s ->
  90s. Eviction order switches to Euclidean distance to match the disc.
- `to_load` is sorted nearest-first before `chunk_pipeline` dispatches it,
  so with tasks serialising on the registry lock the chunks in front of
  the camera land first.
- Region cache capacity and the in-flight cancel radius use the load
  radius, not the render distance.

## Visual consequence

The fog end moves in (rd=10: 258 -> 160 blocks; rd=16: 394 -> 256). It
was overstating what was loaded; what you *saw* in the cardinal
directions was already ending at 160/256, just unfogged. Anyone who wants
today's diagonal reach everywhere turns the slider up — that costs real
chunks, which is the honest trade the old numbers hid.

## Done when

- `cargo test --lib` passes with updated streaming/camera tests.
- Manual: fly around in either app; no pop-in at the frontier, no reloads
  when turning around or doubling back within a few hundred blocks.
