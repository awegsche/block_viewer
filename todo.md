# To-do

Manual/visual checks that need a human at the window — Claude doesn't run
these itself (see CLAUDE.md's "Manual/visual verification").

- [ ] **005-c async pipeline: chunks stream in without stutter.**
  `cargo run` (debug build) against the real save, fly the camera away
  from its streamed-in startup position for a minute or two (005-e deleted
  the old eagerly-loaded startup region — everything, including what's
  under the camera at spawn, now streams in via the pipeline). Confirm:
  new chunk mesh entities appear within a few frames of entering render
  distance (watch the console for the pipeline picking up `to_load`
  coordinates), the frame rate doesn't visibly hitch when a batch of
  chunks completes, and nothing panics. Checklist:
  `finished_tickets/005-c-async-pipeline.manual-verification.md`.
- [ ] **005-e startup wiring: window opens immediately, terrain streams in
  from a cold start.** `cargo run` (debug build) against the real save.
  Confirm the window appears immediately (no multi-second blank/frozen
  startup — nothing loads before `App::run()` anymore), and that terrain
  visibly streams in around the camera's fixed startup position (near the
  save's region-footprint centroid, ~100 blocks up) rather than starting
  from an empty world. Checklist:
  `finished_tickets/005-e-startup-wiring.manual-verification.md`.
- [ ] **005-d unload path: memory stays flat, no stale spawns on
  reversal.** `cargo run` (debug build) against the real save. (1) Fly in
  one direction for a few minutes past several render-distance widths of
  terrain, watching RSS — it should plateau rather than climb
  monotonically once chunks are loading ahead and unloading behind. (2)
  Fly toward the render-distance edge until new chunks start loading
  (watch console for `to_load` pickups), then reverse before they finish;
  confirm no chunk mesh pops in behind the camera's current render
  distance right after the reversal. Checklist:
  `finished_tickets/005-d-unload-and-budget.manual-verification.md`.
- [ ] **005-f boundary re-mesh: no lingering seam at the loading
  frontier.** `cargo run` (debug build) against the real save, fly past
  the loading frontier at normal fly speed. Watch the boundary between a
  chunk that just finished loading and its already-loaded neighbour — any
  seam there should close within the one frame the new chunk is
  legitimately still in flight, not visibly trail the camera. Checklist:
  `finished_tickets/005-f-boundary-reseam.manual-verification.md`.
- [ ] **007 egui explorer UI: save picker, coordinate jump, block
  inspector, status panel — and that egui doesn't fight the camera.**
  `cargo run` (debug build) against the real save(s). Covers switching
  saves at runtime, clicking a region to teleport there, jumping to typed
  coordinates, the block inspector matching rendered terrain, the status
  panel's FPS/chunk counts/render-distance slider, and that dragging a
  slider or typing into a field doesn't also spin the view or fly the
  camera. Checklist:
  `finished_tickets/007-egui-explorer-ui.manual-verification.md`.
- [ ] **008 graceful startup errors: no panic on a missing/empty saves
  directory or a corrupted region file.** Needs a human to run `cargo run`
  a few different ways (pointing it at a nonexistent directory, an empty
  directory, and against a deliberately truncated `.mca` file) and confirm
  the window still opens and the reason shows up in the UI/console each
  time rather than a crash. Full steps and what to look for:
  `finished_tickets/008-graceful-startup-errors.manual-verification.md`.
- [ ] **011 vertex colour channel: purely a no-op visually.** `cargo run`
  (debug build) against the real save. Confirm the world looks **exactly**
  as it did before this ticket — same colours everywhere. This ticket only
  adds a white (`[1,1,1,1]`) multiplicative vertex colour attribute to
  every chunk mesh; if anything looks tinted, darker, or washed out, the
  colour is landing in the wrong colour space (see `src/world/mesh.rs`'s
  module docs on sRGB vs. linear) rather than being a genuine no-op.
- [ ] **012 biome decode: block inspector shows a plausible biome, world
  renders unchanged.** `cargo run` (debug build) against the real save.
  Point the cursor at terrain in a few different spots (ideally spots you
  know the biome of — e.g. a beach vs. a forest) and confirm the "Block
  Inspector" panel's new "Biome: minecraft:..." line names something
  plausible for what's rendered there, and changes as you look at
  different areas rather than sitting stuck on `minecraft:plains`
  everywhere. Also confirm the world looks exactly as before — this ticket
  is decode-only, no rendering changed.
