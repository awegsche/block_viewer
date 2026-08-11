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
