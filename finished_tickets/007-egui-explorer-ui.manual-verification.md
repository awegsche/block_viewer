# 007 - Manual verification checklist

Automated checks already confirmed: `cargo build`/`cargo check`/`cargo clippy`
are clean, and `cargo test` passes (42 tests — unchanged in count, since this
ticket's egui panels have no unit-testable logic of their own beyond what
`camera::tests::raycast_*` already covers for the reused ray-march). The
items below are the ticket's "Done when" criteria plus its "watch out" note,
and need an actual human at the window — `cargo run` and confirm them.

- [ ] **Save picker: switching saves at runtime, no restart.** `cargo run`
  (debug build; needs at least two saves under the Minecraft saves
  directory to fully exercise this — otherwise just confirm the list shows
  the one save with "— current" and no crash). Open the "Save" window,
  click "Load" on a different save. Confirm: the previously spawned
  terrain despawns, the camera jumps to the new save's region centroid,
  and new terrain streams in — all without the process restarting or a
  console panic.
- [ ] **Region grid: clicking a region teleports the camera.** In the
  "Save" window, click a filled cell in the region grid (or, for a save
  with an impractically large footprint, confirm the ASCII grid view
  renders instead without hanging the UI). Confirm the camera jumps to
  that region and terrain streams in under it.
- [ ] **Coordinate jump moves the camera and terrain streams in.** In the
  "Navigate" window, type X/Y/Z values inside the save's region footprint
  and click "Go". Confirm the camera teleports there (switching out of
  orbit mode if it was in it) and terrain streams in around the new
  position. Confirm the block/chunk/region readout above it updates live
  as you fly around afterward.
- [ ] **Block inspector matches the rendered terrain.** Point the cursor at
  a visible block (not over any egui panel) and confirm the "Block
  Inspector" window shows a `Name` that matches what's rendered there
  (e.g. grass/stone/water) and non-garbage `Properties`. Move the cursor
  over empty sky and confirm it says so rather than showing stale data.
- [ ] **Status panel: FPS/counts look sane, render distance slider works.**
  Confirm FPS is a plausible non-zero number, "Loaded chunks"/"Queued
  chunks" move as the camera flies (increasing as new terrain streams in,
  settling once it catches up), and dragging the render distance slider
  visibly grows/shrinks how far terrain is streamed in and rendered
  (fog/far-plane pop should track the new value too, not stay stuck at the
  startup one).
- [ ] **egui doesn't fight the camera (ticket's "watch out" note).** With
  the mouse over any egui panel: dragging shouldn't spin the fly-mode
  view, scrolling over the render-distance slider shouldn't change fly
  speed/orbit zoom, and typing into a coordinate field shouldn't also
  trigger WASD movement or the `Tab` mode toggle. Releasing the right
  mouse button over a panel should still un-grab the cursor if a fly-mode
  look-drag was already in progress.
