# 006 - Manual verification checklist

Automated checks (unit tests, `cargo check`, and a scripted headless run
comparing logged camera transforms across frames) already confirmed: the
window opens and renders terrain, fly-mode WASD movement covers the
expected distance for the configured speed, and the orbit-mode
raycast/toggle math is correct (`src/camera.rs`'s `raycast_terrain` unit
tests). The items below need an actual human at the window — `cargo run`
and go through them.

- [ ] **Mouse-look in fly mode** — hold right mouse button, confirm the
      cursor grabs/hides and the view rotates smoothly; releases cleanly on
      button-up.
- [ ] **Sprint** — Left Shift while moving noticeably speeds up.
- [ ] **Scroll-adjusts-speed** in fly mode — confirm it feels right
      (currently ±20%/notch, exponential).
- [ ] **Orbit mode entry (`Tab`)** — aim at actual terrain, press Tab,
      confirm the view now pivots around that point rather than flying;
      press Tab again to confirm it returns to fly mode without snapping.
- [ ] **Orbit drag** — left-drag while orbiting rotates around the target;
      scroll zooms in/out; verify `min_orbit_radius` doesn't feel too
      close/far.
- [ ] **Tab with cursor over open sky** — should still enter orbit mode,
      targeting a point straight ahead (fallback path) rather than doing
      nothing.
- [ ] **Far plane / fog** — fly toward the edge of the loaded chunks and
      confirm terrain fades via fog rather than hard-popping or getting
      clipped.
- [ ] **Initial spawn placement** — on app start, confirm the camera is
      above the terrain surface, not underground or floating absurdly
      high, across a couple of different real saves if available.
