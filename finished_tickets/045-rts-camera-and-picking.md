# 045 - RTS camera and picking

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 376 tests,
up from 369). See the Resolution.

## Part of
Roadmap E1 (`tickets/CITYBUILDER_ROADMAP.md`) — the first ticket in group E
(placement), and the first ticket in the citybuilder that isn't shared
infrastructure (W/B/C/D). Nothing later in E, F, G or H can be built without
a camera that isn't the viewer's free-flight rig, and without a way to turn
"where's the cursor" into a block coordinate.

## Depends on
- `camera.rs`'s existing `CameraRig`/`drive_camera` (ticket 006) — the
  yaw/pitch state machine, the orbit math, and `block_under_cursor`'s DDA
  ray-march are all reused rather than rebuilt.
- `lib.rs::world_app`/`setup_world` (ticket 027) — where the camera entity
  is actually spawned, shared by both games.
- `city::run` (ticket 030's `RenderFloor` override is the precedent for the
  override-after-`world_app()` shape this ticket reuses for the camera's
  starting mode).

## Problem

`city::run()` calls `world_app()`, which unconditionally spawns the camera
in `CameraMode::Fly` and drives it with `camera::CameraControllerPlugin` —
WASD flight, Q/E vertical, right-mouse-look-and-grab, Tab into an orbit
mode. That's the explorer's rig. The citybuilder needs to pan/zoom/rotate
over the terrain instead, and — because `streaming`, `unload` and `sky` all
find "the camera" via `Query<&Transform, With<camera::CameraRig>>` — it has
to do that *as* a `CameraRig`, not a second, parallel camera stack that
those three shared systems would need to learn about.

Left mouse also has to stay free: E3/E4 will use it to place buildings, so
whatever the RTS camera binds for rotation can't be `Orbit`'s left-drag.

## Goal

A third `CameraMode::Rts`, living in the shared `camera.rs` next to `Fly`
and `Orbit` (not a separate module/component), so `streaming`/`unload`/
`sky`'s `With<CameraRig>` queries keep working unmodified. Controls:

| Input | Effect |
|---|---|
| `W`/`A`/`S`/`D` | pan `orbit_target` on the ground plane, yaw-relative (pitch ignored, so panning never drifts vertically) |
| `Q`/`E` | rotate yaw |
| right-mouse drag | free look (yaw + pitch), **not** grabbed/hidden the way `Fly`'s right-drag is — left mouse stays free for E3/E4, and the cursor stays visible for picking |
| scroll | zoom (`orbit_radius`), same math `Orbit` already uses |
| `Shift` | pan speed multiplier, reusing `sprint_multiplier` |
| `Tab` | no-op — the citybuilder never toggles into `Fly`/`Orbit` |

Pitch clamps to a new `rts_pitch_range` (a steeper "always looking down at
the terrain" band) rather than `Fly`/`Orbit`'s near-vertical `pitch_range`.

Plus picking: a per-frame `city::picking::HoveredBlock` resource giving
E2/E3 a ready-made "what block is the cursor over" answer, built on
`camera::block_under_cursor` (already screen-ray → Minecraft block
coordinate, shared with the block inspector and orbit re-aim) rather than a
second raycast.

## Scope

- `camera::CameraMode::Rts` variant; `CameraMode` gains `impl Default`
  (`Fly`) so the new start-mode resource below can derive `Default`.
- `camera::CameraSettings` gains `rts_pitch_range: Range<f32>`,
  `rts_pan_speed_factor: f32` (pan speed = `orbit_radius *
  rts_pan_speed_factor`, so panning covers the same *visual* ground
  regardless of zoom — the same "scale with the current radius" idea
  `orbit_zoom_sensitivity` already uses for zoom), and `rts_rotate_speed`
  (radians/sec for `Q`/`E`).
- `camera::CameraRig::with_mode(self, mode) -> Self` builder, so a rig can
  start somewhere other than `Fly` without a second constructor.
- `camera::CameraStartMode(pub CameraMode)` resource
  (`CameraControllerPlugin` inits it, default `Fly`) — `setup_world` reads
  it once when constructing the initial `CameraRig`. `city::run()`
  overrides it to `Rts` the same way it already overrides `RenderFloor`:
  inserted after `world_app()` returns, before `.run()`.
- `drive_camera` changes: `look_button` gains `Rts => MouseButton::Right`;
  the pitch clamp branches on `rts_pitch_range` vs `pitch_range` by mode;
  `Q`/`E` yaw rotation is read (gated `!egui_input.keyboard`) *before*
  `rotation` is computed, so a same-frame turn isn't a frame late the way
  computing it inside the mode match would be; a new `Rts` arm handles
  scroll-zoom (identical to `Orbit`'s) plus WASD pan.
- `Tab` is a no-op in `Rts` mode: gated at `drive_camera`'s call site
  (`rig.mode != CameraMode::Rts` alongside the existing `!egui_input.keyboard`
  check) rather than inside `toggle_mode` itself, so `toggle_mode` is never
  even called — it stays exactly the two-mode function it always was.
- `city::picking` (new): `HoveredBlock(pub Option<IVec3>)` resource +
  `PickingPlugin`, updating it once a frame from `camera::block_under_cursor`
  against the primary window's cursor, ordered `.after(camera::CameraSet)`
  so it reads this frame's camera, not last frame's. Wired into `city::run()`.
  No consumer yet — E2's grid fit and E3's ghost preview are the eventual
  readers, the same "proven, not yet used" state ticket 042's `City` landed
  in.

## Watch out

- **Don't touch `Fly`/`Orbit`'s existing behaviour.** `rts_pitch_range` is a
  new field, not a replacement for `pitch_range` — the viewer's clamp must
  stay exactly as it is. Same for `orbit_zoom_sensitivity`/`min_orbit_radius`,
  which `Rts` reuses read-only rather than forking.
- **Q/E ordering matters.** If yaw rotation from keys is applied inside the
  `match rig.mode` block (after `rotation` is computed), a same-frame `Q`/`E`
  press won't show up in `transform.translation`'s orbit offset until the
  *next* frame — a one-frame lag that's easy to not notice by eye but easy
  to catch in a test that checks translation, not just `rig.yaw`, on the
  same `app.update()` the key was pressed in.
- **`rig.orbit_target.y` doesn't follow terrain.** Panning only moves X/Z;
  the target's height stays whatever it started at. That's deliberate —
  sampling terrain height under the target is E2's job (footprint/terrain
  fit), not this ticket's. Don't reach for it here.
- **`DecodedWorld`/window-dependent paths stay untested here.** `Rts`'s
  pan/rotate/zoom don't touch the cursor or the block grid at all, so they're
  testable via a bare `App` with no `Window` entity spawned, the same way
  `sync_render_distance_effects` already is. Only `toggle_mode`'s no-op path
  needs checking for `Rts`, and that returns before touching the window.
  `city::picking::update_hovered_block` is the one new piece that *does* need
  a real window/cursor to exercise meaningfully — same boat `drive_camera`
  itself has always been in (untested as a whole system; its pure pieces are
  tested instead), so this ticket doesn't add an App-level test for it.

## Out of scope

- **Terrain-following pan** (`orbit_target.y` tracking the ground under it).
  E2.
- **The grid, footprint occupancy, or anything reading `HoveredBlock`.** E2
  onward.
- **A picking UI/reticle.** Nothing draws `HoveredBlock` yet — there's no
  egui in the citybuilder (group G).
- **Camera-relative pan directions changing with a snapped-yaw convention.**
  Ticket 020 made the same call for selection's arrow keys and it applies
  here too: yaw-relative is simple and predictable; a "snap to nearest of
  four quadrants" remap is a separate, later decision if flat yaw-relative
  turns out to feel wrong at the window.
- **An upper zoom clamp.** `Orbit` doesn't have one today either; adding one
  only for `Rts` would be an inconsistency with no real motivation yet.

## Done when

- `cargo build`/`cargo test` clean.
- Tests: `W` pans the target forward on the yaw-relative ground plane
  (yaw = 0 moves −Z, not X); `Shift` multiplies pan speed; `Q`/`E` rotate
  yaw in opposite directions and the resulting `transform.translation`
  reflects the turn on the *same* `app.update()`; scroll zooms
  `orbit_radius` and clamps at `min_orbit_radius`; a large right-drag pitch
  delta clamps at `rts_pitch_range`'s bound, not `pitch_range`'s (the two
  must be different values in the test, or this doesn't prove anything);
  `Tab` leaves `Rts` unchanged; `CameraRig::with_mode` sets the requested
  mode.
- `todo.md` gets a manual-verification entry: `cargo run --bin citybuilder`,
  confirm the window opens already panning/zooming/rotating (no Fly-mode
  flash first), WASD pans over the terrain, Q/E and right-drag both rotate,
  scroll zooms, left mouse does nothing (reserved), Tab does nothing, and
  the view never flips past looking-down (the `rts_pitch_range` clamp holds
  at both ends). This is exactly the kind of "does it feel right" check
  Claude doesn't run itself, per `CLAUDE.md`.

## Resolution

Landed as scoped, in `camera.rs` and a new `city/picking.rs`.

`CameraMode::Rts` sits next to `Fly`/`Orbit` in `drive_camera`: `look_button`
gains a `Rts => MouseButton::Right` arm (the existing `rig.mode == Fly` grab
block stays untouched, so `Rts`'s right-drag never grabs/hides the cursor);
the pitch clamp now picks `rts_pitch_range` vs `pitch_range` by mode; `Q`/`E`
yaw rotation is read in a small block placed *before* `rotation` is computed
(not inside the mode match, where `Fly`'s Q/E-for-altitude and `Orbit`'s
zoom already live) — the ticket's watch-out about a one-frame lag, verified
by `rts_q_e_rotate_yaw_and_the_same_frame_reflects_it_in_translation`
asserting `transform.translation` against the *post-rotation* yaw on the
same `app.update()`. The `Rts` match arm itself does scroll-zoom (identical
math to `Orbit`'s) plus WASD pan against a yaw-only rotation (`Quat::
from_rotation_y(rig.yaw)`, not the yaw+pitch `rotation` `Fly`/`Orbit` pan/
orbit with), so pressing `W` while tilted down doesn't also nudge the target
into the ground.

`Tab` is gated `rig.mode != CameraMode::Rts` at `drive_camera`'s call site
(alongside the existing `!egui_input.keyboard` check) rather than inside
`toggle_mode`, which turned out simpler than the ticket's original sketch:
`toggle_mode` never gets called at all for `Rts`, so it stays exactly the
two-mode (`Fly`↔`Orbit`) function it always was, with no new branch to keep
in sync with `CameraMode`'s variants.

`CameraStartMode(pub CameraMode)` (`Default` = `Fly`, via a new `impl
Default for CameraMode`) is inserted by `CameraControllerPlugin` and read
once by `lib.rs::setup_world` via `CameraRig::looking_at(eye,
target).with_mode(camera_start_mode.0)`. `city::run()` overrides it to
`Rts` in the same `insert_resource` chain `RenderFloor`'s override already
lives in — no second Startup system racing `setup_world`, and the viewer
(which never touches the resource) is unaffected: `CameraRig::looking_at`
alone still produces exactly the `Fly`-mode rig it always did.

Three new `CameraSettings` fields: `rts_pitch_range` (`-1.4..-0.2`, roughly
-80°..-11° — always angled down, unlike `pitch_range`'s near-vertical
limits), `rts_pan_speed_factor` (pan speed = `orbit_radius *
rts_pan_speed_factor`, so a lap of the screen takes about as long zoomed out
as zoomed in — the same "scale with the current radius" shape
`orbit_zoom_sensitivity` already uses), and `rts_rotate_speed` (90°/sec via
`Q`/`E`). None of the three are tuned against a real window yet — see the
`todo.md` entry below.

`city::picking::{HoveredBlock, PickingPlugin}` is the "screen ray -> block
coordinate" half: a per-frame resource updated from `camera::
block_under_cursor` (no second raycast), ordered `.after(camera::CameraSet)`
the same way the ticket's docs said to, gated through a closure returning
`Option` rather than a chain of early returns (mirrors
`block_inspector_panel`'s shape, minus the egui). Wired into `city::run()`
via `.add_plugins(picking::PickingPlugin)`. No consumer yet — E2/E3, per the
roadmap.

Testing: `Rts`'s pan/rotate/zoom don't touch the cursor or `DecodedWorld` at
all, so they're covered by a bare `App` (`rts_test_app`, no `Window` entity
spawned, one camera entity, `drive_camera` as the only system) driving
`Time` by hand via `Time::advance_by` — the same deterministic pattern
`sky::time_of_day`'s tests already established, avoiding real-wall-clock
flakiness. Seven new tests: `looking_at_with_mode` sets the mode; `W` pans
yaw-relative and never drifts vertically; `Shift` multiplies pan distance by
exactly `sprint_multiplier`; `Q`/`E` rotate yaw and the *same-frame*
translation reflects it; scroll zooms and clamps at `min_orbit_radius`; a
huge right-drag clamps at `rts_pitch_range`'s bound rather than
`pitch_range`'s (asserting the two bounds actually differ first, so the test
can't pass by accident); `Tab` leaves `Rts` unchanged. `city::picking::
update_hovered_block` has no test of its own, per the ticket's own call —
same untested-as-a-whole-system state `drive_camera` itself has always been
in, with the pure piece it delegates to (`block_under_cursor`) already
covered by `camera.rs`'s existing raycast tests.

`todo.md` gets the manual-verification entry the ticket asked for — feel
under an actual window is not something these tests can stand in for.
