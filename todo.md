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
- [ ] **013 biome tint: grass/leaves/water are coloured and vary by biome,
  nothing looks washed out.** `cargo run` (debug build) against the real
  save. Fly across a biome border (e.g. plains into a forest or desert,
  or toward any ocean) and confirm the tint visibly changes rather than
  staying flat green everywhere. Look at a grass block from above (should
  read green) and then from the side (should still read brown dirt —
  014's problem, not a bug here). If you know a spot with swamp or
  badlands terrain in this save, check those read as a duller
  green/khaki rather than the plains green (the hardcoded overrides in
  `src/world/tint.rs`). Also confirm nothing looks unnaturally bright or
  pastel/washed-out — that would mean the sRGB→linear conversion regressed
  (see `src/world/tint.rs`'s module docs).
- [ ] **014 cutouts and grass overlay: leaves have holes, grass sides are
  green.** `cargo run` (debug build) against the real save. Stand at ground
  level in a grassy biome: grass blocks should read green on top *and*
  around their sides (biome-tinted, matching 013), with brown dirt visible
  below the green fringe. Fly to the render-distance edge and watch the
  overlay quads on distant grass blocks for shimmering/z-fighting (would
  mean `OVERLAY_EPSILON` in `src/world/mesh.rs` needs revisiting). Look at
  a tree: leaves should read as foliage you can see holes through, not a
  solid green block — but note looking *through* a leaf into the canopy
  will show missing geometry behind it (leaves still cull neighbouring
  faces; that's 010's transparency work, out of scope here per the ticket).
- [ ] **015 sun rig and sky colour: lighting doesn't fall off far from
  spawn, no seam at the fog/sky boundary, world isn't bleached out.**
  `cargo run` (debug build) against the real save. Fly a few hundred
  blocks from spawn and confirm terrain out there is lit the same as
  terrain near spawn — the old `PointLight` only lit a small sphere near
  its spawn point, so this checks the `DirectionalLight` replacement
  actually reaches everywhere. Look toward the render-distance edge and
  confirm there's no visible band where terrain fades into fog and fog
  meets the clear-colour "sky" — they should read as one continuous
  colour (`SkyPalette::horizon_color` drives both). Also move the render-
  distance slider and confirm the horizon colour does *not* change (only
  the fog's falloff distance should react — a regression here would look
  like "the horizon goes blue-grey when I move the slider"). Finally,
  check the world doesn't read as washed out/bleached under the
  ~10,000-lux noon sun — if it does, `src/sky/mod.rs`'s `SkyPalette::default`
  needs a lower `sun_illuminance`, or `main.rs`'s camera needs a
  `Tonemapping` other than the default `TonyMcMapface` (try
  `ReinhardLuminance` before reaching for `Exposure` changes — see the
  ticket's "Tone mapping" section for the full order to try).
- [ ] **016 skybox: gradient dome, sun and moon.** `cargo run` (debug
  build) against the real save. Look straight up: should read as a smooth
  zenith→horizon gradient with a large sun disc positioned consistently
  with the lighting/shadows on terrain (opposite the direction it's
  shining from). Look straight down from height: the dome should still be
  visible past the render-distance edge, flat `horizon_color`, no
  visible seam against the fog. Look at the horizon at eye level: confirm
  there's no seam between terrain fading into fog and the dome behind it
  — they should read as one continuous colour. Fly several thousand
  blocks in any direction and confirm the sky/sun/moon don't move,
  shrink, clip, or otherwise change (the sky camera's translation never
  moves from the origin — only rotation follows the main camera).
  Separately, check the moon is visible (opposite the sun, so likely
  below the horizon at noon — worth a quick look with the render distance
  or camera angle tweaked to catch it, or wait for 018's day/night cycle)
  and reads as a plausible full-moon disc, not stretched/cropped from the
  wrong cell of `moon_phases.png`. Finally, confirm the egui panels (save
  picker, block inspector, status bar, ...) still draw on top of
  everything with the second camera and the main camera's `ClearColorConfig::None`
  in place — nothing should look like it's rendering underneath the UI or
  vanishing behind the sky pass.
- [ ] **017 sun shadows: no acne, no peter-panning, and the toggle's cost is
  real.** `cargo run` (debug build) against the real save. Look at a wide
  stretch of flat ground at midday (default noon `SkyPalette`) and confirm
  it's clean — no moiré/banding patterns of self-shadowing on lit ground.
  Then look at flat ground with the sun near the horizon (rotate
  `SkyPalette::sun_direction` in code for now, or wait for 018's time
  slider if that lands first) and check again — grazing angles are where
  acne shows up worst. Separately, find a single block sitting on flat
  ground (a 1-block-tall step) and confirm its shadow touches the block,
  not detached/floating off to one side (peter-panning). If either shows
  up, `src/sky/shadows.rs`'s `SHADOW_NORMAL_BIAS` (raise for acne, the
  starting value is `3.0` against Bevy's default `1.8`) or
  `SHADOW_DEPTH_BIAS` (raise for peter-panning, starting value is Bevy's
  default `0.02`) needs adjusting — record whatever it ends up at, and why,
  in `finished_tickets/017-shadows.md`'s resolution, since these were
  chosen without ever being looked at. Finally, toggle the new "Shadows"
  checkbox in the status panel and read the FPS counter both ways at a
  fixed render distance/camera position — record the before/after numbers
  in the same resolution section (the ticket flagged this as potentially
  the single most expensive item in the whole lighting group).
