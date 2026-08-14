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
- [ ] **025 side-face UV winding fix: side textures read right-way-round,
  not rotated/mirrored, on every face.** `cargo run` (debug build) against
  the real save. Look at a block with an asymmetric or directional side
  texture from multiple angles (a log's bark grain, a crafting table, a
  furnace's front — anything where a mirror or a 90° rotation would be
  visible) and confirm it looks the same "handedness" on every side, not
  flipped/rotated on some faces relative to others (this was the original
  "side textures seem rotated by 90 degrees" report). Grass side overlays
  (014) are drawn through the same per-face UV path — worth a glance too,
  though they're mostly symmetric noise so a regression there would be
  subtler. Checklist: `finished_tickets/025-side-face-uv-winding-fix.md`.
- [ ] **018 day/night cycle: the slider actually looks like a day, and
  nothing fights the camera.** `cargo run` (debug build) against the real
  save. In the status panel's new "Time of day" section: (1) drag the
  slider across its full range and watch for colour *discontinuities* —
  the sky/fog/ambient should read as one smooth gradient through dawn,
  noon, dusk, and night, with no visible pop at any keyframe tick (0, 3000,
  6000, 9000, 12000, 13500, 18000, 22500) and no seam at the 24000->0 wrap.
  (2) Watch the sun/moon handover specifically: as the slider crosses dawn
  (ticks 0) and dusk (ticks 12000), confirm the shadow direction swings
  smoothly rather than snapping 180° in one frame, and that the world
  doesn't flash black or produce a garbage shadow direction right at the
  crossover (`time_of_day::moon_handover_direction`'s "never degenerates"
  test only checks the math stays non-zero, not that it *looks* right).
  (3) Set the slider to early morning (~ticks 1000-2000) and confirm
  shadows fall to the west, matching `tests::morning_shadows_fall_west`.
  (4) Confirm night is dark but still legible (not black) — per the
  ticket's deliberate "moonlight ~50-100 lux" choice, not a bug. (5) Try
  play/pause and each speed multiplier, and the four quick-jump buttons.
  (6) Confirm dragging the slider (and clicking play/pause/speed/quick-jump)
  doesn't also spin or fly the camera — same input-capture plumbing as the
  render-distance slider (007), should just work, but this is the first
  new panel control since that was wired up. If step (1) or (2) shows a
  discontinuity, the fix is almost certainly tuning
  `src/sky/time_of_day.rs`'s `keyframes()` table or `HANDOVER_WIDTH` — record
  whatever changes in `finished_tickets/018-day-night-cycle.md`'s
  resolution, since none of this was tuned by eye. Checklist:
  `finished_tickets/018-day-night-cycle.md`.
- [ ] **019 selection box: nothing to see yet — this check rides along with
  020.** 019 landed the `Selection` resource, the coordinate rules and the
  gizmo drawing, but nothing sets a selection until 020's click handler, so
  there is deliberately nothing on screen after 019 alone (see that
  ticket's "Deviation" note for why no throwaway hardcoded box was
  committed). When 020 lands, check these *019* concerns as part of its
  pass: (1) the box's faces sit exactly on block edges — not half a block
  off, not one block short — with the far faces covering the max block
  rather than stopping at its near corner; (2) the same box read from the
  north and then from the south looks identically placed, which is the
  check for the `bevy.z = -mc.z` sign (a flipped Z puts the box one block
  off in Z and only in Z, which is near-invisible from one viewpoint);
  (3) the depth-bias look — `src/selection/gizmo.rs`'s `DEPTH_BIAS` is
  currently `-1.0`, i.e. the box always draws in front of terrain. Decide
  whether that's right or whether terrain should occlude it (a small
  negative bias instead, e.g. `-0.01`, which still avoids z-fighting where
  the box is coplanar with block faces). Also judge `LINE_WIDTH`,
  `BOX_COLOR` and `ANCHOR_COLOR` against real terrain — the yellow/orange
  pair was picked without ever being looked at. Record whatever they end up
  at, and why, in `finished_tickets/019-selection-volume.md`'s resolution.
- [ ] **020 selection input: click anchors a box, the six keys push the face
  you pressed, and nothing fights the camera or egui.** `cargo run` (debug
  build) against the real save. Do 019's three checks above in the same pass
  — this is what finally puts a box on screen. Then:
  (1) **Anchor.** Left-click a block you can identify (a corner of a roof,
  a lone tree trunk) and confirm a 1x1x1 box lands on *that* block, with the
  orange anchor cube inside it, not on its neighbour and not one block
  toward the camera.
  (2) **Each direction, twice.** Facing **north**, press each of `→` `←` `↑`
  `↓` `PageUp` `PageDown` in turn and confirm the box grows on the side you
  pressed — `↑` must push the box *away* from you (north = −Z), `↓` toward
  you. Then turn the camera to face **south** and repeat the whole set: the
  keys must still move the same world-absolute faces, so now `↑` grows the
  box toward you. That reversal is the point of the check — it's both the
  test for the Z sign and the honest test of whether world-absolute arrows
  are usable at all. If facing south feels genuinely disorienting rather than
  merely unfamiliar, that's the signal to add the camera-relative remap the
  ticket deferred (it goes in front of `input::move_face`, which already
  takes a `Face` rather than a `KeyCode` for exactly this).
  (3) **Modifiers.** `AltLeft` + a direction pulls that face back in; hold it
  past the far side and confirm the box stops at a single block rather than
  inverting or vanishing. `ControlLeft` + a direction steps 16 blocks —
  eyeball it against chunk boundaries if you can see any. `Ctrl`+`Alt`
  together retracts a chunk at a time.
  (4) **Repeat rate.** Hold a direction key and confirm it starts repeating
  after a beat and then runs at a comfortable speed — fast enough to grow a
  40-block box without letting go, slow enough to stop on a block you want.
  If it's wrong, `REPEAT_DELAY` (0.3 s) and `REPEAT_INTERVAL` (0.05 s) in
  `src/selection/input.rs` are the knobs; record what they end up at in
  `finished_tickets/020-selection-input.md`'s resolution, since neither was
  tuned by eye.
  (5) **Build limits.** Hold `PageUp` until the top face stops, and
  `PageDown` until the bottom does — they should stop dead at y=319 and
  y=−64 rather than running away.
  (6) **Nothing fights.** Type into the coordinate-jump panel's fields and
  drag the render-distance and time-of-day sliders: the selection must not
  move at all. In orbit mode (`Tab`), drag the view around with left mouse
  and release over terrain — the selection must not re-anchor. Click a
  button in any panel with terrain behind it — same. Click at open sky and
  confirm the existing box *survives* (a missed click deliberately does
  nothing rather than clearing). `Escape` clears it.
  (7) **Legibility.** With a box grown well away from its anchor, judge
  whether it's clear which face the next keystroke will move. If it isn't,
  019 left face highlighting on the table for this ticket and it's still
  available as a follow-up.
- [ ] **021 selection panel: typed bounds move the box, and the panel doesn't
  fight 020's keys or the anchor click.** `cargo run` (debug build) against
  the real save; do this in the same pass as 020, since the panel is the
  fastest way to read whether the keys are doing what you think.
  (1) **Live readout.** Click a block, then press the face keys and watch the
  "Selection" panel: min/max, size, volume and anchor should all track the box
  every keystroke, and the anchor should stay put while min/max move.
  (2) **Typed edits.** Type a new min/max pair and press `Enter` (and again
  with the "Set bounds" button) — the box should jump to it. Type a min
  *above* the max on one axis and confirm the box comes out the right way
  round and the fields refill sorted. Type garbage (`abc`, `12.5`, an empty
  field) and confirm the red "Bounds must be whole numbers." line appears and
  the box does **not** move. Type a Y past the build limits (say `9999`) and
  confirm it comes back clamped to 319/−64.
  (3) **Arrow keys while typing** — the one this ticket most needs a human
  for. Click into any of the six fields and press `←`/`→`/`↑`/`↓`/`PageUp`/
  `PageDown`: they must move the *text cursor* and leave the selection box
  completely still. Then click away from the field and confirm the same keys
  move the faces again. (Same for `Escape` with a field focused: it should not
  clear the selection.) This is `EguiInputCapture::keyboard` doing its job;
  if it isn't, the gate in `src/selection/input.rs::drive_selection_keys` is
  where to look, not the panel.
  (4) **Clicking the panel over terrain.** Position the "Selection" window so
  terrain is behind it, then click its buttons, drag its title bar, and
  click-drag inside a text field to select text — the selection must never
  re-anchor to the block behind the panel. Same for the "Clear" button:
  it should clear the box and nothing else.
  (5) **Volume warning and the cap.** Hold `Ctrl`+a direction to grow a big
  box and watch the volume line: it should turn yellow with a "will be slow"
  note past 1,000,000 blocks (100x100x100), and red with "too large to
  export" past 16,000,000, at which point the "Export…" button greys out
  (it does nothing when enabled either — that's ticket 024). Judge whether
  those two thresholds are in the right place once 022 gives real extraction
  timings; they're `VOLUME_WARN`/`VOLUME_CAP` in
  `src/ui/selection_panel.rs`, chosen without measurement.
  (6) **The key legend.** Expand the "Keys" header and check the arrow
  glyphs render (`→ (Right)` etc.) rather than showing tofu boxes in egui's
  default font — if any do, the fix is dropping the arrow from
  `Face::key_label` in `src/selection/input.rs` and keeping the word.
- [ ] **022 blueprint extraction: orientation survives, and the box isn't
  limited to loaded chunks.** `cargo run` (debug build) against the real
  save. The "Export…" button in the Selection panel now runs an extraction
  and logs the result; nothing is written to disk yet (that's 023/024).
  (1) **Properties survive.** Select a small box containing
  orientation-sensitive blocks — a staircase, a log wall, a door, a
  repeater or a piece of rail. Hit "Export…" and read the palette lines the
  console prints (`block_viewer:   [7] minecraft:oak_stairs[facing=north,
  half=bottom,shape=straight]`). Point the block inspector at those same
  blocks and confirm the properties match what it reports. This is the
  whole reason 022 doesn't read `DecodedWorld` — if a stair comes back with
  no properties, or every stair in the box comes back identical, that's the
  failure the ticket exists to prevent.
  (2) **Past the render distance.** Grow a selection (Ctrl + a direction
  steps a chunk at a time) until part of it is well outside the loaded
  terrain — far enough that there's visibly no mesh there. Extract, and
  confirm the far part comes back as real blocks, not air: the palette
  should contain terrain block names, and the "distinct states" count
  should be similar to a same-sized box inside the loaded area. The region
  cache loads from disk on demand, so this is meant to work; if that half
  comes back as pure air, the column loop is silently swallowing a region
  error.
  (3) **Progress and responsiveness.** Grow a box past a million blocks
  (the panel turns the volume line yellow) and extract. The panel should
  show `Extracting… n / m chunk columns` and a moving progress bar, the
  window should keep rendering at a normal frame rate throughout, and
  terrain should keep streaming in if you fly while it runs — the
  extraction shares the region-cache lock with chunk loading and takes it
  one column at a time specifically so it can't stall streaming. Note the
  elapsed time the panel reports afterward: the measured figure is ~175 ms
  for 2.1M blocks off a warm disk, so anything wildly slower than that
  (especially if the frame rate drops with it) means the lock is being held
  longer than intended.
  (4) **Cross-check a couple of heights.** Pick two blocks at known
  coordinates at clearly different Y values (say a surface block and one
  deep underground), read them in the block inspector, then extract a box
  containing both and confirm the palette contains both names. This is the
  `sections`-indexing cross-check the ticket asks for — the automated test
  `extraction_agrees_with_the_block_inspectors_path_on_a_real_save` already
  does it for one 8x8x8 box, so this is only worth a minute.
  Resolution notes: `finished_tickets/022-blueprint-extraction.md`.
- [ ] **023 structure writer: an exported `.nbt` loads in Minecraft.** The
  one check for this ticket that can only happen in the game. Ticket 024
  hasn't wired the export button to a save dialog yet, so produce a file
  with the test that does it end to end against the real save:
  `cargo test writes_a_real_box_from_the_save -- --nocapture`. It prints
  the path it wrote to (in `%TEMP%`) and the palette it extracted — a
  16x16x16 box at the middle of the save's first region, Y 60-75, so
  expect terrain rather than a build. Then: copy the file into a world's
  `generated/minecraft/structures/` folder (create it if missing), load it
  with a structure block (place one, set it to Load mode, type the
  filename without `.nbt`, hit Load, then Place). Confirm (1) the detected
  size is 16 x 16 x 16, (2) the terrain matches what's actually at those
  coordinates in the source world — **not mirrored**, which is the thing
  most likely to be wrong: 019's `bevy.z = -mc.z` flip shows up as a
  Z-mirrored build and nowhere else, and (3) blocks with an orientation
  kept it. For (3) the printed palette is the guide — a terrain box will
  usually only have `grass_block[snowy=false]`, so for a real orientation
  check re-run the test with `origin` pointed at a staircase, a log wall
  or a door you know the coordinates of, and confirm those blocks come out
  facing the same way in the placed structure as in the source world.
  Once 024 lands, this is the same check done through the export button
  instead. Resolution notes: `finished_tickets/023-structure-nbt-writer.md`.
