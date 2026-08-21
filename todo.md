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
  (3) the depth-bias look — **answered, see ticket 026**: `-1.0` for the
  whole box turned out to be unreadable underground, and the box is now
  drawn twice (depth-tested solid, always-in-front dotted). What's left of
  this item is judging `LINE_WIDTH`, `BOX_COLOR` and `ANCHOR_COLOR` against
  real terrain — the yellow/orange pair was picked without ever being
  looked at. Record whatever they end up at, and why, in
  `finished_tickets/019-selection-volume.md`'s resolution.
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
  024 has now landed, so this is the same check done through the export
  button instead — the test is only the fallback if the dialog misbehaves.
  Resolution notes: `finished_tickets/023-structure-nbt-writer.md`.
- [ ] **024 save dialog: the export button writes a file where it says it
  does.** The reported symptom was a button that logged a line reading like
  success and wrote nothing at all; the point of this check is that it now
  writes something, somewhere findable. `cargo run --bin block_viewer`,
  select a small box (a dozen blocks is plenty), click **Export…**.
  Confirm: (1) a **native Windows save dialog opens**, filtered to `.nbt`,
  with the filename pre-filled as `blueprint_<x>_<y>_<z>.nbt` from the
  selection's minimum corner, and starting in the save's
  `generated/minecraft/structures/` if it has one, else in the save folder;
  (2) **the window keeps repainting behind it** — orbit the camera with the
  dialog open, and confirm terrain still streams. If the dialog gets *lost
  behind* the app window when you click the game, that's the missing
  `set_parent` call (deliberately skipped — it needs a direct
  `raw-window-handle` dependency and an `unsafe` handle fetch, and the
  panel's "Choosing a file…" line is the cheap mitigation); say so and it
  becomes its own ticket; (3) **cancel** the dialog — nothing is written,
  no error appears, and the button re-enables; (4) pick a name and confirm
  the panel ends on a green "Wrote N blocks to:" with the **full path**
  under it, that **Copy path** puts that path on the clipboard, and that
  the file is actually there on disk with a plausible size; (5) export a
  large selection (over a million blocks, so the panel's yellow warning is
  showing) and watch the **extraction progress bar advance** and then a
  spinner while it writes, rather than the UI freezing; (6) export to a
  location you can't write to (`C:\Windows\System32\` or a read-only
  folder) and confirm a **red "Export failed:" line naming the path**
  rather than a crash. Then run 023's in-game check above through this
  button. Resolution notes: `finished_tickets/024-save-file-dialog.md`.
- [x] **026 selection box legibility: the buried half is distinguishable.**
  Checked, and the answer was *no* — "I don't see which part is underneath
  the surface, is the double pass rendering working at all?". Diagnosed in
  ticket 029: both passes were running, but `SOLID_DEPTH_BIAS = -0.02` is a
  *proportional* bias (~12% of the view distance), so terrain never occluded
  the "depth tested" pass and the two drew identically. Superseded by the
  029 check below.
- [ ] **029 selection box, take two: the depth bias is now tiny.**
  `cargo run --bin block_viewer`, click a block on open ground, then extend
  the box downward into a hillside (arrow keys, or Ctrl + a direction for a
  chunk at a time) until part of it is underground. Confirm: (1) the part in
  open air is a **solid** line and the buried part is **dashed and dimmer**
  — if the whole box still looks solid the depth test still isn't happening,
  if the whole box looks dashed the solid pass is being occluded when it
  shouldn't be; (2) the boundary between the two styles tracks the terrain
  surface as you orbit, which is the cue the whole thing exists for; (3) the
  horizontal slice plates (every 16 blocks on world chunk boundaries,
  coarsening to 32 for a very tall box) read as depth rather than as clutter
  — if they're noise, `MAX_SLICES`/`SLICE_ALPHA` in `src/selection/gizmo.rs`
  are the knobs, and dropping them entirely is a legitimate outcome; (4)
  `SOLID_DEPTH_BIAS` (now `-0.0002`) against a box whose bottom face sits
  flat on a flat surface — flickering along that edge means it's too small
  (try `-0.001`); the box still floating in front of terrain it should be
  behind means it's still too large (try `-0.00005`). Note that this number
  is *proportional to distance*, so judge it far from the camera as well as
  near; (5) the dashed pass at range — if the buried box vanishes,
  `BURIED_ALPHA` (now 0.4) is too low; if the dashes read as a fat noisy
  halo around the solid line in open air, `BURIED_LINE_WIDTH` (now 4.0, and
  it sets the dash length too — 4px on, 4px off) is too wide. Record
  whatever the constants end up at in
  `finished_tickets/029-selection-depth-bias-is-proportional.md`. If the
  two-pass wireframe *still* isn't enough once the bias is right, the
  translucent fill under 026's "considered and not done" is the next thing
  to try.
- [ ] **027 lib + two binary shims: both binaries still behave.** The
  package now builds a lib plus `block_viewer` and `citybuilder` bins, so
  the check is that nothing moved semantically. (1)
  `cargo run --bin block_viewer` — the viewer must behave exactly as it did
  before: terrain streams in, every egui panel draws, the save picker
  switches saves, the selection box responds to clicks and arrow keys, and
  dragging a slider or typing in a coordinate field still doesn't fly the
  camera or move the selection (that last one is the `configure_sets`
  ordering that moved from `main.rs` into `viewer::run`, and it fails
  silently if it got dropped). (2) `cargo run --bin citybuilder` — a window
  on the same save, same terrain streaming, same free-flight camera, and
  **no UI panels at all**; that's milestone M1 and it's the whole of it. It
  has no egui, so confirm the camera still flies normally rather than being
  stuck (`camera::EguiInputCapture` defaults to "nothing captured", which is
  what should carry it). Ticket:
  `finished_tickets/027-lib-and-two-binary-shims.md`.
- [ ] **028 new save layout: the current-version save actually loads in the
  app.** `ranvil` now resolves the overworld's regions to
  `<save>/dimensions/minecraft/overworld/region` when a save has one
  (ticket `../ranvil/finished_tickets/026-dimension-folder-save-layout.md`),
  falling back to the old `<save>/region`. Verified headlessly via
  `cargo run --example region_info` in `../ranvil` (lists `nbt_test` with 29
  regions and reads a block), but not in the window: run
  `cargo run --bin block_viewer` and check that the save picker lists
  `nbt_test`, that selecting it streams real terrain in rather than leaving
  an empty world, and that the startup-issue message is gone. If you still
  have an older save around, load it too — both layouts are supposed to work.
- [ ] **030 citybuilder render floor: the underground is gone and nothing
  visible went with it.** (Only once 030 has landed — nothing to look at
  before that.) `cargo run --bin citybuilder` against the real save. Fly the
  camera over varied terrain — ideally past a **ravine, a cliff face and a cave
  mouth**, which are the three shapes the per-chunk `min(OCEAN_FLOOR)` floor is
  supposed to handle by dragging the whole chunk's floor down. Confirm: (1) no
  visible hole, missing face or "see-through" patch anywhere the camera looks
  from a normal city-building height; (2) no vertical seam or wall at chunk
  boundaries, which is what a neighbour column with a different floor would
  produce if the "below the floor reads as solid" rule in `block_at` isn't
  working; (3) water still reads right at a shoreline and out over an ocean
  (`OCEAN_FLOOR` is the sea *bed*, so an ocean chunk's floor should be well
  under the water, not at it). Then `cargo run --bin block_viewer` on the same
  save and confirm the world looks **exactly** as before — the viewer keeps
  rendering everything, and the automated identical-mesh test only proves that
  for the fixture column. If the margin (starting at one 16-block section)
  looks too thin or too generous, record what it ends up at in
  `finished_tickets/030-...md`'s Resolution, along with the measured
  before/after the ticket asks for.
- [ ] **ranvil 014 (+ 013): does Minecraft actually relight a chunk whose
  `isLightOn` we cleared?** **No longer a gate** — the decision is that this
  project never computes light, and a chunk that stays wrongly lit until the
  game gets round to relighting it is an accepted cost rather than a bug
  (see the roadmap's ordering advice). What this check buys now is knowing
  *which* fallback we're on: if the interior of the test roof is dark, the
  flag alone is enough and nothing more is needed; if it isn't, the next
  thing to try is deleting the affected sections' light arrays. Neither
  outcome stops W from being built. The code half is done and tested
  (`../ranvil/finished_tickets/014-relight-on-load-flag.md`); the flag is
  confirmed to be a root `TAG_Byte` on the real save (`DataVersion` 4438) and
  `set_blocks` now clears it. What's left can only be answered with the game
  open. **Back the world up first** — this writes to a real save.

  Produce an edited chunk. There's no UI for writing yet (that's W8), so the
  quickest route is a throwaway test in `../ranvil` that opens a region of a
  scratch world, `set_blocks` a solid roof — say a 16x16 slab of
  `minecraft:stone` a few blocks above a patch of open ground you can find
  again — and `save`s. A roof is the right shape because it makes all four
  questions visible at once: what's under it must go dark.

  Then load the world and check:
  (1) **Is the flag alone enough?** Under the new roof should be *dark*, and
  the roof should cast a shadow on the ground beside it. If the interior is
  still fully lit, clearing `isLightOn` is not sufficient on its own and the
  next thing to try is also deleting the affected sections' `BlockLight` /
  `SkyLight` arrays — note that vanilla only writes those for sections whose
  light isn't uniform (one chunk in `nbt_test` had `BlockLight` in exactly
  one of its 24 sections), so "delete them" means "delete the ones that are
  there", not "there is one per section". If *that* doesn't work either, the
  answer is still not a lighting engine: buildings stay wrongly lit until
  something else makes the game relight the chunk, and that's accepted.
  (2) **Does it stick?** Quit, and re-read the same chunk's `isLightOn`
  (a two-line test, or `mark_for_relight`'s inverse). It should read 1 again
  — the game relit and re-claimed it. If it's still 0, every load of that
  chunk pays to relight forever, and the write path needs to set the flag
  itself after some notion of "we know the light is right", which it can't.
  (3) **What does it cost?** Edit a few hundred chunks (a long thin run works)
  and note whether loading the world is visibly slower or hitches. The
  citybuilder will edit chunks by the hundred; if relighting is expensive
  that's a scheduling constraint downstream needs to know about now.
  (4) **Same trip, ranvil 013:** delete a chunk's whole `Heightmaps` compound
  (`ChunkRegion::remove_heightmaps`) instead of recomputing it, load the world,
  and see whether the game rebuilds it — check that grass/snow/rain land
  correctly and mobs don't spawn on lit ground. Deleting is what W4 does by
  default, on the same "let the game do it" reasoning as the lighting. This is
  now a *cheaper decision than it was*: ranvil 013 is implemented (packing
  verified against real game bytes; see its resolution), so if the game does
  **not** rebuild them, W4 swaps one call for
  `recompute_heightmaps(x, z, classify)` and moves on. Record the answer in
  both tickets' Resolution sections.

- [ ] **033 write safety: does our `session.lock` actually conflict with
  Minecraft's?** Needs the game installed, ten minutes, and nothing built —
  `../ranvil/tickets/025-verify-session-lock-against-a-running-minecraft.md`
  has the five steps and is where the answers get recorded. It's listed here
  too because ticket 033 now *refuses to write* on the strength of that lock,
  and step 5 in particular ("hold the lock via `SessionLock::acquire`, then try
  to open that world in Minecraft — the game should refuse") is the half
  `WriteSession` depends on: if the game opens the world anyway, holding the
  lock buys nothing and W6's guarantee shrinks to a probe. Do this before W8
  writes to a world you care about.

- [ ] **035 paint/fill command: the blocks are actually there when
  Minecraft opens the world.** This is the roadmap's W8 gate — the point of
  the whole W1-W7 chain was to reach a button that does this, and only a
  human opening the game can confirm it worked. **Back the world up first**
  (or run this against a scratch copy) — this writes to a real save.

  `cargo run` against the save, click a corner, extend the selection to a
  small box (a handful of blocks — nothing near `VOLUME_WARN`), type a block
  name into the new "Paint" section of the Selection panel (try one with
  properties, e.g. `minecraft:oak_stairs[facing=east,half=top]`, to check
  they aren't dropped) and click "Fill". Confirm:
  (1) the status line reports blocks/chunks/regions written, and the
  in-viewer mesh updates to show the new block(s) without a restart (034's
  live re-mesh, W7) — this is the one half a human doesn't need Minecraft
  open to check;
  (2) close the viewer (a `WriteSession` holds `session.lock` for as long as
  it's open — see the 033 item above) and open the same save in Minecraft:
  the filled blocks are there, at the right position (no mirroring — 019's
  Z flip is what the app's own selection gizmo already renders correctly,
  but the write path has its own coordinate arithmetic and this is its
  first real-world check), with the properties intact if you typed any;
  (3) the surrounding area is otherwise undisturbed, and nothing reads as
  corrupted (the world loads at all, chunks near the edit aren't missing or
  glitched);
  (4) try clicking "Fill" again with the world still open in Minecraft —
  it should refuse (033's `WorldIsOpen`), not write underneath the running
  game.

  If any of this is wrong, it's almost certainly one of W4's coordinate
  rules or W5's region routing, not this ticket's own code — `commit_fill`'s
  own tests already prove the write lands where a synthetic fixture says it
  should; what only this check can prove is that the fixture's assumptions
  match a real save.

- [ ] **030 citybuilder render floor: no visible hole, seam or missing face
  from above, and `block_viewer` looks unchanged.** The automated suite
  covers the arithmetic (`render_floor`, `snap_down_to_section`,
  `decode_chunk`'s section skipping) and the mesher's boundary-occlusion
  rule with synthetic fixtures, plus a real-region measurement — what it
  can't cover is what the real save's actual terrain looks like from above
  with a chunk of it missing underneath.

  `cargo run --bin citybuilder` against the real save. Fly/orbit the camera
  over a range of terrain — flat ground, a hillside, and if the save has one
  nearby, a ravine, cave mouth or cliff edge (the case the whole
  minimum-over-256-columns design exists for). Confirm: no hole punched
  through the ground anywhere the camera can see from above; no visible seam
  or z-fighting at chunk boundaries, including where two neighbouring chunks
  would plausibly have computed different floors (near a ravine or a steep
  slope); the framerate/load time visibly benefits versus flying the same
  path in `block_viewer` (the measured numbers in ticket 030's Resolution are
  ~70% fewer sections and ~4.6x faster on one region — this is the "does that
  hold up by eye too" check).

  Then `cargo run --bin block_viewer` over the same area and confirm it looks
  exactly as it did before this ticket — full underground included, no
  missing caves. If it doesn't, the bug is in `world::mesh`/`world::decode`
  shared code, not `city`'s override, since the viewer never sets
  `FloorPolicy::BelowSurface` at all.

- [ ] **039 blueprint asset catalogue: startup loads and logs it without
  panicking.** The automated suite covers the load/validate logic directly
  (missing directory, malformed files, size/palette rejection, the real
  `house01.nbt` fixture) — what it can't cover is the actual `city::run()`
  startup path with the real asset directory in place.

  `cargo run --bin citybuilder` against the real save. Confirm the console
  prints `block_viewer: loaded 1 building from assets/city/blueprints`
  followed by a `house01 — WxHxD (N states)` line, no `skipped` lines, and
  no panic before the window opens. Checklist: none yet — this is a small
  enough surface that the console output above is the whole check.
- [ ] **040 building definitions: startup loads and logs them without
  panicking.** Rides along with the 039 check above — same binary, same
  pass. After the catalogue line, confirm the console also prints
  `block_viewer: loaded 1 building definition from assets/city/buildings`
  followed by a `house01 — "House" tier 1, footprint WxD` line, no `skipped`
  lines, and still no panic before the window opens.
- [ ] **060/063 road type definitions: startup loads and logs them without
  panicking.** Same shape as the 039/040 check above. As of ticket 063,
  `assets/city/road_types/dirt.ron` exists but `assets/city/roads/dirt/` is
  still geometry-less (a `README.md` only, no real `.nbt` pieces — see that
  directory's own note), so `dirt` currently has **no piece loaded** and
  therefore isn't in `RoadCatalogue::styles()` yet. Confirm the console
  prints `block_viewer: loaded 0 road pieces from assets/city/roads`
  followed by `block_viewer: loaded 0 road types from assets/city/road_types`
  and a `block_viewer:   skipped assets/city/road_types/dirt.ron: style
  "dirt" has no geometry in the road catalogue` line — not a silent load —
  and no panic before the window opens. Worth repeating once real `.nbt`
  pieces land under `assets/city/roads/dirt/`: confirm it flips to
  `loaded 1 road type` with a `dirt — "Dirt Path" (speed 1, capacity 4)`
  line and the `skipped` line disappears.
- [ ] **043 city save/load: the real app lifecycle actually fires the
  save.** The automated suite covers `persistence::{save_city, load_city}`
  directly (round trips, the removed-highest-id `next_id` case, corrupt/
  version-mismatched files) — what it can't cover is whether Bevy's real
  `AppExit` fires from an actual window close on this machine, and whether
  the file lands where expected on a real save.

  `cargo run --bin citybuilder` against the real save. Confirm the console
  prints a `no save loaded, starting with an empty city` line only if no
  save was found, or otherwise nothing about a missing city file (a first
  run has no `citybuilder/city.ron` yet, which is silent — `load_city`'s
  `NotFound` path prints nothing, only `run`'s own no-save-loaded branch
  does). Close the window normally (the titlebar X). Confirm the console
  prints `block_viewer: saved city (0 buildings)` before the process exits,
  and that `<save folder>/citybuilder/city.ron` now exists on disk with
  `version: 1, next_id: 0, buildings: [], roads: []`. Run it a second time
  and confirm the console now prints `block_viewer: loaded 0 buildings
  from <path>\citybuilder\city.ron` — proving the load side reads back
  what the exit handler actually wrote, not just what a unit test
  constructed in memory. Also try closing via Alt+F4 and confirm the same
  save line appears — Bevy is expected to route both through the same
  `AppExit` event, but this is the one thing worth actually checking rather
  than assuming.

  **044 addendum**: same run, same window close -- confirm
  `<save folder>/citybuilder/journal.ron` now exists too, with
  `version: 1, entries: []` (silent on the console: `load_journal`/
  `save_journal_on_exit` only print when there is a non-empty journal or a
  real error, and today the journal is always empty -- nothing calls
  `record_placement`/`record_demolition` yet). This is the same "does
  `AppExit` really fire and land the file where expected" question 043 already
  answers above; ride along with that pass rather than running the app a
  second time just for this.
- [ ] **045 RTS camera: controls feel right at the window.** The automated
  suite covers `CameraMode::Rts`'s pan/rotate/zoom math and clamps directly
  (a bare `App`, `Time` advanced by hand, no real window) -- what it can't
  cover is whether the tuning (`rts_pan_speed_factor`, `rts_rotate_speed`,
  `rts_pitch_range`) actually feels usable, and whether Bevy's real cursor/
  input plumbing agrees with the test doubles.

  `cargo run --bin citybuilder` against the real save. Confirm the window
  opens already in the RTS rig -- panning/zooming/rotating, no Fly-mode
  flash first, no free-fly WASD. Checklist: `W`/`A`/`S`/`D` pan across the
  terrain (and the pan direction stays sane after rotating -- `W` should
  still mean "forward" from the camera's current facing, not always world
  `-Z`); `Q`/`E` rotate the view; holding the right mouse button and
  dragging also rotates (yaw and pitch), without hiding or warping the
  cursor the way `Fly`'s right-drag does in `block_viewer`; scroll zooms in
  and out and the zoom-out has a bottom (the view should never flip past
  looking straight down, nor come up level with the horizon -- that's
  `rts_pitch_range` holding at both ends); holding `Shift` while panning
  noticeably speeds it up; left mouse does nothing (reserved for E3/E4);
  `Tab` does nothing. Then open `block_viewer` on the same save and confirm
  *its* camera is untouched -- still starts in `Fly` mode with the free-fly
  controls exactly as before, proving `CameraStartMode`'s override is
  citybuilder-only.
- [ ] **047 ghost preview: it actually reads as a translucent, validity-tinted
  building at the cursor.** The automated suite covers `resolve_placement`/
  `resolve_ghost`/`ghost_mesh`'s decisions and the keyboard selection
  directly (a bare `App`/hand-built fixtures, no real window, no real
  materials rendered) -- what it can't cover is whether the ghost is
  actually legible against real terrain and real lighting.

  `cargo run --bin citybuilder` against the real save, with at least one
  real `.nbt` under `assets/city/blueprints` (`house01.nbt` from ticket 039
  is enough). Press `1` and confirm a translucent building-shaped mesh
  appears at the cursor, following it as the camera pans; confirm it's
  green over open, flat ground and turns red over terrain too steep for
  `fit_footprint` (a hillside). E4 (placement) doesn't exist yet, so there's
  no way to actually occupy a tile and check the red-on-occupied case this
  way -- that half is unit-tested (`resolve_ghost_shows_the_invalid_material_when_occupied`)
  and worth re-checking visually once E4 lands. Confirm pressing `R`
  visibly rotates the mesh 90 degrees each press (four presses back to the
  start) without the ghost momentarily vanishing or flashing the wrong
  mesh. Confirm `Escape` hides the ghost, and that switching back and forth
  between two different catalogue entries (if more than one `.nbt` is
  present) swaps the mesh cleanly. Watch the console once for the
  `can't rotate to DegN` line if any fixture building has an unrotatable
  property at a given rotation -- confirm the ghost simply hides for that
  combination rather than showing stale geometry or panicking. Also worth
  a glance: whether the translucency reads as "preview" rather than "glitchy
  z-fighting" where the ghost's lowest layer sits flush with real terrain
  (E2's up-to-one-block clip on uneven ground) -- this is a judgement call,
  not a pass/fail, and a candidate for tuning (depth bias, a higher alpha,
  an outline) if it looks wrong rather than a bug to fix blind.
- [ ] **048 commit: a placed building is actually in the save, and the
  height keys feel usable.** The automated suite covers `blueprint_edit`,
  the `y_offset` math, and a committed write against a synthetic fixture
  directly -- what it can't cover is a real click against a real save, or
  whether `Page Up`/`Page Down`/`Home` feel right at the window.
  **Back the world up first** (or run this against a scratch copy) -- this
  writes to a real save, same caveat as ticket 035's item above.

  `cargo run --bin citybuilder` against the real save, with `house01.nbt`
  (or another real `.nbt`) under `assets/city/blueprints`. Press `1`, aim
  the green ghost at open ground, left-click. Confirm: (1) the console
  prints a `placed house01 (N block(s) across M chunk(s))` line and the
  building appears in the viewport without a restart (W7's live re-mesh);
  (2) clicking again immediately on the same tile does nothing (occupied)
  rather than stacking a second building; (3) clicking on a *red* ghost does
  nothing at all -- no console line, no city entry. Then close the app
  (which saves `city.ron`/`journal.ron` on exit, per 043/044) and open the
  save in Minecraft: the building's blocks are actually there, right-side-up
  and not mirrored, and any interior air (a doorway, a window) reads as
  actually empty rather than the terrain that was under it -- this is
  `blueprint_edit`'s "air is written, not skipped" decision, and the one
  thing a synthetic fixture can't judge against a real hillside. Re-open the
  save with `cargo run --bin citybuilder` and confirm the building loads
  back from `city.ron` in the same spot (043's round trip, now fed by a
  real placement for the first time).

  Separately, test the height keys: select a building, press `Page Up` a
  few times and confirm the ghost visibly rises one block per press (and
  `Page Down` lowers it, `Home` snaps back to the terrain fit); place one
  raised a block or two and confirm the gap underneath it is real floating
  space in Minecraft, not filled in. Try placing with a *negative* offset
  into a slope and confirm it doesn't corrupt anything, just buries the
  lower courses. Finally, try a commit while the previous one is still
  writing (rapid double-click on adjacent valid tiles) and confirm the
  second click is silently ignored rather than racing the first -- the
  console should show only one `placed ...` line per click that actually
  went through.

- [ ] **049 demolish: a demolished building is actually gone from the save,
  and the terrain underneath is really back.** The automated suite covers
  target resolution, the `City`/journal glue on success and failure, and a
  restore write against a synthetic fixture directly -- what it can't cover
  is a real `Delete` press against a real save, or the write-gate refusal
  actually being reachable through two real inputs in the same session.
  **Back the world up first** (or run this against a scratch copy) -- this
  writes to a real save, same caveat as tickets 035 and 048's items above.

  `cargo run --bin citybuilder` against the real save (ideally the one 048's
  checklist already placed a building into). Hover the placed building and
  press `Delete`. Confirm: (1) the console prints a
  `demolished house01 (N block(s) restored across M chunk(s))` line and the
  building disappears from the viewport without a restart (W7's live
  re-mesh); (2) the tile is immediately buildable again (select the same
  building with `1` and confirm the ghost goes green there); (3) pressing
  `Delete` again over empty ground or a road tile does nothing at all -- no
  console line. Then close the app (saves `city.ron`/`journal.ron` on exit)
  and open the save in Minecraft: the terrain where the building stood
  should read as whatever was there before it was placed, not a hole and not
  leftover building blocks. Re-open with `cargo run --bin citybuilder` and
  confirm the building does *not* reappear (043's `city.ron` round trip,
  now proving a removal survives a reload too).

  Separately, try to catch the write-gate case: place one building, and
  before its `placed ...` console line appears, immediately hover a
  *different*, already-standing building and press `Delete`. This is a
  narrow timing window (the write is usually fast), so it may take a few
  tries -- confirm that if it does land mid-write, the console shows the
  "can't demolish right now, a write is already in progress" line rather
  than a corrupted region file or two writes racing silently. Not a hard
  failure if the window can't be hit by hand; the automated `write_gate`
  tests already cover the logic itself.

- [ ] **050 build menu and city panel: the first real UI actually reads,
  clicks and undoes correctly.** The automated suite covers the pure logic
  (requirement satisfaction, sort order, cost/production formatting,
  building counts, the write-status/undo state machines) directly; what it
  can't cover is whether the two egui windows actually render sensibly and
  whether a real click drives the same pipeline the number keys already
  proved out in 047-049. **Back the world up first** (or run this against a
  scratch copy) -- this can write to a real save, same caveat as every
  other write-path checklist item above.

  `cargo run --bin citybuilder` against a real save with at least two
  entries in `assets/city/buildings` where one `requires` the other.
  Confirm: (1) the Build window lists tiers with headings, shows cost
  ("Cost: 40x oak_planks" or "Cost: Free") and, if the fixture has one, a
  production line; (2) the locked entry (whose `requires` isn't built yet)
  is greyed out and hovering it shows "Requires: <name>", both in the
  tooltip and as a standing "Locked -- requires…" line under the row; (3)
  clicking an *unlocked* entry selects it exactly like pressing its number
  key would -- the ghost preview appears and `R`/`PageUp`/`PageDown`/`Home`
  still work on it. Place the requirement building via a click (not a
  number key) and confirm the previously-locked entry unlocks live, no
  restart needed.

  Then check the City window: building counts update after each placement,
  the road tile count matches (0 if F hasn't landed yet), and the "Last
  edit" section shows "Placed <id>: N block(s) across M chunk(s), K region
  file(s) — not yet saved to disk" in green right after a successful
  placement (ticket 051 moved the backup/region-write reporting to the new
  "World save" section below it — see that ticket's own checklist item).
  Demolish that building (via `Delete`, same as 049) and confirm the status
  line updates to "Demolished …". Finally, click "Undo" and confirm: the console shows
  an "undid …" line, the write-status section shows "Undid …", the building
  reappears (if the last journal entry was the demolition) or disappears
  (if it was the placement), and clicking Undo again with an empty journal
  shows "Undo failed: there is nothing to undo" rather than doing nothing
  silently. Also worth trying once: click Undo with no save loaded (the
  citybuilder's `empty_save` placeholder) and confirm it fails cleanly
  rather than panicking.

- [ ] **051 defer world writes to a manual Save: placements don't touch
  disk until clicked, and nothing is lost on quit.** **Back the world up
  first** (or run this against a scratch copy) — this can write to a real
  save, same caveat as every other write-path checklist item above.

  `cargo run --bin citybuilder` against a real save. Note the `.mca` files'
  modification times for the region(s) you're about to build in (or just
  watch the directory). Place two or three buildings in the same region
  without clicking "Save world": confirm (1) the ghost/live mesh updates
  immediately for each placement, same as before this ticket, (2) the City
  panel's "Last edit" section says "not yet saved to disk" after each one,
  (3) the "World save" section reports the growing unsaved-region count,
  and (4) the region file(s) on disk are untouched (mtime unchanged) the
  whole time. Then click "Save world": confirm the status line reports the
  regions written and any new backups, the unsaved-region count drops to
  0, and the `.mca` file(s) now have a fresh mtime. Open the world in
  Minecraft afterward and confirm the buildings are actually there.

  Then the exit-safety half: place a building, do **not** click "Save
  world", and quit the app (close the window). Reopen `cargo run --bin
  citybuilder` against the same save and confirm the building is still
  there and the region file was in fact written (mtime updated at quit
  time, not at the placement). Console output on quit should show a
  "flushed N unsaved region file(s)" line (or nothing, if you did click
  Save World before quitting — confirm that case doesn't double-flush or
  error).

- [ ] **055 drag-to-build roads: the tool toggle, the drag preview, and a
  real commit.** Mostly moot until real `.nbt` road pieces exist in
  `assets/city/roads` (`isolated.nbt`, `dead_end.nbt`, `straight.nbt`,
  `corner.nbt`, `t.nbt`, `cross.nbt`, each exactly 6 blocks on `x`/`z` —
  see ticket 054) — without them every cell falls back to the flat quad
  preview and nothing gets written to the world, only recorded in `City`.
  **Back the world up first** (or run this against a scratch copy).

  Without real pieces (the state today): `cargo run --bin citybuilder`
  against a real save, press `T` and confirm the console/city panel imply
  the road tool is active (the building ghost should stop appearing even
  with a catalogue entry selected). Left-click-drag across flat ground and
  release: confirm a translucent flat quad follows the cursor per cell
  while dragging, tinted green over free ground and red over an occupied
  tile or a building footprint, and that releasing over invalid ground
  refuses the whole drag (console message, no cells added — check the City
  panel's road-cell count doesn't move). Release over valid ground and
  confirm the City panel's road-cell count goes up by the number of cells
  dragged over, and the console says "not yet saved to disk" is *not*
  printed (no real write happened — only a "no road catalogue loaded"/"no
  matching road pieces loaded" line). Press `T` again and confirm building
  placement (ghost + click to commit) still works exactly as before this
  ticket.

  Once real pieces exist: repeat the same drag and confirm (1) the preview
  shows an actual rotated piece per cell rather than a flat quad, switching
  shape live as you drag past a bend; (2) on release, the blocks actually
  land (open the world in Minecraft after a "Save world" click and check);
  (3) dragging a new branch off an existing straight run re-renders the
  existing cell it attaches to as a T or a corner, not left as whatever it
  was; (4) killing the write (e.g. drag into an ungenerated chunk) leaves
  the City panel's road-cell count unchanged and the console reports a
  rollback of only the newly-attempted cells, not any pre-existing road.

- [ ] **057 terraforming: dig and level tools.** **Back the world up first**
  (or run this against a scratch copy) — this writes to a real save.

  `cargo run --bin citybuilder` against a real save. Press `T` twice to
  cycle to the Terraform tool (Building -> Road -> Terraform); confirm the
  building ghost and the road drag preview both stop reacting to clicks
  while it's active. With the default Dig mode, left-click-drag a small
  rectangle over varied terrain (a hill, a tree) and release: confirm the
  console reports a tile count and block count, the City panel's "Last
  edit" section shows a "Shaped" line ("not yet saved to disk"), and the
  live mesh updates immediately to show the topmost layer cleared across
  the dragged rectangle — including a tree's trunk/leaves, which should
  clear same as stone would. Drag over the same spot again and confirm it
  digs one layer deeper each time.

  Press `Z` to switch to Level mode (no on-screen indicator exists yet —
  watch the console/city-panel line after a drag to tell which mode is
  active). Start a drag on a low or high point and release on an uneven
  patch nearby: confirm every tile in the rectangle ends up visually flush
  with the tile the drag started on — high points cut down to bare dirt/
  stone at that height, low points filled up with dirt, the starting tile
  itself untouched. Try a drag that's already flat and confirm the console
  says there's nothing to change (no edit dispatched).

  Click "Save world" and confirm the region(s) actually get written (mtime
  changes), then open the world in Minecraft and confirm the dig/level
  results are really there — including that a levelled area's fill reads
  as ordinary dirt, not some placeholder block. Also confirm `T` still
  cycles back to Building afterward and placement/road tools are unaffected.

- [ ] **059 multiple road styles: the `[`/`]` style cycle.** Mostly moot
  until real `.nbt` road pieces exist for more than one style under
  `assets/city/roads/<style>/` (see ticket 054's own still-open note) —
  without them every cell falls back to the flat quad preview regardless
  of which style is selected, so there's nothing to *see* differ yet. The
  one thing worth checking without real assets: `cargo run --bin
  citybuilder` against a real save, press `T` to the Road tool, and
  left-click-drag a cell with no `RoadCatalogue` loaded at all (the normal
  case today, no `assets/city/roads` on disk) — confirm the drag still
  commits (console/City panel road-cell count moves), since a missing
  catalogue is a "nothing to render" case, not a "no style selected"
  refusal.

  Once at least two real styles exist (e.g. `assets/city/roads/dirt/` and
  `assets/city/roads/paved/`, each with all six pieces): confirm pressing
  `[`/`]` while the Road tool is active visibly changes which piece the
  drag preview shows (no on-screen style name exists yet — go by the mesh
  itself), that the *first* loaded style is already selected the moment
  you switch to the Road tool with no keypress, that `[`/`]` wrap around
  at both ends, and that they do nothing while a different tool is active
  or while typing in an egui panel. Drag two separate runs with two
  different styles selected, confirm both commit and render correctly
  side by side (each keeps the style it was built with even after
  switching the selection and dragging elsewhere), then click "Save
  world", reload the save, and confirm each run still shows its own style
  after the reload (persistence round-trip, not just the in-memory
  session).

- [ ] **061 definition hot reload and error panel: a live edit actually
  gets picked up, and a bad file actually shows up in the panel.** The
  automated suite covers `dir_snapshot`'s file-level detection directly
  (temp directories, hand-set mtimes) — what it can't cover is a real
  editor saving a real file while the game is running, or whether the new
  "Definition Errors" egui window actually reads as useful rather than
  just present.

  `cargo run --bin citybuilder` against a real save with at least one real
  `.ron` under `assets/city/buildings` (`house01.ron` from ticket 040 is
  enough). Confirm the "Definition Errors" window is there (collapsed,
  title "Definition Errors", body "(no problems)" if expanded) alongside
  the Build menu and City panel. With the game still running, open
  `assets/city/buildings/house01.ron` in a text editor and change its
  `name:` field, save it, and switch back to the game window within a
  couple of seconds — confirm (1) the console prints a `reloaded 1 building
  definition from assets/city/buildings` line without you touching
  anything in-game, and (2) the Build menu's entry for that building now
  shows the new name, live, no restart. Then break the file on purpose —
  change `tier: 1` to `tier: "one"` (a type error) or delete the trailing
  `)` — save again, and confirm: the console prints a `skipped
  assets/city/buildings/house01.ron: ...` line, the "Definition Errors"
  window's title-bar dot/expand shows something changed (it auto-expands
  the first time an error exists after a clean start — check this only
  fires once, not on every subsequent poll), and the body lists the file
  path and a parse-error message under a "Buildings" heading. Fix the file
  back and confirm the error clears within a second or two without a
  restart, and the Build menu's entry comes back. Repeat once for
  `assets/city/road_types/*.ron` if a fixture exists there, to check the
  "Road types" heading and `RoadTypes` reload path independently of the
  buildings one. Finally, confirm editing an unrelated file in either
  directory (e.g. touching a stray `.txt`, or a `.nbt` in the sibling
  geometry directories) does **not** print a reload line — only `.ron`
  changes in `assets/city/buildings`/`assets/city/road_types` should
  trigger anything.

- [ ] **062 fix stale "queued chunks" count and chunks that never load.**
  Reported symptom: loading a new world showed "Queued chunks: 850" and the
  world only displayed some chunks — and after the first fix (the counter
  not draining `PendingChunkWork::to_load`) landed, still no visible change,
  meaning that was cosmetic, not the real blocker. Two real causes were
  found and fixed — see `finished_tickets/062-fix-stale-queued-chunks-count.md`
  for the full diagnosis:
  (1) `RegionCache::failed` permanently blacklisted a region after any single
  load failure (plausible on Windows: Explorer/antivirus/another process
  transiently locking a `.mca` file during the initial burst of ~850
  concurrent-ish loads) — now expires after a 10s cooldown and gets retried;
  (2) the streaming diff only ever recomputed on a chunk-boundary crossing —
  now it also force-recomputes every 2s regardless of camera movement, so
  anything still desired-but-missing gets retried rather than needing the
  camera to move.

  `cargo run --bin block_viewer` against the real save, ideally with the
  render distance slider pushed up toward 14-20 before switching to a save
  (or right after opening) so there's a large backlog to watch. Confirm:
  (1) "Queued chunks" visibly counts *down* over the following seconds
  rather than sitting fixed at one number; (2) it eventually reaches 0 (or
  close to it — chunks at the edge of explored terrain that aren't fully
  generated yet correctly never enter the count, that's expected) while
  "Loaded chunks" climbs to fill the render-distance square; (3) all of the
  visible terrain in view actually renders in — no permanently-missing
  patch inside the render distance once the counter settles at/near 0,
  including watching for a while (up to ~15-20s) in case a region's failure
  needed a cooldown-expiry retry to clear. **If it still gets stuck**, check
  the console for `block_viewer: skipping region (...) — failed to load
  ...` lines repeating for the *same* coordinate well past the 10s cooldown
  (would mean the region genuinely, permanently can't load — a real disk/
  permissions problem worth its own ticket, not this pipeline) versus no
  such lines at all with terrain still missing (would mean the bottleneck is
  purely throughput: the shared `Mutex<BlockRegistry>`/`Mutex<BiomeRegistry>`
  serializing every chunk's decode+mesh — see `chunk_pipeline`'s module
  docs — needs longer than expected, or a per-frame dispatch budget, for a
  backlog this size).


## 064 — save selection on the citybuilder command line

Claude can't run the app; this needs a human at the window. The argument
forms below are all verified to resolve correctly by unit test against your
real saves directory — what still needs eyes is that the *window* opens on
the right world.

Run each and check the **City** panel's "World" section at the top says what
you asked for (name plus region count), and that the terrain around the
camera is that world's:

1. `cargo run --bin citybuilder` — unchanged: the first save found.
2. `cargo run --bin citybuilder -- nbt_test` — the named save.
3. `cargo run --bin citybuilder -- NBT_TEST` — same world (case-insensitive).
4. `cargo run --bin citybuilder -- typo_world` — a **red** message naming
   `typo_world` and listing the saves that do exist, on an empty world
   rather than silently opening a different save.
5. `cargo run --bin citybuilder -- "<full path to a save folder>"` — loads
   that save directly, including one outside any `saves/` folder.
6. `cargo run --bin block_viewer -- nbt_test` — the viewer opens on the same
   save, and its Save picker window still lists and switches saves.

Also worth confirming once: place a building in a save opened by name, hit
"Save world", quit, and reopen that same save by name — the city comes back
(`city.ron`/`journal.ron` are read from the *selected* save's root).

## 065 - roads visible at ground level (manual/visual check)

Roads were being written at a hardcoded Y=0 (buried in the deepslate), so
every road ever built in the citybuilder was invisible. Fixed to write at the
cell's fitted ground, with the piece's surface course flush with the terrain.
Needs a human at the window:

`cargo run --bin citybuilder -- nbt_test`, press `T` for the road tool, then:

1. **Drag a straight run on flat ground.** The dirt path should appear at
   ground level, flush with the surrounding grass — not floating a block
   above it, not sunk a block into it, and not missing entirely.
2. **The ghost should stand where the blocks land.** While dragging, the
   translucent green preview should sit exactly on the terrain the road ends
   up occupying — in particular it must not be slid ~3 blocks diagonally off
   the cell (the corner-vs-centre anchoring this ticket also fixed; it was
   invisible while every cell previewed as the flat fallback quad).
3. **Drag an L.** The corner piece should orient correctly and the two
   straights either side of it should meet it cleanly, all at the same level.
4. **Extend an existing road by one cell.** The previously-placed dead end
   re-tiles to a straight — check it stays at its original height and does
   *not* climb a block. Repeat two or three times: any upward staircase means
   the stored `base_y` isn't being respected somewhere.
5. **Build across a slope.** Each 6x6 cell fits to its own lowest ground, so
   a step between cells is expected — what to look for is whether it's
   tolerable or whether roads need real slope handling (roadmap F3's
   acknowledged iteration-2 gap).
6. **Save, quit, reopen.** The roads come back at the same height.

Note: any road built with a *previous* build of the app is still sitting at
Y=0 in the world and won't be cleaned up by this fix — see the ticket.

- [ ] **066 road piece rotation: corners, dead ends and Ts point the right
  way.** `cargo run --bin citybuilder -- nbt_test`, press `T` for the road
  tool. `canonical_pattern` was claiming a different orientation than
  `assets/city/roads/dirt/*.nbt` were actually exported at, so every dead end
  and corner came out 180° off and every T 90° off. A test now reads the real
  `.nbt` files and checks the table against them
  (`road_catalogue::the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations`),
  but "does the bend actually bend the right way on screen" still needs eyes:

  1. **Drag an L.** The corner cell should turn *into* both of its
     neighbours — the paving continuous through the bend, the grass shoulder
     on the outside of it. A corner rotated 180° looks like a bend pointing
     into two empty cells with grass where the road should join.
  2. **Place a single cell, then extend it by one.** The first cell is a dead
     end: its stub should open toward the cell you extend into, not away from
     it.
  3. **Build a T** (a straight run with one cell branching off the side). All
     three arms should meet the paving; the closed arm should be the one with
     no neighbour.
  4. **Known asset gap, not a code bug:** a *lone* cell with no neighbours
     renders as a south-pointing dead-end stub, because `isolated.nbt` is a
     byte-identical copy of `dead_end.nbt`. Re-export a real island piece in
     Minecraft if you want one.

- [ ] **067 road height: a drag builds one continuous road, not a staircase
  of cliffs.** Same launch. Roads used to fit each 6x6 cell to its own
  ground; now the whole drag gets one level (from its first cell), with
  optional 4-block steps via `stairs.nbt`.

  1. **Drag a long run across a slope.** Every cell should sit at the same Y
     — the one the cell you *started* the drag on was fitted to. No
     one-block steps between cells. Expect the road to be buried where the
     hill rises above it and to stand proud where the ground falls away:
     that's the rule working, and the dig/level tools (`T` again) are the
     current answer to it.
  2. **The ghost previews the same level.** While dragging, the translucent
     preview should already be flat at the start level, not following the
     terrain.
  3. **Start a second drag on (or beside) the end of the first.** The new
     road should continue at the *existing* road's height, not drop back to
     the ground under it. That's `anchor_level`; a seam means it isn't
     firing.
  4. **Ramps** — `assets/city/roads/dirt/stairs.nbt` is in (ticket 068), so
     this is live. Drag a run whose two ends are ~4+ blocks apart in height.
     One stair cell should appear in the middle of the run, its low end flush
     with the flat road behind it and its top step flush with the flat road
     ahead — walkable end to end, no lip at either join. ~8+ blocks apart
     should give two ramps, spread rather than stacked. A run too short for
     the climb should refuse the *whole* drag with a message in the console
     ("the ends are N step(s) apart ... only M cell(s) on the path can be
     stairs") rather than building a cliff.
  5. **Save, quit, reopen.** Heights and stairs come back. Note: an existing
     `<save>/citybuilder/city.ron` is version 4 and will be *refused* on load
     (`city save is version 4, this build reads version 5`) — delete it to
     clear the message; the buildings and roads it recorded go with it.

- [ ] **069 road placement over older chunks writes instead of refusing.**
  `cargo run --bin citybuilder nbt_test` — that save is a real patchwork:
  9327 overworld chunks at `DataVersion` 4438, 12 at 4440, 9215 at 4903,
  mixed region by region (`r.-2.-1.mca` is 1024/1024 on 4438;
  `r.0.-2.mca` is 874/874 on 4903; `r.0.0.mca` is 479 against 383). The
  shipped road pieces are all 4903, and before this ticket every cell
  landing on a 4438 chunk refused the drag with `chunk (x, z) is
  DataVersion 4438, the edit is 4903` and placed nothing. Confirm:
  1. **Build a road over the old half** — head west/south-west (negative
     X, around chunk X -64 and below, region `r.-2.-1`/`r.-2.-2`) and drag
     a road. It should place, and the console should report blocks
     written, with no `DataVersion` line at all.
  2. **Build a road straddling the seam** — `r.0.0` and `r.1.-1` hold both
     versions. A drag crossing from 4903 chunks onto 4438 ones should
     commit as one transaction, not fail partway.
  3. **Save, then open the world in Minecraft.** The point of the change:
     blocks written into the 4438 chunks have to load and look right after
     the game's own datafixers run over them. Walk the road built in step
     1 — dirt path, stairs and all — and confirm nothing came back as
     `air`, a missing block, or the wrong stair facing.

- [ ] **070 citybuilder render distance + lazy unloading.**
  `cargo run --bin citybuilder` (release if the debug build is too slow to
  judge frame rate).
  1. **Distance** — terrain should reach visibly further than before
     (`RenderDistance(16)` against the viewer's 10: a 33x33-chunk square,
     ~528 blocks across, against 21x21/~336). Fog and the far plane follow
     it automatically, so the horizon should still fade into fog rather
     than ending at a hard edge, and shadows should look unchanged (the
     cascade distance is capped well below the far plane). Watch the frame
     rate while panning — if it's meaningfully worse than at 10, say so;
     the render floor is supposed to be paying for this.
  2. **No thrash at the boundary** — pan a few chunks in one direction and
     straight back, inside half a minute. Nothing should reload: no hitch,
     no chunk visibly popping back in, and the console should print no
     `Chunk streaming: ... to load` line for ground you already had. This
     is the hysteresis margin (2 chunks) plus the 30s grace.
  3. **The grace does expire** — pan far away (several render distances),
     park for a minute, and watch memory settle back down rather than
     climbing forever. In `block_viewer` (not the citybuilder) the Status
     panel now reads `Loaded chunks: N (M lingering)`; M should rise as
     you move and fall back toward 0 about 30 seconds after you stop.
  4. **A long flight stays bounded** — in `block_viewer`, fly in one
     direction for several minutes without stopping. `M lingering` should
     plateau at 256 (the cap shedding the farthest ones early) rather than
     growing with the length of the flight, and RSS should plateau with it.

- [ ] **071 road tunnel tiles.** Needs the `-tunnel.nbt` exports first — the
  code path is complete but inert until at least
  `assets/city/roads/dirt/straight-tunnel.nbt` exists (see that directory's
  `README.md` for what a tunnel piece has to keep identical to its surface
  twin). Once one is in, `cargo run --bin citybuilder` and:
  1. **It triggers where it should** — drag a road straight into the side of
     a hill. The cells that end up buried should preview as the *tunnel*
     piece (the ghost and the written blocks share `plan_tunnels`, so a
     mismatch here is a real bug), and the cells out in the open should not.
     The boundary is "more than 18 of the 36 columns just above the piece are
     not air", so a road grazing a low bank should stay a surface road.
  2. **It's walkable** — save, open the world in Minecraft, and walk the
     road through the hill end to end. No suffocation, no stone left hanging
     where the bore should be, and the portal at each end shouldn't step up
     or down (the tunnel piece's subgrade/surface layers have to sit where
     the surface piece's do).
  3. **The tunnel survives a re-tile** — this is the regression the whole
     "store the variant" design exists for. Build a second road that
     branches off a cell *next to* the tunnel, so the tunnel's neighbour is
     re-tiled and the tunnel itself is rewritten. The bore must still be
     there afterwards, not filled back in with hillside.
  4. **Old saves are refused, not misplaced** — a `citybuilder/city.ron`
     written before this change is version 5; loading it should print the
     version-mismatch line rather than silently loading roads.

- [ ] **072 material stock.** `cargo run --bin citybuilder`. Nothing charges
  or credits the stock yet (that's 073) — this check is that the plumbing is
  visible and survives a restart.
  1. **The panel says so** — the City window has a "Stock" section reading
     `(nothing stockpiled)` on a fresh save, below "Roads".
  2. **A bad table is visible, not fatal** — put a deliberate typo in
     `assets/city/drops.ron` (e.g. delete a closing brace) and relaunch. The
     game should still start, the console should say every block will drop
     itself, and the "Definition Errors" window should list the file under a
     "Drops" heading. Undo the typo afterwards.
  3. **It round-trips** — hand-write
     `<save>/citybuilder/stock.ron` as `(version: 1, items: {"minecraft:dirt": 12})`,
     relaunch, and confirm the panel shows `12x dirt`; close the game and
     confirm the file is still there and still says 12.

- [ ] **073 cost and yields.** `cargo run --bin citybuilder`. Note the
  bootstrap gap first: a fresh save has an empty stock and the shipped
  `house01` costs 40 oak_planks + 20 cobblestone, so **start by
  terraforming** (`T` to the terraform tool, drag to dig) — dug blocks are
  what pays for everything else until there's a starting grant.
  1. **Digging pays** — dig a patch of grassy hillside and watch the City
     window's "Stock" section fill with dirt and cobblestone (grass_block
     gives dirt, stone gives cobblestone — `assets/city/drops.ron`).
     Levelling *up* should take dirt back out again.
  2. **A cost you can't meet refuses before anything happens** — with an
     empty stock, the build menu's cost line for House should be red and
     read "(short 40 more oak_planks, ...)"; clicking to place should place
     nothing, and the City window's "Last edit" should say why.
  3. **A placement pays and is paid** — hand-edit `stock.ron` to give
     yourself the planks and cobblestone (or dig first), place a House on a
     slope, and confirm the cost leaves the stock while the terrain it
     displaced arrives in it.
  4. **Undo is exact** — note the stock, place a building, undo it from the
     City panel, and confirm the stock is back to the same numbers.
  5. **Demolish is not a farm** — place a building, demolish it, and confirm
     the restored terrain was charged back (the dirt you gained on placing
     it goes away again). Repeating place/demolish should not grow the pile.

- [ ] **074 founding stock and conversions.** `cargo run --bin citybuilder`.
  1. **A new city is founded with materials** — on a save with no
     `citybuilder/stock.ron`, the console should list the grant and the City
     window's "Stock" section should show 512 dirt, 256 cobblestone, 128
     oak_planks, 64 oak_log (`assets/city/economy.ron`). Place a House and
     confirm the planks and cobblestone leave the pile.
  2. **An emptied city is not refilled** — spend or hand-edit the stock down
     to `(version: 1, items: {})`, relaunch, and confirm the stock stays
     empty. The grant is for a save with *no* stock file, not an empty one.
  3. **Logs pay for planks** — hand-edit `stock.ron` to hold only
     `minecraft:oak_log` (say 40) and nothing else. The build menu's House
     row should be affordable, with an orange "Converts: 10x oak_log" line
     under its cost; placing should eat the logs and leave the spare planks
     in the pile (40 planks made, 40 spent — so with 10 logs exactly, none
     spare; try 41 planks' worth to see the remainder).
  4. **Undo returns the logs** — note the stock, place that House, undo it
     from the City panel, and confirm the *logs* come back rather than
     planks.
  5. **A broken table is visible, not fatal** — typo `economy.ron`, relaunch,
     and confirm the game starts with no grant and no conversions and the
     "Definition Errors" window lists it under "Economy".

- [ ] **075 interchangeable materials.** `cargo run --bin citybuilder`. Hand-
  edit `citybuilder/stock.ron` to hold only `minecraft:birch_log` (say 40)
  and nothing else, then relaunch. The build menu's House row should read
  affordable with an orange "Converts: 10x birch_log" line under its cost;
  placing should eat the birch and leave the spare oak planks in the pile
  (the stock keeps saying `birch_log` for what you cut — only the payment is
  type-blind). Worth a second pass with `stripped_spruce_log` and
  `cherry_wood`, which route through the same group.
