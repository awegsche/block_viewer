# 016 - Skybox: gradient dome, sun and moon

## Status
Open

## Depends on
015 (`SkyPalette` — this ticket draws what that one stores).

## Goal

Replace the flat clear colour with an actual sky: a vertical gradient from
zenith to horizon, a sun disc, a moon, and stars at night. It has to stay
cheap to *re-colour* every frame, because 018 will be driving it from a
clock.

## Approach: a skydome on its own camera layer

Recommended over the alternatives (see below).

- A second `Camera3d` with `order: -1`, `clear_color:
  ClearColorConfig::Custom(zenith)`, rendering only `RenderLayers::layer(1)`.
- The main camera changes to `ClearColorConfig::None` so it draws terrain on
  top of the sky pass instead of erasing it.
- On layer 1: an inverted UV-sphere mesh (normals pointing inward), radius
  ~100, at the origin, with an unlit `StandardMaterial`, plus
  `NotShadowCaster` and `NotShadowReceiver` (017 will care).
- Each frame, copy the main camera's **rotation** onto the sky camera and
  leave its translation at the origin. The dome never translates, so terrain
  can never clip through it regardless of how far the player flies.

### Why not the obvious options

- **`bevy::core_pipeline::Skybox` with a cubemap.** Correct and the cheapest
  thing to draw, but the colour is baked into the texture. Re-generating six
  faces every time 018 ticks the clock is far more work than re-colouring a
  few hundred vertices. Fine for a static sky; wrong for this one.
- **A custom `Material` + WGSL gradient.** Least memory, most flexible, and
  the right long-term answer — but it's the only option here that adds a
  shader to a codebase that currently has none. Not worth it for a
  two-colour gradient.
- **Fog + clear colour only** (i.e. don't do this ticket). Legitimate! 015
  alone already looks decent. This ticket is what makes a sunset read as a
  sunset.

### Colouring the dome

Bake the gradient into the dome mesh's `ATTRIBUTE_COLOR`: for each vertex,
lerp `horizon_color → zenith_color` on the vertex's normalised Y, with a
curve (`y.max(0.0).powf(0.5)` or similar) so the horizon band is tight
rather than a linear wash. Rebuild the colour attribute whenever
`SkyPalette` changes — a 32x16 sphere is ~500 vertices, so this is free even
at every frame, and it avoids writing any shader.

Below the horizon (`y < 0`) hold the horizon colour. There is world down
there, but the dome is visible past the render-distance edge when looking
down from height.

### Sun and moon

Two unlit, alpha-masked quads on layer 1, at the dome radius, billboarded to
face the origin:

- sun: `assets/minecraft/textures/environment/sun.png`,
- moon: `assets/minecraft/textures/environment/moon_phases.png` — a 4x2
  atlas of the eight phases; pick one via UV rect. Phase depends on the
  world's day count, which isn't read anywhere yet (see 018's `level.dat`
  note) — hardcode full moon and leave a comment.

Position both from `SkyPalette::sun_direction`: sun at
`-sun_direction * radius`, moon opposite. They stay consistent with the
lighting for free that way, including once 018 animates it.

Angular size: vanilla's sun is generously large (~2° would be realistic;
Minecraft's is far bigger). Match the game's look rather than reality — size
by eye against a screenshot.

### Stars

Optional in this ticket, cheap if wanted: a few hundred small quads or a
point-list mesh on layer 1, with alpha driven by how far below the horizon
the sun is. Skip it if 018 isn't landing soon — stars against a noon sky are
just wasted draw calls.

## Interactions to get right

- **The horizon band must match the fog colour exactly** (015). Terrain
  fades into fog; the dome sits behind it. Any mismatch shows up as a seam
  right where the eye is drawn. Both read `SkyPalette::horizon_color` — do
  not introduce a second "sky bottom" colour.
- **The sky camera must not run egui or the camera controller.** Both query
  by component; check that adding a second `Camera3d` doesn't break
  `camera::drive_camera`'s `Single<…>` query (it will — `Single` panics or
  silently no-ops on multiple matches). The sky camera has no `CameraRig`,
  so the existing filter should hold, but verify. Same for
  `sync_render_distance_effects` and anything else that queries cameras.
- **`bevy_egui`** renders to the primary window's camera; confirm the UI
  still draws over everything with the added camera and the changed clear
  config.

## Tests

- the dome mesh's vertex colours interpolate zenith→horizon monotonically in
  Y, and the bottom ring is exactly `horizon_color`;
- changing `SkyPalette` rebuilds the dome colours;
- sun and moon positions are antipodal and follow `sun_direction`.

## Done when

- Looking up shows a gradient sky with a sun in it, and looking at the
  horizon shows no seam between fog and sky.
- Flying to the far edge of the loaded world doesn't clip or move the sky.
- `cargo test` passes.
- `todo.md` gets a manual check: look straight up, straight down, and at the
  horizon; fly several thousand blocks and confirm the sky is unchanged;
  confirm the egui panels still draw on top.
