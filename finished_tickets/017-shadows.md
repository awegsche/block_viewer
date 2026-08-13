# 017 - Sun shadows

## Status
Done

## Resolution

Implemented as `src/sky/shadows.rs`, wired into `sky::SkyPlugin` and called
from `main.rs::setup` via the new `sky::spawn_sun` (which replaces the old
inline `DirectionalLight` spawn — shadows now on by default instead of the
`shadows_enabled: false` placeholder 015 left).

- **Cascade config**: `num_cascades: 4`, `minimum_distance: 0.1`,
  `first_cascade_far_bound: 32.0`, `overlap_proportion: 0.2`,
  `maximum_distance: camera::far_plane_distance(render_distance).min(250.0)`
  — exactly the ticket's suggested shape. `shadows::sync_shadow_cascades`
  rebuilds it whenever `RenderDistance` changes, the same pattern
  `camera::sync_render_distance_effects` uses for the far plane/fog.
- **`DirectionalLightShadowMap { size: 2048 }`**: inserted explicitly by
  `SkyPlugin` (matches Bevy's own default, but spelled out since 4096 is
  the first knob to reach for if resolution turns out to be the problem).
- **Bias values**: `shadow_depth_bias` left at Bevy's default (`0.02`);
  `shadow_normal_bias` raised from Bevy's default `1.8` to `3.0` — voxel
  terrain's large coplanar axis-aligned faces at grazing sun angles is
  exactly where normal bias (not depth bias) earns its keep, per the
  ticket's own framing. **These are starting values, not measured ones** —
  per CLAUDE.md's manual/visual-verification policy, nothing in this
  session ran the app to actually look at a low sun angle for banding or a
  single block for peter-panning. `todo.md` carries that check (017's
  entry); if it finds banding, raise `SHADOW_NORMAL_BIAS`
  (`src/sky/shadows.rs`) further before touching depth bias. If it finds
  peter-panning instead, that's the depth-bias knob.
- **Skybox exclusion**: already done by 016 — the sky dome (`sky/mod.rs`)
  and sun/moon billboards (`sky/bodies.rs`) already carry
  `NotShadowCaster`/`NotShadowReceiver`, so there was nothing left to do
  here.
- **Toggle**: `sky::ShadowSettings { enabled: bool }` (default `true`),
  a checkbox in the status panel (`src/ui/status.rs`) next to the render-
  distance slider, plus a derived read-only label showing the current
  cascade `maximum_distance` (`sky::shadow_cascade_distance`) rather than a
  second slider, since it's fully determined by render distance.
- **Before/after FPS**: not measured — same reason as the bias values,
  this needs a human actually running the app with the checkbox. `todo.md`'s
  017 entry asks for it (toggle the checkbox, read the status panel's
  existing FPS counter both ways).

Tests (`src/sky/shadows.rs`): cascade config tracks `RenderDistance`
(including that `maximum_distance` stays capped at 250 blocks well past
where the far plane alone would put it), and the toggle flips
`DirectionalLight::shadows_enabled` and nothing else.

## Depends on
015 (there has to be a directional light to cast them). Best done after
**009**'s profiling pass, or at least with its measurements in hand — this
is the most expensive item in the whole colour/lighting group and the only
one that can plausibly halve the frame rate.

## Goal

Terrain casts and receives shadows from the sun, at a cost the user can see
and turn off.

## Work

### Enable and configure

```rust
DirectionalLight {
    shadows_enabled: true,
    shadow_depth_bias: …,
    shadow_normal_bias: …,
    ..default()
}
CascadeShadowConfigBuilder {
    num_cascades: 4,
    minimum_distance: 0.1,
    first_cascade_far_bound: 32.0,
    maximum_distance: /* see below */,
    overlap_proportion: 0.2,
}.build()
```

plus the `DirectionalLightShadowMap { size: 2048 }` resource (4096 if it
holds up; it's a straight quality/VRAM trade).

### Cascade distance vs render distance

`maximum_distance` should **not** just be `camera::far_plane_distance(rd)`.
At render distance 32 that's ~750 blocks, and stretching four cascades over
750 blocks makes the near ones coarse enough that shadow edges crawl. Cap it
— something like `far_plane_distance(rd).min(250.0)` — and let distant
terrain be unshadowed. Beyond a couple of hundred blocks, fog is doing most
of the work anyway.

Wire it into the existing `sync_render_distance_effects` pattern in
`camera.rs` (or a sibling system in `sky.rs`) so it tracks the
render-distance slider, same as the far plane and fog already do.

### Bias tuning — expect to spend time here

Voxel terrain is the pathological case for shadow acne: enormous coplanar
axis-aligned faces, and a sun that spends part of the day at a grazing angle
to all of them. Too little bias gives moiré banding across flat ground; too
much gives peter-panning (shadows detached from the blocks casting them,
very visible on a 1-block-tall step).

`shadow_normal_bias` is the more useful knob for this geometry than
`shadow_depth_bias`. Start from Bevy's defaults, then tune with the sun at a
low angle (018's time slider makes this much easier — consider doing 018
first if it's queued anyway). **Record the values you land on and why** in
the ticket resolution; nobody will be able to re-derive them.

### Skybox exclusion

016's dome and sun/moon quads need `NotShadowCaster` and
`NotShadowReceiver`. A 100-radius inverted sphere around the origin
otherwise sits inside every cascade and shadows the entire world.

### A toggle, not just a feature

Add to the status panel (`src/ui/status.rs`, next to the render-distance
slider):

- a "Shadows" checkbox,
- a cascade-distance slider (or reuse the derived value and just show it).

Because this can be the difference between 60 fps and 25 on the same
machine, and because the checkbox is also the fastest way for a human to
measure the cost — flip it and watch the FPS counter that's already there.

## Cost — measure, don't guess

Every visible chunk mesh is re-rendered once per cascade it falls in.
With `num_cascades: 4`, that is up to 5x the draw calls of today. This
project draws one entity per chunk column with no batching and no greedy
meshing yet (009's list), so draw-call count is exactly the axis this
pressures hardest.

Record before/after FPS at a fixed render distance and camera position, the
same way 009 asks. If the hit is severe, the options in order are: fewer
cascades (2–3), a smaller `maximum_distance`, a smaller shadow map, and then
009's actual work (vertical section splitting, greedy meshing) which reduces
both passes at once.

## Interaction with streaming

Chunks appear and disappear at the render-distance frontier, so shadows will
pop in and out with them, including shadows cast *into* the visible area by
terrain that hasn't loaded. This is inherent to streaming and not worth
fixing (the fix is loading a shadow-only ring beyond render distance —
strictly more work than it's worth here). Note it as expected behaviour so
it isn't reported as a bug later.

## Tests

Essentially none are meaningful — this is a renderer configuration change.
Test what's testable: that the cascade config tracks `RenderDistance`, and
that the shadows toggle actually flips `DirectionalLight::shadows_enabled`.

## Done when

- Terrain casts sun shadows, with no visible acne on flat ground and no
  peter-panning on 1-block steps, at both a high and a low sun angle.
- The status panel can turn shadows off.
- Before/after FPS numbers are written into the ticket resolution, along
  with the bias values chosen.
- `todo.md` gets a manual check: look at flat ground at midday and at a low
  sun angle for banding; look at a single block on flat ground for a
  detached shadow; toggle the checkbox and record the FPS delta.
