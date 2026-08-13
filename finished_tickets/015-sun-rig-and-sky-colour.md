# 015 - Sun rig and a single source of truth for sky colour

## Status
Done

## Resolution

Implemented as `src/sky.rs`'s `SkyPlugin`/`SkyPalette`/`sync_sky_palette`,
wired into `main.rs` (directional-light spawn replaces the old `PointLight`,
`setup` reads `SkyPalette::horizon_color` for the camera's initial fog) and
`camera.rs` (`atmosphere_fog` takes a `color` param; `fog_falloff` factors
out the falloff-only half so `sync_render_distance_effects` never touches
fog colour). Tone mapping was left at the default (`TonyMcMapface`) and
`sun_illuminance` at the ticket's suggested `AMBIENT_DAYLIGHT` (~10,000 lux)
— whether that reads as bleached is a visual call that needs the manual
check in `todo.md` (015's entry); if it does, that entry documents the
tuning order (`sun_illuminance` → `Tonemapping::ReinhardLuminance` →
`Exposure`) to try, in place of guessing blind here.

## Depends on
Nothing. Independent of the biome-tint track (011–014); the two can land in
either order.

## What's there now

`main.rs::setup` spawns a bare `PointLight::default()` about 30 blocks above
the camera's startup position and never moves it again. It was a
"light up the scene" placeholder from before there was a scene. In a
streaming voxel world it lights a small sphere near spawn and leaves
everything else to ambient — fly a hundred blocks and the world goes flat.

There is also a hardcoded sky-ish colour in `camera::atmosphere_fog`
(`Color::srgb(0.7, 0.8, 0.92)`) with no relationship to anything else, and
the window's clear colour is Bevy's default.

## Goal

One `SkyPalette` resource that everything visual about the atmosphere reads
from: the sun, the ambient fill, the window clear colour, and the fog. Set
it to a fixed noon and the world looks like a lit world. 016 draws an actual
sky from it, 017 adds shadows to its sun, 018 animates it — all three become
small because this ticket puts the plumbing in one place.

## Work

### `src/sky.rs` (new) — `SkyPlugin`

```rust
#[derive(Resource, Debug, Clone)]
pub struct SkyPalette {
    /// Direction the sunlight travels (from the sun toward the ground).
    pub sun_direction: Vec3,
    pub sun_color: Color,
    pub sun_illuminance: f32,
    pub ambient_color: Color,
    pub ambient_brightness: f32,
    /// Colour straight up. 016 draws the gradient; 015 only stores it.
    pub zenith_color: Color,
    /// Colour at the horizon — **and** the fog colour. See below.
    pub horizon_color: Color,
}
```

`Default` = noon. Suggested starting values, to be tuned by eye:
sun `Vec3::new(-0.35, -0.85, -0.4).normalize()`, illuminance
`light_consts::lux::AMBIENT_DAYLIGHT` (~10 000), sun colour near-white with
a touch of warmth, ambient brightness a few hundred lux in a cool blue,
zenith `#78A7FF`, horizon `#C6DBFF`.

One system, `sync_sky_palette`, runs when `SkyPalette` is changed
(`Res::is_changed()`) and writes:

- the `DirectionalLight`'s `color`/`illuminance` and its transform's
  rotation (`Transform::default().looking_to(sun_direction, Vec3::Y)`),
- the `AmbientLight` resource,
- the `ClearColor` resource — set it to `horizon_color`, since without 016
  the "sky" *is* the clear colour and terrain fades into it,
- every `DistanceFog`'s `color`.

### `main.rs::setup`

Delete the `PointLight`. Spawn instead:

```rust
commands.spawn((
    Name::new("Sun"),
    DirectionalLight { shadows_enabled: false, ..default() },  // 017 flips this
    Transform::default(),
));
```

Position is irrelevant for a directional light — only rotation matters. Do
not parent it to the camera or offset it by `target`; that was a property of
the point light being local.

### `camera.rs` — the fog-colour collision

`sync_render_distance_effects` currently does `*fog = atmosphere_fog(rd)`,
rebuilding the whole `DistanceFog` including its colour. Once the palette
owns the colour, this **clobbers it** every time the render-distance slider
moves. Split the responsibilities:

- `camera::atmosphere_fog` keeps only the falloff distances
  (`start = far * 0.6`, `end = far`) and takes a colour parameter, or
- `sync_render_distance_effects` mutates `fog.falloff` only and leaves
  `fog.color` alone.

The second is smaller and makes the ownership obvious. Either way, add a
test that changing `RenderDistance` does not change the fog colour — it's
the kind of regression that only shows up as "the horizon went blue-grey
again after I moved the slider".

### Fog colour must equal the horizon colour

This is the single detail that decides whether the result looks like a sky
or like a bug. Terrain at the render-distance edge fades into the fog
colour; if that colour differs from what's behind it, there's a visible band
where the world ends. Keep them the same field, not two fields that happen
to be set alike.

## Tone mapping

Bevy 0.15 defaults to `Tonemapping::TonyMcMapface`. A 10 000-lux directional
light against a mostly-albedo-1.0 texture atlas can come out washed. If the
world reads as bleached after this lands, in order: lower `illuminance`,
then try `Tonemapping::ReinhardLuminance`, then adjust the camera's
`Exposure`. Don't reach for `Tonemapping::None` — it looks flat and will
fight 018's night. Record whatever combination ends up working in the
ticket resolution, since 016/017/018 all inherit it.

## Tests

Rendering isn't unit-testable, but the plumbing is. With a minimal `App`
(`MinimalPlugins` + the resources under test):

- changing `SkyPalette` updates `AmbientLight`, `ClearColor` and the
  `DirectionalLight`'s illuminance;
- `sun_direction` produces a `DirectionalLight` transform whose forward axis
  matches it;
- changing `RenderDistance` updates fog falloff but not fog colour.

## Done when

- The `PointLight` is gone and the world is lit consistently everywhere, not
  just near spawn.
- Sky colour, fog colour, ambient and sun all come from one resource.
- `cargo test` passes.
- `todo.md` gets a manual check: fly a long way from spawn and confirm
  lighting doesn't fall off; confirm the horizon has no visible band where
  fog meets sky; confirm the world isn't bleached out (tone mapping).
