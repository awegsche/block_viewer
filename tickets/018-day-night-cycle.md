# 018 - Day/night cycle

## Status
Open

## Depends on
015 (`SkyPalette`). Much better with 016 (a sky to change) and 017 (shadows
that swing). Technically landable after 015 alone.

## Goal

A `TimeOfDay` the user can scrub, that drives the sun's angle and the whole
`SkyPalette` through dawn, noon, dusk and night — so a build can be looked
at under the light it was built in.

## Design

### `TimeOfDay`, in `src/sky.rs`

```rust
#[derive(Resource)]
pub struct TimeOfDay {
    /// Minecraft ticks, 0..24000. 0 = dawn, 6000 = noon,
    /// 12000 = dusk, 18000 = midnight.
    pub ticks: f32,
    /// Ticks advanced per real second. 0 = paused.
    pub rate: f32,
}
```

Use Minecraft's tick convention rather than inventing hours — the numbers
then match anything the user knows from the game, and `level.dat`'s
`DayTime` (below) drops straight in.

**Default to paused, at noon.** This is a viewer: a sky that keeps changing
while someone is inspecting a build is an irritation, not a feature. The
cycle is something to turn on, and 018's real value is the *slider*, not the
animation.

### Sun direction

The sun rises in the east and sets in the west. Minecraft east is +X, and
`world::mesh`'s axis mapping keeps Bevy X = Minecraft X, so the sun travels
in the X–Y plane and rotates about the Z axis:

```rust
let angle = (ticks / 24000.0) * std::f32::consts::TAU;
let sun_direction = Quat::from_rotation_z(angle) * Vec3::NEG_Y;  // verify sign
```

Check the sign against a real observation rather than trusting the formula:
at morning (ticks ≈ 2000) shadows must fall **west** (−X). Getting it
backwards is invisible in code and obvious on screen. Write the expectation
down as a test on the noon and dawn cases.

### Palette keyframes

Compute the rest of `SkyPalette` by lerping a small `const` keyframe table.
A table beats analytic curves here: it's directly tunable by whoever is
looking at the screen, and it's readable.

```rust
// (ticks, zenith, horizon, sun_color, sun_illuminance, ambient…)
const KEYFRAMES: &[SkyKeyframe] = &[
    /* 0     dawn     */  horizon ~#FFA95E, low warm sun
    /* 3000  morning  */
    /* 6000  noon     */  zenith #78A7FF, horizon #C6DBFF, full daylight
    /* 9000  afternoon*/
    /* 12000 dusk     */  horizon ~#FF7B42
    /* 13500 night    */  zenith ~#0B0B1A
    /* 18000 midnight */
    /* 22500 predawn  */
];
```

Those hex values are starting points to tune, not gospel. Lerp in **linear**
space (`Color::to_linear()` then mix), not sRGB — an sRGB lerp through a
sunset goes muddy grey in the middle. Wrap around 24000 for the last
segment.

### Night must not be black

Realistically, midnight with no moon is unusable. Give the moon a weak cool
fill — illuminance around 50–100 lux, a blue-white tint, plus a slightly
raised ambient — so a night world is legible while still reading as night.
State this as a deliberate viewer-usability choice in the code comment;
otherwise someone will later "fix" it toward realism.

### Sun ↔ moon handover

Below the horizon the sun's illuminance goes to zero and the moon becomes
the light source, from the opposite direction. Simplest implementation: one
`DirectionalLight`, whose direction flips to the moon's and whose
colour/illuminance switch to the moon's values once the sun is below the
horizon. Cross-fade over the last few degrees so the shadow direction
doesn't snap 180° in one frame. A second light entity is the alternative;
one light with a blend is less code and avoids doubling 017's shadow cost at
dusk.

## UI

In the status panel (`src/ui/status.rs`), below the render-distance slider:

- a slider over `0..=24000`, labelled with the derived clock time
  (`ticks → HH:MM`, where tick 0 = 06:00 and 1000 ticks = 1 hour);
- a play/pause toggle bound to `rate`;
- a speed control when running (a real Minecraft day is 20 minutes = 20
  ticks/sec; offer that and a few multiples);
- optionally quick buttons for dawn / noon / dusk / midnight, which are what
  someone actually wants nine times out of ten.

Watch the existing input-capture plumbing: dragging a slider must not also
fly the camera. That's already handled by `ui::sync_egui_input_capture` and
`camera::CameraSet` ordering, so just don't bypass it.

## Optional: seed from the save

`level.dat`'s `Data.DayTime` is the world's actual clock, and starting there
is a nice touch. It is genuinely optional and it is not free:

- `ranvil` does **not** read `level.dat` today — `SaveMeta` is
  name/path/regions only.
- `level.dat` is **gzip**-compressed, not zlib. `ranvil` already depends on
  `flate2` (it zlib-inflates region chunks), but `block_viewer` does not
  depend on it directly, and `rnbt` has no decompression of its own.

So the clean version of this is a small addition to `ranvil` (that crate
already owns save I/O and already has `flate2`), exposing `DayTime` on
`SaveMeta` — not a `flate2` dependency added to `block_viewer` to reach
around it. Treat it as a follow-up ticket if it's wanted; note in the
resolution which way it went.

## Tests

- keyframe interpolation at exact keyframes, at midpoints, and across the
  24000→0 wrap;
- interpolation happens in linear space (a lerp between two saturated
  colours does not desaturate at the midpoint the way an sRGB lerp does);
- the sun points straight down at ticks 6000, and morning shadows fall west;
- `ticks` wraps rather than growing unbounded when the cycle runs;
- `ticks → HH:MM` formatting, including the tick-0-is-06:00 offset.

## Done when

- The slider scrubs the world through a full day, with the sun, sky, fog,
  ambient and (if 017 has landed) shadows all moving together.
- Night is dark but legible.
- Paused at noon by default; nothing animates unless asked.
- `cargo test` passes.
- `todo.md` gets a manual check: scrub the full slider range watching for
  colour discontinuities, especially across the wrap and at the sun/moon
  handover; confirm morning shadows fall west; confirm the slider doesn't
  fly the camera.
