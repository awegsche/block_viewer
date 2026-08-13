//! Day/night cycle (ticket 018): a scrubbable [`TimeOfDay`] that drives
//! [`super::SkyPalette`] — and, through it, the sun's colour/illuminance,
//! the sky dome, ambient light and fog (via [`super::sync_sky_palette`],
//! [`super::dome::rebuild_dome_on_palette_change`],
//! [`super::bodies::sync_celestial_positions`]) — through dawn, noon, dusk
//! and night. [`SkyPalette::default`](super::SkyPalette::default) stays a
//! fixed, hand-picked noon for anything that spins up [`super::SkyPlugin`]
//! without also caring about the clock (tests, mainly); as soon as
//! [`TimeOfDay`] is inserted (every real app, via [`super::SkyPlugin`]),
//! [`sync_palette_from_time_of_day`] overwrites it with
//! [`palette_for_ticks`]'s noon on the very first frame and keeps it in
//! step with the clock from then on.
//!
//! ## Sun direction
//!
//! [`raw_sun_direction`] rotates about Bevy's Z axis (Minecraft's
//! north-south axis) so the sun travels in the X-Y plane — east (+X) to
//! zenith (+Y) to west (-X) to nadir (-Y) — matching the parent ticket's
//! "sun rises in the east, sets in the west". The ticket's own placeholder
//! formula (`Quat::from_rotation_z(angle) * Vec3::NEG_Y`) doesn't survive
//! contact with its own convention (`ticks: 0 = dawn, 6000 = noon`): at
//! `ticks = 0` that formula points straight down, i.e. noon, not dawn. This
//! module uses `Vec3::NEG_X` instead — verified, per the ticket's
//! instruction to check the sign against a real observation rather than
//! trust the formula, against both cases it calls out:
//! `tests::sun_points_straight_down_at_noon` and
//! `tests::morning_shadows_fall_west`.
//!
//! ## Sun ↔ moon handover
//!
//! [`super::bodies`] already places the sun billboard at `-sun_direction`
//! and the moon billboard at `+sun_direction` — the two are always
//! antipodal, by construction, at every tick, not just near the horizon.
//! That's a problem for the *light*, not the billboards: naively lerping
//! the `DirectionalLight`'s direction between "the sun's" and "the moon's"
//! is a lerp between two always-exactly-opposite vectors, which passes
//! through the zero vector at the midpoint and produces a degenerate
//! (`looking_to`-panicking, in the limit) transform right at the
//! crossover — not just a sharp cut, actively broken.
//!
//! [`moon_handover_direction`] sidesteps this by blending in *rotation
//! angle* space instead of vector space: within [`HANDOVER_WIDTH`] of the
//! horizon it rotates the given direction the extra half turn to the
//! antipodal (moon's) direction smoothly, rather than snapping 180° in one
//! frame. It's a pure function of the direction vector, not of `ticks` —
//! [`super::sync_sky_palette`] applies it only to the `DirectionalLight`'s
//! own transform, leaving [`super::SkyPalette::sun_direction`] itself (what
//! [`super::bodies`] reads for billboard placement) as the true,
//! unblended, continuous direction throughout.

use bevy::color::Mix;
use bevy::prelude::*;

use super::SkyPalette;

/// Minecraft ticks in a full day — [`TimeOfDay::ticks`] wraps at this.
/// `pub`, not `pub(crate)` — reachable outside this private module only
/// through `sky`'s re-export (same pattern as [`super::ShadowSettings`]),
/// but the status panel (`ui::status`) needs the slider's range.
pub const TICKS_PER_DAY: f32 = 24000.0;

/// Ticks per real second at 1x speed: a real Minecraft day (20 minutes) is
/// 24000 ticks, so 24000 / (20*60) = 20 ticks/sec — the status panel's
/// baseline "play" speed (ticket 018's UI section), with multiples of this
/// offered alongside it.
pub const REAL_TIME_TICKS_PER_SECOND: f32 = 20.0;

/// The clock the day/night cycle scrubs — see the module docs.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct TimeOfDay {
    /// Minecraft ticks, wrapped into `0..TICKS_PER_DAY`. 0 = dawn, 6000 =
    /// noon, 12000 = dusk, 18000 = midnight.
    pub ticks: f32,
    /// Ticks advanced per real second by [`advance_time_of_day`]. 0 =
    /// paused.
    pub rate: f32,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        // Paused, at noon: this is a viewer, and a sky that keeps changing
        // while someone is inspecting a build is an irritation, not a
        // feature (see the parent ticket's "Default to paused, at noon").
        // 018's value is the slider, not the animation running
        // unattended.
        Self {
            ticks: 6000.0,
            rate: 0.0,
        }
    }
}

/// Advances [`TimeOfDay::ticks`] by `rate` ticks/second and wraps it back
/// into `0..TICKS_PER_DAY` — never lets it grow unboundedly across a long
/// session. Only writes to the resource (and so only marks it
/// `is_changed()`, which [`sync_palette_from_time_of_day`] gates on) when
/// `rate != 0.0`; touching a `ResMut` unconditionally every frame would
/// mark the clock "changed" even while paused, which would in turn make
/// every downstream palette/dome/body system redo work for a sky that
/// never actually moved.
pub(crate) fn advance_time_of_day(time: Res<Time>, mut time_of_day: ResMut<TimeOfDay>) {
    if time_of_day.rate == 0.0 {
        return;
    }
    let ticks = (time_of_day.ticks + time_of_day.rate * time.delta_secs()).rem_euclid(TICKS_PER_DAY);
    time_of_day.ticks = ticks;
}

/// Rewrites [`SkyPalette`] from [`palette_for_ticks`] whenever
/// [`TimeOfDay`] changes — playing, dragging the status panel's slider, or
/// hitting one of its dawn/noon/dusk/midnight buttons all go through
/// mutating this one resource, so this is the only place that needs to
/// know about [`palette_for_ticks`] at all.
pub(crate) fn sync_palette_from_time_of_day(
    time_of_day: Res<TimeOfDay>,
    mut palette: ResMut<SkyPalette>,
) {
    if !time_of_day.is_changed() {
        return;
    }
    *palette = palette_for_ticks(time_of_day.ticks);
}

/// `ticks -> "HH:MM"`, the status panel's slider label. Tick 0 is
/// Minecraft's own dawn-tick convention, 06:00; 1000 ticks is one in-game
/// hour. (Sanity check: 6000 ticks lands on 12:00 (noon) and 18000 lands on
/// 00:00 (midnight) — the same ticks Minecraft itself uses for noon and
/// midnight, which is a good sign this offset is the right one.)
pub fn ticks_to_clock_string(ticks: f32) -> String {
    let ticks = ticks.rem_euclid(TICKS_PER_DAY);
    const DAWN_OFFSET_MINUTES: i64 = 6 * 60;
    let minutes_since_dawn = (ticks / 1000.0 * 60.0) as i64;
    let total_minutes = (minutes_since_dawn + DAWN_OFFSET_MINUTES).rem_euclid(24 * 60);
    format!("{:02}:{:02}", total_minutes / 60, total_minutes % 60)
}

/// One point in the day/night colour curve [`palette_for_ticks`]
/// interpolates through — a lookup table beats an analytic curve here per
/// the parent ticket: directly tunable by whoever is looking at the
/// screen, and readable without doing trig in your head.
struct SkyKeyframe {
    ticks: f32,
    zenith: Color,
    horizon: Color,
    sun_color: Color,
    /// Lux. Doubles as the night entries' moonlight strength — see the
    /// module docs and [`super::SkyPalette`]'s "night must not be black"
    /// requirement: [`palette_for_ticks`] has no separate moon-illuminance
    /// concept, the night keyframes' own `sun_illuminance` values already
    /// are the moon's.
    sun_illuminance: f32,
    ambient_color: Color,
    ambient_brightness: f32,
}

/// The keyframe table, ticks ascending, `[0]` always at ticks 0 (dawn) so
/// [`palette_for_ticks`]'s wraparound segment (last keyframe back to `[0]`)
/// doesn't need special-casing "before the first keyframe" separately from
/// "after the last". Colours are starting points to tune by eye once
/// someone can actually watch the slider move (see CLAUDE.md's
/// manual-verification note) — not gospel.
///
/// Illuminance values are real [`light_consts::lux`] presets, not
/// invented numbers: [`light_consts::lux::AMBIENT_DAYLIGHT`] at noon
/// (matching [`super::SkyPalette::default`]'s own value),
/// [`light_consts::lux::CLEAR_SUNRISE`] at dawn/dusk, and
/// [`light_consts::lux::HALLWAY`] (~80 lux) for full night — inside the
/// ticket's suggested 50-100 lux range for a night that's dark but still
/// legible, deliberately unrealistic (see [`super::SkyPalette`]'s "night
/// must not be black" doc: this is a viewer-usability choice, not a bug to
/// "fix" toward a true moonless ~0.0001 lux).
fn keyframes() -> [SkyKeyframe; 8] {
    use bevy::pbr::light_consts::lux;
    [
        SkyKeyframe {
            ticks: 0.0, // dawn
            zenith: Color::srgb_u8(0x4A, 0x6F, 0xA5),
            horizon: Color::srgb_u8(0xFF, 0xA9, 0x5E),
            sun_color: Color::srgb_u8(0xFF, 0xB3, 0x7A),
            sun_illuminance: lux::CLEAR_SUNRISE,
            ambient_color: Color::srgb(0.55, 0.55, 0.75),
            ambient_brightness: 120.0,
        },
        SkyKeyframe {
            ticks: 3000.0, // morning
            zenith: Color::srgb_u8(0x6C, 0x9A, 0xE0),
            horizon: Color::srgb_u8(0xD9, 0xE6, 0xFF),
            sun_color: Color::srgb(0.95, 0.90, 0.85),
            sun_illuminance: lux::OVERCAST_DAY * 6.0,
            ambient_color: Color::srgb(0.60, 0.70, 0.95),
            ambient_brightness: 220.0,
        },
        SkyKeyframe {
            ticks: 6000.0, // noon — matches SkyPalette::default exactly.
            zenith: Color::srgb_u8(0x78, 0xA7, 0xFF),
            horizon: Color::srgb_u8(0xC6, 0xDB, 0xFF),
            sun_color: Color::srgb(1.0, 0.97, 0.92),
            sun_illuminance: lux::AMBIENT_DAYLIGHT,
            ambient_color: Color::srgb(0.65, 0.75, 1.0),
            ambient_brightness: 300.0,
        },
        SkyKeyframe {
            ticks: 9000.0, // afternoon
            zenith: Color::srgb_u8(0x6C, 0x9A, 0xE0),
            horizon: Color::srgb_u8(0xD9, 0xE6, 0xFF),
            sun_color: Color::srgb(0.98, 0.88, 0.78),
            sun_illuminance: lux::OVERCAST_DAY * 6.0,
            ambient_color: Color::srgb(0.60, 0.70, 0.95),
            ambient_brightness: 220.0,
        },
        SkyKeyframe {
            ticks: 12000.0, // dusk
            zenith: Color::srgb_u8(0x3A, 0x4E, 0x7A),
            horizon: Color::srgb_u8(0xFF, 0x7B, 0x42),
            sun_color: Color::srgb(1.0, 0.55, 0.30),
            sun_illuminance: lux::CLEAR_SUNRISE,
            ambient_color: Color::srgb(0.50, 0.45, 0.60),
            ambient_brightness: 120.0,
        },
        SkyKeyframe {
            ticks: 13500.0, // night
            zenith: Color::srgb_u8(0x0B, 0x0B, 0x1A),
            horizon: Color::srgb_u8(0x13, 0x1A, 0x33),
            sun_color: Color::srgb(0.70, 0.78, 1.0),
            sun_illuminance: lux::HALLWAY,
            // Slightly raised relative to a "realistic" near-zero ambient
            // — see the module-level "night must not be black" reasoning.
            ambient_color: Color::srgb(0.25, 0.30, 0.45),
            ambient_brightness: 40.0,
        },
        SkyKeyframe {
            ticks: 18000.0, // midnight — the dimmest point of the cycle.
            zenith: Color::srgb_u8(0x06, 0x06, 0x12),
            horizon: Color::srgb_u8(0x0D, 0x11, 0x22),
            sun_color: Color::srgb(0.65, 0.75, 1.0),
            sun_illuminance: lux::HALLWAY * 0.75,
            ambient_color: Color::srgb(0.22, 0.27, 0.42),
            ambient_brightness: 35.0,
        },
        SkyKeyframe {
            ticks: 22500.0, // predawn
            zenith: Color::srgb_u8(0x17, 0x24, 0x4A),
            horizon: Color::srgb_u8(0x3B, 0x38, 0x66),
            sun_color: Color::srgb(0.75, 0.70, 0.85),
            sun_illuminance: lux::CIVIL_TWILIGHT * 40.0,
            ambient_color: Color::srgb(0.35, 0.35, 0.50),
            ambient_brightness: 55.0,
        },
    ]
}

/// Lerps two colours in **linear**, not sRGB, space (see
/// [`super::super::world::mesh`]'s module docs on why this repo is careful
/// about that everywhere it touches colour): a straight sRGB lerp between
/// two saturated colours (e.g. dusk's orange horizon and night's near-black
/// blue) visibly desaturates through a muddy grey at the midpoint, which a
/// linear-space lerp doesn't.
fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    Color::from(a.to_linear().mix(&b.to_linear(), t))
}

/// Direction the sunlight travels (sun -> ground), for `ticks` alone — no
/// day/night handover here (see [`moon_handover_direction`] for that);
/// [`super::bodies`] and [`palette_for_ticks`] both need this exact
/// continuous rotation, never the handover-blended one, to place the sun/
/// moon billboards and [`SkyPalette::sun_direction`] correctly. See the
/// module docs' "Sun direction" section for the derivation.
fn raw_sun_direction(ticks: f32) -> Vec3 {
    let angle = (ticks / TICKS_PER_DAY) * std::f32::consts::TAU;
    Quat::from_rotation_z(angle) * Vec3::NEG_X
}

/// How many degrees (in `sin(elevation)` units — 0.06 ≈ 3.4°) either side
/// of the horizon [`moon_handover_direction`] spends rotating the light
/// direction the extra half turn, rather than snapping it. Narrow on
/// purpose: it should read as "the sun set, the moon's now doing the
/// lighting" over a couple of frames near the horizon, not as a visibly
/// slow re-aim of the whole shadow field.
const HANDOVER_WIDTH: f32 = 0.06;

/// 0.0 = the sun alone is lighting the scene, 1.0 = the moon alone is,
/// smoothly in between within [`HANDOVER_WIDTH`] of the horizon.
/// `elevation` is `sin` of the sun's angle above the horizon — positive
/// above, negative below, exactly `-sun_direction.y` for any direction
/// [`raw_sun_direction`] or [`moon_handover_direction`] can produce.
fn night_fraction(elevation: f32) -> f32 {
    ((HANDOVER_WIDTH - elevation) / (2.0 * HANDOVER_WIDTH)).clamp(0.0, 1.0)
}

/// The direction a `DirectionalLight` transform should actually point,
/// given a [`SkyPalette::sun_direction`] — see the module docs' "Sun ↔ moon
/// handover" section for why this exists as a separate function rather
/// than being baked into `sun_direction` itself. Identical to the input
/// while the sun is up (`night_fraction == 0.0`, so the extra rotation is
/// zero and this is the identity), rotated a full extra half turn to the
/// antipodal (moon's) direction once fully below [`HANDOVER_WIDTH`], and
/// smoothly in between.
pub(crate) fn moon_handover_direction(sun_direction: Vec3) -> Vec3 {
    let elevation = -sun_direction.y;
    let extra_angle = std::f32::consts::PI * night_fraction(elevation);
    Quat::from_rotation_z(extra_angle) * sun_direction
}

/// The full [`SkyPalette`] for a given clock reading — wraps `ticks` into
/// `0..TICKS_PER_DAY`, finds the bracketing pair of [`keyframes`] (wrapping
/// the segment after the last keyframe back to the first, `+ TICKS_PER_DAY`,
/// rather than treating it as "off the end of the table"), and lerps every
/// field ([`lerp_color`] for the colours, plain `f32` lerp for illuminance/
/// brightness) between them. `sun_direction` doesn't come from the table at
/// all — see [`raw_sun_direction`].
pub(crate) fn palette_for_ticks(ticks: f32) -> SkyPalette {
    let ticks = ticks.rem_euclid(TICKS_PER_DAY);
    let frames = keyframes();

    // `frames[0].ticks == 0.0` always, so `ticks >= frames[0].ticks`
    // always holds and `lo` never needs a "before the first keyframe"
    // fallback — this loop always finds a real bracketing pair.
    let mut lo = 0;
    for (i, frame) in frames.iter().enumerate() {
        if frame.ticks <= ticks {
            lo = i;
        }
    }
    let hi = (lo + 1) % frames.len();

    let lo_ticks = frames[lo].ticks;
    let (hi_ticks, ticks) = if hi == 0 {
        // Wrapping segment: the last keyframe back to the first, one lap
        // later.
        (frames[hi].ticks + TICKS_PER_DAY, ticks.max(lo_ticks))
    } else {
        (frames[hi].ticks, ticks)
    };
    let t = ((ticks - lo_ticks) / (hi_ticks - lo_ticks)).clamp(0.0, 1.0);

    let a = &frames[lo];
    let b = &frames[hi];

    SkyPalette {
        sun_direction: raw_sun_direction(ticks),
        sun_color: lerp_color(a.sun_color, b.sun_color, t),
        sun_illuminance: a.sun_illuminance + (b.sun_illuminance - a.sun_illuminance) * t,
        ambient_color: lerp_color(a.ambient_color, b.ambient_color, t),
        ambient_brightness: a.ambient_brightness + (b.ambient_brightness - a.ambient_brightness) * t,
        zenith_color: lerp_color(a.zenith, b.zenith, t),
        horizon_color: lerp_color(a.horizon, b.horizon, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Color` is a tagged union over colour spaces and `lerp_color` always
    /// returns the `LinearRgba` variant (see its docs) — comparing against
    /// a keyframe's `Srgba`-variant colour with `assert_eq!` would fail on
    /// the variant tag alone even when the numbers agree, so this compares
    /// through `to_srgba()` instead, which normalises both sides to the
    /// same representation.
    fn assert_colors_approx_eq(a: Color, b: Color) {
        let a = a.to_srgba();
        let b = b.to_srgba();
        assert!(
            (a.red - b.red).abs() < 1e-4
                && (a.green - b.green).abs() < 1e-4
                && (a.blue - b.blue).abs() < 1e-4,
            "expected {b:?}, got {a:?}"
        );
    }

    #[test]
    fn keyframe_interpolation_is_exact_at_keyframe_ticks() {
        for frame in keyframes() {
            let palette = palette_for_ticks(frame.ticks);
            assert_eq!(palette.sun_illuminance, frame.sun_illuminance);
            assert_eq!(palette.ambient_brightness, frame.ambient_brightness);
            assert_colors_approx_eq(palette.zenith_color, frame.zenith);
            assert_colors_approx_eq(palette.horizon_color, frame.horizon);
        }
    }

    #[test]
    fn keyframe_interpolation_is_a_true_midpoint_between_neighbours() {
        // Between dusk (12000, illuminance = CLEAR_SUNRISE) and night
        // (13500, illuminance = HALLWAY).
        let mid_ticks = 12750.0;
        let palette = palette_for_ticks(mid_ticks);
        let expected = (bevy::pbr::light_consts::lux::CLEAR_SUNRISE
            + bevy::pbr::light_consts::lux::HALLWAY)
            / 2.0;
        assert!(
            (palette.sun_illuminance - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            palette.sun_illuminance
        );
    }

    /// Ticket 018's "Tests" section, verbatim: "interpolation happens in
    /// linear space (a lerp between two saturated colours does not
    /// desaturate at the midpoint the way an sRGB lerp does)".
    #[test]
    fn color_interpolation_happens_in_linear_space_not_srgb() {
        let red = Color::srgb(1.0, 0.0, 0.0);
        let blue = Color::srgb(0.0, 0.0, 1.0);

        let linear_mid = lerp_color(red, blue, 0.5).to_srgba();

        // A naive sRGB-space lerp of the same two colours: 0.5 in each
        // sRGB channel, gathered here (not via `lerp_color`) purely as the
        // wrong answer to compare against.
        let srgb_lerp_mid = Color::srgb(0.5, 0.0, 0.5).to_srgba();

        assert!(
            (linear_mid.red - srgb_lerp_mid.red).abs() > 0.05
                || (linear_mid.blue - srgb_lerp_mid.blue).abs() > 0.05,
            "linear-space and sRGB-space lerps should visibly disagree on a \
             red/blue midpoint; got linear {linear_mid:?} vs sRGB-lerped {srgb_lerp_mid:?}"
        );
    }

    #[test]
    fn keyframe_interpolation_wraps_from_the_last_keyframe_back_to_the_first() {
        // Halfway between the last keyframe (22500, predawn) and the first
        // (24000 == 0, dawn) one lap later.
        let mid_ticks = 23250.0;
        let palette = palette_for_ticks(mid_ticks);
        let frames = keyframes();
        let predawn = frames.last().unwrap();
        let dawn = &frames[0];
        let expected = (predawn.sun_illuminance + dawn.sun_illuminance) / 2.0;
        assert!(
            (palette.sun_illuminance - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            palette.sun_illuminance
        );
    }

    /// Ticket 018's "Sun direction" section, verbatim requirement: "the sun
    /// points straight down at ticks 6000".
    #[test]
    fn sun_points_straight_down_at_noon() {
        let direction = raw_sun_direction(6000.0);
        assert!(
            direction.dot(Vec3::NEG_Y) > 0.999,
            "expected straight down at noon, got {direction:?}"
        );
    }

    /// Ticket 018's "Sun direction" section, verbatim requirement:
    /// "morning shadows fall west" (at ticks ≈ 2000). A shadow falls in the
    /// direction the light continues travelling past the ground, i.e. the
    /// horizontal component of `sun_direction` itself.
    #[test]
    fn morning_shadows_fall_west() {
        let direction = raw_sun_direction(2000.0);
        assert!(
            direction.x < 0.0,
            "expected a westward (-X) horizontal component at ticks 2000, got {direction:?}"
        );
    }

    #[test]
    fn dawn_and_dusk_sit_exactly_on_the_horizon() {
        assert!(raw_sun_direction(0.0).y.abs() < 1e-5);
        assert!(raw_sun_direction(12000.0).y.abs() < 1e-5);
    }

    /// Drives [`Time`] by hand via [`Time::advance_by`] rather than adding
    /// `TimePlugin` and relying on real wall-clock deltas between
    /// `app.update()` calls — deterministic (no flakiness from how fast the
    /// test machine happens to run), and it makes the exact wrap amount
    /// assertable instead of just "somewhere in range".
    #[test]
    fn ticks_advance_and_wrap_rather_than_growing_unbounded() {
        let mut app = App::new();
        app.init_resource::<Time>();
        app.add_systems(Update, advance_time_of_day);
        app.insert_resource(TimeOfDay {
            ticks: TICKS_PER_DAY - 10.0,
            rate: 20.0,
        });

        // 1 real second at 20 ticks/sec = 20 ticks: TICKS_PER_DAY - 10 + 20
        // wraps to 10.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs_f32(1.0));
        app.update();

        let ticks = app.world().resource::<TimeOfDay>().ticks;
        assert!(
            (0.0..TICKS_PER_DAY).contains(&ticks),
            "ticks should stay wrapped into 0..{TICKS_PER_DAY}, got {ticks}"
        );
        assert!(
            (ticks - 10.0).abs() < 1e-2,
            "expected wrap to land on ~10.0, got {ticks}"
        );
    }

    #[test]
    fn paused_time_of_day_never_advances() {
        let mut app = App::new();
        app.init_resource::<Time>();
        app.add_systems(Update, advance_time_of_day);
        app.insert_resource(TimeOfDay {
            ticks: 6000.0,
            rate: 0.0,
        });

        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs_f32(1.0));
        app.update();
        app.update();

        assert_eq!(app.world().resource::<TimeOfDay>().ticks, 6000.0);
    }

    /// Ticket 018's "Tests" section: "`ticks -> HH:MM` formatting, including
    /// the tick-0-is-06:00 offset".
    #[test]
    fn ticks_to_clock_string_matches_minecrafts_own_dawn_noon_dusk_midnight() {
        assert_eq!(ticks_to_clock_string(0.0), "06:00");
        assert_eq!(ticks_to_clock_string(6000.0), "12:00");
        assert_eq!(ticks_to_clock_string(12000.0), "18:00");
        assert_eq!(ticks_to_clock_string(18000.0), "00:00");
        // 500 ticks = 30 minutes.
        assert_eq!(ticks_to_clock_string(500.0), "06:30");
    }

    #[test]
    fn ticks_to_clock_string_wraps_past_a_full_day() {
        assert_eq!(ticks_to_clock_string(TICKS_PER_DAY), "06:00");
        assert_eq!(ticks_to_clock_string(-1000.0), "05:00");
    }

    /// The handover must be the identity while the sun's well above the
    /// horizon — otherwise the light's direction would visibly disagree
    /// with [`super::super::bodies`]'s billboard placement (which always
    /// uses the unblended [`raw_sun_direction`]) any time it's not exactly
    /// noon.
    #[test]
    fn moon_handover_is_identity_well_above_the_horizon() {
        let noon = raw_sun_direction(6000.0);
        assert!((moon_handover_direction(noon) - noon).length() < 1e-5);
    }

    /// Deep at night, the handover has fully rotated to the antipodal
    /// (moon's) direction, matching what [`super::super::bodies`] already
    /// treats as "the moon's position": `+sun_direction` there is
    /// `-raw_sun_direction` here.
    #[test]
    fn moon_handover_is_fully_flipped_deep_at_night() {
        let midnight = raw_sun_direction(18000.0);
        let handed_over = moon_handover_direction(midnight);
        assert!(
            handed_over.dot(-midnight) > 0.999,
            "expected the fully-flipped antipodal direction, got {handed_over:?} vs -sun {:?}",
            -midnight
        );
    }

    /// The handover must never produce a zero (or near-zero) vector — the
    /// exact failure mode a naive `Vec3::lerp` between the always-antipodal
    /// sun/moon directions has at the crossover (see the module docs).
    #[test]
    fn moon_handover_never_degenerates_across_a_full_sweep() {
        let mut ticks = 0.0;
        while ticks < TICKS_PER_DAY {
            let direction = moon_handover_direction(raw_sun_direction(ticks));
            assert!(
                direction.length() > 0.5,
                "handover direction degenerated at ticks {ticks}: {direction:?}"
            );
            ticks += 37.0; // odd step so it doesn't land only on round ticks
        }
    }
}
