//! Single source of truth for everything visual about the atmosphere
//! (ticket 015): the sun, the ambient fill, the window clear colour and the
//! distance fog all read from one [`SkyPalette`] resource instead of each
//! picking its own hardcoded colour.
//!
//! Before this, `main.rs::setup` spawned a bare `PointLight` a fixed offset
//! above the camera's startup position — it lit a small sphere near spawn
//! and left everything else to flat ambient light, and the sky-ish fog
//! colour in `camera::atmosphere_fog` had no relationship to anything else
//! in the scene. [`SkyPlugin`] replaces both with one resource and one
//! system that keeps every dependent piece of scene state in sync with it.
//!
//! ## An actual sky (ticket 016)
//!
//! [`dome`] and [`bodies`] draw what [`SkyPalette`] stores: a gradient dome
//! plus a sun and moon, on a second `Camera3d` (`order: -1`, its own
//! [`RenderLayers`] layer — see [`SKY_LAYER`]) that renders first and never
//! translates, so the sky can never clip through no matter how far the main
//! camera flies. [`sync_sky_camera_rotation`] is the only thing that ties
//! the two cameras together: it copies the main camera's *rotation* (never
//! its translation) onto the sky camera every frame.
//!
//! [`shadows`] (017) owns everything about the sun casting shadows: the
//! bias constants and cascade config [`spawn_sun`] gives the light at
//! startup, keeping the cascade config in step with render distance from
//! then on, and the status panel's on/off toggle.
//!
//! [`time_of_day`] (018) owns the clock: [`TimeOfDay`] is what the status
//! panel's slider/play-pause/speed controls actually mutate, and
//! [`time_of_day::sync_palette_from_time_of_day`] is the only thing that
//! writes [`SkyPalette`] wholesale from then on — every other system in
//! this module still only ever *reads* the palette, unchanged since 016/017.

mod bodies;
mod dome;
mod shadows;
mod time_of_day;

use bevy::pbr::{DirectionalLightShadowMap, NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::view::RenderLayers;

pub use shadows::ShadowSettings;
pub use time_of_day::{ticks_to_clock_string, TimeOfDay, REAL_TIME_TICKS_PER_SECOND, TICKS_PER_DAY};

/// Everything visual about the atmosphere, in one place. `Default` is a
/// fixed, hand-picked noon, used as-is by anything that spins up
/// [`SkyPlugin`] without also caring about the clock (mainly tests) — any
/// real app gets [`TimeOfDay`] (018) too, whose
/// [`time_of_day::sync_palette_from_time_of_day`] overwrites this resource
/// wholesale from [`time_of_day::palette_for_ticks`] on the very first
/// frame and every frame the clock moves after that.
#[derive(Resource, Debug, Clone)]
pub struct SkyPalette {
    /// Direction the sunlight travels (from the sun toward the ground) —
    /// this, not a position, is all a [`DirectionalLight`] cares about.
    pub sun_direction: Vec3,
    pub sun_color: Color,
    pub sun_illuminance: f32,
    pub ambient_color: Color,
    pub ambient_brightness: f32,
    /// Colour straight up — [`dome::build_dome_mesh`] draws the gradient
    /// between this and [`Self::horizon_color`].
    pub zenith_color: Color,
    /// Colour at the horizon — **and** the fog colour ([`sync_sky_palette`]
    /// writes it to both [`ClearColor`] and every [`DistanceFog`]). Without
    /// 016's sky gradient, the clear colour *is* the sky, and terrain at
    /// the render-distance edge fades into the fog colour — if that ever
    /// drifts from what's behind it there's a visible band where the world
    /// ends, so both read this one field rather than two fields that
    /// happen to be set alike.
    pub horizon_color: Color,
}

impl Default for SkyPalette {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(-0.35, -0.85, -0.4).normalize(),
            sun_color: Color::srgb(1.0, 0.97, 0.92),
            sun_illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
            ambient_color: Color::srgb(0.65, 0.75, 1.0),
            ambient_brightness: 300.0,
            zenith_color: Color::srgb_u8(0x78, 0xA7, 0xFF),
            horizon_color: Color::srgb_u8(0xC6, 0xDB, 0xFF),
        }
    }
}

/// The [`RenderLayers`] layer the sky camera, dome, sun and moon all live
/// on — kept off the default layer (`0`, what the main camera and every
/// terrain mesh render without needing to say so) so neither camera ever
/// double-draws the other's geometry.
pub(crate) const SKY_LAYER: usize = 1;

/// Tags the sky camera so [`sync_sky_camera_rotation`] can tell it apart
/// from the main camera — both are plain [`Camera3d`]s otherwise.
///
/// Deliberately the *only* thing that marks it: it never gets a
/// `camera::CameraRig`, so every existing camera query filtered on that
/// (`camera::drive_camera`'s `Single<…>`, `sync_render_distance_effects`,
/// `streaming::update_pending_chunk_work`, `unload`'s equivalent) keeps
/// matching exactly one entity — the main camera — with this one added
/// alongside it, rather than needing to learn about a second camera at all.
#[derive(Component)]
pub(crate) struct SkyCamera;

/// Adds [`SkyPalette`] (defaulting to noon), [`ShadowSettings`] (defaulting
/// to on), [`sync_sky_palette`], and (016) the systems that keep the dome's
/// colours and the sun/moon's positions following the palette, plus (017)
/// the sun's shadow cascade config following render distance and its
/// `shadows_enabled` following [`ShadowSettings`], plus the sky camera's
/// rotation following the main camera.
///
/// Does **not** spawn the sky camera/dome/sun/moon itself — those need
/// [`Assets<Image>`] loaded from disk the same eager, panic-on-failure way
/// `main.rs::setup` loads the block atlas, and nothing in this repo runs
/// that kind of I/O from inside a `Startup` system that `cargo test` would
/// also execute (see `world::atlas`'s own tests for why: relative asset
/// paths only resolve from the crate root, which is true for `cargo run`
/// but not guaranteed for every test harness). Call [`spawn_sky_scene`] and
/// [`spawn_sun`] from `main.rs::setup` instead, right alongside the atlas/
/// colormap loads it already does.
pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyPalette>()
            .init_resource::<ShadowSettings>()
            .init_resource::<TimeOfDay>()
            // Explicit rather than relying on `PbrPlugin`'s own default
            // (also 2048) — this is the knob the ticket calls out as the
            // quality/VRAM trade to reach for first (4096) if shadow
            // resolution turns out to be the problem, so it's spelled out
            // here rather than left implicit.
            .insert_resource(DirectionalLightShadowMap { size: 2048 })
            .add_systems(
                Update,
                (
                    // 018: advancing/scrubbing the clock has to land in
                    // `SkyPalette` before anything below reads it, or a
                    // slider drag/the play toggle would lag a frame behind
                    // — `.chain()` on the whole tuple makes that ordering
                    // explicit rather than relying on Bevy's default
                    // (unordered/parallel) system scheduling.
                    time_of_day::advance_time_of_day,
                    time_of_day::sync_palette_from_time_of_day,
                    sync_sky_palette,
                    dome::rebuild_dome_on_palette_change,
                    bodies::sync_celestial_positions,
                    sync_sky_camera_rotation,
                    shadows::sync_shadow_cascades,
                    shadows::sync_shadow_settings,
                )
                    .chain(),
            );
    }
}

/// Spawns the sun's `DirectionalLight` with shadows configured for
/// `render_distance_chunks` (ticket 017) — a plain function `main.rs::setup`
/// calls, for the same reason [`spawn_sky_scene`] is (see [`SkyPlugin`]'s
/// docs). Kept separate from [`spawn_sky_scene`] because it needs no image/
/// mesh assets, and because `main.rs::setup` spawns the sun before the sky
/// scene.
///
/// Only sets the fields [`sync_sky_palette`] doesn't already own on the very
/// next tick (colour, illuminance, rotation): the shadow bias constants
/// ([`shadows::SHADOW_DEPTH_BIAS`], [`shadows::SHADOW_NORMAL_BIAS`]), the
/// initial cascade config ([`shadows::cascade_config`] — the same shape
/// [`shadows::sync_shadow_cascades`] rebuilds at runtime), and
/// `shadows_enabled` matching [`ShadowSettings::default`] (owned by
/// [`shadows::sync_shadow_settings`] from then on).
pub(crate) fn spawn_sun(commands: &mut Commands, render_distance_chunks: u32) {
    commands.spawn((
        Name::new("Sun"),
        DirectionalLight {
            shadows_enabled: ShadowSettings::default().enabled,
            shadow_depth_bias: shadows::SHADOW_DEPTH_BIAS,
            shadow_normal_bias: shadows::SHADOW_NORMAL_BIAS,
            ..default()
        },
        shadows::cascade_config(render_distance_chunks),
        Transform::default(),
    ));
}

/// Shadow cascade far distance for `render_distance_chunks` — what the
/// status panel shows next to the shadows checkbox (ticket 017), computed
/// the same way [`spawn_sun`]'s initial cascade config and
/// [`shadows::sync_shadow_cascades`]'s rebuilt one both do.
pub(crate) fn shadow_cascade_distance(render_distance_chunks: u32) -> f32 {
    shadows::max_distance(render_distance_chunks)
}

/// Spawns the sky camera, the gradient dome, and the sun/moon billboards —
/// see [`SkyPlugin`]'s docs for why this is a plain function `main.rs::setup`
/// calls rather than a `Startup` system this plugin adds itself.
///
/// The sky camera renders first (`order: -1`) with its own
/// [`ClearColorConfig::Custom`] zenith colour, restricted to [`SKY_LAYER`]
/// so it never draws terrain; the caller is responsible for setting the
/// *main* camera's `clear_color` to [`ClearColorConfig::None`] so it draws
/// on top of this pass instead of erasing it (see the parent ticket).
pub(crate) fn spawn_sky_scene(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    palette: &SkyPalette,
) {
    commands.spawn((
        Name::new("Sky Camera"),
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(palette.zenith_color),
            ..default()
        },
        RenderLayers::layer(SKY_LAYER),
        SkyCamera,
    ));

    let dome_mesh = meshes.add(dome::build_dome_mesh(
        palette.zenith_color,
        palette.horizon_color,
    ));
    let dome_material = materials.add(StandardMaterial {
        unlit: true,
        // See `bodies::spawn_one`'s comment on `cull_mode` — same reasoning
        // applies to the dome's winding.
        cull_mode: None,
        ..default()
    });
    commands.spawn((
        Name::new("Sky Dome"),
        Mesh3d(dome_mesh.clone()),
        MeshMaterial3d(dome_material),
        Transform::IDENTITY,
        RenderLayers::layer(SKY_LAYER),
        NotShadowCaster,
        NotShadowReceiver,
        dome::SkyDome,
    ));
    commands.insert_resource(dome::SkyDomeMesh(dome_mesh));

    bodies::spawn_celestial_bodies(commands, meshes, materials, images, palette.sun_direction);
}

/// Copies the main camera's *rotation* (never its translation, which stays
/// pinned to the origin) onto the sky camera every frame — the only thing
/// that keeps the dome/sun/moon aligned with what the main camera is
/// actually looking at. Silently no-ops if either camera isn't there yet
/// (e.g. before [`spawn_sky_scene`] has run, or in a test app that never
/// calls it).
fn sync_sky_camera_rotation(
    main_camera: Query<&Transform, (With<Camera3d>, Without<SkyCamera>)>,
    mut sky_camera: Query<&mut Transform, With<SkyCamera>>,
) {
    let Ok(main_transform) = main_camera.get_single() else {
        return;
    };
    let Ok(mut sky_transform) = sky_camera.get_single_mut() else {
        return;
    };
    sky_transform.rotation = main_transform.rotation;
}

/// Pushes [`SkyPalette`] out to every dependent piece of scene state
/// whenever it changes (`Res::is_changed()` — true both the frame it's
/// first inserted and any frame something mutates it, e.g. 018's
/// [`time_of_day::sync_palette_from_time_of_day`]): the sun's colour/
/// illuminance/rotation, [`AmbientLight`], [`ClearColor`], and every
/// [`DistanceFog`]'s colour.
///
/// The `DirectionalLight`'s `Transform` is [`time_of_day::moon_handover_direction`]
/// applied to `palette.sun_direction`, not that field directly — see that
/// function's docs for why the light's actual pointing direction and
/// [`SkyPalette::sun_direction`] (what [`bodies`] reads to place the sun/
/// moon billboards) have to diverge near the horizon.
///
/// Does not touch `DistanceFog::falloff` — that tracks
/// [`crate::streaming::RenderDistance`] instead (see
/// `camera::sync_render_distance_effects`), and clobbering it here would
/// fight that system every time either resource changed.
fn sync_sky_palette(
    palette: Res<SkyPalette>,
    mut ambient: ResMut<AmbientLight>,
    mut clear_color: ResMut<ClearColor>,
    mut sun_query: Query<(&mut DirectionalLight, &mut Transform)>,
    mut fog_query: Query<&mut DistanceFog>,
) {
    if !palette.is_changed() {
        return;
    }

    let light_direction = time_of_day::moon_handover_direction(palette.sun_direction);
    for (mut light, mut transform) in &mut sun_query {
        light.color = palette.sun_color;
        light.illuminance = palette.sun_illuminance;
        *transform = Transform::default().looking_to(light_direction, Vec3::Y);
    }

    ambient.color = palette.ambient_color;
    ambient.brightness = palette.ambient_brightness;

    clear_color.0 = palette.horizon_color;

    for mut fog in &mut fog_query {
        fog.color = palette.horizon_color;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal app with [`SkyPlugin`] plus the vanilla-Bevy resources it
    /// writes into — `MinimalPlugins` doesn't insert [`AmbientLight`] or
    /// [`ClearColor`] itself (those come from `DefaultPlugins` in the real
    /// app), so tests provide stand-ins the same way `main.rs` would end up
    /// with real ones.
    fn test_app() -> App {
        let mut app = App::new();
        app.init_resource::<AmbientLight>()
            .init_resource::<ClearColor>()
            // `dome::rebuild_dome_on_palette_change` (016) reads/writes
            // this — real apps get it from `AssetPlugin`, which these
            // `SkyPlugin`-only tests don't add.
            .init_resource::<Assets<Mesh>>()
            .add_plugins(SkyPlugin);
        app
    }

    #[test]
    fn changing_the_palette_updates_ambient_clear_color_and_the_suns_illuminance() {
        let mut app = test_app();
        let sun = app
            .world_mut()
            .spawn((DirectionalLight::default(), Transform::default()))
            .id();

        // First tick: `TimeOfDay` defaults to paused noon, and
        // `time_of_day::sync_palette_from_time_of_day` overwrites the
        // just-inserted default `SkyPalette` with `palette_for_ticks(6000.0)`
        // on this very frame — this test doesn't care what that intermediate
        // value is, only that the manual override below sticks afterward
        // (see the next comment for why it does).
        app.update();

        {
            let mut palette = app.world_mut().resource_mut::<SkyPalette>();
            palette.sun_illuminance = 1234.5;
            palette.ambient_brightness = 42.0;
            palette.ambient_color = Color::srgb(0.1, 0.2, 0.3);
            palette.horizon_color = Color::srgb(0.4, 0.5, 0.6);
        }
        // `TimeOfDay` itself was never touched between the two updates
        // (still paused, still ticks 6000), so `advance_time_of_day` never
        // wrote it and `sync_palette_from_time_of_day` sees `is_changed() ==
        // false` this frame — otherwise it would clobber the manual
        // override above right back to `palette_for_ticks(6000.0)`.
        app.update();

        let light = app.world().get::<DirectionalLight>(sun).unwrap();
        assert_eq!(light.illuminance, 1234.5);

        let ambient = app.world().resource::<AmbientLight>();
        assert_eq!(ambient.brightness, 42.0);
        assert_eq!(ambient.color, Color::srgb(0.1, 0.2, 0.3));

        let clear = app.world().resource::<ClearColor>();
        assert_eq!(clear.0, Color::srgb(0.4, 0.5, 0.6));
    }

    /// With `TimeOfDay` wired into `SkyPlugin` (018), the very first frame
    /// overwrites the hand-picked `SkyPalette::default()` with
    /// `palette_for_ticks(6000.0)` (`TimeOfDay`'s own default: paused,
    /// noon) — so the light's forward axis is expected to match *that*,
    /// straight down, not the old hardcoded default's arbitrary vector.
    #[test]
    fn sun_direction_matches_the_directional_lights_forward_axis() {
        let mut app = test_app();
        let sun = app
            .world_mut()
            .spawn((DirectionalLight::default(), Transform::default()))
            .id();

        app.update();

        let transform = app.world().get::<Transform>(sun).unwrap();
        let forward = transform.rotation * Vec3::NEG_Z;
        assert!(
            forward.dot(Vec3::NEG_Y) > 0.999,
            "expected forward {forward:?} to point straight down at noon"
        );
    }

    #[test]
    fn palette_change_also_updates_any_distance_fog_in_the_scene() {
        let mut app = test_app();
        let camera = app
            .world_mut()
            .spawn(DistanceFog {
                color: Color::srgb(0.0, 0.0, 0.0),
                falloff: FogFalloff::Linear {
                    start: 10.0,
                    end: 20.0,
                },
                ..default()
            })
            .id();

        app.update();

        let fog = app.world().get::<DistanceFog>(camera).unwrap();
        // Not `assert_eq!`: `TimeOfDay`'s default (noon) drives this via
        // `time_of_day::palette_for_ticks`, which lerps through `Color`'s
        // `LinearRgba` variant (see `time_of_day::lerp_color`) even at an
        // exact keyframe tick, while `SkyPalette::default()`'s literal is
        // the `Srgba` variant — the two numerically agree (the noon
        // keyframe's horizon colour is deliberately the same hex as the
        // hardcoded default) but `Color`'s derived `PartialEq` compares the
        // variant tag first, so a bitwise `assert_eq!` would fail on two
        // representations of the identical colour.
        let got = fog.color.to_srgba();
        let expected = SkyPalette::default().horizon_color.to_srgba();
        assert!(
            (got.red - expected.red).abs() < 1e-4
                && (got.green - expected.green).abs() < 1e-4
                && (got.blue - expected.blue).abs() < 1e-4,
            "expected {expected:?}, got {got:?}"
        );
        // Falloff is untouched — that's `camera.rs`'s job.
        match &fog.falloff {
            FogFalloff::Linear { start, end } => {
                assert_eq!(*start, 10.0);
                assert_eq!(*end, 20.0);
            }
            other => panic!("expected linear falloff, got {other:?}"),
        }
    }
}
