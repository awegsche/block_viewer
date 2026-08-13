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
//! 016 draws an actual sky gradient from [`SkyPalette::zenith_color`] /
//! [`SkyPalette::horizon_color`], 017 adds shadows to the sun this spawns,
//! 018 animates the palette over a day/night cycle — all three build on top
//! of this rather than re-deriving their own lighting state.

use bevy::prelude::*;

/// Everything visual about the atmosphere, in one place. `Default` is a
/// fixed noon — there is no day/night animation yet (that's ticket 018);
/// changing this resource at runtime (by mutating it directly, or replacing
/// it wholesale) is what 018 will eventually drive from a clock.
#[derive(Resource, Debug, Clone)]
pub struct SkyPalette {
    /// Direction the sunlight travels (from the sun toward the ground) —
    /// this, not a position, is all a [`DirectionalLight`] cares about.
    pub sun_direction: Vec3,
    pub sun_color: Color,
    pub sun_illuminance: f32,
    pub ambient_color: Color,
    pub ambient_brightness: f32,
    /// Colour straight up. 016 draws the gradient between this and
    /// [`Self::horizon_color`]; this ticket only stores it, so nothing
    /// reads it yet.
    #[allow(dead_code)]
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

/// Adds [`SkyPalette`] (defaulting to noon) and [`sync_sky_palette`].
pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyPalette>()
            .add_systems(Update, sync_sky_palette);
    }
}

/// Pushes [`SkyPalette`] out to every dependent piece of scene state
/// whenever it changes (`Res::is_changed()` — true both the frame it's
/// first inserted and any frame something mutates it, e.g. a future 018
/// day/night system): the sun's colour/illuminance/rotation, [`AmbientLight`],
/// [`ClearColor`], and every [`DistanceFog`]'s colour.
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

    for (mut light, mut transform) in &mut sun_query {
        light.color = palette.sun_color;
        light.illuminance = palette.sun_illuminance;
        *transform = Transform::default().looking_to(palette.sun_direction, Vec3::Y);
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

        // First tick applies the just-inserted default palette.
        app.update();

        {
            let mut palette = app.world_mut().resource_mut::<SkyPalette>();
            palette.sun_illuminance = 1234.5;
            palette.ambient_brightness = 42.0;
            palette.ambient_color = Color::srgb(0.1, 0.2, 0.3);
            palette.horizon_color = Color::srgb(0.4, 0.5, 0.6);
        }
        app.update();

        let light = app.world().get::<DirectionalLight>(sun).unwrap();
        assert_eq!(light.illuminance, 1234.5);

        let ambient = app.world().resource::<AmbientLight>();
        assert_eq!(ambient.brightness, 42.0);
        assert_eq!(ambient.color, Color::srgb(0.1, 0.2, 0.3));

        let clear = app.world().resource::<ClearColor>();
        assert_eq!(clear.0, Color::srgb(0.4, 0.5, 0.6));
    }

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
        let expected = SkyPalette::default().sun_direction;
        assert!(
            forward.dot(expected) > 0.999,
            "expected forward {forward:?} to match sun_direction {expected:?}"
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
        assert_eq!(fog.color, SkyPalette::default().horizon_color);
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
