//! Sun shadows (ticket 017): cascaded shadow maps on the `DirectionalLight`
//! [`super::spawn_sun`] spawns — the bias constants baked into that spawn,
//! the cascade config that tracks [`streaming::RenderDistance`], and the
//! status-panel toggle that flips `shadows_enabled` without touching
//! anything else about the light (colour/illuminance/rotation stay
//! `sync_sky_palette`'s job — same "one field, one owner" split as the rest
//! of this module).

use bevy::pbr::{CascadeShadowConfig, CascadeShadowConfigBuilder};
use bevy::prelude::*;

use crate::{camera, streaming};

/// Kept at Bevy's default (`DirectionalLight::DEFAULT_SHADOW_DEPTH_BIAS` =
/// 0.02). Depth bias is the peter-panning knob; the acne problem the ticket
/// calls out is a normal-bias problem (see [`SHADOW_NORMAL_BIAS`]'s docs),
/// so there's no reason yet to move this off Bevy's tuned default.
pub(crate) const SHADOW_DEPTH_BIAS: f32 = DirectionalLight::DEFAULT_SHADOW_DEPTH_BIAS;

/// Above Bevy's default (`DirectionalLight::DEFAULT_SHADOW_NORMAL_BIAS` =
/// 1.8). Voxel terrain is close to the pathological case for cascaded
/// shadow maps: enormous, perfectly flat, axis-aligned faces, lit by a sun
/// that (once 018 animates it) spends part of the day at a grazing angle to
/// all of them — exactly where shadow acne is worst, since self-shadowing
/// error grows with the angle between the surface and the light. Normal
/// bias pushes the sampled shadow-map position along the surface normal,
/// which directly counters that grazing-angle error; depth bias doesn't,
/// since it pushes along the light's view direction regardless of how the
/// surface is tilted relative to it.
///
/// This is a starting value, not a measured one — nothing in this repo
/// drives the camera to actually look at a render (see CLAUDE.md's
/// "Manual/visual verification"), so it hasn't been checked against real
/// banding or peter-panning at a low sun angle. `todo.md` carries that
/// check; adjust this constant if it turns out wrong in either direction.
pub(crate) const SHADOW_NORMAL_BIAS: f32 = 3.0;

/// Cap on cascade `maximum_distance`, independent of render distance — see
/// [`max_distance`]'s docs for why stretching cascades all the way to the
/// far plane is worse than just leaving distant terrain unshadowed.
const CASCADE_MAX_DISTANCE_CAP: f32 = 250.0;

/// User-facing shadow toggle (status panel, ticket 017) — [`sync_shadow_settings`]
/// is the only system that reads it, and `DirectionalLight::shadows_enabled`
/// is the only thing it touches.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowSettings {
    pub enabled: bool,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Cascade `maximum_distance` for a sun casting shadows out to
/// `render_distance_chunks` chunks. Deliberately **not**
/// [`camera::far_plane_distance`] outright: at render distance 32 that's
/// ~530 blocks, and stretching 4 cascades over 530 blocks makes the near
/// ones coarse enough that shadow edges crawl as the camera moves. Capping
/// at [`CASCADE_MAX_DISTANCE_CAP`] keeps the near cascades sharp and leaves
/// distant terrain unshadowed — `camera::atmosphere_fog` is doing most of
/// the work hiding distant terrain by then anyway.
pub(crate) fn max_distance(render_distance_chunks: u32) -> f32 {
    camera::far_plane_distance(render_distance_chunks).min(CASCADE_MAX_DISTANCE_CAP)
}

/// The cascade config for a sun casting shadows out to `render_distance_chunks`
/// chunks — the same shape [`super::spawn_sun`] uses at startup and
/// [`sync_shadow_cascades`] rebuilds at runtime, so the two can never drift
/// apart.
pub(crate) fn cascade_config(render_distance_chunks: u32) -> CascadeShadowConfig {
    CascadeShadowConfigBuilder {
        num_cascades: 4,
        minimum_distance: 0.1,
        first_cascade_far_bound: 32.0,
        maximum_distance: max_distance(render_distance_chunks),
        overlap_proportion: 0.2,
    }
    .build()
}

/// Keeps the sun's [`CascadeShadowConfig`] tracking [`streaming::RenderDistance`]
/// (ticket 007's status-panel slider) the same way
/// `camera::sync_render_distance_effects` keeps the far plane and fog in
/// step with it — without this, widening render distance at runtime would
/// leave shadow cascades sized for the old, narrower one.
pub(crate) fn sync_shadow_cascades(
    render_distance: Res<streaming::RenderDistance>,
    mut query: Query<&mut CascadeShadowConfig, With<DirectionalLight>>,
) {
    if !render_distance.is_changed() {
        return;
    }
    for mut config in &mut query {
        *config = cascade_config(render_distance.0);
    }
}

/// Flips `DirectionalLight::shadows_enabled` to match [`ShadowSettings`]
/// whenever it changes — the status panel checkbox's only effect. Leaves
/// every other `DirectionalLight` field alone (`sync_sky_palette` owns
/// colour/illuminance/rotation).
pub(crate) fn sync_shadow_settings(
    settings: Res<ShadowSettings>,
    mut query: Query<&mut DirectionalLight>,
) {
    if !settings.is_changed() {
        return;
    }
    for mut light in &mut query {
        light.shadows_enabled = settings.enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ticket 017's "Tests" section: "that the cascade config tracks
    /// `RenderDistance`".
    #[test]
    fn cascade_config_follows_render_distance_changes() {
        let mut app = App::new();
        app.add_systems(Update, sync_shadow_cascades);
        app.insert_resource(streaming::RenderDistance(4));

        let sun = app
            .world_mut()
            .spawn((DirectionalLight::default(), cascade_config(4)))
            .id();

        // First tick applies the just-inserted RenderDistance.
        app.update();
        app.world_mut()
            .resource_mut::<streaming::RenderDistance>()
            .0 = 20;
        app.update();

        let config = app.world().get::<CascadeShadowConfig>(sun).unwrap();
        let expected = cascade_config(20);
        assert_eq!(config.bounds, expected.bounds);
    }

    /// Ticket 017's "Tests" section: "that the shadows toggle actually
    /// flips `DirectionalLight::shadows_enabled`".
    #[test]
    fn shadow_toggle_flips_shadows_enabled() {
        let mut app = App::new();
        app.add_systems(Update, sync_shadow_settings);
        app.insert_resource(ShadowSettings { enabled: true });

        let sun = app
            .world_mut()
            .spawn(DirectionalLight {
                shadows_enabled: false,
                ..default()
            })
            .id();

        // First tick applies the just-inserted ShadowSettings.
        app.update();
        assert!(
            app.world()
                .get::<DirectionalLight>(sun)
                .unwrap()
                .shadows_enabled
        );

        app.world_mut().resource_mut::<ShadowSettings>().enabled = false;
        app.update();
        assert!(
            !app.world()
                .get::<DirectionalLight>(sun)
                .unwrap()
                .shadows_enabled
        );
    }

    /// Regression guard for the `.min(250.0)` cap the ticket calls out
    /// explicitly: at a high render distance the far plane is well past the
    /// cap, so cascades must not stretch out to it.
    #[test]
    fn cascade_maximum_distance_is_capped_at_high_render_distance() {
        let far = camera::far_plane_distance(32);
        assert!(
            far > CASCADE_MAX_DISTANCE_CAP,
            "test assumes render distance 32 exceeds the cap"
        );

        let config = cascade_config(32);
        let maximum_distance = *config.bounds.last().unwrap();
        assert!((maximum_distance - CASCADE_MAX_DISTANCE_CAP).abs() < 1e-3);
    }
}
