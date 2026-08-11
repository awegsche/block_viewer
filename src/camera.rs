//! Free-flying + orbit/inspect camera (ticket 006).
//!
//! This is the one camera implementation in the repo: it replaces both the
//! fixed-target `orbit` system that used to live in `main.rs` and the
//! never-compiled `pan_orbit_camera_bundle.rs` (deleted — two half-camera
//! implementations was the actual bug ticket 006 called out).
//!
//! ## Modes
//!
//! - [`CameraMode::Fly`]: WASD (+ Q/E for straight up/down) move relative to
//!   where the camera is looking, sprint on Left Shift, mouse-look while the
//!   right mouse button is held (the cursor is grabbed and hidden for the
//!   duration), scroll adjusts fly speed.
//! - [`CameraMode::Orbit`]: orbits [`CameraRig::orbit_target`] — left-drag
//!   to rotate, scroll to zoom. `Tab` toggles between the two modes; going
//!   Fly -> Orbit re-aims the target by ray-marching the loaded voxel grid
//!   under the cursor (falling back to a point straight ahead if the cursor
//!   isn't over any loaded terrain), so orbit is never stuck on the origin.
//!
//! Ray-marching against the decoded block grid (rather than mesh picking)
//! keeps this independent of `bevy_picking`/render-side raycasting — it
//! only needs the same `DecodedWorld` the mesher already reads.

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};
use std::ops::Range;

use crate::{world, DecodedWorld};

/// Fly speed (blocks/sec) a freshly spawned rig starts at — brisk enough to
/// cross a chunk in well under a second without being uncontrollable.
const DEFAULT_FLY_SPEED: f32 = 24.0;
const MIN_FLY_SPEED: f32 = 2.0;
const MAX_FLY_SPEED: f32 = 400.0;

/// Chunk radius assumed when picking the far clip plane and fog falloff.
/// Ticket 005 will introduce a real, tunable `render_distance` resource;
/// until chunk streaming lands there's nothing to read, so this mirrors the
/// modest default (8-12 chunks) that ticket documents for itself.
const PLACEHOLDER_RENDER_DISTANCE_CHUNKS: f32 = 12.0;

/// Adds [`CameraSettings`] and the system that drives every
/// [`CameraRig`]-tagged camera. Spawn the camera itself (with [`CameraRig`],
/// [`far_plane_distance`] wired into its `Projection`, and [`atmosphere_fog`])
/// wherever the app already knows where to place it — this plugin only
/// handles input, not initial placement.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraSettings>()
            .add_systems(Update, drive_camera);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Fly,
    Orbit,
}

/// Per-camera runtime state. Yaw/pitch are shared between both modes so
/// switching mode never snaps the view: only how translation is derived
/// (free movement vs. orbiting a target) changes.
#[derive(Component, Debug)]
pub struct CameraRig {
    pub mode: CameraMode,
    yaw: f32,
    pitch: f32,
    fly_speed: f32,
    orbit_target: Vec3,
    orbit_radius: f32,
}

impl CameraRig {
    /// A rig starting in [`CameraMode::Fly`] at `eye`, looking at `target`.
    /// `target` also seeds `orbit_target`, so toggling straight into orbit
    /// mode without ever re-aiming still orbits something sane instead of
    /// the world origin.
    pub fn looking_at(eye: Vec3, target: Vec3) -> Self {
        let rotation = Transform::from_translation(eye)
            .looking_at(target, Vec3::Y)
            .rotation;
        let (yaw, pitch, _roll) = rotation.to_euler(EulerRot::YXZ);
        Self {
            mode: CameraMode::Fly,
            yaw,
            pitch,
            fly_speed: DEFAULT_FLY_SPEED,
            orbit_target: target,
            orbit_radius: eye.distance(target).max(2.0),
        }
    }
}

#[derive(Resource, Debug)]
pub struct CameraSettings {
    /// Radians of look rotation per pixel of mouse motion, shared by
    /// fly-mode mouse-look and orbit-mode dragging.
    pub mouse_sensitivity: f32,
    pub pitch_range: Range<f32>,
    pub sprint_multiplier: f32,
    /// Multiplier applied to `fly_speed` per unit of scroll (exponential,
    /// so repeated scroll steps feel consistent at any speed).
    pub speed_scroll_factor: f32,
    pub orbit_zoom_sensitivity: f32,
    pub min_orbit_radius: f32,
    /// How far (blocks) the "what's under the cursor" ray-march searches
    /// before giving up and aiming straight ahead instead.
    pub max_ray_distance: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        // Limiting pitch stops the view flipping past straight up/down.
        let pitch_limit = std::f32::consts::FRAC_PI_2 - 0.01;
        Self {
            mouse_sensitivity: 0.0025,
            pitch_range: -pitch_limit..pitch_limit,
            sprint_multiplier: 4.0,
            speed_scroll_factor: 0.2,
            orbit_zoom_sensitivity: 0.15,
            min_orbit_radius: 2.0,
            max_ray_distance: 300.0,
        }
    }
}

/// Far clip plane distance (world units) that comfortably covers
/// [`PLACEHOLDER_RENDER_DISTANCE_CHUNKS`] chunks in every horizontal
/// direction, including the diagonal — Bevy's default (1000) happens to
/// already clear this, but wiring it through explicitly keeps the far plane
/// correct once ticket 005 makes render distance real and tunable.
pub fn far_plane_distance() -> f32 {
    PLACEHOLDER_RENDER_DISTANCE_CHUNKS * world::SECTION_SIZE as f32 * std::f32::consts::SQRT_2
        + 32.0
}

/// A soft distance fog fading terrain out before the far plane, so chunks
/// don't visibly pop out of existence at the render-distance edge.
pub fn atmosphere_fog() -> DistanceFog {
    let far = far_plane_distance();
    DistanceFog {
        color: Color::srgb(0.7, 0.8, 0.92),
        falloff: FogFalloff::Linear {
            start: far * 0.6,
            end: far,
        },
        ..default()
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_camera(
    rig: Single<(&mut CameraRig, &mut Transform, &Camera, &GlobalTransform)>,
    settings: Res<CameraSettings>,
    decoded_world: Res<DecodedWorld>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    time: Res<Time>,
) {
    let (mut rig, mut transform, camera, camera_transform) = rig.into_inner();
    let mut window = windows.get_single_mut().ok();

    if keys.just_pressed(KeyCode::Tab) {
        toggle_mode(
            &mut rig,
            &transform,
            camera,
            camera_transform,
            &settings,
            &decoded_world,
            window.as_deref(),
        );
        // Orbiting reads the cursor freely for re-aiming; a grab held over
        // from fly mode would just be confusing.
        if rig.mode == CameraMode::Orbit {
            set_cursor_grab(window.as_deref_mut(), false);
        }
    }

    if rig.mode == CameraMode::Fly {
        if mouse_buttons.just_pressed(MouseButton::Right) {
            set_cursor_grab(window.as_deref_mut(), true);
        }
        if mouse_buttons.just_released(MouseButton::Right) {
            set_cursor_grab(window.as_deref_mut(), false);
        }
    }

    let look_button = match rig.mode {
        CameraMode::Fly => MouseButton::Right,
        CameraMode::Orbit => MouseButton::Left,
    };
    if mouse_buttons.pressed(look_button) {
        let delta = mouse_motion.delta;
        rig.yaw -= delta.x * settings.mouse_sensitivity;
        rig.pitch = (rig.pitch - delta.y * settings.mouse_sensitivity)
            .clamp(settings.pitch_range.start, settings.pitch_range.end);
    }
    let rotation = Quat::from_euler(EulerRot::YXZ, rig.yaw, rig.pitch, 0.0);
    transform.rotation = rotation;

    let scroll = mouse_scroll.delta.y;
    match rig.mode {
        CameraMode::Fly => {
            if scroll != 0.0 {
                rig.fly_speed = (rig.fly_speed * (1.0 + scroll * settings.speed_scroll_factor))
                    .clamp(MIN_FLY_SPEED, MAX_FLY_SPEED);
            }

            let mut move_dir = Vec3::ZERO;
            if keys.pressed(KeyCode::KeyW) {
                move_dir += rotation * Vec3::NEG_Z;
            }
            if keys.pressed(KeyCode::KeyS) {
                move_dir += rotation * Vec3::Z;
            }
            if keys.pressed(KeyCode::KeyD) {
                move_dir += rotation * Vec3::X;
            }
            if keys.pressed(KeyCode::KeyA) {
                move_dir += rotation * Vec3::NEG_X;
            }
            // Q/E move straight up/down in world space regardless of pitch,
            // rather than along the (possibly tilted) view direction like
            // WASD does — keeping a way to gain/lose altitude on command
            // even while looking level.
            if keys.pressed(KeyCode::KeyE) {
                move_dir += Vec3::Y;
            }
            if keys.pressed(KeyCode::KeyQ) {
                move_dir += Vec3::NEG_Y;
            }

            if move_dir != Vec3::ZERO {
                let speed = rig.fly_speed
                    * if keys.pressed(KeyCode::ShiftLeft) {
                        settings.sprint_multiplier
                    } else {
                        1.0
                    };
                transform.translation += move_dir.normalize() * speed * time.delta_secs();
            }
        }
        CameraMode::Orbit => {
            if scroll != 0.0 {
                rig.orbit_radius = (rig.orbit_radius
                    * (1.0 - scroll * settings.orbit_zoom_sensitivity))
                    .max(settings.min_orbit_radius);
            }
            transform.translation = rig.orbit_target + rotation * Vec3::new(0.0, 0.0, rig.orbit_radius);
        }
    }
}

/// Handles a `Tab` press: leaves orbit mode as-is, or (from fly mode)
/// ray-marches the loaded voxel grid under the cursor to pick a new orbit
/// target. Falls back to a point straight ahead at `max_ray_distance` if the
/// cursor isn't over any solid, loaded block, so orbit mode is never
/// unreachable just because you're looking at open sky.
fn toggle_mode(
    rig: &mut CameraRig,
    transform: &Transform,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    settings: &CameraSettings,
    decoded_world: &DecodedWorld,
    window: Option<&Window>,
) {
    if rig.mode == CameraMode::Orbit {
        rig.mode = CameraMode::Fly;
        return;
    }

    let Some(cursor) = window.and_then(Window::cursor_position) else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };

    let target = raycast_terrain(ray.origin, *ray.direction, settings.max_ray_distance, decoded_world)
        .unwrap_or_else(|| ray.origin + *ray.direction * settings.max_ray_distance);

    rig.orbit_radius = transform.translation.distance(target).max(settings.min_orbit_radius);
    rig.orbit_target = target;
    rig.mode = CameraMode::Orbit;
}

fn set_cursor_grab(window: Option<&mut Window>, grab: bool) {
    let Some(window) = window else { return };
    if grab {
        window.cursor_options.grab_mode = CursorGrabMode::Locked;
        window.cursor_options.visible = false;
    } else {
        window.cursor_options.grab_mode = CursorGrabMode::None;
        window.cursor_options.visible = true;
    }
}

/// Steps a ray (Bevy space, `direction` normalized) through the loaded
/// voxel grid one block at a time (Amanatides-Woo DDA) and returns the
/// world-space entry point of the first solid block hit, or `None` if
/// nothing solid is within `max_distance`.
fn raycast_terrain(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    decoded_world: &DecodedWorld,
) -> Option<Vec3> {
    // The decoded grid is addressed in Minecraft block coordinates; bevy.z
    // = -mc.z (see `world::mesh` docs) is the only axis that flips.
    let mc_origin = Vec3::new(origin.x, origin.y, -origin.z);
    let mc_dir = Vec3::new(direction.x, direction.y, -direction.z);

    let mut voxel = mc_origin.floor().as_ivec3();
    let step = IVec3::new(signum(mc_dir.x), signum(mc_dir.y), signum(mc_dir.z));
    let t_delta = Vec3::new(safe_inv(mc_dir.x), safe_inv(mc_dir.y), safe_inv(mc_dir.z));
    let mut t_max = Vec3::new(
        next_boundary_t(mc_origin.x, voxel.x, step.x, mc_dir.x),
        next_boundary_t(mc_origin.y, voxel.y, step.y, mc_dir.y),
        next_boundary_t(mc_origin.z, voxel.z, step.z, mc_dir.z),
    );

    let mut traveled = 0.0f32;
    loop {
        if world::is_solid(
            block_at_world(decoded_world, voxel.x, voxel.y, voxel.z),
            &decoded_world.registry,
        ) {
            let hit = mc_origin + mc_dir * traveled;
            return Some(Vec3::new(hit.x, hit.y, -hit.z));
        }

        traveled = if t_max.x < t_max.y && t_max.x < t_max.z {
            voxel.x += step.x;
            let t = t_max.x;
            t_max.x += t_delta.x;
            t
        } else if t_max.y < t_max.z {
            voxel.y += step.y;
            let t = t_max.y;
            t_max.y += t_delta.y;
            t
        } else {
            voxel.z += step.z;
            let t = t_max.z;
            t_max.z += t_delta.z;
            t
        };

        if traveled > max_distance {
            return None;
        }
    }
}

fn signum(v: f32) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn safe_inv(v: f32) -> f32 {
    if v == 0.0 {
        f32::INFINITY
    } else {
        (1.0 / v).abs()
    }
}

fn next_boundary_t(origin: f32, voxel: i32, step: i32, dir: f32) -> f32 {
    if step == 0 {
        return f32::INFINITY;
    }
    let boundary = if step > 0 { (voxel + 1) as f32 } else { voxel as f32 };
    (boundary - origin) / dir
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::BlockRegistry;
    use crate::world::decode::{ChunkColumn, ChunkSection, SECTION_VOLUME};
    use std::collections::HashMap;

    /// A [`DecodedWorld`] containing exactly one solid block at Minecraft
    /// coordinates `(x, y, z)` — enough to exercise [`raycast_terrain`]
    /// without needing a real save.
    fn single_block_world(x: i32, y: i32, z: i32) -> DecodedWorld {
        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");

        let size = world::SECTION_SIZE as i32;
        let (cx, cz) = (x.div_euclid(size), z.div_euclid(size));
        let (local_x, local_z) = (x.rem_euclid(size) as usize, z.rem_euclid(size) as usize);
        let section_y = y.div_euclid(size) as i8;
        let local_y = y.rem_euclid(size) as usize;

        let mut blocks = Box::new([BlockRegistry::AIR; SECTION_VOLUME]);
        blocks[ChunkSection::index(local_x, local_y, local_z)] = stone;

        let mut columns = HashMap::new();
        columns.insert(
            (cx, cz),
            ChunkColumn {
                x: cx,
                z: cz,
                sections: vec![ChunkSection { y: section_y, blocks }],
            },
        );
        DecodedWorld { registry, columns }
    }

    #[test]
    fn raycast_hits_a_placed_block() {
        let decoded = single_block_world(5, 10, 5);
        // Marching along +mc.x (bevy.z = -mc.z, so bevy.z = -5.5 for mc.z =
        // 5.5) through the middle of the block's y/z span, starting well
        // outside it on the west side.
        let origin = Vec3::new(-10.0, 10.5, -5.5);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raycast_terrain(origin, direction, 100.0, &decoded).expect("should hit the block");
        // Entry point is the block's west face, at mc.x = 5.
        assert_eq!(hit.x.floor() as i32, 5);
    }

    #[test]
    fn raycast_misses_when_nothing_is_in_the_path() {
        let decoded = single_block_world(5, 10, 5);
        // Same horizontal path, but well above the block.
        let origin = Vec3::new(-10.0, 200.0, -5.5);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        assert!(raycast_terrain(origin, direction, 100.0, &decoded).is_none());
    }

    #[test]
    fn raycast_gives_up_past_max_distance() {
        let decoded = single_block_world(5, 10, 5);
        // The block's west face is 15 blocks out; a shorter budget should
        // never reach it.
        let origin = Vec3::new(-10.0, 10.5, -5.5);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        assert!(raycast_terrain(origin, direction, 5.0, &decoded).is_none());
    }
}

/// Block at Minecraft-space `(x, y, z)`, across whichever loaded column it
/// falls in. Unloaded columns/sections both read back as air, same as the
/// mesher's neighbour lookups.
fn block_at_world(decoded_world: &DecodedWorld, x: i32, y: i32, z: i32) -> world::BlockId {
    let size = world::SECTION_SIZE as i32;
    let Some(column) = decoded_world.columns.get(&(x.div_euclid(size), z.div_euclid(size))) else {
        return world::BlockRegistry::AIR;
    };
    let local_x = x.rem_euclid(size) as usize;
    let local_z = z.rem_euclid(size) as usize;
    let section_y = y.div_euclid(size) as i8;
    let local_y = y.rem_euclid(size) as usize;
    column
        .sections
        .iter()
        .find(|s| s.y == section_y)
        .map(|s| s.get(local_x, local_y, local_z))
        .unwrap_or(world::BlockRegistry::AIR)
}
