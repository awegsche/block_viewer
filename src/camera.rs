//! Free-flying + orbit/inspect camera (ticket 006).
//!
//! This is the one camera implementation in the repo: it replaces both the
//! fixed-target `orbit` system that used to live in the old `main.rs` and the
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

use crate::{streaming, world, DecodedWorld};

/// Fly speed (blocks/sec) a freshly spawned rig starts at — brisk enough to
/// cross a chunk in well under a second without being uncontrollable.
const DEFAULT_FLY_SPEED: f32 = 24.0;
const MIN_FLY_SPEED: f32 = 2.0;
const MAX_FLY_SPEED: f32 = 400.0;

/// Adds [`CameraSettings`] and the system that drives every
/// [`CameraRig`]-tagged camera. Spawn the camera itself (with [`CameraRig`],
/// [`far_plane_distance`] wired into its `Projection`, and [`atmosphere_fog`])
/// wherever the app already knows where to place it — this plugin only
/// handles input, not initial placement.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraSettings>()
            .init_resource::<EguiInputCapture>()
            .init_resource::<CameraStartMode>()
            .add_systems(
                Update,
                (drive_camera.in_set(CameraSet), sync_render_distance_effects),
            );
    }
}

/// Marker [`SystemSet`] for [`drive_camera`] — `viewer::run` orders this
/// `.after()` the UI plugin's own panel-drawing set (ticket 007) so
/// [`EguiInputCapture`], which the UI updates once per frame right after it
/// finishes drawing, reflects *this* frame's panels by the time
/// [`drive_camera`] reads it rather than lagging a frame behind.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraSet;

/// Whether egui claimed this frame's pointer/keyboard input, set once per
/// frame by the UI layer (ticket 007, `ui::sync_egui_input_capture`) after
/// it finishes drawing every panel. [`drive_camera`] checks this so
/// dragging a slider or typing into a coordinate field doesn't also spin
/// the view or fly the camera — see the parent ticket's "watch out" note.
/// Defaults to `false` (nothing captured) so the camera behaves normally
/// before the UI plugin's first frame, and degrades the same way if the UI
/// plugin isn't present at all.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct EguiInputCapture {
    pub pointer: bool,
    pub keyboard: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Fly,
    Orbit,
    /// Pan/zoom/rotate over the terrain (ticket 045, roadmap E1) — the
    /// citybuilder's rig. Orbits `orbit_target` the same way [`CameraMode::Orbit`]
    /// does, but `W`/`A`/`S`/`D` pan the target instead of the camera flying
    /// free, and rotation lives on the right mouse button (ungrabbed) rather
    /// than the left, which the citybuilder reserves for picking/placement.
    Rts,
}

impl Default for CameraMode {
    /// [`CameraRig::looking_at`]'s mode, and [`CameraStartMode`]'s default —
    /// the viewer's rig unless something overrides it.
    fn default() -> Self {
        CameraMode::Fly
    }
}

/// Which [`CameraMode`] a freshly spawned [`CameraRig`] starts in, read once
/// by `lib.rs::setup_world` when it constructs the camera entity.
/// [`CameraControllerPlugin`] inits this to [`CameraMode::Fly`] (the
/// viewer's rig); [`crate::city::run`] overrides it to [`CameraMode::Rts`]
/// after [`crate::world_app`] returns and before `App::run()`, the same
/// override-after-`world_app()` shape ticket 030's `RenderFloor` already
/// uses.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct CameraStartMode(pub CameraMode);

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
            mode: CameraMode::default(),
            yaw,
            pitch,
            fly_speed: DEFAULT_FLY_SPEED,
            orbit_target: target,
            orbit_radius: eye.distance(target).max(2.0),
        }
    }

    /// Overrides the mode a rig starts in — `lib.rs::setup_world` calls this
    /// with [`CameraStartMode`]'s value right after [`Self::looking_at`],
    /// rather than [`looking_at`](Self::looking_at) taking the mode as a
    /// third argument, so every existing call site (and every test that
    /// doesn't care what mode it gets) stays untouched.
    pub fn with_mode(mut self, mode: CameraMode) -> Self {
        self.mode = mode;
        self
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
    /// [`CameraMode::Rts`]'s own pitch clamp — steeper than `pitch_range`,
    /// so the citybuilder's camera always looks down at the terrain rather
    /// than up toward the horizon or straight down at its own feet.
    pub rts_pitch_range: Range<f32>,
    /// `Rts` pan speed in blocks/sec — constant regardless of zoom. Used to
    /// scale with `orbit_radius`, which made panning crawl to a halt when
    /// zoomed in; this is 20% faster than that old formula produced at the
    /// rig's default spawn zoom level (`orbit_radius` ≈ 39.4, from
    /// `lib.rs::setup_world`'s `eye = target + Vec3::new(-24.0, 20.0, 24.0)`
    /// offset): `39.395 * 1.2 * 1.2 ≈ 56.73`.
    pub rts_pan_speed: f32,
    /// Radians/sec of yaw rotation from `Rts`'s `Q`/`E` keys.
    pub rts_rotate_speed: f32,
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
            // Roughly -80°..-11°: always angled down at the terrain, never
            // up toward the horizon and never near straight down.
            rts_pitch_range: -1.4..-0.2,
            rts_pan_speed: 56.73,
            rts_rotate_speed: std::f32::consts::FRAC_PI_2,
        }
    }
}

/// Far clip plane distance (world units) that comfortably covers
/// `render_distance_chunks` chunks in every horizontal direction, including
/// the diagonal — wired to the real, tunable
/// [`RenderDistance`](crate::streaming::RenderDistance) resource (ticket
/// 005-e) rather than a fixed placeholder.
pub fn far_plane_distance(render_distance_chunks: u32) -> f32 {
    render_distance_chunks as f32 * world::SECTION_SIZE as f32 * std::f32::consts::SQRT_2 + 32.0
}

/// The falloff half of [`atmosphere_fog`] — split out so
/// [`sync_render_distance_effects`] can update just this when
/// [`streaming::RenderDistance`] changes, without touching `DistanceFog`'s
/// colour. That colour belongs to [`crate::sky::SkyPalette`] (ticket 015)
/// now; rebuilding the whole `DistanceFog` here would clobber it back to
/// whatever `color` was passed in every time the render-distance slider
/// moved.
fn fog_falloff(render_distance_chunks: u32) -> FogFalloff {
    let far = far_plane_distance(render_distance_chunks);
    FogFalloff::Linear {
        start: far * 0.6,
        end: far,
    }
}

/// A soft distance fog fading terrain out before the far plane, so chunks
/// don't visibly pop out of existence at the render-distance edge. `color`
/// is a caller-supplied starting value (`lib.rs::setup_world` passes
/// [`crate::sky::SkyPalette::horizon_color`]) — `sky::sync_sky_palette`
/// keeps it in sync with the palette from then on, and
/// [`sync_render_distance_effects`] never touches it, only the falloff.
pub fn atmosphere_fog(render_distance_chunks: u32, color: Color) -> DistanceFog {
    DistanceFog {
        color,
        falloff: fog_falloff(render_distance_chunks),
        ..default()
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_camera(
    rig: Single<(&mut CameraRig, &mut Transform, &Camera, &GlobalTransform)>,
    settings: Res<CameraSettings>,
    decoded_world: Res<DecodedWorld>,
    egui_input: Res<EguiInputCapture>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    time: Res<Time>,
) {
    let (mut rig, mut transform, camera, camera_transform) = rig.into_inner();
    let mut window = windows.get_single_mut().ok();

    // Ticket 007: while egui has keyboard/pointer focus (typing into a
    // coordinate field, dragging a slider, ...), the same key/mouse state
    // this system reads is also what egui just consumed — skip driving the
    // camera from it so e.g. dragging the render-distance slider doesn't
    // also spin the view. `just_released` stays ungated below so a grab
    // that started before a panel opened over the cursor still lets go
    // cleanly.
    if !egui_input.keyboard && rig.mode != CameraMode::Rts && keys.just_pressed(KeyCode::Tab) {
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
        if !egui_input.pointer && mouse_buttons.just_pressed(MouseButton::Right) {
            set_cursor_grab(window.as_deref_mut(), true);
        }
        if mouse_buttons.just_released(MouseButton::Right) {
            set_cursor_grab(window.as_deref_mut(), false);
        }
    }

    let look_button = match rig.mode {
        CameraMode::Fly => MouseButton::Right,
        CameraMode::Orbit => MouseButton::Left,
        // Right, same as `Fly`, but never grabbed (see the `Fly`-only grab
        // block above) — left mouse stays free for the citybuilder's
        // picking/placement (E3/E4), and the cursor stays visible.
        CameraMode::Rts => MouseButton::Right,
    };
    if !egui_input.pointer && mouse_buttons.pressed(look_button) {
        let delta = mouse_motion.delta;
        rig.yaw -= delta.x * settings.mouse_sensitivity;
        let pitch_range = if rig.mode == CameraMode::Rts {
            &settings.rts_pitch_range
        } else {
            &settings.pitch_range
        };
        rig.pitch = (rig.pitch - delta.y * settings.mouse_sensitivity)
            .clamp(pitch_range.start, pitch_range.end);
    }
    // `Rts`'s `Q`/`E` yaw rotation is read here, before `rotation` is
    // computed below, rather than inside the mode match further down — a
    // same-frame press would otherwise show up in `rig.yaw` but not yet in
    // `transform.translation`'s orbit offset until the *next* frame.
    if rig.mode == CameraMode::Rts && !egui_input.keyboard {
        let mut rotate = 0.0;
        if keys.pressed(KeyCode::KeyQ) {
            rotate += settings.rts_rotate_speed;
        }
        if keys.pressed(KeyCode::KeyE) {
            rotate -= settings.rts_rotate_speed;
        }
        rig.yaw += rotate * time.delta_secs();
    }
    let rotation = Quat::from_euler(EulerRot::YXZ, rig.yaw, rig.pitch, 0.0);
    transform.rotation = rotation;

    let scroll = if egui_input.pointer { 0.0 } else { mouse_scroll.delta.y };
    match rig.mode {
        CameraMode::Fly => {
            if scroll != 0.0 {
                rig.fly_speed = (rig.fly_speed * (1.0 + scroll * settings.speed_scroll_factor))
                    .clamp(MIN_FLY_SPEED, MAX_FLY_SPEED);
            }

            let mut move_dir = Vec3::ZERO;
            if !egui_input.keyboard {
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
                // Q/E move straight up/down in world space regardless of
                // pitch, rather than along the (possibly tilted) view
                // direction like WASD does — keeping a way to gain/lose
                // altitude on command even while looking level.
                if keys.pressed(KeyCode::KeyE) {
                    move_dir += Vec3::Y;
                }
                if keys.pressed(KeyCode::KeyQ) {
                    move_dir += Vec3::NEG_Y;
                }
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
        CameraMode::Rts => {
            if scroll != 0.0 {
                rig.orbit_radius = (rig.orbit_radius
                    * (1.0 - scroll * settings.orbit_zoom_sensitivity))
                    .max(settings.min_orbit_radius);
            }

            if !egui_input.keyboard {
                // Yaw-only, so panning stays on the ground plane regardless
                // of the current pitch — `rotation` (yaw + pitch) would tilt
                // "forward" up/down into the terrain or the sky.
                let yaw_only = Quat::from_rotation_y(rig.yaw);
                let mut pan_dir = Vec3::ZERO;
                if keys.pressed(KeyCode::KeyW) {
                    pan_dir += yaw_only * Vec3::NEG_Z;
                }
                if keys.pressed(KeyCode::KeyS) {
                    pan_dir += yaw_only * Vec3::Z;
                }
                if keys.pressed(KeyCode::KeyD) {
                    pan_dir += yaw_only * Vec3::X;
                }
                if keys.pressed(KeyCode::KeyA) {
                    pan_dir += yaw_only * Vec3::NEG_X;
                }

                if pan_dir != Vec3::ZERO {
                    let speed = settings.rts_pan_speed
                        * if keys.pressed(KeyCode::ShiftLeft) {
                            settings.sprint_multiplier
                        } else {
                            1.0
                        };
                    rig.orbit_target += pan_dir.normalize() * speed * time.delta_secs();
                }
            }

            transform.translation = rig.orbit_target + rotation * Vec3::new(0.0, 0.0, rig.orbit_radius);
        }
    }
}

/// Keeps the camera's far clip plane and distance fog matched to
/// [`streaming::RenderDistance`] whenever it changes at runtime (ticket
/// 007's status-panel slider) — without this, widening the render distance
/// would stream terrain in past the old, now-stale far plane, where it
/// would simply never be drawn.
fn sync_render_distance_effects(
    render_distance: Res<streaming::RenderDistance>,
    mut query: Query<(&mut Projection, &mut DistanceFog), With<CameraRig>>,
) {
    if !render_distance.is_changed() {
        return;
    }
    for (mut projection, mut fog) in &mut query {
        if let Projection::Perspective(perspective) = projection.as_mut() {
            perspective.far = far_plane_distance(render_distance.0);
        }
        // Falloff only — `fog.color` belongs to `SkyPalette` (ticket 015),
        // see `fog_falloff`'s docs.
        fog.falloff = fog_falloff(render_distance.0);
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
        .map(|(_voxel, hit)| hit)
        .unwrap_or_else(|| ray.origin + *ray.direction * settings.max_ray_distance);

    rig.orbit_radius = transform.translation.distance(target).max(settings.min_orbit_radius);
    rig.orbit_target = target;
    rig.mode = CameraMode::Orbit;
}

/// Minecraft block coordinates of the solid block the camera ray through
/// `cursor` (window pixel coordinates) first hits within `max_ray_distance`
/// — the block inspector panel's (ticket 007) "what's under the cursor",
/// sharing [`raycast_terrain`] with [`toggle_mode`]'s orbit re-aiming so
/// both agree on what block the cursor is over.
pub(crate) fn block_under_cursor(
    camera: &Camera,
    camera_transform: &GlobalTransform,
    cursor: Vec2,
    max_ray_distance: f32,
    decoded_world: &DecodedWorld,
) -> Option<IVec3> {
    let ray = camera.viewport_to_world(camera_transform, cursor).ok()?;
    raycast_terrain(ray.origin, *ray.direction, max_ray_distance, decoded_world)
        .map(|(voxel, _hit)| voxel)
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
/// Minecraft-space block coordinate of the first solid block hit, together
/// with the world-space entry point on its surface — or `None` if nothing
/// solid is within `max_distance`.
fn raycast_terrain(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    decoded_world: &DecodedWorld,
) -> Option<(IVec3, Vec3)> {
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

    // Locked once up front rather than per voxel step — background
    // chunk-load tasks (005-c) hold this lock only briefly (one chunk's
    // decode+mesh at a time), so a raycast spanning a few hundred voxels
    // holding it for its whole walk is a non-issue in practice.
    let registry = decoded_world
        .registry
        .lock()
        .expect("block registry mutex poisoned");

    let mut traveled = 0.0f32;
    loop {
        if world::is_solid(
            block_at_world(decoded_world, voxel.x, voxel.y, voxel.z),
            &registry,
        ) {
            let hit = mc_origin + mc_dir * traveled;
            return Some((voxel, Vec3::new(hit.x, hit.y, -hit.z)));
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
    use crate::world::biome::BiomeRegistry;
    use crate::world::block::BlockRegistry;
    use crate::world::decode::{ChunkColumn, ChunkSection, BIOME_GRID_VOLUME, SECTION_VOLUME};
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
        let biomes = Box::new([BiomeRegistry::PLAINS; BIOME_GRID_VOLUME]);

        let mut columns = HashMap::new();
        columns.insert(
            (cx, cz),
            ChunkColumn {
                x: cx,
                z: cz,
                sections: vec![ChunkSection { y: section_y, blocks, biomes }],
                floor_y: world::decode::WORLD_MIN_Y,
            },
        );
        DecodedWorld {
            registry: std::sync::Arc::new(std::sync::Mutex::new(registry)),
            biomes: std::sync::Arc::new(std::sync::Mutex::new(BiomeRegistry::new())),
            columns,
        }
    }

    #[test]
    fn raycast_hits_a_placed_block() {
        let decoded = single_block_world(5, 10, 5);
        // Marching along +mc.x (bevy.z = -mc.z, so bevy.z = -5.5 for mc.z =
        // 5.5) through the middle of the block's y/z span, starting well
        // outside it on the west side.
        let origin = Vec3::new(-10.0, 10.5, -5.5);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let (voxel, hit) =
            raycast_terrain(origin, direction, 100.0, &decoded).expect("should hit the block");
        assert_eq!(voxel, IVec3::new(5, 10, 5));
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

    /// Ticket 015: `sync_render_distance_effects` used to rebuild the whole
    /// `DistanceFog` (colour included) on every render-distance change,
    /// clobbering whatever `SkyPalette` had set — "the horizon went
    /// blue-grey again after I moved the slider". It should only ever touch
    /// `falloff`.
    #[test]
    fn render_distance_change_updates_falloff_but_leaves_fog_color_alone() {
        let mut app = App::new();
        app.add_systems(Update, sync_render_distance_effects);
        app.insert_resource(streaming::RenderDistance(4));

        let custom_color = Color::srgb(0.1, 0.2, 0.3);
        let camera = app
            .world_mut()
            .spawn((
                CameraRig::looking_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0)),
                Projection::Perspective(PerspectiveProjection::default()),
                atmosphere_fog(4, custom_color),
            ))
            .id();

        // First tick applies the just-inserted RenderDistance.
        app.update();
        app.world_mut().resource_mut::<streaming::RenderDistance>().0 = 8;
        app.update();

        let fog = app.world().get::<DistanceFog>(camera).unwrap();
        assert_eq!(fog.color, custom_color, "fog color must not follow RenderDistance");

        let far = far_plane_distance(8);
        match &fog.falloff {
            FogFalloff::Linear { start, end } => {
                assert_eq!(*start, far * 0.6);
                assert_eq!(*end, far);
            }
            other => panic!("expected linear falloff, got {other:?}"),
        }

        let projection = app.world().get::<Projection>(camera).unwrap();
        match projection {
            Projection::Perspective(perspective) => assert_eq!(perspective.far, far),
            other => panic!("expected a perspective projection, got {other:?}"),
        }
    }

    // --- Ticket 045, roadmap E1: `CameraMode::Rts` ---

    /// A [`DecodedWorld`] with nothing decoded in it — `drive_camera` needs
    /// the resource to exist to run at all, but none of `Rts`'s pan/rotate/
    /// zoom paths read it (only `Tab`'s orbit-reaim raycast does, and these
    /// tests never press `Tab` while able to reach that path).
    fn empty_world() -> DecodedWorld {
        DecodedWorld {
            registry: std::sync::Arc::new(std::sync::Mutex::new(world::BlockRegistry::new())),
            biomes: std::sync::Arc::new(std::sync::Mutex::new(BiomeRegistry::new())),
            columns: HashMap::new(),
        }
    }

    /// A bare `App` with `drive_camera` and every resource it reads, plus one
    /// spawned camera entity carrying `rig`. No `Window` entity — `Rts`'s
    /// pan/rotate/zoom never touch the cursor, so `windows.get_single_mut()`
    /// simply comes back `None`, the same as `sync_render_distance_effects`'s
    /// existing test not needing one either.
    fn rts_test_app(rig: CameraRig) -> (App, Entity) {
        let mut app = App::new();
        app.init_resource::<CameraSettings>()
            .init_resource::<EguiInputCapture>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<AccumulatedMouseScroll>()
            .init_resource::<Time>()
            .insert_resource(empty_world())
            .add_systems(Update, drive_camera);

        let entity = app
            .world_mut()
            .spawn((rig, Transform::default(), Camera::default(), GlobalTransform::default()))
            .id();
        (app, entity)
    }

    fn set_rig(app: &mut App, entity: Entity, f: impl FnOnce(&mut CameraRig)) {
        f(&mut app.world_mut().get_mut::<CameraRig>(entity).unwrap());
    }

    fn rig_of(app: &App, entity: Entity) -> &CameraRig {
        app.world().get::<CameraRig>(entity).unwrap()
    }

    #[test]
    fn looking_at_with_mode_sets_the_requested_mode() {
        let rig = CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);
        assert_eq!(rig.mode, CameraMode::Rts);
    }

    #[test]
    fn rts_w_pans_the_target_forward_on_the_yaw_relative_ground_plane() {
        let rig = CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);
        let (mut app, entity) = rts_test_app(rig);
        // Yaw = 0 faces -Z (see `world::mesh`'s bevy.z = -mc.z convention);
        // pinning it removes any ambiguity about which way "forward" is.
        set_rig(&mut app, entity, |r| {
            r.yaw = 0.0;
            r.pitch = -0.5;
            r.orbit_target = Vec3::ZERO;
        });

        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::KeyW);
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs_f32(1.0));
        app.update();

        let target = rig_of(&app, entity).orbit_target;
        assert!(target.z < 0.0, "W at yaw 0 should pan toward -Z, got {target}");
        assert_eq!(target.x, 0.0, "W at yaw 0 must not drift on X");
        assert_eq!(target.y, 0.0, "panning must not move the target vertically");
    }

    #[test]
    fn rts_shift_multiplies_pan_speed() {
        let new_rig = || CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);

        let (mut plain_app, plain_entity) = rts_test_app(new_rig());
        set_rig(&mut plain_app, plain_entity, |r| {
            r.yaw = 0.0;
            r.orbit_target = Vec3::ZERO;
        });
        plain_app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::KeyW);
        plain_app
            .world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs_f32(1.0));
        plain_app.update();
        let plain_distance = rig_of(&plain_app, plain_entity).orbit_target.distance(Vec3::ZERO);

        let (mut sprint_app, sprint_entity) = rts_test_app(new_rig());
        set_rig(&mut sprint_app, sprint_entity, |r| {
            r.yaw = 0.0;
            r.orbit_target = Vec3::ZERO;
        });
        sprint_app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::KeyW);
        sprint_app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::ShiftLeft);
        sprint_app
            .world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs_f32(1.0));
        sprint_app.update();
        let sprint_distance = rig_of(&sprint_app, sprint_entity).orbit_target.distance(Vec3::ZERO);

        let multiplier = plain_app.world().resource::<CameraSettings>().sprint_multiplier;
        assert!(
            (sprint_distance - plain_distance * multiplier).abs() < 1e-3,
            "expected {multiplier}x the plain pan distance, got {sprint_distance} vs {plain_distance}"
        );
    }

    #[test]
    fn rts_q_e_rotate_yaw_and_the_same_frame_reflects_it_in_translation() {
        let rig = CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);
        let (mut app, entity) = rts_test_app(rig);
        set_rig(&mut app, entity, |r| {
            r.yaw = 0.0;
            r.pitch = -0.5;
            r.orbit_target = Vec3::ZERO;
            r.orbit_radius = 10.0;
        });
        let rotate_speed = app.world().resource::<CameraSettings>().rts_rotate_speed;

        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::KeyE);
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_secs_f32(1.0));
        app.update();

        let rig = rig_of(&app, entity);
        assert!((rig.yaw - (-rotate_speed)).abs() < 1e-4, "E should rotate yaw by -rotate_speed");

        // Proves the ordering fix in `drive_camera`: if `Q`/`E` were applied
        // after `rotation` is computed, `transform.translation` would still
        // reflect *last* frame's yaw here.
        let expected_rotation = Quat::from_euler(EulerRot::YXZ, rig.yaw, rig.pitch, 0.0);
        let expected_translation = rig.orbit_target + expected_rotation * Vec3::new(0.0, 0.0, rig.orbit_radius);
        let translation = app.world().get::<Transform>(entity).unwrap().translation;
        assert!(
            translation.distance(expected_translation) < 1e-3,
            "translation lagged a frame behind the yaw it was rotated to: expected {expected_translation}, got {translation}"
        );
    }

    #[test]
    fn rts_scroll_zooms_and_clamps_at_min_orbit_radius() {
        let rig = CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);
        let (mut app, entity) = rts_test_app(rig);
        set_rig(&mut app, entity, |r| r.orbit_radius = 10.0);

        app.world_mut().resource_mut::<AccumulatedMouseScroll>().delta.y = 1.0;
        app.update();
        assert!(rig_of(&app, entity).orbit_radius < 10.0, "scrolling in should shrink the radius");

        app.world_mut().resource_mut::<AccumulatedMouseScroll>().delta.y = 1000.0;
        app.update();
        let min = app.world().resource::<CameraSettings>().min_orbit_radius;
        assert_eq!(rig_of(&app, entity).orbit_radius, min, "must clamp at min_orbit_radius, not overshoot to 0 or negative");
    }

    #[test]
    fn rts_pitch_clamps_to_its_own_range_not_flys() {
        let rig = CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);
        let (mut app, entity) = rts_test_app(rig);
        set_rig(&mut app, entity, |r| r.pitch = 0.0);

        let (rts_min, fly_min) = {
            let settings = app.world().resource::<CameraSettings>();
            (settings.rts_pitch_range.start, settings.pitch_range.start)
        };
        assert!(rts_min > fly_min, "test requires the two ranges to actually differ");

        app.world_mut().resource_mut::<ButtonInput<MouseButton>>().press(MouseButton::Right);
        app.world_mut().resource_mut::<AccumulatedMouseMotion>().delta = Vec2::new(0.0, 1_000_000.0);
        app.update();

        let pitch = rig_of(&app, entity).pitch;
        assert!(
            (pitch - rts_min).abs() < 1e-4,
            "expected the drag to clamp at rts_pitch_range's bound ({rts_min}), got {pitch} (fly's bound is {fly_min})"
        );
    }

    #[test]
    fn tab_is_a_no_op_in_rts_mode() {
        let rig = CameraRig::looking_at(Vec3::new(0.0, 10.0, 10.0), Vec3::ZERO).with_mode(CameraMode::Rts);
        let (mut app, entity) = rts_test_app(rig);

        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::Tab);
        app.update();

        assert_eq!(rig_of(&app, entity).mode, CameraMode::Rts);
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
