use bevy::app::{Plugin, Startup, Update};
use bevy::core::Name;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::input::ButtonInput;
use bevy::math::{EulerRot, Quat, Vec3};
use bevy::pbr::DirectionalLight;
use bevy::prelude::{Camera, Camera3d, Commands, KeyCode, MouseButton, Res, ResMut, Resource, Single, Time, Transform, With};
use std::f32::consts::FRAC_PI_2;
use std::ops::Range;
use bevy::input::keyboard::KeyboardInput;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, init_camera)
            .init_resource::<CameraSettings>()
            .add_systems(Update, orbit);
    }
}

#[derive(Debug, Resource)]
struct CameraSettings {
    //pub orbit_distance: f32,
    pub pitch_speed: f32,
    // Clamp pitch to this range
    pub pitch_range: Range<f32>,
    pub roll_speed: f32,
    pub yaw_speed: f32,
    pub target: Vec3,
}

impl Default for CameraSettings {
    fn default() -> Self {
        // Limiting pitch stops some unexpected rotation past 90° up or down.
        let pitch_limit = FRAC_PI_2 - 0.01;
        Self {
            // These values are completely arbitrary, chosen because they seem to produce
            // "sensible" results for this example. Adjust as required.
            //orbit_distance: 20.0,
            pitch_speed: 0.003,
            pitch_range: -pitch_limit..pitch_limit,
            roll_speed: 1.0,
            yaw_speed: 0.004,
            target: Vec3::new(0.0, 150.0, 0.0),
        }
    }
}

pub(crate) fn init_camera(mut commands: Commands) {
    // Transform for the camera and lighting, looking at (0,0,0) (the position of the mesh).
    let camera_and_light_transform =
        Transform::from_xyz(100.8, 400.0, 100.8).looking_at(Vec3::ZERO, Vec3::Y);

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Transform::from_xyz(50.0, 200.0, 50.0).looking_at(Vec3::new(0.0, 200.0, 0.0), Vec3::Y),
    ));

    // Light up the scene.
    commands.spawn((DirectionalLight::default(), camera_and_light_transform));
}

fn orbit(
    mut camera: Single<&mut Transform, With<Camera>>,
    mut camera_settings: ResMut<CameraSettings>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let delta = mouse_motion.delta;
    let mut delta_roll = 0.0;

    let distance = (camera.translation - camera_settings.target).length();

    if mouse_buttons.pressed(MouseButton::Left) {
        delta_roll -= 1.0;
    }
    if mouse_buttons.pressed(MouseButton::Right) {
        delta_roll += 1.0;
    }

    if mouse_buttons.pressed(MouseButton::Middle) {
        // Mouse motion is one of the few inputs that should not be multiplied by delta time,
        // as we are already receiving the full movement since the last frame was rendered. Multiplying
        // by delta time here would make the movement slower that it should be.
        let delta_pitch = -delta.y * camera_settings.pitch_speed;
        let delta_yaw = -delta.x * camera_settings.yaw_speed;

        // Conversely, we DO need to factor in delta time for mouse button inputs.
        delta_roll *= camera_settings.roll_speed * time.delta_secs();

        // Obtain the existing pitch, yaw, and roll values from the transform.
        let (yaw, pitch, roll) = camera.rotation.to_euler(EulerRot::YXZ);

        // Establish the new yaw and pitch, preventing the pitch value from exceeding our limits.
        let pitch = (pitch + delta_pitch).clamp(
            camera_settings.pitch_range.start,
            camera_settings.pitch_range.end,
        );
        let roll = roll + delta_roll;
        let yaw = yaw + delta_yaw;
        camera.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll);
    }

    if keys.pressed(KeyCode::KeyA) {
        camera_settings.target += camera.left() * time.delta_secs() * distance;
    }
    if keys.pressed(KeyCode::KeyD) {
        camera_settings.target -= camera.left() * time.delta_secs() * distance;
    }
    if keys.pressed(KeyCode::KeyW) {
        camera_settings.target += camera.forward() * time.delta_secs() * distance;
    }
    if keys.pressed(KeyCode::KeyS) {
        camera_settings.target += camera.back() * time.delta_secs() * distance;
    }

    // Adjust the translation to maintain the correct orientation toward the orbit target.
    // In our example it's a static target, but this could easily be customized.
    camera.translation = camera_settings.target - camera.forward() * distance * (1.0 - mouse_scroll.delta.y * 0.1);
}
