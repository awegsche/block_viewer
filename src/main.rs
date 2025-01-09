use bevy::{
    asset::RenderAssetUsages,
    image::{ImageLoaderSettings, ImageSampler},
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    render::mesh::Indices,
};
use chunks::{check_block_texture_loading, create_block_mesh, load_block_textures, Chunk, RenderBlock};
use ranvil::{get_saves, Save};
use rnbt::{from_bytes, NbtField, NbtList, NbtValue};
use std::{f32::consts::FRAC_PI_2, ops::Range};

mod chunks;

#[derive(Debug, States, Default, Clone, Eq, PartialEq, Hash)]
enum AppState {
    #[default]
    LoadingTextures,
    Running,
}

#[derive(Component)]
struct StateText;

fn main() {
    println!("Hello, world!");

    let mut saves = get_saves().unwrap();

    for save in saves.iter() {
        println!("{}", save);
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .init_state::<AppState>()
        .init_resource::<CameraSettings>()
        .add_systems(OnEnter(AppState::LoadingTextures), load_block_textures)
        .add_systems(Update, check_block_texture_loading.run_if(in_state(AppState::LoadingTextures)))
        .add_systems(Startup, setup)
        //.add_systems(OnEnter(AppState::Running), setup)
        .add_systems(Update, orbit)
        .run();
}

#[derive(Component)]
struct BlockMesh;

#[derive(Debug, Resource)]
struct CameraSettings {
    //pub orbit_distance: f32,
    pub pitch_speed: f32,
    // Clamp pitch to this range
    pub pitch_range: Range<f32>,
    pub roll_speed: f32,
    pub yaw_speed: f32,
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
        }
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let block_texture_handle: Handle<Image> = asset_server.load_with_settings(
        "minecraft/textures/block/stone.png",
        |settings: &mut ImageLoaderSettings| {
            // Need to use nearest filtering to avoid bleeding between the slices with tiling
            settings.sampler = ImageSampler::nearest();
        },
    );
    let mesh_handle: Handle<Mesh> = meshes.add(create_block_mesh());

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(block_texture_handle),
            ..default()
        })),
        BlockMesh,
    ));

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

    commands.spawn((
            Text::new("Startup"),
            TextFont::default(),
            StateText,
    ));
}

fn orbit(
    mut camera: Single<&mut Transform, With<Camera>>,
    camera_settings: Res<CameraSettings>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
) {
    let delta = mouse_motion.delta;
    let mut delta_roll = 0.0;

    let target = Vec3::new(0.0, 140.0, 0.0);
    let distance = (camera.translation - target).length();

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

    // Adjust the translation to maintain the correct orientation toward the orbit target.
    // In our example it's a static target, but this could easily be customized.
    camera.translation = target - camera.forward() * distance * (1.0 - mouse_scroll.delta.y * 0.1);
}

fn coord_to_idx(x: i32, y: i32, z: i32, width: i32, depth: i32) -> usize {
    (x + z * width + y * width * depth) as usize
}

pub enum Side {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
    None,
}
