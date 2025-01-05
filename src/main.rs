use bevy::{
    asset::RenderAssetUsages, image::{ImageLoaderSettings, ImageSampler}, input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll}, prelude::*, render::mesh::Indices
};
use mc_anvil::{Save, get_saves};
use std::{f32::consts::FRAC_PI_2, ops::Range};

fn main() {
    println!("Hello, world!");

    let mut saves = get_saves().unwrap();

    for save in saves.iter() {
        println!("{:?}", save);
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<CameraSettings>()
        .add_systems(Startup, setup)
        .add_systems(Update, orbit)
        .run();
}

#[derive(Component)]
struct BlockMesh;

#[derive(Debug, Resource)]
struct CameraSettings {
    pub orbit_distance: f32,
    pub pitch_speed: f32,
    // Clamp pitch to this range
    pub pitch_range: Range<f32>,
    pub roll_speed: f32,
    pub yaw_speed: f32,
    pub is_roatatin: bool,
}

impl Default for CameraSettings {
    fn default() -> Self {
        // Limiting pitch stops some unexpected rotation past 90° up or down.
        let pitch_limit = FRAC_PI_2 - 0.01;
        Self {
            // These values are completely arbitrary, chosen because they seem to produce
            // "sensible" results for this example. Adjust as required.
            orbit_distance: 20.0,
            pitch_speed: 0.003,
            pitch_range: -pitch_limit..pitch_limit,
            roll_speed: 1.0,
            yaw_speed: 0.004,
            is_roatatin: false,
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
        Transform::from_xyz(10.8, 10.8, 10.8).looking_at(Vec3::ZERO, Vec3::Y);

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Transform::from_xyz(10.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light up the scene.
    commands.spawn((PointLight::default(), camera_and_light_transform));
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

    let target = Vec3::ZERO;
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

fn get_bottom_tris(x: i32, y: i32, z: i32) -> [[f32; 3]; 4] {
    [
        [x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5], // vertex with index 0
        [x as f32 + 0.5, y as f32 - 0.5, z as f32 - 0.5], // vertex with index 1
        [x as f32 + 0.5, y as f32 - 0.5, z as f32 + 0.5], // etc. until 23
        [x as f32 - 0.5, y as f32 - 0.5, z as f32 + 0.5],
    ]
}

fn get_top_tris(x: i32, y: i32, z: i32) -> [[f32; 3]; 4] {
    [
        [x as f32 - 0.5, y as f32 + 0.5, z as f32 - 0.5], // vertex with index 0
        [x as f32 + 0.5, y as f32 + 0.5, z as f32 - 0.5], // vertex with index 1
        [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5], // etc. until 23
        [x as f32 - 0.5, y as f32 + 0.5, z as f32 + 0.5],
    ]
}

fn get_left_tris(x: i32, y: i32, z: i32) -> [[f32; 3]; 4] {
    [
        [x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5],
        [x as f32 - 0.5, y as f32 - 0.5, z as f32 + 0.5],
        [x as f32 - 0.5, y as f32 + 0.5, z as f32 + 0.5],
        [x as f32 - 0.5, y as f32 + 0.5, z as f32 - 0.5],
    ]
}

fn get_right_tris(x: i32, y: i32, z: i32) -> [[f32; 3]; 4] {
    [
        [x as f32 + 0.5, y as f32 - 0.5, z as f32 - 0.5],
        [x as f32 + 0.5, y as f32 - 0.5, z as f32 + 0.5],
        [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5], // This vertex is at the same position as vertex with index 2, but they'll have different UV and normal
        [x as f32 + 0.5, y as f32 + 0.5, z as f32 - 0.5],
    ]
}

fn get_front_tris(x: i32, y: i32, z: i32) -> [[f32; 3]; 4] {
    [
        [x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5],
        [x as f32 - 0.5, y as f32 + 0.5, z as f32 - 0.5],
        [x as f32 + 0.5, y as f32 + 0.5, z as f32 - 0.5],
        [x as f32 + 0.5, y as f32 - 0.5, z as f32 - 0.5],
    ]
}

fn get_back_tris(x: i32, y: i32, z: i32) -> [[f32; 3]; 4] {
    [
        [x as f32 - 0.5, y as f32 - 0.5, z as f32 + 0.5],
        [x as f32 - 0.5, y as f32 + 0.5, z as f32 + 0.5],
        [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5],
        [x as f32 + 0.5, y as f32 - 0.5, z as f32 + 0.5],
    ]
}

fn create_block_mesh() -> Mesh {
    // let's test our edge finding algorithm.
    //
    const WIDTH: i32 = 8;
    const DEPTH: i32 = 8;
    const HEIGHT: i32 = 8;
    let mut blocks = vec![0; (WIDTH * DEPTH * HEIGHT) as usize];

    blocks[coord_to_idx(1, 0, 1, WIDTH, DEPTH)] = 1;
    blocks[coord_to_idx(1, 1, 1, WIDTH, DEPTH)] = 1;
    blocks[coord_to_idx(1, 2, 1, WIDTH, DEPTH)] = 1;
    blocks[coord_to_idx(0, 2, 1, WIDTH, DEPTH)] = 1;
    blocks[coord_to_idx(2, 2, 1, WIDTH, DEPTH)] = 1;
    blocks[coord_to_idx(1, 3, 1, WIDTH, DEPTH)] = 1;

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for x in 0..WIDTH {
        for z in 0..DEPTH {
            for y in 0..HEIGHT {
                if blocks[coord_to_idx(x, y, z, WIDTH, DEPTH)] == 0 {
                    continue;
                }

                if x == 0 || blocks[coord_to_idx(x - 1, y, z, WIDTH, DEPTH)] == 0 {
                    let n = vertices.len() as u32;
                    indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    vertices.extend_from_slice(&get_left_tris(x, y, z));
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [-1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                    ]);
                }
                if x == WIDTH - 1 || blocks[coord_to_idx(x + 1, y, z, WIDTH, DEPTH)] == 0 {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_right_tris(x, y, z));
                    indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                    //indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]);
                }
                if z == 0 || blocks[coord_to_idx(x, y, z - 1, WIDTH, DEPTH)] == 0 {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_front_tris(x, y, z));
                    indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ]);
                }
                if z == DEPTH - 1 || blocks[coord_to_idx(x, y, z + 1, WIDTH, DEPTH)] == 0 {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_back_tris(x, y, z));
                    indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ]);
                }
                if y == 0 || blocks[coord_to_idx(x, y - 1, z, WIDTH, DEPTH)] == 0 {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_bottom_tris(x, y, z));
                    indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0],
                    ]);
                }
                if y == HEIGHT - 1 || blocks[coord_to_idx(x, y + 1, z, WIDTH, DEPTH)] == 0 {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_top_tris(x, y, z));
                    indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]);
                }
            }
        }
    }

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}
