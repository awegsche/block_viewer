use bevy::{
    image::{ImageLoaderSettings, ImageSampler},
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
};
use mc_anvil::{chunkregion::ChunkRegion, Save, get_saves};
use std::{collections::HashMap, f32::consts::FRAC_PI_2, ops::Range, time::Instant};

mod world;

/// The currently loaded Minecraft save, populated at startup from a real
/// save directory (`%AppData%\.minecraft\saves` on Windows).
#[derive(Resource)]
struct LoadedSave(Save);

/// Every chunk column decoded from the loaded save's first region (ticket
/// 002), keyed by world chunk coordinates, plus the [`world::BlockRegistry`]
/// their block names were interned into. Consumed by [`setup`] to mesh one
/// entity per chunk column (ticket 003).
#[derive(Resource)]
struct DecodedWorld {
    registry: world::BlockRegistry,
    columns: HashMap<(i32, i32), world::ChunkColumn>,
}

/// Loads the first save found under the Minecraft saves directory
/// (`dirs::config_dir()/.minecraft/saves`, i.e.
/// `C:\Users\<user>\AppData\Roaming\.minecraft\saves` on Windows) and eagerly
/// parses the chunks of its first region so we know real save data is
/// reachable.
fn load_real_save() -> Save {
    let saves = get_saves().expect("could not read the Minecraft saves directory");
    let meta = saves
        .into_iter()
        .next()
        .expect("no Minecraft saves found in the saves directory");

    println!("Loading save {}", meta.get_grid_view());

    let mut save: Save = meta.into();
    if let Some(first_region) = save.regions.first_mut() {
        first_region
            .load_chunks()
            .expect("failed to load chunks for the first region");
        let chunk_count = first_region
            .chunks
            .as_ref()
            .map(|chunks| chunks.iter().filter(|c| c.is_some()).count())
            .unwrap_or(0);
        println!(
            "Loaded {} chunks from region ({}, {})",
            chunk_count,
            first_region.region.get_x_coord(),
            first_region.region.get_z_coord()
        );
    }

    save
}

/// Decodes every populated, fully-generated chunk in `region` into a
/// [`world::ChunkColumn`], sharing one [`world::BlockRegistry`] across all
/// of them so [`world::BlockId`]s stay comparable.
fn decode_region(region: &ChunkRegion) -> DecodedWorld {
    let mut registry = world::BlockRegistry::new();
    let mut columns = HashMap::new();

    let Some(chunks) = &region.chunks else {
        return DecodedWorld { registry, columns };
    };

    let start = Instant::now();
    let mut skipped = 0usize;
    for chunk in chunks.iter().flatten() {
        match world::decode_chunk(chunk, &mut registry) {
            Ok(column) => {
                columns.insert((column.x, column.z), column);
            }
            Err(_) => skipped += 1,
        }
    }
    println!(
        "Decoded {} chunk columns ({} skipped) from region ({}, {}) in {:?}",
        columns.len(),
        skipped,
        region.region.get_x_coord(),
        region.region.get_z_coord(),
        start.elapsed()
    );

    DecodedWorld { registry, columns }
}

fn main() {
    let save = load_real_save();
    let decoded_world = save
        .regions
        .first()
        .map(decode_region)
        .unwrap_or_else(|| DecodedWorld {
            registry: world::BlockRegistry::new(),
            columns: HashMap::new(),
        });

    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<CameraSettings>()
        .insert_resource(LoadedSave(save))
        .insert_resource(decoded_world)
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
    loaded_save: Res<LoadedSave>,
    decoded_world: Res<DecodedWorld>,
) {
    println!(
        "Active save: {} ({} regions)",
        loaded_save.0.meta.name,
        loaded_save.0.meta.regions.len()
    );

    let block_texture_handle: Handle<Image> = asset_server.load_with_settings(
        "minecraft/textures/block/stone.png",
        |settings: &mut ImageLoaderSettings| {
            // Need to use nearest filtering to avoid bleeding between the slices with tiling
            settings.sampler = ImageSampler::nearest();
        },
    );
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(block_texture_handle),
        ..default()
    });

    // One entity per chunk column (ticket 003) — never one mesh for a whole
    // region, that would be a single giant draw call with no culling.
    let mut spawned = 0usize;
    for (&(cx, cz), column) in &decoded_world.columns {
        let neighbors = world::Neighbors {
            north: decoded_world.columns.get(&(cx, cz - 1)),
            south: decoded_world.columns.get(&(cx, cz + 1)),
            east: decoded_world.columns.get(&(cx + 1, cz)),
            west: decoded_world.columns.get(&(cx - 1, cz)),
        };
        let Some(mesh) = world::mesh_chunk_column(column, &decoded_world.registry, &neighbors)
        else {
            continue; // fully-air column: nothing to render
        };

        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(material_handle.clone()),
            // Chunk mesh vertices are chunk-local; place the entity at the
            // chunk's world origin under this module's axis mapping
            // (bevy.x = mc.x, bevy.z = -mc.z — see `world::mesh` docs).
            Transform::from_xyz(cx as f32 * world::SECTION_SIZE as f32, 0.0, -(cz as f32 * world::SECTION_SIZE as f32)),
            BlockMesh,
        ));
        spawned += 1;
    }
    println!(
        "Spawned {spawned} chunk mesh entities ({} columns decoded)",
        decoded_world.columns.len()
    );

    // Aim the (still-fixed, ticket 006 will make this real navigation)
    // camera at the middle of the loaded terrain instead of world origin —
    // real saves are rarely centred on (0,0).
    let target = column_center(&decoded_world.columns);
    let camera_and_light_transform =
        Transform::from_xyz(target.x + 10.8, target.y + 30.0, target.z + 10.8)
            .looking_at(target, Vec3::Y);

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Transform::from_xyz(target.x + 40.0, target.y + 60.0, target.z + 40.0)
            .looking_at(target, Vec3::Y),
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

/// World-space (Bevy axes, see `world::mesh` docs) centre of the horizontal
/// span of the decoded columns, at Y=0 — a reasonable point for the
/// placeholder camera to orbit until ticket 006 replaces it with real
/// navigation.
fn column_center(columns: &HashMap<(i32, i32), world::ChunkColumn>) -> Vec3 {
    if columns.is_empty() {
        return Vec3::ZERO;
    }
    let n = columns.len() as f32;
    let (sum_x, sum_z) = columns.keys().fold((0i64, 0i64), |(sx, sz), &(cx, cz)| {
        (sx + cx as i64, sz + cz as i64)
    });
    let avg_cx = sum_x as f32 / n;
    let avg_cz = sum_z as f32 / n;
    let size = world::SECTION_SIZE as f32;
    // Chunk-centre in Minecraft blocks, then through the same x/-z mapping
    // as chunk-entity transforms.
    Vec3::new(
        avg_cx * size + size / 2.0,
        0.0,
        -(avg_cz * size + size / 2.0),
    )
}
