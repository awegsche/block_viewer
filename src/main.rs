use bevy::prelude::*;
use mc_anvil::{chunkregion::ChunkRegion, Save, get_saves};
use std::{collections::HashMap, path::Path, time::Instant};

mod camera;
mod streaming;
mod world;

/// The currently loaded Minecraft save, populated at startup from a real
/// save directory (`%AppData%\.minecraft\saves` on Windows).
#[derive(Resource)]
struct LoadedSave(Save);

/// Every chunk column decoded from the loaded save's first region (ticket
/// 002), keyed by world chunk coordinates, plus the [`world::BlockRegistry`]
/// their block names were interned into. Consumed by [`setup`] to mesh one
/// entity per chunk column (ticket 003), and by [`camera`]'s
/// under-the-cursor ray-march for orbit targeting (ticket 006).
#[derive(Resource)]
pub(crate) struct DecodedWorld {
    pub(crate) registry: world::BlockRegistry,
    pub(crate) columns: HashMap<(i32, i32), world::ChunkColumn>,
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
        .add_plugins(camera::CameraControllerPlugin)
        .add_plugins(streaming::ChunkStreamingPlugin)
        .insert_resource(LoadedSave(save))
        .insert_resource(decoded_world)
        .add_systems(Startup, setup)
        .run();
}

#[derive(Component)]
struct BlockMesh;

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    loaded_save: Res<LoadedSave>,
    decoded_world: Res<DecodedWorld>,
) {
    println!(
        "Active save: {} ({} regions)",
        loaded_save.0.meta.name,
        loaded_save.0.meta.regions.len()
    );

    // Pack every block texture into one atlas and resolve each interned
    // block name to its per-face atlas rect (ticket 004) — replaces the
    // single hardcoded stone.png the mesher previously stretched over
    // every face.
    let atlas = world::atlas::build(Path::new("assets/minecraft/textures/block"))
        .expect("failed to build the block texture atlas");
    let uv_table = world::atlas::build_block_uv_table(&decoded_world.registry, &atlas);
    let atlas_handle = images.add(atlas.image);
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(atlas_handle),
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
        let Some(mesh) =
            world::mesh_chunk_column(column, &decoded_world.registry, &neighbors, &uv_table)
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

    // Place the camera above the terrain surface near the middle of the
    // loaded columns (ticket 006) instead of a fixed point like the old
    // `(10, 5, 10)`, which on a real save may well be underground.
    let target = spawn_point(&decoded_world.columns);
    let eye = target + Vec3::new(-24.0, 20.0, 24.0);

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: camera::far_plane_distance(),
            ..default()
        }),
        camera::atmosphere_fog(),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        camera::CameraRig::looking_at(eye, target),
    ));

    // Light up the scene.
    commands.spawn((
        PointLight::default(),
        Transform::from_xyz(target.x + 10.8, target.y + 30.0, target.z + 10.8)
            .looking_at(target, Vec3::Y),
    ));
}

/// Bevy-space point on (or just above) the terrain surface near the middle
/// of the loaded columns — used to place the camera somewhere sensible at
/// startup and as its initial orbit target. Real saves are rarely centred
/// on (0,0), and a fixed height would just as easily land underground.
fn spawn_point(columns: &HashMap<(i32, i32), world::ChunkColumn>) -> Vec3 {
    let Some((&(cx, cz), column)) = nearest_to_average(columns) else {
        return Vec3::new(0.0, 80.0, 0.0);
    };

    let size = world::SECTION_SIZE as i32;
    let local = world::SECTION_SIZE / 2;
    // Fall back to a plausible sea-level-ish height if the centre column
    // happens to be a void (e.g. an unloaded/void chunk in a partial save).
    let world_y = column.topmost_non_air(local, local).map_or(72, |(y, _)| y);

    Vec3::new(
        (cx * size + local as i32) as f32,
        world_y as f32 + 2.0, // stand a couple of blocks above the surface
        -(cz * size + local as i32) as f32,
    )
}

/// The loaded column closest to the horizontal centroid of every loaded
/// column — the centroid itself may land on a gap (an unloaded or void
/// chunk), so this snaps to whatever's actually there.
fn nearest_to_average(
    columns: &HashMap<(i32, i32), world::ChunkColumn>,
) -> Option<(&(i32, i32), &world::ChunkColumn)> {
    if columns.is_empty() {
        return None;
    }
    let n = columns.len() as f64;
    let (sum_x, sum_z) = columns.keys().fold((0i64, 0i64), |(sx, sz), &(cx, cz)| {
        (sx + cx as i64, sz + cz as i64)
    });
    let avg_cx = sum_x as f64 / n;
    let avg_cz = sum_z as f64 / n;

    columns.iter().min_by(|a, b| {
        let dist2 =
            |&(cx, cz): &(i32, i32)| (cx as f64 - avg_cx).powi(2) + (cz as f64 - avg_cz).powi(2);
        dist2(a.0).total_cmp(&dist2(b.0))
    })
}
