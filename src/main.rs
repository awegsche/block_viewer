use bevy::prelude::*;
use mc_anvil::{get_saves, region::REGION_WIDTH_IN_CHUNKS, Save, SaveMeta};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

mod camera;
mod chunk_pipeline;
mod region_cache;
mod streaming;
mod unload;
mod world;

/// The currently loaded Minecraft save, populated at startup from a real
/// save directory (`%AppData%\.minecraft\saves` on Windows).
#[derive(Resource)]
struct LoadedSave(Save);

/// Every chunk column decoded so far (ticket 002), keyed by world chunk
/// coordinates, plus the [`world::BlockRegistry`] their block names were
/// interned into. Starts empty at startup (ticket 005-e deleted the old
/// eager pre-`App::run()` load) and fills in as [`chunk_pipeline`]'s
/// background chunk-load tasks (ticket 005-c) stream columns in around the
/// camera; also read by [`camera`]'s under-the-cursor ray-march for orbit
/// targeting (ticket 006).
///
/// `registry` is behind an `Arc<Mutex<_>>` — not just an owned
/// [`world::BlockRegistry`] — because those background tasks intern newly
/// decoded chunks' block names directly into it, off the main thread, and
/// [`world::BlockId`]s need to stay globally stable regardless of whether a
/// chunk was decoded eagerly at startup or streamed in later.
#[derive(Resource)]
pub(crate) struct DecodedWorld {
    pub(crate) registry: Arc<Mutex<world::BlockRegistry>>,
    pub(crate) columns: HashMap<(i32, i32), world::ChunkColumn>,
}

/// Picks the first save found under the Minecraft saves directory
/// (`dirs::config_dir()/.minecraft/saves`, i.e.
/// `C:\Users\<user>\AppData\Roaming\.minecraft\saves` on Windows). Metadata
/// only (`get_saves`/`SaveMeta` -> `Save`) — cheap and synchronous, unlike
/// chunk data, which streams in after `App::run()` via the async pipeline
/// (ticket 005-c) instead of being loaded here (ticket 005-e removed the old
/// eager pre-`App::run()` region load).
fn load_real_save() -> Save {
    let saves = get_saves().expect("could not read the Minecraft saves directory");
    let meta = saves
        .into_iter()
        .next()
        .expect("no Minecraft saves found in the saves directory");

    println!("Loading save {}", meta.get_grid_view());

    meta.into()
}

fn main() {
    let save = load_real_save();
    let decoded_world = DecodedWorld {
        registry: Arc::new(Mutex::new(world::BlockRegistry::new())),
        columns: HashMap::new(),
    };

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(camera::CameraControllerPlugin)
        .add_plugins(streaming::ChunkStreamingPlugin)
        .add_plugins(chunk_pipeline::ChunkLoadPipelinePlugin)
        .add_plugins(unload::ChunkUnloadPlugin)
        .insert_resource(LoadedSave(save))
        .insert_resource(decoded_world)
        .add_systems(Startup, setup)
        .run();
}

/// Marker on every spawned chunk mesh entity — `pub(crate)` so
/// [`chunk_pipeline`]'s polling system can tag entities it spawns the same
/// way [`setup`]'s eager spawn does.
#[derive(Component)]
pub(crate) struct BlockMesh;

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    loaded_save: Res<LoadedSave>,
    render_distance: Res<streaming::RenderDistance>,
) {
    println!(
        "Active save: {} ({} regions)",
        loaded_save.0.meta.name,
        loaded_save.0.meta.regions.len()
    );

    // Pack every block texture into one atlas (ticket 004) — replaces the
    // single hardcoded stone.png the mesher previously stretched over every
    // face. Per-block-id UV resolution (`build_block_uv_table`) happens
    // per-chunk inside `chunk_pipeline`'s background tasks instead of once
    // here, against whatever names are interned into the registry *at the
    // moment that chunk decodes* — block names get interned over the app's
    // whole lifetime as streaming loads new chunks (ticket 005-e), not just
    // once at startup like the old eager decode, so there's no fixed set of
    // names to build a table from up front.
    let atlas = world::atlas::build(Path::new("assets/minecraft/textures/block"))
        .expect("failed to build the block texture atlas");
    let uv_index = atlas.uv_index();
    let atlas_handle = images.add(atlas.image);
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(atlas_handle),
        ..default()
    });

    // Nothing is decoded yet — chunks stream in via the async pipeline
    // (ticket 005-c) as the camera moves, driven by these three resources
    // plus `DecodedWorld` (already inserted in `main()`). Ticket 005-e
    // deleted the old eager single-region load and per-column spawn loop
    // that used to populate the world here.
    let region_cache = region_cache::RegionCache::new(
        loaded_save.0.meta.clone(),
        region_cache::recommended_capacity(render_distance.0),
    );
    commands.insert_resource(chunk_pipeline::SharedRegionCache(Arc::new(Mutex::new(
        region_cache,
    ))));
    commands.insert_resource(chunk_pipeline::SharedAtlasIndex(Arc::new(uv_index)));
    commands.insert_resource(chunk_pipeline::TerrainMaterial(material_handle));

    // Place the camera near the middle of the save's region footprint
    // (ticket 005-e) instead of reading real terrain height like the old
    // eager-decode version did — nothing is decoded yet at startup to read
    // a height from. The camera free-flies at a fixed, generally-safe
    // height while terrain streams in underneath, per the ticket's
    // preference over deferring camera spawn until the first chunk loads.
    let target = spawn_point(&loaded_save.0.meta);
    let eye = target + Vec3::new(-24.0, 20.0, 24.0);

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: camera::far_plane_distance(render_distance.0),
            ..default()
        }),
        camera::atmosphere_fog(render_distance.0),
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

/// Bevy-space point used to place the camera at startup: the horizontal
/// middle of the save's region footprint (metadata only — `SaveMeta`'s
/// region-coordinate list, no chunk I/O — so this is safe to call before any
/// streaming has happened), at a fixed height generally above ground level.
/// Real saves are rarely centred on (0,0); landing the camera near where
/// regions actually exist means streaming has something to load in view
/// immediately, rather than the camera free-flying over empty space until
/// it happens to reach one.
fn spawn_point(meta: &SaveMeta) -> Vec3 {
    const DEFAULT_HEIGHT: f32 = 100.0;

    let Some((rx, rz)) = region_centroid(&meta.regions) else {
        return Vec3::new(0.0, DEFAULT_HEIGHT, 0.0);
    };

    let region_size = REGION_WIDTH_IN_CHUNKS as i32 * world::SECTION_SIZE as i32;
    let mc_x = rx * region_size + region_size / 2;
    let mc_z = rz * region_size + region_size / 2;

    // bevy.x = mc.x, bevy.z = -mc.z — see `world::mesh` docs.
    Vec3::new(mc_x as f32, DEFAULT_HEIGHT, -(mc_z as f32))
}

/// The save's region closest to the horizontal centroid of every region it
/// has — the centroid itself may land on a coordinate that isn't actually a
/// region (a save's regions needn't form a filled rectangle), so this snaps
/// to whatever region is actually there.
fn region_centroid(regions: &[(i32, i32)]) -> Option<(i32, i32)> {
    if regions.is_empty() {
        return None;
    }
    let n = regions.len() as f64;
    let (sum_x, sum_z) = regions.iter().fold((0i64, 0i64), |(sx, sz), &(rx, rz)| {
        (sx + rx as i64, sz + rz as i64)
    });
    let avg_x = sum_x as f64 / n;
    let avg_z = sum_z as f64 / n;

    regions.iter().copied().min_by(|a, b| {
        let dist2 =
            |&(rx, rz): &(i32, i32)| (rx as f64 - avg_x).powi(2) + (rz as f64 - avg_z).powi(2);
        dist2(a).total_cmp(&dist2(b))
    })
}
