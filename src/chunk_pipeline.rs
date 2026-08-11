//! Async load/decode/mesh pipeline (ticket 005-c): turns a `to_load` chunk
//! coordinate from [`PendingChunkWork`](crate::streaming::PendingChunkWork)
//! into a spawned mesh entity via [`AsyncComputeTaskPool`], off the main
//! thread except for the final mesh upload / entity spawn.
//!
//! ## Send boundary
//!
//! `mc_anvil`'s [`ChunkRegion`](mc_anvil::chunkregion::ChunkRegion) and
//! `rnbt`'s `NbtField` are both plain owned data (`String`/`Vec`/primitives,
//! no interior mutability) — auto-`Send`/`Sync`, no `unsafe impl` needed.
//! That means [`RegionCache`] itself can be shared across task-pool workers
//! behind an `Arc<Mutex<_>>` (005-b's cache stays synchronous internally;
//! this is what puts calls to it on a background task), rather than falling
//! back to the bytes-only path 005-c's ticket describes for the case where
//! that isn't true.
//!
//! [`BlockRegistry`] gets the same treatment for the same reason: chunk
//! decode interns block names into it, and those interned [`world::BlockId`]s
//! need to stay valid globally (shared with whatever `DecodedWorld` already
//! holds), not just within one task.
//!
//! A task holds the registry lock across both decode *and* mesh (mesh's
//! face culling calls back into the registry to resolve names via
//! [`world::is_solid`]) — that serializes background chunk tasks against
//! each other, but never blocks the main thread, which is what actually
//! avoids frame stutter. A per-worker registry (sharded, merged back on
//! poll) would restore inter-task parallelism if this ever shows up in a
//! profile; not worth the complexity yet at one save's worth of block names.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::region_cache::{chunk_to_region_coord, RegionCache};
use crate::streaming::PendingChunkWork;
use crate::world::{self, AtlasUvIndex, BlockRegistry, ChunkColumn};
use crate::{BlockMesh, DecodedWorld};

/// Shared, lockable handle onto the region LRU cache (005-b) so every
/// chunk-load task can resolve a chunk's region without owning a copy of
/// the cache itself. `main.rs`'s `setup()` builds the one instance of this,
/// sized to the save and render distance.
#[derive(Resource, Clone)]
pub struct SharedRegionCache(pub Arc<Mutex<RegionCache>>);

/// Shared, read-only per-face UV lookup — the small half of
/// [`world::TextureAtlas`] that doesn't own render-side [`Image`] data (see
/// [`world::atlas::TextureAtlas::uv_index`]), so it can cross the `Send`
/// boundary into a background task.
#[derive(Resource, Clone)]
pub struct SharedAtlasIndex(pub Arc<AtlasUvIndex>);

/// The one material every streamed-in chunk mesh uses, built once at
/// startup (`main.rs::setup`) from the packed atlas.
#[derive(Resource, Clone)]
pub struct TerrainMaterial(pub Handle<StandardMaterial>);

/// How many completed chunk-load tasks get uploaded (mesh handed to
/// `Assets<Mesh>`, entity spawned) per frame — mirrors the parent ticket's
/// "budget spawns per frame": the expensive work already happened off the
/// main thread, but a big batch of completions landing in the same frame
/// would still spike it if uploaded all at once.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ChunkUploadBudget(pub usize);

impl Default for ChunkUploadBudget {
    fn default() -> Self {
        Self(4)
    }
}

/// In-flight chunk-load tasks, keyed by chunk coordinate.
/// [`start_chunk_loads`] checks this before spawning a task so a
/// coordinate that's already loading never gets a duplicate task, and
/// [`poll_completed_chunk_loads`] removes an entry once its task resolves
/// (whether or not it produced a mesh).
#[derive(Resource, Default)]
pub struct InFlightChunkLoads(HashMap<(i32, i32), Task<Option<ChunkLoadResult>>>);

impl InFlightChunkLoads {
    /// Drops every in-flight task whose coordinate is no longer in
    /// `desired` — e.g. the camera reversed near the loading edge before
    /// the task finished (005-d). Simply dropping a [`Task`] cancels it;
    /// see `bevy_tasks::Task::cancel`'s doc comment ("it's possible to
    /// simply drop the `Task` to cancel it"), so no `.await` is needed
    /// here to actually stop the work.
    pub(crate) fn cancel_out_of_range(&mut self, desired: &HashSet<(i32, i32)>) {
        self.0.retain(|coord, _| desired.contains(coord));
    }
}

/// Chunk coordinate -> spawned chunk-mesh entity, so
/// [`crate::unload`] (005-d) knows which entity to despawn for a
/// coordinate leaving render distance. Populated here in
/// [`poll_completed_chunk_loads`] and by `main.rs::setup`'s eager startup
/// spawn — until 005-e deletes that eager path, both need to register into
/// this the same way for unload to work regardless of which one spawned a
/// given chunk. A coordinate with no entry either hasn't spawned yet or was
/// a fully-air column with nothing to render (see
/// [`world::mesh_chunk_column`]'s `None` case) — either way, nothing for
/// unload to despawn.
#[derive(Resource, Default)]
pub struct SpawnedChunkEntities(pub HashMap<(i32, i32), Entity>);

/// The four already-loaded neighbour columns available at the moment a
/// chunk's load task was kicked off — an owned snapshot (cloned out of
/// [`DecodedWorld`] on the main thread before spawning the task), since a
/// background task can't reach back into a live Bevy [`Resource`].
/// Neighbours that finish loading *after* this task started still leave a
/// seam on this chunk's edge; re-meshing the frontier is 005-f's problem.
#[derive(Default)]
struct OwnedNeighbors {
    north: Option<ChunkColumn>,
    south: Option<ChunkColumn>,
    east: Option<ChunkColumn>,
    west: Option<ChunkColumn>,
}

/// What a completed chunk-load task hands back: the decoded column (always
/// recorded into [`DecodedWorld`] so later neighbours and the camera's
/// under-cursor raycast see it) and its mesh, if it had any exposed faces
/// at all (`None` for a fully-air column, same as
/// [`world::mesh_chunk_column`]).
pub struct ChunkLoadResult {
    coord: (i32, i32),
    column: ChunkColumn,
    mesh: Option<Mesh>,
}

/// Adds the in-flight task map and upload budget, and the two systems that
/// drive them. [`SharedRegionCache`], [`SharedAtlasIndex`], and
/// [`TerrainMaterial`] all need real save/asset data that only exists once
/// `main.rs::setup` has run, so this plugin doesn't insert those — it just
/// owns what's meaningful without them.
pub struct ChunkLoadPipelinePlugin;

impl Plugin for ChunkLoadPipelinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InFlightChunkLoads>()
            .init_resource::<ChunkUploadBudget>()
            .init_resource::<SpawnedChunkEntities>()
            .add_systems(
                Update,
                (poll_completed_chunk_loads, start_chunk_loads).chain(),
            );
    }
}

/// Spawns an [`AsyncComputeTaskPool`] task for every
/// [`PendingChunkWork::to_load`] coordinate that isn't already loaded or
/// already in flight — re-requesting a coordinate that's already loading
/// never spawns a duplicate task.
///
/// `pub(crate)` so [`crate::unload`] (005-d) can order its own systems
/// `.before()` this one.
pub(crate) fn start_chunk_loads(
    pending: Res<PendingChunkWork>,
    mut in_flight: ResMut<InFlightChunkLoads>,
    decoded_world: Res<DecodedWorld>,
    region_cache: Option<Res<SharedRegionCache>>,
    atlas: Option<Res<SharedAtlasIndex>>,
) {
    // Both are inserted by `main.rs::setup` once the real save/atlas exist;
    // before that (Startup hasn't finished) there's nothing to load with.
    let (Some(region_cache), Some(atlas)) = (region_cache, atlas) else {
        return;
    };

    let pool = AsyncComputeTaskPool::get();
    for &coord in &pending.to_load {
        if in_flight.0.contains_key(&coord) || decoded_world.columns.contains_key(&coord) {
            continue;
        }

        let (cx, cz) = coord;
        let neighbors = OwnedNeighbors {
            north: decoded_world.columns.get(&(cx, cz - 1)).cloned(),
            south: decoded_world.columns.get(&(cx, cz + 1)).cloned(),
            east: decoded_world.columns.get(&(cx + 1, cz)).cloned(),
            west: decoded_world.columns.get(&(cx - 1, cz)).cloned(),
        };

        let region_cache = region_cache.0.clone();
        let registry = decoded_world.registry.clone();
        let atlas = atlas.0.clone();
        let task = pool
            .spawn(async move { load_and_mesh_chunk(coord, region_cache, registry, atlas, neighbors) });
        in_flight.0.insert(coord, task);
    }
}

/// Polls every in-flight task with `block_on(poll_once(&mut task))` — a
/// single non-blocking poll, Bevy's standard pattern for checking an
/// `AsyncComputeTaskPool` task from a normal system without an executor of
/// its own. For up to [`ChunkUploadBudget`] completed tasks this frame,
/// uploads the mesh and spawns the entity (mirroring `main.rs::setup`'s
/// eager spawn: `Mesh3d`/`MeshMaterial3d`/`Transform`/[`BlockMesh`]) and
/// records the decoded column into [`DecodedWorld`] either way.
///
/// `pub(crate)` so [`crate::unload`] (005-d) can order its own systems
/// `.before()` this one.
pub(crate) fn poll_completed_chunk_loads(
    mut in_flight: ResMut<InFlightChunkLoads>,
    mut decoded_world: ResMut<DecodedWorld>,
    mut spawned: ResMut<SpawnedChunkEntities>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    material: Option<Res<TerrainMaterial>>,
    budget: Res<ChunkUploadBudget>,
) {
    let Some(material) = material else {
        return; // Set by `setup()`; nothing to spawn with before that.
    };

    let mut completed = Vec::new();
    for (&coord, task) in in_flight.0.iter_mut() {
        if let Some(result) = block_on(poll_once(task)) {
            completed.push((coord, result));
            if completed.len() >= budget.0 {
                break;
            }
        }
    }

    for (coord, result) in completed {
        in_flight.0.remove(&coord);
        let Some(result) = result else { continue };

        if let Some(mesh) = result.mesh {
            let (cx, cz) = result.coord;
            let entity = commands
                .spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(material.0.clone()),
                    // Chunk mesh vertices are chunk-local; place the entity at
                    // the chunk's world origin (bevy.x = mc.x, bevy.z = -mc.z —
                    // see `world::mesh` docs), same as `main.rs::setup`.
                    Transform::from_xyz(
                        cx as f32 * world::SECTION_SIZE as f32,
                        0.0,
                        -(cz as f32 * world::SECTION_SIZE as f32),
                    ),
                    BlockMesh,
                ))
                .id();
            spawned.0.insert(result.coord, entity);
        }
        decoded_world.columns.insert(result.coord, result.column);
    }
}

/// Runs entirely inside a background task: resolves `coord`'s region via
/// the shared [`RegionCache`] (005-b), decodes the chunk (002) if it hasn't
/// been already, and meshes it (003) against whatever neighbour columns
/// were already loaded when the task was kicked off. Returns `None` if the
/// chunk doesn't exist in the save or fails to decode (not fully generated,
/// missing/malformed NBT — the same cases [`world::decode_chunk`] itself
/// reports as an `Err` for the caller to skip).
fn load_and_mesh_chunk(
    coord: (i32, i32),
    region_cache: Arc<Mutex<RegionCache>>,
    registry: Arc<Mutex<BlockRegistry>>,
    atlas: Arc<AtlasUvIndex>,
    neighbors: OwnedNeighbors,
) -> Option<ChunkLoadResult> {
    let region_coord = chunk_to_region_coord(coord);
    let (local_x, local_z) = local_chunk_index(coord, region_coord);

    let nbt = {
        let mut cache = region_cache.lock().expect("region cache mutex poisoned");
        let region = cache.get_or_load(region_coord).ok()?;
        region.get_chunk(local_x, local_z)?.clone()
    };

    let mut registry = registry.lock().expect("block registry mutex poisoned");
    let column = world::decode_chunk(&nbt, &mut registry).ok()?;
    let uv_table = world::build_block_uv_table(&registry, &atlas);
    let borrowed_neighbors = world::Neighbors {
        north: neighbors.north.as_ref(),
        south: neighbors.south.as_ref(),
        east: neighbors.east.as_ref(),
        west: neighbors.west.as_ref(),
    };
    let mesh = world::mesh_chunk_column(&column, &registry, &borrowed_neighbors, &uv_table);

    Some(ChunkLoadResult { coord, column, mesh })
}

/// `coord`'s position within `region_coord`'s 32x32 chunk grid, as
/// [`ChunkRegion::get_chunk`](mc_anvil::chunkregion::ChunkRegion::get_chunk)
/// expects it (each axis `0..32`).
fn local_chunk_index((cx, cz): (i32, i32), (rx, rz): (i32, i32)) -> (usize, usize) {
    use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;
    let w = REGION_WIDTH_IN_CHUNKS as i32;
    ((cx - rx * w) as usize, (cz - rz * w) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_chunk_index_matches_region_relative_position() {
        assert_eq!(local_chunk_index((0, 0), (0, 0)), (0, 0));
        assert_eq!(local_chunk_index((31, 31), (0, 0)), (31, 31));
        assert_eq!(local_chunk_index((32, 0), (1, 0)), (0, 0));
        assert_eq!(local_chunk_index((-1, 0), (-1, 0)), (31, 0));
        assert_eq!(local_chunk_index((-32, -33), (-1, -2)), (0, 31));
    }

    /// 005-d: a coordinate that leaves the desired set (the camera reversed
    /// near the loading edge) should have its in-flight task dropped —
    /// dropping cancels it, per `bevy_tasks::Task::cancel`'s doc comment —
    /// while one still inside the desired set is left running.
    #[test]
    fn cancel_out_of_range_drops_tasks_outside_the_desired_set() {
        use bevy::tasks::{AsyncComputeTaskPool, TaskPool};

        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut in_flight = InFlightChunkLoads::default();
        in_flight.0.insert((0, 0), pool.spawn(async { None }));
        in_flight.0.insert((5, 5), pool.spawn(async { None }));

        let desired: HashSet<(i32, i32)> = HashSet::from([(0, 0)]);
        in_flight.cancel_out_of_range(&desired);

        assert!(in_flight.0.contains_key(&(0, 0)), "still desired, should survive");
        assert!(
            !in_flight.0.contains_key(&(5, 5)),
            "left the desired set, should have been canceled"
        );
    }

    /// End-to-end against the real save (see `region_cache.rs`'s tests for
    /// the same convention): resolves a region, decodes one of its chunks,
    /// and meshes it, off the main thread the same way a real task would.
    #[test]
    fn load_and_mesh_chunk_decodes_and_meshes_a_real_chunk() {
        use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;

        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        let region_cache = Arc::new(Mutex::new(RegionCache::new(meta, 4)));
        let registry = Arc::new(Mutex::new(BlockRegistry::new()));
        let atlas = Arc::new(AtlasUvIndex::default());

        // The centre of a region a player has actually visited is the best
        // bet for a fully-generated chunk (edges of the explored area are
        // more likely to be partially generated and get skipped).
        let half = (REGION_WIDTH_IN_CHUNKS / 2) as i32;
        let coord = (rx * REGION_WIDTH_IN_CHUNKS as i32 + half, rz * REGION_WIDTH_IN_CHUNKS as i32 + half);

        let result = load_and_mesh_chunk(
            coord,
            region_cache,
            registry,
            atlas,
            OwnedNeighbors::default(),
        )
        .expect("a real save's region centre should have a fully-generated chunk");

        assert_eq!(result.coord, coord);
        assert_eq!((result.column.x, result.column.z), coord);
    }

    /// A load for a chunk whose region the save doesn't have should fail
    /// cleanly (`None`), not panic — exercised without a real save by
    /// pointing the cache at an empty [`SaveMeta`].
    #[test]
    fn load_and_mesh_chunk_returns_none_for_a_region_the_save_does_not_have() {
        let meta = mc_anvil::SaveMeta {
            name: "empty".to_string(),
            path: std::path::PathBuf::from("does-not-exist"),
            regions: vec![],
        };
        let region_cache = Arc::new(Mutex::new(RegionCache::new(meta, 1)));
        let registry = Arc::new(Mutex::new(BlockRegistry::new()));
        let atlas = Arc::new(AtlasUvIndex::default());

        let result = load_and_mesh_chunk(
            (0, 0),
            region_cache,
            registry,
            atlas,
            OwnedNeighbors::default(),
        );
        assert!(result.is_none());
    }
}
