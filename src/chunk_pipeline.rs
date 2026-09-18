//! Async load/decode/mesh pipeline (ticket 005-c): turns a `to_load` chunk
//! coordinate from [`crate::streaming::PendingChunkWork`]
//! into a spawned mesh entity via [`AsyncComputeTaskPool`], off the main
//! thread except for the final mesh upload / entity spawn.
//!
//! ## Re-meshing the loading frontier (005-f)
//!
//! 003's mesher treats a missing neighbour as air, so a chunk meshed before
//! its neighbour arrives bakes in an exposed face at that edge — a seam
//! that continuously trails the camera at the streaming frontier. When a
//! load completes, [`poll_completed_chunk_loads`] checks that chunk's four
//! neighbour coordinates and queues ([`PendingChunkRemeshes`]) any that are
//! already loaded; [`start_chunk_remeshes`]/[`poll_completed_chunk_remeshes`]
//! rebuild and swap in their mesh the same async-task way loads work,
//! closing the seam instead of leaving it until that neighbour happens to
//! re-mesh for some other reason.
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
//! A task holds the registry lock for decode only (ticket 123). Decode is
//! the one `&mut` user — it interns — and it's ~0.5 ms; mesh only *reads*
//! the registry (face culling resolves names via [`world::is_solid`]) and
//! is ~10 ms, so a task clones both registries the moment decode is done
//! and meshes against the snapshots with the lock released. Until 123 the
//! lock was held across both, which serialised every load, re-mesh and
//! reload task on one thread's worth of meshing — fine while the only
//! source of mesh work was the streaming frontier, but the citybuilder's
//! mines (ticket 116) fire `ChunksEdited` continuously, and their reloads
//! and neighbour re-meshes starved streaming down to ~1 chunk/s. A
//! snapshot taken after this chunk's own decode holds every id the column
//! and its (already-decoded) neighbours can contain, so meshing against it
//! resolves exactly what meshing under the lock would have.
//!
//! ## Live re-mesh on edit (ticket 034, roadmap W7)
//!
//! [`crate::edit`] mutates the very `ChunkRegion` this pipeline reads
//! through (both go through the same shared [`RegionCache`]), so a fresh
//! decode already sees post-edit blocks — but the *already-decoded*
//! [`world::ChunkColumn`]s sitting in [`DecodedWorld`] and their meshes are
//! now stale, and nothing re-derives them on its own. [`ChunksEdited`]
//! is how an edit says so: any system that commits an edit fires it with
//! [`crate::edit::EditReport::chunks`], and [`queue_edited_chunk_reloads`]
//! turns that into two kinds of work, using 005-f's re-mesh queue for the
//! second rather than a second dirty-chunk mechanism:
//!
//! - the edited chunks themselves go through [`PendingChunkReloads`] /
//!   [`start_chunk_reloads`] / [`poll_completed_chunk_reloads`] — a full
//!   re-decode *and* re-mesh, because unlike a 005-f neighbour, these
//!   chunks' own blocks changed;
//! - their loaded neighbours go into [`PendingChunkRemeshes`] — same as
//!   005-f, since only the mesh at the shared boundary can have changed —
//!   but (ticket 123) only the neighbours across a border the edit
//!   actually wrote on ([`crate::edit::ChunkBorders`]), and only once the
//!   reload has *landed*. Queuing them at edit time, as 034 first did,
//!   re-meshed every neighbour against the edited chunk's stale column and
//!   nothing re-meshed them again afterwards.
//!
//! A coordinate that isn't currently in [`DecodedWorld`] is dropped rather
//! than queued: it isn't on screen, and whenever it does stream in,
//! [`load_and_mesh_chunk`] decodes it from the (already-edited) region for
//! free. Reloads of one chunk are rate-limited by [`ChunkReloadThrottle`].

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::edit::{ChunkBorders, EditReport};
use crate::region_cache::{chunk_to_region_coord, RegionCache};
use crate::streaming::PendingChunkWork;
use crate::world::warn::WarnLedger;
use crate::world::{self, AtlasUvIndex, BiomeRegistry, BlockRegistry, ChunkColumn, ColorMaps};
use crate::{BlockMesh, DecodedWorld};

/// Shared, lockable handle onto the region LRU cache (005-b) so every
/// chunk-load task can resolve a chunk's region without owning a copy of
/// the cache itself. `lib.rs`'s `setup_world()` builds the one instance of this,
/// sized to the save and render distance.
#[derive(Resource, Clone)]
pub struct SharedRegionCache(pub Arc<Mutex<RegionCache>>);

/// Shared, read-only per-face UV lookup — the small half of
/// [`world::TextureAtlas`] that doesn't own render-side [`Image`] data (see
/// [`world::atlas::TextureAtlas::uv_index`]), so it can cross the `Send`
/// boundary into a background task.
#[derive(Resource, Clone)]
pub struct SharedAtlasIndex(pub Arc<AtlasUvIndex>);

/// Shared, read-only colormap pair (ticket 013) — plain `[u8; 3]` texel
/// data, so like [`SharedAtlasIndex`] it can cross the `Send` boundary into
/// a background task without dragging any render-side type along.
#[derive(Resource, Clone)]
pub struct SharedColorMaps(pub Arc<ColorMaps>);

/// The one material every streamed-in chunk mesh uses, built once at
/// startup (`lib.rs::setup_world`) from the packed atlas.
#[derive(Resource, Clone)]
pub struct TerrainMaterial(pub Handle<StandardMaterial>);

/// How much of a chunk column [`load_and_mesh_chunk`] actually decodes
/// (ticket 030) — a thin `Resource` wrapper around
/// [`world::decode::FloorPolicy`], which stays free of Bevy itself (see
/// that module's docs). [`ChunkLoadPipelinePlugin`] gives every app
/// `FloorPolicy::WholeWorld` by default via `init_resource` — what
/// `block_viewer` keeps using — and `city::run()` overrides it with
/// `insert_resource` once [`crate::world_app`] returns.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct RenderFloor(pub world::decode::FloorPolicy);

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

    /// Number of load tasks currently in flight — ticket 007's status panel
    /// folds this into "queued chunks" alongside
    /// [`PendingChunkWork::to_load`] (coordinates not yet started).
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

/// Chunk coordinate -> spawned chunk-mesh entity, so
/// [`crate::unload`] (005-d) knows which entity to despawn for a
/// coordinate leaving render distance. Populated here in
/// [`poll_completed_chunk_loads`] and by `lib.rs::setup_world`'s eager startup
/// spawn — until 005-e deletes that eager path, both need to register into
/// this the same way for unload to work regardless of which one spawned a
/// given chunk. A coordinate with no entry either hasn't spawned yet or was
/// a fully-air column with nothing to render (see
/// [`world::mesh_chunk_column`]'s `None` case) — either way, nothing for
/// unload to despawn.
#[derive(Resource, Default)]
pub struct SpawnedChunkEntities(pub HashMap<(i32, i32), Entity>);

/// How many completed re-mesh tasks get uploaded (mesh handle swapped onto
/// its existing entity) per frame — the re-mesh equivalent of
/// [`ChunkUploadBudget`] (ticket 005-f), so a burst of chunks all finishing
/// their re-mesh in the same frame (e.g. a whole ring of the loading
/// frontier closing at once) can't reintroduce the upload-spike stutter
/// 005-c's budget was added to avoid. Kept as a separate counter from
/// `ChunkUploadBudget` rather than sharing it, since initial loads and
/// frontier re-meshes are different kinds of pressure.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ChunkRemeshBudget(pub usize);

impl Default for ChunkRemeshBudget {
    fn default() -> Self {
        Self(4)
    }
}

/// Chunk coordinates queued for a re-mesh (ticket 005-f) because a neighbour
/// they were missing at their last mesh build has since loaded. Populated by
/// [`poll_completed_chunk_loads`] for the four neighbours of every chunk
/// that finishes loading; a `HashSet` rather than a `Vec` so two neighbours
/// of the same coordinate completing in one frame collapse into a single
/// entry (the ticket's "dedupe before spawning tasks"). Drained by
/// [`start_chunk_remeshes`], which drops any coordinate that isn't actually
/// loaded (any more, or yet) — nothing to re-mesh.
#[derive(Resource, Default)]
pub struct PendingChunkRemeshes(HashSet<(i32, i32)>);

impl PendingChunkRemeshes {
    /// Drops any queued coordinate that has since left render distance —
    /// mirrors [`InFlightChunkRemeshes::cancel_out_of_range`]; called from
    /// the same place (`unload`, 005-d/005-f) for the same reason.
    pub(crate) fn cancel_out_of_range(&mut self, desired: &HashSet<(i32, i32)>) {
        self.0.retain(|coord| desired.contains(coord));
    }
}

/// In-flight re-mesh tasks (ticket 005-f), keyed by chunk coordinate — the
/// re-mesh equivalent of [`InFlightChunkLoads`]. Kept separate from that map
/// because a re-mesh task carries no [`ChunkLoadResult`] (the column is
/// already decoded; only the mesh is being rebuilt).
#[derive(Resource, Default)]
pub struct InFlightChunkRemeshes(HashMap<(i32, i32), Task<ChunkRemeshResult>>);

impl InFlightChunkRemeshes {
    /// Cancels (drops) any in-flight re-mesh task whose coordinate has left
    /// render distance since it was queued — same reasoning and mechanism as
    /// [`InFlightChunkLoads::cancel_out_of_range`]: without this, a task that
    /// outlives the chunk's unload would complete later and, finding no
    /// entity in [`SpawnedChunkEntities`] to update, respawn one — a ghost
    /// chunk reappearing after it was meant to stay unloaded.
    pub(crate) fn cancel_out_of_range(&mut self, desired: &HashSet<(i32, i32)>) {
        self.0.retain(|coord, _| desired.contains(coord));
    }
}

/// Fired when an edit ([`crate::edit`], roadmap W4/W5) commits: the chunks
/// whose blocks actually changed, straight from
/// [`crate::edit::EditReport::chunks`], each paired with which of its
/// borders the edit wrote on ([`EditReport::borders`], ticket 123) so the
/// pipeline re-meshes only the neighbours that can have changed. The
/// pipeline that reacts to it ([`queue_edited_chunk_reloads`]) is W7's,
/// ticket 034.
#[derive(Event, Debug, Clone)]
pub struct ChunksEdited(pub Vec<((i32, i32), ChunkBorders)>);

impl ChunksEdited {
    /// Every chunk `report` wrote, with its borders. A report built without
    /// them (a hand-made one in a test) falls back to
    /// [`ChunkBorders::ALL`] per chunk — the conservative reading.
    pub fn from_report(report: &EditReport) -> Self {
        Self(
            report
                .chunks
                .iter()
                .enumerate()
                .map(|(i, &chunk)| (chunk, report.borders.get(i).copied().unwrap_or(ChunkBorders::ALL)))
                .collect(),
        )
    }

    /// `chunks` with every border assumed touched — for a sender that
    /// doesn't know which.
    pub fn all_borders(chunks: impl IntoIterator<Item = (i32, i32)>) -> Self {
        Self(chunks.into_iter().map(|chunk| (chunk, ChunkBorders::ALL)).collect())
    }

    /// Just the coordinates, in order.
    pub fn chunks(&self) -> Vec<(i32, i32)> {
        self.0.iter().map(|&(chunk, _)| chunk).collect()
    }
}

/// How many completed chunk-reload tasks (ticket 034) get uploaded per
/// frame — the reload equivalent of [`ChunkUploadBudget`]/[`ChunkRemeshBudget`].
/// A single edit (a placed building, a fill command) can touch many chunks
/// at once, and each needs a full re-decode, not just a re-mesh, so this
/// stays a separate counter for the same "different kind of pressure"
/// reason [`ChunkRemeshBudget`] does.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ChunkReloadBudget(pub usize);

impl Default for ChunkReloadBudget {
    fn default() -> Self {
        Self(4)
    }
}

/// Chunk coordinates queued for a full reload (ticket 034) because an edit
/// changed their blocks — as opposed to [`PendingChunkRemeshes`], whose
/// coordinates only need their mesh rebuilt against unchanged blocks.
/// Populated by [`queue_edited_chunk_reloads`], drained by
/// [`start_chunk_reloads`]. Each carries the union of the borders every
/// edit since its last dispatch wrote on (ticket 123): two edits to the same
/// chunk before it gets a turn collapse into one reload whose completion
/// re-meshes the neighbours either of them reached.
#[derive(Resource, Default)]
pub struct PendingChunkReloads(HashMap<(i32, i32), ChunkBorders>);

impl PendingChunkReloads {
    /// Mirrors [`PendingChunkRemeshes::cancel_out_of_range`]: a coordinate
    /// the camera has since left render distance shouldn't reload.
    pub(crate) fn cancel_out_of_range(&mut self, desired: &HashSet<(i32, i32)>) {
        self.0.retain(|coord, _| desired.contains(coord));
    }

    /// Queues `coord`, folding `borders` into whatever is already queued.
    fn queue(&mut self, coord: (i32, i32), borders: ChunkBorders) {
        let queued = self.0.entry(coord).or_default();
        *queued = queued.union(borders);
    }
}

/// One reload in flight: its task, plus the borders the edit that caused it
/// touched, carried along so [`poll_completed_chunk_reloads`] knows which
/// neighbours to re-mesh once the fresh column has actually landed.
struct InFlightReload {
    task: Task<Option<ChunkLoadResult>>,
    borders: ChunkBorders,
}

/// In-flight reload tasks (ticket 034), keyed by chunk coordinate — the
/// reload equivalent of [`InFlightChunkRemeshes`]. Reuses [`ChunkLoadResult`]
/// as its task output rather than a distinct type: a reload *is* a load in
/// everything but which queue triggered it, decoding and meshing the same
/// way [`load_and_mesh_chunk`] already does.
#[derive(Resource, Default)]
pub struct InFlightChunkReloads(HashMap<(i32, i32), InFlightReload>);

impl InFlightChunkReloads {
    /// Mirrors [`InFlightChunkRemeshes::cancel_out_of_range`].
    pub(crate) fn cancel_out_of_range(&mut self, desired: &HashSet<(i32, i32)>) {
        self.0.retain(|coord, _| desired.contains(coord));
    }
}

/// How often one chunk may be reloaded (ticket 123). A source of edits that
/// keeps hitting the same chunk every frame — a mine fast-forwarding
/// through air at a job per frame, a gallery advancing a block at a time —
/// would otherwise reload it every frame, each reload a full decode + mesh
/// plus the neighbour re-meshes, and that alone can saturate the task pool.
///
/// Leading-edge with a trailing catch-up: a chunk that hasn't reloaded for
/// `min_interval` reloads *immediately* (a placed building shows up as fast
/// as ever); anything queued for it within the interval waits in
/// [`PendingChunkReloads`] — accumulating borders — and goes out as one
/// reload when the interval is up. So a chunk is never more than
/// `min_interval` stale, and never reloads more than `1 / min_interval`
/// times a second, however many edits land on it.
#[derive(Resource, Debug, Clone)]
pub struct ChunkReloadThrottle {
    pub min_interval: std::time::Duration,
    /// When each coordinate's last reload was dispatched, on
    /// [`Time::elapsed`]'s clock. Pruned of entries older than the interval
    /// on every pass, so it holds at most a recent frame's worth of chunks.
    last_dispatch: HashMap<(i32, i32), std::time::Duration>,
}

impl Default for ChunkReloadThrottle {
    fn default() -> Self {
        Self {
            min_interval: std::time::Duration::from_millis(250),
            last_dispatch: HashMap::new(),
        }
    }
}

impl ChunkReloadThrottle {
    /// Whether `coord` may reload at `now`; records the dispatch if so.
    /// Also prunes entries the interval has already expired, since nothing
    /// will ever consult them again.
    fn admit(&mut self, coord: (i32, i32), now: std::time::Duration) -> bool {
        let interval = self.min_interval;
        self.last_dispatch.retain(|_, at| now.saturating_sub(*at) < interval);
        if self.last_dispatch.contains_key(&coord) {
            return false;
        }
        self.last_dispatch.insert(coord, now);
        true
    }
}

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

/// What a completed re-mesh task (ticket 005-f) hands back. No `column` —
/// unlike a load, a re-mesh doesn't decode anything new, it only rebuilds
/// the mesh of a chunk already recorded in [`DecodedWorld`]. `mesh` is
/// `None` for the (rare, since only horizontal neighbours factor into a
/// column's own top/bottom faces) case where the chunk still has no exposed
/// faces at all.
pub struct ChunkRemeshResult {
    coord: (i32, i32),
    mesh: Option<Mesh>,
}

/// Adds the in-flight task maps and upload budgets, and the four systems
/// that drive them: [`start_chunk_loads`]/[`poll_completed_chunk_loads`]
/// (005-c) and their re-mesh counterparts
/// [`start_chunk_remeshes`]/[`poll_completed_chunk_remeshes`] (005-f).
/// [`SharedRegionCache`], [`SharedAtlasIndex`], and [`TerrainMaterial`] all
/// need real save/asset data that only exists once `lib.rs::setup_world` has
/// run, so this plugin doesn't insert those — it just owns what's
/// meaningful without them.
pub struct ChunkLoadPipelinePlugin;

impl Plugin for ChunkLoadPipelinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InFlightChunkLoads>()
            .init_resource::<ChunkUploadBudget>()
            .init_resource::<SpawnedChunkEntities>()
            .init_resource::<PendingChunkRemeshes>()
            .init_resource::<InFlightChunkRemeshes>()
            .init_resource::<ChunkRemeshBudget>()
            .init_resource::<PendingChunkReloads>()
            .init_resource::<InFlightChunkReloads>()
            .init_resource::<ChunkReloadBudget>()
            .init_resource::<ChunkReloadThrottle>()
            .init_resource::<RenderFloor>()
            .add_event::<ChunksEdited>()
            .add_systems(
                Update,
                (
                    poll_completed_chunk_loads,
                    poll_completed_chunk_remeshes,
                    poll_completed_chunk_reloads,
                    queue_edited_chunk_reloads,
                    start_chunk_loads,
                    start_chunk_remeshes,
                    start_chunk_reloads,
                )
                    .chain(),
            );
    }
}

/// The four horizontal neighbour coordinates of `coord`, in the same
/// north/south/east/west order [`OwnedNeighbors`] and [`world::Neighbors`]
/// use. Shared by every place that needs to look a chunk's neighbours up by
/// coordinate: dispatching a load/re-mesh task's neighbour snapshot
/// ([`owned_neighbors_of`]), and deciding which already-loaded neighbours a
/// completed load should queue for a re-mesh ([`poll_completed_chunk_loads`]).
fn neighbor_coords((cx, cz): (i32, i32)) -> [(i32, i32); 4] {
    [(cx, cz - 1), (cx, cz + 1), (cx + 1, cz), (cx - 1, cz)]
}

/// Clones `coord`'s four neighbour columns out of `columns` (whichever are
/// currently loaded), for handing to a background load/re-mesh task as an
/// [`OwnedNeighbors`] snapshot — see that type's docs for why an owned clone
/// rather than a reference.
fn owned_neighbors_of(
    coord: (i32, i32),
    columns: &HashMap<(i32, i32), ChunkColumn>,
) -> OwnedNeighbors {
    let [north, south, east, west] = neighbor_coords(coord);
    OwnedNeighbors {
        north: columns.get(&north).cloned(),
        south: columns.get(&south).cloned(),
        east: columns.get(&east).cloned(),
        west: columns.get(&west).cloned(),
    }
}

/// Spawns an [`AsyncComputeTaskPool`] task for every
/// [`PendingChunkWork::to_load`] coordinate that isn't already loaded or
/// already in flight — re-requesting a coordinate that's already loading
/// never spawns a duplicate task.
///
/// Drains `pending.to_load` as it goes (mirroring [`start_chunk_remeshes`]/
/// [`start_chunk_reloads`]) rather than leaving dispatched coordinates
/// sitting in it: every entry here gets *some* resolution in this same pass
/// (a task spawned into `in_flight`, or dropped as already-loaded/already-
/// in-flight), so nothing is left to re-check next frame. Previously this
/// read `&pending.to_load` without consuming it, so a coordinate stayed in
/// the list forever after being dispatched — `status_panel`'s "Queued
/// chunks" (`pending.to_load.len() + in_flight_loads.len()`) would then
/// double-count in-flight work and never count down as loads actually
/// completed, making a large render distance's worth of streaming look
/// stalled even while it was progressing normally in the background.
///
/// `pub(crate)` so [`crate::unload`] (005-d) can order its own systems
/// `.before()` this one.
pub(crate) fn start_chunk_loads(
    mut pending: ResMut<PendingChunkWork>,
    mut in_flight: ResMut<InFlightChunkLoads>,
    decoded_world: Res<DecodedWorld>,
    region_cache: Option<Res<SharedRegionCache>>,
    atlas: Option<Res<SharedAtlasIndex>>,
    color_maps: Option<Res<SharedColorMaps>>,
    render_floor: Res<RenderFloor>,
) {
    // All three are inserted by `lib.rs::setup_world` once the real save/atlas/
    // colormaps exist; before that (Startup hasn't finished) there's nothing
    // to load with — leave `pending.to_load` untouched so it's still there
    // to drain once setup finishes.
    let (Some(region_cache), Some(atlas), Some(color_maps)) = (region_cache, atlas, color_maps)
    else {
        return;
    };
    let floor_policy = render_floor.0;

    let pool = AsyncComputeTaskPool::get();
    for coord in std::mem::take(&mut pending.to_load) {
        if in_flight.0.contains_key(&coord) || decoded_world.columns.contains_key(&coord) {
            continue;
        }

        let neighbors = owned_neighbors_of(coord, &decoded_world.columns);

        let region_cache = region_cache.0.clone();
        let registry = decoded_world.registry.clone();
        let biome_registry = decoded_world.biomes.clone();
        let atlas = atlas.0.clone();
        let color_maps = color_maps.0.clone();
        let task = pool.spawn(async move {
            load_and_mesh_chunk(
                coord,
                region_cache,
                registry,
                biome_registry,
                atlas,
                color_maps,
                neighbors,
                floor_policy,
            )
        });
        in_flight.0.insert(coord, task);
    }
}

/// Spawns an [`AsyncComputeTaskPool`] task for every coordinate in
/// [`PendingChunkRemeshes`] that isn't already being re-meshed (ticket
/// 005-f) — drains the queue each call, so a coordinate that no longer has a
/// loaded column (unloaded again before its re-mesh got a turn) is silently
/// dropped rather than re-queued.
///
/// `pub(crate)` so [`crate::unload`] (005-d/005-f) can order its own systems
/// `.before()` this one, the same way it does for [`start_chunk_loads`].
pub(crate) fn start_chunk_remeshes(
    mut pending: ResMut<PendingChunkRemeshes>,
    mut in_flight: ResMut<InFlightChunkRemeshes>,
    decoded_world: Res<DecodedWorld>,
    atlas: Option<Res<SharedAtlasIndex>>,
    color_maps: Option<Res<SharedColorMaps>>,
) {
    // Mirrors `start_chunk_loads`: nothing to mesh with before `setup()`.
    let (Some(atlas), Some(color_maps)) = (atlas, color_maps) else {
        return;
    };

    let pool = AsyncComputeTaskPool::get();
    for coord in pending.0.drain() {
        if in_flight.0.contains_key(&coord) {
            continue;
        }
        let Some(column) = decoded_world.columns.get(&coord).cloned() else {
            continue; // Unloaded again before the re-mesh could start.
        };

        let neighbors = owned_neighbors_of(coord, &decoded_world.columns);
        let registry = decoded_world.registry.clone();
        let biome_registry = decoded_world.biomes.clone();
        let atlas = atlas.0.clone();
        let color_maps = color_maps.0.clone();
        let task = pool.spawn(async move {
            remesh_chunk_column(coord, column, registry, biome_registry, atlas, color_maps, neighbors)
        });
        in_flight.0.insert(coord, task);
    }
}

/// Polls every in-flight task with `block_on(poll_once(&mut task))` — a
/// single non-blocking poll, Bevy's standard pattern for checking an
/// `AsyncComputeTaskPool` task from a normal system without an executor of
/// its own. For up to [`ChunkUploadBudget`] completed tasks this frame,
/// uploads the mesh and spawns the entity (mirroring `lib.rs::setup_world`'s
/// eager spawn: `Mesh3d`/`MeshMaterial3d`/`Transform`/[`BlockMesh`]) and
/// records the decoded column into [`DecodedWorld`] either way. Also queues
/// a re-mesh (ticket 005-f) for any of the four neighbour coordinates that
/// are already loaded — they were meshed before this chunk existed, so their
/// edge facing it was baked in as a seam (see the module's neighbour-clone
/// doc comment and 003's `missing_neighbor_leaves_a_seam`); now that this
/// chunk has landed in [`DecodedWorld`], that neighbour's mesh can close it.
///
/// `pub(crate)` so [`crate::unload`] (005-d) can order its own systems
/// `.before()` this one.
pub(crate) fn poll_completed_chunk_loads(
    mut in_flight: ResMut<InFlightChunkLoads>,
    mut decoded_world: ResMut<DecodedWorld>,
    mut spawned: ResMut<SpawnedChunkEntities>,
    mut pending_remeshes: ResMut<PendingChunkRemeshes>,
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
                    // see `world::mesh` docs), same as `lib.rs::setup_world`.
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

        for neighbor in neighbor_coords(result.coord) {
            if decoded_world.columns.contains_key(&neighbor) {
                pending_remeshes.0.insert(neighbor);
            }
        }
    }
}

/// Polls every in-flight re-mesh task (ticket 005-f) the same
/// `block_on(poll_once(&mut task))` way [`poll_completed_chunk_loads`] polls
/// loads. For up to [`ChunkRemeshBudget`] completed tasks this frame, swaps
/// the entity's existing `Mesh3d` handle for the freshly-built one in place
/// (freeing the old mesh asset) rather than despawning/respawning — per the
/// ticket, re-meshing a chunk never changes anything about its own entity
/// beyond the mesh geometry. Falls back to spawning or despawning only for
/// the edge cases where a re-mesh flips whether the chunk has any exposed
/// faces at all (see [`ChunkRemeshResult`]'s docs).
///
/// `pub(crate)` so [`crate::unload`] (005-d/005-f) can order its own systems
/// `.before()` this one.
pub(crate) fn poll_completed_chunk_remeshes(
    mut in_flight: ResMut<InFlightChunkRemeshes>,
    mut spawned: ResMut<SpawnedChunkEntities>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_of: Query<&mut Mesh3d>,
    material: Option<Res<TerrainMaterial>>,
    budget: Res<ChunkRemeshBudget>,
) {
    let Some(material) = material else {
        return; // Set by `setup()`; nothing to spawn with before that.
    };

    let mut completed = Vec::new();
    for (_coord, task) in in_flight.0.iter_mut() {
        if let Some(result) = block_on(poll_once(task)) {
            completed.push(result);
            if completed.len() >= budget.0 {
                break;
            }
        }
    }

    for result in completed {
        in_flight.0.remove(&result.coord);
        apply_mesh_update(
            result.coord,
            result.mesh,
            &mut spawned,
            &mut commands,
            &mut meshes,
            &mut mesh_of,
            &material.0,
        );
    }
}

/// Reads every [`ChunksEdited`] event fired this frame (ticket 034) and
/// queues every edited chunk into [`PendingChunkReloads`] — its own blocks
/// changed, so it needs a full re-decode — with the borders the edit wrote
/// on. Neighbours are *not* queued here (ticket 123): that's
/// [`poll_completed_chunk_reloads`]'s job once the fresh column has landed,
/// so their re-mesh sees the post-edit blocks rather than the stale ones
/// still in [`DecodedWorld`] at this point.
pub(crate) fn queue_edited_chunk_reloads(
    mut events: EventReader<ChunksEdited>,
    mut pending_reloads: ResMut<PendingChunkReloads>,
) {
    for ChunksEdited(chunks) in events.read() {
        for &(coord, borders) in chunks {
            pending_reloads.queue(coord, borders);
        }
    }
}

/// Spawns an [`AsyncComputeTaskPool`] task for every coordinate in
/// [`PendingChunkReloads`] that isn't already reloading, is currently in
/// [`DecodedWorld`] (ticket 034) — a coordinate not loaded isn't on screen,
/// and the next real load will decode it from the already-edited region for
/// free, so it's dropped rather than queued for later — and that
/// [`ChunkReloadThrottle`] admits (ticket 123).
///
/// Unlike [`start_chunk_remeshes`], a coordinate already in flight (or
/// throttled) is left in the pending set instead of being dropped: see the
/// ticket's "Watch out" — a second edit to a chunk mid-reload must still get
/// its own reload once the first one clears, because nothing else will
/// re-trigger it the way a later neighbour arrival does for 005-f's
/// frontier case.
pub(crate) fn start_chunk_reloads(
    mut pending: ResMut<PendingChunkReloads>,
    mut in_flight: ResMut<InFlightChunkReloads>,
    mut throttle: ResMut<ChunkReloadThrottle>,
    time: Res<Time>,
    decoded_world: Res<DecodedWorld>,
    region_cache: Option<Res<SharedRegionCache>>,
    atlas: Option<Res<SharedAtlasIndex>>,
    color_maps: Option<Res<SharedColorMaps>>,
    render_floor: Res<RenderFloor>,
) {
    // Mirrors `start_chunk_loads`/`start_chunk_remeshes`: nothing to
    // decode/mesh with before `setup()`.
    let (Some(region_cache), Some(atlas), Some(color_maps)) = (region_cache, atlas, color_maps)
    else {
        return;
    };
    let floor_policy = render_floor.0;
    let now = time.elapsed();

    // Dropped-because-unloaded first, so an unloaded chunk never occupies a
    // throttle slot.
    pending.0.retain(|coord, _| decoded_world.columns.contains_key(coord));
    let ready: Vec<(i32, i32)> = pending
        .0
        .keys()
        .copied()
        .filter(|coord| !in_flight.0.contains_key(coord) && throttle.admit(*coord, now))
        .collect();

    let pool = AsyncComputeTaskPool::get();
    for coord in ready {
        let Some(borders) = pending.0.remove(&coord) else { continue };

        let neighbors = owned_neighbors_of(coord, &decoded_world.columns);
        let region_cache = region_cache.0.clone();
        let registry = decoded_world.registry.clone();
        let biome_registry = decoded_world.biomes.clone();
        let atlas = atlas.0.clone();
        let color_maps = color_maps.0.clone();
        let task = pool.spawn(async move {
            load_and_mesh_chunk(
                coord,
                region_cache,
                registry,
                biome_registry,
                atlas,
                color_maps,
                neighbors,
                floor_policy,
            )
        });
        in_flight.0.insert(coord, InFlightReload { task, borders });
    }
}

/// Polls every in-flight reload task (ticket 034) the same
/// `block_on(poll_once(&mut task))` way the other two polling systems do.
/// For up to [`ChunkReloadBudget`] completed tasks this frame, records the
/// freshly-decoded column into [`DecodedWorld`] (replacing the stale one —
/// the whole reason this queue exists rather than reusing
/// [`poll_completed_chunk_remeshes`]) and applies the mesh update the same
/// spawn/swap/despawn way a re-mesh does.
///
/// Then (ticket 123) queues the neighbours whose mesh the edit can have
/// changed, now that the column they'd mesh against is the fresh one: the
/// loaded neighbour across each border the edit wrote on, or across every
/// border if the reload moved the column's render floor (a heightmap
/// recompute after a surface edit) — a neighbour's faces against this
/// chunk are culled by that floor along the whole shared side, see
/// `world::mesh::occludes`. A neighbour whose own reload is still in
/// flight is re-queued for a reload rather than re-meshed: its running
/// task snapshotted *this* chunk before this reload landed, so its result
/// is already stale at the border, and a plain re-mesh now would clone its
/// own not-yet-landed column. The follow-up reload carries no borders of
/// its own, so the two chunks can't keep re-queuing each other.
///
/// A `None` result (the edit somehow left the chunk undecodable) is
/// dropped with nothing rendered — [`load_and_mesh_chunk`] already logs a
/// real decode failure; the routine "not fully generated" case can't
/// happen here since the edit itself refuses ungenerated chunks (`edit`
/// module, `require_full_status`).
pub(crate) fn poll_completed_chunk_reloads(
    mut in_flight: ResMut<InFlightChunkReloads>,
    mut pending_reloads: ResMut<PendingChunkReloads>,
    mut pending_remeshes: ResMut<PendingChunkRemeshes>,
    mut decoded_world: ResMut<DecodedWorld>,
    mut spawned: ResMut<SpawnedChunkEntities>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_of: Query<&mut Mesh3d>,
    material: Option<Res<TerrainMaterial>>,
    budget: Res<ChunkReloadBudget>,
) {
    let Some(material) = material else {
        return; // Set by `setup()`; nothing to spawn with before that.
    };

    let mut completed = Vec::new();
    for (&coord, reload) in in_flight.0.iter_mut() {
        if let Some(result) = block_on(poll_once(&mut reload.task)) {
            completed.push((coord, reload.borders, result));
            if completed.len() >= budget.0 {
                break;
            }
        }
    }

    for (coord, borders, result) in completed {
        in_flight.0.remove(&coord);
        let Some(result) = result else { continue };

        let previous = decoded_world.columns.insert(coord, result.column);
        apply_mesh_update(
            coord,
            result.mesh,
            &mut spawned,
            &mut commands,
            &mut meshes,
            &mut mesh_of,
            &material.0,
        );

        let floor_moved = previous.is_some_and(|old| old.floor_y != decoded_world.columns[&coord].floor_y);
        let borders = if floor_moved { ChunkBorders::ALL } else { borders };
        for (neighbor, touched) in neighbor_coords(coord).into_iter().zip(borders.as_neighbor_order()) {
            if !touched || !decoded_world.columns.contains_key(&neighbor) {
                continue;
            }
            if in_flight.0.contains_key(&neighbor) {
                pending_reloads.queue(neighbor, ChunkBorders::default());
            } else {
                pending_remeshes.0.insert(neighbor);
            }
        }
    }
}

/// Shared by [`poll_completed_chunk_remeshes`] and
/// [`poll_completed_chunk_reloads`]: the three-way outcome a completed mesh
/// build can have against whatever entity already exists for `coord`. Swaps
/// the `Mesh3d` handle in place (freeing the old one) when both a mesh and
/// an entity exist — the common case, since re-meshing/reloading a chunk
/// never itself changes anything about the entity beyond the geometry —
/// and falls back to spawning or despawning for the edge cases where the
/// update flips whether the chunk has any exposed faces at all.
fn apply_mesh_update(
    coord: (i32, i32),
    mesh: Option<Mesh>,
    spawned: &mut SpawnedChunkEntities,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mesh_of: &mut Query<&mut Mesh3d>,
    material: &Handle<StandardMaterial>,
) {
    let existing_entity = spawned.0.get(&coord).copied();

    match (mesh, existing_entity) {
        (Some(mesh), Some(entity)) => {
            if let Ok(mut mesh3d) = mesh_of.get_mut(entity) {
                let old_handle = mesh3d.0.clone();
                mesh3d.0 = meshes.add(mesh);
                meshes.remove(&old_handle);
            }
        }
        (Some(mesh), None) => {
            // Rare: the chunk had no exposed faces at its last mesh build
            // (no entity yet) but now does — spawn one, same as a load.
            let (cx, cz) = coord;
            let entity = commands
                .spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(material.clone()),
                    Transform::from_xyz(
                        cx as f32 * world::SECTION_SIZE as f32,
                        0.0,
                        -(cz as f32 * world::SECTION_SIZE as f32),
                    ),
                    BlockMesh,
                ))
                .id();
            spawned.0.insert(coord, entity);
        }
        (None, Some(entity)) => {
            // Rare: the chunk's last exposed face is now interior — free
            // the mesh and despawn, mirroring `unload::unload_chunks`.
            if let Ok(mesh3d) = mesh_of.get(entity) {
                meshes.remove(&mesh3d.0);
            }
            commands.entity(entity).despawn();
            spawned.0.remove(&coord);
        }
        (None, None) => {} // Still nothing to render; nothing to do.
    }
}

/// Shared by [`load_and_mesh_chunk`] and [`remesh_chunk_column`]: builds the
/// per-block-id UV table (004) and meshes (003) `column` against whichever
/// of `neighbors` are `Some`. The only difference between a load's and a
/// re-mesh's (005-f) mesh build is where `column` came from — freshly
/// decoded vs. already sitting in [`DecodedWorld`] — so both funnel through
/// here rather than duplicating the `Neighbors` borrow dance.
fn mesh_column_with_neighbors(
    column: &ChunkColumn,
    registry: &BlockRegistry,
    biome_registry: &BiomeRegistry,
    atlas: &AtlasUvIndex,
    color_maps: &ColorMaps,
    neighbors: &OwnedNeighbors,
) -> Option<Mesh> {
    let uv_table = world::build_block_uv_table(registry, atlas);
    let block_tint = world::build_block_tint_table(registry, atlas);
    let biome_colors = world::build_biome_tint_table(biome_registry, color_maps);
    let borrowed_neighbors = world::Neighbors {
        north: neighbors.north.as_ref(),
        south: neighbors.south.as_ref(),
        east: neighbors.east.as_ref(),
        west: neighbors.west.as_ref(),
    };
    world::mesh_chunk_column(
        column,
        registry,
        &borrowed_neighbors,
        &uv_table,
        &block_tint,
        &biome_colors,
    )
}

/// Chunk coordinates already reported as undecodable (ticket 081), so a
/// corrupt chunk is logged once per run rather than once per re-stream.
static UNDECODABLE_CHUNK: WarnLedger = WarnLedger::new();

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
    biome_registry: Arc<Mutex<BiomeRegistry>>,
    atlas: Arc<AtlasUvIndex>,
    color_maps: Arc<ColorMaps>,
    neighbors: OwnedNeighbors,
    floor_policy: world::decode::FloorPolicy,
) -> Option<ChunkLoadResult> {
    let region_coord = chunk_to_region_coord(coord);
    let (local_x, local_z) = local_chunk_index(coord, region_coord);

    let nbt = {
        let mut cache = region_cache.lock().expect("region cache mutex poisoned");
        let region = cache.get_or_load(region_coord).ok()?;
        region.get_chunk(local_x, local_z)?.clone()
    };

    // Both locks taken here, in this order, for decode only — the one lock
    // ordering to reason about — and released (with a snapshot of each
    // registry taken) before the mesh; see the module docs' "Send boundary".
    let (column, registry, biome_registry) = {
        let mut registry = registry.lock().expect("block registry mutex poisoned");
        let mut biome_registry = biome_registry.lock().expect("biome registry mutex poisoned");
        let column = match world::decode_chunk(&nbt, &mut registry, &mut biome_registry, floor_policy) {
            Ok(column) => column,
            // Not fully generated is routine at the edge of explored terrain —
            // every real save has plenty of these, so logging it would just be
            // startup-log noise, not a problem to report (ticket 008 only asks
            // for genuine failures — corrupt/unexpected NBT — to be logged).
            Err(world::DecodeError::NotFullyGenerated(_)) => return None,
            Err(err) => {
                // Once per chunk coordinate for the whole run (ticket 081), not
                // once per attempt: a chunk that streams out and back in is the
                // same corrupt chunk, and re-reporting it every time the camera
                // revisits the area drowns out everything else.
                if UNDECODABLE_CHUNK.first_time(&format!("{coord:?}")) {
                    println!("block_viewer: skipping chunk {coord:?} — failed to decode: {err}");
                }
                return None;
            }
        };
        (column, registry.clone(), biome_registry.clone())
    };
    let mesh = mesh_column_with_neighbors(
        &column,
        &registry,
        &biome_registry,
        &atlas,
        &color_maps,
        &neighbors,
    );

    Some(ChunkLoadResult { coord, column, mesh })
}

/// Runs entirely inside a background task (ticket 005-f): rebuilds `coord`'s
/// mesh from its already-decoded `column` against whatever neighbour columns
/// are loaded *now* — unlike the snapshot a load task's [`OwnedNeighbors`]
/// was taken from, this one is built fresh in [`start_chunk_remeshes`] right
/// before dispatch, so it picks up the neighbour that just triggered this
/// re-mesh (and any others that have since arrived).
fn remesh_chunk_column(
    coord: (i32, i32),
    column: ChunkColumn,
    registry: Arc<Mutex<BlockRegistry>>,
    biome_registry: Arc<Mutex<BiomeRegistry>>,
    atlas: Arc<AtlasUvIndex>,
    color_maps: Arc<ColorMaps>,
    neighbors: OwnedNeighbors,
) -> ChunkRemeshResult {
    // Nothing to intern here — the column is already decoded — so the locks
    // are held only long enough to snapshot (ticket 123); the mesh itself
    // never contends with a load task's decode.
    let registry = registry.lock().expect("block registry mutex poisoned").clone();
    let biome_registry = biome_registry.lock().expect("biome registry mutex poisoned").clone();
    let mesh = mesh_column_with_neighbors(
        &column,
        &registry,
        &biome_registry,
        &atlas,
        &color_maps,
        &neighbors,
    );
    ChunkRemeshResult { coord, mesh }
}

/// `coord`'s position within `region_coord`'s 32x32 chunk grid, as
/// [`ChunkRegion::get_chunk`](mc_anvil::chunkregion::ChunkRegion::get_chunk)
/// expects it (each axis `0..32`).
///
/// `pub(crate)` so ticket 022's blueprint extraction resolves a column
/// through the same arithmetic this pipeline does, rather than writing a
/// second copy of it.
pub(crate) fn local_chunk_index((cx, cz): (i32, i32), (rx, rz): (i32, i32)) -> (usize, usize) {
    use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;
    let w = REGION_WIDTH_IN_CHUNKS as i32;
    ((cx - rx * w) as usize, (cz - rz * w) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A colormap pair with no real pixels — fine for these tests, which
    /// exercise face-culling/lock-passing behaviour, not what colour a
    /// block actually ends up (that's `world::tint`'s own tests).
    fn stub_color_maps() -> Arc<ColorMaps> {
        Arc::new(ColorMaps {
            grass: vec![[0u8, 0, 0]; 256 * 256],
            foliage: vec![[0u8, 0, 0]; 256 * 256],
        })
    }

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

    /// 005-f: same cancellation behaviour as `InFlightChunkLoads`, but for
    /// in-flight re-mesh tasks — a coordinate that's left render distance
    /// shouldn't get its re-mesh uploaded later and respawn a ghost entity.
    #[test]
    fn in_flight_remeshes_cancel_out_of_range_drops_tasks_outside_the_desired_set() {
        use bevy::tasks::{AsyncComputeTaskPool, TaskPool};

        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut in_flight = InFlightChunkRemeshes::default();
        in_flight.0.insert(
            (0, 0),
            pool.spawn(async { ChunkRemeshResult { coord: (0, 0), mesh: None } }),
        );
        in_flight.0.insert(
            (5, 5),
            pool.spawn(async { ChunkRemeshResult { coord: (5, 5), mesh: None } }),
        );

        let desired: HashSet<(i32, i32)> = HashSet::from([(0, 0)]);
        in_flight.cancel_out_of_range(&desired);

        assert!(in_flight.0.contains_key(&(0, 0)), "still desired, should survive");
        assert!(
            !in_flight.0.contains_key(&(5, 5)),
            "left the desired set, should have been canceled"
        );
    }

    /// 005-f: a coordinate queued for re-mesh that's left render distance
    /// before `start_chunk_remeshes` got to it should be dropped, not spawn
    /// a task for a chunk that's about to be (or already was) unloaded.
    #[test]
    fn pending_remeshes_cancel_out_of_range_drops_coords_outside_the_desired_set() {
        let mut pending = PendingChunkRemeshes::default();
        pending.0.insert((0, 0));
        pending.0.insert((5, 5));

        let desired: HashSet<(i32, i32)> = HashSet::from([(0, 0)]);
        pending.cancel_out_of_range(&desired);

        assert_eq!(pending.0, HashSet::from([(0, 0)]));
    }

    /// The four neighbour coordinates, in the north/south/east/west order
    /// [`OwnedNeighbors`] and [`world::Neighbors`] both use.
    #[test]
    fn neighbor_coords_matches_the_north_south_east_west_convention() {
        assert_eq!(
            neighbor_coords((5, -3)),
            [(5, -4), (5, -2), (6, -3), (4, -3)]
        );
    }

    /// 005-f end to end at the task level: a chunk meshed with a missing
    /// neighbour bakes in an exposed face facing it (003's
    /// `missing_neighbor_leaves_a_seam`); re-meshing the same column once
    /// that neighbour is available closes it, the same way
    /// `world::mesh`'s own `loaded_neighbor_closes_the_seam` does for
    /// `mesh_chunk_column` directly — this checks `remesh_chunk_column`,
    /// the async-task entry point [`start_chunk_remeshes`] actually spawns.
    #[test]
    fn remesh_chunk_column_closes_a_seam_once_the_neighbor_is_available() {
        use bevy::render::mesh::Indices;

        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");

        let mut blocks = Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]);
        blocks[world::ChunkSection::index(15, 5, 0)] = stone;
        let biomes = Box::new([world::BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]);
        let column = world::ChunkColumn {
            x: 0,
            z: 0,
            sections: vec![world::ChunkSection { y: 0, blocks, biomes }],
            floor_y: world::WORLD_MIN_Y,
        };

        let registry = Arc::new(Mutex::new(registry));
        let biome_registry = Arc::new(Mutex::new(world::BiomeRegistry::new()));
        let atlas = Arc::new(AtlasUvIndex::default());
        let color_maps = stub_color_maps();

        // No neighbours loaded yet: all 6 faces of the lone block render,
        // including the east face facing the not-yet-loaded neighbour.
        let without_neighbor = remesh_chunk_column(
            (0, 0),
            column.clone(),
            registry.clone(),
            biome_registry.clone(),
            atlas.clone(),
            color_maps.clone(),
            OwnedNeighbors::default(),
        );
        let mesh = without_neighbor.mesh.unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 6 * 6);

        // The east neighbour now has a matching block across the boundary —
        // re-meshing with it present should cull that one face.
        let mut east_blocks = Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]);
        east_blocks[world::ChunkSection::index(0, 5, 0)] = stone;
        let east_biomes = Box::new([world::BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]);
        let east_neighbor = world::ChunkColumn {
            x: 1,
            z: 0,
            sections: vec![world::ChunkSection { y: 0, blocks: east_blocks, biomes: east_biomes }],
            floor_y: world::WORLD_MIN_Y,
        };
        let neighbors = OwnedNeighbors {
            east: Some(east_neighbor),
            ..Default::default()
        };

        let with_neighbor =
            remesh_chunk_column((0, 0), column, registry, biome_registry, atlas, color_maps, neighbors);
        let mesh = with_neighbor.mesh.unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 5 * 6);
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
        let biome_registry = Arc::new(Mutex::new(world::BiomeRegistry::new()));
        let atlas = Arc::new(AtlasUvIndex::default());
        let color_maps = stub_color_maps();

        // The centre of a region a player has actually visited is the best
        // bet for a fully-generated chunk (edges of the explored area are
        // more likely to be partially generated and get skipped).
        let half = (REGION_WIDTH_IN_CHUNKS / 2) as i32;
        let coord = (rx * REGION_WIDTH_IN_CHUNKS as i32 + half, rz * REGION_WIDTH_IN_CHUNKS as i32 + half);

        let result = load_and_mesh_chunk(
            coord,
            region_cache,
            registry,
            biome_registry,
            atlas,
            color_maps,
            OwnedNeighbors::default(),
            world::decode::FloorPolicy::WholeWorld,
        )
        .expect("a real save's region centre should have a fully-generated chunk");

        assert_eq!(result.coord, coord);
        assert_eq!((result.column.x, result.column.z), coord);
    }

    /// Throughput probe, not a pass/fail test (`#[ignore]`d): times what one
    /// background task costs, stage by stage, over a real render-distance
    /// disc of the real save — the region load, then the region-cache lock
    /// + NBT clone, decode, and mesh for every chunk — then the same disc
    /// through the real task body on a pool, which is what ticket 123's
    /// lock release buys (before it, the pooled number was the serial one).
    /// Run whenever streaming "feels slow":
    ///
    /// `cargo test --lib chunk_pipeline::tests::probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_serial_load_throughput_over_a_real_disc() {
        use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;
        use std::time::Instant;

        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        println!("save: {} ({} regions)", meta.name, meta.regions.len());
        let (rx, rz) = meta.regions[0];

        let region_cache = Arc::new(Mutex::new(RegionCache::new(meta, 25)));
        let registry = Arc::new(Mutex::new(BlockRegistry::new()));
        let biome_registry = Arc::new(Mutex::new(world::BiomeRegistry::new()));
        let atlas = Arc::new(AtlasUvIndex::default());
        let color_maps = stub_color_maps();

        let half = (REGION_WIDTH_IN_CHUNKS / 2) as i32;
        let center = (rx * REGION_WIDTH_IN_CHUNKS as i32 + half, rz * REGION_WIDTH_IN_CHUNKS as i32 + half);

        let started = Instant::now();
        {
            let mut cache = region_cache.lock().unwrap();
            cache.get_or_load(chunk_to_region_coord(center)).unwrap();
        }
        println!("region load: {:?}", started.elapsed());

        let disc = crate::streaming::desired_chunks(center, 12);
        let mut coords: Vec<(i32, i32)> = disc.into_iter().collect();
        coords.sort_unstable();

        let mut fetch = std::time::Duration::ZERO;
        let mut decode = std::time::Duration::ZERO;
        let mut mesh = std::time::Duration::ZERO;
        let mut loaded = 0usize;
        let total = Instant::now();
        for coord in &coords {
            let region_coord = chunk_to_region_coord(*coord);
            let (lx, lz) = local_chunk_index(*coord, region_coord);
            let t = Instant::now();
            let nbt = {
                let mut cache = region_cache.lock().unwrap();
                let Ok(region) = cache.get_or_load(region_coord) else { continue };
                let Some(nbt) = region.get_chunk(lx, lz) else { continue };
                nbt.clone()
            };
            fetch += t.elapsed();

            let t = Instant::now();
            let mut reg = registry.lock().unwrap();
            let mut bio = biome_registry.lock().unwrap();
            let Ok(column) =
                world::decode_chunk(&nbt, &mut reg, &mut bio, world::decode::FloorPolicy::WholeWorld)
            else {
                continue;
            };
            decode += t.elapsed();

            let t = Instant::now();
            let _ = mesh_column_with_neighbors(
                &column,
                &reg,
                &bio,
                &atlas,
                &color_maps,
                &OwnedNeighbors::default(),
            );
            mesh += t.elapsed();
            loaded += 1;
        }
        let wall = total.elapsed();
        println!(
            "serial: {loaded}/{} chunks in {wall:?}: fetch {fetch:?}, decode {decode:?}, mesh {mesh:?} — {:.2} ms/chunk, registry {} names",
            coords.len(),
            wall.as_secs_f64() * 1000.0 / loaded.max(1) as f64,
            registry.lock().unwrap().len(),
        );

        // The same disc through the real task body on a pool, the way the
        // pipeline runs it — what ticket 123's lock release buys. The
        // registries are shared exactly as `DecodedWorld`'s are.
        use bevy::tasks::{block_on, AsyncComputeTaskPool, TaskPoolBuilder};
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
        let pool = AsyncComputeTaskPool::get_or_init(|| TaskPoolBuilder::new().num_threads(threads).build());
        let total = Instant::now();
        let tasks: Vec<_> = coords
            .iter()
            .map(|&coord| {
                let (region_cache, registry, biome_registry, atlas, color_maps) =
                    (region_cache.clone(), registry.clone(), biome_registry.clone(), atlas.clone(), color_maps.clone());
                pool.spawn(async move {
                    load_and_mesh_chunk(
                        coord,
                        region_cache,
                        registry,
                        biome_registry,
                        atlas,
                        color_maps,
                        OwnedNeighbors::default(),
                        world::decode::FloorPolicy::WholeWorld,
                    )
                    .is_some()
                })
            })
            .collect();
        let loaded = tasks.into_iter().map(block_on).filter(|ok| *ok).count();
        let wall = total.elapsed();
        println!(
            "pooled ({threads} threads): {loaded}/{} chunks in {wall:?} — {:.2} ms/chunk wall",
            coords.len(),
            wall.as_secs_f64() * 1000.0 / loaded.max(1) as f64,
        );
    }

    /// A load for a chunk whose region the save doesn't have should fail
    /// cleanly (`None`), not panic — exercised without a real save by
    /// pointing the cache at an empty [`SaveMeta`].
    #[test]
    fn load_and_mesh_chunk_returns_none_for_a_region_the_save_does_not_have() {
        let meta = mc_anvil::SaveMeta {
            name: "empty".to_string(),
            path: std::path::PathBuf::from("does-not-exist"),
            region_dir: std::path::PathBuf::from("does-not-exist/region"),
            regions: vec![],
        };
        let region_cache = Arc::new(Mutex::new(RegionCache::new(meta, 1)));
        let registry = Arc::new(Mutex::new(BlockRegistry::new()));
        let biome_registry = Arc::new(Mutex::new(world::BiomeRegistry::new()));
        let atlas = Arc::new(AtlasUvIndex::default());
        let color_maps = stub_color_maps();

        let result = load_and_mesh_chunk(
            (0, 0),
            region_cache,
            registry,
            biome_registry,
            atlas,
            color_maps,
            OwnedNeighbors::default(),
            world::decode::FloorPolicy::WholeWorld,
        );
        assert!(result.is_none());
    }

    /// Ticket 034: same cancellation behaviour as [`InFlightChunkRemeshes`],
    /// for in-flight reloads.
    #[test]
    fn in_flight_reloads_cancel_out_of_range_drops_tasks_outside_the_desired_set() {
        use bevy::tasks::{AsyncComputeTaskPool, TaskPool};

        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let mut in_flight = InFlightChunkReloads::default();
        let reload = |task| InFlightReload { task, borders: ChunkBorders::ALL };
        in_flight.0.insert((0, 0), reload(pool.spawn(async { None })));
        in_flight.0.insert((5, 5), reload(pool.spawn(async { None })));

        let desired: HashSet<(i32, i32)> = HashSet::from([(0, 0)]);
        in_flight.cancel_out_of_range(&desired);

        assert!(in_flight.0.contains_key(&(0, 0)), "still desired, should survive");
        assert!(
            !in_flight.0.contains_key(&(5, 5)),
            "left the desired set, should have been canceled"
        );
    }

    /// Ticket 034: same as [`PendingChunkRemeshes`]'s equivalent test — a
    /// coordinate queued for reload that's left render distance before
    /// `start_chunk_reloads` got to it should be dropped.
    #[test]
    fn pending_reloads_cancel_out_of_range_drops_coords_outside_the_desired_set() {
        let mut pending = PendingChunkReloads::default();
        pending.queue((0, 0), ChunkBorders::ALL);
        pending.queue((5, 5), ChunkBorders::ALL);

        let desired: HashSet<(i32, i32)> = HashSet::from([(0, 0)]);
        pending.cancel_out_of_range(&desired);

        assert_eq!(pending.0.keys().copied().collect::<Vec<_>>(), vec![(0, 0)]);
    }

    /// Ticket 034/123: firing [`ChunksEdited`] queues every edited chunk for
    /// a full reload, folding the borders of repeated edits to the same
    /// chunk together, and queues *no* re-mesh yet — neighbours wait for
    /// the reload to land (see [`poll_completed_chunk_reloads`]).
    #[test]
    fn queue_edited_chunk_reloads_queues_reloads_with_merged_borders_and_no_remeshes() {
        let mut app = App::new();
        app.add_event::<ChunksEdited>()
            .init_resource::<PendingChunkReloads>()
            .init_resource::<PendingChunkRemeshes>()
            .add_systems(Update, queue_edited_chunk_reloads);

        let north = ChunkBorders { north: true, ..Default::default() };
        let east = ChunkBorders { east: true, ..Default::default() };
        app.world_mut().send_event(ChunksEdited(vec![((0, 0), north), ((0, 1), ChunkBorders::default())]));
        app.world_mut().send_event(ChunksEdited(vec![((0, 0), east)]));
        app.update();

        let reloads = &app.world().resource::<PendingChunkReloads>().0;
        assert_eq!(reloads.len(), 2);
        assert_eq!(reloads[&(0, 0)], ChunkBorders { north: true, east: true, ..Default::default() });
        assert_eq!(reloads[&(0, 1)], ChunkBorders::default());

        assert!(app.world().resource::<PendingChunkRemeshes>().0.is_empty(), "neighbours are queued on completion, not here");
    }

    /// Ticket 123: [`ChunksEdited::from_report`] pairs each chunk with its
    /// borders, and reads every border as touched for a report that has
    /// none recorded.
    #[test]
    fn chunks_edited_from_report_pairs_chunks_with_borders_and_defaults_to_all() {
        let west = ChunkBorders { west: true, ..Default::default() };
        let report = EditReport {
            chunks: vec![(0, 0), (3, 4)],
            borders: vec![west],
            ..Default::default()
        };
        let edited = ChunksEdited::from_report(&report);
        assert_eq!(edited.0, vec![((0, 0), west), ((3, 4), ChunkBorders::ALL)]);
    }

    /// Ticket 123: the throttle admits a chunk's first reload immediately,
    /// refuses another inside `min_interval`, and admits again once it has
    /// passed. Other chunks are unaffected.
    #[test]
    fn reload_throttle_admits_once_per_interval_per_chunk() {
        use std::time::Duration;
        let mut throttle = ChunkReloadThrottle { min_interval: Duration::from_millis(250), ..Default::default() };

        assert!(throttle.admit((0, 0), Duration::from_millis(0)));
        assert!(!throttle.admit((0, 0), Duration::from_millis(100)), "inside the interval");
        assert!(throttle.admit((1, 0), Duration::from_millis(100)), "another chunk has its own slot");
        assert!(throttle.admit((0, 0), Duration::from_millis(250)), "interval elapsed");
        assert!(!throttle.admit((0, 0), Duration::from_millis(300)));
        assert_eq!(throttle.last_dispatch.len(), 2);
        // Entries the interval has expired are pruned on the next pass.
        assert!(throttle.admit((9, 9), Duration::from_millis(10_000)));
        assert_eq!(throttle.last_dispatch.len(), 1);
    }

    /// A [`ChunkColumn`] with nothing in it, at `floor_y` — enough for
    /// [`poll_completed_chunk_reloads`]'s bookkeeping, which never looks at
    /// blocks.
    fn empty_column(coord: (i32, i32), floor_y: i32) -> ChunkColumn {
        ChunkColumn { x: coord.0, z: coord.1, sections: Vec::new(), floor_y }
    }

    /// An `App` with just what [`poll_completed_chunk_reloads`] reads and
    /// writes, `columns` already in [`DecodedWorld`], and one finished
    /// reload of `coord` (yielding `result`, borders `borders`) waiting to
    /// be polled.
    fn reload_poll_app(
        columns: &[((i32, i32), i32)],
        coord: (i32, i32),
        borders: ChunkBorders,
        result: Option<ChunkLoadResult>,
        also_in_flight: &[(i32, i32)],
    ) -> App {
        use bevy::tasks::{AsyncComputeTaskPool, TaskPool};
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);

        let mut app = App::new();
        app.add_plugins(bevy::asset::AssetPlugin::default())
            .init_asset::<Mesh>()
            .init_resource::<InFlightChunkReloads>()
            .init_resource::<PendingChunkReloads>()
            .init_resource::<PendingChunkRemeshes>()
            .init_resource::<SpawnedChunkEntities>()
            .init_resource::<ChunkReloadBudget>()
            .insert_resource(TerrainMaterial(Handle::default()))
            .add_systems(Update, poll_completed_chunk_reloads);

        let mut world = DecodedWorld {
            registry: Arc::new(Mutex::new(BlockRegistry::new())),
            biomes: Arc::new(Mutex::new(world::BiomeRegistry::new())),
            columns: HashMap::new(),
        };
        for &(c, floor_y) in columns {
            world.columns.insert(c, empty_column(c, floor_y));
        }
        app.insert_resource(world);

        let mut in_flight = app.world_mut().resource_mut::<InFlightChunkReloads>();
        let task = pool.spawn(async move { result });
        // Finished tasks poll ready on the first try.
        in_flight.0.insert(coord, InFlightReload { task, borders });
        for &other in also_in_flight {
            let task = pool.spawn(std::future::pending::<Option<ChunkLoadResult>>());
            in_flight.0.insert(other, InFlightReload { task, borders: ChunkBorders::default() });
        }
        app
    }

    /// Ticket 123: once a reload lands, only the loaded neighbours across
    /// the borders the edit wrote on are queued for a re-mesh — an unloaded
    /// one and one across an untouched border are left alone.
    #[test]
    fn reload_completion_remeshes_only_loaded_neighbours_across_touched_borders() {
        let borders = ChunkBorders { north: true, east: true, ..Default::default() };
        let fresh = ChunkLoadResult { coord: (0, 0), column: empty_column((0, 0), 0), mesh: None };
        // North (0, -1) and west (-1, 0) loaded; east (1, 0) is not.
        let mut app = reload_poll_app(&[((0, 0), 0), ((0, -1), 0), ((-1, 0), 0)], (0, 0), borders, Some(fresh), &[]);

        // A finished task may need the pool a moment; poll until it lands.
        for _ in 0..100 {
            app.update();
            if app.world().resource::<InFlightChunkReloads>().0.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let remeshes = &app.world().resource::<PendingChunkRemeshes>().0;
        assert_eq!(*remeshes, HashSet::from([(0, -1)]), "north is touched and loaded; east touched but unloaded; west untouched");
        assert!(app.world().resource::<PendingChunkReloads>().0.is_empty());
    }

    /// Ticket 123: a neighbour whose own reload is still in flight is
    /// re-queued for a reload (with no borders of its own) rather than
    /// re-meshed against a column that hasn't landed.
    #[test]
    fn reload_completion_requeues_a_neighbour_that_is_itself_reloading() {
        let borders = ChunkBorders { south: true, ..Default::default() };
        let fresh = ChunkLoadResult { coord: (0, 0), column: empty_column((0, 0), 0), mesh: None };
        let mut app = reload_poll_app(&[((0, 0), 0), ((0, 1), 0)], (0, 0), borders, Some(fresh), &[(0, 1)]);

        for _ in 0..100 {
            app.update();
            if app.world().resource::<InFlightChunkReloads>().0.len() == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        assert!(app.world().resource::<PendingChunkRemeshes>().0.is_empty());
        let reloads = &app.world().resource::<PendingChunkReloads>().0;
        assert_eq!(reloads.get(&(0, 1)), Some(&ChunkBorders::default()));
    }

    /// Ticket 123: a reload that moved the column's render floor re-meshes
    /// every loaded neighbour, whatever borders the edit wrote on — the
    /// floor culls a neighbour's faces along the whole shared side.
    #[test]
    fn reload_completion_remeshes_all_neighbours_when_the_render_floor_moved() {
        let fresh = ChunkLoadResult { coord: (0, 0), column: empty_column((0, 0), 16), mesh: None };
        let all_four = [((0, 0), 0), ((0, -1), 0), ((0, 1), 0), ((1, 0), 0), ((-1, 0), 0)];
        let mut app = reload_poll_app(&all_four, (0, 0), ChunkBorders::default(), Some(fresh), &[]);

        for _ in 0..100 {
            app.update();
            if app.world().resource::<InFlightChunkReloads>().0.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let remeshes = &app.world().resource::<PendingChunkRemeshes>().0;
        assert_eq!(*remeshes, HashSet::from([(0, -1), (0, 1), (1, 0), (-1, 0)]));
    }
}
