//! Commit (ticket 048, roadmap E4): turns a valid ghost preview into a real
//! building — a [`state::City`] entry, blocks applied to the shared
//! [`RegionCache`] (see "Applied to memory, not written to disk" below), and
//! a journal entry carrying the as-built baseline (roadmap I1, which "ships
//! with E4, in iteration 1" per the roadmap).
//!
//! ## Recomputing, not reusing, the ghost's answer
//!
//! [`try_commit_placement`] calls [`placement::resolve_placement`] itself —
//! the same function ticket 047's ghost preview reads every frame — rather
//! than trusting whatever the ghost displayed last frame. A click has to
//! commit *exactly* what's on screen at the moment of the click, with this
//! frame's [`picking::HoveredBlock`] and [`placement::PlacementSelection`]
//! (rotation, and ticket 048's own `y_offset`), not a stale answer from
//! whenever the ghost system last ran.
//!
//! ## Applied to memory, not written to disk (ticket 051)
//!
//! [`apply_building_edit`] mutates the shared [`RegionCache`] and leaves the
//! touched region(s) dirty — it does **not** open a
//! [`crate::edit::session::WriteSession`] and does **not** touch disk.
//! Ticket 048 originally had every commit open its own session and save
//! immediately, which meant a street of houses rewrote a whole `.mca` file
//! per house; ticket 051 deferred the actual write to a manual Save
//! (`city::save`), which flushes every dirty region at once via
//! [`crate::edit::session::WriteSession::flush`]. The live mesh still
//! updates immediately — [`ChunksEdited`] fires off the in-memory apply, the
//! same as before — only the disk write is deferred.
//!
//! ## Synchronous city entry, asynchronous apply
//!
//! [`state::City::place_building`] is cheap (an occupancy check over a
//! `HashMap`, no I/O) and runs the instant a click is accepted — the tile
//! claim exists before the apply starts, which is what stops a second click
//! on the same spot from racing the first. [`apply_building_edit`] still
//! runs on [`AsyncComputeTaskPool`] (a region not yet resident in the cache
//! is a disk read), the same shape `viewer::paint`'s
//! `start_paint`/`poll_paint` established for W8. [`poll_commit`]
//! is this ticket's mirror of `poll_paint`: on success it journals the
//! baseline and fires [`ChunksEdited`]; on failure it calls
//! [`state::City::remove_building`], which is the "transactionally" half of
//! the roadmap's own wording for E4 — the synchronous entry doesn't survive
//! an apply that didn't happen.
//!
//! ## `blueprint_edit`: air is written, not skipped
//!
//! [`crate::edit::WorldEdit`]'s own docs already name this as an E4
//! decision to make: a building's declared-empty interior (a doorway, the
//! space above a floor) writes `minecraft:air` over whatever was there,
//! rather than leaving it standing. That's also what clears whatever terrain
//! [`grid::fit_footprint`]'s lowest-point `base_y` leaves poking into a
//! footprint on uneven ground — since ticket 058 removed the old refusal on
//! steep ground, that sliver can now be a good deal more than one block on a
//! genuinely rough site, and there is still no second pass here that goes
//! looking for it: the blueprint's own bottom layer already covers whatever
//! there is. [`blueprint_edit`] mirrors [`crate::blueprint::mesh_blueprint`]'s own
//! `dy*sz*sx + dz*sx + dx` indexing (see `blueprint::mesh`'s `block_at`) so
//! a build and the mesh that previewed it never disagree about which corner
//! is which.
//!
//! ## One commit in flight at a time
//!
//! [`CommitState::pending`] is a single slot, the same backpressure
//! [`crate::blueprint::BlueprintExtraction`]/`viewer::paint::PaintCommand`
//! already use. [`try_commit_placement`] simply does nothing while a commit
//! is pending; there's no build-menu affordance yet to disable, the same "no
//! UI beyond what already exists" state ticket 047 left this whole feature
//! area in.
//!
//! Before ticket 051, that slot alone only ruled out a second *commit* —
//! `city::demolish` opened its own `WriteSession` the same way, on its own
//! tiles, and the two modules knew nothing about each other's `pending`, so
//! a shared `WriteGate` serialized their session locks. Neither module opens
//! a session per edit any more (see "Applied to memory, not written to
//! disk" above), so that race no longer exists — the shared
//! `Arc<Mutex<RegionCache>>` still serializes concurrent applies correctly
//! on its own, and `WriteGate` was removed.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::blueprint::{self, Blueprint, BuildingCatalogue, Rotation};
use crate::camera;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;
use crate::DecodedWorld;

use super::journal::{self, Journal};
use super::picking::{HoveredBlock, PickingSet};
use super::placement::{self, GhostPlacement, PlacementSelection};
use super::state::{self, BuildingId, PlacedBuilding};
use super::tool::ActiveTool;
use super::write_status::{WriteKind, WriteStatus};

/// A commit's write, in flight — see the module docs.
struct PendingCommit {
    building: BuildingId,
    placement: PlacedBuilding,
    /// Kept alongside the task so [`poll_commit`] can build the baseline
    /// ([`journal::Baseline::capture`] needs the edit *and* the report it
    /// produced) without recomputing it from the blueprint a second time.
    edit: WorldEdit,
    task: Task<Result<EditReport, EditRefusal>>,
}

/// One commit at a time — see the module docs.
#[derive(Resource, Default)]
struct CommitState {
    pending: Option<PendingCommit>,
}

pub struct CommitPlugin;

impl Plugin for CommitPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CommitState>()
            // `city::demolish::DemolishPlugin`/`city::undo::UndoPlugin`
            // initialize the same resource — `init_resource` only inserts a
            // default when one isn't already present, so it doesn't matter
            // which plugin adds to the app first. See `write_status`'s
            // module docs.
            .init_resource::<WriteStatus>()
            // Registered here rather than assumed from `ChunkLoadPipelinePlugin`
            // — `add_event` is idempotent, the same defensive call
            // `viewer::paint::PaintPlugin` makes, and it's what lets a
            // standalone test app spawn `CommitPlugin` on its own.
            .add_event::<ChunksEdited>()
            // After `PickingSet` for the same reason ticket 047's ghost
            // preview orders there — `try_commit_placement` needs *this*
            // frame's `HoveredBlock`, not last frame's.
            .add_systems(Update, (try_commit_placement, poll_commit).chain().after(PickingSet));
    }
}

/// Every grid position in `blueprint`, in Minecraft world coordinates at
/// `origin` — one [`WorldEdit::set`] per position, air included (see the
/// module docs). `origin` is `blueprint`-local `(0, 0, 0)`; the caller is
/// responsible for it already being wherever the rotated blueprint should
/// sit ([`placement::resolve_placement`]'s `origin`, not the raw hovered
/// block).
///
/// `pub(super)`: `city::road_build` (ticket 055, roadmap F2/F3) reuses this
/// verbatim for a road piece's own blueprint — building a `WorldEdit` from a
/// rotated blueprint at an origin doesn't care whether the blueprint is a
/// building or a road piece.
pub(super) fn blueprint_edit(blueprint: &Blueprint, origin: IVec3) -> WorldEdit {
    let (sx, sy, sz) = (blueprint.size.x, blueprint.size.y, blueprint.size.z);
    let mut edit = WorldEdit::new().with_data_version(blueprint.data_version);

    for dy in 0..sy {
        for dz in 0..sz {
            for dx in 0..sx {
                let index = (dy as usize) * (sz as usize) * (sx as usize) + (dz as usize) * (sx as usize) + (dx as usize);
                let Some(&palette_index) = blueprint.blocks.get(index) else { continue };
                let Some(state) = blueprint.palette.get(palette_index as usize) else { continue };
                edit.set(origin + IVec3::new(dx, dy, dz), state.clone());
            }
        }
    }

    edit
}

/// Applies `edit` to `cache` in memory — no [`crate::edit::session::WriteSession`],
/// no disk write; see the module docs' "Applied to memory, not written to
/// disk". Pulled out on its own the same way `viewer::paint::commit_fill`
/// is, so it's callable directly from a test rather than only through a real
/// `AsyncComputeTaskPool` task. `pub(super)`: `city::demolish` and
/// `city::undo` reuse this verbatim for their own edits — applying an edit
/// to the cache doesn't care which direction it's going or why.
pub(super) fn apply_building_edit(cache: &mut RegionCache, edit: &WorldEdit, policy: &EditPolicy) -> Result<EditReport, EditRefusal> {
    crate::edit::apply_routed(edit, cache, policy)
}

/// Left-click on a valid placement: claims the tile in [`state::City`]
/// synchronously, then dispatches the actual write onto
/// [`AsyncComputeTaskPool`] — see the module docs.
#[allow(clippy::too_many_arguments)]
fn try_commit_placement(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_input: Res<camera::EguiInputCapture>,
    selection: Res<PlacementSelection>,
    hovered: Res<HoveredBlock>,
    catalogue: Option<Res<BuildingCatalogue>>,
    world: Res<DecodedWorld>,
    mut city: ResMut<state::City>,
    mut commit: ResMut<CommitState>,
    region_cache: Option<Res<SharedRegionCache>>,
    tool: Option<Res<ActiveTool>>,
) {
    // Ticket 055, roadmap F2: a left click while the road tool is active is
    // `city::road_build`'s to react to, not this. `Option` and a default of
    // `Building` — see `tool`'s module docs — so a minimal test `App` that
    // never adds `tool::ToolPlugin` keeps committing exactly as it did before
    // this ticket.
    if !matches!(tool.as_deref(), None | Some(ActiveTool::Building)) {
        return;
    }
    if commit.pending.is_some() || egui_input.pointer || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(id) = selection.catalogue_id.clone() else { return };
    let Some(catalogue) = catalogue else { return };
    let Some(entry) = catalogue.get(&id) else { return };
    let Some(hovered) = hovered.0 else { return };

    let GhostPlacement { origin, valid } =
        placement::resolve_placement(hovered, entry.footprint, selection.rotation, selection.y_offset, &world, &city);
    if !valid {
        return;
    }

    // `Deg0` never touches `rotate_blueprint` — the identity case can't fail
    // on a property nothing recognises, mirroring `placement::ghost_mesh`'s
    // own shortcut, and it avoids cloning a blueprint that can run into the
    // millions of blocks.
    let rotated;
    let blueprint = if selection.rotation == Rotation::Deg0 {
        &entry.blueprint
    } else {
        match blueprint::rotate_blueprint(&entry.blueprint, selection.rotation) {
            Ok(b) => {
                rotated = b;
                &rotated
            }
            Err(err) => {
                println!("block_viewer: can't place {id}: {err}");
                return;
            }
        }
    };

    let Some(region_cache) = region_cache else {
        println!("block_viewer: can't place a building: no save is loaded");
        return;
    };

    let edit = blueprint_edit(blueprint, origin);
    if edit.is_empty() {
        // An empty blueprint (or an empty catalogue entry, which 039's
        // loader already refuses) isn't reachable in practice, but an empty
        // `WorldEdit` is itself refused by the write path — better to say
        // nothing than to spawn a task doomed to fail on `EditRefusal::Empty`.
        return;
    }

    let building = match city.place_building(id.clone(), origin, selection.rotation, entry.footprint) {
        Ok(building) => building,
        Err(err) => {
            println!("block_viewer: placement refused: {err}");
            return;
        }
    };
    let placed = PlacedBuilding { definition: id, origin, rotation: selection.rotation, footprint: entry.footprint };

    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
    let task_edit = edit.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        apply_building_edit(&mut cache, &task_edit, &policy)
    });

    commit.pending = Some(PendingCommit { building, placement: placed, edit, task });
}

/// Single non-blocking poll of the in-flight commit, the same
/// `block_on(poll_once(..))` pattern `viewer::paint::poll_paint` uses. On
/// success: journals the as-built baseline and fires [`ChunksEdited`] (W7).
/// On failure: rolls the synchronous [`state::City::place_building`] back —
/// the "transactionally" half of the roadmap's own wording for E4.
fn poll_commit(
    mut commit: ResMut<CommitState>,
    mut city: ResMut<state::City>,
    mut journal: ResMut<Journal>,
    mut write_status: ResMut<WriteStatus>,
    mut edited: EventWriter<ChunksEdited>,
) {
    let result = {
        let Some(pending) = &mut commit.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still applying.
        };
        result
    };
    let PendingCommit { building, placement, edit, .. } = commit.pending.take().expect("just matched Some above");

    match result {
        Ok(report) => {
            let blocks = report.blocks_written;
            let chunks = report.chunks.len();
            println!(
                "block_viewer: placed {} ({blocks} block(s) across {chunks} chunk(s), not yet saved to disk)",
                placement.definition
            );
            // `EditPolicy::capture_replaced` was on, so `report.replaced` is
            // `Some` and this always succeeds — the `if let` is the same
            // defensive shape `Baseline::capture`'s own doc comment expects
            // of a caller, not a case this path expects to actually miss.
            if let Some(baseline) = journal::Baseline::capture(&edit, &report) {
                journal.record_placement(building, placement.clone(), baseline);
            }
            write_status.record_success(WriteKind::Placed, placement.definition, &report);
            edited.send(ChunksEdited(report.chunks));
        }
        Err(err) => {
            city.remove_building(building);
            println!("block_viewer: placement of {} failed, rolled back: {err}", placement.definition);
            write_status.record_failure(WriteKind::Placed, placement.definition, err.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::BlockState;
    use crate::region_cache::RegionCache as Cache;
    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use mc_anvil::SaveMeta as Meta;
    use rnbt::{NbtField, NbtList, NbtValue};

    // --- blueprint_edit ----------------------------------------------------

    fn state_named(name: &str) -> BlockState {
        name.parse().unwrap()
    }

    /// A 2x2x1 blueprint: air at `(0,0,0)` (palette index 0, per every real
    /// extraction/structure read), stone everywhere else — small enough to
    /// enumerate by hand in an assertion.
    fn small_blueprint() -> Blueprint {
        Blueprint {
            size: IVec3::new(2, 1, 2),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state_named("minecraft:stone")],
            blocks: vec![0, 1, 1, 1], // dz-major, dx-minor at dy=0: (0,0,0) air, rest stone
            data_version: 4438,
            failed_columns: 0,
        }
    }

    #[test]
    fn blueprint_edit_writes_every_position_air_included() {
        let blueprint = small_blueprint();
        let edit = blueprint_edit(&blueprint, IVec3::new(10, 64, 10));

        assert_eq!(edit.len(), 4, "every grid position, air included");
        assert_eq!(edit.data_version(), Some(4438));

        let by_pos: std::collections::HashMap<IVec3, &BlockState> =
            edit.edits().iter().map(|e| (e.at, &e.state)).collect();
        assert_eq!(by_pos[&IVec3::new(10, 64, 10)].name, "minecraft:air");
        assert_eq!(by_pos[&IVec3::new(11, 64, 10)].name, "minecraft:stone");
        assert_eq!(by_pos[&IVec3::new(10, 64, 11)].name, "minecraft:stone");
        assert_eq!(by_pos[&IVec3::new(11, 64, 11)].name, "minecraft:stone");
    }

    #[test]
    fn blueprint_edit_offsets_every_position_by_the_origin() {
        let blueprint = small_blueprint();
        let edit = blueprint_edit(&blueprint, IVec3::new(0, 0, 0));
        let positions: std::collections::HashSet<IVec3> = edit.edits().iter().map(|e| e.at).collect();
        assert_eq!(
            positions,
            std::collections::HashSet::from([
                IVec3::new(0, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, 0, 1),
                IVec3::new(1, 0, 1),
            ])
        );
    }

    // --- apply_building_edit: the actual apply, tested directly and synchronously -

    /// A single-region, single-chunk fixture save — a slimmed-down copy of
    /// `viewer::paint::tests::Fixture`, which this module can't reach since
    /// it's private to that one.
    struct Fixture {
        dir: std::path::PathBuf,
        meta: Meta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let dir = std::env::temp_dir().join(format!("block_viewer-commit-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk()));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = Meta { name: "commit-fixture".to_string(), path: dir.clone(), region_dir, regions: vec![(0, 0)] };
            Self { dir, meta }
        }

        fn cache(&self) -> Cache {
            Cache::new(self.meta.clone(), 4)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A finished chunk with one all-stone section at `Y = 0` (world Y
    /// 0..15), `DataVersion` 4438 — the same shape `viewer::paint::tests::full_chunk`
    /// builds, copied for the reason [`Fixture`]'s own docs give.
    fn full_chunk() -> NbtField {
        let palette = NbtList::Compound(vec![NbtField::new_compound("", vec![NbtField::new_string("Name", "minecraft:stone")])]);
        let section = NbtField::new_compound(
            "",
            vec![
                NbtField { name: "Y".to_string(), value: NbtValue::Byte(0) },
                NbtField::new_compound("block_states", vec![NbtField::new_list("palette", palette)]),
            ],
        );
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_list("sections", NbtList::Compound(vec![section])),
                NbtField::new_i32("xPos", 0),
                NbtField::new_i32("zPos", 0),
                NbtField::new_i32("yPos", -4),
                NbtField::new_i32("DataVersion", 4438),
                NbtField::new_string("Status", "minecraft:full"),
                NbtField { name: "isLightOn".to_string(), value: NbtValue::Byte(1) },
            ],
        )
    }

    fn block_name_at(cache: &mut Cache, at: IVec3) -> String {
        let address = crate::edit::address_of(at);
        cache
            .get_or_load(address.region)
            .expect("resident")
            .get_block(address.local_x, address.y, address.local_z)
            .expect("a populated chunk")
            .get_string("Name")
            .expect("a palette entry")
            .clone()
    }

    #[test]
    fn apply_building_edit_writes_the_blueprint_and_reports_a_baseline() {
        let fixture = Fixture::new("write");
        let mut cache = fixture.cache();
        let blueprint = Blueprint {
            size: IVec3::new(2, 1, 2),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state_named("minecraft:dirt")],
            blocks: vec![1, 1, 1, 1],
            data_version: 4438,
            failed_columns: 0,
        };
        let edit = blueprint_edit(&blueprint, IVec3::new(1, 5, 1));
        let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };

        let report = apply_building_edit(&mut cache, &edit, &policy).expect("a valid placement");
        assert_eq!(report.blocks_written, 4);
        assert_eq!(cache.dirty_regions().count(), 1, "the edit lands in memory only — no save call here");

        for at in [IVec3::new(1, 5, 1), IVec3::new(2, 5, 1), IVec3::new(1, 5, 2), IVec3::new(2, 5, 2)] {
            assert_eq!(block_name_at(&mut cache, at), "minecraft:dirt");
        }

        let baseline = journal::Baseline::capture(&edit, &report).expect("capture_replaced was on");
        assert_eq!(baseline.written.len(), 4);
        assert_eq!(baseline.previous.len(), 4);
        assert!(baseline.previous.iter().all(|(_, s)| s.name == "minecraft:stone"), "the ground the building overwrote");
    }

    #[test]
    fn apply_building_edit_refuses_ungenerated_terrain_and_writes_nothing() {
        let fixture = Fixture::new("refuse");
        let mut cache = fixture.cache();
        let blueprint = small_blueprint();
        // Chunk (5, 0) is outside the fixture's one generated chunk.
        let edit = blueprint_edit(&blueprint, IVec3::new(5 * 16, 5, 0));
        let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };

        let err = apply_building_edit(&mut cache, &edit, &policy).unwrap_err();
        assert!(matches!(err, EditRefusal::ChunkNotGenerated { chunk: (5, 0) }));
        assert_eq!(cache.dirty_regions().count(), 0);
    }

    #[test]
    fn apply_building_edit_accepts_a_region_an_earlier_placement_already_dirtied() {
        // The whole point of ticket 051's `allow_dirty_regions: true`: a
        // second placement in the same region as an unsaved first one must
        // not be refused with `RegionHasUnsavedChanges`.
        let fixture = Fixture::new("batch");
        let mut cache = fixture.cache();
        let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };

        let first = blueprint_edit(&small_blueprint(), IVec3::new(1, 5, 1));
        apply_building_edit(&mut cache, &first, &policy).expect("first placement");
        assert_eq!(cache.dirty_regions().count(), 1);

        let second = blueprint_edit(&small_blueprint(), IVec3::new(4, 5, 4));
        let report = apply_building_edit(&mut cache, &second, &policy).expect("second placement, same region");
        assert_eq!(report.blocks_written, 4);
        assert_eq!(cache.dirty_regions().count(), 1, "still one dirty region, now carrying both edits");
    }

    // --- poll_commit: the City/journal glue, plumbing tests -------------------
    //
    // Same split `viewer::paint`'s own tests use: `apply_building_edit` (above)
    // is tested directly and synchronously against a real fixture; `poll_commit`
    // is tested through a real `App` with a task whose result is fixed ahead
    // of time, the same way `viewer::paint::tests::a_finished_commit_reports_done_and_fires_chunks_edited`
    // avoids needing a second real region-file fixture just to prove the
    // transition logic.

    use bevy::tasks::TaskPool;

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn commit_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(CommitPlugin)
            .insert_resource(state::City::default())
            .insert_resource(Journal::default());
        app
    }

    /// Ticks `app` until `CommitState` is no longer pending, or gives up —
    /// same reasoning as `viewer::paint::tests::run_until_settled`: a task
    /// spawned directly onto `AsyncComputeTaskPool` can finish on its worker
    /// thread before or after any particular `app.update()`.
    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if app.world().resource::<CommitState>().pending.is_none() {
                return;
            }
        }
        panic!("commit never settled");
    }

    fn a_placement() -> PlacedBuilding {
        PlacedBuilding { definition: "house01".to_string(), origin: IVec3::new(0, 64, 0), rotation: Rotation::Deg0, footprint: IVec2::new(2, 2) }
    }

    #[test]
    fn poll_commit_success_records_the_baseline_and_fires_chunks_edited() {
        let mut app = commit_test_app();
        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), state_named("minecraft:stone"));
        let task_edit = edit.clone();
        let report = EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(vec![(IVec3::new(0, 64, 0), BlockState::air())]),
        };
        let task = pool().spawn(async move { Ok(report) });

        app.world_mut().resource_mut::<CommitState>().pending =
            Some(PendingCommit { building, placement: a_placement(), edit: task_edit, task });

        run_until_settled(&mut app);

        assert!(app.world().resource::<CommitState>().pending.is_none());
        let journal = app.world().resource::<Journal>();
        assert_eq!(journal.len(), 1, "a successful write journals the baseline");
        assert_eq!(journal.placement_baseline(building).unwrap().written, edit.edits().iter().map(|e| (e.at, e.state.clone())).collect::<Vec<_>>());

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].0, vec![(0, 0)]);

        // The city entry the click made synchronously survives a successful
        // write untouched.
        assert!(!app.world().resource::<state::City>().is_tile_free(IVec2::new(0, 0)));
    }

    #[test]
    fn poll_commit_failure_rolls_back_the_city_entry_and_journals_nothing() {
        let mut app = commit_test_app();
        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        let task = pool().spawn(async { Err(EditRefusal::Empty) });
        app.world_mut().resource_mut::<CommitState>().pending =
            Some(PendingCommit { building, placement: a_placement(), edit: WorldEdit::new(), task });

        run_until_settled(&mut app);

        let city = app.world().resource::<state::City>();
        assert!(city.is_empty(), "the synchronous place_building must not survive a failed write");
        assert!(city.is_tile_free(IVec2::new(0, 0)), "the tile claim is released along with the entry");
        assert!(app.world().resource::<Journal>().is_empty(), "nothing to journal for a write that never happened");

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "nothing changed in the world, so nothing needs re-meshing");
    }
}
