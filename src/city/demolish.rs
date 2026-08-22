//! Demolish (ticket 049, roadmap E5): the inverse of E4's commit — removes a
//! placed building from [`City`] and restores the terrain its own
//! *placement* baseline (roadmap I1) says stood there before it, applied
//! back to the shared region cache (W4/W5; see `city::commit`'s module docs'
//! "Applied to memory, not written to disk" for why this no longer touches
//! disk itself, since ticket 051). Also lands [`Journal::record_demolition`]'s
//! first real caller, and with it the `Demolished` half of D3's journal
//! (ticket 044) — undo can now reverse either direction.
//!
//! ## Finding the target off the occupancy grid, not a menu
//!
//! No G1 build menu, and no new selection concept: `Delete` demolishes
//! whichever building [`picking::HoveredBlock`]'s tile currently belongs to
//! ([`City::occupant_at`]), the same "keyboard stand-in until G1"
//! role ticket 047's number keys/`R`/`Escape` and ticket 048's height keys
//! already play for *placing* one. Hovering empty ground or a road tile does
//! nothing; hovering a building with no recorded placement baseline (a save
//! from before ticket 048, or a journal that didn't round-trip) refuses with
//! a message rather than guessing what terrain to put back — see
//! [`resolve_demolition_target`].
//!
//! ## The apply doesn't need to read the world first
//!
//! Demolition's restoring edit is exactly the *placement*'s own baseline
//! `previous` — the terrain that stood there before the building went in —
//! at exactly the positions that baseline recorded
//! ([`Baseline::restore_edit`]). Nothing here re-derives a volume
//! from the blueprint or walks the footprint again; the placement baseline
//! already *is* the answer. What the apply *does* still capture
//! (`EditPolicy::capture_replaced`) is what the building's blocks actually
//! were at the moment of demolition — not re-derived from the blueprint
//! either, so a building that was already damaged demolishes, and undoes, as
//! what it actually was. That's [`Baseline::capture`]'s ordinary
//! job, called here exactly like `city::commit::poll_commit` calls it, just
//! reading the *other* half of the baseline it produces.
//!
//! ## Removed from `City` only *after* the apply succeeds — the mirror image
//! of commit's ordering
//!
//! `city::commit`'s [`City::place_building`] runs *before* its apply
//! starts, because a tile has to read occupied the instant a click lands, or
//! a second click could claim the same tile while the first apply is still
//! in flight. Demolition has the opposite problem: if [`try_demolish`] freed
//! the tile immediately, a placement could land on it while the restoring
//! apply was still in flight, and whichever of the two applies landed
//! last would silently clobber the other's blocks — the tile has to keep
//! reading occupied for exactly as long as *something* still intends to
//! write there. So [`City::remove_building`] is called from
//! [`poll_demolish`], only once the restoring apply has actually succeeded,
//! not from [`try_demolish`] at all. A failed apply therefore needs no
//! rollback on the `City` side — nothing was mutated there yet — which is
//! also why this module's failure path is shorter than commit's.
//!
//! ## One demolition in flight at a time
//!
//! [`DemolishState::pending`] is this module's own single slot, the same
//! backpressure shape `city::commit::CommitState` uses. Before ticket 051 a
//! shared `WriteGate` also ruled out racing a concurrent `city::commit` over
//! the save's `session.lock` — see `city::commit`'s module docs for why
//! neither module opens a session per edit any more, and why the shared
//! `Arc<Mutex<RegionCache>>` alone is now enough.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::camera;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;

use super::commit::apply_building_edit;
use super::drops::DropTable;
use super::inventory::Stock;
use super::journal::{Baseline, Journal, Ledger};
use super::picking::{HoveredBlock, PickingSet};
use super::state::{BuildingId, City, Occupant, PlacedBuilding};
use super::write_status::{WriteKind, WriteStatus};

/// A demolition's write, in flight — see the module docs.
struct PendingDemolition {
    building: BuildingId,
    /// The building as [`City`] still has it — `City` isn't touched
    /// until [`poll_demolish`] sees this succeed, so there's nothing to look
    /// back up there once it does.
    placement: PlacedBuilding,
    /// Kept alongside the task so [`poll_demolish`] can build the
    /// demolition's own baseline ([`Baseline::capture`] needs the
    /// edit *and* the report it produced) without recomputing it.
    edit: WorldEdit,
    task: Task<Result<EditReport, EditRefusal>>,
}

/// One demolition at a time — see the module docs.
#[derive(Resource, Default)]
struct DemolishState {
    pending: Option<PendingDemolition>,
}

pub struct DemolishPlugin;

impl Plugin for DemolishPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DemolishState>()
            // Ticket 073 — same `init_resource` reasoning `city::commit`'s
            // own three carry.
            .init_resource::<Stock>()
            .init_resource::<DropTable>()
            // `city::commit::CommitPlugin`/`city::undo::UndoPlugin`
            // initialize the same resource — `init_resource` only inserts a
            // default when one isn't already present. See `write_status`'s
            // module docs.
            .init_resource::<WriteStatus>()
            .add_event::<ChunksEdited>()
            // After `PickingSet` for the same reason ticket 047's ghost
            // preview and ticket 048's commit both order there —
            // `try_demolish` needs *this* frame's `HoveredBlock`.
            .add_systems(Update, (try_demolish, poll_demolish).chain().after(PickingSet));
    }
}

/// What [`try_demolish`] found at `hovered`'s tile, and why — see the module
/// docs' "Finding the target off the occupancy grid".
enum DemolitionTarget {
    /// The tile is empty, or held by a road. Nothing to demolish, and
    /// nothing worth a console line about it.
    Nothing,
    /// A building is there, but the journal has no placement baseline for
    /// it — see the module docs.
    NoBaseline { building: BuildingId },
    /// A building is there, with a baseline to restore from.
    Found {
        building: BuildingId,
        placement: PlacedBuilding,
        baseline: Baseline,
    },
}

/// Resolves what `Delete` should demolish at `hovered`'s `(x, z)` tile — a
/// plain function, not a system, so it's callable directly from a test
/// against a bare [`City`]/[`Journal`] with no `App` involved, the same split
/// `city::commit`'s `blueprint_edit`/`apply_building_edit` use.
fn resolve_demolition_target(hovered: IVec3, city: &City, journal: &Journal) -> DemolitionTarget {
    let tile = IVec2::new(hovered.x, hovered.z);
    let Some(Occupant::Building(building)) = city.occupant_at(tile) else {
        return DemolitionTarget::Nothing;
    };
    let Some(baseline) = journal.placement_baseline(building) else {
        return DemolitionTarget::NoBaseline { building };
    };
    // `occupant_at` just found this exact id in this exact `City`, so the
    // lookup below can't miss — the occupancy grid and `buildings` are kept
    // in lockstep by every mutator in `state.rs`.
    let placement = city.building(building).cloned().expect("occupant_at found this building in the same City");
    DemolitionTarget::Found { building, placement, baseline: baseline.clone() }
}

/// `Delete` on a hovered building: dispatches its restoring apply onto
/// [`AsyncComputeTaskPool`] — see the module docs for why [`City`]
/// itself isn't touched here at all.
fn try_demolish(
    keys: Res<ButtonInput<KeyCode>>,
    egui_input: Res<camera::EguiInputCapture>,
    hovered: Res<HoveredBlock>,
    city: Res<City>,
    journal: Res<Journal>,
    mut demolish: ResMut<DemolishState>,
    region_cache: Option<Res<SharedRegionCache>>,
) {
    // Same guard `camera.rs`'s own input systems use — a keystroke egui is
    // already handling shouldn't also drive the game underneath it.
    if demolish.pending.is_some() || egui_input.keyboard || !keys.just_pressed(KeyCode::Delete) {
        return;
    }
    let Some(hovered) = hovered.0 else { return };

    let (building, placement, baseline) = match resolve_demolition_target(hovered, &city, &journal) {
        DemolitionTarget::Nothing => return,
        DemolitionTarget::NoBaseline { building } => {
            println!(
                "block_viewer: can't demolish building {building:?}: no placement baseline was recorded for it \
                 (a save from before ticket 048?)"
            );
            return;
        }
        DemolitionTarget::Found { building, placement, baseline } => (building, placement, baseline),
    };

    let edit = baseline.restore_edit();
    if edit.is_empty() {
        // A journaled placement with an empty baseline isn't reachable in
        // practice (`blueprint_edit`'s own "air is written, not skipped"
        // means every placement's baseline covers at least one position),
        // but an empty `WorldEdit` is refused by the write path regardless —
        // better to say nothing than spawn a task doomed to fail.
        return;
    }

    let Some(region_cache) = region_cache else {
        println!("block_viewer: can't demolish a building: no save is loaded");
        return;
    };

    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
    let task_edit = edit.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        apply_building_edit(&mut cache, &task_edit, &policy)
    });

    demolish.pending = Some(PendingDemolition { building, placement, edit, task });
}

/// Single non-blocking poll of the in-flight demolition, the same
/// `block_on(poll_once(..))` pattern `city::commit::poll_commit` uses. On
/// success: removes the building from [`City`] (only now — see the
/// module docs), journals the demolition's own baseline, and fires
/// [`ChunksEdited`] (W7). On failure: nothing to undo — `City` was never
/// touched.
fn poll_demolish(
    mut demolish: ResMut<DemolishState>,
    mut city: ResMut<City>,
    mut journal: ResMut<Journal>,
    mut write_status: ResMut<WriteStatus>,
    mut edited: EventWriter<ChunksEdited>,
    mut stock: ResMut<Stock>,
    drops: Res<DropTable>,
) {
    let result = {
        let Some(pending) = &mut demolish.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still applying.
        };
        result
    };
    let PendingDemolition { building, placement, edit, .. } = demolish.pending.take().expect("just matched Some above");

    match result {
        Ok(report) => {
            // Only now does the building actually leave `City` — the tile
            // was kept "occupied" for the entire apply, on purpose.
            city.remove_building(building);
            let blocks = report.blocks_written;
            let chunks = report.chunks.len();
            println!(
                "block_viewer: demolished {} ({blocks} block(s) restored across {chunks} chunk(s), not yet saved to disk)",
                placement.catalogue_id
            );
            // `EditPolicy::capture_replaced` was on, so this is always
            // `Some` here — see the module docs on what each half of this
            // particular baseline means for a demolition.
            if let Some(baseline) = Baseline::capture(&edit, &report) {
                // Ticket 073: the terrain this demolition put back is paid
                // for out of the stock. Without that, `place -> demolish ->
                // place` clears the same hillside over and over and hands
                // back its stone every time. Nothing is credited in return —
                // the building's own blocks are lost, not salvaged; salvage
                // is a mechanic (a fraction, a rubble state), not a rounding
                // decision, and belongs to a ticket that designs it.
                //
                // Clamped, not refused: `remove_parcel` takes what's there
                // and reports it, because a demolition is a world-state
                // change that has already landed, and blocking one for want
                // of dirt would leave city state and world unable to agree.
                let wanted = drops.parcel_for(baseline.written.iter().map(|(_, state)| state));
                let debited = stock.remove_parcel(&wanted);
                journal.record_demolition(
                    building,
                    placement.clone(),
                    baseline,
                    Ledger { credited: Default::default(), debited },
                );
            }
            write_status.record_success(WriteKind::Demolished, placement.catalogue_id, &report);
            edited.send(ChunksEdited(report.chunks));
        }
        Err(err) => {
            println!("block_viewer: demolition of {} failed, nothing was changed: {err}", placement.catalogue_id);
            write_status.record_failure(WriteKind::Demolished, placement.catalogue_id, err.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::road::RoadPieceVariant;
    use super::*;
    use crate::blueprint::{BlockState, Rotation};
    use crate::region_cache::RegionCache as Cache;
    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use mc_anvil::SaveMeta as Meta;
    use rnbt::{NbtField, NbtList, NbtValue};

    fn state_named(name: &str) -> BlockState {
        name.parse().unwrap()
    }

    fn a_placement() -> PlacedBuilding {
        PlacedBuilding {
            catalogue_id: "house01".to_string(),
            definition_id: None,
            origin: IVec3::new(0, 64, 0),
            rotation: Rotation::Deg0,
            footprint: IVec2::new(2, 2),
        }
    }

    fn a_baseline() -> Baseline {
        Baseline {
            written: vec![(IVec3::new(0, 64, 0), state_named("minecraft:stone"))],
            previous: vec![(IVec3::new(0, 64, 0), state_named("minecraft:dirt"))],
            data_version: Some(4438),
        }
    }

    // --- resolve_demolition_target ------------------------------------------

    #[test]
    fn resolve_demolition_target_is_nothing_on_empty_ground() {
        let city = City::default();
        let journal = Journal::default();
        let result = resolve_demolition_target(IVec3::new(5, 64, 5), &city, &journal);
        assert!(matches!(result, DemolitionTarget::Nothing));
    }

    #[test]
    fn resolve_demolition_target_is_nothing_on_a_road_tile() {
        let mut city = City::default();
        // Cell (0, 0) covers block tiles 0..6 x 0..6, which includes (5, 5).
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        let journal = Journal::default();
        let result = resolve_demolition_target(IVec3::new(5, 64, 5), &city, &journal);
        assert!(matches!(result, DemolitionTarget::Nothing));
    }

    #[test]
    fn resolve_demolition_target_refuses_a_building_with_no_baseline() {
        let mut city = City::default();
        let id = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1)).unwrap();
        let journal = Journal::default();

        let result = resolve_demolition_target(IVec3::new(0, 64, 0), &city, &journal);
        assert!(matches!(result, DemolitionTarget::NoBaseline { building } if building == id));
    }

    #[test]
    fn resolve_demolition_target_finds_a_journaled_building() {
        let mut city = City::default();
        let id = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1)).unwrap();
        let mut journal = Journal::default();
        journal.record_placement(id, a_placement(), a_baseline(), Ledger::default());

        let result = resolve_demolition_target(IVec3::new(0, 64, 0), &city, &journal);
        let DemolitionTarget::Found { building, placement, baseline } = result else { panic!("expected Found") };
        assert_eq!(building, id);
        assert_eq!(placement.catalogue_id, "house01");
        assert_eq!(baseline, a_baseline());
    }

    // --- try_demolish / poll_demolish: through a real App -------------------
    //
    // Same split `city::commit`'s own tests use: the pure decision logic
    // (above) is tested directly; the apply itself is proven once by
    // `city::commit`'s own `apply_building_edit` tests (this module reuses
    // that exact function), so what's left to prove here is the
    // `City`/`Journal` glue, through a task whose result is fixed ahead of
    // time.

    use bevy::tasks::{AsyncComputeTaskPool, TaskPool};

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn demolish_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(DemolishPlugin).insert_resource(City::default()).insert_resource(Journal::default());
        app
    }

    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if app.world().resource::<DemolishState>().pending.is_none() {
                return;
            }
        }
        panic!("demolition never settled");
    }

    #[test]
    fn poll_demolish_success_removes_the_building_and_journals_the_restore() {
        let mut app = demolish_test_app();
        let building = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1))
            .unwrap();

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), state_named("minecraft:dirt"));
        let task_edit = edit.clone();
        let report = EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(vec![(IVec3::new(0, 64, 0), state_named("minecraft:stone"))]),
        };
        let task = pool().spawn(async move { Ok(report) });

        app.world_mut().resource_mut::<DemolishState>().pending =
            Some(PendingDemolition { building, placement: a_placement(), edit: task_edit, task });

        run_until_settled(&mut app);

        let city = app.world().resource::<City>();
        assert!(city.is_empty(), "a successful demolition removes the building");
        assert!(city.is_tile_free(IVec2::new(0, 0)), "and frees its tile");

        let journal = app.world().resource::<Journal>();
        assert_eq!(journal.len(), 1);
        let demolition = journal.entries().last().unwrap();
        assert_eq!(demolition.building(), building);
        // `written` is the restored terrain (what `edit` wrote); `previous`
        // is the building's own block, read off the world at demolition
        // time via `EditPolicy::capture_replaced` — not re-derived from a
        // blueprint.
        assert_eq!(demolition.baseline().written, vec![(IVec3::new(0, 64, 0), state_named("minecraft:dirt"))]);
        assert_eq!(demolition.baseline().previous, vec![(IVec3::new(0, 64, 0), state_named("minecraft:stone"))]);

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].0, vec![(0, 0)]);
    }

    // --- ticket 073: a demolition pays for its own backfill -----------------

    #[test]
    fn a_demolition_charges_for_the_terrain_it_puts_back() {
        // The loop this closes: placing cleared this dirt and credited it,
        // so restoring it has to take it back — otherwise `place ->
        // demolish -> place` clears the same ground over and over and pays
        // out every time.
        let mut app = demolish_test_app();
        app.world_mut().resource_mut::<Stock>().add("minecraft:dirt", 1);
        let stock_before = app.world().resource::<Stock>().clone();

        let building = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1))
            .unwrap();

        // The restoring edit writes the dirt back; `replaced` is the
        // building's own block, which is lost rather than salvaged.
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), state_named("minecraft:dirt"));
        let report = EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(vec![(IVec3::new(0, 64, 0), state_named("minecraft:oak_planks"))]),
        };
        let task_edit = edit.clone();
        let task = pool().spawn(async move { Ok(report) });
        app.world_mut().resource_mut::<DemolishState>().pending =
            Some(PendingDemolition { building, placement: a_placement(), edit: task_edit, task });

        run_until_settled(&mut app);

        let stock = app.world().resource::<Stock>();
        assert_eq!(stock.count("minecraft:dirt"), 0, "the restored terrain is paid for");
        assert_eq!(stock.count("minecraft:oak_planks"), 0, "and the building's own blocks are not salvaged");
        assert!(stock.is_empty(), "back exactly where the placement found it");
        assert_ne!(*stock, stock_before);

        let journal = app.world().resource::<Journal>();
        let ledger = journal.entries().last().unwrap().ledger();
        assert!(ledger.credited.is_empty());
        assert_eq!(ledger.debited.get("minecraft:dirt"), 1);
    }

    #[test]
    fn a_backfill_the_stock_cannot_cover_is_clamped_rather_than_refused() {
        // Demolishing is a world-state change; being short of dirt must not
        // be able to block it — see `poll_demolish`'s own comment.
        let mut app = demolish_test_app();
        let building = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1))
            .unwrap();

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), state_named("minecraft:dirt"));
        let report = EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(vec![(IVec3::new(0, 64, 0), state_named("minecraft:oak_planks"))]),
        };
        let task_edit = edit.clone();
        let task = pool().spawn(async move { Ok(report) });
        app.world_mut().resource_mut::<DemolishState>().pending =
            Some(PendingDemolition { building, placement: a_placement(), edit: task_edit, task });

        run_until_settled(&mut app);

        assert!(app.world().resource::<City>().is_empty(), "the demolition still went through");
        let ledger = app.world().resource::<Journal>().entries().last().unwrap().ledger().clone();
        assert!(ledger.debited.is_empty(), "the ledger records what was actually taken, which was nothing");
    }

    #[test]
    fn poll_demolish_failure_leaves_the_building_and_journal_untouched() {
        let mut app = demolish_test_app();
        let building = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1))
            .unwrap();

        let task = pool().spawn(async { Err(EditRefusal::Empty) });
        app.world_mut().resource_mut::<DemolishState>().pending =
            Some(PendingDemolition { building, placement: a_placement(), edit: WorldEdit::new(), task });

        run_until_settled(&mut app);

        let city = app.world().resource::<City>();
        assert!(!city.is_tile_free(IVec2::new(0, 0)), "the building must still be standing after a failed write");
        assert_eq!(city.len(), 1);
        assert!(app.world().resource::<Journal>().is_empty(), "nothing to journal for a restore that never happened");

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0);
    }

    // --- a real write, end to end: try_demolish + poll_demolish against a ---
    // --- real fixture region file --------------------------------------------

    struct Fixture {
        dir: std::path::PathBuf,
        meta: Meta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let dir = std::env::temp_dir().join(format!("block_viewer-demolish-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk()));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = Meta { name: "demolish-fixture".to_string(), path: dir.clone(), region_dir, regions: vec![(0, 0)] };
            Self { dir, meta }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A finished chunk, one all-dirt section at `Y = 0` (world Y 0..15) —
    /// the "building" the fixture's own placement will sit on top of and
    /// then demolish back to.
    fn full_chunk() -> NbtField {
        let palette = NbtList::Compound(vec![NbtField::new_compound("", vec![NbtField::new_string("Name", "minecraft:dirt")])]);
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
    fn a_demolition_restores_the_terrain_the_placement_baseline_recorded() {
        let fixture = Fixture::new("restore");
        let mut cache = Cache::new(fixture.meta.clone(), 4);

        // Place first: write "stone" over the dirt at (1, 5, 1), the way
        // `city::commit`'s own `blueprint_edit` would, and capture the
        // baseline exactly like `poll_commit` does.
        let mut place_edit = WorldEdit::new();
        place_edit.set(IVec3::new(1, 5, 1), state_named("minecraft:stone"));
        let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
        let place_report = apply_building_edit(&mut cache, &place_edit, &policy).expect("the placement apply");
        let placement_baseline = Baseline::capture(&place_edit, &place_report).unwrap();
        assert_eq!(block_name_at(&mut cache, IVec3::new(1, 5, 1)), "minecraft:stone");

        // Now demolish it: the restoring edit is exactly the placement
        // baseline's own `previous` half.
        let restore_edit = placement_baseline.restore_edit();
        let restore_report = apply_building_edit(&mut cache, &restore_edit, &policy).expect("the restoring apply");
        assert_eq!(block_name_at(&mut cache, IVec3::new(1, 5, 1)), "minecraft:dirt", "the original terrain is back");

        // And the demolition's own baseline recorded the building's actual
        // block (stone) as `previous` — not re-derived from anything.
        let demolition_baseline = Baseline::capture(&restore_edit, &restore_report).unwrap();
        assert_eq!(demolition_baseline.previous, vec![(IVec3::new(1, 5, 1), state_named("minecraft:stone"))]);
        assert_eq!(demolition_baseline.written, vec![(IVec3::new(1, 5, 1), state_named("minecraft:dirt"))]);
    }
}
