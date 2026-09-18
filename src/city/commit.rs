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

//! ## Paying for it, and being paid for the hole (ticket 073)
//!
//! A placement is the first half of iteration 2's rule — *a write that
//! removes blocks credits their drops; a write that restores blocks debits
//! them; a building's own blocks are what `cost` buys*:
//!
//! - The definition's `cost` is spent the instant
//!   [`state::City::place_building`] claims the tiles, in the same
//!   synchronous step and for the same reason: two clicks in flight must not
//!   both be able to afford the last forty planks. Unaffordable is a refusal
//!   before anything is claimed at all, reported through [`WriteStatus`] so
//!   the city panel says why nothing happened.
//! - The terrain the placement cleared is credited on success, out of the
//!   baseline's `previous` (what the write overwrote) through
//!   [`super::drops::DropTable`] — the same record roadmap I1 already
//!   captures, read a second way.
//! - A failed apply refunds exactly what was spent, beside the
//!   [`state::City::remove_building`] rollback that was already there.
//!
//! Both halves land on the journal entry's [`journal::Ledger`], so undo can
//! reverse this placement's own numbers rather than recomputing a cost that
//! may since have been edited under it.
//!
//! ## Converting on the way (ticket 074)
//!
//! A cost is priced through [`economy::plan_payment`], which covers whatever
//! the stock is short of by running `assets/city/economy.ron`'s conversion
//! table — a city holding logs can pay a cost in planks. The conversion is
//! applied at the same instant the cost is spent, and both halves of it join
//! the same ledger: what it consumed is `debited` alongside the cost, what it
//! produced is `credited` alongside the terrain drops. So undo hands the
//! *logs* back rather than the planks they became, and a failed apply
//! reverses it the same way ([`PendingCommit::gained`]).
//!
//! This is the only place conversions run. A demolition's backfill and a
//! terraform's fill *debit*, and debits clamp — see `city::demolish`.
//!
//! ## Which definition's cost, though
//!
//! [`PlacementSelection`] names a *catalogue* id (the `.nbt` stem); `cost`
//! lives on a *definition* (the `.ron`), and the two are allowed to differ.
//! [`PlacementSelection::definition_id`] is the bridge the build menu fills
//! in; a selection made through `city::placement`'s keyboard stand-in has no
//! definition behind it and is therefore **free**, the same hole that
//! already leaves such a placement with no requirements and no production.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::blueprint::{self, Blueprint, BuildingCatalogue, Rotation};
use crate::camera;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;
use crate::DecodedWorld;

use super::definition::BuildingDefinitions;
use super::drops::DropTable;
use super::economy::{self, EconomyConfig};
use super::inventory::{Parcel, Stock};
use super::journal::{self, Journal, Ledger};
use super::picking::{HoveredBlock, PickingSet};
use super::placement::{self, GhostPlacement, PlacementSelection};
use super::road_build::BuildingFootprintChanged;
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
    /// What [`Stock::spend`] actually took for this placement — refunded
    /// verbatim if the apply fails, journaled as the entry's
    /// [`Ledger::debited`] if it succeeds. Kept here rather than looked up
    /// again later: the definition it came from can be hot-reloaded mid-write.
    spent: Parcel,
    /// What ticket 074's conversions *produced* on the way to paying —
    /// materials the placement put into the stock, which a failed apply has
    /// to take back out again or the rollback would leave the player with
    /// planks they never had and a log they no longer do.
    gained: Parcel,
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
            // Ticket 073's three, all `init_resource` for the same reason
            // `WriteStatus` below is: `city::run` inserts the real ones
            // before adding any plugin, and a bare test `App` gets empty
            // defaults (no definitions -> free, no drop table -> every block
            // drops itself) instead of a missing-resource panic.
            .init_resource::<Stock>()
            .init_resource::<DropTable>()
            .init_resource::<BuildingDefinitions>()
            .init_resource::<EconomyConfig>()
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
            // Ticket 110 — same idempotent registration, so this plugin can
            // fire it in a test app without `RoadBuildPlugin` present.
            .add_event::<BuildingFootprintChanged>()
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
    definitions: Res<BuildingDefinitions>,
    mut stock: ResMut<Stock>,
    mut write_status: ResMut<WriteStatus>,
    economy: Res<EconomyConfig>,
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

    // Ticket 085: the same lookup the ghost preview makes off
    // `selection.definition_id` — see `BuildingDefinitions::ground_level`'s
    // own docs for why a keyboard-stand-in selection (no definition) reads
    // as `0`, unshifted.
    let ground_level = definitions.ground_level(selection.definition_id.as_deref());
    let GhostPlacement { origin, valid } = placement::resolve_placement(
        hovered,
        entry.footprint,
        selection.rotation,
        selection.y_offset,
        ground_level,
        &world,
        &city,
    );
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

    // Ticket 073: what this placement costs, if a definition was selected at
    // all — see the module docs' "Which definition's cost, though".
    let costs = selection
        .definition_id
        .as_deref()
        .and_then(|definition| definitions.get(definition))
        .map(|definition| definition.building.cost.clone())
        .unwrap_or_default();

    // Priced before the tile is claimed, so a refusal leaves nothing behind
    // to roll back; the conversion and the `spend` below then can't fail,
    // since nothing between here and there touches the stock. Ticket 074:
    // the shortfall reported is the one that survives conversions, so the
    // message never asks for planks a log in the pile would have covered.
    let payment = economy::plan_payment(&stock, &costs, &economy);
    if !payment.affordable() {
        let shortfall = &payment.shortfall;
        println!("block_viewer: can't afford {id}: needs {shortfall}");
        write_status.record_failure(WriteKind::Placed, id, format!("can't afford it — needs {shortfall}"));
        return;
    }

    // Ticket 076: the definition id travels onto the placement itself, not
    // just into this frame's pricing — a *placed* building has to be able to
    // find its own `.ron` for anything per-instance (production rates,
    // warehouse radii) to read it later. `None` here is the keyboard
    // stand-in's placement, priced free a few lines up for the same reason.
    let definition_id = selection.definition_id.clone();
    let building = match city.place_building(id.clone(), definition_id.clone(), origin, selection.rotation, entry.footprint) {
        Ok(building) => building,
        Err(err) => {
            println!("block_viewer: placement refused: {err}");
            return;
        }
    };
    // The conversion first, then the cost out of what it made — one
    // transaction as far as the stock is concerned, recorded as one ledger.
    let economy::ConversionPlan { consumed, produced } = payment.conversion;
    stock.remove_parcel(&consumed);
    stock.add_parcel(&produced);
    let mut spent = stock.spend(&costs).expect("plan_payment said this was affordable, and nothing since has touched the stock");
    spent.add_all(&consumed);
    let gained = produced;
    let placed =
        PlacedBuilding { catalogue_id: id, definition_id, origin, rotation: selection.rotation, footprint: entry.footprint, work_area: None };

    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
    let task_edit = edit.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        apply_building_edit(&mut cache, &task_edit, &policy)
    });

    commit.pending = Some(PendingCommit { building, placement: placed, edit, spent, gained, task });
}

/// Single non-blocking poll of the in-flight commit, the same
/// `block_on(poll_once(..))` pattern `viewer::paint::poll_paint` uses. On
/// success: journals the as-built baseline and fires [`ChunksEdited`] (W7).
/// On failure: rolls the synchronous [`state::City::place_building`] back —
/// the "transactionally" half of the roadmap's own wording for E4.
#[allow(clippy::too_many_arguments)]
fn poll_commit(
    mut commit: ResMut<CommitState>,
    mut city: ResMut<state::City>,
    mut journal: ResMut<Journal>,
    mut write_status: ResMut<WriteStatus>,
    mut edited: EventWriter<ChunksEdited>,
    mut footprints: EventWriter<BuildingFootprintChanged>,
    mut stock: ResMut<Stock>,
    drops: Res<DropTable>,
    capacity: Option<Res<super::warehouse::StorageCapacity>>,
) {
    let capacity = super::warehouse::storage_capacity(capacity.as_deref());
    let result = {
        let Some(pending) = &mut commit.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still applying.
        };
        result
    };
    let PendingCommit { building, placement, edit, spent, gained, .. } =
        commit.pending.take().expect("just matched Some above");

    match result {
        Ok(report) => {
            let blocks = report.blocks_written;
            let chunks = report.chunks.len();
            println!(
                "block_viewer: placed {} ({blocks} block(s) across {chunks} chunk(s), not yet saved to disk)",
                placement.catalogue_id
            );
            // `EditPolicy::capture_replaced` was on, so `report.replaced` is
            // `Some` and this always succeeds — the `if let` is the same
            // defensive shape `Baseline::capture`'s own doc comment expects
            // of a caller, not a case this path expects to actually miss.
            if let Some(baseline) = journal::Baseline::capture(&edit, &report) {
                // Ticket 073: the terrain this placement cleared, credited
                // out of the very record roadmap I1 already keeps. `previous`
                // is what each written position held before — air included,
                // which the drop table drops on the floor.
                let mut credited = drops.parcel_for(baseline.previous.iter().map(|(_, state)| state));
                // Ticket 079: the cleared terrain goes in under the city's
                // storage cap, and what doesn't fit is reported rather than
                // silently lost.
                let overflow = stock.add_parcel_capped(&credited, capacity);
                if !overflow.is_empty() {
                    println!("block_viewer: storage full — {overflow} could not be stored");
                }
                // Ticket 074's conversions are part of the same action: what
                // they made is already in the stock, and belongs on the
                // ledger so undo takes it back out with everything else.
                credited.add_all(&gained);
                if !credited.is_empty() || !spent.is_empty() {
                    println!(
                        "block_viewer:   paid {} unit(s), recovered {} unit(s) of material",
                        spent.total(),
                        credited.total()
                    );
                }
                journal.record_placement(building, placement.clone(), baseline, Ledger { credited, debited: spent });
            }
            write_status.record_success(WriteKind::Placed, placement.catalogue_id.clone(), &report);
            edited.send(ChunksEdited::from_report(&report));
            // Ticket 110: a road beside this footprint may now want its
            // connected piece. Only from this arm — a rolled-back placement
            // never changed what any road cell touches.
            footprints.send(BuildingFootprintChanged(placement));
        }
        Err(err) => {
            city.remove_building(building);
            // The cost was taken the instant the tile was claimed; both halves
            // of that claim come back together, conversions included — the
            // logs return and the planks they became do not.
            stock.add_parcel(&spent);
            stock.remove_parcel(&gained);
            println!("block_viewer: placement of {} failed, rolled back: {err}", placement.catalogue_id);
            write_status.record_failure(WriteKind::Placed, placement.catalogue_id, err.to_string());
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
        PlacedBuilding {
            catalogue_id: "house01".to_string(),
            definition_id: None,
            origin: IVec3::new(0, 64, 0),
            rotation: Rotation::Deg0,
            footprint: IVec2::new(2, 2),
            work_area: None,
        }
    }

    #[test]
    fn poll_commit_success_records_the_baseline_and_fires_chunks_edited() {
        let mut app = commit_test_app();
        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), state_named("minecraft:stone"));
        let task_edit = edit.clone();
        let report = EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(vec![(IVec3::new(0, 64, 0), BlockState::air())]), ..Default::default()
        };
        let task = pool().spawn(async move { Ok(report) });

        app.world_mut().resource_mut::<CommitState>().pending =
            Some(PendingCommit { building, placement: a_placement(), edit: task_edit, spent: Parcel::default(), gained: Parcel::default(), task });

        run_until_settled(&mut app);

        assert!(app.world().resource::<CommitState>().pending.is_none());
        let journal = app.world().resource::<Journal>();
        assert_eq!(journal.len(), 1, "a successful write journals the baseline");
        assert_eq!(journal.placement_baseline(building).unwrap().written, edit.edits().iter().map(|e| (e.at, e.state.clone())).collect::<Vec<_>>());

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].chunks(), vec![(0, 0)]);

        // Ticket 110: the road re-tile hears about the footprint that landed.
        let footprints: Vec<_> = app.world_mut().resource_mut::<Events<BuildingFootprintChanged>>().drain().collect();
        assert_eq!(footprints.len(), 1);
        assert_eq!(footprints[0].0.origin, a_placement().origin);

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
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        let task = pool().spawn(async { Err(EditRefusal::Empty) });
        app.world_mut().resource_mut::<CommitState>().pending =
            Some(PendingCommit { building, placement: a_placement(), edit: WorldEdit::new(), spent: Parcel::default(), gained: Parcel::default(), task });

        run_until_settled(&mut app);

        let city = app.world().resource::<state::City>();
        assert!(city.is_empty(), "the synchronous place_building must not survive a failed write");
        assert!(city.is_tile_free(IVec2::new(0, 0)), "the tile claim is released along with the entry");
        assert!(app.world().resource::<Journal>().is_empty(), "nothing to journal for a write that never happened");

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "nothing changed in the world, so nothing needs re-meshing");
        let footprints = app.world_mut().resource_mut::<Events<BuildingFootprintChanged>>().drain().count();
        assert_eq!(footprints, 0, "a rolled-back placement never touched what any road cell borders");
    }

    // --- ticket 073: the ledger ---------------------------------------------

    fn a_parcel(items: &[(&str, u64)]) -> Parcel {
        let mut parcel = Parcel::default();
        for &(item, count) in items {
            parcel.add(item, count);
        }
        parcel
    }

    /// A pending commit that cleared `replaced` out of the world and paid
    /// `spent` for the privilege.
    fn pending_that_replaced(
        building: BuildingId,
        replaced: Vec<(IVec3, BlockState)>,
        spent: Parcel,
        result: Result<EditReport, EditRefusal>,
    ) -> PendingCommit {
        let mut edit = WorldEdit::new();
        for (at, _) in &replaced {
            edit.set(*at, state_named("minecraft:oak_planks"));
        }
        let task = pool().spawn(async move { result });
        PendingCommit { building, placement: a_placement(), edit, spent, gained: Parcel::default(), task }
    }

    #[test]
    fn a_successful_placement_credits_the_terrain_it_cleared() {
        let mut app = commit_test_app();
        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        // Two stone, one air: with the default (empty) drop table stone
        // drops itself, and air is never a drop at all.
        let replaced = vec![
            (IVec3::new(0, 64, 0), state_named("minecraft:stone")),
            (IVec3::new(1, 64, 0), state_named("minecraft:stone")),
            (IVec3::new(2, 64, 0), BlockState::air()),
        ];
        let report = EditReport {
            blocks_written: 3,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(replaced.clone()), ..Default::default()
        };
        app.world_mut().resource_mut::<CommitState>().pending =
            Some(pending_that_replaced(building, replaced, a_parcel(&[("minecraft:oak_planks", 40)]), Ok(report)));

        run_until_settled(&mut app);

        let stock = app.world().resource::<Stock>();
        assert_eq!(stock.count("minecraft:stone"), 2);
        assert_eq!(stock.count(BlockState::AIR), 0, "air is not a material");

        let journal = app.world().resource::<Journal>();
        let ledger = journal.entries().last().unwrap().ledger();
        assert_eq!(ledger.credited, a_parcel(&[("minecraft:stone", 2)]));
        assert_eq!(ledger.debited, a_parcel(&[("minecraft:oak_planks", 40)]), "the ledger records what was actually paid");
    }

    /// Ticket 074, end to end on the ledger: a placement paid for by
    /// converting logs into planks has to undo back to *logs*. The
    /// placement's own numbers are simulated here the way
    /// `try_commit_placement` computes them (`economy::plan_payment` has its
    /// own tests for that); what this pins down is that reversing the ledger
    /// `poll_commit` writes puts the stock back exactly as it was.
    #[test]
    fn a_placement_paid_for_by_converting_undoes_back_to_what_was_converted() {
        let mut app = commit_test_app();
        let before = {
            let mut stock = Stock::default();
            stock.add("minecraft:oak_log", 20);
            stock
        };

        // Ten logs became forty planks, and the forty planks were spent.
        *app.world_mut().resource_mut::<Stock>() = {
            let mut stock = before.clone();
            stock.remove("minecraft:oak_log", 10);
            stock
        };
        let mut spent = a_parcel(&[("minecraft:oak_log", 10), ("minecraft:oak_planks", 40)]);
        let gained = a_parcel(&[("minecraft:oak_planks", 40)]);

        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();
        let replaced = vec![(IVec3::new(0, 64, 0), state_named("minecraft:dirt"))];
        let report = EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(replaced.clone()), ..Default::default()
        };
        let mut pending = pending_that_replaced(building, replaced, Parcel::default(), Ok(report));
        std::mem::swap(&mut pending.spent, &mut spent);
        pending.gained = gained;
        app.world_mut().resource_mut::<CommitState>().pending = Some(pending);

        run_until_settled(&mut app);

        let ledger = app.world().resource::<Journal>().entries().last().unwrap().ledger().clone();
        assert_eq!(ledger.credited.get("minecraft:oak_planks"), 40, "what the conversion made is credited");
        assert_eq!(ledger.credited.get("minecraft:dirt"), 1, "alongside the terrain the placement cleared");
        assert_eq!(ledger.debited.get("minecraft:oak_log"), 10, "and what it consumed is debited");

        // Undo's own settlement (`city::undo::settle_reverse`), applied here
        // to the ledger this commit actually wrote.
        let mut stock = app.world().resource::<Stock>().clone();
        stock.add_parcel(&ledger.debited);
        stock.remove_parcel(&ledger.credited);

        assert_eq!(stock, before, "undoing gives the logs back, not the planks they became");
    }

    #[test]
    fn a_failed_placement_undoes_its_conversion_too() {
        let mut app = commit_test_app();
        // Mid-placement: ten logs are already gone and the planks they made
        // have already been spent on the cost.
        app.world_mut().resource_mut::<Stock>().add("minecraft:oak_log", 10);
        let before = app.world().resource::<Stock>().clone();

        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();
        let mut pending = pending_that_replaced(
            building,
            vec![(IVec3::new(0, 64, 0), state_named("minecraft:stone"))],
            a_parcel(&[("minecraft:oak_log", 10), ("minecraft:oak_planks", 40)]),
            Err(EditRefusal::Empty),
        );
        pending.gained = a_parcel(&[("minecraft:oak_planks", 40)]);
        app.world_mut().resource_mut::<CommitState>().pending = Some(pending);

        run_until_settled(&mut app);

        let stock = app.world().resource::<Stock>();
        assert_eq!(stock.count("minecraft:oak_log"), 20, "the logs come back");
        assert_eq!(stock.count("minecraft:oak_planks"), 0, "and the planks they became do not stay");
        assert_ne!(*stock, before, "…which is the pile as it stood before the conversion, not mid-payment");
    }

    #[test]
    fn a_failed_placement_refunds_exactly_what_it_spent() {
        let mut app = commit_test_app();
        app.world_mut().resource_mut::<Stock>().add("minecraft:oak_planks", 10);
        let building = app
            .world_mut()
            .resource_mut::<state::City>()
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        app.world_mut().resource_mut::<CommitState>().pending = Some(pending_that_replaced(
            building,
            vec![(IVec3::new(0, 64, 0), state_named("minecraft:stone"))],
            a_parcel(&[("minecraft:oak_planks", 40)]),
            Err(EditRefusal::Empty),
        ));

        run_until_settled(&mut app);

        let stock = app.world().resource::<Stock>();
        assert_eq!(stock.count("minecraft:oak_planks"), 50, "the cost comes back with the rolled-back city entry");
        assert_eq!(stock.count("minecraft:stone"), 0, "and a write that never landed cleared nothing to credit");
    }
}
