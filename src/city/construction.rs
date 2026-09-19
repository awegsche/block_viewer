//! Site clearing (ticket 128): the tick that digs a placement's own volume
//! clear before its blueprint is written, so building on a hillside costs
//! the same thing the gatherer charges for levelling one — time — instead of
//! being free.
//!
//! ## Why this exists
//!
//! `city::commit`'s write is one big `blueprint_edit` that overwrites
//! *everything* in a placement's volume in a single apply — a hillside, a
//! forest, the top of a mountain — and ticket 073 already credits the drops
//! of whatever that write removed. What it never did is make the player
//! *wait* for it: a Gatherer's Hut needs minutes to level a yard, while a
//! house dropped on the same slope levels it for free in one click. This
//! module is what closes that gap, for both a building's placement and a
//! road cell's.
//!
//! ## What a site is
//!
//! A **site** is a claimed placement — a row in [`super::state::City`] with
//! [`super::state::PlacedBuilding::under_construction`] (or
//! [`super::state::RoadCell::under_construction`]) set — whose blueprint
//! hasn't been written yet because the ground it displaces isn't clear.
//! `city::commit::try_commit_placement` and `city::road_build::try_commit_drag`
//! are what decide whether a placement becomes a site at all: a volume
//! that's already all air (or entirely undecoded) never becomes one, and
//! takes exactly the instant path it always has — see those modules' own
//! docs for the "nothing to clear" fork. A building site is journalled the
//! instant it's entered, with an empty baseline, so cancel/undo/persistence
//! all fall out of machinery `city::journal` already has
//! ([`super::journal::Journal::extend_placement_baseline`]/
//! [`super::journal::Journal::remove_site_entry`]). A road cell site is not
//! — roads aren't journalled at all (`city::road_build`'s own module docs),
//! so a cell's clearing credits the stock and nothing else.
//!
//! ## The tick
//!
//! [`SiteId`] names one site — a building or a road cell — and
//! [`ConstructionState`] tracks two things per site: a fractional block
//! `carry` (the same shape `city::gatherer::Producer::dig_carry` uses) and
//! at most one clearing write in flight ([`PendingClear`]). Sites don't
//! queue behind each other — several can be clearing on the same tick, the
//! same "one write in flight per building, not one shared slot" reasoning
//! [`super::gatherer`]'s own module docs give.
//!
//! Every tick with [`super::clock::GameClock::delta_minutes`] `> 0`, for
//! every site without a pending write: [`scan_site`] reads a fresh, top-down
//! scan of the site's volume out of [`crate::DecodedWorld`].
//!
//! - **Something left to clear**: carry accrues at
//!   [`super::economy::EconomyConfig::site_clearing_blocks_per_minute`];
//!   once a whole block is owed, a [`crate::edit::WorldEdit`] clearing that
//!   many positions to air — topmost layer first, then column order within
//!   the layer, so a hill shrinks rather than gets hollowed — is dispatched
//!   onto `AsyncComputeTaskPool`, the same [`crate::edit::EditPolicy`]
//!   (`capture_replaced: true`, `allow_dirty_regions: true`) the original
//!   commit uses. The carry is charged at dispatch and refunded on a refused
//!   write, exactly as `city::gatherer::plan_dig`/`settle_dig` do — see
//!   those functions' own docs. No [`super::write_status::WriteStatus`]
//!   line for a dig, for the same reason a gatherer's own dig gets none.
//! - **Nothing left**: the site is done clearing. A building's completion
//!   dispatches the real blueprint write through
//!   [`super::commit::dispatch_blueprint_write`] into
//!   `city::commit`'s own single `CommitState::pending` slot; a road cell's
//!   dispatches [`super::road_build::dispatch_road_cell_write`] into
//!   `city::road_build`'s own single slot. Both are single slots shared with
//!   every other write those modules make — a site that finds its module's
//!   slot busy simply tries again next tick, which is also how a completion
//!   write that fails ever gets retried: nothing here remembers "this site
//!   already tried to complete," so the very next tick's empty scan asks
//!   again.
//!
//! ## Settling a dig
//!
//! [`poll_clears`] is the single non-blocking poll every other write path in
//! this crate uses. On success: [`super::journal::Baseline::capture`] reads
//! what the dig actually removed, [`super::drops::DropTable`] turns it into
//! materials credited into the [`super::inventory::Stock`] (capped by
//! [`super::warehouse::StorageCapacity`], overflow printed — the same shape
//! every other credit in this crate follows). A building site's dig also
//! extends its journal entry
//! ([`super::journal::Journal::extend_placement_baseline`]); a road cell's
//! credits the stock and nowhere else. [`crate::chunk_pipeline::ChunksEdited`]
//! still fires either way, so the dug blocks vanish from the mesh.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::blueprint::BuildingCatalogue;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;
use crate::world::BlockRegistry;
use crate::DecodedWorld;

use super::clock::GameClock;
use super::commit::CommitState;
use super::drops::DropTable;
use super::economy::EconomyConfig;
use super::grid;
use super::inventory::Stock;
use super::journal::{Baseline, Journal};
use super::loading::GameplaySet;
use super::road_build::RoadBuildState;
use super::road_catalogue::RoadCatalogue;
use super::state::{BuildingId, City};
use super::warehouse::{storage_capacity, StorageCapacity};

/// Names one site — a placement claimed in [`super::state::City`] but not
/// yet built, see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum SiteId {
    Building(BuildingId),
    RoadCell(IVec2),
}

/// One clearing dig, in flight — [`ConstructionState::pending`]'s value.
struct PendingClear {
    edit: WorldEdit,
    task: Task<Result<EditReport, EditRefusal>>,
}

/// Per-site clearing state — see the module docs. `pub(super)`:
/// `city::demolish`/`city::undo` read [`is_dig_in_flight`](Self::is_dig_in_flight)
/// to refuse cancelling or undoing a site whose dig hasn't settled yet,
/// rather than racing it.
#[derive(Resource, Default)]
pub(super) struct ConstructionState {
    carry: HashMap<SiteId, f32>,
    pending: HashMap<SiteId, PendingClear>,
}

impl ConstructionState {
    /// Whether `site` has a clearing dig in flight right now.
    pub(super) fn is_dig_in_flight(&self, site: SiteId) -> bool {
        self.pending.contains_key(&site)
    }

    /// Test-only: fakes a dig in flight for `site`, so `city::demolish`/
    /// `city::undo`'s own tests can exercise their "refuses while a dig is
    /// in flight" precondition without a real `AsyncComputeTaskPool` task
    /// that actually clears anything.
    #[cfg(test)]
    pub(super) fn mark_dig_in_flight_for_tests(&mut self, site: SiteId) {
        let task = AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default).spawn(async { Err(EditRefusal::Empty) });
        self.pending.insert(site, PendingClear { edit: WorldEdit::new(), task });
    }
}

pub struct ConstructionPlugin;

impl Plugin for ConstructionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConstructionState>()
            // Idempotent-either-order shape `city::commit`/`city::gatherer`
            // already document for these resources — `CommitState`/
            // `RoadBuildState` included, since this plugin dispatches a
            // site's completion write into each module's own single slot.
            .init_resource::<City>()
            .init_resource::<Journal>()
            .init_resource::<Stock>()
            .init_resource::<DropTable>()
            .init_resource::<EconomyConfig>()
            .init_resource::<CommitState>()
            .init_resource::<RoadBuildState>()
            .add_event::<ChunksEdited>()
            .add_systems(Update, (dispatch_clears, poll_clears).chain().after(super::picking::PickingSet).in_set(GameplaySet));
    }
}

/// Every non-air position in `origin .. origin + size`, ordered topmost
/// layer first and then column order within a layer — see the module docs.
/// A column whose chunk isn't decoded contributes nothing at all: nothing
/// here can tell "genuinely empty" apart from "not streamed in yet," and
/// the same commit that reads this scan already treats an undecoded column
/// as nothing to clear, exactly the way it always overwrote one for free.
///
/// `pub(super)`: `city::commit`/`city::road_build` call this once, at
/// commit time, to decide whether a placement needs to become a site at
/// all; this module calls it again every tick a site without a pending
/// write has, on a fresh read, which is what makes a neighbouring edit
/// (another site's dig, a terraform) safe to run alongside this one.
pub(super) fn scan_site(origin: IVec3, size: IVec3, world: &DecodedWorld) -> Vec<IVec3> {
    let mut found = Vec::new();
    for dy in (0..size.y).rev() {
        for dz in 0..size.z {
            for dx in 0..size.x {
                let at = origin + IVec3::new(dx, dy, dz);
                if grid::block_at(at, world).is_some_and(|id| id != BlockRegistry::AIR) {
                    found.push(at);
                }
            }
        }
    }
    found
}

/// The `WorldEdit` for one dig batch: the first `blocks` positions of
/// `remaining` (already ordered top-down by [`scan_site`]), cleared to air.
fn clear_edit(remaining: &[IVec3], blocks: u32) -> WorldEdit {
    let mut edit = WorldEdit::new();
    for &at in remaining.iter().take(blocks as usize) {
        edit.set(at, crate::blueprint::BlockState::air());
    }
    edit
}

/// Dispatches `edit` onto `AsyncComputeTaskPool`, tracked under `site` —
/// shared by both the building and the road-cell half of [`dispatch_clears`].
fn dispatch_dig(state: &mut ConstructionState, site: SiteId, edit: WorldEdit, region_cache: &SharedRegionCache) {
    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
    let task_edit = edit.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        super::commit::apply_building_edit(&mut cache, &task_edit, &policy)
    });
    state.pending.insert(site, PendingClear { edit, task });
}

/// A building site's own volume — the rotated blueprint's box at its
/// origin, exactly what [`super::commit::blueprint_edit`] writes.
/// `None` when the catalogue has nothing for `catalogue_id` (shouldn't
/// happen for a real site) or the rotation fails — either way the site is
/// simply skipped this tick, retried the next.
fn building_site_box(catalogue: &BuildingCatalogue, catalogue_id: &str, rotation: crate::blueprint::Rotation) -> Option<IVec3> {
    let entry = catalogue.get(catalogue_id)?;
    Some(match rotation {
        crate::blueprint::Rotation::Deg90 | crate::blueprint::Rotation::Deg270 => {
            IVec3::new(entry.blueprint.size.z, entry.blueprint.size.y, entry.blueprint.size.x)
        }
        crate::blueprint::Rotation::Deg0 | crate::blueprint::Rotation::Deg180 => entry.blueprint.size,
    })
}

/// One tick: accrues every site's carry and dispatches a dig or a
/// completion write — see the module docs.
#[allow(clippy::too_many_arguments)]
fn dispatch_clears(
    clock: Res<GameClock>,
    mut city: ResMut<City>,
    catalogue: Option<Res<BuildingCatalogue>>,
    road_catalogue: Option<Res<RoadCatalogue>>,
    economy: Res<EconomyConfig>,
    world: Res<DecodedWorld>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut state: ResMut<ConstructionState>,
    mut commit: ResMut<CommitState>,
    mut road_build: ResMut<RoadBuildState>,
) {
    let minutes = clock.delta_minutes();
    if minutes <= 0.0 {
        return; // paused, or a frame the clock clamped to nothing
    }
    let Some(region_cache) = region_cache else { return };
    let rate = economy.site_clearing_blocks_per_minute;

    // Buildings -------------------------------------------------------------
    if let Some(catalogue) = catalogue.as_deref() {
        for (id, placed) in city.placements() {
            if !placed.under_construction {
                continue;
            }
            let site = SiteId::Building(id);
            if state.pending.contains_key(&site) {
                continue;
            }
            let Some(size) = building_site_box(catalogue, &placed.catalogue_id, placed.rotation) else { continue };
            let remaining = scan_site(placed.origin, size, &world);

            if remaining.is_empty() {
                if commit.is_busy() {
                    continue; // the module's write slot is busy — retry next tick
                }
                let rotated;
                let entry = catalogue.get(&placed.catalogue_id).expect("building_site_box just resolved this entry");
                let blueprint = if placed.rotation == crate::blueprint::Rotation::Deg0 {
                    &entry.blueprint
                } else {
                    match crate::blueprint::rotate_blueprint(&entry.blueprint, placed.rotation) {
                        Ok(b) => {
                            rotated = b;
                            &rotated
                        }
                        Err(err) => {
                            println!("block_viewer: site for {} can't rotate to {:?}, retrying: {err}", placed.catalogue_id, placed.rotation);
                            continue;
                        }
                    }
                };
                super::commit::dispatch_blueprint_write(
                    &mut commit,
                    &region_cache,
                    id,
                    placed.clone(),
                    blueprint,
                    placed.origin,
                    super::inventory::Parcel::default(),
                    super::inventory::Parcel::default(),
                    true,
                );
                continue;
            }

            let carry = state.carry.entry(site).or_insert(0.0);
            *carry += rate * minutes;
            let blocks = carry.floor();
            if blocks < 1.0 {
                continue;
            }
            let edit = clear_edit(&remaining, blocks as u32);
            *carry -= edit.len() as f32;
            dispatch_dig(&mut state, site, edit, &region_cache);
        }
    }

    // Road cells --------------------------------------------------------------
    if let Some(road_catalogue) = road_catalogue.as_deref() {
        let sites: Vec<IVec2> = city.road_cells_with_data().filter(|(_, road)| road.under_construction).map(|(&cell, _)| cell).collect();
        for cell in sites {
            let site = SiteId::RoadCell(cell);
            if state.pending.contains_key(&site) {
                continue;
            }
            let Some((origin, size)) = super::road_build::road_site_box(cell, &city, road_catalogue) else { continue };
            let remaining = scan_site(origin, size, &world);

            if remaining.is_empty() {
                if road_build.is_busy() {
                    continue;
                }
                super::road_build::dispatch_road_cell_write(&mut road_build, &region_cache, cell, road_catalogue, &mut city);
                continue;
            }

            let carry = state.carry.entry(site).or_insert(0.0);
            *carry += rate * minutes;
            let blocks = carry.floor();
            if blocks < 1.0 {
                continue;
            }
            let edit = clear_edit(&remaining, blocks as u32);
            *carry -= edit.len() as f32;
            dispatch_dig(&mut state, site, edit, &region_cache);
        }
    }
}

/// Single non-blocking poll of every in-flight dig — see the module docs'
/// "Settling a dig".
fn poll_clears(
    mut state: ResMut<ConstructionState>,
    mut journal: ResMut<Journal>,
    mut stock: ResMut<Stock>,
    drops: Res<DropTable>,
    capacity: Option<Res<StorageCapacity>>,
    mut edited: EventWriter<ChunksEdited>,
) {
    let capacity = storage_capacity(capacity.as_deref());
    let mut settled: Vec<(SiteId, WorldEdit, Result<EditReport, EditRefusal>)> = Vec::new();
    state.pending.retain(|&site, pending| match block_on(poll_once(&mut pending.task)) {
        Some(result) => {
            settled.push((site, std::mem::take(&mut pending.edit), result));
            false
        }
        None => true,
    });

    for (site, edit, result) in settled {
        match result {
            Ok(report) => {
                if let Some(baseline) = Baseline::capture(&edit, &report) {
                    let credited = drops.parcel_for(baseline.previous.iter().map(|(_, state)| state));
                    let overflow = stock.add_parcel_capped(&credited, capacity);
                    if !overflow.is_empty() {
                        println!("block_viewer: storage full — {overflow} could not be stored");
                    }
                    if let SiteId::Building(building) = site {
                        journal.extend_placement_baseline(building, baseline, credited);
                    }
                }
                edited.send(ChunksEdited::from_report(&report));
            }
            Err(err) => {
                println!("block_viewer: site clearing dig failed: {err}");
                *state.carry.entry(site).or_insert(0.0) += edit.len() as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{BiomeRegistry, BlockId, BlockRegistry as Registry, ChunkColumn, ChunkSection};
    use std::collections::HashMap as StdHashMap;

    fn registry_with_stone() -> (Registry, BlockId) {
        let mut registry = Registry::new();
        let stone = registry.intern("minecraft:stone");
        (registry, stone)
    }

    /// A single chunk `(0, 0)`, solid stone from `y=0` to `y=height-1` at
    /// tile `(0, 0)`, air everywhere else in the chunk — enough to give
    /// [`scan_site`] a small, hand-checkable volume.
    fn world_with_column(height: i32) -> DecodedWorld {
        let (registry, stone) = registry_with_stone();
        let size = crate::world::SECTION_SIZE as i32;
        let mut column = ChunkColumn { x: 0, z: 0, sections: Vec::new(), floor_y: crate::world::WORLD_MIN_Y };

        for y in 0..height {
            let section_y = y.div_euclid(size) as i8;
            let local_y = y.rem_euclid(size) as usize;
            let section = match column.sections.iter().position(|s| s.y == section_y) {
                Some(index) => index,
                None => {
                    column.sections.push(ChunkSection {
                        y: section_y,
                        blocks: Box::new([Registry::AIR; crate::world::SECTION_VOLUME]),
                        biomes: Box::new([BiomeRegistry::PLAINS; crate::world::BIOME_GRID_VOLUME]),
                    });
                    column.sections.len() - 1
                }
            };
            column.sections[section].blocks[ChunkSection::index(0, local_y, 0)] = stone;
        }

        let mut columns: StdHashMap<(i32, i32), ChunkColumn> = StdHashMap::new();
        columns.insert((0, 0), column);
        DecodedWorld { registry: Arc::new(Mutex::new(registry)), biomes: Arc::new(Mutex::new(BiomeRegistry::new())), columns }
    }

    #[test]
    fn scan_site_counts_every_non_air_block_top_layer_first() {
        let world = world_with_column(3);
        // A 1x5x1 volume starting at y=0: the bottom three are stone (0,1,2),
        // the top two are air.
        let found = scan_site(IVec3::new(0, 0, 0), IVec3::new(1, 5, 1), &world);
        assert_eq!(found, vec![IVec3::new(0, 2, 0), IVec3::new(0, 1, 0), IVec3::new(0, 0, 0)], "topmost first");
    }

    #[test]
    fn scan_site_is_empty_over_pure_air() {
        let world = world_with_column(0);
        let found = scan_site(IVec3::new(0, 0, 0), IVec3::new(2, 2, 2), &world);
        assert!(found.is_empty());
    }

    #[test]
    fn scan_site_treats_an_undecoded_column_as_nothing_to_clear() {
        let world = world_with_column(3);
        let found = scan_site(IVec3::new(500, 0, 500), IVec3::new(1, 5, 1), &world);
        assert!(found.is_empty());
    }

    #[test]
    fn clear_edit_takes_the_first_n_positions_to_air() {
        let remaining = vec![IVec3::new(0, 2, 0), IVec3::new(0, 1, 0), IVec3::new(0, 0, 0)];
        let edit = clear_edit(&remaining, 2);
        assert_eq!(edit.len(), 2);
        let positions: std::collections::HashSet<IVec3> = edit.edits().iter().map(|e| e.at).collect();
        assert_eq!(positions, std::collections::HashSet::from([IVec3::new(0, 2, 0), IVec3::new(0, 1, 0)]));
        for e in edit.edits() {
            assert_eq!(e.state.name, "minecraft:air");
        }
    }

    #[test]
    fn clear_edit_never_takes_more_than_is_left() {
        let remaining = vec![IVec3::new(0, 0, 0)];
        let edit = clear_edit(&remaining, 5);
        assert_eq!(edit.len(), 1, "there was only one block to take");
    }

    // --- poll_clears: settling a dig, through a real App ---------------------

    use bevy::tasks::TaskPool;
    use crate::edit::EditReport;

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn poll_test_app() -> App {
        let mut app = App::new();
        app.init_resource::<ConstructionState>()
            .init_resource::<Journal>()
            .init_resource::<Stock>()
            .init_resource::<DropTable>()
            .add_event::<ChunksEdited>()
            .add_systems(Update, poll_clears);
        app
    }

    fn run_until_settled(app: &mut App, site: SiteId) {
        for _ in 0..200 {
            app.update();
            if !app.world().resource::<ConstructionState>().pending.contains_key(&site) {
                return;
            }
        }
        panic!("clearing dig never settled");
    }

    fn dig_report(at: IVec3, previous: &str) -> EditReport {
        EditReport {
            blocks_written: 1,
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(vec![(at, crate::blueprint::BlockState { name: previous.to_string(), properties: Vec::new() })]),
            ..Default::default()
        }
    }

    fn a_placement() -> super::super::state::PlacedBuilding {
        super::super::state::PlacedBuilding {
            catalogue_id: "house01".to_string(),
            definition_id: None,
            origin: IVec3::new(0, 64, 0),
            rotation: crate::blueprint::Rotation::Deg0,
            footprint: IVec2::ONE,
            work_area: None,
            under_construction: true,
        }
    }

    #[test]
    fn a_settled_building_dig_credits_the_stock_and_extends_the_journal() {
        let mut app = poll_test_app();
        let building = BuildingId::from_u64(0);
        app.world_mut().resource_mut::<Journal>().record_placement(
            building,
            a_placement(),
            Baseline { written: Vec::new(), previous: Vec::new(), data_version: None },
            super::super::journal::Ledger::default(),
        );

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), crate::blueprint::BlockState::air());
        let task = pool().spawn(async move { Ok(dig_report(IVec3::new(0, 64, 0), "minecraft:stone")) });
        app.world_mut().resource_mut::<ConstructionState>().pending.insert(SiteId::Building(building), PendingClear { edit, task });

        run_until_settled(&mut app, SiteId::Building(building));

        assert_eq!(app.world().resource::<Stock>().count("minecraft:stone"), 1);
        let baseline = app.world().resource::<Journal>().placement_baseline(building).unwrap();
        assert_eq!(baseline.previous, vec![(IVec3::new(0, 64, 0), crate::blueprint::BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() })]);
        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 1);
    }

    #[test]
    fn a_settled_road_cell_dig_credits_the_stock_and_never_touches_the_journal() {
        let mut app = poll_test_app();
        let cell = IVec2::new(3, 4);

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), crate::blueprint::BlockState::air());
        let task = pool().spawn(async move { Ok(dig_report(IVec3::new(0, 64, 0), "minecraft:dirt")) });
        app.world_mut().resource_mut::<ConstructionState>().pending.insert(SiteId::RoadCell(cell), PendingClear { edit, task });

        run_until_settled(&mut app, SiteId::RoadCell(cell));

        assert_eq!(app.world().resource::<Stock>().count("minecraft:dirt"), 1);
        assert!(app.world().resource::<Journal>().is_empty(), "roads are never journalled");
    }

    #[test]
    fn a_failed_dig_refunds_the_carry_it_claimed() {
        let mut app = poll_test_app();
        let site = SiteId::RoadCell(IVec2::new(0, 0));

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), crate::blueprint::BlockState::air());
        edit.set(IVec3::new(1, 64, 0), crate::blueprint::BlockState::air());
        let task = pool().spawn(async { Err(EditRefusal::Empty) });
        app.world_mut().resource_mut::<ConstructionState>().pending.insert(site, PendingClear { edit, task });

        run_until_settled(&mut app, site);

        assert_eq!(app.world().resource::<ConstructionState>().carry.get(&site), Some(&2.0));
        assert!(app.world().resource::<Stock>().is_empty());
    }
}
