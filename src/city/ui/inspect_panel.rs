//! The inspect panel (ticket 083, roadmap G3): a third window alongside the
//! build menu and the city panel, and the reason `ActiveTool::Inspect` can
//! be the resting state at all — a left-click that no longer commits a
//! placement has to mean *something*, and clicking a placed building to see
//! what it's doing is that something. `city::picking::SelectedBuilding` is
//! where the click itself lands; this module is only the display.
//!
//! ## Nothing shown with nothing selected
//!
//! Unlike the build menu and city panel — which always draw their window,
//! with a placeholder line for an empty state — this one draws no window at
//! all while [`SelectedBuilding`] is `None`. A resting-state panel that's
//! visible constantly, one way or another, would be exactly the clutter
//! `ActiveTool::Inspect` becoming the *default* was trying to avoid.
//!
//! ## No health field
//!
//! Group I ("the world answers back") doesn't exist yet — nothing in this
//! crate reads block damage. A stub health number here would be something to
//! rip out later rather than fill in now; this panel grows one when I4 lands
//! and not before.
//!
//! ## The working-area section (ticket 111)
//!
//! For a gatherer, the panel also shows its drawn working area (or a
//! warning that there is none — a hut without one does nothing) and is the
//! one place that switches [`ActiveTool`] to `DrawWorkArea`: the area
//! belongs to *this* building, and the panel is what knows which one is
//! selected. The drawing itself, and the tool's way back to `Inspect`, live
//! in `city::work_area`; this is only the button. Both this and ticket 107's
//! "Clear buffer" mutate after the window closure, not inside it — the
//! window borrows the resources it displays.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::blueprint::{BuildingCatalogue, Rotation};
use crate::DecodedWorld;

use super::super::construction::scan_site;
use super::super::definition::{BuildingDefinitions, Mine};
use super::super::economy::EconomyConfig;
use super::super::gatherer::gatherer_buffer_capacity;
use super::super::inventory::{short_name, Parcel};
use super::super::journal::Journal;
use super::super::mine::layout::{Arm, MineFrame};
use super::super::mine::progress::{LevelCursor, MineProgress, Phase, RowStep};
use super::super::mine::{mine_buffer_capacity, MineState};
use super::super::picking::SelectedBuilding;
use super::super::placement::rotation_degrees;
use super::super::production::{buffer_capacity, Producer, ProductionState};
use super::super::state::{self, BuildingId, PlacedBuilding, WorkArea};
use super::super::tool::ActiveTool;
use super::super::warehouse::Coverage;

/// The selected building's own display name — its definition's, via
/// [`state::City::definition_of`], falling back to the catalogue id the same
/// way `build_menu::requirement_label` falls back to a raw id it can't
/// resolve.
fn building_name(id: state::BuildingId, catalogue_id: &str, city: &state::City, definitions: &BuildingDefinitions) -> String {
    city.definition_of(id)
        .and_then(|definition_id| definitions.get(definition_id))
        .map(|entry| entry.building.name.clone())
        .unwrap_or_else(|| catalogue_id.to_string())
}

/// The buffer's contents, one line per item, each against the buffer's
/// shared cap (`buffer.total() >= capacity` is what `production::tick` stops
/// on, per item and combined both read the same way) — `"  oak_log: 37/256"`.
/// A plain function, testable without an `egui::Context`, the same shape
/// `city_panel::stock_lines` uses.
fn buffer_lines(producer: &Producer, capacity: u64) -> Vec<String> {
    producer.buffer.iter().map(|(item, count)| format!("  {}: {count}/{capacity}", short_name(item))).collect()
}

/// This building's own buffer cap, if it has a `production` or (ticket 086) a
/// `gatherer` block — `0` otherwise, which reads as an empty buffer rather
/// than a divide-by-zero anywhere downstream (nothing here divides by it). A
/// definition with both is read as `production`'s own cap; the shipped set
/// never declares both on one building.
fn producer_capacity(placed: &PlacedBuilding, definitions: &BuildingDefinitions, economy: &EconomyConfig) -> u64 {
    let Some(entry) = placed.definition_id.as_deref().and_then(|id| definitions.get(id)) else { return 0 };
    if let Some(spec) = entry.building.production.as_ref() {
        return buffer_capacity(spec, economy);
    }
    if let Some(gatherer) = entry.building.gatherer.as_ref() {
        return gatherer_buffer_capacity(gatherer, economy);
    }
    if let Some(mine) = entry.building.mine.as_ref() {
        return mine_buffer_capacity(mine, economy);
    }
    0
}

/// Ticket 107: the panel's line for whether `id` is reachable by a warehouse
/// at all — [`super::super::warehouse::Coverage`] already answers this for
/// haulage's own dispatch; this is its first display. `None` means "not
/// connected", which the panel shows as a warning rather than a line naming
/// nothing.
fn warehouse_status(
    id: state::BuildingId,
    coverage: Option<&Coverage>,
    city: &state::City,
    definitions: &BuildingDefinitions,
) -> Option<String> {
    let served = coverage?.served(id)?;
    let name = city
        .building(served.warehouse)
        .map(|warehouse| building_name(served.warehouse, &warehouse.catalogue_id, city, definitions))
        .unwrap_or_else(|| "warehouse".to_string());
    Some(format!("Warehouse: {name} ({:.1} min away)", served.travel_minutes))
}

/// Whether `placed` is a gatherer — the one kind of building with a working
/// area to show and draw (ticket 111).
fn is_gatherer(placed: &PlacedBuilding, definitions: &BuildingDefinitions) -> bool {
    placed.definition_id.as_deref().and_then(|id| definitions.get(id)).is_some_and(|entry| entry.building.gatherer.is_some())
}

/// The panel's line for a drawn working area: both corners, Minecraft
/// `(x, z)`, and the tile count.
fn work_area_line(area: WorkArea) -> String {
    format!("Working area: ({}, {}) - ({}, {}), {} tiles", area.min.x, area.min.y, area.max.x, area.max.y, area.len())
}

/// Ticket 128: the panel's line for a **site** — `N of M blocks (~x min
/// left)`. `M` is recounted every frame from the world plus the entry's own
/// `written` count rather than cached anywhere (`N + written` is invariant
/// across the site's whole lifetime, since every dig moves exactly one block
/// from one side of that sum to the other — see `city::construction`'s
/// module docs and `city::persistence`'s "M is recounted" note). `None` when
/// `placed` isn't a site, or there's nothing to compute it from (no
/// catalogue, no matching entry).
fn site_clearing_line(
    id: BuildingId,
    placed: &PlacedBuilding,
    catalogue: Option<&BuildingCatalogue>,
    world: &DecodedWorld,
    journal: &Journal,
    economy: &EconomyConfig,
) -> Option<String> {
    if !placed.under_construction {
        return None;
    }
    let entry = catalogue?.get(&placed.catalogue_id)?;
    let size = match placed.rotation {
        Rotation::Deg90 | Rotation::Deg270 => IVec3::new(entry.blueprint.size.z, entry.blueprint.size.y, entry.blueprint.size.x),
        Rotation::Deg0 | Rotation::Deg180 => entry.blueprint.size,
    };
    let remaining = scan_site(placed.origin, size, world).len() as u32;
    let written = journal.placement_baseline(id).map(|baseline| baseline.written.len()).unwrap_or(0) as u32;
    let total = remaining + written;
    let rate = economy.site_clearing_blocks_per_minute.max(f32::MIN_POSITIVE);
    let minutes_left = remaining as f32 / rate;
    Some(format!("Clearing site: {remaining} of {total} block(s) (~{minutes_left:.1} min left)"))
}

/// What the panel's working-area buttons asked for this frame — resolved
/// after the window closes, the way `clear_buffer` is.
#[derive(Default)]
struct WorkAreaRequest {
    draw: bool,
    clear: bool,
}

/// This building's `Mine` definition and blueprint ground level, if it has
/// one — [`is_gatherer`]'s counterpart for the mine section (ticket 117);
/// bundled with `ground_level` since [`MineFrame::from_placement`] needs
/// both.
fn mine_definition<'a>(placed: &PlacedBuilding, definitions: &'a BuildingDefinitions) -> Option<(&'a Mine, u32)> {
    let entry = definitions.get(placed.definition_id.as_deref()?)?;
    let mine = entry.building.mine.as_ref()?;
    Some((mine, entry.building.ground_level))
}

/// How many mining levels this shaft ever reaches: every level whose floor
/// is still at or above [`Mine::min_level_y`], counting from level 0 —
/// ticket 117's "total = levels from level_floor(0) down to min_level_y".
fn total_levels(frame: &MineFrame, mine: &Mine) -> u32 {
    let diff = frame.level_floor(0) - mine.min_level_y;
    (diff / frame.level_spacing()) as u32 + 1
}

/// The panel's "Level" line: the level currently being sunk toward or
/// mined, 1-based of the shaft's total, and the world Y its floor sits (or
/// will sit) at.
fn level_line(frame: &MineFrame, mine: &Mine, progress: &MineProgress) -> String {
    let total = total_levels(frame, mine);
    let (index, floor_y) = match &progress.phase {
        Phase::Sinking { target } => (frame.level_at_floor(*target).unwrap_or(0), *target),
        Phase::Mining(cursor) => (cursor.level, progress.bottom),
        Phase::MinedOut => (total - 1, frame.level_floor(total - 1)),
    };
    format!("Level: {} of {total} (floor Y {floor_y})", index + 1)
}

/// The panel's "Shaft" line: the primary shaft's current bottom and how far
/// it has been sunk from the surface.
fn shaft_line(frame: &MineFrame, progress: &MineProgress) -> String {
    format!("Shaft: bottom Y {}, {} blocks deep", progress.bottom, frame.floor_y - progress.bottom)
}

fn arm_name(arm: Arm) -> &'static str {
    match arm {
        Arm::North => "north",
        Arm::South => "south",
    }
}

/// How many of a level's `rows_per_arm * 4` galleries are closed (dug to
/// their end, void-run-stopped, refused or bedrock — the panel doesn't
/// distinguish, ticket 117) as of `cursor`'s position. Rows behind the
/// cursor are always fully resolved one way or another (dug to completion,
/// or never opened because their arm closed first — see `MINES_DESIGN.md`'s
/// "row 0 north, row 0 south, row 1 north, …"), so they count in full; the
/// current row counts whichever of its two arms the cursor has already
/// finished or is presently working.
fn closed_galleries(cursor: &LevelCursor) -> u32 {
    let side_closed = |side: Arm| -> u32 {
        if cursor.arm_closed[side.index()] {
            return 2;
        }
        if cursor.arm != side {
            // North always precedes South within a row; a side that isn't
            // the cursor's current one and isn't closed is either done
            // (South's turn, North finished) or not started yet (North's
            // turn, South waiting).
            return if side == Arm::North { 2 } else { 0 };
        }
        match &cursor.step {
            RowStep::Galleries { faces, .. } => faces.iter().filter(|f| f.closed).count() as u32,
            RowStep::Secondary => 0,
        }
    };
    cursor.row * 4 + side_closed(Arm::North) + side_closed(Arm::South)
}

/// The panel's "Phase" line: which of mining/sinking/mined-out this mine is
/// in, with that phase's own progress detail.
fn phase_line(frame: &MineFrame, mine: &Mine, progress: &MineProgress) -> String {
    match &progress.phase {
        Phase::Mining(cursor) => {
            let level = frame.level(cursor.level);
            let rows_per_arm = level.rows_per_arm(mine.level_reach as i32);
            let total_galleries = rows_per_arm * 4;
            let closed = closed_galleries(cursor);
            format!(
                "Phase: mining — north arm {} m, south arm {} m, row {} of {rows_per_arm} ({}), galleries {closed} of {total_galleries} closed",
                cursor.arm_reach[Arm::North.index()],
                cursor.arm_reach[Arm::South.index()],
                cursor.row + 1,
                arm_name(cursor.arm),
            )
        }
        Phase::Sinking { target } => {
            let flights = (progress.bottom - target) / frame.level_spacing();
            let plural = if flights == 1 { "" } else { "s" };
            format!("Phase: sinking to Y {target} ({flights} flight{plural} left)")
        }
        Phase::MinedOut => "Phase: mined out".to_string(),
    }
}

/// The panel's "Job" line — only shown while a job is actually pending for
/// this building (ticket 117): the same "in flight" hint a gatherer's dig
/// has none of, useful here since a mine job can take a moment.
fn job_line(budget: u32) -> String {
    format!("Job: digging ({budget} blocks budget)")
}

/// Egui window: the selected building's name, position, rotation, and — if
/// it's a producer — its running state, warehouse connection, and buffer
/// (with a "Clear buffer" debug button, ticket 107); for a gatherer, its
/// working area and the buttons to draw or clear one (ticket 111); for a
/// mine, its level/shaft/phase/job progress (ticket 117). Draws nothing at
/// all with nothing selected; see the module docs.
#[allow(clippy::too_many_arguments)]
pub(super) fn inspect_panel(
    mut contexts: EguiContexts,
    selected: Res<SelectedBuilding>,
    mut city: ResMut<state::City>,
    definitions: Res<BuildingDefinitions>,
    mut production: ResMut<ProductionState>,
    coverage: Option<Res<Coverage>>,
    economy: Res<EconomyConfig>,
    mine_state: Option<Res<MineState>>,
    mut tool: Option<ResMut<ActiveTool>>,
    catalogue: Option<Res<BuildingCatalogue>>,
    world: Res<DecodedWorld>,
    journal: Res<Journal>,
) {
    let Some(id) = selected.0 else { return };
    // A stale selection (the building was demolished since) draws nothing —
    // the same "clears itself out" behaviour a fresh click on empty ground
    // gives it, just arrived at without one.
    let Some(placed) = city.building(id) else { return };

    let mut clear_buffer = false;
    let mut work_area = WorkAreaRequest::default();
    let drawing = matches!(tool.as_deref(), Some(ActiveTool::DrawWorkArea));

    egui::Window::new("Inspect").show(contexts.ctx_mut(), |ui| {
        ui.label(building_name(id, &placed.catalogue_id, &city, &definitions));
        ui.label(format!("Position: ({}, {}, {})", placed.origin.x, placed.origin.y, placed.origin.z));
        ui.label(format!("Rotation: {}°", rotation_degrees(placed.rotation)));

        // Ticket 128: a site isn't producing anything yet — its line
        // replaces the producer section rather than sitting beside it.
        if let Some(line) = site_clearing_line(id, placed, catalogue.as_deref(), &world, &journal, &economy) {
            ui.separator();
            ui.label(line);
        } else if let Some(producer) = production.get(id) {
            ui.separator();
            ui.label(format!("State: {}", producer.state.label()));

            match warehouse_status(id, coverage.as_deref(), &city, &definitions) {
                Some(line) => {
                    ui.label(line);
                }
                None => {
                    ui.colored_label(egui::Color32::from_rgb(220, 60, 60), "⚠ Not connected to a warehouse");
                }
            }

            let capacity = producer_capacity(placed, &definitions, &economy);
            if producer.buffer.is_empty() {
                ui.label("Buffer: (empty)");
            } else {
                ui.label("Buffer:");
                for line in buffer_lines(producer, capacity) {
                    ui.label(line);
                }
                if ui.button("Clear buffer").clicked() {
                    clear_buffer = true;
                }
            }
        }

        if is_gatherer(placed, &definitions) {
            ui.separator();
            match placed.work_area {
                Some(area) => {
                    ui.label(work_area_line(area));
                }
                None => {
                    ui.colored_label(egui::Color32::from_rgb(220, 60, 60), "⚠ No working area - draw one");
                }
            }
            ui.horizontal(|ui| {
                let label = if drawing { "Drawing... (Escape to cancel)" } else { "Draw working area" };
                if ui.add_enabled(!drawing, egui::Button::new(label)).clicked() {
                    work_area.draw = true;
                }
                if placed.work_area.is_some() && ui.button("Clear working area").clicked() {
                    work_area.clear = true;
                }
            });
        }

        if let Some((mine, ground_level)) = mine_definition(placed, &definitions) {
            let frame = MineFrame::from_placement(placed, mine, ground_level);
            let fresh_progress;
            let progress = match mine_state.as_deref().and_then(|state| state.progress.get(&id)) {
                Some(progress) => progress,
                None => {
                    fresh_progress = MineProgress::new(&frame);
                    &fresh_progress
                }
            };

            ui.separator();
            ui.label("Mine");
            ui.label(level_line(&frame, mine, progress));
            ui.label(shaft_line(&frame, progress));
            ui.label(phase_line(&frame, mine, progress));
            if let Some(budget) = mine_state.as_deref().and_then(|state| state.pending_budget(id)) {
                ui.label(job_line(budget));
            }
        }
    });

    if clear_buffer {
        production.entry(id).buffer = Parcel::default();
    }
    if work_area.clear {
        city.set_work_area(id, None);
    }
    if work_area.draw {
        if let Some(tool) = tool.as_deref_mut() {
            *tool = ActiveTool::DrawWorkArea;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::{
        Building, Category, FootprintSpec, Gatherer, Integrity, LoadedBuilding, Production, ProductionItem, Warehouse,
    };
    use crate::city::production::ProducerState;
    use crate::city::road::RoadPieceVariant;
    use crate::city::road_definition::RoadTypes;
    use crate::city::state::ROAD_CELL_SIZE;
    use crate::city::warehouse::compute_coverage;
    use std::path::PathBuf;

    fn placement(catalogue_id: &str, definition_id: Option<&str>) -> PlacedBuilding {
        PlacedBuilding {
            catalogue_id: catalogue_id.to_string(),
            definition_id: definition_id.map(str::to_string),
            origin: IVec3::new(1, 64, 2),
            rotation: Rotation::Deg90,
            footprint: IVec2::ONE,
            work_area: None,
            under_construction: false,
        }
    }

    /// A [`state::City`] holding one building placed with `catalogue_id`/
    /// `definition_id`, plus the id it was placed under — [`building_name`]
    /// reads through [`state::City::definition_of`], not `PlacedBuilding`
    /// directly, so its tests need a real `City` to look the id up in.
    fn city_with(catalogue_id: &str, definition_id: Option<&str>) -> (state::City, state::BuildingId) {
        let mut city = state::City::default();
        let id = city
            .place_building(catalogue_id, definition_id.map(str::to_string), IVec3::new(1, 64, 2), Rotation::Deg90, IVec2::ONE)
            .unwrap();
        (city, id)
    }

    // --- building_name -------------------------------------------------------

    #[test]
    fn building_name_prefers_the_definitions_own_name() {
        let building = Building {
            name: "Lumberjack's Hut".to_string(),
            blueprint: "lumber.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: None,
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: None,
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        let defs = BuildingDefinitions::from_entries(vec![LoadedBuilding {
            id: "lumber".to_string(),
            path: PathBuf::new(),
            building,
            footprint: IVec2::ONE,
            catalogue_id: "lumber".to_string(),
        }]);

        let (city, id) = city_with("lumber", Some("lumber"));
        assert_eq!(building_name(id, "lumber", &city, &defs), "Lumberjack's Hut");
    }

    #[test]
    fn building_name_falls_back_to_the_catalogue_id_with_no_definition() {
        let defs = BuildingDefinitions::default();
        let (city, id) = city_with("house01", None);
        assert_eq!(building_name(id, "house01", &city, &defs), "house01");
    }

    // --- buffer_lines ----------------------------------------------------------

    #[test]
    fn buffer_lines_are_counted_against_the_shared_capacity() {
        let mut producer = Producer::default();
        producer.buffer.add("minecraft:oak_log", 37);
        assert_eq!(buffer_lines(&producer, 256), vec!["  oak_log: 37/256".to_string()]);
    }

    #[test]
    fn an_empty_buffer_has_no_lines() {
        assert!(buffer_lines(&Producer::default(), 256).is_empty());
    }

    // --- producer_capacity ------------------------------------------------------

    #[test]
    fn producer_capacity_reads_the_definitions_own_buffer_size() {
        let building = Building {
            name: "hut".to_string(),
            blueprint: "hut.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: Some(Production {
                outputs: vec![ProductionItem { item: "minecraft:oak_log".to_string(), per_minute: 8.0 }],
                inputs: Vec::new(),
                radius: None,
                buffer_stacks: 4,
                haul_at_stacks: None,
            }),
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: None,
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        let defs = BuildingDefinitions::from_entries(vec![LoadedBuilding {
            id: "hut".to_string(),
            path: PathBuf::new(),
            building,
            footprint: IVec2::ONE,
            catalogue_id: "hut".to_string(),
        }]);
        let placed = placement("hut", Some("hut"));
        let economy = EconomyConfig::default();
        let expected = buffer_capacity(defs.get("hut").unwrap().building.production.as_ref().unwrap(), &economy);

        assert_eq!(producer_capacity(&placed, &defs, &economy), expected);
    }

    #[test]
    fn producer_capacity_is_zero_with_no_definition() {
        let placed = placement("house01", None);
        assert_eq!(producer_capacity(&placed, &BuildingDefinitions::default(), &EconomyConfig::default()), 0);
    }

    // Sanity: `ProducerState::label` is what the panel prints for "State:" —
    // pinned here so a relabel doesn't silently drift.
    // --- the working-area section (ticket 111) ---------------------------------------------

    // --- site_clearing_line (ticket 128) --------------------------------------

    fn small_catalogue(id: &str, size: IVec3) -> BuildingCatalogue {
        let dir = std::env::temp_dir().join(format!("block_viewer_inspect_site_{}_{}", id, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let volume = (size.x * size.y * size.z) as usize;
        let blueprint = crate::blueprint::Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![crate::blueprint::BlockState::air(), "minecraft:stone".parse().unwrap()],
            blocks: vec![1; volume],
            data_version: 0,
            failed_columns: 0,
        };
        crate::blueprint::write_structure_file(&dir.join(format!("{id}.nbt")), &blueprint).unwrap();
        let (catalogue, _) = crate::blueprint::load_catalogue_dir(&dir);
        std::fs::remove_dir_all(&dir).ok();
        catalogue
    }

    fn empty_world() -> DecodedWorld {
        DecodedWorld {
            registry: std::sync::Arc::new(std::sync::Mutex::new(crate::world::BlockRegistry::new())),
            biomes: std::sync::Arc::new(std::sync::Mutex::new(crate::world::BiomeRegistry::new())),
            columns: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn site_clearing_line_is_none_for_a_completed_building() {
        let placed = placement("house01", Some("house01"));
        let catalogue = small_catalogue("house01", IVec3::new(1, 1, 1));
        assert!(site_clearing_line(
            state::BuildingId::from_u64(0),
            &placed,
            Some(&catalogue),
            &empty_world(),
            &Journal::default(),
            &EconomyConfig::default(),
        )
        .is_none());
    }

    #[test]
    fn site_clearing_line_reports_the_remaining_and_total_block_count() {
        let mut placed = placement("house01", Some("house01"));
        placed.under_construction = true;
        placed.origin = IVec3::new(500, 64, 500); // an undecoded column: nothing left to clear
        let catalogue = small_catalogue("house01", IVec3::new(2, 1, 2));
        let id = state::BuildingId::from_u64(0);

        let mut journal = Journal::default();
        journal.record_placement(
            id,
            placed.clone(),
            super::super::super::journal::Baseline {
                written: vec![(IVec3::ZERO, crate::blueprint::BlockState::air())],
                previous: vec![(IVec3::ZERO, "minecraft:stone".parse().unwrap())],
                data_version: None,
            },
            crate::city::journal::Ledger::default(),
        );

        let line = site_clearing_line(id, &placed, Some(&catalogue), &empty_world(), &journal, &EconomyConfig::default())
            .expect("a site under construction has a line");
        assert_eq!(line, "Clearing site: 0 of 1 block(s) (~0.0 min left)");
    }

    #[test]
    fn work_area_line_names_both_corners_and_the_tile_count() {
        let area = WorkArea::new(IVec2::new(-3, 2), IVec2::new(4, 5));
        assert_eq!(work_area_line(area), "Working area: (-3, 2) - (4, 5), 32 tiles");
    }

    #[test]
    fn is_gatherer_reads_the_definitions_gatherer_block() {
        let definitions = warehouse_status_defs();
        assert!(!is_gatherer(&placement("warehouse01", Some("warehouse01")), &definitions));
        assert!(!is_gatherer(&placement("house01", None), &definitions));
        assert!(is_gatherer(&placement("gatherer_hut", Some("gatherer_hut")), &definitions));
    }

    #[test]
    fn producer_state_labels_read_as_a_sentence_fragment() {
        assert_eq!(ProducerState::Running.label(), "running");
    }

    // --- warehouse_status (ticket 107) --------------------------------------

    fn warehouse_status_defs() -> BuildingDefinitions {
        let warehouse = Building {
            name: "Warehouse".to_string(),
            blueprint: "warehouse.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: None,
            cost: Vec::new(),
            warehouse: Some(Warehouse { radius_cells: 8, concurrent_hauls: 2, handling_minutes: 0.0, storage: 4096 }),
            farm: None,
            gatherer: None,
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        let farm = Building {
            name: "Farm".to_string(),
            blueprint: "farm.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: Some(Production {
                outputs: vec![ProductionItem { item: "minecraft:wheat".to_string(), per_minute: 12.0 }],
                inputs: Vec::new(),
                radius: None,
                buffer_stacks: 4,
                haul_at_stacks: None,
            }),
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: None,
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        let hut = Building {
            name: "Gatherer's Hut".to_string(),
            blueprint: "gatherer_hut.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: None,
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: Some(Gatherer { radius_blocks: 18, blocks_per_minute: 20.0, buffer_stacks: 4, haul_at_stacks: None }),
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        BuildingDefinitions::from_entries(vec![
            LoadedBuilding { id: "warehouse01".to_string(), path: PathBuf::new(), building: warehouse, footprint: IVec2::ONE, catalogue_id: "warehouse01".to_string() },
            LoadedBuilding { id: "farm01".to_string(), path: PathBuf::new(), building: farm, footprint: IVec2::ONE, catalogue_id: "farm01".to_string() },
            LoadedBuilding { id: "gatherer_hut".to_string(), path: PathBuf::new(), building: hut, footprint: IVec2::ONE, catalogue_id: "gatherer_hut".to_string() },
        ])
    }

    #[test]
    fn warehouse_status_names_the_serving_warehouse_and_its_travel_time() {
        let mut city = state::City::default();
        for x in 0..=1 {
            city.add_road_cell(IVec2::new(x, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        }
        let at = |cell: i32| IVec3::new(cell * ROAD_CELL_SIZE, 64, -1);
        city.place_building("warehouse01", Some("warehouse01".to_string()), at(0), Rotation::Deg0, IVec2::ONE).unwrap();
        let farm =
            city.place_building("farm01", Some("farm01".to_string()), at(1), Rotation::Deg0, IVec2::ONE).unwrap();

        let defs = warehouse_status_defs();
        let coverage = compute_coverage(&city, &defs, &RoadTypes::default());

        let status =
            warehouse_status(farm, Some(&coverage), &city, &defs).expect("the farm is one cell from the warehouse");
        assert!(status.starts_with("Warehouse: Warehouse ("), "{status}");
    }

    #[test]
    fn warehouse_status_warns_when_nothing_serves_the_producer() {
        let mut city = state::City::default();
        let farm = city
            .place_building("farm01", Some("farm01".to_string()), IVec3::new(500, 64, 500), Rotation::Deg0, IVec2::ONE)
            .unwrap();

        let defs = warehouse_status_defs();
        let coverage = compute_coverage(&city, &defs, &RoadTypes::default());

        assert!(warehouse_status(farm, Some(&coverage), &city, &defs).is_none());
    }

    /// The tolerant `Option<Res<Coverage>>` shape: a minimal test `App` that
    /// never adds `WarehousePlugin` must read as "not connected", not panic.
    #[test]
    fn warehouse_status_is_none_with_no_coverage_resource() {
        let city = state::City::default();
        let defs = BuildingDefinitions::default();
        assert!(warehouse_status(state::BuildingId::from_u64(0), None, &city, &defs).is_none());
    }

    // --- the mine section (ticket 117) ------------------------------------

    use crate::city::definition::ShaftAt;
    use crate::city::mine::layout::GallerySide;
    use crate::city::mine::progress::Face;

    fn mine_defs() -> BuildingDefinitions {
        let mine = Mine {
            shaft: ShaftAt { x: 5, z: 5 },
            shaft_size: 6,
            first_level_depth: 12,
            min_level_y: 16,
            level_reach: 100,
            gallery_length: 200,
            torch_spacing: 8,
            max_void_run: 6,
            blocks_per_minute: 60.0,
            buffer_stacks: 512,
            haul_at_stacks: Some(64),
            valuables: Vec::new(),
        };
        let building = Building {
            name: "Mine".to_string(),
            blueprint: "mine.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: None,
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: None,
            mine: Some(mine),
            category: Category::Production,
            ground_level: 2,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        BuildingDefinitions::from_entries(vec![LoadedBuilding {
            id: "mine01".to_string(),
            path: PathBuf::new(),
            building,
            footprint: IVec2::ONE,
            catalogue_id: "mine01".to_string(),
        }])
    }

    fn mine_frame() -> MineFrame {
        MineFrame { shaft_min: IVec2::new(100, 200), shaft_size: 6, floor_y: 64, first_level_depth: 12 }
    }

    #[test]
    fn mine_definition_is_none_for_a_non_mine_building() {
        let definitions = warehouse_status_defs();
        assert!(mine_definition(&placement("warehouse01", Some("warehouse01")), &definitions).is_none());
        assert!(mine_definition(&placement("house01", None), &definitions).is_none());
    }

    #[test]
    fn mine_definition_reads_the_mine_block_and_ground_level() {
        let defs = mine_defs();
        let (mine, ground_level) =
            mine_definition(&placement("mine01", Some("mine01")), &defs).expect("mine01 has a mine block");
        assert_eq!(mine.min_level_y, 16);
        assert_eq!(ground_level, 2);
    }

    #[test]
    fn total_levels_counts_every_level_floor_down_to_min_level_y() {
        // level_floor(0) = 64 - 12 = 52, spacing = 6 - 2 = 4: 52, 48, ..., 16 is 10 levels.
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        assert_eq!(total_levels(&f, mine), 10);
    }

    #[test]
    fn level_line_shows_the_current_mining_level_one_based() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let cursor = LevelCursor {
            level: 1,
            row: 0,
            arm: Arm::North,
            step: RowStep::Secondary,
            arm_reach: [0, 0],
            arm_closed: [false, false],
        };
        let floor = f.level_floor(1);
        let progress = MineProgress { bottom: floor, phase: Phase::Mining(cursor) };
        assert_eq!(level_line(&f, mine, &progress), format!("Level: 2 of 10 (floor Y {floor})"));
    }

    #[test]
    fn level_line_shows_the_level_being_sunk_toward() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let target = f.level_floor(2);
        let progress = MineProgress { bottom: target + f.level_spacing(), phase: Phase::Sinking { target } };
        assert_eq!(level_line(&f, mine, &progress), format!("Level: 3 of 10 (floor Y {target})"));
    }

    #[test]
    fn level_line_shows_the_last_level_once_mined_out() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let progress = MineProgress { bottom: 16, phase: Phase::MinedOut };
        assert_eq!(level_line(&f, mine, &progress), "Level: 10 of 10 (floor Y 16)");
    }

    #[test]
    fn shaft_line_reports_bottom_and_depth() {
        let f = mine_frame();
        let progress = MineProgress { bottom: 44, phase: Phase::MinedOut };
        assert_eq!(shaft_line(&f, &progress), "Shaft: bottom Y 44, 20 blocks deep");
    }

    #[test]
    fn phase_line_reports_sinking_with_flights_left() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let progress = MineProgress { bottom: 44, phase: Phase::Sinking { target: 40 } };
        assert_eq!(phase_line(&f, mine, &progress), "Phase: sinking to Y 40 (1 flight left)");
    }

    #[test]
    fn phase_line_reports_mined_out() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let progress = MineProgress { bottom: 16, phase: Phase::MinedOut };
        assert_eq!(phase_line(&f, mine, &progress), "Phase: mined out");
    }

    #[test]
    fn phase_line_reports_a_fresh_row_as_no_galleries_closed() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let cursor = LevelCursor {
            level: 0,
            row: 0,
            arm: Arm::North,
            step: RowStep::Secondary,
            arm_reach: [0, 0],
            arm_closed: [false, false],
        };
        let progress = MineProgress { bottom: f.level_floor(0), phase: Phase::Mining(cursor) };
        assert_eq!(
            phase_line(&f, mine, &progress),
            "Phase: mining — north arm 0 m, south arm 0 m, row 1 of 25 (north), galleries 0 of 100 closed"
        );
    }

    #[test]
    fn phase_line_counts_finished_rows_in_full_and_the_current_rows_own_faces() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let faces = [
            Face { distance: 5, void_run: 0, closed: true },
            Face { distance: 2, void_run: 0, closed: false },
        ];
        let cursor = LevelCursor {
            level: 0,
            row: 3,
            arm: Arm::South,
            step: RowStep::Galleries { faces, next: GallerySide::West },
            arm_reach: [36, 32],
            arm_closed: [false, false],
        };
        let progress = MineProgress { bottom: f.level_floor(0), phase: Phase::Mining(cursor) };
        // 3 finished rows (12) + north done this row (2) + one closed south face (1) = 15.
        assert_eq!(
            phase_line(&f, mine, &progress),
            "Phase: mining — north arm 36 m, south arm 32 m, row 4 of 25 (south), galleries 15 of 100 closed"
        );
    }

    #[test]
    fn phase_line_counts_a_permanently_closed_arm_as_closed_for_every_later_row() {
        let f = mine_frame();
        let defs = mine_defs();
        let mine = defs.get("mine01").unwrap().building.mine.as_ref().unwrap();
        let cursor = LevelCursor {
            level: 0,
            row: 5,
            arm: Arm::South,
            step: RowStep::Secondary,
            arm_reach: [8, 20],
            arm_closed: [true, false],
        };
        let progress = MineProgress { bottom: f.level_floor(0), phase: Phase::Mining(cursor) };
        // 5 finished rows (20) + north permanently closed (2) + south not started this row (0) = 22.
        assert_eq!(
            phase_line(&f, mine, &progress),
            "Phase: mining — north arm 8 m, south arm 20 m, row 6 of 25 (south), galleries 22 of 100 closed"
        );
    }

    #[test]
    fn job_line_names_the_pending_budget() {
        assert_eq!(job_line(96), "Job: digging (96 blocks budget)");
    }
}
