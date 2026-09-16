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

use super::super::definition::BuildingDefinitions;
use super::super::economy::EconomyConfig;
use super::super::gatherer::gatherer_buffer_capacity;
use super::super::inventory::{short_name, Parcel};
use super::super::picking::SelectedBuilding;
use super::super::placement::rotation_degrees;
use super::super::production::{buffer_capacity, Producer, ProductionState};
use super::super::state::{self, PlacedBuilding, WorkArea};
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

/// What the panel's working-area buttons asked for this frame — resolved
/// after the window closes, the way `clear_buffer` is.
#[derive(Default)]
struct WorkAreaRequest {
    draw: bool,
    clear: bool,
}

/// Egui window: the selected building's name, position, rotation, and — if
/// it's a producer — its running state, warehouse connection, and buffer
/// (with a "Clear buffer" debug button, ticket 107); for a gatherer, its
/// working area and the buttons to draw or clear one (ticket 111). Draws
/// nothing at all with nothing selected; see the module docs.
#[allow(clippy::too_many_arguments)]
pub(super) fn inspect_panel(
    mut contexts: EguiContexts,
    selected: Res<SelectedBuilding>,
    mut city: ResMut<state::City>,
    definitions: Res<BuildingDefinitions>,
    mut production: ResMut<ProductionState>,
    coverage: Option<Res<Coverage>>,
    economy: Res<EconomyConfig>,
    mut tool: Option<ResMut<ActiveTool>>,
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

        if let Some(producer) = production.get(id) {
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
}
