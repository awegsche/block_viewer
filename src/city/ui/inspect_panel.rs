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

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::super::definition::BuildingDefinitions;
use super::super::economy::EconomyConfig;
use super::super::gatherer::gatherer_buffer_capacity;
use super::super::inventory::short_name;
use super::super::picking::SelectedBuilding;
use super::super::placement::rotation_degrees;
use super::super::production::{buffer_capacity, Producer, ProductionState};
use super::super::state::{self, PlacedBuilding};

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

/// Egui window: the selected building's name, position, rotation, and — if
/// it's a producer — its running state and buffer. Draws nothing at all with
/// nothing selected; see the module docs.
pub(super) fn inspect_panel(
    mut contexts: EguiContexts,
    selected: Res<SelectedBuilding>,
    city: Res<state::City>,
    definitions: Res<BuildingDefinitions>,
    production: Res<ProductionState>,
    economy: Res<EconomyConfig>,
) {
    let Some(id) = selected.0 else { return };
    // A stale selection (the building was demolished since) draws nothing —
    // the same "clears itself out" behaviour a fresh click on empty ground
    // gives it, just arrived at without one.
    let Some(placed) = city.building(id) else { return };

    egui::Window::new("Inspect").show(contexts.ctx_mut(), |ui| {
        ui.label(building_name(id, &placed.catalogue_id, &city, &definitions));
        ui.label(format!("Position: ({}, {}, {})", placed.origin.x, placed.origin.y, placed.origin.z));
        ui.label(format!("Rotation: {}°", rotation_degrees(placed.rotation)));

        if let Some(producer) = production.get(id) {
            ui.separator();
            ui.label(format!("State: {}", producer.state.label()));
            let capacity = producer_capacity(placed, &definitions, &economy);
            if producer.buffer.is_empty() {
                ui.label("Buffer: (empty)");
            } else {
                ui.label("Buffer:");
                for line in buffer_lines(producer, capacity) {
                    ui.label(line);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::{Building, Category, FootprintSpec, Integrity, LoadedBuilding, Production, ProductionItem};
    use crate::city::production::ProducerState;
    use std::path::PathBuf;

    fn placement(catalogue_id: &str, definition_id: Option<&str>) -> PlacedBuilding {
        PlacedBuilding {
            catalogue_id: catalogue_id.to_string(),
            definition_id: definition_id.map(str::to_string),
            origin: IVec3::new(1, 64, 2),
            rotation: Rotation::Deg90,
            footprint: IVec2::ONE,
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
    #[test]
    fn producer_state_labels_read_as_a_sentence_fragment() {
        assert_eq!(ProducerState::Running.label(), "running");
    }
}
