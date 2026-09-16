//! The build menu (ticket 050, roadmap G1): the catalogue grouped by tier,
//! locked entries visible but disabled and showing what unlocks them, costs
//! and production shown from C1's data even while iteration 1 leaves both
//! inert.
//!
//! ## Replaces the number-key stand-in, keeps the rest
//!
//! Ticket 047's `city::placement::cycle_selection` picked a catalogue entry
//! with `1`-`9`, explicitly as a stand-in "until a real menu exists" — this
//! is that menu, and clicking an entry here sets exactly the same
//! [`PlacementSelection::catalogue_id`] the number keys did. Rotation
//! (`R`), height (`PageUp`/`PageDown`/`Home`) and clearing (`Escape`) stay on
//! the keyboard; a build menu doesn't need to reinvent those, only "which
//! building".
//!
//! ## Selecting by definition, placing by catalogue id
//!
//! [`PlacementSelection::catalogue_id`] has always named a
//! [`BuildingCatalogue`](crate::blueprint::BuildingCatalogue) entry — the raw
//! shape — not a [`BuildingDefinitions`] entry — the game data. A menu built
//! off definitions (for tier/cost/production/`requires`) has to bridge the
//! two on every click, which is exactly what
//! [`LoadedBuilding::catalogue_id`] (ticket 050's own addition to
//! `definition.rs`) is for: resolved once at load time from
//! `Building::blueprint`'s filename stem, rather than this panel re-deriving
//! it from a path every frame.
//!
//! ## Unlocking, defined for the first time here
//!
//! Nothing before this ticket ever *read* [`Building::requires`] outside of
//! ticket 041's own cycle/dangling validation — no economy, no separate
//! "unlocked techs" resource exists in iteration 1 (see the roadmap's C3:
//! "production simulation... nothing consumes it"). The definition this
//! panel needs, and the smallest one that uses data already on hand: a
//! requirement is met once at least one building of that type has actually
//! been placed — [`missing_requirements`] checks
//! [`state::City::buildings`]'s own `definition` field, the same "has this
//! type been built" signal a real tech tree would gate on, without inventing
//! a second piece of state to track it in.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::super::definition::{Building, BuildingDefinitions, Category, Cost, LoadedBuilding, Production};
use super::super::economy::{self, EconomyConfig};
use super::super::inventory::Stock;
use super::super::placement::{rotation_degrees, PlacementSelection};
use super::super::road_build::RoadStyleSelection;
use super::super::road_definition::{LoadedRoadType, RoadTypes};
use super::super::state;
use super::super::tool::ActiveTool;

/// The build menu's sections, outer to inner (ticket 082, roadmap G1) — a
/// player thinking "I want to build a farm" goes to one section rather than
/// scanning every tier for it. `Building::tier` stays the sub-heading within
/// a section.
const CATEGORIES: [Category; 3] = [Category::Production, Category::Residential, Category::Street];

fn category_label(category: Category) -> &'static str {
    match category {
        Category::Production => "Production",
        Category::Residential => "Residential",
        Category::Street => "Street",
    }
}

/// A block or item id's short display form — `"minecraft:oak_planks"` ->
/// `"oak_planks"`. Every id this panel shows is already namespaced
/// `minecraft:`, and repeating that prefix on every cost line would be pure
/// noise in a menu meant to be scanned quickly.
fn short_name(name: &str) -> &str {
    name.strip_prefix("minecraft:").unwrap_or(name)
}

/// `Building::cost` as one line — `"Free"` for an empty list (iteration 1's
/// starter buildings, per the roadmap's own sketch) rather than a blank line
/// that reads as a loading glitch.
fn cost_line(cost: &[Cost]) -> String {
    if cost.is_empty() {
        return "Free".to_string();
    }
    cost.iter().map(|c| format!("{}x {}", c.count, short_name(&c.block))).collect::<Vec<_>>().join(", ")
}

/// `Building::production`'s outputs as one line, `None` for iteration 1's
/// non-functional buildings (`production: None`) — the roadmap's own
/// wording, "C1 parses and displays these; nothing simulates them yet",
/// which is exactly what this line is: a display, not a rate anything reads.
fn production_line(production: &Production) -> Option<String> {
    if production.outputs.is_empty() {
        return None;
    }
    Some(production.outputs.iter().map(|item| format!("{} {:.1}/min", item.item, item.per_minute)).collect::<Vec<_>>().join(", "))
}

/// [`Building::farm`]'s numbers as one line, for a row whose `production` is
/// scaled rather than fixed (ticket 084) — "Scales with Lumberjack's Farm
/// Tile nearby (3 needed, within 16 tiles)". Shown on the catalogue row
/// itself rather than only once placed: a player deciding whether to build
/// the hub at all needs to know it does nothing without fields around it,
/// before they've committed the tile it costs.
fn farm_line(farm: &super::super::definition::Farm, definitions: &BuildingDefinitions) -> String {
    let tile_name = definitions.get(&farm.tile).map(|entry| entry.building.name.clone()).unwrap_or_else(|| farm.tile.clone());
    format!(
        "  Scales with {tile_name} nearby ({} needed, within {} tiles)",
        farm.tiles_for_full_rate, farm.radius_blocks,
    )
}

/// Every `requires` id `building` names that no placed building's own
/// definition currently satisfies — empty means unlocked. See the module
/// docs' "Unlocking, defined for the first time here". A plain function, not
/// a system, so it's testable directly against a bare [`state::City`] the
/// same way [`crate::city::placement::resolve_placement`] is.
///
/// Ticket 076: a `requires` entry is a *definition* id, so this compares
/// against [`state::PlacedBuilding::definition_id`] and not, as it did until
/// then, against the placement's catalogue id — a comparison that only
/// happened to work while every definition's `.ron` stem matched its
/// blueprint's. A placement with no definition behind it (the keyboard
/// stand-in's) unlocks nothing, which is the same "no game data" it already
/// gets no cost and no production from.
/// `"10x oak_log"` — what a row's cost will eat out of the stock through
/// ticket 074's conversion table. Same `count`/short-name shape
/// [`cost_line`] uses, because it's read in the same glance.
fn conversion_line(consumed: &super::super::inventory::Parcel) -> String {
    consumed.iter().map(|(item, count)| format!("{count}x {}", short_name(item))).collect::<Vec<_>>().join(", ")
}

fn missing_requirements(building: &Building, city: &state::City) -> Vec<String> {
    building
        .requires
        .iter()
        .filter(|req| !city.buildings().any(|(_, placed)| placed.definition_id.as_deref() == Some(req.as_str())))
        .cloned()
        .collect()
}

/// [`BuildingDefinitions::iter`]'s entries matching `category`, sorted by
/// tier then id within it — `BuildingDefinitions` is keyed by a `HashMap`,
/// whose iteration order a menu can't be built on.
fn entries_in_category(definitions: &BuildingDefinitions, category: Category) -> Vec<&LoadedBuilding> {
    let mut entries: Vec<&LoadedBuilding> = definitions.iter().filter(|entry| entry.building.category == category).collect();
    sort_by_tier_then_id(&mut entries);
    entries
}

/// The comparison [`entries_in_category`] applies, split out so it's directly
/// testable against a hand-built `Vec<&LoadedBuilding>` — every field of
/// [`LoadedBuilding`] is public, but [`BuildingDefinitions`] itself has no
/// public constructor beyond loading real files off disk (see
/// `definition`'s own tests for that path), so a unit test for *ordering
/// alone* is cheaper built this way, the same reason
/// [`crate::city::placement::rotate_clockwise`] is tested apart from any
/// `PlacementSelection`.
fn sort_by_tier_then_id(entries: &mut [&LoadedBuilding]) {
    entries.sort_by(|a, b| a.building.tier.cmp(&b.building.tier).then_with(|| a.id.cmp(&b.id)));
}

/// A missing-requirement id's display name — the definition it names if one
/// loaded, the raw id otherwise (a dangling `requires` is caught at load
/// time by ticket 041's `resolve_requirements`, so this fallback is
/// unreachable through a real [`BuildingDefinitions`], but a menu is a bad
/// place to `expect()` on it).
fn requirement_label(id: &str, definitions: &BuildingDefinitions) -> String {
    definitions.get(id).map(|entry| entry.building.name.clone()).unwrap_or_else(|| id.to_string())
}

/// One row: name, footprint, cost, production, and either a click target (if
/// unlocked) or a disabled row naming what's missing (if not).
///
/// Clicking also switches `*tool` to [`ActiveTool::Building`] (ticket 082) —
/// the menu is now how placement mode is *entered*, mirroring the Street
/// section's own row switching to [`ActiveTool::Road`].
#[allow(clippy::too_many_arguments)]
fn entry_row(
    ui: &mut egui::Ui,
    entry: &LoadedBuilding,
    definitions: &BuildingDefinitions,
    selection: &mut PlacementSelection,
    city: &state::City,
    stock: &Stock,
    economy: &EconomyConfig,
    tool: &mut ActiveTool,
) {
    let missing = missing_requirements(&entry.building, city);
    let unlocked = missing.is_empty();
    let missing_names = || -> String { missing.iter().map(|id| requirement_label(id, definitions)).collect::<Vec<_>>().join(", ") };
    let selected = selection.catalogue_id.as_deref() == Some(entry.catalogue_id.as_str());

    let label = format!("{}  ({}x{})", entry.building.name, entry.footprint.x, entry.footprint.y);
    let response = ui.add_enabled(unlocked, egui::SelectableLabel::new(selected, label));
    if response.clicked() {
        selection.catalogue_id = Some(entry.catalogue_id.clone());
        // Ticket 073: the *definition*, so `city::commit` can find the cost
        // this row is displaying. The catalogue id above can't stand in for
        // it — see `PlacementSelection::definition_id`.
        selection.definition_id = Some(entry.id.clone());
        selection.y_offset = 0;
        *tool = ActiveTool::Building;
    }
    if !unlocked {
        response.on_disabled_hover_text(format!("Requires: {}", missing_names()));
    }

    // Ticket 073: the cost line turns red the moment the stockpile can't
    // cover it, and says what's short — a placement click would otherwise be
    // refused with the reason only in the city panel's last-edit line.
    //
    // Ticket 074: priced through the same `plan_payment` the commit pays
    // with, conversions included, so a row can't read red above a click that
    // succeeds. A row the player can only afford *by* converting says so —
    // materials disappearing out of the pile is worth a word of warning.
    let cost_text = format!("  Cost: {}", cost_line(&entry.building.cost));
    let payment = economy::plan_payment(stock, &entry.building.cost, economy);
    if !payment.affordable() {
        ui.colored_label(egui::Color32::RED, format!("{cost_text}  (short {})", payment.shortfall));
    } else if payment.conversion.consumed.is_empty() {
        ui.label(cost_text);
    } else {
        ui.label(cost_text);
        ui.colored_label(
            egui::Color32::from_rgb(220, 160, 90),
            format!("  Converts: {}", conversion_line(&payment.conversion.consumed)),
        );
    }
    if let Some(production) = &entry.building.production {
        if let Some(line) = production_line(production) {
            ui.label(format!("  Produces: {line}"));
        }
    }
    if let Some(farm) = &entry.building.farm {
        ui.colored_label(egui::Color32::from_rgb(150, 190, 220), farm_line(farm, definitions));
    }
    if !unlocked {
        ui.colored_label(egui::Color32::from_rgb(220, 160, 90), format!("  Locked — requires {}", missing_names()));
    }
}

/// The currently selected entry's own line, above the tiers — "what am I
/// about to place, and how" is worth one glance without scrolling a
/// collapsed tier open.
fn selected_line(ui: &mut egui::Ui, selection: &PlacementSelection, definitions: &BuildingDefinitions) {
    let Some(id) = selection.catalogue_id.as_deref() else {
        ui.label("(nothing selected — click a building below)");
        return;
    };
    let name = definitions
        .iter()
        .find(|entry| entry.catalogue_id == id)
        .map(|entry| entry.building.name.clone())
        .unwrap_or_else(|| id.to_string());
    ui.label(format!(
        "Selected: {name} — rotation {}°{}",
        rotation_degrees(selection.rotation),
        if selection.y_offset != 0 { format!(", height {:+}", selection.y_offset) } else { String::new() },
    ));
}

/// `R` rotate, `PageUp`/`PageDown`/`Home` height, `Delete` demolish, `Esc`
/// clear, `T` switch tool, `Z` switch dig/level — the keyboard half
/// ticket 047/048/049/055/057 already built, restated here so it's
/// discoverable from the one panel a player actually looks at while placing
/// something. Collapsed by default, the same call
/// `viewer::ui::selection_panel::key_legend` makes.
fn key_legend(ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Keys").show(ui, |ui| {
        egui::Grid::new("build_menu_key_legend").num_columns(2).show(ui, |ui| {
            ui.label("T");
            ui.label("switch tool (Building / Road / Terraform)");
            ui.end_row();
            ui.label("R");
            ui.label("rotate the selection 90°");
            ui.end_row();
            ui.label("Page Up / Page Down");
            ui.label("nudge the placement height");
            ui.end_row();
            ui.label("Home");
            ui.label("reset the height to the terrain's own fit");
            ui.end_row();
            ui.label("Delete");
            ui.label("demolish the hovered building");
            ui.end_row();
            ui.label("Esc");
            ui.label("clear the selection");
            ui.end_row();
            ui.label("Z");
            ui.label("switch Dig / Level (Terraform tool)");
            ui.end_row();
            ui.label("Left-click drag");
            ui.label("dig or level the dragged area (Terraform tool)");
            ui.end_row();
        });
    });
}

/// The `Street` section (ticket 082): one row per loaded [`RoadType`]
/// (`super::super::road_definition::RoadType`), not a [`BuildingDefinitions`]
/// entry — see the module docs' "Street category". Clicking a row selects
/// that style (mirroring `road_build`'s `[`/`]` stand-in, which keeps working
/// alongside this) and switches `*tool` to [`ActiveTool::Road`], the road
/// tool's counterpart of a building row switching to [`ActiveTool::Building`].
fn street_section(ui: &mut egui::Ui, road_types: Option<&RoadTypes>, style_selection: &mut RoadStyleSelection, tool: &mut ActiveTool) {
    let mut entries: Vec<&LoadedRoadType> = road_types.map(|types| types.iter().collect()).unwrap_or_default();
    entries.sort_by(|a, b| a.id.cmp(&b.id));

    if entries.is_empty() {
        // Today's actual state per ticket 060/063's own note: `dirt` has a
        // type file but no loaded geometry yet — see
        // `assets/city/roads/dirt/README.md`.
        ui.label("(no road styles loaded)");
        return;
    }

    for entry in entries {
        let selected = style_selection.current.as_deref() == Some(entry.id.as_str());
        let label =
            format!("{}  (speed {}, capacity {})", entry.road_type.name, entry.road_type.travel_speed, entry.road_type.capacity);
        if ui.add(egui::SelectableLabel::new(selected, label)).clicked() {
            style_selection.current = Some(entry.id.clone());
            *tool = ActiveTool::Road;
        }
    }
}

/// Egui window: the build menu. Empty definitions (nothing loaded, or an
/// `assets/city/buildings` directory that doesn't exist) shows a plain
/// message rather than an empty, confusing window — the same "(nothing
/// selected...)"-style tone the rest of the crate's panels use for an empty
/// state that isn't an error.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_menu_panel(
    mut contexts: EguiContexts,
    definitions: Option<Res<BuildingDefinitions>>,
    city: Res<state::City>,
    stock: Res<Stock>,
    economy: Res<EconomyConfig>,
    mut selection: ResMut<PlacementSelection>,
    road_types: Option<Res<RoadTypes>>,
    mut style_selection: Option<ResMut<RoadStyleSelection>>,
    mut tool: Option<ResMut<ActiveTool>>,
) {
    // `entry_row`/`street_section` need `&mut ActiveTool`/`RoadStyleSelection`
    // to switch tools on a click — both are optional resources (same
    // tolerant shape `tool::ActiveTool`'s own docs describe) so a minimal
    // test `App` that never adds `ToolPlugin`/`RoadBuildPlugin` doesn't panic
    // here. A fallback owned value absorbs the write when either is missing.
    let mut fallback_tool = ActiveTool::default();
    let tool = tool.as_deref_mut().unwrap_or(&mut fallback_tool);
    let mut fallback_style_selection = RoadStyleSelection::default();
    let style_selection = style_selection.as_deref_mut().unwrap_or(&mut fallback_style_selection);
    egui::Window::new("Build").show(contexts.ctx_mut(), |ui| {
        let Some(definitions) = definitions else {
            ui.label("(no building definitions loaded)");
            return;
        };
        if definitions.is_empty() {
            ui.label("(no buildings in assets/city/buildings)");
            return;
        }

        selected_line(ui, &selection, &definitions);
        if ui.button("Clear selection").clicked() {
            selection.catalogue_id = None;
            selection.definition_id = None;
            selection.y_offset = 0;
        }
        ui.separator();

        for category in CATEGORIES {
            ui.heading(category_label(category));
            if category == Category::Street {
                street_section(ui, road_types.as_deref(), &mut *style_selection, &mut *tool);
                ui.separator();
                continue;
            }

            let entries = entries_in_category(&definitions, category);
            if entries.is_empty() {
                ui.label("(nothing here yet)");
            } else {
                let mut current_tier = None;
                for entry in entries {
                    if current_tier != Some(entry.building.tier) {
                        current_tier = Some(entry.building.tier);
                        ui.strong(format!("Tier {}", entry.building.tier));
                    }
                    entry_row(ui, entry, &definitions, &mut selection, &city, &stock, &economy, &mut *tool);
                    ui.separator();
                }
            }
            ui.separator();
        }

        key_legend(ui);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::{Category, FootprintSpec, Integrity, ProductionItem};
    use std::path::PathBuf;

    fn building(name: &str, tier: u32, requires: Vec<&str>) -> Building {
        Building {
            name: name.to_string(),
            blueprint: format!("{name}.nbt"),
            tier,
            requires: requires.into_iter().map(str::to_string).collect(),
            footprint: FootprintSpec::FromBlueprint,
            production: None,
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        }
    }

    fn loaded(id: &str, building: Building) -> LoadedBuilding {
        let catalogue_id = building.blueprint.trim_end_matches(".nbt").to_string();
        LoadedBuilding { id: id.to_string(), path: PathBuf::new(), building, footprint: IVec2::new(2, 2), catalogue_id }
    }

    // --- missing_requirements --------------------------------------------

    #[test]
    fn no_requires_is_always_unlocked() {
        let city = state::City::default();
        assert!(missing_requirements(&building("house", 1, vec![]), &city).is_empty());
    }

    #[test]
    fn an_unmet_requirement_is_reported() {
        let city = state::City::default();
        let missing = missing_requirements(&building("carpenter", 2, vec!["house01"]), &city);
        assert_eq!(missing, vec!["house01".to_string()]);
    }

    #[test]
    fn a_requirement_is_met_once_that_type_is_placed() {
        let mut city = state::City::default();
        city.place_building("house01", Some("house01".to_string()), IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let missing = missing_requirements(&building("carpenter", 2, vec!["house01"]), &city);
        assert!(missing.is_empty());
    }

    #[test]
    fn only_a_matching_definition_id_satisfies_a_requirement() {
        let mut city = state::City::default();
        city.place_building(
            "some_other_building",
            Some("some_other_building".to_string()),
            IVec3::new(0, 64, 0),
            Rotation::Deg0,
            IVec2::ONE,
        )
        .unwrap();
        let missing = missing_requirements(&building("carpenter", 2, vec!["house01"]), &city);
        assert_eq!(missing, vec!["house01".to_string()]);
    }

    /// Ticket 076: the two ids are separate keyspaces, and a `requires` entry
    /// names a *definition*. A placement whose blueprint happens to be called
    /// `house01` but whose game data is `manor.ron` does not unlock what
    /// `house01.ron` unlocks — which is what this check silently got wrong
    /// while it compared against the catalogue id.
    #[test]
    fn a_matching_catalogue_id_alone_does_not_satisfy_a_requirement() {
        let mut city = state::City::default();
        city.place_building("house01", Some("manor".to_string()), IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let missing = missing_requirements(&building("carpenter", 2, vec!["house01"]), &city);
        assert_eq!(missing, vec!["house01".to_string()], "the requirement names house01.ron, not house01.nbt");
    }

    /// A placement made through `city::placement`'s keyboard stand-in has no
    /// definition behind it at all, so it unlocks nothing — the same "no game
    /// data" that already leaves it with no cost and no production.
    #[test]
    fn a_placement_with_no_definition_unlocks_nothing() {
        let mut city = state::City::default();
        city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        let missing = missing_requirements(&building("carpenter", 2, vec!["house01"]), &city);
        assert_eq!(missing, vec!["house01".to_string()]);
    }

    // --- sort_by_tier_then_id ------------------------------------------------

    #[test]
    fn entries_are_sorted_by_tier_then_id() {
        let a = loaded("zzz_tier1", building("Z", 1, vec![]));
        let b = loaded("aaa_tier1", building("A", 1, vec![]));
        let c = loaded("mid_tier2", building("M", 2, vec![]));
        let mut entries = vec![&a, &b, &c];

        sort_by_tier_then_id(&mut entries);

        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["aaa_tier1", "zzz_tier1", "mid_tier2"], "tier 1 before tier 2, alphabetical within a tier");
    }

    // --- entries_in_category (ticket 082) ------------------------------------

    #[test]
    fn entries_in_category_only_returns_matching_entries() {
        let mut farm = building("farm", 1, vec![]);
        farm.category = Category::Production;
        let mut house = building("house", 1, vec![]);
        house.category = Category::Residential;
        let defs = BuildingDefinitions::from_entries(vec![loaded("farm", farm), loaded("house", house)]);

        let production: Vec<&str> = entries_in_category(&defs, Category::Production).iter().map(|e| e.id.as_str()).collect();
        assert_eq!(production, vec!["farm"]);

        let residential: Vec<&str> = entries_in_category(&defs, Category::Residential).iter().map(|e| e.id.as_str()).collect();
        assert_eq!(residential, vec!["house"]);

        assert!(entries_in_category(&defs, Category::Street).is_empty(), "Street is never a BuildingDefinitions entry");
    }

    // --- cost_line / production_line ---------------------------------------

    #[test]
    fn an_empty_cost_reads_free() {
        assert_eq!(cost_line(&[]), "Free");
    }

    #[test]
    fn a_cost_is_formatted_count_and_short_name() {
        let cost = [Cost { block: "minecraft:oak_planks".to_string(), count: 40 }];
        assert_eq!(cost_line(&cost), "40x oak_planks");
    }

    #[test]
    fn a_conversion_line_reads_as_what_leaves_the_stock() {
        let mut consumed = super::super::super::inventory::Parcel::default();
        consumed.add("minecraft:oak_log", 10);
        assert_eq!(conversion_line(&consumed), "10x oak_log");
    }

    #[test]
    fn production_with_no_outputs_is_none() {
        let production = Production { outputs: vec![], inputs: vec![], radius: None, buffer_stacks: 4 };
        assert!(production_line(&production).is_none());
    }

    #[test]
    fn production_outputs_are_joined() {
        let production = Production {
            outputs: vec![
                ProductionItem { item: "wood".to_string(), per_minute: 4.0 },
                ProductionItem { item: "planks".to_string(), per_minute: 2.5 },
            ],
            inputs: vec![],
            radius: None,
            buffer_stacks: 4,
        };
        assert_eq!(production_line(&production), Some("wood 4.0/min, planks 2.5/min".to_string()));
    }

    // --- farm_line (ticket 084) ---------------------------------------------

    #[test]
    fn a_farm_line_names_the_tiles_building_by_its_loaded_name() {
        let mut tile = building("tile", 1, vec![]);
        tile.name = "Lumberjack's Farm Tile".to_string();
        let defs = BuildingDefinitions::from_entries(vec![loaded("tile", tile)]);
        let farm = crate::city::definition::Farm { tile: "tile".to_string(), radius_blocks: 16, tiles_for_full_rate: 3 };

        assert_eq!(
            farm_line(&farm, &defs),
            "  Scales with Lumberjack's Farm Tile nearby (3 needed, within 16 tiles)".to_string()
        );
    }

    #[test]
    fn a_farm_line_falls_back_to_the_raw_id_when_the_tile_definition_is_missing() {
        let defs = BuildingDefinitions::default();
        let farm = crate::city::definition::Farm { tile: "ghost_tile".to_string(), radius_blocks: 4, tiles_for_full_rate: 1 };

        assert_eq!(farm_line(&farm, &defs), "  Scales with ghost_tile nearby (1 needed, within 4 tiles)".to_string());
    }
}
