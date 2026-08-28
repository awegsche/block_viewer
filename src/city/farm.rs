//! Farm-tile coverage (ticket 084, roadmap C1/H2): how many of a `farm`
//! link's named tile buildings are in range of, and nearest to, each hub —
//! which `production::tick` reads to scale that hub's `production` rates.
//!
//! This is the open half of `lumber.ron`'s original design note, closed:
//! "whether the footprint stays `FromBlueprint` or widens to reserve ground
//! for Anno-style farm tiles placed around it (working radius, tile count ->
//! production %) is still open." The answer landed on here is separate
//! tile *buildings*, not a widened footprint — [`super::definition::Farm`]
//! names another [`super::definition::Building`] by its definition id, placed
//! through the exact same catalogue/footprint/occupancy machinery every other
//! building already uses, so nothing about placement, the ghost preview,
//! commit, journalling, undo or demolish had to change to add it.
//!
//! ## Not a warehouse's radius
//!
//! [`super::warehouse::Coverage`] measures its radius *along the road*, in
//! hops, because a warehouse's whole point is that the road network is what
//! makes it reachable. A field has no such requirement — Anno's own
//! farmhouses don't need a road to their crops — so [`Farm::radius_blocks`]
//! is plain straight-line distance between the two buildings' own footprint
//! rectangles: the [`rect_distance`] gap, in blocks, 0 when they touch or
//! overlap.
//!
//! ## Nearest hub wins
//!
//! Two hubs of the same kind can have overlapping catchments — one field
//! sitting in range of both. Rather than let it count toward both (double
//! spending the same tile), [`compute_farm_coverage`] walks every placed
//! tile once and assigns it to whichever qualifying hub (same
//! [`Farm::tile`], within [`Farm::radius_blocks`]) is *nearest*; a tie goes
//! to the lower [`BuildingId`] — arbitrary but deterministic, the same kind
//! of tie-break [`super::warehouse::compute_coverage`] already uses for a
//! producer equidistant from two warehouses.
//!
//! ## Linear scaling, not a threshold
//!
//! `production::scale_production` (the reader) turns a tile count into a
//! ratio — `count / tiles_for_full_rate`, clamped to `1.0` — and multiplies
//! every output *and input* rate by it (see that function's own docs for
//! why inputs scale too). Zero tiles is zero rate, not a stalled/starved
//! state: a hub with no fields yet simply has nothing to scale, the same way
//! a producer with no `production` block isn't a producer at all.
//!
//! ## Derived every time, never stored
//!
//! Same rule [`super::warehouse::Coverage`]'s own docs give: [`FarmCoverage`]
//! is rebuilt from [`City`] whenever it or the definitions change, so it can
//! never itself go stale, and nothing here is persisted.

use std::collections::HashMap;

use bevy::math::IVec2;
use bevy::prelude::*;

use super::definition::{BuildingDefinitions, Farm};
use super::state::{footprint_extent, BuildingId, City, PlacedBuilding};

/// How many `farm.tile` instances are within range of each placed hub — see
/// the module docs.
#[derive(Resource, Debug, Default)]
pub struct FarmCoverage {
    tiles: HashMap<BuildingId, u32>,
}

impl FarmCoverage {
    /// `0` for a hub with no tiles in range yet, and for anything that isn't
    /// a placed farm hub at all — the same "no data, treat it as none"
    /// collapse [`City::definition_of`]'s own docs describe for a similar
    /// pair of `None`s.
    pub fn tiles_near(&self, hub: BuildingId) -> u32 {
        self.tiles.get(&hub).copied().unwrap_or(0)
    }

    /// Sets `hub`'s tile count directly — for `production`'s own tests of
    /// `scale_production`, which need a [`FarmCoverage`] with a chosen answer
    /// rather than one [`compute_farm_coverage`] derives from a real [`City`].
    /// Same reasoning as [`super::production::ProductionState::push_shipment`].
    #[cfg(test)]
    pub fn set_tiles_near(&mut self, hub: BuildingId, tiles: u32) {
        self.tiles.insert(hub, tiles);
    }
}

pub struct FarmPlugin;

/// The system set [`recompute`] runs in — `production::tick` orders itself
/// after this so a tile placed this frame counts toward this frame's output,
/// the same reason `production::tick` already orders itself after
/// [`super::warehouse::CoverageSet`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct FarmCoverageSet;

impl Plugin for FarmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FarmCoverage>().add_systems(Update, recompute.in_set(FarmCoverageSet));
    }
}

/// Rebuilds [`FarmCoverage`] when [`City`] or [`BuildingDefinitions`] has
/// changed — change detection rather than every frame, the same call
/// [`super::warehouse::recompute`] makes and for the same reason: a city's
/// buildings change on a click, not on a tick. `is_added` is implicit in
/// `is_changed` firing on insertion, so the first frame still computes it.
fn recompute(city: Res<City>, definitions: Res<BuildingDefinitions>, mut coverage: ResMut<FarmCoverage>) {
    if !(city.is_changed() || definitions.is_changed()) {
        return;
    }
    *coverage = compute_farm_coverage(&city, &definitions);
}

/// The whole pass — a plain function over its two inputs, so it is testable
/// without an `App`, the same shape [`super::warehouse::compute_coverage`]
/// uses. See the module docs' "Nearest hub wins" for what this does with two
/// overlapping catchments.
pub fn compute_farm_coverage(city: &City, definitions: &BuildingDefinitions) -> FarmCoverage {
    // Every placed hub, with its own bounds and `farm` spec resolved once —
    // sorted ascending by id, the order the tie-break below relies on.
    let mut hubs: Vec<(BuildingId, IVec2, IVec2, &Farm)> = city
        .buildings()
        .filter_map(|(id, placed)| {
            let farm =
                placed.definition_id.as_deref().and_then(|def_id| definitions.get(def_id)).and_then(|def| def.building.farm.as_ref())?;
            let (min, max) = rect_bounds(placed);
            Some((id, min, max, farm))
        })
        .collect();
    hubs.sort_by_key(|&(id, ..)| id);

    let mut tiles: HashMap<BuildingId, u32> = HashMap::new();
    for (tile_id, tile_placed) in city.buildings() {
        let Some(tile_definition) = tile_placed.definition_id.as_deref() else { continue };
        let (tile_min, tile_max) = rect_bounds(tile_placed);

        // The nearest hub this tile is both named by (`farm.tile` matches)
        // and within range of. `hubs` is ascending by id and `best` is only
        // ever replaced by a *strictly* nearer candidate, so the first hub
        // reached at the minimum distance keeps it — ties go to the lower id.
        let mut best: Option<(BuildingId, i32)> = None;
        for &(hub_id, hub_min, hub_max, farm) in &hubs {
            if hub_id == tile_id || farm.tile != tile_definition {
                continue;
            }
            let distance = rect_distance(hub_min, hub_max, tile_min, tile_max);
            if distance > farm.radius_blocks as i32 {
                continue;
            }
            if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                best = Some((hub_id, distance));
            }
        }

        if let Some((hub_id, _)) = best {
            *tiles.entry(hub_id).or_insert(0) += 1;
        }
    }

    FarmCoverage { tiles }
}

/// `placed`'s footprint as `(min, max)` Minecraft `(x, z)` corners — `max`
/// exclusive, the same convention [`super::state::footprint_tiles`] walks.
/// Rotation-aware through [`footprint_extent`], the horizontal counterpart of
/// [`super::warehouse`]'s own road-cell math.
fn rect_bounds(placed: &PlacedBuilding) -> (IVec2, IVec2) {
    let extent = footprint_extent(placed.footprint, placed.rotation);
    let min = IVec2::new(placed.origin.x, placed.origin.z);
    (min, min + extent)
}

/// The Chebyshev (chessboard) gap between two axis-aligned rectangles, each
/// given as `(min, max)` with `max` exclusive: `0` when they touch or
/// overlap, growing outward from there. Chebyshev rather than the road
/// graph's own Manhattan hop-count: a radius here is an *area* around a hub
/// (diagonal neighbours count the same as cardinal ones), not a path through
/// discrete cells.
pub fn rect_distance(a_min: IVec2, a_max: IVec2, b_min: IVec2, b_max: IVec2) -> i32 {
    let dx = (a_min.x - b_max.x).max(b_min.x - a_max.x).max(0);
    let dz = (a_min.y - b_max.y).max(b_min.y - a_max.y).max(0);
    dx.max(dz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::{Building, Category, Farm, FootprintSpec, Integrity, LoadedBuilding, Production, ProductionItem};
    use bevy::math::IVec3;
    use std::path::PathBuf;

    fn building(farm: Option<Farm>, production: Option<Production>) -> Building {
        Building {
            name: "test".to_string(),
            blueprint: "test.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production,
            cost: Vec::new(),
            warehouse: None,
            farm,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        }
    }

    fn wood_production() -> Production {
        Production {
            outputs: vec![ProductionItem { item: "minecraft:oak_log".to_string(), per_minute: 8.0 }],
            inputs: Vec::new(),
            radius: None,
            buffer_stacks: 4,
        }
    }

    fn definitions(entries: &[(&str, Building)]) -> BuildingDefinitions {
        BuildingDefinitions::from_entries(
            entries
                .iter()
                .map(|(id, building)| LoadedBuilding {
                    id: id.to_string(),
                    path: PathBuf::from(format!("{id}.ron")),
                    building: building.clone(),
                    footprint: IVec2::new(3, 3),
                    catalogue_id: id.to_string(),
                })
                .collect(),
        )
    }

    fn defs() -> BuildingDefinitions {
        definitions(&[
            ("hut", building(Some(Farm { tile: "tile".to_string(), radius_blocks: 4, tiles_for_full_rate: 3 }), Some(wood_production()))),
            ("tile", building(None, None)),
        ])
    }

    fn place(city: &mut City, definition: &str, origin: IVec3) -> BuildingId {
        city.place_building(definition, Some(definition.to_string()), origin, Rotation::Deg0, IVec2::new(3, 3)).unwrap()
    }

    #[test]
    fn a_hub_with_no_tiles_counts_zero() {
        let mut city = City::default();
        let hub = place(&mut city, "hut", IVec3::new(0, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub), 0);
    }

    #[test]
    fn a_tile_touching_the_hub_counts() {
        let mut city = City::default();
        let hub = place(&mut city, "hut", IVec3::new(0, 64, 0));
        // Footprint 3x3 at (0,0) occupies x/z 0..3; a tile placed right
        // against its east edge touches, i.e. distance 0.
        place(&mut city, "tile", IVec3::new(3, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub), 1);
    }

    #[test]
    fn a_tile_beyond_the_radius_does_not_count() {
        let mut city = City::default();
        let hub = place(&mut city, "hut", IVec3::new(0, 64, 0));
        // Gap from the hub's east edge (x=3) to the tile's west edge (x=8) is
        // 5 tiles — outside `hut`'s radius of 4.
        place(&mut city, "tile", IVec3::new(8, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub), 0);
    }

    #[test]
    fn a_tile_exactly_at_the_radius_counts() {
        let mut city = City::default();
        let hub = place(&mut city, "hut", IVec3::new(0, 64, 0));
        // Gap is exactly 4: hub occupies 0..3, tile placed at x=7 occupies
        // 7..10, gap = 7 - 3 = 4.
        place(&mut city, "tile", IVec3::new(7, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub), 1, "a gap exactly equal to the radius still counts");
    }

    #[test]
    fn a_building_of_the_wrong_definition_never_counts_as_a_tile() {
        let mut city = City::default();
        let hub = place(&mut city, "hut", IVec3::new(0, 64, 0));
        // Another hut, not a tile — must not count toward its own or the
        // other hub's total.
        place(&mut city, "hut", IVec3::new(3, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub), 0);
    }

    #[test]
    fn multiple_tiles_in_range_all_count() {
        let mut city = City::default();
        let hub = place(&mut city, "hut", IVec3::new(0, 64, 0));
        place(&mut city, "tile", IVec3::new(3, 64, 0));
        place(&mut city, "tile", IVec3::new(-3, 64, 0));
        place(&mut city, "tile", IVec3::new(0, 64, 3));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub), 3);
    }

    #[test]
    fn a_building_with_no_farm_link_is_not_a_hub() {
        let mut city = City::default();
        let tile_as_hub = place(&mut city, "tile", IVec3::new(0, 64, 0));
        place(&mut city, "tile", IVec3::new(3, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(tile_as_hub), 0, "`tile` has no `farm` block, so it is never a hub");
    }

    // --- nearest hub wins ----------------------------------------------------

    #[test]
    fn an_overlapping_catchment_assigns_the_tile_to_the_strictly_nearer_hub() {
        let mut city = City::default();
        // Two hubs four tiles apart on the x axis, footprints 3x3 each:
        // hub_a at x=0..3, hub_b at x=7..10. A tile at x=4..7 sits 1 away
        // from hub_a and 0 away (touching) hub_b.
        let hub_a = place(&mut city, "hut", IVec3::new(0, 64, 0));
        let hub_b = place(&mut city, "hut", IVec3::new(7, 64, 0));
        place(&mut city, "tile", IVec3::new(4, 64, 0));

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(hub_b), 1, "hub_b (distance 0) is strictly nearer than hub_a (distance 1)");
        assert_eq!(coverage.tiles_near(hub_a), 0, "the tile is not double-counted");
    }

    #[test]
    fn an_exact_tie_goes_to_the_lower_building_id() {
        let mut city = City::default();
        // Both hubs touch the tile (distance 0 to each) — a genuine tie.
        let lower = place(&mut city, "hut", IVec3::new(0, 64, 0));
        let higher = place(&mut city, "hut", IVec3::new(6, 64, 0));
        place(&mut city, "tile", IVec3::new(3, 64, 0));
        assert!(lower < higher, "placed first, so it minted the lower id");

        let coverage = compute_farm_coverage(&city, &defs());
        assert_eq!(coverage.tiles_near(lower), 1);
        assert_eq!(coverage.tiles_near(higher), 0);
    }

    #[test]
    fn rect_distance_is_zero_for_overlapping_or_touching_rectangles() {
        assert_eq!(rect_distance(IVec2::new(0, 0), IVec2::new(3, 3), IVec2::new(1, 1), IVec2::new(4, 4)), 0, "overlapping");
        assert_eq!(rect_distance(IVec2::new(0, 0), IVec2::new(3, 3), IVec2::new(3, 0), IVec2::new(6, 3)), 0, "touching edges");
    }

    #[test]
    fn rect_distance_is_the_chebyshev_gap() {
        // Rectangles separated by 2 on x and 5 on z — Chebyshev takes the max.
        let a = (IVec2::new(0, 0), IVec2::new(2, 2));
        let b = (IVec2::new(4, 7), IVec2::new(6, 9));
        assert_eq!(rect_distance(a.0, a.1, b.0, b.1), 5);
    }
}
