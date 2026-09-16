//! Warehouses (ticket 079, roadmap H2): which producers a warehouse serves,
//! how far each one is along the road, and how much the city can hold.
//!
//! The roadmap's own wording for this half of H2 — "a producer's output has to
//! reach a warehouse, and how long that takes comes from the road distance and
//! each road type's `travel_speed`". This module answers *which* and *how
//! far*; ticket 080's haulage is what then moves the goods.
//!
//! It is also the first consumer of roadmap F4 (ticket 056), whose
//! [`touching_road_cells`](super::road::touching_road_cells) and friends have
//! carried a `#[allow(dead_code)]` and the note "logistics in a later
//! iteration are the eventual readers" ever since they landed.
//!
//! ## Coverage: one BFS, then one Dijkstra
//!
//! Per placed warehouse:
//!
//! 1. **Coverage** is a breadth-first walk over road cells from the
//!    warehouse's own touching cells, out to `radius_cells` hops. Hops, not
//!    travel time — a working radius is a *distance*, and a faster road
//!    should make a haul quicker rather than make the warehouse reach
//!    further.
//! 2. **Travel time** is then a Dijkstra *restricted to that covered set*,
//!    charging `1.0 / travel_speed` minutes for each cell entered.
//!
//! Two passes rather than one Dijkstra pruned by hop count, because that
//! would answer neither question exactly: the hop count along a *fastest*
//! path can exceed the radius while a slower path stays inside it, so one
//! pass has to approximate one of the two. Restricting the second pass to
//! what the first found keeps "is it in range" and "how long does it take"
//! separately and exactly answerable, and both passes are cheap over a city's
//! worth of road cells.
//!
//! ## A producer off the road is unserved, and that is the point
//!
//! Measuring the radius along the road means a producer that touches no road
//! cell — or whose road island holds no warehouse — is served by nobody, and
//! (ticket 078) fills its buffer and stops. This is what makes the road
//! network the thing the economy runs on rather than decoration, and what
//! `RoadType::travel_speed` has been waiting for since ticket 060.
//!
//! A road style with no `.ron` under `assets/city/road_types` falls back to
//! `travel_speed: 1.0` rather than being impassable — an undescribed road is
//! undescribed, not broken, which is the call
//! [`super::road_definition`]'s own docs already make.
//!
//! ## Derived every time, never stored
//!
//! [`Coverage`] is rebuilt from [`City`] whenever the city, the definitions
//! or the road types change — the same "no new stored state, so it can never
//! go stale" rule [`super::road`]'s module docs set for the road graph
//! itself. Nothing here is persisted.
//!
//! ## The storage capacity, and ticket 072
//!
//! [`StorageCapacity`] is `economy.base_storage` plus every placed
//! warehouse's `storage`. This does cut against ticket 072's "one global
//! pile, unbounded", and the reconciliation is that a cap is a *warehouse*
//! property enforced against the pile rather than per-warehouse storage:
//! goods still don't live anywhere. [`Stock::add_parcel_capped`] is where it
//! bites, and every caller that credits the stock reports what overflowed
//! rather than dropping it quietly.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use bevy::math::IVec2;
use bevy::prelude::*;

use super::definition::{BuildingDefinitions, Warehouse};
use super::economy::EconomyConfig;
use super::road::{self, Direction};
use super::road_definition::RoadTypes;
use super::state::{BuildingId, City};

/// The `travel_speed` used for a road style with no definition file — see
/// the module docs.
const DEFAULT_TRAVEL_SPEED: f32 = 1.0;

/// Which warehouse serves one producer, and how long a one-way haul takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Served {
    pub warehouse: BuildingId,
    /// Game minutes, one way, from the warehouse to the producer along the
    /// fastest road route inside the radius. Ticket 080 adds the warehouse's
    /// own `handling_minutes` on top of this rather than folding it in here,
    /// so this number stays "how far away is it" and can be shown as such.
    pub travel_minutes: f32,
}

/// Who serves whom, recomputed from [`City`] — see the module docs.
#[derive(Resource, Debug, Default)]
pub struct Coverage {
    served: HashMap<BuildingId, Served>,
    warehouses: Vec<BuildingId>,
}

impl Coverage {
    /// Which warehouse serves `producer`, if any is in range along the road.
    pub fn served(&self, producer: BuildingId) -> Option<Served> {
        self.served.get(&producer).copied()
    }

    /// Every placed warehouse, ascending by id.
    pub fn warehouses(&self) -> &[BuildingId] {
        &self.warehouses
    }

    /// Every producer this warehouse serves — ticket 080 counts its in-flight
    /// hauls against `concurrent_hauls` this way.
    #[allow(dead_code)] // ticket 080's dispatch is the first non-test caller
    pub fn producers_of(&self, warehouse: BuildingId) -> impl Iterator<Item = BuildingId> + '_ {
        self.served.iter().filter(move |(_, served)| served.warehouse == warehouse).map(|(&id, _)| id)
    }
}

/// The city's total storage, `economy.base_storage` plus every placed
/// warehouse's own — see the module docs.
#[derive(Resource, Debug, Clone, Copy)]
pub struct StorageCapacity(pub u64);

impl Default for StorageCapacity {
    fn default() -> Self {
        // Not zero: a `StorageCapacity` that hasn't been computed yet must
        // not read as "the city can hold nothing", which would bounce every
        // credit in the game for one frame.
        StorageCapacity(u64::MAX)
    }
}

/// `capacity`'s value, or "no ceiling" when the resource isn't present — the
/// same tolerant `Option<Res<..>>` shape every other citybuilder system uses
/// for a resource a minimal test `App` may not have added
/// ([`super::tool::ActiveTool`]'s own module docs set the pattern). A test
/// `App` that never adds [`WarehousePlugin`] therefore credits the stock
/// exactly as it did before ticket 079.
pub fn storage_capacity(capacity: Option<&StorageCapacity>) -> u64 {
    capacity.map(|capacity| capacity.0).unwrap_or(u64::MAX)
}

pub struct WarehousePlugin;

/// The system set [`recompute`] runs in — ticket 080's dispatch orders itself
/// after this so a haul is never planned against last frame's coverage.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoverageSet;

impl Plugin for WarehousePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Coverage>()
            .init_resource::<StorageCapacity>()
            .add_systems(Update, recompute.in_set(CoverageSet));
    }
}

/// Rebuilds [`Coverage`] and [`StorageCapacity`] when anything they derive
/// from has changed.
///
/// Change detection rather than every frame: this walks the road graph once
/// per warehouse, and a city's roads and buildings change on a click, not on
/// a tick. `is_added` is in the condition so the first frame computes
/// them — a `Coverage` that stayed empty until the player's next click would
/// leave every producer reading as unserved on load.
fn recompute(
    city: Res<City>,
    definitions: Res<BuildingDefinitions>,
    road_types: Res<RoadTypes>,
    economy: Res<EconomyConfig>,
    mut coverage: ResMut<Coverage>,
    mut capacity: ResMut<StorageCapacity>,
) {
    if !(city.is_changed() || definitions.is_changed() || road_types.is_changed() || economy.is_changed()) {
        return;
    }

    *coverage = compute_coverage(&city, &definitions, &road_types);
    *capacity = StorageCapacity(compute_capacity(&city, &definitions, &economy));
}

/// `economy.base_storage` plus every placed warehouse's `storage`, saturating
/// — a hand-written definition with an absurd number should give a very
/// roomy city rather than a wrapped one.
pub fn compute_capacity(city: &City, definitions: &BuildingDefinitions, economy: &EconomyConfig) -> u64 {
    city.buildings()
        .filter_map(|(id, _)| warehouse_of(city, definitions, id))
        .fold(economy.base_storage, |total, warehouse| total.saturating_add(warehouse.storage))
}

/// `id`'s [`Warehouse`] block, if it is a placed warehouse at all. Goes
/// through [`City::definition_of`] (ticket 076) rather than the catalogue id,
/// which is the whole reason that ticket had to land first.
pub fn warehouse_of<'a>(
    city: &City,
    definitions: &'a BuildingDefinitions,
    id: BuildingId,
) -> Option<&'a Warehouse> {
    let definition = definitions.get(city.definition_of(id)?)?;
    definition.building.warehouse.as_ref()
}

/// Whether `id` is a placed building that produces anything — a
/// `production` block, or (ticket 086) a `gatherer` one: both fill a
/// [`super::production::Producer`] buffer a warehouse needs to reach the
/// same way, and a gatherer with no road in range should stall exactly like
/// an unserved farm does, not silently skip coverage because its output
/// isn't a chosen recipe.
fn is_producer(city: &City, definitions: &BuildingDefinitions, id: BuildingId) -> bool {
    city.definition_of(id).and_then(|definition| definitions.get(definition)).is_some_and(|definition| {
        definition.building.production.is_some()
            || definition.building.gatherer.is_some()
            || definition.building.mine.is_some()
    })
}

/// The whole coverage pass — see the module docs. A plain function over its
/// three inputs, so it is testable without an `App`, the same shape
/// [`super::grid::fit_footprint`] and [`super::road::select_piece`] use.
pub fn compute_coverage(city: &City, definitions: &BuildingDefinitions, road_types: &RoadTypes) -> Coverage {
    let mut warehouses: Vec<BuildingId> =
        city.buildings().map(|(id, _)| id).filter(|&id| warehouse_of(city, definitions, id).is_some()).collect();
    warehouses.sort();

    // The producers' own road cells, resolved once rather than per warehouse
    // — `touching_road_cells` walks every footprint tile, and a warehouse
    // pass that redid it would multiply that by the number of warehouses.
    let producer_cells: Vec<(BuildingId, HashSet<IVec2>)> = city
        .buildings()
        .filter(|&(id, _)| is_producer(city, definitions, id))
        .map(|(id, placed)| (id, road::touching_road_cells(city, placed)))
        .collect();

    let mut served: HashMap<BuildingId, Served> = HashMap::new();
    for &warehouse in &warehouses {
        let Some(spec) = warehouse_of(city, definitions, warehouse) else { continue };
        let Some(placed) = city.building(warehouse) else { continue };

        let start = road::touching_road_cells(city, placed);
        if start.is_empty() {
            continue;
        }

        let covered = cells_within(city, &start, spec.radius_cells);
        let times = travel_times(city, road_types, &start, &covered);

        for (producer, cells) in &producer_cells {
            if *producer == warehouse {
                continue;
            }
            let Some(minutes) = cells.iter().filter_map(|cell| times.get(cell).copied()).min_by(cmp_f32) else {
                continue;
            };

            // Ties go to the lower id, so a producer equidistant from two
            // warehouses doesn't flip between them frame to frame.
            match served.get(producer) {
                Some(existing) if (existing.travel_minutes, existing.warehouse) <= (minutes, warehouse) => {}
                _ => {
                    served.insert(*producer, Served { warehouse, travel_minutes: minutes });
                }
            }
        }
    }

    Coverage { served, warehouses }
}

/// Total ordering over the `f32` travel times. They are finite by
/// construction — every edge weight is `1.0 / travel_speed` for a
/// `travel_speed > 0.0` that `road_definition` already validated — so this
/// never has a NaN to break the ordering on.
fn cmp_f32(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.partial_cmp(b).expect("travel times are finite by construction")
}

/// Every road cell within `radius` hops of `start`, `start` included — the
/// coverage half of the pass.
fn cells_within(city: &City, start: &HashSet<IVec2>, radius: u32) -> HashSet<IVec2> {
    let mut covered: HashSet<IVec2> = start.clone();
    let mut queue: VecDeque<(IVec2, u32)> = start.iter().map(|&cell| (cell, 0)).collect();

    while let Some((cell, depth)) = queue.pop_front() {
        if depth == radius {
            continue;
        }
        for direction in Direction::ALL {
            let next = cell + direction.offset();
            if city.is_road_cell(next) && covered.insert(next) {
                queue.push_back((next, depth + 1));
            }
        }
    }
    covered
}

/// Fastest travel time from `start` to every cell of `covered`, in game
/// minutes — a Dijkstra restricted to `covered`, charging `1.0 /
/// travel_speed` for each cell *entered*.
///
/// Charging on entry rather than exit is what makes a stair or a tunnel cost
/// what its own style says: the price of a route is the sum of the cells it
/// travels through, and the warehouse's own cells are free because it is
/// standing on them.
fn travel_times(
    city: &City,
    road_types: &RoadTypes,
    start: &HashSet<IVec2>,
    covered: &HashSet<IVec2>,
) -> HashMap<IVec2, f32> {
    let mut best: HashMap<IVec2, f32> = HashMap::new();
    let mut queue: BinaryHeap<Step> = BinaryHeap::new();
    for &cell in start {
        best.insert(cell, 0.0);
        queue.push(Step { minutes: 0.0, cell });
    }

    while let Some(Step { minutes, cell }) = queue.pop() {
        if best.get(&cell).is_some_and(|&known| known < minutes) {
            continue;
        }
        for direction in Direction::ALL {
            let next = cell + direction.offset();
            if !covered.contains(&next) {
                continue;
            }
            let cost = minutes + cell_minutes(city, road_types, next);
            if best.get(&next).is_none_or(|&known| cost < known) {
                best.insert(next, cost);
                queue.push(Step { minutes: cost, cell: next });
            }
        }
    }
    best
}

/// What entering `cell` costs, in game minutes — the inverse of its style's
/// `travel_speed`, or of [`DEFAULT_TRAVEL_SPEED`] for a style with no
/// definition file.
fn cell_minutes(city: &City, road_types: &RoadTypes, cell: IVec2) -> f32 {
    let speed = city
        .road_style_at(cell)
        .and_then(|style| road_types.get(style))
        .map(|road_type| road_type.road_type.travel_speed)
        .unwrap_or(DEFAULT_TRAVEL_SPEED);
    1.0 / speed
}

/// A Dijkstra frontier entry. `Ord` is reversed on the cost so
/// [`BinaryHeap`] — a max-heap — pops the *cheapest* first; the cell breaks
/// ties so the ordering is total and the walk is deterministic.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Step {
    minutes: f32,
    cell: IVec2,
}

impl Eq for Step {}

impl Ord for Step {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        cmp_f32(&other.minutes, &self.minutes).then_with(|| (other.cell.x, other.cell.y).cmp(&(self.cell.x, self.cell.y)))
    }
}

impl PartialOrd for Step {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::{Building, Category, FootprintSpec, Integrity, LoadedBuilding, Mine, Production, ProductionItem, ShaftAt};
    use crate::city::road::RoadPieceVariant;
    use crate::city::road_definition::{LoadedRoadType, RoadType};
    use crate::city::state::ROAD_CELL_SIZE;
    use bevy::math::{IVec2, IVec3};
    use std::path::PathBuf;

    fn building(warehouse: Option<Warehouse>, production: Option<Production>) -> Building {
        Building {
            name: "test".to_string(),
            blueprint: "test.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production,
            cost: Vec::new(),
            warehouse,
            farm: None,
            gatherer: None,
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        }
    }

    fn farm_production() -> Production {
        Production {
            outputs: vec![ProductionItem { item: "minecraft:wheat".to_string(), per_minute: 12.0 }],
            inputs: Vec::new(),
            radius: None,
            buffer_stacks: 4,
            haul_at_stacks: None,
        }
    }

    fn warehouse(radius_cells: u32, storage: u64) -> Warehouse {
        Warehouse { radius_cells, concurrent_hauls: 2, handling_minutes: 0.0, storage }
    }

    fn mine_only_building() -> Building {
        let mut def = building(None, None);
        def.mine = Some(Mine {
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
        });
        def
    }

    /// Ticket 116: a mine fills the same buffer a `production`/`gatherer`
    /// block does, so it has to be recognised as a producer for the same
    /// reason 086's own note gives — without this a mine never gets a
    /// warehouse.
    #[test]
    fn is_producer_is_true_for_a_mine_only_definition() {
        let definitions = definitions(&[("mine01", mine_only_building())]);
        let mut city = City::default();
        let id = place(&mut city, "mine01", IVec2::new(0, 0));
        assert!(is_producer(&city, &definitions, id));
    }

    fn definitions(entries: &[(&str, Building)]) -> BuildingDefinitions {
        BuildingDefinitions::from_entries(
            entries
                .iter()
                .map(|(id, building)| LoadedBuilding {
                    id: id.to_string(),
                    path: PathBuf::from(format!("{id}.ron")),
                    building: building.clone(),
                    footprint: IVec2::ONE,
                    catalogue_id: id.to_string(),
                })
                .collect(),
        )
    }

    fn road_types(entries: &[(&str, f32)]) -> RoadTypes {
        RoadTypes::from_entries(
            entries
                .iter()
                .map(|(id, speed)| LoadedRoadType {
                    id: id.to_string(),
                    path: PathBuf::from(format!("{id}.ron")),
                    road_type: RoadType { name: id.to_string(), travel_speed: *speed, capacity: 4 },
                })
                .collect(),
        )
    }

    /// A building placed against the *south* edge of road cell `cell` — one
    /// block below its minimum corner, so `touching_road_cells` finds it.
    fn place(city: &mut City, definition: &str, cell: IVec2) -> BuildingId {
        let origin = IVec3::new(cell.x * ROAD_CELL_SIZE, 64, cell.y * ROAD_CELL_SIZE - 1);
        city.place_building(definition, Some(definition.to_string()), origin, Rotation::Deg0, IVec2::ONE).unwrap()
    }

    fn road(city: &mut City, style: &str, from_x: i32, to_x: i32, z: i32) {
        for x in from_x..=to_x {
            city.add_road_cell(IVec2::new(x, z), style, 64, None, RoadPieceVariant::Surface).unwrap();
        }
    }

    fn defs() -> BuildingDefinitions {
        definitions(&[
            ("warehouse01", building(Some(warehouse(4, 4096)), None)),
            ("far_warehouse", building(Some(warehouse(1, 1024)), None)),
            ("farm01", building(None, Some(farm_production()))),
        ])
    }

    #[test]
    fn a_farm_on_the_same_road_is_served() {
        let mut city = City::default();
        road(&mut city, "dirt", 0, 3, 0);
        let warehouse = place(&mut city, "warehouse01", IVec2::new(0, 0));
        let farm = place(&mut city, "farm01", IVec2::new(3, 0));

        let coverage = compute_coverage(&city, &defs(), &road_types(&[("dirt", 1.0)]));
        let served = coverage.served(farm).expect("the farm is three cells along the road");
        assert_eq!(served.warehouse, warehouse);
        assert!((served.travel_minutes - 3.0).abs() < 1e-5, "three cells at 1.0 speed is three minutes");
    }

    /// The deliberate consequence of measuring the radius along the road: a
    /// producer with no road touching it is served by nobody, however close
    /// it physically stands.
    #[test]
    fn a_farm_with_no_road_is_unserved_however_near_it_stands() {
        let mut city = City::default();
        road(&mut city, "dirt", 0, 0, 0);
        place(&mut city, "warehouse01", IVec2::new(0, 0));
        // Far from any road cell, in the middle of nowhere.
        let farm = city
            .place_building("farm01", Some("farm01".to_string()), IVec3::new(500, 64, 500), Rotation::Deg0, IVec2::ONE)
            .unwrap();

        let coverage = compute_coverage(&city, &defs(), &road_types(&[("dirt", 1.0)]));
        assert!(coverage.served(farm).is_none());
    }

    #[test]
    fn a_farm_beyond_the_radius_is_unserved() {
        let mut city = City::default();
        road(&mut city, "dirt", 0, 6, 0);
        place(&mut city, "far_warehouse", IVec2::new(0, 0)); // radius 1
        let near = place(&mut city, "farm01", IVec2::new(1, 0));
        let far = place(&mut city, "farm01", IVec2::new(6, 0));

        let coverage = compute_coverage(&city, &defs(), &road_types(&[("dirt", 1.0)]));
        assert!(coverage.served(near).is_some(), "one cell away is inside a radius of 1");
        assert!(coverage.served(far).is_none(), "six cells away is not");
    }

    /// Two disconnected road islands: a warehouse on one cannot serve a farm
    /// on the other, even within the hop count.
    #[test]
    fn a_farm_on_a_disconnected_road_island_is_unserved() {
        let mut city = City::default();
        road(&mut city, "dirt", 0, 1, 0);
        road(&mut city, "dirt", 8, 9, 0);
        place(&mut city, "warehouse01", IVec2::new(0, 0));
        let farm = place(&mut city, "farm01", IVec2::new(9, 0));

        let coverage = compute_coverage(&city, &defs(), &road_types(&[("dirt", 1.0)]));
        assert!(coverage.served(farm).is_none());
    }

    /// The roadmap's own reason for `travel_speed`: a faster road makes the
    /// haul quicker. It does *not* make the warehouse reach further — that is
    /// what the hop-count radius is for.
    #[test]
    fn a_faster_road_shortens_the_haul_without_widening_the_radius() {
        let mut city = City::default();
        road(&mut city, "paved", 0, 3, 0);
        place(&mut city, "warehouse01", IVec2::new(0, 0));
        let farm = place(&mut city, "farm01", IVec2::new(3, 0));

        let fast = compute_coverage(&city, &defs(), &road_types(&[("paved", 3.0)]));
        assert!((fast.served(farm).unwrap().travel_minutes - 1.0).abs() < 1e-5, "three cells at 3x is one minute");

        let mut far = City::default();
        road(&mut far, "paved", 0, 6, 0);
        place(&mut far, "far_warehouse", IVec2::new(0, 0)); // radius 1
        let distant = place(&mut far, "farm01", IVec2::new(6, 0));
        let coverage = compute_coverage(&far, &defs(), &road_types(&[("paved", 9.0)]));
        assert!(coverage.served(distant).is_none(), "speed must not buy range");
    }

    /// A style with no `.ron` is undescribed, not impassable.
    #[test]
    fn an_undefined_road_style_travels_at_the_default_speed() {
        let mut city = City::default();
        road(&mut city, "gravel", 0, 2, 0);
        place(&mut city, "warehouse01", IVec2::new(0, 0));
        let farm = place(&mut city, "farm01", IVec2::new(2, 0));

        let coverage = compute_coverage(&city, &defs(), &RoadTypes::default());
        assert!((coverage.served(farm).unwrap().travel_minutes - 2.0).abs() < 1e-5);
    }

    /// Dijkstra, not breadth-first: the route with more cells wins when its
    /// cells are fast enough. The detour is 4 paved cells at 4x (1.0 min)
    /// against 2 dirt cells at 0.5x (4.0 min).
    #[test]
    fn the_fastest_route_wins_over_the_shortest_one() {
        let mut city = City::default();
        // Slow direct link east from the warehouse.
        city.add_road_cell(IVec2::new(1, 0), "mud", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(2, 0), "mud", 64, None, RoadPieceVariant::Surface).unwrap();
        // Fast detour south and back — south rather than north because the
        // `place` helper puts a building against a cell's *north* edge, and a
        // road cell there would be sitting on the tiles it wants.
        for cell in [IVec2::new(0, 1), IVec2::new(1, 1), IVec2::new(2, 1), IVec2::new(3, 1)] {
            city.add_road_cell(cell, "paved", 64, None, RoadPieceVariant::Surface).unwrap();
        }
        city.add_road_cell(IVec2::new(0, 0), "paved", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(3, 0), "paved", 64, None, RoadPieceVariant::Surface).unwrap();

        place(&mut city, "warehouse01", IVec2::new(0, 0));
        let farm = place(&mut city, "farm01", IVec2::new(3, 0));

        let types = road_types(&[("mud", 0.25), ("paved", 4.0)]);
        let coverage = compute_coverage(&city, &defs(), &types);
        let minutes = coverage.served(farm).unwrap().travel_minutes;
        assert!(minutes < 2.0, "the fast four-cell detour ({minutes}) must beat the slow three-cell run");
    }

    #[test]
    fn the_nearest_warehouse_wins() {
        let mut city = City::default();
        road(&mut city, "dirt", 0, 8, 0);
        let near = place(&mut city, "warehouse01", IVec2::new(4, 0));
        place(&mut city, "warehouse01", IVec2::new(8, 0));
        let farm = place(&mut city, "farm01", IVec2::new(3, 0));

        let coverage = compute_coverage(&city, &defs(), &road_types(&[("dirt", 1.0)]));
        assert_eq!(coverage.served(farm).unwrap().warehouse, near);
    }

    #[test]
    fn a_warehouse_lists_the_producers_it_serves() {
        let mut city = City::default();
        road(&mut city, "dirt", 0, 3, 0);
        let warehouse = place(&mut city, "warehouse01", IVec2::new(0, 0));
        let a = place(&mut city, "farm01", IVec2::new(2, 0));
        let b = place(&mut city, "farm01", IVec2::new(3, 0));

        let coverage = compute_coverage(&city, &defs(), &road_types(&[("dirt", 1.0)]));
        let mut served: Vec<BuildingId> = coverage.producers_of(warehouse).collect();
        served.sort();
        assert_eq!(served, vec![a, b]);
        assert_eq!(coverage.warehouses(), &[warehouse]);
    }

    // --- storage capacity ----------------------------------------------------

    #[test]
    fn capacity_is_the_base_plus_every_warehouse() {
        let mut city = City::default();
        place(&mut city, "warehouse01", IVec2::new(0, 0)); // 4096
        place(&mut city, "far_warehouse", IVec2::new(5, 0)); // 1024
        place(&mut city, "farm01", IVec2::new(9, 0)); // adds nothing

        let economy = EconomyConfig::default();
        assert_eq!(compute_capacity(&city, &defs(), &economy), economy.base_storage + 4096 + 1024);
    }

    /// A city with no warehouse still has to be able to hold its founding
    /// grant, which is what `base_storage` is for.
    #[test]
    fn a_city_with_no_warehouse_still_has_the_base_capacity() {
        let economy = EconomyConfig::default();
        assert_eq!(compute_capacity(&City::default(), &defs(), &economy), economy.base_storage);
        assert!(economy.base_storage > 0);
    }
}
