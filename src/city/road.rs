//! Road adjacency and connectivity (ticket 053, roadmap F1) — the graph F3's
//! auto-tiling and F4's connectivity queries both read.
//!
//! No new stored state: a road graph is *derived* from
//! [`super::state::City::occupant_at`] on every call, the same "the city
//! state is authoritative" rule [`super::state`]'s own module docs lay out.
//! `City` already tracks which tiles are road (`add_road`/`remove_road`/
//! `roads()`, landed with D1); this module only adds queries over that set,
//! nothing that could itself go stale.
//!
//! ## Coordinates
//!
//! This module works in the same Minecraft `(x, z)` tile space
//! [`super::state::footprint_tiles`] does — an `IVec2`'s `x`/`y` fields are
//! Minecraft's `x`/`z`. There is no `bevy.z = -mc.z` flip to apply here;
//! that convention only matters where a tile becomes a Bevy transform, which
//! nothing in this module does. [`Direction`]'s offsets are Minecraft's own
//! cardinal convention: north is `-z`, south is `+z`, east is `+x`, west is
//! `-x`.
//!
//! ## What F3 and F4 build on this
//!
//! [`connections_at`] is the per-tile answer F3's auto-tiling switches on —
//! which of a road tile's four neighbours are also road decides whether it's
//! a straight, a corner, a T, a cross, or a dead end. [`reachable_from`] is
//! the BFS primitive F4's "what does this road segment reach" is built on;
//! [`is_connected`] is the two-tile special case ("is this building on the
//! network" tests membership of a footprint's adjacent tiles in a
//! [`reachable_from`] set, which is F4's own job, not this ticket's).

use std::collections::{HashSet, VecDeque};

use bevy::math::IVec2;

use super::state::{City, Occupant};

/// One of the four cardinal directions a road tile can connect in, in
/// Minecraft's own `x`/`z` convention — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    /// Every direction, in a fixed order — iterated by [`connections_at`]
    /// and [`reachable_from`] so both visit neighbours the same way.
    pub const ALL: [Direction; 4] = [Direction::North, Direction::South, Direction::East, Direction::West];

    /// The `(x, z)` step one tile in this direction — north is `-z`, south
    /// is `+z`, east is `+x`, west is `-x`.
    pub fn offset(self) -> IVec2 {
        match self {
            Direction::North => IVec2::new(0, -1),
            Direction::South => IVec2::new(0, 1),
            Direction::East => IVec2::new(1, 0),
            Direction::West => IVec2::new(-1, 0),
        }
    }
}

/// Which of a road tile's four neighbours are also road tiles — the input
/// F3's auto-tiling picks a piece from. Built by [`connections_at`], which
/// is the only way to construct one: this type is a pure answer, not
/// something a caller assembles field-by-field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RoadConnections {
    pub north: bool,
    pub south: bool,
    pub east: bool,
    pub west: bool,
}

impl RoadConnections {
    /// How many of the four neighbours are road — 0 (isolated), 1 (dead
    /// end), 2 (straight or corner, depending on *which* two — see the
    /// individual fields), 3 (a T), or 4 (a cross). F3's own job to map this
    /// (and the specific fields) onto a piece; this is only the count.
    #[allow(dead_code)] // no caller yet — F3's auto-tiling is
    pub fn count(self) -> u32 {
        self.north as u32 + self.south as u32 + self.east as u32 + self.west as u32
    }
}

/// Whether `city` has a road tile at `tile` — [`Occupant::Road`] specifically,
/// not just "occupied" (a building on `tile` does not count).
fn is_road(city: &City, tile: IVec2) -> bool {
    city.occupant_at(tile) == Some(Occupant::Road)
}

/// Which of `tile`'s four cardinal neighbours are also road tiles in `city`.
/// `tile` itself need not be a road — a caller checking "would a road placed
/// here connect to anything" (F2's live preview) can ask before committing,
/// the same way E3's ghost preview asks [`super::grid::fit_footprint`] before
/// [`super::state::City::place_building`].
#[allow(dead_code)] // no caller yet — F3's auto-tiling is
pub fn connections_at(city: &City, tile: IVec2) -> RoadConnections {
    RoadConnections {
        north: is_road(city, tile + Direction::North.offset()),
        south: is_road(city, tile + Direction::South.offset()),
        east: is_road(city, tile + Direction::East.offset()),
        west: is_road(city, tile + Direction::West.offset()),
    }
}

/// Every tile reachable from `start` through an unbroken chain of
/// orthogonally-adjacent road tiles, `start` included. Empty if `start`
/// itself isn't a road tile — a non-road tile reaches nothing, not even
/// itself, so callers can't mistake "reaches only itself" (a one-tile road
/// island) for "isn't a road at all."
#[allow(dead_code)] // no caller yet — F4's connectivity queries are
pub fn reachable_from(city: &City, start: IVec2) -> HashSet<IVec2> {
    let mut visited = HashSet::new();
    if !is_road(city, start) {
        return visited;
    }

    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(tile) = queue.pop_front() {
        for direction in Direction::ALL {
            let neighbour = tile + direction.offset();
            if is_road(city, neighbour) && visited.insert(neighbour) {
                queue.push_back(neighbour);
            }
        }
    }

    visited
}

/// Whether `a` and `b` are both road tiles connected through an unbroken
/// chain of road tiles — the two-tile special case of [`reachable_from`].
/// `false` if either isn't a road tile at all, not just if they're
/// unreachable from each other.
#[allow(dead_code)] // no caller yet — F4's connectivity queries are
pub fn is_connected(city: &City, a: IVec2, b: IVec2) -> bool {
    is_road(city, a) && is_road(city, b) && reachable_from(city, a).contains(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_offsets_are_mutually_opposite_and_cover_the_four_cardinals() {
        assert_eq!(Direction::North.offset(), -Direction::South.offset());
        assert_eq!(Direction::East.offset(), -Direction::West.offset());

        let offsets: HashSet<IVec2> = Direction::ALL.iter().map(|d| d.offset()).collect();
        assert_eq!(
            offsets,
            HashSet::from([IVec2::new(0, -1), IVec2::new(0, 1), IVec2::new(1, 0), IVec2::new(-1, 0)])
        );
    }

    #[test]
    fn connections_at_reports_no_neighbours_for_an_isolated_road_tile() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, 0)).unwrap();

        let connections = connections_at(&city, IVec2::new(0, 0));
        assert_eq!(connections, RoadConnections::default());
        assert_eq!(connections.count(), 0);
    }

    #[test]
    fn connections_at_reports_exactly_the_road_neighbours() {
        let mut city = City::default();
        // A road tile with north and east neighbours, but not south or west.
        city.add_road(IVec2::new(0, 0)).unwrap();
        city.add_road(IVec2::new(0, -1)).unwrap(); // north
        city.add_road(IVec2::new(1, 0)).unwrap(); // east

        let connections = connections_at(&city, IVec2::new(0, 0));
        assert_eq!(connections, RoadConnections { north: true, south: false, east: true, west: false });
        assert_eq!(connections.count(), 2);
    }

    #[test]
    fn connections_at_ignores_a_building_neighbour() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, 0)).unwrap();
        city.place_building("house01", bevy::math::IVec3::new(0, 64, -1), crate::blueprint::Rotation::Deg0, IVec2::ONE)
            .unwrap();

        // A building sits north of the road tile — that is not a road
        // connection, even though the tile is occupied.
        let connections = connections_at(&city, IVec2::new(0, 0));
        assert!(!connections.north);
    }

    #[test]
    fn connections_at_works_from_a_tile_that_is_not_itself_a_road() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, -1)).unwrap();

        // Asking "what would connect here" before placing anything.
        let connections = connections_at(&city, IVec2::new(0, 0));
        assert_eq!(connections, RoadConnections { north: true, south: false, east: false, west: false });
    }

    #[test]
    fn connections_at_reports_a_full_cross() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, 0)).unwrap();
        for offset in [IVec2::new(0, -1), IVec2::new(0, 1), IVec2::new(1, 0), IVec2::new(-1, 0)] {
            city.add_road(offset).unwrap();
        }

        assert_eq!(connections_at(&city, IVec2::new(0, 0)).count(), 4);
    }

    #[test]
    fn reachable_from_a_non_road_tile_is_empty() {
        let city = City::default();
        assert!(reachable_from(&city, IVec2::new(0, 0)).is_empty());
    }

    #[test]
    fn reachable_from_a_single_tile_island_is_only_itself() {
        let mut city = City::default();
        city.add_road(IVec2::new(5, 5)).unwrap();
        assert_eq!(reachable_from(&city, IVec2::new(5, 5)), HashSet::from([IVec2::new(5, 5)]));
    }

    #[test]
    fn reachable_from_walks_a_straight_run() {
        let mut city = City::default();
        for x in 0..5 {
            city.add_road(IVec2::new(x, 0)).unwrap();
        }

        let reached = reachable_from(&city, IVec2::new(0, 0));
        let expected: HashSet<IVec2> = (0..5).map(|x| IVec2::new(x, 0)).collect();
        assert_eq!(reached, expected);
    }

    #[test]
    fn reachable_from_walks_a_branch() {
        let mut city = City::default();
        // A horizontal run with one tile branching south from the middle.
        for x in 0..3 {
            city.add_road(IVec2::new(x, 0)).unwrap();
        }
        city.add_road(IVec2::new(1, 1)).unwrap();

        let reached = reachable_from(&city, IVec2::new(0, 0));
        assert_eq!(
            reached,
            HashSet::from([IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(2, 0), IVec2::new(1, 1)])
        );
    }

    #[test]
    fn reachable_from_a_loop_terminates_and_does_not_double_count() {
        let mut city = City::default();
        // A 2x2 loop of road tiles.
        for tile in [IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(0, 1), IVec2::new(1, 1)] {
            city.add_road(tile).unwrap();
        }

        let reached = reachable_from(&city, IVec2::new(0, 0));
        assert_eq!(reached.len(), 4);
    }

    #[test]
    fn reachable_from_does_not_cross_to_a_disconnected_island() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, 0)).unwrap();
        city.add_road(IVec2::new(1, 0)).unwrap();
        // A second island, far away and not adjacent to the first.
        city.add_road(IVec2::new(100, 100)).unwrap();

        let reached = reachable_from(&city, IVec2::new(0, 0));
        assert_eq!(reached, HashSet::from([IVec2::new(0, 0), IVec2::new(1, 0)]));
    }

    #[test]
    fn is_connected_is_true_within_one_island() {
        let mut city = City::default();
        for x in 0..3 {
            city.add_road(IVec2::new(x, 0)).unwrap();
        }
        assert!(is_connected(&city, IVec2::new(0, 0), IVec2::new(2, 0)));
    }

    #[test]
    fn is_connected_is_false_across_two_islands() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, 0)).unwrap();
        city.add_road(IVec2::new(100, 100)).unwrap();
        assert!(!is_connected(&city, IVec2::new(0, 0), IVec2::new(100, 100)));
    }

    #[test]
    fn is_connected_is_false_when_an_endpoint_is_not_a_road_tile() {
        let mut city = City::default();
        city.add_road(IVec2::new(0, 0)).unwrap();
        assert!(!is_connected(&city, IVec2::new(0, 0), IVec2::new(1, 0)));
        assert!(!is_connected(&city, IVec2::new(1, 0), IVec2::new(0, 0)));
    }
}
