//! Road adjacency, connectivity, and piece selection (ticket 053, roadmap
//! F1; revised to cell space and given F3's selection logic by ticket 054).
//!
//! No new stored state: a road graph is *derived* from
//! [`super::state::City::is_road_cell`] on every call, the same "the city
//! state is authoritative" rule [`super::state`]'s own module docs lay out.
//! `City` already tracks which cells are road (`add_road_cell`/
//! `remove_road_cell`/`road_cells()`, ticket 054); this module only adds
//! queries over that set, nothing that could itself go stale.
//!
//! ## Coordinates: cells, not blocks
//!
//! Ticket 053 originally worked in single-block tile space. Ticket 054
//! redefined a road as a [`super::state::ROAD_CELL_SIZE`]-block-wide cell —
//! a real cross-section (shoulder/kerb/road/road/kerb/shoulder), not a
//! 1x1-block dot — so this module's `IVec2`s are **cell coordinates**:
//! `cell * ROAD_CELL_SIZE` is the cell's minimum block corner
//! ([`super::state::road_cell_tiles`]). There is still no `bevy.z = -mc.z`
//! flip to apply here; that convention only matters where a tile becomes a
//! Bevy transform, which nothing in this module does. [`Direction`]'s
//! offsets are Minecraft's own cardinal convention: north is `-z`, south is
//! `+z`, east is `+x`, west is `-x` — one *cell* in that direction, not one
//! block.
//!
//! ## What F2, F3 and F4 build on this
//!
//! [`connections_at`] is the per-cell answer F3's auto-tiling switches on —
//! which of a road cell's four neighbours are also road decides its shape.
//! [`select_piece`] carries that the rest of the way: a
//! [`RoadPieceKind`] plus the [`Rotation`] that reproduces the actual
//! connections when applied to that kind's canonically-authored blueprint —
//! the exact pair [`crate::blueprint::rotate_blueprint`] needs once F3 wires
//! this to a mesh, and what [`super::road_catalogue`] indexes its pieces by.
//! [`reachable_from`] is the BFS primitive F4's "what does this road segment
//! reach" is built on; [`is_connected`] is the two-cell special case ("is
//! this building on the network" tests membership of a footprint's adjacent
//! cells in a [`reachable_from`] set, which is F4's own job, not this
//! ticket's).

use std::collections::{HashSet, VecDeque};

use bevy::math::IVec2;

use crate::blueprint::Rotation;

use super::state::City;

/// One of the four cardinal directions a road cell can connect in, in
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

    /// The `(x, z)` step one *cell* in this direction — north is `-z`, south
    /// is `+z`, east is `+x`, west is `-x`. See the module docs' "cells, not
    /// blocks."
    pub fn offset(self) -> IVec2 {
        match self {
            Direction::North => IVec2::new(0, -1),
            Direction::South => IVec2::new(0, 1),
            Direction::East => IVec2::new(1, 0),
            Direction::West => IVec2::new(-1, 0),
        }
    }

    /// This direction, turned `turns` quarter-turns clockwise (viewed from
    /// above: north -> east -> south -> west), the same rotation
    /// [`Rotation::quarter_turns`](crate::blueprint::rotate)-style geometry
    /// and `blueprint::rotate`'s own `Cardinal` table use — [`select_piece`]
    /// rotates a canonical connection pattern through this to find the
    /// [`Rotation`] that reproduces an actual one.
    fn rotated(self, turns: u8) -> Direction {
        const ORDER: [Direction; 4] = [Direction::North, Direction::East, Direction::South, Direction::West];
        let index = ORDER.iter().position(|&d| d == self).expect("ORDER covers every Direction");
        ORDER[(index + turns as usize) % 4]
    }
}

/// Which of a road cell's four neighbours are also road cells — the input
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
    /// end), 2 (straight or corner, depending on *which* two), 3 (a T), or 4
    /// (a cross). [`select_piece`] is what turns this, and *which* two
    /// neighbours, into a [`RoadPieceKind`].
    pub fn count(self) -> u32 {
        self.north as u32 + self.south as u32 + self.east as u32 + self.west as u32
    }

    /// This pattern, rotated `turns` quarter-turns clockwise — every `true`
    /// direction walked through [`Direction::rotated`]. [`select_piece`]'s
    /// own primitive for matching a canonical pattern against an actual one.
    fn rotated(self, turns: u8) -> RoadConnections {
        let mut out = RoadConnections::default();
        for (present, dir) in [
            (self.north, Direction::North),
            (self.south, Direction::South),
            (self.east, Direction::East),
            (self.west, Direction::West),
        ] {
            if !present {
                continue;
            }
            match dir.rotated(turns) {
                Direction::North => out.north = true,
                Direction::South => out.south = true,
                Direction::East => out.east = true,
                Direction::West => out.west = true,
            }
        }
        out
    }
}

/// Whether `city` has a road cell at `cell` — [`City::is_road_cell`], not
/// [`super::state::City::occupant_at`] against one of its 36 tiles.
fn is_road(city: &City, cell: IVec2) -> bool {
    city.is_road_cell(cell)
}

/// Which of `cell`'s four cardinal neighbour *cells* are also road cells in
/// `city`. `cell` itself need not be a road — a caller checking "would a
/// road placed here connect to anything" (F2's live preview) can ask before
/// committing, the same way E3's ghost preview asks
/// [`super::grid::fit_footprint`] before
/// [`super::state::City::place_building`].
pub fn connections_at(city: &City, cell: IVec2) -> RoadConnections {
    RoadConnections {
        north: is_road(city, cell + Direction::North.offset()),
        south: is_road(city, cell + Direction::South.offset()),
        east: is_road(city, cell + Direction::East.offset()),
        west: is_road(city, cell + Direction::West.offset()),
    }
}

/// Every cell reachable from `start` through an unbroken chain of
/// orthogonally-adjacent road cells, `start` included. Empty if `start`
/// itself isn't a road cell — a non-road cell reaches nothing, not even
/// itself, so callers can't mistake "reaches only itself" (a one-cell road
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

    while let Some(cell) = queue.pop_front() {
        for direction in Direction::ALL {
            let neighbour = cell + direction.offset();
            if is_road(city, neighbour) && visited.insert(neighbour) {
                queue.push_back(neighbour);
            }
        }
    }

    visited
}

/// Whether `a` and `b` are both road cells connected through an unbroken
/// chain of road cells — the two-cell special case of [`reachable_from`].
/// `false` if either isn't a road cell at all, not just if they're
/// unreachable from each other.
#[allow(dead_code)] // no caller yet — F4's connectivity queries are
pub fn is_connected(city: &City, a: IVec2, b: IVec2) -> bool {
    is_road(city, a) && is_road(city, b) && reachable_from(city, a).contains(&b)
}

/// Which shape a road cell's connections call for — the piece-selection half
/// of F3's auto-tiling. Six kinds, one per connection count except 2 (which
/// splits into straight and corner, since a rotated straight piece can't
/// stand in for a bend — see ticket 054). [`super::road_catalogue`] loads
/// exactly one `.nbt` blueprint per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoadPieceKind {
    /// No neighbours at all.
    Isolated,
    /// Exactly one neighbour.
    DeadEnd,
    /// Two *opposite* neighbours (north+south, or east+west) — passes
    /// straight through.
    Straight,
    /// Two *adjacent* neighbours (e.g. north+east) — bends 90°.
    Corner,
    /// Three neighbours.
    T,
    /// All four neighbours.
    Cross,
}

impl RoadPieceKind {
    /// Every kind, for [`super::road_catalogue`] to iterate when loading the
    /// fixed set of `.nbt` files it expects one of.
    pub const ALL: [RoadPieceKind; 6] = [
        RoadPieceKind::Isolated,
        RoadPieceKind::DeadEnd,
        RoadPieceKind::Straight,
        RoadPieceKind::Corner,
        RoadPieceKind::T,
        RoadPieceKind::Cross,
    ];
}

/// The canonical connection pattern each oriented [`RoadPieceKind`] is
/// authored at — the shape its own `.nbt` file is assumed to have been built
/// to match. [`select_piece`] rotates these until one equals the actual
/// connections; [`RoadPieceKind::Isolated`]/[`RoadPieceKind::Cross`] aren't
/// here because neither has an orientation to search over (see
/// [`select_piece`]).
fn canonical_pattern(kind: RoadPieceKind) -> RoadConnections {
    match kind {
        RoadPieceKind::Isolated => RoadConnections::default(),
        RoadPieceKind::DeadEnd => RoadConnections { north: true, ..RoadConnections::default() },
        RoadPieceKind::Straight => RoadConnections { north: true, south: true, ..RoadConnections::default() },
        RoadPieceKind::Corner => RoadConnections { north: true, east: true, ..RoadConnections::default() },
        // Missing south: a T pointing away from south.
        RoadPieceKind::T => RoadConnections { north: true, east: true, west: true, ..RoadConnections::default() },
        RoadPieceKind::Cross => RoadConnections { north: true, south: true, east: true, west: true },
    }
}

/// The quarter-turn count (0..4) that rotates `canonical` into `actual`, via
/// [`RoadConnections::rotated`]. Every call site already knows a match
/// exists (`canonical`'s neighbour *count* was chosen to match `actual`'s —
/// see [`select_piece`]), so this panics rather than returning `Option` if
/// it doesn't: a mismatch here is a bug in this module's own table, not
/// something a caller needs to handle.
fn matching_rotation(canonical: RoadConnections, actual: RoadConnections) -> Rotation {
    for turns in 0..4u8 {
        if canonical.rotated(turns) == actual {
            return match turns {
                0 => Rotation::Deg0,
                1 => Rotation::Deg90,
                2 => Rotation::Deg180,
                _ => Rotation::Deg270,
            };
        }
    }
    unreachable!("canonical_pattern's shape always has some rotation matching an actual pattern of the same count/adjacency");
}

/// The piece kind and rotation `connections` calls for. Every oriented piece
/// is authored once at [`canonical_pattern`]'s shape and rotated to match —
/// the [`Rotation`] returned is exactly what
/// [`crate::blueprint::rotate_blueprint`] needs to turn that piece's
/// blueprint into the shape `connections` actually describes.
/// [`RoadPieceKind::Isolated`] and [`RoadPieceKind::Cross`] always come back
/// [`Rotation::Deg0`] — neither has an orientation (no neighbours, or all
/// four) for a rotation to mean anything about.
pub fn select_piece(connections: RoadConnections) -> (RoadPieceKind, Rotation) {
    match connections.count() {
        0 => (RoadPieceKind::Isolated, Rotation::Deg0),
        4 => (RoadPieceKind::Cross, Rotation::Deg0),
        1 => (RoadPieceKind::DeadEnd, matching_rotation(canonical_pattern(RoadPieceKind::DeadEnd), connections)),
        3 => (RoadPieceKind::T, matching_rotation(canonical_pattern(RoadPieceKind::T), connections)),
        2 => {
            let opposite = (connections.north && connections.south) || (connections.east && connections.west);
            if opposite {
                (RoadPieceKind::Straight, matching_rotation(canonical_pattern(RoadPieceKind::Straight), connections))
            } else {
                (RoadPieceKind::Corner, matching_rotation(canonical_pattern(RoadPieceKind::Corner), connections))
            }
        }
        _ => unreachable!("RoadConnections::count() is always 0..=4"),
    }
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
    fn connections_at_reports_no_neighbours_for_an_isolated_road_cell() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0)).unwrap();

        let connections = connections_at(&city, IVec2::new(0, 0));
        assert_eq!(connections, RoadConnections::default());
        assert_eq!(connections.count(), 0);
    }

    #[test]
    fn connections_at_reports_exactly_the_road_neighbours() {
        let mut city = City::default();
        // A road cell with north and east neighbours, but not south or west.
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        city.add_road_cell(IVec2::new(0, -1)).unwrap(); // north
        city.add_road_cell(IVec2::new(1, 0)).unwrap(); // east

        let connections = connections_at(&city, IVec2::new(0, 0));
        assert_eq!(connections, RoadConnections { north: true, south: false, east: true, west: false });
        assert_eq!(connections.count(), 2);
    }

    #[test]
    fn connections_at_ignores_a_building_neighbour() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        // Cell (0, -1) is north of (0, 0): block tiles -6..0 x z. Place a
        // building inside that block range so it's the road cell's north
        // neighbour, but as a building, not a road.
        city.place_building("house01", bevy::math::IVec3::new(0, 64, -6), crate::blueprint::Rotation::Deg0, IVec2::ONE)
            .unwrap();

        let connections = connections_at(&city, IVec2::new(0, 0));
        assert!(!connections.north);
    }

    #[test]
    fn connections_at_works_from_a_cell_that_is_not_itself_a_road() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, -1)).unwrap();

        // Asking "what would connect here" before placing anything.
        let connections = connections_at(&city, IVec2::new(0, 0));
        assert_eq!(connections, RoadConnections { north: true, south: false, east: false, west: false });
    }

    #[test]
    fn connections_at_reports_a_full_cross() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        for offset in [IVec2::new(0, -1), IVec2::new(0, 1), IVec2::new(1, 0), IVec2::new(-1, 0)] {
            city.add_road_cell(offset).unwrap();
        }

        assert_eq!(connections_at(&city, IVec2::new(0, 0)).count(), 4);
    }

    #[test]
    fn reachable_from_a_non_road_cell_is_empty() {
        let city = City::default();
        assert!(reachable_from(&city, IVec2::new(0, 0)).is_empty());
    }

    #[test]
    fn reachable_from_a_single_cell_island_is_only_itself() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(5, 5)).unwrap();
        assert_eq!(reachable_from(&city, IVec2::new(5, 5)), HashSet::from([IVec2::new(5, 5)]));
    }

    #[test]
    fn reachable_from_walks_a_straight_run() {
        let mut city = City::default();
        for x in 0..5 {
            city.add_road_cell(IVec2::new(x, 0)).unwrap();
        }

        let reached = reachable_from(&city, IVec2::new(0, 0));
        let expected: HashSet<IVec2> = (0..5).map(|x| IVec2::new(x, 0)).collect();
        assert_eq!(reached, expected);
    }

    #[test]
    fn reachable_from_walks_a_branch() {
        let mut city = City::default();
        // A horizontal run with one cell branching south from the middle.
        for x in 0..3 {
            city.add_road_cell(IVec2::new(x, 0)).unwrap();
        }
        city.add_road_cell(IVec2::new(1, 1)).unwrap();

        let reached = reachable_from(&city, IVec2::new(0, 0));
        assert_eq!(
            reached,
            HashSet::from([IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(2, 0), IVec2::new(1, 1)])
        );
    }

    #[test]
    fn reachable_from_a_loop_terminates_and_does_not_double_count() {
        let mut city = City::default();
        // A 2x2 loop of road cells.
        for cell in [IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(0, 1), IVec2::new(1, 1)] {
            city.add_road_cell(cell).unwrap();
        }

        let reached = reachable_from(&city, IVec2::new(0, 0));
        assert_eq!(reached.len(), 4);
    }

    #[test]
    fn reachable_from_does_not_cross_to_a_disconnected_island() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        city.add_road_cell(IVec2::new(1, 0)).unwrap();
        // A second island, far away and not adjacent to the first.
        city.add_road_cell(IVec2::new(100, 100)).unwrap();

        let reached = reachable_from(&city, IVec2::new(0, 0));
        assert_eq!(reached, HashSet::from([IVec2::new(0, 0), IVec2::new(1, 0)]));
    }

    #[test]
    fn is_connected_is_true_within_one_island() {
        let mut city = City::default();
        for x in 0..3 {
            city.add_road_cell(IVec2::new(x, 0)).unwrap();
        }
        assert!(is_connected(&city, IVec2::new(0, 0), IVec2::new(2, 0)));
    }

    #[test]
    fn is_connected_is_false_across_two_islands() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        city.add_road_cell(IVec2::new(100, 100)).unwrap();
        assert!(!is_connected(&city, IVec2::new(0, 0), IVec2::new(100, 100)));
    }

    #[test]
    fn is_connected_is_false_when_an_endpoint_is_not_a_road_cell() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        assert!(!is_connected(&city, IVec2::new(0, 0), IVec2::new(1, 0)));
        assert!(!is_connected(&city, IVec2::new(1, 0), IVec2::new(0, 0)));
    }

    // -- select_piece (ticket 054, roadmap F3's selection half) -------------

    #[test]
    fn select_piece_picks_isolated_and_cross_with_no_rotation_search() {
        assert_eq!(select_piece(RoadConnections::default()), (RoadPieceKind::Isolated, Rotation::Deg0));
        let full = RoadConnections { north: true, south: true, east: true, west: true };
        assert_eq!(select_piece(full), (RoadPieceKind::Cross, Rotation::Deg0));
    }

    #[test]
    fn select_piece_distinguishes_straight_from_corner_at_the_same_count() {
        let straight = RoadConnections { north: true, south: true, ..RoadConnections::default() };
        assert_eq!(select_piece(straight).0, RoadPieceKind::Straight);

        let corner = RoadConnections { north: true, east: true, ..RoadConnections::default() };
        assert_eq!(select_piece(corner).0, RoadPieceKind::Corner);
    }

    /// Every one of the 16 possible neighbour combinations resolves to the
    /// kind its neighbour count/adjacency implies, and rotating that kind's
    /// own canonical pattern by the returned [`Rotation`] reproduces the
    /// exact input — the round-trip property, not just one example per kind.
    #[test]
    fn select_piece_round_trips_every_connection_pattern() {
        for north in [false, true] {
            for south in [false, true] {
                for east in [false, true] {
                    for west in [false, true] {
                        let connections = RoadConnections { north, south, east, west };
                        let (kind, rotation) = select_piece(connections);

                        let expected_kind = match connections.count() {
                            0 => RoadPieceKind::Isolated,
                            4 => RoadPieceKind::Cross,
                            1 => RoadPieceKind::DeadEnd,
                            3 => RoadPieceKind::T,
                            2 if (north && south) || (east && west) => RoadPieceKind::Straight,
                            2 => RoadPieceKind::Corner,
                            _ => unreachable!(),
                        };
                        assert_eq!(kind, expected_kind, "{connections:?}");

                        if matches!(kind, RoadPieceKind::Isolated | RoadPieceKind::Cross) {
                            assert_eq!(rotation, Rotation::Deg0);
                        }

                        let turns = match rotation {
                            Rotation::Deg0 => 0,
                            Rotation::Deg90 => 1,
                            Rotation::Deg180 => 2,
                            Rotation::Deg270 => 3,
                        };
                        assert_eq!(
                            canonical_pattern(kind).rotated(turns),
                            connections,
                            "rotating {kind:?}'s canonical pattern by {rotation:?} should reproduce {connections:?}"
                        );
                    }
                }
            }
        }
    }
}
