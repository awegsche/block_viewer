//! The `City` resource (ticket 042, roadmap D1): placed buildings, roads,
//! and a footprint occupancy grid. This is the authoritative half of the
//! rule the whole citybuilder plan follows:
//!
//! > The city state is authoritative. The blocks in the world are a
//! > projection of it.
//!
//! Everything here is pure data plus the invariants that keep it internally
//! consistent — no picking (roadmap E1), no terrain fit (E2), no ghost
//! preview (E3), and no glue to the write path (E4, `crate::edit`). Those
//! are all *readers* or *writers* of a [`City`]; none of them are this
//! ticket. Neither is persistence (D2) or a journal (D3) — [`City`] lives in
//! memory only, for now.
//!
//! ## Two kinds of id
//!
//! A [`BuildingId`] names one *placed instance* — assigned by
//! [`City::place_building`], never reused. A definition id (`"house01"`,
//! [`super::definition::BuildingDefinitions`]'s key) names a building
//! *type*. Two houses placed side by side share the second and must not
//! share the first, which is why [`PlacedBuilding::definition`] is a
//! `String` (the type) while [`City::place_building`]'s return value is a
//! fresh [`BuildingId`] (the instance).
//!
//! ## The occupancy grid
//!
//! One `HashMap<IVec2, Occupant>`, keyed by Minecraft `(x, z)` — the same
//! horizontal-only convention [`super::blueprint`]'s `CatalogueEntry`
//! already uses for a footprint (Y doesn't factor in; the grid a building
//! sits on is horizontal). "Is this tile free" is a lookup, not a scan over
//! every building — the roadmap names this explicitly as D1's job.
//!
//! ## Rotation and the occupied rectangle
//!
//! A footprint stored on [`PlacedBuilding`] or a catalogue entry is always
//! the *unrotated* size. [`blueprint::rotate_blueprint`](crate::blueprint::rotate_blueprint)
//! already established that 90°/270° about Y is a real transform on the
//! block grid, not just the mesh; [`footprint_extent`] is the same swap
//! applied to the footprint alone, and [`footprint_tiles`] is what every
//! occupancy computation in this module goes through rather than reading
//! `footprint.x`/`footprint.y` directly — the difference between a building
//! occupying the rectangle it actually covers and one occupying the
//! rectangle it would have covered unrotated.
//!
//! ## All-or-nothing placement
//!
//! [`City::place_building`] computes every tile a footprint covers, checks
//! all of them are free, and only *then* mutates the grid — the same "plan
//! before apply" shape [`crate::edit::route::apply_routed`] uses for the
//! write path. A footprint spans many tiles; refusing partway through and
//! leaving the first few marked occupied would leave the grid holding a
//! phantom building nobody actually placed.
//!
//! ## Real callers, as of tickets 048 and 049
//!
//! [`City::place_building`]/[`City::remove_building`] were exercised only by
//! this module's own tests until `city::commit` (ticket 048, roadmap E4)
//! started calling them — `place_building` on a click, `remove_building` as
//! the rollback when the write that click started fails. `city::demolish`
//! (ticket 049, roadmap E5) is `remove_building`'s second, deliberate
//! caller, and adds two more: [`City::occupant_at`] finds what's under the
//! cursor, [`City::building`] looks up its placement. Everything past those
//! four (`remove_road_cell`, `len`, `is_empty`) is still only proven by tests;
//! their own `#[allow(dead_code)]` marks are that same "no caller yet"
//! situation, not a note-worthy call each time it recurs.

use std::collections::{HashMap, HashSet};

use bevy::math::{IVec2, IVec3};
use bevy::prelude::Resource;

use crate::blueprint::Rotation;

/// Identifies one *placed* building instance — never a building *type* (see
/// the module docs). Assigned monotonically by [`City::place_building`];
/// never reused, even after [`City::remove_building`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuildingId(u64);

impl BuildingId {
    /// The raw instance counter value — ticket 043's persistence module is
    /// the only caller; everything else names a building by [`BuildingId`]
    /// itself; not the `u64` underneath it.
    pub(crate) fn as_u64(self) -> u64 {
        self.0
    }

    /// The inverse of [`as_u64`](Self::as_u64), for reconstructing a
    /// [`BuildingId`] read back off disk. Not a `From<u64>` impl —
    /// deliberately awkward to reach for outside `persistence::load_city`,
    /// where alone it's correct to hand back an id [`City::place_building`]
    /// never minted itself.
    pub(crate) fn from_u64(id: u64) -> Self {
        BuildingId(id)
    }
}

/// One building placed in the city.
///
/// `Clone` (ticket 044, roadmap D3) so a journal entry can hold its own copy
/// of a demolished building's placement — the record undo needs to put it
/// back — without the entry borrowing from `City` and outliving it.
#[derive(Debug, Clone)]
pub struct PlacedBuilding {
    /// The building's *type* — a key into
    /// [`super::definition::BuildingDefinitions`]/[`super::blueprint::BuildingCatalogue`],
    /// not this instance's own [`BuildingId`].
    pub definition: String,
    /// Minecraft world coordinates of the footprint's minimum `(x, z)`
    /// corner (the same convention [`crate::edit`]/[`crate::selection`] use
    /// throughout — no `bevy.z = -mc.z` flip belongs in this module).
    pub origin: IVec3,
    pub rotation: Rotation,
    /// The building's *unrotated* footprint — what
    /// [`super::blueprint::CatalogueEntry::footprint`] or
    /// [`super::definition::LoadedBuilding::footprint`] already resolved.
    /// Kept on the placement itself so [`City::remove_building`] can free
    /// the right tiles without the caller having to look the definition back
    /// up.
    pub footprint: IVec2,
}

/// What one tile of the occupancy grid holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occupant {
    Building(BuildingId),
    Road,
}

/// Why a placement was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementError {
    /// `tile` is already held by `by` — a building, or a road, in the way.
    TileOccupied { tile: IVec2, by: Occupant },
}

impl std::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementError::TileOccupied { tile, by } => {
                let what = match by {
                    Occupant::Building(id) => format!("building {}", id.0),
                    Occupant::Road => "a road".to_string(),
                };
                write!(f, "tile ({}, {}) is already occupied by {what}", tile.x, tile.y)
            }
        }
    }
}

impl std::error::Error for PlacementError {}

/// How a footprint's `(x, z)` extent changes under a Y rotation: 0°/180°
/// keep the axes, 90°/270° swap them — see the module docs.
pub fn footprint_extent(footprint: IVec2, rotation: Rotation) -> IVec2 {
    match rotation {
        Rotation::Deg0 | Rotation::Deg180 => footprint,
        Rotation::Deg90 | Rotation::Deg270 => IVec2::new(footprint.y, footprint.x),
    }
}

/// Every Minecraft `(x, z)` tile a footprint covers when placed at `origin`
/// with `rotation` — the rectangle [`footprint_extent`] describes, walked
/// one tile per block, with `origin` as its minimum corner regardless of
/// rotation.
pub fn footprint_tiles(origin: IVec3, footprint: IVec2, rotation: Rotation) -> impl Iterator<Item = IVec2> {
    let extent = footprint_extent(footprint, rotation);
    let base = IVec2::new(origin.x, origin.z);
    (0..extent.x).flat_map(move |dx| (0..extent.y).map(move |dz| base + IVec2::new(dx, dz)))
}

/// Width and depth, in blocks, of one road cell (ticket 054, roadmap F1/F3):
/// 1 shoulder (grass, etc.) + 1 kerb + 2 road surface + 1 kerb + 1 shoulder
/// on each cross-section, laid out symmetrically — `1+1+2+1+1 = 6`. A road
/// is placed and connects to its neighbours one cell at a time, not one
/// block at a time, the way a building's footprint is one unit regardless of
/// how many blocks it covers.
pub const ROAD_CELL_SIZE: i32 = 6;

/// Every Minecraft `(x, z)` block tile one road cell covers — `cell` scaled
/// up by [`ROAD_CELL_SIZE`], `cell`'s own `(x, z)` naming its *minimum*
/// corner. The cell-space counterpart of [`footprint_tiles`]; a road cell
/// has no rotation of its own to fold in here (unlike a footprint) —
/// [`super::road::select_piece`] is what a cell's orientation feeds into,
/// downstream of occupancy, not upstream of it.
pub fn road_cell_tiles(cell: IVec2) -> impl Iterator<Item = IVec2> {
    let base = cell * ROAD_CELL_SIZE;
    (0..ROAD_CELL_SIZE).flat_map(move |dx| (0..ROAD_CELL_SIZE).map(move |dz| base + IVec2::new(dx, dz)))
}

/// The authoritative city state: placed buildings, roads, and the occupancy
/// grid both are checked against. See the module docs for the rule this
/// implements and what is deliberately not here yet (a journal, anything
/// that reads or writes the actual world). Persistence (D2, ticket 043) is
/// [`super::persistence`], which reads and writes this through
/// [`buildings`](Self::buildings)/[`roads`](Self::roads)/[`insert_loaded`](Self::insert_loaded)/
/// [`add_road`](Self::add_road) rather than serializing this struct
/// directly — `occupancy` is derived, not stored.
#[allow(dead_code)] // remaining fields/methods are exercised only by tests — see the module docs
#[derive(Resource, Default, Debug)]
pub struct City {
    buildings: HashMap<BuildingId, PlacedBuilding>,
    next_id: u64,
    /// Cell coordinates (ticket 054) — not block tiles. `occupancy` still
    /// holds a block-tile [`Occupant::Road`] for every one of a cell's 36
    /// tiles, kept in lockstep by [`add_road_cell`](Self::add_road_cell)/
    /// [`remove_road_cell`](Self::remove_road_cell); this set is the source
    /// of truth for "is this cell a road," the way `buildings` is for
    /// buildings.
    road_cells: HashSet<IVec2>,
    occupancy: HashMap<IVec2, Occupant>,
}

impl City {
    /// Places a building of type `definition` at `origin`/`rotation`,
    /// covering `footprint`'s rotated extent. All-or-nothing: every tile is
    /// checked before any of them is marked occupied, so a refusal never
    /// leaves a partial building behind — see the module docs.
    ///
    /// Called synchronously by `city::commit::try_commit_placement` (ticket
    /// 048, roadmap E4) the instant a click is accepted — before the write
    /// even starts, so the tile claim exists first. See that module's docs
    /// for why the write is transactional against this call:
    /// [`remove_building`](Self::remove_building) is the rollback.
    pub fn place_building(
        &mut self,
        definition: impl Into<String>,
        origin: IVec3,
        rotation: Rotation,
        footprint: IVec2,
    ) -> Result<BuildingId, PlacementError> {
        let tiles: Vec<IVec2> = footprint_tiles(origin, footprint, rotation).collect();
        for &tile in &tiles {
            if let Some(&by) = self.occupancy.get(&tile) {
                return Err(PlacementError::TileOccupied { tile, by });
            }
        }

        let id = BuildingId(self.next_id);
        self.next_id += 1;
        for &tile in &tiles {
            self.occupancy.insert(tile, Occupant::Building(id));
        }
        self.buildings.insert(
            id,
            PlacedBuilding { definition: definition.into(), origin, rotation, footprint },
        );
        Ok(id)
    }

    /// Removes a placed building and frees exactly the tiles its own record
    /// covers. `None` if `id` isn't a currently-placed building — removing
    /// twice, or an id that was never valid, isn't a panic.
    ///
    /// `city::commit::poll_commit` (ticket 048) calls this on a *failed*
    /// write — the rollback half of [`place_building`](Self::place_building)'s
    /// docs. `city::demolish::poll_demolish` (ticket 049, roadmap E5) is the
    /// second, deliberate caller — and, unlike commit's rollback, only after
    /// its own restoring write has already succeeded; see that module's docs
    /// for why the two callers free the tile at opposite ends of their write.
    pub fn remove_building(&mut self, id: BuildingId) -> Option<PlacedBuilding> {
        let building = self.buildings.remove(&id)?;
        for tile in footprint_tiles(building.origin, building.footprint, building.rotation) {
            self.occupancy.remove(&tile);
        }
        Some(building)
    }

    /// `city::demolish::resolve_demolition_target` (ticket 049, roadmap E5)
    /// is the real caller: `occupant_at` finds the id, this looks up the
    /// placement itself.
    pub fn building(&self, id: BuildingId) -> Option<&PlacedBuilding> {
        self.buildings.get(&id)
    }

    pub fn buildings(&self) -> impl Iterator<Item = (BuildingId, &PlacedBuilding)> {
        self.buildings.iter().map(|(&id, b)| (id, b))
    }

    /// Inserts a building under an id the caller supplies, rather than
    /// minting a fresh one — [`place_building`](Self::place_building)'s
    /// counterpart for [`persistence::load_city`](super::persistence::load_city),
    /// the only caller: a save file already recorded a [`BuildingId`] for
    /// every placement, and reissuing new ones on load would let two
    /// sessions disagree about which id names which building.
    ///
    /// Same all-or-nothing shape as `place_building` — every tile is checked
    /// before any of them is marked occupied — so a save file with two
    /// overlapping buildings fails the load rather than producing a City
    /// whose occupancy grid disagrees with itself. Also raises `next_id`
    /// past `id`, the same bump a fresh placement gets, so an id read back
    /// off disk is never handed out again by a later `place_building` call.
    pub(crate) fn insert_loaded(&mut self, id: BuildingId, building: PlacedBuilding) -> Result<(), PlacementError> {
        let tiles: Vec<IVec2> = footprint_tiles(building.origin, building.footprint, building.rotation).collect();
        for &tile in &tiles {
            if let Some(&by) = self.occupancy.get(&tile) {
                return Err(PlacementError::TileOccupied { tile, by });
            }
        }

        for &tile in &tiles {
            self.occupancy.insert(tile, Occupant::Building(id));
        }
        self.next_id = self.next_id.max(id.as_u64() + 1);
        self.buildings.insert(id, building);
        Ok(())
    }

    /// The raw instance counter, for [`persistence::save_city`](super::persistence::save_city)
    /// to record — see that module's docs for why it has to be persisted
    /// rather than recomputed from the surviving buildings alone.
    pub(crate) fn next_id_raw(&self) -> u64 {
        self.next_id
    }

    /// Raises `next_id` to `at_least` if it isn't there already — never
    /// lowers it. [`persistence::load_city`](super::persistence::load_city)'s
    /// last step: `insert_loaded` already bumped `next_id` past every id it
    /// inserted, but a building can be removed *after* being placed and
    /// *before* being saved, which drops its id from `buildings` without
    /// rolling `next_id` back — the file's own recorded value is what this
    /// restores, see the module docs' removed-highest-building scenario.
    pub(crate) fn raise_next_id(&mut self, at_least: u64) {
        self.next_id = self.next_id.max(at_least);
    }

    /// Marks `cell` (cell coordinates, ticket 054 — see [`ROAD_CELL_SIZE`])
    /// as a road, occupying all 36 block tiles [`road_cell_tiles`] lists for
    /// it. Idempotent if `cell` is already a road; all-or-nothing and
    /// refused if *any* of those 36 tiles are held by a building or another
    /// road cell — the same "plan every tile before marking any of them"
    /// shape [`place_building`](Self::place_building) uses, so a refused
    /// road cell never leaves a partial one behind.
    pub fn add_road_cell(&mut self, cell: IVec2) -> Result<(), PlacementError> {
        if self.road_cells.contains(&cell) {
            return Ok(());
        }

        let tiles: Vec<IVec2> = road_cell_tiles(cell).collect();
        for &tile in &tiles {
            if let Some(&by) = self.occupancy.get(&tile) {
                return Err(PlacementError::TileOccupied { tile, by });
            }
        }

        for &tile in &tiles {
            self.occupancy.insert(tile, Occupant::Road);
        }
        self.road_cells.insert(cell);
        Ok(())
    }

    /// Clears a road cell and frees all 36 of its block tiles. Returns
    /// whether `cell` was actually a road cell beforehand.
    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn remove_road_cell(&mut self, cell: IVec2) -> bool {
        if self.road_cells.remove(&cell) {
            for tile in road_cell_tiles(cell) {
                self.occupancy.remove(&tile);
            }
            true
        } else {
            false
        }
    }

    pub fn road_cells(&self) -> impl Iterator<Item = &IVec2> {
        self.road_cells.iter()
    }

    /// Whether `cell` is a road cell — [`city::road`](super::road)'s own
    /// adjacency queries go through this rather than re-deriving it from
    /// [`occupant_at`](Self::occupant_at) at one of the cell's 36 tiles.
    pub fn is_road_cell(&self, cell: IVec2) -> bool {
        self.road_cells.contains(&cell)
    }

    pub fn is_tile_free(&self, tile: IVec2) -> bool {
        !self.occupancy.contains_key(&tile)
    }

    /// `city::demolish::resolve_demolition_target` (ticket 049, roadmap E5)
    /// is the real caller — the first check on a `Delete` press: is there
    /// even a building on this tile.
    pub fn occupant_at(&self, tile: IVec2) -> Option<Occupant> {
        self.occupancy.get(&tile).copied()
    }

    /// Number of placed buildings. Not roads or occupied tiles — the same
    /// "how many things has the player built" count a city panel (G2) would
    /// want, not a grid-size figure.
    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn len(&self) -> usize {
        self.buildings.len()
    }

    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn is_empty(&self) -> bool {
        self.buildings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placing_a_building_occupies_exactly_its_footprint() {
        let mut city = City::default();
        let origin = IVec3::new(10, 64, 20);
        let footprint = IVec2::new(3, 2);
        let id = city
            .place_building("house01", origin, Rotation::Deg0, footprint)
            .expect("should place on an empty grid");

        for x in 10..13 {
            for z in 20..22 {
                assert_eq!(city.occupant_at(IVec2::new(x, z)), Some(Occupant::Building(id)));
            }
        }
        // One tile outside the footprint on every side stays free.
        assert!(city.is_tile_free(IVec2::new(9, 20)));
        assert!(city.is_tile_free(IVec2::new(13, 20)));
        assert!(city.is_tile_free(IVec2::new(10, 19)));
        assert!(city.is_tile_free(IVec2::new(10, 22)));
    }

    #[test]
    fn a_90_degree_rotation_swaps_the_occupied_extent() {
        let mut city = City::default();
        let origin = IVec3::new(0, 64, 0);
        let footprint = IVec2::new(3, 5); // 3 wide (x), 5 deep (z), unrotated
        let id = city
            .place_building("house01", origin, Rotation::Deg90, footprint)
            .expect("should place on an empty grid");

        // Rotated 90°: the occupied rectangle is 5 wide (x), 3 deep (z).
        for x in 0..5 {
            for z in 0..3 {
                assert_eq!(city.occupant_at(IVec2::new(x, z)), Some(Occupant::Building(id)));
            }
        }
        assert!(city.is_tile_free(IVec2::new(5, 0)), "unrotated width would have stopped at x=3, not x=5");
        assert!(city.is_tile_free(IVec2::new(0, 3)), "unrotated depth would have stopped at z=5, not z=3");
    }

    #[test]
    fn an_overlapping_placement_is_refused_and_leaves_the_grid_unchanged() {
        let mut city = City::default();
        let first = city
            .place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(4, 4))
            .unwrap();

        // Overlaps the first building's last column/row (x=3, z=3..7).
        let err = city
            .place_building("house01", IVec3::new(3, 64, 3), Rotation::Deg0, IVec2::new(4, 4))
            .unwrap_err();
        assert!(matches!(
            err,
            PlacementError::TileOccupied { tile, by: Occupant::Building(id) }
                if tile == IVec2::new(3, 3) && id == first
        ));

        // None of the second building's other tiles got marked occupied —
        // an all-or-nothing refusal, not a partial one.
        assert!(city.is_tile_free(IVec2::new(6, 6)));
        assert!(city.is_tile_free(IVec2::new(4, 4)));
        assert_eq!(city.len(), 1, "the failed placement must not have been recorded");
    }

    #[test]
    fn placing_a_road_cell_occupies_exactly_its_36_tiles() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(1, 2)).unwrap();

        for x in 6..12 {
            for z in 12..18 {
                assert_eq!(city.occupant_at(IVec2::new(x, z)), Some(Occupant::Road));
            }
        }
        // One tile outside the cell on every side stays free.
        assert!(city.is_tile_free(IVec2::new(5, 12)));
        assert!(city.is_tile_free(IVec2::new(12, 12)));
        assert!(city.is_tile_free(IVec2::new(6, 11)));
        assert!(city.is_tile_free(IVec2::new(6, 18)));
    }

    #[test]
    fn a_road_cell_cannot_be_placed_on_a_building_and_vice_versa() {
        let mut city = City::default();
        let building_id = city
            .place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        // Cell (0, 0) covers block tiles 0..6 x 0..6, which overlaps the
        // building's 2x2 footprint at its very first checked tile.
        let err = city.add_road_cell(IVec2::new(0, 0)).unwrap_err();
        assert!(matches!(
            err,
            PlacementError::TileOccupied { tile, by: Occupant::Building(id) }
                if tile == IVec2::new(0, 0) && id == building_id
        ));

        city.add_road_cell(IVec2::new(5, 5)).expect("an empty cell should accept a road");
        // Cell (5, 5) covers block tiles 30..36 x 30..36.
        let err = city
            .place_building("house01", IVec3::new(30, 64, 30), Rotation::Deg0, IVec2::new(1, 1))
            .unwrap_err();
        assert!(matches!(
            err,
            PlacementError::TileOccupied { tile, by: Occupant::Road } if tile == IVec2::new(30, 30)
        ));
    }

    #[test]
    fn adding_the_same_road_cell_twice_is_a_no_op() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(1, 1)).unwrap();
        city.add_road_cell(IVec2::new(1, 1)).expect("re-adding the same road cell should succeed");
        assert_eq!(city.road_cells().count(), 1);
    }

    #[test]
    fn removing_a_building_frees_its_tiles_for_reuse() {
        let mut city = City::default();
        let origin = IVec3::new(0, 64, 0);
        let footprint = IVec2::new(2, 2);
        let id = city.place_building("house01", origin, Rotation::Deg0, footprint).unwrap();

        let removed = city.remove_building(id).expect("the id was just placed");
        assert_eq!(removed.definition, "house01");
        assert!(city.is_tile_free(IVec2::new(0, 0)));
        assert!(city.is_empty());

        // The freed tiles accept a new placement.
        city.place_building("house01", origin, Rotation::Deg0, footprint)
            .expect("tiles freed by removal should be placeable again");
    }

    #[test]
    fn removing_an_unknown_id_is_none_not_a_panic() {
        let mut city = City::default();
        let id = city.place_building("house01", IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap();
        city.remove_building(id);
        assert!(city.remove_building(id).is_none(), "already removed");
    }

    #[test]
    fn remove_road_cell_reports_whether_a_cell_was_actually_a_road() {
        let mut city = City::default();
        assert!(!city.remove_road_cell(IVec2::new(0, 0)), "never added");
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        assert!(city.remove_road_cell(IVec2::new(0, 0)));
        assert!(city.is_tile_free(IVec2::new(0, 0)));
        assert!(!city.is_road_cell(IVec2::new(0, 0)));

        // Removing frees every one of the cell's 36 tiles, not just its
        // corner.
        city.add_road_cell(IVec2::new(0, 0)).unwrap();
        city.remove_road_cell(IVec2::new(0, 0));
        for x in 0..6 {
            for z in 0..6 {
                assert!(city.is_tile_free(IVec2::new(x, z)));
            }
        }
    }

    #[test]
    fn footprint_extent_swaps_axes_only_on_a_quarter_turn() {
        let footprint = IVec2::new(3, 5);
        assert_eq!(footprint_extent(footprint, Rotation::Deg0), footprint);
        assert_eq!(footprint_extent(footprint, Rotation::Deg180), footprint);
        assert_eq!(footprint_extent(footprint, Rotation::Deg90), IVec2::new(5, 3));
        assert_eq!(footprint_extent(footprint, Rotation::Deg270), IVec2::new(5, 3));
    }

    #[test]
    fn buildings_and_roads_iterate_what_was_added() {
        let mut city = City::default();
        let a = city.place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        let b = city.place_building("house01", IVec3::new(5, 64, 5), Rotation::Deg0, IVec2::ONE).unwrap();
        city.add_road_cell(IVec2::new(2, 2)).unwrap();
        city.add_road_cell(IVec2::new(2, 3)).unwrap();

        let ids: HashSet<BuildingId> = city.buildings().map(|(id, _)| id).collect();
        assert_eq!(ids, HashSet::from([a, b]));
        assert_eq!(city.road_cells().count(), 2);
        assert_eq!(city.len(), 2);
    }
}
