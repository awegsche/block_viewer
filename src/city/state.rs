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
//! ## Three kinds of id
//!
//! A [`BuildingId`] names one *placed instance* — assigned by
//! [`City::place_building`], never reused. Two houses placed side by side
//! must not share it, which is why it's minted per placement.
//!
//! The other two both name a building *type*, and ticket 076 is where this
//! module stopped pretending they were one:
//!
//! - a **catalogue id** ([`PlacedBuilding::catalogue_id`]) is a blueprint's
//!   filename stem, [`super::blueprint::BuildingCatalogue`]'s key — the
//!   building's geometry;
//! - a **definition id** ([`PlacedBuilding::definition_id`]) is a `.ron`'s
//!   filename stem, [`super::definition::BuildingDefinitions`]'s key — the
//!   building's game data (tier, cost, production, integrity).
//!
//! They are separate keyspaces that happen to agree for the single shipped
//! `house01.nbt`/`house01.ron` pair. Until 076 only the first was stored,
//! under the name `definition`, which meant a *placed* building could not
//! find its own `.ron` — roadmap H2's own "known gap", and the thing every
//! per-instance mechanic from production onward needs.
//! [`City::definition_of`] is the lookup that replaced the guess.
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

use std::collections::HashMap;

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
    /// Which *geometry* this instance was placed from — a key into
    /// [`super::blueprint::BuildingCatalogue`] (a blueprint's filename stem),
    /// not this instance's own [`BuildingId`] and **not** a
    /// [`super::definition::BuildingDefinitions`] key.
    ///
    /// Ticket 076 renamed this from `definition`, which is what it had been
    /// called since 042 while holding a catalogue id the whole time: the two
    /// keyspaces are separate (`house01.nbt` -> `house01` for the catalogue,
    /// `house01.ron` -> `house01` for the definitions) and coincided only
    /// because the one shipped pair of files shares a stem. See
    /// [`definition_id`](Self::definition_id) for the other half.
    pub catalogue_id: String,
    /// Which *game data* this instance was placed from — a key into
    /// [`super::definition::BuildingDefinitions`], and the thing every
    /// per-instance mechanic from roadmap H2 onward (production rates,
    /// warehouse radii, `integrity` thresholds) has to go through to find
    /// this building's own `.ron`.
    ///
    /// `None` for a placement made through `super::placement`'s keyboard
    /// stand-in, which picks a *catalogue* entry and has no definition
    /// behind it at all — the same hole that already leaves such a placement
    /// with no requirements and no cost (ticket 073). A `None` here is
    /// "this building has no game data", not "look it up by
    /// [`catalogue_id`](Self::catalogue_id)": guessing that the stems
    /// coincide is precisely the bug ticket 076 exists to remove.
    pub definition_id: Option<String>,
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
    /// Ticket 111: where a gatherer digs — a player-drawn rectangle, `None`
    /// until one has been drawn (a hut with no area does nothing, see
    /// `city::gatherer`). Meaningless for a building without a `gatherer`
    /// block, and always `None` for one. Kept on the placement itself, not
    /// in a side map, for the same reason [`footprint`](Self::footprint) is:
    /// everything that already clones, saves or restores a `PlacedBuilding`
    /// (the journal's demolish-undo copy, `persistence`) carries it for
    /// free, so demolish-then-undo brings the area back without anyone
    /// learning it exists. Only [`City::set_work_area`] ever writes it.
    pub work_area: Option<WorkArea>,
    /// Ticket 128: `true` while this placement is still a **site** — its
    /// blueprint hasn't been written yet because the ground it displaces
    /// isn't clear (`city::construction`'s clearing tick is digging it out
    /// first). `false` for an ordinary, already-built placement, and for
    /// every placement that predates this ticket (see `persistence`'s
    /// `#[serde(default)]`). Lives on the placement itself, not a side map,
    /// for the same reason [`work_area`](Self::work_area) does — a site is a
    /// claimed row in [`City`], and everything that already clones/saves a
    /// `PlacedBuilding` carries the flag for free. Only
    /// [`City::mark_under_construction`]/[`City::complete_building`] ever
    /// write it.
    pub under_construction: bool,
}

/// An inclusive axis-aligned rectangle of Minecraft `(x, z)` block tiles —
/// a gatherer's working area (ticket 111), drawn by the player. Both
/// corners inclusive, unlike the `(min, max)`-exclusive pairs
/// [`footprint_tiles`]/`farm::rect_distance` deal in: this is the corner
/// pair a click-drag naturally produces (a one-tile drag is `min == max`),
/// and [`WorkArea::new`] takes either corner order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkArea {
    pub min: IVec2,
    pub max: IVec2,
}

impl WorkArea {
    /// The rectangle `a` and `b` span, either corner order.
    pub fn new(a: IVec2, b: IVec2) -> Self {
        Self { min: a.min(b), max: a.max(b) }
    }

    /// Every tile in the rectangle, row by row. Order isn't meaningful to
    /// any caller.
    pub fn tiles(self) -> impl Iterator<Item = IVec2> {
        let (min, max) = (self.min, self.max);
        (min.x..=max.x).flat_map(move |x| (min.y..=max.y).map(move |z| IVec2::new(x, z)))
    }

    /// Number of tiles.
    pub fn len(self) -> u32 {
        let extent = self.max - self.min + IVec2::ONE;
        (extent.x * extent.y) as u32
    }

    /// The part of this rectangle that lies within `radius` blocks
    /// (Chebyshev) of a footprint spanning `footprint_min..footprint_max`
    /// (`max` exclusive, the [`footprint_tiles`] convention) — the box that
    /// footprint expanded by `radius` on every side. `None` if nothing of it
    /// does: a drag entirely out of reach is refused, not silently moved.
    /// The footprint's own tiles are *not* cut out — the gatherer's
    /// occupancy guard already never digs them, and cutting a hole would
    /// turn one rectangle into up to four.
    pub fn clamp_to_reach(self, footprint_min: IVec2, footprint_max: IVec2, radius: i32) -> Option<Self> {
        let reach_min = footprint_min - IVec2::splat(radius);
        let reach_max = footprint_max + IVec2::splat(radius) - IVec2::ONE; // inclusive
        let min = self.min.max(reach_min);
        let max = self.max.min(reach_max);
        (min.x <= max.x && min.y <= max.y).then_some(Self { min, max })
    }
}

/// One placed road cell — the style it was built as, plus the world Y its
/// piece is written at.
///
/// `base_y` is resolved *once*, by `super::road_build::try_commit_drag`,
/// off the same [`super::grid::fit_footprint`] read the drag preview
/// already uses, and then never re-derived. Ticket 065: re-deriving it is
/// what broke roads in the first place — once a piece has been written,
/// [`super::grid::ground_height_at`] samples the road surface as ground, so
/// a neighbour being *re-tiled* (a dead end that just grew a neighbour) would
/// resample one block higher every time and walk itself up into the sky. The
/// cell remembers its own ground, the way [`PlacedBuilding::origin`] does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoadCell {
    /// `assets/city/roads/<style>` — [`super::road_catalogue::RoadCatalogue`]'s
    /// own key (ticket 059).
    pub style: String,
    /// The Minecraft world Y [`super::grid::fit_footprint`] fitted this cell
    /// at — "one above the ground", the same convention
    /// [`PlacedBuilding::origin`]'s own Y carries.
    ///
    /// Not the piece's write origin: a road piece carries subgrade layers
    /// below its surface course, so `super::road_build::cell_write_origin`
    /// offsets this before handing it to `super::commit::blueprint_edit`.
    /// What's stored here is the *ground reading*, so a piece whose geometry
    /// changes shape doesn't invalidate every recorded cell.
    ///
    /// For a **stair** cell ([`Self::ascent`]) this is the *low* end: the
    /// piece climbs from here to `base_y + ROAD_STAIR_RISE` across the cell.
    pub base_y: i32,
    /// `None` for an ordinary flat cell. `Some(direction)` marks this cell a
    /// [`RoadPieceKind::Stair`](super::road::RoadPieceKind::Stair) climbing
    /// `super::road_build::ROAD_STAIR_RISE` blocks toward `direction` — so
    /// its low end (and [`Self::base_y`]) is on the `direction.opposite()`
    /// side, and the flat cell it meets on the `direction` side records
    /// `base_y + ROAD_STAIR_RISE`.
    ///
    /// Ticket 067. Stored rather than derived from the neighbours' heights
    /// for the same reason `base_y` is: a cell's shape has to survive its
    /// neighbours changing, and a ramp that recomputed which way it pointed
    /// every time a road was built beside it would flip under the player.
    /// Nothing about a cell's *connections* can imply this either — a stair
    /// and a straight connect identically, which is why
    /// [`super::road::select_piece`] never returns `Stair`.
    pub ascent: Option<super::road::Direction>,
    /// Which of the style's alternate pieces this cell resolved to — the
    /// ordinary open-sky one, the tunnelled one (ticket 071), or the
    /// connected one that signals the cell reaches a building — see
    /// [`super::road::RoadPieceVariant`]. [`super::road_catalogue::RoadCatalogue`]'s
    /// second key alongside the kind.
    ///
    /// Stored, like [`Self::base_y`] and [`Self::ascent`], rather than
    /// re-derived on every read. For [`super::road::RoadPieceVariant::Tunnel`]
    /// that isn't a subtle drift but a straight contradiction: a tunnel piece
    /// **carves away the very cover that made it a tunnel**. Re-sample a
    /// written tunnel cell and the 36 blocks over it are the air the piece
    /// just cut, so it reads as `Surface`, and the next re-tile (a neighbour
    /// growing a connection — `super::road_build::affected_cells`) writes the
    /// open-sky piece back into the hillside, filling the bore in around the
    /// player.
    ///
    /// [`super::road::RoadPieceVariant::Connected`] has no such
    /// self-destroying check — `super::road::touches_building` is safe to
    /// call again any time — so it is the one variant that *is* revisited
    /// after the cell exists: ticket 110's
    /// `super::road_build::retile_beside_buildings` re-reads it for exactly
    /// the cells a building's footprint just appeared or disappeared next
    /// to, through [`City::set_road_cell_variant`], and never for a cell
    /// already recorded `Tunnel`. Still stored rather than derived on every
    /// read, so the write path and the preview keep reading one recorded
    /// answer rather than each taking their own.
    pub variant: super::road::RoadPieceVariant,
    /// Ticket 128: `true` while this cell is still a **site** — claimed in
    /// [`City`] (so occupancy, `connections_at`/`select_piece` and a drag's
    /// own re-crossing all see it as road already) but not yet cleared and
    /// written. `false` once its piece is actually in the world, and for
    /// every cell that predates this ticket. `warehouse` coverage is the one
    /// reader that treats a `true` cell as *not* road — see that module's
    /// docs. Only [`City::set_road_cell_under_construction`] ever writes it.
    pub under_construction: bool,
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

/// Where a single unrotated local `(x, z)` tile inside a `footprint`-sized
/// footprint lands once that footprint is rotated — the same per-cell corner
/// math [`crate::blueprint::rotate_blueprint`] applies to a block's grid
/// position (`rotate90_pos`), just dropped to two dimensions and driven by
/// [`footprint_extent`]'s axis swap instead of a blueprint's own `size`.
///
/// `tile` is expected to be inside `0..footprint.x` / `0..footprint.y`; nothing
/// here checks that, the same way `rotate_blueprint` trusts its grid indices.
/// Ticket 114 (`city::mine::layout::MineFrame::from_placement`) is the first
/// caller — rotating a mine's `shaft` corner into world space — and rotates
/// *both* corners of the shaft square through this and takes the
/// componentwise min, since a single corner's image under a 90°/270° turn is
/// the square's opposite corner, not its new minimum.
///
/// No non-test caller outside `city::mine::layout` yet — that module itself
/// has no caller until ticket 115/116 wire it in, the gap ticket 113's
/// `Mine::is_valuable` sat in.
#[allow(dead_code)]
pub fn rotate_local_tile(tile: IVec2, footprint: IVec2, rotation: Rotation) -> IVec2 {
    match rotation {
        Rotation::Deg0 => tile,
        Rotation::Deg90 => IVec2::new(footprint.y - 1 - tile.y, tile.x),
        Rotation::Deg180 => IVec2::new(footprint.x - 1 - tile.x, footprint.y - 1 - tile.y),
        Rotation::Deg270 => IVec2::new(tile.y, footprint.x - 1 - tile.x),
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

/// The road cell a Minecraft `(x, z)` block tile falls inside — the inverse
/// of [`road_cell_tiles`]'s `cell * ROAD_CELL_SIZE` corner math. Uses
/// `div_euclid`, not plain `/`, so it stays correct for negative coordinates
/// — `/` truncates toward zero, which would put tile `(-1, -1)` in cell
/// `(0, 0)` instead of the cell that actually covers it, `(-1, -1)`. Shared
/// by [`super::road_build::cell_of`] (this exact math specialised to a 3D
/// block coordinate) and [`super::road`]'s connectivity queries (ticket 056,
/// roadmap F4), which need it from a building's `(x, z)` footprint tiles.
pub fn cell_of(tile: IVec2) -> IVec2 {
    IVec2::new(tile.x.div_euclid(ROAD_CELL_SIZE), tile.y.div_euclid(ROAD_CELL_SIZE))
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
    /// Cell coordinates (ticket 054) — not block tiles, keyed to the
    /// [`RoadCell`] record holding the style (ticket 059) the cell was built
    /// as and the world Y it was built at (ticket 065). `occupancy` still
    /// holds a block-tile [`Occupant::Road`] for every one of a cell's 36
    /// tiles, kept in lockstep by [`add_road_cell`](Self::add_road_cell)/
    /// [`remove_road_cell`](Self::remove_road_cell); this map is the source
    /// of truth for "is this cell a road, and which style," the way
    /// `buildings` is for buildings.
    road_cells: HashMap<IVec2, RoadCell>,
    occupancy: HashMap<IVec2, Occupant>,
}

impl City {
    /// Places a building of geometry `catalogue_id` — and, if the placement
    /// came from anything that knows one, game data `definition_id` (ticket
    /// 076; see [`PlacedBuilding::definition_id`] for why a placement can
    /// legitimately have none) — at `origin`/`rotation`, covering
    /// `footprint`'s rotated extent. All-or-nothing: every tile is checked
    /// before any of them is marked occupied, so a refusal never leaves a
    /// partial building behind — see the module docs.
    ///
    /// Called synchronously by `city::commit::try_commit_placement` (ticket
    /// 048, roadmap E4) the instant a click is accepted — before the write
    /// even starts, so the tile claim exists first. See that module's docs
    /// for why the write is transactional against this call:
    /// [`remove_building`](Self::remove_building) is the rollback.
    pub fn place_building(
        &mut self,
        catalogue_id: impl Into<String>,
        definition_id: Option<String>,
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
            PlacedBuilding {
                catalogue_id: catalogue_id.into(),
                definition_id,
                origin,
                rotation,
                footprint,
                work_area: None,
                under_construction: false,
            },
        );
        Ok(id)
    }

    /// Marks `id` a **site** (ticket 128) — its blueprint hasn't been
    /// written yet, `city::construction` is clearing the ground under it
    /// first. `false` if `id` isn't a placed building. Called once, by
    /// `city::commit::try_commit_placement`, right after
    /// [`place_building`](Self::place_building) claims the tile — see
    /// [`PlacedBuilding::under_construction`].
    pub fn mark_under_construction(&mut self, id: BuildingId) -> bool {
        match self.buildings.get_mut(&id) {
            Some(building) => {
                building.under_construction = true;
                true
            }
            None => false,
        }
    }

    /// The inverse of [`mark_under_construction`](Self::mark_under_construction):
    /// a site's clearing finished and its blueprint has actually been
    /// written. `false` if `id` isn't a placed building. Called by
    /// `city::commit::poll_commit`'s site arm once that write succeeds.
    pub fn complete_building(&mut self, id: BuildingId) -> bool {
        match self.buildings.get_mut(&id) {
            Some(building) => {
                building.under_construction = false;
                true
            }
            None => false,
        }
    }

    /// Sets (or, with `None`, clears) `id`'s [`PlacedBuilding::work_area`]
    /// (ticket 111). `false` if `id` isn't a placed building. Not
    /// journalled — drawing an area is a setting, like a road cell's
    /// style, not a build; see [`PlacedBuilding::work_area`].
    pub fn set_work_area(&mut self, id: BuildingId, area: Option<WorkArea>) -> bool {
        match self.buildings.get_mut(&id) {
            Some(building) => {
                building.work_area = area;
                true
            }
            None => false,
        }
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

    /// Which [`super::definition::BuildingDefinitions`] entry `id` was placed
    /// from, if it was placed from one at all — the lookup ticket 076 exists
    /// to make possible and every per-instance mechanic after it goes
    /// through, so nothing else has to reach into
    /// [`PlacedBuilding::definition_id`] and decide for itself what a `None`
    /// means.
    ///
    /// Two distinct `None`s collapse here on purpose: `id` isn't a placed
    /// building, and `id` is a placed building with no definition behind it.
    /// Every caller so far treats them the same — no definition, no game
    /// data, no production — and [`building`](Self::building) is right there
    /// for anything that needs to tell them apart.
    ///
    /// `city::ui::inspect_panel` (ticket 083, roadmap G3) is the real
    /// caller: the selected building's own name, off its definition.
    pub fn definition_of(&self, id: BuildingId) -> Option<&str> {
        self.buildings.get(&id)?.definition_id.as_deref()
    }

    /// Every **completed** placement — a site (ticket 128,
    /// [`PlacedBuilding::under_construction`]) is excluded, because it isn't
    /// a building yet: no production/gatherer/mine tick runs for it, it adds
    /// no storage or coverage, and it unlocks nothing in the build menu. See
    /// [`placements`](Self::placements) for the unfiltered iterator, which
    /// the handful of callers that *do* need to see a site (persistence, the
    /// construction tick, the site marker, the journal's row lookup) use
    /// instead.
    pub fn buildings(&self) -> impl Iterator<Item = (BuildingId, &PlacedBuilding)> {
        self.buildings.iter().filter(|(_, b)| !b.under_construction).map(|(&id, b)| (id, b))
    }

    /// Every placement, completed or still a site — see
    /// [`buildings`](Self::buildings)'s own docs for the distinction and why
    /// most callers want that one instead.
    pub fn placements(&self) -> impl Iterator<Item = (BuildingId, &PlacedBuilding)> {
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
    /// as a road built in `style` (ticket 059 —
    /// `super::road_catalogue::RoadCatalogue`'s own key), occupying all 36
    /// block tiles [`road_cell_tiles`] lists for it. Idempotent if `cell` is
    /// already a road — `style`, `base_y`, `ascent` and `variant` are all
    /// ignored in
    /// that case and the cell's *existing* record is kept; repainting a road
    /// cell to a different style is out of scope here (a demolish-and-rebuild
    /// away, for now), keeping the original `base_y` is what stops a re-tiled
    /// cell from drifting upward (ticket 065 — see [`RoadCell`]), keeping
    /// the original `ascent` is what stops a drag that merely *crosses* an
    /// existing stair from flattening it (ticket 067), and keeping the
    /// original `variant` is what stops a re-tiled tunnel from being filled
    /// back in with the hillside it was cut out of (ticket 071). Refused,
    /// all-or-nothing, if *any* of the 36 tiles a *new* cell would claim are
    /// held by a building or another road cell — the same "plan every tile
    /// before marking any of them" shape [`place_building`](Self::place_building)
    /// uses, so a refused road cell never leaves a partial one behind.
    pub fn add_road_cell(
        &mut self,
        cell: IVec2,
        style: impl Into<String>,
        base_y: i32,
        ascent: Option<super::road::Direction>,
        variant: super::road::RoadPieceVariant,
    ) -> Result<(), PlacementError> {
        if self.road_cells.contains_key(&cell) {
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
        self.road_cells.insert(cell, RoadCell { style: style.into(), base_y, ascent, variant, under_construction: false });
        Ok(())
    }

    /// Sets `cell`'s [`RoadCell::under_construction`] flag (ticket 128).
    /// `false` if `cell` isn't a road cell — the same "no such cell" report
    /// [`set_road_cell_variant`](Self::set_road_cell_variant) gives with
    /// `None`, spelled as a `bool` here since no caller needs the old value
    /// back.
    pub fn set_road_cell_under_construction(&mut self, cell: IVec2, value: bool) -> bool {
        match self.road_cells.get_mut(&cell) {
            Some(road) => {
                road.under_construction = value;
                true
            }
            None => false,
        }
    }

    /// Repaints an existing road cell's [`RoadCell::variant`] in place
    /// (ticket 110), returning what it was before so the caller can put it
    /// back if the world write that follows fails. `None` — and nothing
    /// changed — if `cell` isn't a road cell.
    ///
    /// The one field of a [`RoadCell`] that is allowed to change after the
    /// cell exists: [`add_road_cell`](Self::add_road_cell)'s idempotence
    /// deliberately protects `base_y`/`ascent`/`variant` from a drag that
    /// merely re-crosses the cell, so a *reactive* re-tile — a building
    /// appearing or disappearing beside a road that was already there —
    /// needs its own way in. `style`, `base_y` and `ascent` stay
    /// untouchable: the reasons [`RoadCell`] gives for freezing them (drift,
    /// a stair flipping under the player) still hold; only the connected
    /// marker is a reading that stays true to re-take.
    pub fn set_road_cell_variant(&mut self, cell: IVec2, variant: super::road::RoadPieceVariant) -> Option<super::road::RoadPieceVariant> {
        let road = self.road_cells.get_mut(&cell)?;
        Some(std::mem::replace(&mut road.variant, variant))
    }

    /// Clears a road cell and frees all 36 of its block tiles. Returns
    /// whether `cell` was actually a road cell beforehand.
    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn remove_road_cell(&mut self, cell: IVec2) -> bool {
        if self.road_cells.remove(&cell).is_some() {
            for tile in road_cell_tiles(cell) {
                self.occupancy.remove(&tile);
            }
            true
        } else {
            false
        }
    }

    /// Cell coordinates only — every currently-placed road cell, regardless
    /// of style. [`road_cells_with_data`](Self::road_cells_with_data) is
    /// the counterpart that also names each cell's style; most callers (a
    /// count for the city panel, `city::road`'s connectivity queries) only
    /// ever wanted "is this a road cell," so this stays the cheaper, more
    /// common shape rather than making every caller unpack a tuple it
    /// doesn't need.
    pub fn road_cells(&self) -> impl Iterator<Item = &IVec2> {
        self.road_cells.keys()
    }

    /// Every currently-placed road cell alongside its [`RoadCell`] record —
    /// [`super::persistence::save_city`]'s own iteration order.
    pub fn road_cells_with_data(&self) -> impl Iterator<Item = (&IVec2, &RoadCell)> {
        self.road_cells.iter()
    }

    /// `cell`'s full record, if it's a road cell at all — what
    /// [`super::road_build::road_write_edit`] reads per affected cell once a
    /// commit already recorded it here, since it needs the style *and* the
    /// height together.
    pub fn road_cell_at(&self, cell: IVec2) -> Option<&RoadCell> {
        self.road_cells.get(&cell)
    }

    /// The style `cell` was built as, if it's a road cell at all — `None`
    /// for a cell that isn't road, never an empty string.
    pub fn road_style_at(&self, cell: IVec2) -> Option<&str> {
        self.road_cells.get(&cell).map(|road| road.style.as_str())
    }

    /// Whether `cell` is a road cell — [`city::road`](super::road)'s own
    /// adjacency queries go through this rather than re-deriving it from
    /// [`occupant_at`](Self::occupant_at) at one of the cell's 36 tiles.
    pub fn is_road_cell(&self, cell: IVec2) -> bool {
        self.road_cells.contains_key(&cell)
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
    use std::collections::HashSet;

    use super::super::road::RoadPieceVariant;
    use super::*;

    #[test]
    fn placing_a_building_occupies_exactly_its_footprint() {
        let mut city = City::default();
        let origin = IVec3::new(10, 64, 20);
        let footprint = IVec2::new(3, 2);
        let id = city
            .place_building("house01", None, origin, Rotation::Deg0, footprint)
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
            .place_building("house01", None, origin, Rotation::Deg90, footprint)
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
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(4, 4))
            .unwrap();

        // Overlaps the first building's last column/row (x=3, z=3..7).
        let err = city
            .place_building("house01", None, IVec3::new(3, 64, 3), Rotation::Deg0, IVec2::new(4, 4))
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
        city.add_road_cell(IVec2::new(1, 2), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

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
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 2))
            .unwrap();

        // Cell (0, 0) covers block tiles 0..6 x 0..6, which overlaps the
        // building's 2x2 footprint at its very first checked tile.
        let err = city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap_err();
        assert!(matches!(
            err,
            PlacementError::TileOccupied { tile, by: Occupant::Building(id) }
                if tile == IVec2::new(0, 0) && id == building_id
        ));

        city.add_road_cell(IVec2::new(5, 5), "dirt", 64, None, RoadPieceVariant::Surface).expect("an empty cell should accept a road");
        // Cell (5, 5) covers block tiles 30..36 x 30..36.
        let err = city
            .place_building("house01", None, IVec3::new(30, 64, 30), Rotation::Deg0, IVec2::new(1, 1))
            .unwrap_err();
        assert!(matches!(
            err,
            PlacementError::TileOccupied { tile, by: Occupant::Road } if tile == IVec2::new(30, 30)
        ));
    }

    #[test]
    fn adding_the_same_road_cell_twice_is_a_no_op() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(1, 1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(1, 1), "dirt", 64, None, RoadPieceVariant::Surface).expect("re-adding the same road cell should succeed");
        assert_eq!(city.road_cells().count(), 1);
    }

    // -- road cell style (ticket 059) ----------------------------------------

    #[test]
    fn road_style_at_reports_the_style_a_cell_was_built_as() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "paved", 64, None, RoadPieceVariant::Surface).unwrap();
        assert_eq!(city.road_style_at(IVec2::new(0, 0)), Some("paved"));
    }

    #[test]
    fn road_style_at_is_none_for_a_tile_that_is_not_a_road_cell() {
        let city = City::default();
        assert_eq!(city.road_style_at(IVec2::new(0, 0)), None);
    }

    /// Re-adding an already-road cell is a no-op (per its own doc comment)
    /// — a second `add_road_cell` call with a *different* style must not
    /// silently repaint it. Changing a cell's style is out of scope for
    /// ticket 059; see `add_road_cell`'s docs.
    #[test]
    fn re_adding_an_existing_road_cell_keeps_its_original_style() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 0), "paved", 64, None, RoadPieceVariant::Surface).unwrap();
        assert_eq!(city.road_style_at(IVec2::new(0, 0)), Some("dirt"));
    }

    /// Ticket 065's own reason the height lives here rather than being
    /// resampled: a cell re-crossed by a later drag (which resolves its
    /// height against terrain that now contains the *road*, one block higher)
    /// must keep the ground it was originally fitted to, or every re-tile
    /// walks it upward.
    #[test]
    fn re_adding_an_existing_road_cell_keeps_its_original_base_y() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 65, None, RoadPieceVariant::Surface).unwrap();
        assert_eq!(city.road_cell_at(IVec2::new(0, 0)).map(|road| road.base_y), Some(64));
    }

    #[test]
    fn road_cell_at_reports_the_style_and_height_a_cell_was_built_with() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(3, -2), "paved", 71, None, RoadPieceVariant::Surface).unwrap();
        assert_eq!(
            city.road_cell_at(IVec2::new(3, -2)),
            Some(&RoadCell { style: "paved".to_string(), base_y: 71, ascent: None, variant: RoadPieceVariant::Surface, under_construction: false })
        );
        assert_eq!(city.road_cell_at(IVec2::new(0, 0)), None);
    }

    #[test]
    fn road_cells_with_data_iterates_every_cell_with_its_style_and_height() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(5, 5), "paved", 71, None, RoadPieceVariant::Surface).unwrap();

        let found: HashSet<(IVec2, String, i32)> =
            city.road_cells_with_data().map(|(&cell, road)| (cell, road.style.clone(), road.base_y)).collect();
        assert_eq!(
            found,
            HashSet::from([
                (IVec2::new(0, 0), "dirt".to_string(), 64),
                (IVec2::new(5, 5), "paved".to_string(), 71)
            ])
        );
    }

    #[test]
    fn removing_a_building_frees_its_tiles_for_reuse() {
        let mut city = City::default();
        let origin = IVec3::new(0, 64, 0);
        let footprint = IVec2::new(2, 2);
        let id = city.place_building("house01", None, origin, Rotation::Deg0, footprint).unwrap();

        let removed = city.remove_building(id).expect("the id was just placed");
        assert_eq!(removed.catalogue_id, "house01");
        assert!(city.is_tile_free(IVec2::new(0, 0)));
        assert!(city.is_empty());

        // The freed tiles accept a new placement.
        city.place_building("house01", None, origin, Rotation::Deg0, footprint)
            .expect("tiles freed by removal should be placeable again");
    }

    #[test]
    fn removing_an_unknown_id_is_none_not_a_panic() {
        let mut city = City::default();
        let id = city.place_building("house01", None, IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap();
        city.remove_building(id);
        assert!(city.remove_building(id).is_none(), "already removed");
    }

    #[test]
    fn remove_road_cell_reports_whether_a_cell_was_actually_a_road() {
        let mut city = City::default();
        assert!(!city.remove_road_cell(IVec2::new(0, 0)), "never added");
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        assert!(city.remove_road_cell(IVec2::new(0, 0)));
        assert!(city.is_tile_free(IVec2::new(0, 0)));
        assert!(!city.is_road_cell(IVec2::new(0, 0)));

        // Removing frees every one of the cell's 36 tiles, not just its
        // corner.
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
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
    fn rotate_local_tile_walks_a_corner_around_the_footprint() {
        // A 3x5 footprint's four corners, walked clockwise under each
        // rotation — the same check `footprint_extent_swaps_axes_only_on_a_quarter_turn`
        // makes for the bounding box, one level down at a single tile.
        let footprint = IVec2::new(3, 5);
        let nw = IVec2::new(0, 0);
        assert_eq!(rotate_local_tile(nw, footprint, Rotation::Deg0), IVec2::new(0, 0));
        // NW rotated 90 degrees clockwise lands at the new footprint's NE
        // corner: (extent.x - 1, 0) = (4, 0).
        assert_eq!(rotate_local_tile(nw, footprint, Rotation::Deg90), IVec2::new(4, 0));
        // 180 degrees: the opposite corner of the original footprint.
        assert_eq!(rotate_local_tile(nw, footprint, Rotation::Deg180), IVec2::new(2, 4));
        // 270 degrees: the new footprint's SW corner.
        assert_eq!(rotate_local_tile(nw, footprint, Rotation::Deg270), IVec2::new(0, 2));
    }

    #[test]
    fn cell_of_maps_a_tile_to_the_cell_covering_it_including_negative_coordinates() {
        assert_eq!(cell_of(IVec2::new(0, 0)), IVec2::new(0, 0));
        assert_eq!(cell_of(IVec2::new(5, 5)), IVec2::new(0, 0));
        assert_eq!(cell_of(IVec2::new(6, 5)), IVec2::new(1, 0));
        // Plain truncating division would put -1 in cell 0, not cell -1.
        assert_eq!(cell_of(IVec2::new(-1, -1)), IVec2::new(-1, -1));
        assert_eq!(cell_of(IVec2::new(-6, -6)), IVec2::new(-1, -1));
        assert_eq!(cell_of(IVec2::new(-7, -6)), IVec2::new(-2, -1));
    }

    #[test]
    fn buildings_and_roads_iterate_what_was_added() {
        let mut city = City::default();
        let a = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        let b = city.place_building("house01", None, IVec3::new(5, 64, 5), Rotation::Deg0, IVec2::ONE).unwrap();
        city.add_road_cell(IVec2::new(2, 2), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(2, 3), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let ids: HashSet<BuildingId> = city.buildings().map(|(id, _)| id).collect();
        assert_eq!(ids, HashSet::from([a, b]));
        assert_eq!(city.road_cells().count(), 2);
        assert_eq!(city.len(), 2);
    }

    // --- work areas (ticket 111) ---------------------------------------------------------------

    #[test]
    fn a_placed_building_starts_with_no_work_area_and_set_work_area_sets_it() {
        let mut city = City::default();
        let id = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        assert_eq!(city.building(id).unwrap().work_area, None);

        let area = WorkArea::new(IVec2::new(3, -2), IVec2::new(-1, 4));
        assert!(city.set_work_area(id, Some(area)));
        assert_eq!(city.building(id).unwrap().work_area, Some(area));
        assert!(city.set_work_area(id, None));
        assert_eq!(city.building(id).unwrap().work_area, None);
    }

    #[test]
    fn set_work_area_on_a_missing_building_is_false() {
        let mut city = City::default();
        assert!(!city.set_work_area(BuildingId(7), Some(WorkArea::new(IVec2::ZERO, IVec2::ONE))));
    }

    #[test]
    fn work_area_new_orders_the_corners_and_counts_tiles() {
        let area = WorkArea::new(IVec2::new(3, -2), IVec2::new(-1, 4));
        assert_eq!(area.min, IVec2::new(-1, -2));
        assert_eq!(area.max, IVec2::new(3, 4));
        assert_eq!(area.len(), 5 * 7);
        assert_eq!(area.tiles().count(), 35);
        assert!(area.tiles().any(|t| t == IVec2::new(3, 4)), "max corner is inclusive");
        assert!(!area.tiles().any(|t| t == IVec2::new(4, 4)));
        assert_eq!(WorkArea::new(IVec2::ONE, IVec2::ONE).len(), 1, "a click is one tile");
    }

    #[test]
    fn clamp_to_reach_keeps_a_rectangle_already_inside() {
        // 2x2 footprint at (0,0), radius 3: reach covers -3..=4 on both axes.
        let area = WorkArea::new(IVec2::new(-3, -3), IVec2::new(4, 4));
        assert_eq!(area.clamp_to_reach(IVec2::ZERO, IVec2::splat(2), 3), Some(area));
    }

    #[test]
    fn clamp_to_reach_cuts_the_part_beyond_the_radius() {
        let area = WorkArea::new(IVec2::new(-10, 1), IVec2::new(10, 20));
        let clamped = area.clamp_to_reach(IVec2::ZERO, IVec2::splat(2), 3).unwrap();
        assert_eq!(clamped, WorkArea::new(IVec2::new(-3, 1), IVec2::new(4, 4)));
    }

    #[test]
    fn clamp_to_reach_refuses_a_rectangle_entirely_out_of_reach() {
        let area = WorkArea::new(IVec2::new(5, 0), IVec2::new(9, 0));
        assert_eq!(area.clamp_to_reach(IVec2::ZERO, IVec2::splat(2), 3), None);
    }

    // --- construction sites (ticket 128) -------------------------------------

    #[test]
    fn buildings_skips_a_site_but_placements_includes_it() {
        let mut city = City::default();
        let a = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        let b = city.place_building("house01", None, IVec3::new(5, 64, 5), Rotation::Deg0, IVec2::ONE).unwrap();
        assert!(city.mark_under_construction(b));

        let completed: HashSet<BuildingId> = city.buildings().map(|(id, _)| id).collect();
        assert_eq!(completed, HashSet::from([a]), "a site under construction isn't a building yet");

        let all: HashSet<BuildingId> = city.placements().map(|(id, _)| id).collect();
        assert_eq!(all, HashSet::from([a, b]), "placements() sees every row, sites included");

        assert!(city.complete_building(b));
        let completed: HashSet<BuildingId> = city.buildings().map(|(id, _)| id).collect();
        assert_eq!(completed, HashSet::from([a, b]), "completing the site makes it a building again");
    }

    #[test]
    fn mark_and_complete_are_false_for_an_unknown_building() {
        let mut city = City::default();
        assert!(!city.mark_under_construction(BuildingId(7)));
        assert!(!city.complete_building(BuildingId(7)));
    }

    #[test]
    fn set_road_cell_under_construction_reports_whether_the_cell_exists() {
        let mut city = City::default();
        assert!(!city.set_road_cell_under_construction(IVec2::new(0, 0), true));
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        assert!(!city.road_cell_at(IVec2::new(0, 0)).unwrap().under_construction, "a freshly-added cell starts complete");
        assert!(city.set_road_cell_under_construction(IVec2::new(0, 0), true));
        assert!(city.road_cell_at(IVec2::new(0, 0)).unwrap().under_construction);
    }

    #[test]
    fn clamp_to_reach_leaves_the_footprint_itself_inside() {
        let area = WorkArea::new(IVec2::new(0, 0), IVec2::new(1, 1));
        assert_eq!(area.clamp_to_reach(IVec2::ZERO, IVec2::splat(2), 3), Some(area));
    }
}
