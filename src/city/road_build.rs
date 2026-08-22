//! Drag-to-build roads (ticket 055, roadmap F2), wired straight through to
//! F3's auto-tiling: [`super::road::select_piece`] decides which of the six
//! [`super::road_catalogue::RoadCatalogue`] pieces a cell needs, and this
//! module is what actually spawns the preview mesh for it and writes its
//! blocks once a drag commits.
//!
//! ## Active only while the road tool is selected
//!
//! [`super::tool::ActiveTool::Road`] — `T` toggles it (see [`super::tool`]).
//! While it's active, [`super::placement`]'s building ghost hides itself and
//! [`super::commit`]'s left-click handler no-ops, so a click only ever means
//! one thing at a time. See both modules' own docs for their half of that
//! guard.
//!
//! ## The drag: click, hold, release
//!
//! [`RoadDragState::start`] is set on the frame a left click lands on a road
//! cell tool with something hovered, and cleared on release — [`update_drag_preview`]
//! recomputes [`drag_path`] every frame in between off *this* frame's
//! [`super::picking::HoveredBlock`], the same "recompute, don't cache" shape
//! ticket 047's ghost preview and ticket 048's commit both already use for a
//! building. When nothing is being dragged, the "path" is just the one
//! hovered cell — a single click with no drag is a valid way to place one
//! road cell.
//!
//! [`drag_path`] routes an L-shape: every cell from `start` to `end` along
//! `start`'s own row, then every cell from that corner to `end` along `end`'s
//! own column. Not the shortest Chebyshev/diagonal path — an L keeps every
//! cell in the path a *cardinal* neighbour of the next one, which is what
//! [`super::road::connections_at`] and [`super::road::select_piece`] assume
//! throughout; a diagonal "path" would need pieces this catalogue doesn't
//! have.
//!
//! ## Which style: `[`/`]`, auto-picking the first loaded one (ticket 059)
//!
//! [`super::road_catalogue::RoadCatalogue`] can hold more than one road
//! style side by side (`assets/city/roads/<style>/...`), so *placing* a road
//! cell needs to say which one — the same "no build menu yet" gap ticket 047
//! filled for buildings with [`super::placement::PlacementSelection`].
//! [`RoadStyleSelection`] is that stand-in here: `[`/`]` cycle through
//! [`super::road_catalogue::RoadCatalogue::styles`] (sorted, so the order is
//! stable), gated on [`super::tool::ActiveTool::Road`] being active — unlike
//! the building keys, these are new and unused elsewhere, so there's no
//! reason not to keep them scoped to the tool they mean something for. The
//! first loaded style is auto-selected the moment the catalogue has one, so
//! a single-style setup needs no keypress at all.
//!
//! A road cell's style, once placed, lives on [`super::state::City`] itself
//! (`add_road_cell`'s own `style` argument, `City::road_style_at`) — not
//! carried around separately — so [`road_write_edit`] and the preview both
//! read a *placed* cell's style back off `City` rather than needing it
//! threaded through as a parameter. [`RoadStyleSelection::current`] only
//! matters for a cell that's brand new to the current drag and has no
//! recorded style yet.
//!
//! ## The preview: a real piece where one exists, a flat quad otherwise
//!
//! [`road::select_piece`] decides shape against a *hypothetical*
//! connectivity that treats every other cell in the current drag path as
//! road too — so a straight run previews as a run of
//! [`super::road::RoadPieceKind::Straight`] pieces while the drag is still in
//! progress, not six disconnected dead ends. When [`super::road_catalogue::RoadCatalogue`]
//! actually has a piece for the resolved `(style, kind)`, [`preview_mesh`]
//! meshes it the same way ticket 047's `placement::ghost_mesh` meshes a
//! building — rotated, cached by `(style, RoadPieceKind, Rotation)` (a
//! handful of styles times six kinds times four rotations: small, no
//! eviction needed). Ticket 063's `dirt` style is the first to ship real
//! `.nbt` pieces; a style (or a kind) with none still falls back to
//! [`quad_mesh`]: one flat, unrotated plane per cell, tinted the same
//! green/red [`super::placement`] uses.
//!
//! The two are anchored differently — a meshed piece starts at its own
//! minimum corner, the quad is centred on its origin — which is what
//! [`PreviewAnchor`] and [`cell_transform`] exist to keep straight. While
//! every cell fell back to the quad this didn't matter; ticket 065 is where
//! it started to.
//!
//! ## Height: decided once, for the whole placement
//!
//! Ticket 065 established that a cell's height is read *once* and then
//! remembered on [`super::state::RoadCell::base_y`], never re-derived —
//! because once a piece has been written [`super::grid::ground_height_at`]
//! samples the road surface as ground, so a re-tiled neighbour would climb a
//! block per rewrite. [`cell_write_origin`] turns that reading into the
//! piece's actual write origin, dropping it by [`ROAD_PIECE_SUBGRADE_DEPTH`]
//! so the surface course lands flush with the terrain rather than perched
//! above it. (Before 065 the write origin's Y was a hardcoded `0`: every road
//! ever built went into the deepslate, reported as written, and was never
//! seen.)
//!
//! Ticket 067 changed *what* is read once. 065 fitted every cell to its own
//! 6x6 patch of ground, which across a slope produced a run of individually
//! correct pieces separated by one-block cliffs — a road that isn't
//! continuous isn't a road. The unit that owns a height is the **placement**,
//! not the cell:
//!
//! - the drag's **first** cell fixes the starting level;
//! - the **last** cell fixes the ending level, snapped to the first's plus a
//!   whole number of [`ROAD_STAIR_RISE`] steps, because a
//!   [`RoadPieceKind::Stair`] piece is the only thing that bridges a level
//!   change and it bridges exactly four blocks;
//! - the cells **between** run flat, except for the few [`spread_evenly`]
//!   picks to be stairs, each climbing one step;
//! - with no `stairs.nbt` loaded, or with both ends on one level, the whole
//!   drag is flat at the first cell's level — the plain "one Y per road"
//!   rule, as the degenerate case rather than a second mode.
//!
//! [`plan_drag`] is all of that, as one function shared by the preview and
//! the commit so the ghost can't disagree with the blocks — the same thing
//! 065 did for [`cell_write_origin`], one level up. Which levels the ends
//! *land* on comes from [`anchor_level`]: an existing road cell's own
//! recorded height wins over the terrain, so a new drag joins an old road
//! flush instead of at whatever the ground under it happens to be.
//!
//! ## Tunnels: when the terrain closes over the road (ticket 071)
//!
//! Height planning above is about a road running *over* terrain. It says
//! nothing about terrain running over the *road* — drag across the foot of a
//! hill and the profile happily puts the surface course several blocks under
//! the hillside, and the piece's own three layers of clearance mow exactly
//! three layers of it before the player walks into a wall.
//!
//! The rule for spotting that is a single reading, taken at the first Y the
//! cell's piece does not occupy: **if more than half of the 36 columns over
//! the cell are not air, the cell is a tunnel** ([`cover_at`],
//! [`ROAD_TUNNEL_COVER_MAJORITY`]). It resolves to a
//! [`RoadPieceVariant::Tunnel`] piece — `<kind>-tunnel.nbt`, the same kind
//! and the same rotation, a different `.nbt` — so nothing in
//! [`road::select_piece`] or the connection logic changes.
//!
//! [`plan_tunnels`] is that pass, run over [`plan_drag`]'s output rather
//! than inside it, because it needs both the heights that pass resolves and
//! the catalogue that pass deliberately doesn't see. Two things gate it, and
//! the second is the same shape [`stair_available`] gives ramps: a cell is
//! only recorded as a tunnel if the catalogue really holds the piece, so a
//! style with no `-tunnel` exports keeps building exactly what it built
//! before rather than leaving unresolvable cells as holes.
//!
//! And, like the heights, the variant is decided **once** and remembered on
//! [`super::state::RoadCell::variant`]. Here that isn't the subtle drift
//! 065 fixed but a straight contradiction: writing the tunnel piece *carves
//! away the cover that chose it*, so a cell re-measured after its own write
//! reads as open sky and the next re-tile would fill the bore back in with
//! hillside.
//!
//! ## Terrain fit, reusing E2 rather than reinventing it for cells
//!
//! A road cell is exactly a [`super::state::ROAD_CELL_SIZE`]-square footprint
//! at [`crate::blueprint::Rotation::Deg0`] — [`cell_fit`] calls
//! [`super::grid::fit_footprint`] with that footprint directly rather than a
//! second height-sampling walk. Slopes are the acknowledged iteration-1 gap
//! the roadmap's own F3 entry names ("a legitimate iteration-2 deferral if it
//! bites") — this ticket doesn't relax it or work around it, just reuses
//! whatever E2 already decided about a 6x6 patch of ground.
//!
//! ## Committing: all-or-nothing across the whole drag, then re-tile
//!
//! [`try_commit_drag`] refuses the *entire* path if any one cell in it is
//! invalid — the same "plan every tile before marking any of them" shape
//! [`super::state::City::place_building`]/`add_road_cell` already use, lifted
//! one level up to a multi-cell drag. Once every cell in the path is
//! confirmed road (`City::add_road_cell`, synchronously, before the write
//! starts — the same ordering `city::commit` uses and for the same reason: a
//! second drag over the same cells must see them as taken immediately, not
//! once an async task gets around to it), the write batches every **affected**
//! cell: the path itself, plus any already-road neighbour whose own piece
//! might now need to change (a dead end that just grew a neighbour becomes a
//! straight or a corner) — [`affected_cells`]. Each with a catalogue piece
//! contributes its own [`super::commit::blueprint_edit`]; [`merge_edits`]
//! folds them into one [`crate::edit::WorldEdit`] so a multi-cell drag is one
//! routed transaction (W5), not one per cell. A cell with no catalogue piece
//! contributes nothing to the write — it's still recorded in [`super::state::City`]
//! (the city state is authoritative regardless of whether there's an asset to
//! render it with yet), just invisible until a real piece exists.
//!
//! If the write fails, only the cells *this* drag actually added are rolled
//! back via `City::remove_road_cell` — a pre-existing neighbour that merely
//! needed re-tiling was never newly claimed by this drag and must not be
//! un-built by its failure.
//!
//! ## Not journaled
//!
//! Unlike a building (roadmap I1, landed with ticket 048), a road placement
//! writes no [`super::journal::Journal`] entry — there is no undo or demolish
//! for a road cell yet. `City`'s own persistence (ticket 043) still saves
//! `road_cells` regardless; only the *journal* side is out of scope here.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::render::{mesh::Indices, render_asset::RenderAssetUsages};
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::blueprint::{self, Blueprint, Rotation};
use crate::camera;
use crate::chunk_pipeline::{ChunksEdited, SharedAtlasIndex, SharedColorMaps, SharedRegionCache, TerrainMaterial};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;
use crate::world::{self, BiomeColors};
use crate::DecodedWorld;

use super::commit::blueprint_edit;
use super::grid::{self, FootprintFit};
use super::picking::{HoveredBlock, PickingSet};
use super::road::{self, RoadConnections, RoadPieceKind, RoadPieceVariant};
use super::road_catalogue::RoadCatalogue;
use super::state::{self, City, ROAD_CELL_SIZE};
use super::tool::ActiveTool;
use super::write_status::{WriteKind, WriteStatus};

/// Which cell a left click started dragging from, if any — `None` between
/// drags. See the module docs' "The drag: click, hold, release".
#[derive(Resource, Default)]
struct RoadDragState {
    start: Option<IVec2>,
}

/// Which road style (ticket 059) new cells get built as — the road tool's
/// counterpart of `city::placement::PlacementSelection::catalogue_id`. See
/// the module docs' "Which style". `pub(super)` so `city::ui`'s eventual
/// style picker (mirroring the build menu's role for
/// `PlacementSelection`) can read/set it without this whole module needing
/// to be `pub`.
#[derive(Resource, Default)]
pub(super) struct RoadStyleSelection {
    pub(super) current: Option<String>,
}

/// `[`/`]` cycle [`RoadStyleSelection::current`] through
/// [`RoadCatalogue::styles`], and auto-pick the first one the moment the
/// catalogue has any and nothing is selected yet — see the module docs.
/// Gated on [`ActiveTool::Road`] (unlike `placement::cycle_selection`'s
/// number keys, which react regardless of tool): these are new keys with no
/// meaning outside the road tool, so there's nothing lost by scoping them.
fn cycle_road_style(
    keys: Res<ButtonInput<KeyCode>>,
    egui_input: Res<camera::EguiInputCapture>,
    tool: Option<Res<ActiveTool>>,
    catalogue: Option<Res<RoadCatalogue>>,
    mut selection: ResMut<RoadStyleSelection>,
) {
    if egui_input.keyboard || !matches!(tool.as_deref(), Some(ActiveTool::Road)) {
        return;
    }
    let Some(catalogue) = catalogue else { return };
    let styles = catalogue.styles();
    if styles.is_empty() {
        return;
    }

    if selection.current.is_none() {
        selection.current = Some(styles[0].to_string());
    }

    let forward = keys.just_pressed(KeyCode::BracketRight);
    let backward = keys.just_pressed(KeyCode::BracketLeft);
    if !forward && !backward {
        return;
    }

    let current_index = selection.current.as_deref().and_then(|current| styles.iter().position(|&s| s == current)).unwrap_or(0);
    let next_index = if forward { (current_index + 1) % styles.len() } else { (current_index + styles.len() - 1) % styles.len() };
    selection.current = Some(styles[next_index].to_string());
}

/// A committed drag's write, in flight — the road-cell counterpart of
/// `city::commit::PendingCommit`.
struct PendingRoadBuild {
    /// Cells *this* drag actually added (excludes an already-road cell it
    /// merely crossed) — [`City::remove_road_cell`]'s rollback list on a
    /// failed write. See the module docs' "Not journaled".
    newly_added: Vec<IVec2>,
    task: Task<Result<EditReport, EditRefusal>>,
}

/// One road build in flight at a time — the same single-slot backpressure
/// `city::commit::CommitState`/`city::demolish::DemolishState` already use.
#[derive(Resource, Default)]
struct RoadBuildState {
    pending: Option<PendingRoadBuild>,
}

/// The pool of preview entities for the current drag path, plus the caches
/// [`preview_mesh`] fills — the road-tool counterpart of
/// `city::placement::GhostState`. `entities` grows to the longest path drawn
/// so far and never shrinks; extra entities beyond the current path's length
/// are hidden, not despawned, mirroring the ghost's own "update in place, one
/// entity, sixty times a second" reasoning, just for a pool instead of one.
#[derive(Resource, Default)]
struct RoadPreviewState {
    entities: Vec<Entity>,
    quad_mesh: Option<Handle<Mesh>>,
    materials: Option<PreviewMaterials>,
    piece_meshes: HashMap<(String, RoadPieceKind, RoadPieceVariant, Rotation), Option<Handle<Mesh>>>,
}

/// The two translucent preview materials — a separate pair from
/// `placement::GhostMaterials` rather than a shared resource; see
/// `city::ui::city_panel`'s own `WROTE_COLOR` doc comment for why duplicating
/// one small constant across two otherwise-independent modules is the call
/// this crate already makes rather than coupling them for it.
struct PreviewMaterials {
    valid: Handle<StandardMaterial>,
    invalid: Handle<StandardMaterial>,
}

/// Marks a road preview entity, so [`update_drag_preview`] can find and reuse
/// [`RoadPreviewState::entities`] without a second bookkeeping list.
#[derive(Component)]
struct RoadPreview;

pub struct RoadBuildPlugin;

impl Plugin for RoadBuildPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RoadDragState>()
            .init_resource::<RoadBuildState>()
            .init_resource::<RoadPreviewState>()
            .init_resource::<RoadStyleSelection>()
            // Same idempotent-either-order shape `city::commit`/`city::demolish`
            // already document for this resource.
            .init_resource::<WriteStatus>()
            .add_event::<ChunksEdited>()
            // After `PickingSet`, same reason every other per-frame reader of
            // `HoveredBlock` orders there. `cycle_road_style` first, so a
            // `[`/`]` press this frame is reflected in this same frame's
            // preview and commit.
            .add_systems(
                Update,
                (cycle_road_style, update_drag_state, update_drag_preview, try_commit_drag, poll_road_build)
                    .chain()
                    .after(PickingSet),
            );
    }
}

// -----------------------------------------------------------------------------------------------
// ---- pure geometry/logic: cell coordinates, the path, and per-cell validity -------------------
// -----------------------------------------------------------------------------------------------

/// The road cell a Minecraft block coordinate's `(x, z)` falls in —
/// [`state::cell_of`], specialised to a 3D block coordinate (`y` dropped,
/// the grid is horizontal-only, same as everywhere else in this module).
pub(super) fn cell_of(block: IVec3) -> IVec2 {
    state::cell_of(IVec2::new(block.x, block.z))
}

/// An L-shaped path of cells from `start` to `end`, inclusive of both — see
/// the module docs for why an L, not a diagonal. `start == end` is a
/// single-cell path (a plain click, no drag).
pub(super) fn drag_path(start: IVec2, end: IVec2) -> Vec<IVec2> {
    let mut path = Vec::new();

    let step_x = (end.x - start.x).signum();
    let mut x = start.x;
    loop {
        path.push(IVec2::new(x, start.y));
        if x == end.x {
            break;
        }
        x += step_x;
    }

    let step_z = (end.y - start.y).signum();
    let mut z = start.y;
    loop {
        // The corner cell (end.x, start.y) was already pushed by the first
        // leg when `z == start.y`.
        if z != start.y {
            path.push(IVec2::new(end.x, z));
        }
        if z == end.y {
            break;
        }
        z += step_z;
    }

    path
}

/// Whether every one of `cell`'s 36 block tiles is free for a road — already
/// being a road cell counts as free (a drag re-crossing existing road, or a
/// neighbour this drag didn't touch, must not read as blocked).
fn cell_occupancy_ok(cell: IVec2, city: &City) -> bool {
    city.is_road_cell(cell) || state::road_cell_tiles(cell).all(|tile| city.is_tile_free(tile))
}

/// The base Minecraft `(x, z)` corner of `cell` — every terrain/occupancy
/// sample in this module starts here. The `y` is a placeholder `0`, not a
/// height: [`cell_fit`] only ever reads the `(x, z)` out of this, and the
/// one caller that needs a real Y ([`road_write_edit`]) overwrites it with
/// the cell's own recorded [`super::state::RoadCell::base_y`].
fn cell_min_corner(cell: IVec2) -> IVec3 {
    IVec3::new(cell.x * ROAD_CELL_SIZE, 0, cell.y * ROAD_CELL_SIZE)
}

/// How many layers of a road piece sit *below* its surface course — the
/// subgrade a piece carries under the paving a player actually walks on.
///
/// The shipped `dirt` pieces (`assets/city/roads/dirt`, ticket 063) are
/// 6x5x6 and laid out `y=0` solid dirt, `y=1` the surface course
/// (`dirt_path`/`grass_block`/`cobblestone_stairs`), `y=2..4` air. Those air
/// layers are deliberate clearance — they mow whatever grew over the road —
/// which only does its job if `y=1` lands *at* the terrain surface (the
/// topmost ground block, [`super::grid::ground_height_at`]'s answer minus
/// one), leaving the clearance directly above it. Anchoring the piece's
/// bottom at that surface instead would put the paving a block proud of the
/// grass beside it and waste the clearance on empty sky.
///
/// A constant rather than per-style data: [`super::road_definition::RoadType`]
/// (`assets/city/road_types/*.ron`) is *game* data — travel speed, capacity —
/// and isn't threaded into the write path at all. If a style ever ships
/// pieces with a different subgrade depth, this is the thing that becomes a
/// field there.
pub(super) const ROAD_PIECE_SUBGRADE_DEPTH: i32 = 1;

/// Where a cell's piece is written: [`cell_min_corner`]'s `(x, z)`, and a Y
/// that puts the piece's surface course at the terrain surface — see
/// [`ROAD_PIECE_SUBGRADE_DEPTH`]. `base_y` is the cell's recorded
/// [`super::state::RoadCell::base_y`], i.e. `fit_footprint`'s "one above the
/// ground", so the surface itself is `base_y - 1`.
///
/// Shared by [`road_write_edit`] and the drag preview so the ghost stands
/// exactly where the blocks will land — before ticket 065 the two disagreed
/// by the whole height of the world.
fn cell_write_origin(cell: IVec2, base_y: i32) -> IVec3 {
    cell_min_corner(cell).with_y(base_y - 1 - ROAD_PIECE_SUBGRADE_DEPTH)
}

/// [`super::grid::fit_footprint`] against `cell`'s own square footprint at
/// [`Rotation::Deg0`] — see the module docs' "Terrain fit, reusing E2".
fn cell_fit(cell: IVec2, world: &DecodedWorld) -> FootprintFit {
    grid::fit_footprint(cell_min_corner(cell), IVec2::splat(ROAD_CELL_SIZE), Rotation::Deg0, world)
}

/// Whether `cell` could actually become (or continue to be) a road cell —
/// terrain fit *and* occupancy, the same "two answers, ANDed" shape
/// `placement::resolve_placement` already uses for a building.
fn cell_valid(cell: IVec2, world: &DecodedWorld, city: &City) -> bool {
    matches!(cell_fit(cell, world), FootprintFit::Fits { .. }) && cell_occupancy_ok(cell, city)
}

/// A height to preview or place `cell` at: [`cell_fit`]'s own `base_y` when
/// it fits, or a best-effort fallback off [`super::grid::ground_height_at`]
/// so an invalid cell's preview still lands somewhere sensible rather than
/// nothing — the same fallback shape `placement::resolve_placement` uses for
/// a refused building footprint. `None` only when neither answers anything
/// (the cell's chunk isn't decoded at all).
fn cell_height(cell: IVec2, world: &DecodedWorld) -> Option<i32> {
    match cell_fit(cell, world) {
        FootprintFit::Fits { base_y } => Some(base_y),
        FootprintFit::Refused(_) => {
            let corner = cell_min_corner(cell);
            grid::ground_height_at(IVec2::new(corner.x, corner.z), world)
        }
    }
}

// -----------------------------------------------------------------------------------------------
// ---- height: one level per placement, stairs between levels (ticket 067) -----------------------
// -----------------------------------------------------------------------------------------------

/// How many blocks of world Y one [`RoadPieceKind::Stair`] piece climbs
/// across its cell — and therefore the *only* level change a road can make,
/// which is why a drag's end level snaps to a multiple of this.
///
/// A constant for the same reason [`ROAD_PIECE_SUBGRADE_DEPTH`] is one: it
/// describes the shipped geometry (`assets/city/roads/<style>/stairs.nbt`,
/// authored to climb four blocks), and `assets/city/road_types/*.ron` is
/// game data that never reaches the write path. A style shipping a
/// differently-pitched stair is what turns this into a field there.
pub(super) const ROAD_STAIR_RISE: i32 = 4;

/// One cell of a planned drag: where its piece goes and, if it's a stair,
/// which way it climbs. [`plan_drag`]'s output, shared by the preview and
/// the commit so the ghost can't stand somewhere the blocks won't land —
/// the same reason ticket 065 made both go through [`cell_write_origin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CellPlan {
    pub(super) cell: IVec2,
    /// The cell's [`state::RoadCell::base_y`]: its road surface for a flat
    /// cell, its *low* end for a stair.
    pub(super) base_y: i32,
    /// `None` for a flat cell, else the direction this cell climbs — see
    /// [`state::RoadCell::ascent`].
    pub(super) ascent: Option<road::Direction>,
    /// Surface or tunnel (ticket 071) — decided by [`plan_tunnels`] in a
    /// second pass over [`plan_drag`]'s output, because it depends on the
    /// heights that pass resolves *and* on the catalogue, which `plan_drag`
    /// deliberately doesn't see.
    pub(super) variant: RoadPieceVariant,
}

/// Why a drag has no contiguous height profile, and so can't be built at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PlanRefusal {
    /// No ground reading for this cell — its chunk isn't decoded, the same
    /// state [`cell_height`] answers `None` for.
    NoGround(IVec2),
    /// The drag's two ends are `steps` stair-pieces apart in Y, and the path
    /// between them has only `eligible` cells that could *be* a stair (see
    /// [`stair_eligible`]). Refused whole rather than built with a cliff in
    /// it, the same all-or-nothing call [`try_commit_drag`] makes about
    /// validity.
    NotEnoughRoomToClimb { steps: i32, eligible: usize },
}

impl std::fmt::Display for PlanRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanRefusal::NoGround(cell) => write!(f, "no ground height for cell {cell}"),
            PlanRefusal::NotEnoughRoomToClimb { steps, eligible } => write!(
                f,
                "the ends are {} step(s) apart in height but only {eligible} cell(s) on the path can be stairs \
                 — drag further, or level the ground first",
                steps.abs()
            ),
        }
    }
}

/// The road-surface level `road` presents on the edge facing `edge`.
///
/// A flat cell is the same height all the way round. A stair is
/// [`state::RoadCell::base_y`] on its low side and `base_y + ROAD_STAIR_RISE`
/// on the side it climbs toward — so a road meeting a stair has to know
/// *which* side it is meeting it on, and this is that answer.
fn level_on_edge(road: &state::RoadCell, edge: road::Direction) -> i32 {
    if road.ascent == Some(edge) {
        road.base_y + ROAD_STAIR_RISE
    } else {
        road.base_y
    }
}

/// The level one end of a drag should be pinned at, preferring what the city
/// already knows over what the terrain says — see the ticket's "Joining an
/// existing road":
///
/// 1. `cell`'s own record, if it is already a road cell, read on the `edge`
///    the path runs through it;
/// 2. otherwise an adjacent existing road cell's record, read on *its* edge
///    facing `cell`, so a new road starting beside an old one meets it flush
///    rather than at whatever the ground under it happens to be;
/// 3. otherwise [`cell_height`]'s terrain reading.
///
/// `edge` is `None` for a one-cell drag, which has no direction to run in.
fn anchor_level(cell: IVec2, edge: Option<road::Direction>, world: &DecodedWorld, city: &City) -> Option<i32> {
    if let Some(road) = city.road_cell_at(cell) {
        return Some(match edge {
            Some(edge) => level_on_edge(road, edge),
            None => road.base_y,
        });
    }
    for direction in road::Direction::ALL {
        if let Some(neighbour) = city.road_cell_at(cell + direction.offset()) {
            return Some(level_on_edge(neighbour, direction.opposite()));
        }
    }
    cell_height(cell, world)
}

/// The cardinal step from `from` to `to`, which every consecutive pair in a
/// [`drag_path`] is by construction (that's the whole reason the path is an
/// L and not a diagonal). `None` for any other pair.
fn step_direction(from: IVec2, to: IVec2) -> Option<road::Direction> {
    road::Direction::ALL.into_iter().find(|d| from + d.offset() == to)
}

/// Whether the cell at `index` could be a stair in this drag.
///
/// Four things have to hold, and each rules out a way a ramp would come out
/// broken rather than merely ugly:
///
/// - it is **not an end** of the drag — those two are the fixed levels the
///   climb runs between;
/// - the path runs **straight through** it (the previous and next cells are
///   on opposite sides), so the L's corner cell is never a ramp;
/// - **nothing else connects** to it — a stair is shaped like a straight, so
///   a cell that a third road branches into needs a T or a cross, and can't
///   be both;
/// - it is **not already a road cell**, whose recorded height and shape
///   `City::add_road_cell` keeps (deliberately — ticket 065) and this drag
///   therefore cannot change.
fn stair_eligible(path: &[IVec2], index: usize, city: &City) -> bool {
    if index == 0 || index + 1 >= path.len() {
        return false;
    }
    let cell = path[index];
    if city.is_road_cell(cell) {
        return false;
    }
    let (Some(into), Some(out_of)) = (step_direction(path[index - 1], cell), step_direction(cell, path[index + 1]))
    else {
        return false;
    };
    if into != out_of {
        return false; // the L's corner
    }
    connections_with_path(city, path, cell).count() == 2
}

/// `count` indices picked out of `eligible`, spread as evenly along it as
/// integer arithmetic allows: the midpoint of each of `count` equal buckets.
/// Distinct by construction whenever `count <= eligible.len()`, which
/// [`plan_drag`] checks before calling.
fn spread_evenly(eligible: &[usize], count: usize) -> Vec<usize> {
    (0..count).map(|i| eligible[(2 * i + 1) * eligible.len() / (2 * count)]).collect()
}

/// The height profile for a whole drag — the rule ticket 067 exists for.
///
/// The first cell's level is fixed by [`anchor_level`]; the last cell's is
/// too, then **snapped** to the first's plus a whole number of
/// [`ROAD_STAIR_RISE`] steps, because a stair piece is the only thing that
/// can bridge a level change and it bridges exactly that much. Every cell in
/// between runs flat at the level it is on, except for the `steps` of them
/// [`spread_evenly`] picks out of the [`stair_eligible`] ones, each of which
/// climbs one step in the direction the path is travelling (or, on a
/// descending drag, back the way it came — a stair's `ascent` always points
/// uphill).
///
/// With no stair piece loaded for the style, or with the two ends on the
/// same level, the whole path comes back flat at the first cell's level.
/// That is the user's original "Y of the first tile dictates Y of the entire
/// road" rule, and it's the degenerate case of this one rather than a
/// separate mode.
///
/// Cells that are *already* road are still planned (a plan covers the whole
/// path), but `City::add_road_cell` will keep their own recorded height and
/// shape — see [`stair_eligible`]'s last bullet.
fn plan_drag(
    path: &[IVec2],
    world: &DecodedWorld,
    city: &City,
    stair_available: bool,
) -> Result<Vec<CellPlan>, PlanRefusal> {
    let Some(&first) = path.first() else { return Ok(Vec::new()) };
    let leaving = path.get(1).and_then(|&next| step_direction(first, next));
    let start_y = anchor_level(first, leaving, world, city).ok_or(PlanRefusal::NoGround(first))?;

    // Every cell starts `Surface`; `plan_tunnels` is what turns any of them
    // over, once these heights exist for it to measure the cover above.
    let flat = |level: i32| {
        path.iter()
            .map(|&cell| CellPlan { cell, base_y: level, ascent: None, variant: RoadPieceVariant::Surface })
            .collect::<Vec<_>>()
    };
    if path.len() < 2 || !stair_available {
        return Ok(flat(start_y));
    }

    let last = path[path.len() - 1];
    let arriving = step_direction(path[path.len() - 2], last);
    // The edge of the last cell the path arrives *through* is the one facing
    // back the way it came.
    let end_y =
        anchor_level(last, arriving.map(road::Direction::opposite), world, city).ok_or(PlanRefusal::NoGround(last))?;

    // Round to the nearest whole stair, halves away from zero, without
    // floating point: a 6-block difference climbs two steps rather than one.
    let difference = end_y - start_y;
    let steps = (difference * 2 + difference.signum() * ROAD_STAIR_RISE) / (ROAD_STAIR_RISE * 2);
    if steps == 0 {
        return Ok(flat(start_y));
    }

    let eligible: Vec<usize> = (0..path.len()).filter(|&i| stair_eligible(path, i, city)).collect();
    let needed = steps.unsigned_abs() as usize;
    if needed > eligible.len() {
        return Err(PlanRefusal::NotEnoughRoomToClimb { steps, eligible: eligible.len() });
    }
    let stairs = spread_evenly(&eligible, needed);

    let mut plan = Vec::with_capacity(path.len());
    let mut level = start_y;
    for (index, &cell) in path.iter().enumerate() {
        if !stairs.contains(&index) {
            plan.push(CellPlan { cell, base_y: level, ascent: None, variant: RoadPieceVariant::Surface });
            continue;
        }
        // `stair_eligible` already established this cell has a next one and
        // that the path runs straight through it.
        let travel = step_direction(cell, path[index + 1]).expect("a stair-eligible cell has a cardinal successor");
        if steps > 0 {
            // Climbing: this cell's low end meets the flat run behind it.
            plan.push(CellPlan { cell, base_y: level, ascent: Some(travel), variant: RoadPieceVariant::Surface });
            level += ROAD_STAIR_RISE;
        } else {
            // Descending: the *high* end meets the run behind it, so the
            // recorded low end is a step down and the ascent points back.
            plan.push(CellPlan {
                cell,
                base_y: level - ROAD_STAIR_RISE,
                ascent: Some(travel.opposite()),
                variant: RoadPieceVariant::Surface,
            });
            level -= ROAD_STAIR_RISE;
        }
    }
    Ok(plan)
}

/// Whether `style` has a stair piece loaded — the one thing [`plan_drag`]
/// needs to know about the catalogue. `false` (every cell flat) when there's
/// no catalogue, no selected style, or no `stairs.nbt` for it, which is the
/// state a style is in until a `stairs.nbt` is exported for it.
///
/// The **surface** stair specifically (ticket 071), not "either variant": a
/// style that somehow shipped only `stairs-tunnel.nbt` could plan a ramp it
/// then couldn't write on any cell the terrain didn't happen to close over,
/// which is a hole in a road rather than a flat one.
fn stair_available(catalogue: Option<&RoadCatalogue>, style: Option<&str>) -> bool {
    matches!(
        (catalogue, style),
        (Some(catalogue), Some(style))
            if catalogue.get(style, RoadPieceKind::Stair, RoadPieceVariant::Surface).is_some()
    )
}

// -----------------------------------------------------------------------------------------------
// ---- tunnels: the pieces for a cell the terrain closes over (ticket 071) -----------------------
// -----------------------------------------------------------------------------------------------

/// How many of a road cell's 36 columns have to be roofed over before the
/// cell needs a [`RoadPieceVariant::Tunnel`] piece — a strict majority, so
/// the test is `cover > ROAD_TUNNEL_COVER_MAJORITY`.
///
/// Derived from [`ROAD_CELL_SIZE`] rather than written as `18`: 36 is the
/// cell's own footprint, and the rule is "is the majority of this cell
/// roofed", not a magic count that would quietly become a minority if a cell
/// ever stopped being 6x6.
pub(super) const ROAD_TUNNEL_COVER_MAJORITY: usize = (ROAD_CELL_SIZE * ROAD_CELL_SIZE / 2) as usize;

/// How many of `cell`'s 36 columns hold something other than air at exactly
/// world Y `y`.
///
/// **One layer, and literally not-air.** Not [`grid::ground_height_at`]'s
/// top-down scan — that answers "where does the ground stop", which a cell
/// buried in a hillside answers from somewhere far above the road — and not
/// its clutter-skipping notion of ground either: a cell roofed by 36 leaves
/// is exactly as unwalkable as one roofed by 36 stone, and a road that
/// tunnels under a tree is no worse for it.
///
/// A column whose chunk isn't decoded counts as *not* covered. Every cell
/// this is asked about has already passed [`cell_fit`], which refuses
/// outright on an undecoded column, so that case is unreachable in practice
/// — and "don't carve a tunnel through terrain you can't see" is the right
/// way for it to fail if it ever isn't.
fn cover_at(cell: IVec2, y: i32, world: &DecodedWorld) -> usize {
    state::road_cell_tiles(cell)
        .filter(|tile| {
            grid::block_at(IVec3::new(tile.x, y, tile.y), world)
                .is_some_and(|id| id != world::BlockRegistry::AIR)
        })
        .count()
}

/// The first world Y a cell's piece does **not** occupy — where
/// [`cover_at`] takes its reading.
///
/// Read off the *surface* piece's own blueprint rather than a constant,
/// because the pieces aren't all the same height: the shipped flat `dirt`
/// pieces are `6x5x6` and `stairs.nbt` is `6x8x6`, so a hardcoded envelope
/// would sample four blocks inside the stair on one hand or four blocks of
/// sky above a straight on the other. `None` when the style has no piece for
/// this kind at all — there is nothing to measure the top of, and nothing
/// that would be written there either.
fn piece_top_y(
    cell: IVec2,
    base_y: i32,
    style: &str,
    kind: RoadPieceKind,
    catalogue: &RoadCatalogue,
) -> Option<i32> {
    let piece = catalogue.get(style, kind, RoadPieceVariant::Surface)?;
    Some(cell_write_origin(cell, base_y).y + piece.size.y)
}

/// Fills in every [`CellPlan::variant`] in `plan`, the second pass over
/// [`plan_drag`]'s output — see the module docs' "Tunnels".
///
/// A cell that is **already** road keeps its own recorded variant, exactly
/// as `City::add_road_cell` will: re-measuring a written tunnel reads the
/// air the tunnel piece itself carved, which is the whole reason
/// [`state::RoadCell::variant`] is stored rather than derived.
///
/// A cell new to this drag is a tunnel when both of these hold:
///
/// - a strict majority of the 36 columns just above its piece are not air
///   ([`cover_at`], [`ROAD_TUNNEL_COVER_MAJORITY`]);
/// - the catalogue actually has the `-tunnel` piece for its `(style, kind)`.
///
/// The second condition is the same shape [`stair_available`] gives ramps: a
/// style with no tunnel exports keeps building the surface pieces it has,
/// rather than recording cells whose blueprint can't be resolved and leaving
/// holes in the road where they were.
fn plan_tunnels(
    plan: &mut [CellPlan],
    path: &[IVec2],
    world: &DecodedWorld,
    city: &City,
    catalogue: Option<&RoadCatalogue>,
    selected_style: Option<&str>,
) {
    for entry in plan.iter_mut() {
        if let Some(existing) = city.road_cell_at(entry.cell) {
            entry.variant = existing.variant;
            continue;
        }
        let Some(catalogue) = catalogue else { continue };
        let Some(style) = style_for_cell(entry.cell, city, selected_style) else { continue };
        let (kind, _) = piece_for(connections_with_path(city, path, entry.cell), entry.ascent);
        if catalogue.get(style, kind, RoadPieceVariant::Tunnel).is_none() {
            continue;
        }
        let Some(top) = piece_top_y(entry.cell, entry.base_y, style, kind, catalogue) else { continue };
        if cover_at(entry.cell, top, world) > ROAD_TUNNEL_COVER_MAJORITY {
            entry.variant = RoadPieceVariant::Tunnel;
        }
    }
}

/// The piece kind and rotation a cell calls for: a stair if it has an
/// [`state::RoadCell::ascent`], else whatever its connections imply. The
/// write path's and the preview's shared answer — see
/// [`RoadPieceKind::Stair`]'s docs for why a cell's connections alone can
/// never say "ramp".
fn piece_for(connections: RoadConnections, ascent: Option<road::Direction>) -> (RoadPieceKind, Rotation) {
    match ascent {
        Some(ascent) => (RoadPieceKind::Stair, road::stair_rotation(ascent)),
        None => road::select_piece(connections),
    }
}

/// [`road::connections_at`], but treating every cell in `path` as road too —
/// see the module docs' "The preview" for why a drag in progress needs this
/// rather than the real `City`-only answer.
fn connections_with_path(city: &City, path: &[IVec2], cell: IVec2) -> RoadConnections {
    let is_road_or_pending = |c: IVec2| city.is_road_cell(c) || path.contains(&c);
    RoadConnections {
        north: is_road_or_pending(cell + road::Direction::North.offset()),
        south: is_road_or_pending(cell + road::Direction::South.offset()),
        east: is_road_or_pending(cell + road::Direction::East.offset()),
        west: is_road_or_pending(cell + road::Direction::West.offset()),
    }
}

/// The style to preview or write `cell` at (ticket 059): its own recorded
/// style if `city` already has it as a road cell, or `selected` (the
/// currently-picked [`RoadStyleSelection::current`]) if it's new to this
/// drag. `None` only when `cell` is new *and* nothing is selected — see the
/// module docs' "Which style".
fn style_for_cell<'a>(cell: IVec2, city: &'a City, selected: Option<&'a str>) -> Option<&'a str> {
    city.road_style_at(cell).or(selected)
}

/// Every cell a commit needs to (re)render/(re)write: `path` itself, plus any
/// of their cardinal neighbours that are *already* road cells in `city` — a
/// pre-existing dead end that just grew a neighbour may need to become a
/// straight or a corner. Deduplicated; order doesn't matter to any caller.
fn affected_cells(path: &[IVec2], city: &City) -> Vec<IVec2> {
    let mut affected: Vec<IVec2> = path.to_vec();
    for &cell in path {
        for direction in road::Direction::ALL {
            let neighbour = cell + direction.offset();
            if city.is_road_cell(neighbour) && !affected.contains(&neighbour) {
                affected.push(neighbour);
            }
        }
    }
    affected
}

// -----------------------------------------------------------------------------------------------
// ---- the drag: start/track/clear -----------------------------------------------------------------
// -----------------------------------------------------------------------------------------------

/// Starts, tracks and clears [`RoadDragState::start`] off the mouse and the
/// road tool being active. Left click with nothing already pending starts a
/// drag at the hovered cell; releasing hands off to [`try_commit_drag`] and
/// clears the drag regardless of whether a commit actually happened (a
/// refused drag shouldn't leave the state waiting for a click that already
/// landed).
fn update_drag_state(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_input: Res<camera::EguiInputCapture>,
    tool: Option<Res<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    mut drag: ResMut<RoadDragState>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::Road)) || egui_input.pointer {
        drag.start = None;
        return;
    }

    if mouse.just_pressed(MouseButton::Left) {
        if let Some(hovered) = hovered.0 {
            drag.start = Some(cell_of(hovered));
        }
    }
}

/// This frame's drag path — the single hovered cell when nothing is being
/// dragged, the [`drag_path`] between the drag's start and the hovered cell
/// otherwise. `None` with nothing hovered at all.
fn current_path(drag: &RoadDragState, hovered: Option<IVec3>) -> Option<Vec<IVec2>> {
    let hovered_cell = cell_of(hovered?);
    Some(match drag.start {
        Some(start) => drag_path(start, hovered_cell),
        None => vec![hovered_cell],
    })
}

// -----------------------------------------------------------------------------------------------
// ---- preview: quad fallback + real piece meshes ------------------------------------------------
// -----------------------------------------------------------------------------------------------

/// A flat, unrotated [`ROAD_CELL_SIZE`]-square plane in the XZ plane,
/// centred on the origin — the fallback preview shape for a
/// [`RoadPieceKind`] the catalogue has no `.nbt` for yet. See the module
/// docs' "The preview".
fn quad_mesh() -> Mesh {
    let half = ROAD_CELL_SIZE as f32 / 2.0;
    let positions = vec![[-half, 0.0, -half], [half, 0.0, -half], [half, 0.0, half], [-half, 0.0, half]];
    let normals = vec![[0.0, 1.0, 0.0]; 4];
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    Mesh::new(bevy::render::mesh::PrimitiveTopology::TriangleList, RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]))
}

/// A plain-white fallback biome, mirroring `placement::white_biome` — see
/// that function's own doc comment for why this is practically unreachable.
fn white_biome() -> BiomeColors {
    BiomeColors { grass: LinearRgba::WHITE, foliage: LinearRgba::WHITE, water: LinearRgba::WHITE }
}

fn plains_biome_colors(world: &DecodedWorld, maps: &world::ColorMaps) -> BiomeColors {
    let registry = world.biomes.lock().expect("biome registry mutex poisoned");
    let table = world::build_biome_tint_table(&registry, maps);
    table.get(world::BiomeRegistry::PLAINS.0 as usize).copied().unwrap_or_else(white_biome)
}

/// Resolves (and caches) the preview mesh for `(style, kind, variant,
/// rotation)`: the
/// real piece from `catalogue`, meshed via B2/B3's own path, if one has
/// loaded — [`quad_mesh`] otherwise (including when `style` is `None`: a
/// cell new to this drag with nothing selected yet still needs *some*
/// preview). The quad itself is cached once, not per style/kind — every
/// fallback is the same flat square.
///
/// The [`PreviewAnchor`] alongside the handle says which of the two came
/// back, since [`cell_transform`] has to place them differently — see its
/// own docs.
#[allow(clippy::too_many_arguments)]
fn preview_mesh(
    preview: &mut RoadPreviewState,
    style: Option<&str>,
    kind: RoadPieceKind,
    variant: RoadPieceVariant,
    rotation: Rotation,
    catalogue: Option<&RoadCatalogue>,
    atlas: &world::AtlasUvIndex,
    world: &DecodedWorld,
    color_maps: &world::ColorMaps,
    meshes: &mut Assets<Mesh>,
) -> (Handle<Mesh>, PreviewAnchor) {
    let Some(style) = style else {
        return (preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone(), PreviewAnchor::Quad);
    };
    let Some(piece) = catalogue.and_then(|c| c.get(style, kind, variant)) else {
        return (preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone(), PreviewAnchor::Quad);
    };

    let key = (style.to_string(), kind, variant, rotation);
    if let Some(cached) = preview.piece_meshes.get(&key) {
        if let Some(handle) = cached {
            return (handle.clone(), PreviewAnchor::Piece);
        }
        // A cached rotation failure for a *real* piece: still show
        // something rather than nothing.
        return (preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone(), PreviewAnchor::Quad);
    }

    let rotated;
    let blueprint: &Blueprint = if rotation == Rotation::Deg0 {
        piece
    } else {
        match blueprint::rotate_blueprint(piece, rotation) {
            Ok(b) => {
                rotated = b;
                &rotated
            }
            Err(err) => {
                println!("block_viewer: road preview: {style}/{kind:?}/{variant:?} can't rotate to {rotation:?}: {err}");
                preview.piece_meshes.insert(key, None);
                return (preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone(), PreviewAnchor::Quad);
            }
        }
    };

    let biome = plains_biome_colors(world, color_maps);
    let handle = blueprint::mesh_blueprint(blueprint, atlas, biome).map(|mesh| meshes.add(mesh));
    preview.piece_meshes.insert(key, handle.clone());
    match handle {
        Some(handle) => (handle, PreviewAnchor::Piece),
        None => (preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone(), PreviewAnchor::Quad),
    }
}

/// The two translucent preview materials, built once — same shape and same
/// reasoning as `placement::ensure_materials`, see that function's docs.
fn ensure_materials<'a>(
    preview: &'a mut RoadPreviewState,
    terrain_material: &Handle<StandardMaterial>,
    materials: &mut Assets<StandardMaterial>,
) -> &'a PreviewMaterials {
    preview.materials.get_or_insert_with(|| {
        let texture = materials.get(terrain_material).and_then(|m| m.base_color_texture.clone());
        let valid = materials.add(StandardMaterial {
            base_color: Color::srgba(0.35, 0.95, 0.4, 0.55),
            base_color_texture: texture.clone(),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
        let invalid = materials.add(StandardMaterial {
            base_color: Color::srgba(0.95, 0.3, 0.3, 0.55),
            base_color_texture: texture,
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
        PreviewMaterials { valid, invalid }
    })
}

/// Which of [`preview_mesh`]'s two possible meshes came back, because the
/// two are anchored differently and so need different transforms — the
/// distinction ticket 065 had to draw once real pieces started resolving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviewAnchor {
    /// A real catalogue piece meshed by [`blueprint::mesh_blueprint`], whose
    /// geometry starts at the blueprint's own *minimum corner* — exactly
    /// like `placement`'s building ghost.
    Piece,
    /// [`quad_mesh`]'s flat fallback plane, which is built centred on its
    /// own origin and so needs half a cell added back in `(x, z)`.
    Quad,
}

/// World-space transform for a cell's preview — the same `bevy.z = -mc.z`
/// translation `placement::ghost_transform` uses, at
/// [`cell_write_origin`]'s Y so the ghost stands where the blocks will
/// actually land.
///
/// `anchor` is why this isn't one expression: a real piece's mesh is
/// corner-anchored and drops straight onto the cell's minimum corner, while
/// the fallback quad is centred on its own origin and needs half a cell
/// added to reach the same footprint. Before ticket 065 this centred *both*
/// — invisible while every cell fell back to the quad, a 3-block diagonal
/// slide the moment ticket 063's real `dirt` pieces started resolving.
fn cell_transform(cell: IVec2, base_y: i32, anchor: PreviewAnchor) -> Transform {
    let origin = cell_write_origin(cell, base_y);
    let offset = match anchor {
        PreviewAnchor::Piece => 0.0,
        PreviewAnchor::Quad => ROAD_CELL_SIZE as f32 / 2.0,
    };
    Transform::from_xyz(origin.x as f32 + offset, origin.y as f32, -(origin.z as f32 + offset))
}

/// Updates every entity in the preview pool for this frame's drag path —
/// spawning new ones as the path grows, hiding the tail of the pool once the
/// path shrinks, per the module docs.
#[allow(clippy::too_many_arguments)]
fn update_drag_preview(
    mut commands: Commands,
    hovered: Res<HoveredBlock>,
    drag: Res<RoadDragState>,
    tool: Option<Res<ActiveTool>>,
    catalogue: Option<Res<RoadCatalogue>>,
    world: Res<DecodedWorld>,
    city: Res<City>,
    atlas: Option<Res<SharedAtlasIndex>>,
    color_maps: Option<Res<SharedColorMaps>>,
    terrain_material: Option<Res<TerrainMaterial>>,
    style_selection: Option<Res<RoadStyleSelection>>,
    mut preview: ResMut<RoadPreviewState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(&mut Visibility, &mut Transform, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>), With<RoadPreview>>,
) {
    let path = if matches!(tool.as_deref(), Some(ActiveTool::Road)) { current_path(&drag, hovered.0) } else { None };

    let (Some(path), Some(atlas), Some(color_maps), Some(terrain_material)) = (path, atlas, color_maps, terrain_material) else {
        for &entity in &preview.entities {
            if let Ok((mut visibility, ..)) = query.get_mut(entity) {
                *visibility = Visibility::Hidden;
            }
        }
        return;
    };

    let selected_style = style_selection.as_deref().and_then(|s| s.current.as_deref());

    // Ticket 067: the ghost stands on the *plan*'s heights, not on each
    // cell's own terrain — that difference is the whole visible half of the
    // rule, and previewing per-cell ground while committing a plan would put
    // the ghost somewhere the blocks never land.
    //
    // A refused plan (not enough room to climb) still previews, flat at the
    // path's own first-cell level and tinted invalid throughout, so the
    // player sees *where* the road would go and that it won't build, rather
    // than the preview blinking out with no explanation.
    let stairs = stair_available(catalogue.as_deref(), selected_style);
    let plan = plan_drag(&path, &world, &city, stairs);
    let refused = plan.is_err();
    let mut plan = plan.unwrap_or_else(|_| {
        plan_drag(&path, &world, &city, false).unwrap_or_default()
    });
    // Ticket 071: the ghost shows the *tunnel* piece where one is going to be
    // written, for the same reason it stands on the plan's heights — the
    // preview and the commit share one answer so the player can't be shown
    // paving and given a bore.
    plan_tunnels(&mut plan, &path, &world, &city, catalogue.as_deref(), selected_style);

    for (index, planned) in plan.iter().enumerate() {
        let CellPlan { cell, base_y, ascent, variant } = *planned;
        let valid = !refused && cell_valid(cell, &world, &city);
        let style = style_for_cell(cell, &city, selected_style);
        let (kind, rotation) = piece_for(connections_with_path(&city, &path, cell), ascent);
        let (mesh, anchor) = preview_mesh(
            &mut preview,
            style,
            kind,
            variant,
            rotation,
            catalogue.as_deref(),
            &atlas.0,
            &world,
            &color_maps.0,
            &mut meshes,
        );
        let ghost_materials = ensure_materials(&mut preview, &terrain_material.0, &mut materials);
        let material = if valid { ghost_materials.valid.clone() } else { ghost_materials.invalid.clone() };
        let transform = cell_transform(cell, base_y, anchor);

        let entity = match preview.entities.get(index) {
            Some(&entity) => entity,
            None => {
                let entity = commands.spawn((Name::new("Road preview"), Visibility::Hidden, RoadPreview)).id();
                preview.entities.push(entity);
                entity
            }
        };
        if let Ok((mut visibility, mut existing_transform, mut mesh3d, mut material3d)) = query.get_mut(entity) {
            *visibility = Visibility::Visible;
            *existing_transform = transform;
            mesh3d.0 = mesh;
            material3d.0 = material;
        } else {
            commands.entity(entity).insert((Mesh3d(mesh), MeshMaterial3d(material), transform, Visibility::Visible));
        }
    }

    for &entity in preview.entities.iter().skip(plan.len()) {
        if let Ok((mut visibility, ..)) = query.get_mut(entity) {
            *visibility = Visibility::Hidden;
        }
    }
}

// -----------------------------------------------------------------------------------------------
// ---- commit: validate, claim, write ------------------------------------------------------------
// -----------------------------------------------------------------------------------------------

/// Builds one merged [`WorldEdit`] out of every affected cell that has a
/// catalogue piece — see the module docs' "Committing". A cell with no
/// recorded style yet (shouldn't happen — every affected cell is either
/// already in `city`, or was just added there by this same commit, before
/// this runs) or no matching piece for its style contributes nothing (its
/// `City` entry alone is the whole of what this ticket can do for it — see
/// "The preview").
fn road_write_edit(affected: &[IVec2], catalogue: &RoadCatalogue, city: &City) -> WorldEdit {
    let mut merged = WorldEdit::new();
    let mut data_version = None;

    for &cell in affected {
        let Some(road) = city.road_cell_at(cell) else { continue };
        // Ticket 067: a cell recorded with an ascent is a stair, whatever its
        // connections look like — see `piece_for`.
        let (kind, rotation) = piece_for(road::connections_at(city, cell), road.ascent);
        // Ticket 071: and the *variant* the cell was recorded with, never a
        // fresh reading of what's above it — see `state::RoadCell::variant`.
        let Some(piece) = catalogue.get(&road.style, kind, road.variant) else { continue };

        let rotated;
        let blueprint: &Blueprint = if rotation == Rotation::Deg0 {
            piece
        } else {
            match blueprint::rotate_blueprint(piece, rotation) {
                Ok(b) => {
                    rotated = b;
                    &rotated
                }
                Err(err) => {
                    println!(
                        "block_viewer: road build: {}/{kind:?}/{:?} can't rotate to {rotation:?}, skipping cell: {err}",
                        road.style, road.variant
                    );
                    continue;
                }
            }
        };

        // Ticket 065: the piece goes at the cell's *recorded* ground, not at
        // `cell_min_corner`'s placeholder Y of 0 (which buried every road in
        // the deepslate) and not at a freshly resampled height either — see
        // `state::RoadCell`'s docs for why re-deriving it drifts.
        let edit = blueprint_edit(blueprint, cell_write_origin(cell, road.base_y));
        data_version = data_version.or(edit.data_version());
        for crate::edit::BlockEdit { at, state } in edit.edits() {
            merged.set(*at, state.clone());
        }
    }

    if let Some(version) = data_version {
        merged = merged.with_data_version(version);
    }
    merged
}

/// Left-click release with the road tool active: validates the whole drag,
/// claims every cell in [`City`] synchronously, and dispatches the write onto
/// [`AsyncComputeTaskPool`] — see the module docs' "Committing".
#[allow(clippy::too_many_arguments)]
fn try_commit_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    tool: Option<Res<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    world: Res<DecodedWorld>,
    mut city: ResMut<City>,
    catalogue: Option<Res<RoadCatalogue>>,
    style_selection: Option<Res<RoadStyleSelection>>,
    mut build: ResMut<RoadBuildState>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut drag: ResMut<RoadDragState>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::Road)) || build.pending.is_some() || !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(start) = drag.start else { return };
    drag.start = None; // Cleared regardless of what happens below — see update_drag_state's docs.

    let Some(hovered) = hovered.0 else { return };
    let path = drag_path(start, cell_of(hovered));

    if !path.iter().all(|&cell| cell_valid(cell, &world, &city)) {
        println!("block_viewer: road drag refused: not every cell in the path is buildable");
        return;
    }

    // Ticket 059: any *new* cell in this path needs a style to be recorded
    // under. An already-road cell keeps whatever it already has — see
    // `add_road_cell`'s own docs — so it's fine for `selected_style` to go
    // unused in a drag that only re-crosses existing road.
    let selected_style = style_selection.and_then(|selection| selection.current.clone());
    if path.iter().any(|&cell| !city.is_road_cell(cell)) && selected_style.is_none() {
        println!("block_viewer: road drag refused: no road style selected ([ or ] to pick one)");
        return;
    }
    let build_style = selected_style.unwrap_or_default();

    // Tickets 065/067: the whole drag's height profile, resolved here — once,
    // off the same `plan_drag` the preview meshed against — and handed cell
    // by cell to `add_road_cell` to be remembered. 065's lesson was that a
    // height re-derived later drifts; 067's is that it has to be decided for
    // the *placement*, not per cell, or the road comes out as a run of
    // one-block cliffs.
    let stairs = stair_available(catalogue.as_deref(), Some(build_style.as_str()));
    let mut plan = match plan_drag(&path, &world, &city, stairs) {
        Ok(plan) => plan,
        Err(refusal) => {
            println!("block_viewer: road drag refused: {refusal}");
            return;
        }
    };
    // Ticket 071: which cells the terrain closes over, measured off the
    // heights above and *before* anything is written — once the tunnel
    // pieces are in the world the cover they were chosen for is gone.
    plan_tunnels(&mut plan, &path, &world, &city, catalogue.as_deref(), Some(build_style.as_str()));

    let newly_added: Vec<IVec2> = path.iter().copied().filter(|&cell| !city.is_road_cell(cell)).collect();
    for &CellPlan { cell, base_y, ascent, variant } in &plan {
        // Already validated above; `add_road_cell` only fails on occupancy,
        // which `cell_valid` just confirmed clear (or already-road, which is
        // idempotent — and keeps its own recorded style, height and ascent) —
        // see the module docs' "Committing".
        if let Err(err) = city.add_road_cell(cell, build_style.clone(), base_y, ascent, variant) {
            println!("block_viewer: road drag refused partway through (a race with another edit?): {err}");
            for cell in &newly_added {
                city.remove_road_cell(*cell);
            }
            return;
        }
    }

    let affected = affected_cells(&path, &city);
    let Some(catalogue) = catalogue else {
        // No `RoadCatalogue` resource at all — the cells are recorded; there
        // is nothing to mesh or write. See the module docs' "The preview" for
        // why a `City` entry without geometry is still a legitimate state.
        println!("block_viewer: built {} road cell(s) (no road catalogue loaded, nothing written to the world)", path.len());
        return;
    };
    let edit = road_write_edit(&affected, &catalogue, &city);
    if edit.is_empty() {
        println!(
            "block_viewer: built {} road cell(s) (no matching road pieces loaded, nothing written to the world)",
            path.len()
        );
        return;
    }

    let Some(region_cache) = region_cache else {
        println!("block_viewer: can't write road cells: no save is loaded");
        return;
    };

    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy { capture_replaced: false, allow_dirty_regions: true, ..EditPolicy::default() };
    let task_edit = edit;

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        super::commit::apply_building_edit(&mut cache, &task_edit, &policy)
    });

    build.pending = Some(PendingRoadBuild { newly_added, task });
}

/// Single non-blocking poll of the in-flight write, the same
/// `block_on(poll_once(..))` pattern `city::commit::poll_commit` uses. On
/// failure: rolls back exactly the cells this drag newly added — see the
/// module docs.
fn poll_road_build(
    mut build: ResMut<RoadBuildState>,
    mut city: ResMut<City>,
    mut write_status: ResMut<WriteStatus>,
    mut edited: EventWriter<ChunksEdited>,
) {
    let result = {
        let Some(pending) = &mut build.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still applying.
        };
        result
    };
    let PendingRoadBuild { newly_added, .. } = build.pending.take().expect("just matched Some above");

    match result {
        Ok(report) => {
            let blocks = report.blocks_written;
            let chunks = report.chunks.len();
            println!(
                "block_viewer: built {} road cell(s) ({blocks} block(s) across {chunks} chunk(s), not yet saved to disk)",
                newly_added.len()
            );
            write_status.record_success(WriteKind::Road, format!("{} road cell(s)", newly_added.len()), &report);
            edited.send(ChunksEdited(report.chunks));
        }
        Err(err) => {
            for cell in &newly_added {
                city.remove_road_cell(*cell);
            }
            println!("block_viewer: road build failed, rolled back {} cell(s): {err}", newly_added.len());
            write_status.record_failure(WriteKind::Road, format!("{} road cell(s)", newly_added.len()), err.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- drag_path -------------------------------------------------------

    #[test]
    fn a_single_click_is_a_one_cell_path() {
        let path = drag_path(IVec2::new(3, 3), IVec2::new(3, 3));
        assert_eq!(path, vec![IVec2::new(3, 3)]);
    }

    #[test]
    fn a_straight_horizontal_drag_walks_every_cell_in_between() {
        let path = drag_path(IVec2::new(0, 0), IVec2::new(3, 0));
        assert_eq!(path, vec![IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(2, 0), IVec2::new(3, 0)]);
    }

    #[test]
    fn a_straight_vertical_drag_walks_every_cell_in_between() {
        let path = drag_path(IVec2::new(0, 0), IVec2::new(0, 3));
        assert_eq!(path, vec![IVec2::new(0, 0), IVec2::new(0, 1), IVec2::new(0, 2), IVec2::new(0, 3)]);
    }

    #[test]
    fn a_diagonal_drag_is_an_l_shape_horizontal_then_vertical() {
        let path = drag_path(IVec2::new(0, 0), IVec2::new(2, 2));
        assert_eq!(
            path,
            vec![IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(2, 0), IVec2::new(2, 1), IVec2::new(2, 2)],
            "the corner (2, 0) is counted once"
        );
    }

    #[test]
    fn a_drag_toward_negative_cells_still_walks_a_contiguous_path() {
        let path = drag_path(IVec2::new(2, 2), IVec2::new(0, 0));
        assert_eq!(
            path,
            vec![IVec2::new(2, 2), IVec2::new(1, 2), IVec2::new(0, 2), IVec2::new(0, 1), IVec2::new(0, 0)]
        );
        // Every consecutive pair is a cardinal neighbour of the next.
        for pair in path.windows(2) {
            let delta = (pair[1] - pair[0]).abs();
            assert_eq!(delta.x + delta.y, 1, "{pair:?} is not cardinally adjacent");
        }
    }

    // --- cell_of -----------------------------------------------------------

    #[test]
    fn cell_of_maps_block_coordinates_to_their_cell() {
        assert_eq!(cell_of(IVec3::new(0, 64, 0)), IVec2::new(0, 0));
        assert_eq!(cell_of(IVec3::new(5, 64, 5)), IVec2::new(0, 0));
        assert_eq!(cell_of(IVec3::new(6, 64, 0)), IVec2::new(1, 0));
        assert_eq!(cell_of(IVec3::new(-1, 64, -1)), IVec2::new(-1, -1), "div_euclid, not truncating division");
    }

    // --- cell_occupancy_ok / affected_cells --------------------------------

    #[test]
    fn cell_occupancy_ok_is_true_for_free_or_already_road_ground() {
        let mut city = City::default();
        assert!(cell_occupancy_ok(IVec2::new(0, 0), &city));
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        assert!(cell_occupancy_ok(IVec2::new(0, 0), &city), "already-road counts as ok, not blocked");
    }

    #[test]
    fn cell_occupancy_ok_is_false_over_a_building() {
        let mut city = City::default();
        city.place_building("house01", None, IVec3::new(0, 64, 0), crate::blueprint::Rotation::Deg0, IVec2::new(2, 2)).unwrap();
        assert!(!cell_occupancy_ok(IVec2::new(0, 0), &city));
    }

    #[test]
    fn affected_cells_includes_the_path_and_its_already_road_neighbours() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(-1, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap(); // west neighbour of (0, 0)
        city.add_road_cell(IVec2::new(5, 5), "dirt", 64, None, RoadPieceVariant::Surface).unwrap(); // unrelated, far away

        let affected = affected_cells(&[IVec2::new(0, 0)], &city);
        assert!(affected.contains(&IVec2::new(0, 0)));
        assert!(affected.contains(&IVec2::new(-1, 0)));
        assert!(!affected.contains(&IVec2::new(5, 5)));
        assert_eq!(affected.len(), 2);
    }

    #[test]
    fn affected_cells_does_not_duplicate_a_neighbour_shared_by_two_path_cells() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(1, 1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        // Both (0, 1) and (2, 1) border (1, 1); (1, 0)/(1, 2) also border it.
        let affected = affected_cells(&[IVec2::new(0, 1), IVec2::new(2, 1)], &city);
        assert_eq!(affected.iter().filter(|&&c| c == IVec2::new(1, 1)).count(), 1);
    }

    // --- connections_with_path ----------------------------------------------

    #[test]
    fn connections_with_path_treats_the_rest_of_the_path_as_road() {
        let city = City::default();
        let path = vec![IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(2, 0)];
        let connections = connections_with_path(&city, &path, IVec2::new(1, 0));
        assert!(connections.west, "(0,0) is in the path");
        assert!(connections.east, "(2,0) is in the path");
        assert!(!connections.north);
        assert!(!connections.south);
    }

    #[test]
    fn connections_with_path_still_sees_real_city_roads() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, -1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap(); // north of (0, 0)
        let connections = connections_with_path(&city, &[IVec2::new(0, 0)], IVec2::new(0, 0));
        assert!(connections.north);
    }

    // --- road_write_edit: the pure edit-building logic ----------------------

    use crate::blueprint::{write_structure_file, BlockState};
    use super::super::road_catalogue::{load_road_catalogue_dir, piece_path};

    fn state_named(name: &str) -> BlockState {
        name.parse().unwrap()
    }

    /// A `ROAD_CELL_SIZE`-square, one-block-tall piece: stone everywhere
    /// except `(0,0,0)` — small and cheap to write into the fixture and to
    /// enumerate by hand in an assertion, the same shape
    /// `road_catalogue::tests::one_stone` builds.
    fn one_stone_piece() -> Blueprint {
        let size = IVec3::new(ROAD_CELL_SIZE, 1, ROAD_CELL_SIZE);
        let volume = (size.x * size.y * size.z) as usize;
        let mut blocks = vec![1u16; volume];
        blocks[0] = 0; // (0,0,0) stays air.
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state_named("minecraft:stone")],
            blocks,
            data_version: 4438,
            failed_columns: 0,
        }
    }

    /// A fixture directory of this test's own. The pid alone isn't unique
    /// enough: `cycle_test_app` hands every one of its four callers the same
    /// `name`, and `cargo test` runs them in parallel — so one test's
    /// `remove_dir_all` below would delete a fixture another was still
    /// reading, or write a *different* set of styles into it. The counter
    /// makes each call its own directory regardless of what it's called.
    fn temp_dir(name: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("block_viewer_test_road_build_{name}_{}_{seq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// Writes `blueprint` at `piece_path(dir, style, kind, Surface)`,
    /// creating the style subdirectory first — same reason
    /// `road_catalogue::tests::write_piece` needs to.
    fn write_piece(dir: &std::path::Path, style: &str, kind: RoadPieceKind, blueprint: &Blueprint) {
        write_variant_piece(dir, style, kind, RoadPieceVariant::Surface, blueprint);
    }

    /// [`write_piece`], for a named variant — ticket 071's tests need to put
    /// a `-tunnel` file beside a surface one.
    fn write_variant_piece(
        dir: &std::path::Path,
        style: &str,
        kind: RoadPieceKind,
        variant: RoadPieceVariant,
        blueprint: &Blueprint,
    ) {
        std::fs::create_dir_all(dir.join(style)).expect("should create style dir");
        write_structure_file(&piece_path(dir, style, kind, variant), blueprint).unwrap();
    }

    /// A catalogue with only [`RoadPieceKind::Isolated`] loaded, under style
    /// `"dirt"` — enough for a single, disconnected cell.
    fn catalogue_with_isolated(dir: &std::path::Path) -> RoadCatalogue {
        write_piece(dir, "dirt", RoadPieceKind::Isolated, &one_stone_piece());
        let (catalogue, _skipped) = load_road_catalogue_dir(dir);
        catalogue
    }

    /// A catalogue with [`RoadPieceKind::Isolated`] *and*
    /// [`RoadPieceKind::Straight`], both under style `"dirt"` — enough to
    /// prove [`road_write_edit`] picks a different piece per cell and merges
    /// both into one edit.
    fn catalogue_with_isolated_and_straight(dir: &std::path::Path) -> RoadCatalogue {
        write_piece(dir, "dirt", RoadPieceKind::Isolated, &one_stone_piece());
        write_piece(dir, "dirt", RoadPieceKind::Straight, &one_stone_piece());
        let (catalogue, _skipped) = load_road_catalogue_dir(dir);
        catalogue
    }

    #[test]
    fn road_write_edit_writes_the_isolated_piece_for_a_lone_cell() {
        let dir = temp_dir("write_edit_isolated");
        let catalogue = catalogue_with_isolated(&dir);
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        assert_eq!(edit.len(), (ROAD_CELL_SIZE * ROAD_CELL_SIZE) as usize, "every block in the one cell's piece");
        assert_eq!(edit.data_version(), Some(4438));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A cell recorded under a style the catalogue has no pieces for at all
    /// contributes nothing — same "skipped, not guessed at" contract a
    /// missing *kind* gets.
    #[test]
    fn road_write_edit_is_empty_for_a_cell_whose_style_is_not_in_the_catalogue() {
        let dir = temp_dir("write_edit_unknown_style");
        let catalogue = catalogue_with_isolated(&dir); // only "dirt" is loaded
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "paved", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        assert!(edit.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn road_write_edit_is_empty_when_the_catalogue_has_no_matching_piece() {
        let dir = temp_dir("write_edit_no_piece");
        // A catalogue with only Straight — this cell resolves to Isolated,
        // which the catalogue doesn't have.
        write_piece(&dir, "dirt", RoadPieceKind::Straight, &one_stone_piece());
        let (catalogue, _skipped) = load_road_catalogue_dir(&dir);
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        assert!(edit.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn road_write_edit_offsets_each_cells_piece_by_its_own_corner() {
        let dir = temp_dir("write_edit_offset");
        let catalogue = catalogue_with_isolated(&dir);
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(2, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0), IVec2::new(2, 0)], &catalogue, &city);
        let positions: std::collections::HashSet<IVec3> = edit.edits().iter().map(|e| e.at).collect();
        // Cell (2, 0)'s corner is (12, _, 0) — two cells over. The Y is
        // `cell_write_origin`'s, not `base_y` itself; that's this test's
        // neighbours' business, not its own.
        let y = cell_write_origin(IVec2::ZERO, 64).y;
        assert!(positions.contains(&IVec3::new(0, y, 0)));
        assert!(positions.contains(&IVec3::new(2 * ROAD_CELL_SIZE, y, 0)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Ticket 065, the bug that made every built road invisible: the piece
    /// is written relative to the cell's *recorded* `base_y`, not at
    /// `cell_min_corner`'s placeholder Y of 0 (which buried it ~60 blocks
    /// down in the deepslate). The exact offset from `base_y` is
    /// `cell_write_origin`'s business — see the test below it; all this one
    /// asserts is that the height is being read at all.
    #[test]
    fn road_write_edit_writes_relative_to_the_cells_recorded_base_y_not_zero() {
        let dir = temp_dir("write_edit_base_y");
        let catalogue = catalogue_with_isolated(&dir);
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 71, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        assert!(!edit.is_empty());
        let y = cell_write_origin(IVec2::ZERO, 71).y;
        assert!(y > 60, "a road on ground at 71 must not land anywhere near the deepslate");
        assert!(
            edit.edits().iter().all(|e| e.at.y == y),
            "every block of a one-layer piece should land on the cell's own ground, not at Y=0"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The anchoring itself: a piece's *surface course*
    /// ([`ROAD_PIECE_SUBGRADE_DEPTH`] layers up from its bottom) lands on the
    /// terrain's topmost ground block — `base_y - 1`, `base_y` being
    /// `fit_footprint`'s "one *above* the ground". Flush with the grass
    /// beside it, with the shipped pieces' air layers as clearance above.
    #[test]
    fn cell_write_origin_puts_the_surface_course_at_the_terrain_surface() {
        let surface = cell_write_origin(IVec2::ZERO, 64).y + ROAD_PIECE_SUBGRADE_DEPTH;
        assert_eq!(surface, 63, "ground at 64 means the topmost ground block is 63");
    }

    /// Two cells fitted to different ground each keep their own height —
    /// a single drag across a step doesn't flatten to one shared Y.
    #[test]
    fn road_write_edit_uses_each_cells_own_base_y() {
        let dir = temp_dir("write_edit_two_heights");
        let catalogue = catalogue_with_isolated(&dir);
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(2, 0), "dirt", 70, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0), IVec2::new(2, 0)], &catalogue, &city);
        let positions: std::collections::HashSet<IVec3> = edit.edits().iter().map(|e| e.at).collect();
        assert!(positions.contains(&IVec3::new(0, cell_write_origin(IVec2::ZERO, 64).y, 0)));
        assert!(positions.contains(&IVec3::new(2 * ROAD_CELL_SIZE, cell_write_origin(IVec2::new(2, 0), 70).y, 0)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The preview and the write must agree, cell for cell — the disagreement
    /// ticket 065 exists to close. A real piece's ghost is corner-anchored
    /// and stands exactly at `cell_write_origin`; only the fallback quad
    /// (centred on its own origin) gets half a cell added back.
    #[test]
    fn cell_transform_stands_a_piece_ghost_exactly_where_the_write_lands() {
        let cell = IVec2::new(3, -2);
        let origin = cell_write_origin(cell, 68);
        let piece = cell_transform(cell, 68, PreviewAnchor::Piece);
        assert_eq!(piece.translation, Vec3::new(origin.x as f32, origin.y as f32, -(origin.z as f32)));

        let half = ROAD_CELL_SIZE as f32 / 2.0;
        let quad = cell_transform(cell, 68, PreviewAnchor::Quad);
        assert_eq!(quad.translation, Vec3::new(origin.x as f32 + half, origin.y as f32, -(origin.z as f32 + half)));
    }

    #[test]
    fn road_write_edit_picks_a_different_piece_per_cells_own_connections() {
        let dir = temp_dir("write_edit_mixed");
        // Only Isolated and Straight are loaded — deliberately no DeadEnd.
        let catalogue = catalogue_with_isolated_and_straight(&dir);
        let mut city = City::default();
        // A straight run of three cells: the middle one sees a north *and*
        // a south neighbour, resolving to Straight — the two ends resolve
        // to DeadEnd, which this catalogue has no piece for.
        city.add_road_cell(IVec2::new(0, -1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, -1), IVec2::new(0, 0), IVec2::new(0, 1)], &catalogue, &city);
        assert_eq!(
            edit.len(),
            (ROAD_CELL_SIZE * ROAD_CELL_SIZE) as usize,
            "only the middle (Straight) cell has a matching piece; the two DeadEnd ends are skipped, not guessed at"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Ticket 059's own reason to exist: two cells built as *different*
    /// styles each pull their piece from their own style's set, not one
    /// style's set applied to both.
    #[test]
    fn road_write_edit_reads_each_cells_own_recorded_style() {
        let dir = temp_dir("write_edit_two_styles");
        write_piece(&dir, "dirt", RoadPieceKind::Isolated, &one_stone_piece());
        write_piece(&dir, "paved", RoadPieceKind::Isolated, &one_stone_piece());
        let (catalogue, _skipped) = load_road_catalogue_dir(&dir);

        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(100, 100), "paved", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0), IVec2::new(100, 100)], &catalogue, &city);
        assert_eq!(
            edit.len(),
            2 * (ROAD_CELL_SIZE * ROAD_CELL_SIZE) as usize,
            "both cells' pieces should be written, one from each style"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- poll_road_build: the City/write-status glue, through a real App ---
    //
    // Same split `city::commit`/`city::demolish`'s own tests use: the pure
    // decision logic (above) is tested directly; `apply_building_edit` itself
    // is proven once by `city::commit`'s own tests (this module reuses that
    // exact function via `super::commit::apply_building_edit`); what's left
    // to prove here is the `City`/`WriteStatus` glue, through a task whose
    // result is fixed ahead of time.

    use bevy::tasks::TaskPool;
    use crate::edit::EditRefusal;

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn road_build_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(RoadBuildPlugin).insert_resource(City::default());
        app
    }

    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if app.world().resource::<RoadBuildState>().pending.is_none() {
                return;
            }
        }
        panic!("road build never settled");
    }

    #[test]
    fn poll_road_build_success_fires_chunks_edited_and_records_the_write() {
        let mut app = road_build_test_app();
        app.world_mut().resource_mut::<City>().add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let report = EditReport { blocks_written: 36, chunks: vec![(0, 0)], regions: vec![(0, 0)], replaced: None };
        let task = pool().spawn(async move { Ok(report) });
        app.world_mut().resource_mut::<RoadBuildState>().pending =
            Some(PendingRoadBuild { newly_added: vec![IVec2::new(0, 0)], task });

        run_until_settled(&mut app);

        assert!(app.world().resource::<RoadBuildState>().pending.is_none());
        assert!(app.world().resource::<City>().is_road_cell(IVec2::new(0, 0)), "a successful write keeps the cell");

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].0, vec![(0, 0)]);
    }

    #[test]
    fn poll_road_build_failure_rolls_back_only_the_newly_added_cells() {
        let mut app = road_build_test_app();
        {
            let mut city = app.world_mut().resource_mut::<City>();
            city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap(); // pre-existing neighbour, not newly added
            city.add_road_cell(IVec2::new(1, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap(); // this drag's own new cell
        }

        let task = pool().spawn(async { Err(EditRefusal::Empty) });
        app.world_mut().resource_mut::<RoadBuildState>().pending =
            Some(PendingRoadBuild { newly_added: vec![IVec2::new(1, 0)], task });

        run_until_settled(&mut app);

        let city = app.world().resource::<City>();
        assert!(!city.is_road_cell(IVec2::new(1, 0)), "the newly-added cell is rolled back");
        assert!(city.is_road_cell(IVec2::new(0, 0)), "the pre-existing neighbour must not be touched");

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "nothing changed in the world, so nothing needs re-meshing");
    }

    // --- style_for_cell (ticket 059) ----------------------------------------

    #[test]
    fn style_for_cell_prefers_a_cells_own_recorded_style_over_the_selection() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        assert_eq!(style_for_cell(IVec2::new(0, 0), &city, Some("paved")), Some("dirt"));
    }

    #[test]
    fn style_for_cell_falls_back_to_the_selection_for_a_new_cell() {
        let city = City::default();
        assert_eq!(style_for_cell(IVec2::new(0, 0), &city, Some("paved")), Some("paved"));
    }

    #[test]
    fn style_for_cell_is_none_for_a_new_cell_with_nothing_selected() {
        let city = City::default();
        assert_eq!(style_for_cell(IVec2::new(0, 0), &city, None), None);
    }

    // --- cycle_road_style ---------------------------------------------------

    fn cycle_test_app(styles: &[&str]) -> (App, std::path::PathBuf) {
        let dir = temp_dir("cycle_road_style");
        for &style in styles {
            for kind in RoadPieceKind::ALL {
                write_piece(&dir, style, kind, &one_stone_piece());
            }
        }
        let (catalogue, _skipped) = load_road_catalogue_dir(&dir);

        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<camera::EguiInputCapture>()
            .init_resource::<RoadStyleSelection>()
            .insert_resource(ActiveTool::Road)
            .insert_resource(catalogue)
            .add_systems(Update, cycle_road_style);
        (app, dir)
    }

    fn press(app: &mut App, key: KeyCode) {
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(key);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().release(key);
    }

    #[test]
    fn cycle_road_style_auto_picks_the_first_style_with_no_keypress() {
        let (mut app, dir) = cycle_test_app(&["dirt", "paved"]);
        app.update();
        assert_eq!(app.world().resource::<RoadStyleSelection>().current.as_deref(), Some("dirt"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bracket_right_cycles_forward_and_wraps() {
        let (mut app, dir) = cycle_test_app(&["dirt", "gravel", "paved"]);
        app.update();
        press(&mut app, KeyCode::BracketRight);
        assert_eq!(app.world().resource::<RoadStyleSelection>().current.as_deref(), Some("gravel"));
        press(&mut app, KeyCode::BracketRight);
        assert_eq!(app.world().resource::<RoadStyleSelection>().current.as_deref(), Some("paved"));
        press(&mut app, KeyCode::BracketRight);
        assert_eq!(app.world().resource::<RoadStyleSelection>().current.as_deref(), Some("dirt"), "wraps back around");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bracket_left_cycles_backward_and_wraps() {
        let (mut app, dir) = cycle_test_app(&["dirt", "gravel", "paved"]);
        app.update();
        press(&mut app, KeyCode::BracketLeft);
        assert_eq!(
            app.world().resource::<RoadStyleSelection>().current.as_deref(),
            Some("paved"),
            "wraps to the last style"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cycle_road_style_does_nothing_outside_the_road_tool() {
        let (mut app, dir) = cycle_test_app(&["dirt", "paved"]);
        app.insert_resource(ActiveTool::Building);
        app.update();
        assert_eq!(app.world().resource::<RoadStyleSelection>().current, None, "not auto-picked outside the road tool");
        std::fs::remove_dir_all(&dir).ok();
    }

    // -- height planning (ticket 067) ----------------------------------------

    use super::super::road::Direction;

    /// A `DecodedWorld` whose ground is flat at the given height across each
    /// listed *cell*'s full 6x6 footprint — `fit_footprint` samples all 36
    /// tiles and takes the minimum, so a half-filled cell would read as the
    /// lower half's height. Built the same way `city::grid`'s own fixtures
    /// build theirs: one stone block per tile, everything else air, chunks
    /// that cover nothing simply absent.
    fn world_with_cell_ground(cells: &[(IVec2, i32)]) -> DecodedWorld {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        use crate::world::{BiomeRegistry, BlockRegistry, ChunkColumn, ChunkSection};

        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");
        let size = world::SECTION_SIZE as i32;
        let mut columns: HashMap<(i32, i32), ChunkColumn> = HashMap::new();

        for &(cell, height) in cells {
            // `ground_height_at` answers "topmost solid + 1", so the block
            // itself goes one below the height the caller means.
            let y = height - 1;
            for tile in state::road_cell_tiles(cell) {
                let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
                let (local_x, local_z) = (tile.x.rem_euclid(size) as usize, tile.y.rem_euclid(size) as usize);
                let section_y = y.div_euclid(size) as i8;
                let local_y = y.rem_euclid(size) as usize;

                let column = columns.entry(chunk).or_insert_with(|| ChunkColumn {
                    x: chunk.0,
                    z: chunk.1,
                    sections: Vec::new(),
                    floor_y: world::WORLD_MIN_Y,
                });
                let section = match column.sections.iter().position(|s| s.y == section_y) {
                    Some(index) => index,
                    None => {
                        column.sections.push(ChunkSection {
                            y: section_y,
                            blocks: Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]),
                            biomes: Box::new([BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]),
                        });
                        column.sections.len() - 1
                    }
                };
                column.sections[section].blocks[ChunkSection::index(local_x, local_y, local_z)] = stone;
            }
        }

        DecodedWorld {
            registry: Arc::new(Mutex::new(registry)),
            biomes: Arc::new(Mutex::new(BiomeRegistry::new())),
            columns,
        }
    }

    fn levels(plan: &[CellPlan]) -> Vec<i32> {
        plan.iter().map(|c| c.base_y).collect()
    }

    fn ascents(plan: &[CellPlan]) -> Vec<Option<Direction>> {
        plan.iter().map(|c| c.ascent).collect()
    }

    /// The rule the user asked for first: one drag, one level, taken from
    /// the cell the drag *started* on — even though the ground under the run
    /// climbs a block per cell.
    #[test]
    fn a_drag_across_a_slope_is_flat_at_the_first_cells_level_when_no_stair_exists() {
        let path: Vec<IVec2> = (0..5).map(|x| IVec2::new(x, 0)).collect();
        let world = world_with_cell_ground(&path.iter().enumerate().map(|(i, &c)| (c, 64 + i as i32)).collect::<Vec<_>>());

        let plan = plan_drag(&path, &world, &City::default(), false).unwrap();
        assert_eq!(levels(&plan), vec![64; 5], "every cell should sit at the first cell's ground, not its own");
        assert!(ascents(&plan).iter().all(Option::is_none));
    }

    /// Even with a stair piece available, a run whose ends are less than half
    /// a step apart snaps to one level — the "Y of the first tile dictates Y
    /// of the entire road" case, reached through the same code path as the
    /// stepped one.
    #[test]
    fn a_gentle_slope_snaps_to_a_single_level_even_with_stairs_available() {
        let path: Vec<IVec2> = (0..4).map(|x| IVec2::new(x, 0)).collect();
        // Ends one block apart: nowhere near the four a stair bridges.
        let world = world_with_cell_ground(&[(path[0], 64), (path[1], 64), (path[2], 65), (path[3], 65)]);

        let plan = plan_drag(&path, &world, &City::default(), true).unwrap();
        assert_eq!(levels(&plan), vec![64; 4]);
        assert!(ascents(&plan).iter().all(Option::is_none));
    }

    /// One four-block step: a single stair somewhere in the middle, the run
    /// before it at the start level and the run after it a step up.
    #[test]
    fn a_four_block_climb_becomes_one_stair_with_flat_runs_either_side() {
        let path: Vec<IVec2> = (0..5).map(|x| IVec2::new(x, 0)).collect();
        let world = world_with_cell_ground(&[
            (path[0], 64),
            (path[1], 65),
            (path[2], 66),
            (path[3], 67),
            (path[4], 68),
        ]);

        let plan = plan_drag(&path, &world, &City::default(), true).unwrap();
        let stair_count = plan.iter().filter(|c| c.ascent.is_some()).count();
        assert_eq!(stair_count, 1, "{plan:?}");

        let stair = plan.iter().position(|c| c.ascent.is_some()).unwrap();
        assert!(stair > 0 && stair < plan.len() - 1, "a stair must not be an end of the drag");
        assert_eq!(plan[stair].ascent, Some(Direction::East), "the path travels east, so the climb does too");
        // Low end level with everything behind it, high end level with
        // everything ahead: that is what "contiguous" means here.
        assert!(plan[..stair].iter().all(|c| c.base_y == 64), "{plan:?}");
        assert_eq!(plan[stair].base_y, 64);
        assert!(plan[stair + 1..].iter().all(|c| c.base_y == 68), "{plan:?}");
    }

    /// A descending drag: the stair still *ascends* — back the way the path
    /// came — and its recorded `base_y` is its low end, on the downhill side.
    #[test]
    fn a_descending_drag_records_the_stairs_low_end_and_an_uphill_ascent() {
        let path: Vec<IVec2> = (0..5).map(|x| IVec2::new(x, 0)).collect();
        let world = world_with_cell_ground(&[
            (path[0], 68),
            (path[1], 67),
            (path[2], 66),
            (path[3], 65),
            (path[4], 64),
        ]);

        let plan = plan_drag(&path, &world, &City::default(), true).unwrap();
        let stair = plan.iter().position(|c| c.ascent.is_some()).expect("one stair");
        assert_eq!(plan[stair].ascent, Some(Direction::West), "the path runs east and drops, so the climb faces west");
        assert_eq!(plan[stair].base_y, 64, "a stair's base_y is its low end");
        assert!(plan[..stair].iter().all(|c| c.base_y == 68), "{plan:?}");
        assert!(plan[stair + 1..].iter().all(|c| c.base_y == 64), "{plan:?}");
    }

    /// Two steps get two stairs, spread rather than stacked, and the levels
    /// walk 64 -> 68 -> 72 exactly.
    #[test]
    fn an_eight_block_climb_spreads_two_stairs_along_the_path() {
        let path: Vec<IVec2> = (0..9).map(|x| IVec2::new(x, 0)).collect();
        let mut ground: Vec<(IVec2, i32)> = path.iter().map(|&c| (c, 64)).collect();
        ground[8].1 = 72;

        let plan = plan_drag(&path, &world_with_cell_ground(&ground), &City::default(), true).unwrap();
        let stairs: Vec<usize> = plan.iter().enumerate().filter(|(_, c)| c.ascent.is_some()).map(|(i, _)| i).collect();
        assert_eq!(stairs.len(), 2, "{plan:?}");
        assert!(stairs[1] - stairs[0] > 1, "the two stairs should be spread, not adjacent: {stairs:?}");

        let mut expected_levels: Vec<i32> = Vec::new();
        let mut level = 64;
        for (index, cell) in plan.iter().enumerate() {
            expected_levels.push(level);
            if cell.ascent.is_some() {
                level += ROAD_STAIR_RISE;
            }
            let _ = index;
        }
        assert_eq!(levels(&plan), expected_levels);
        assert_eq!(*levels(&plan).last().unwrap(), 72, "the last cell lands on the snapped end level");
    }

    /// The end level always snaps to a whole number of stairs off the start,
    /// halves away from zero — six blocks is closer to two steps than one.
    #[test]
    fn the_end_level_snaps_to_a_whole_number_of_stair_steps() {
        for (raw_rise, expected_end) in [(0, 64), (1, 64), (2, 68), (5, 68), (6, 72), (7, 72), (-2, 60), (-6, 56)] {
            let path: Vec<IVec2> = (0..9).map(|x| IVec2::new(x, 0)).collect();
            let mut ground: Vec<(IVec2, i32)> = path.iter().map(|&c| (c, 64)).collect();
            ground[8].1 = 64 + raw_rise;

            let plan = plan_drag(&path, &world_with_cell_ground(&ground), &City::default(), true).unwrap();
            assert_eq!(
                *levels(&plan).last().unwrap(),
                expected_end,
                "a {raw_rise}-block difference should snap to {expected_end}"
            );
            assert_eq!(levels(&plan)[0], 64, "the start level is never snapped — it's the anchor");
        }
    }

    /// The L's corner cell can't be a bend and a ramp at once, so it's never
    /// picked as a stair.
    #[test]
    fn the_corner_of_an_l_shaped_drag_is_never_a_stair() {
        let path = drag_path(IVec2::new(0, 0), IVec2::new(3, 3));
        let corner = IVec2::new(3, 0);
        let corner_index = path.iter().position(|&c| c == corner).expect("the L turns at (3, 0)");

        let mut ground: Vec<(IVec2, i32)> = path.iter().map(|&c| (c, 64)).collect();
        ground.last_mut().unwrap().1 = 68;

        let plan = plan_drag(&path, &world_with_cell_ground(&ground), &City::default(), true).unwrap();
        assert_eq!(plan[corner_index].ascent, None, "{plan:?}");
        assert_eq!(plan.iter().filter(|c| c.ascent.is_some()).count(), 1);
    }

    /// A three-cell drag has exactly one interior cell, so it can bridge one
    /// step and no more. Two is refused whole — no partial road, no cliff.
    #[test]
    fn a_climb_with_nowhere_to_put_the_stairs_is_refused_whole() {
        let path: Vec<IVec2> = (0..3).map(|x| IVec2::new(x, 0)).collect();
        let one_step = plan_drag(&path, &world_with_cell_ground(&[(path[0], 64), (path[1], 64), (path[2], 68)]), &City::default(), true);
        assert!(one_step.is_ok(), "one interior cell can carry one step");

        let two_steps =
            plan_drag(&path, &world_with_cell_ground(&[(path[0], 64), (path[1], 64), (path[2], 72)]), &City::default(), true);
        assert_eq!(two_steps, Err(PlanRefusal::NotEnoughRoomToClimb { steps: 2, eligible: 1 }));
    }

    /// A drag starting *on* an existing road cell continues at that cell's
    /// recorded level, not at the terrain under it — which is what stops a
    /// road extended in two drags from having a seam.
    #[test]
    fn a_drag_starting_on_an_existing_road_cell_continues_at_its_recorded_level() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 80, None, RoadPieceVariant::Surface).unwrap();

        let path: Vec<IVec2> = (0..4).map(|x| IVec2::new(x, 0)).collect();
        let world = world_with_cell_ground(&path.iter().map(|&c| (c, 64)).collect::<Vec<_>>());

        let plan = plan_drag(&path, &world, &city, false).unwrap();
        assert_eq!(levels(&plan), vec![80; 4], "the ground says 64; the road that's already there says 80");
    }

    /// A drag starting *beside* an existing road meets it flush, rather than
    /// dropping to the ground the new cells happen to sit on.
    #[test]
    fn a_drag_starting_beside_an_existing_road_cell_meets_it_flush() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(-1, 0), "dirt", 80, None, RoadPieceVariant::Surface).unwrap();

        let path: Vec<IVec2> = (0..4).map(|x| IVec2::new(x, 0)).collect();
        let world = world_with_cell_ground(&path.iter().map(|&c| (c, 64)).collect::<Vec<_>>());

        let plan = plan_drag(&path, &world, &city, false).unwrap();
        assert_eq!(levels(&plan), vec![80; 4]);
    }

    /// Joining an existing *stair* picks up the level of the edge actually
    /// being joined: its high end on the side it climbs to, its low end on
    /// the other.
    #[test]
    fn joining_a_stair_reads_the_level_of_the_edge_being_joined() {
        let mut city = City::default();
        // A stair at (-1, 0) climbing east — so its east edge (facing the
        // drag below) is four blocks above its recorded base.
        city.add_road_cell(IVec2::new(-1, 0), "dirt", 80, Some(Direction::East), RoadPieceVariant::Surface).unwrap();

        let path: Vec<IVec2> = (0..4).map(|x| IVec2::new(x, 0)).collect();
        let world = world_with_cell_ground(&path.iter().map(|&c| (c, 64)).collect::<Vec<_>>());
        let plan = plan_drag(&path, &world, &city, false).unwrap();
        assert_eq!(levels(&plan), vec![84; 4], "the stair's high end, not its recorded low end");

        // The same stair, but climbing *away* from the drag: now the edge
        // being joined is its low end, and the new road runs at the recorded
        // base instead of four blocks up.
        let mut low_side = City::default();
        low_side.add_road_cell(IVec2::new(-1, 0), "dirt", 80, Some(Direction::West), RoadPieceVariant::Surface).unwrap();
        let plan = plan_drag(&path, &world, &low_side, false).unwrap();
        assert_eq!(levels(&plan), vec![80; 4]);
    }

    /// A stair's high end sits exactly on its uphill neighbour's surface —
    /// the arithmetic the whole thing rests on, checked against
    /// `cell_write_origin` rather than restated.
    #[test]
    fn a_stairs_high_end_is_level_with_its_uphill_neighbours_surface() {
        let path: Vec<IVec2> = (0..5).map(|x| IVec2::new(x, 0)).collect();
        let mut ground: Vec<(IVec2, i32)> = path.iter().map(|&c| (c, 64)).collect();
        ground[4].1 = 68;

        let plan = plan_drag(&path, &world_with_cell_ground(&ground), &City::default(), true).unwrap();
        let stair = plan.iter().position(|c| c.ascent.is_some()).unwrap();
        let uphill = plan[stair + 1];

        // The stair piece is written from its own origin; its top step is
        // ROAD_STAIR_RISE above its bottom one.
        let stair_low_surface = cell_write_origin(plan[stair].cell, plan[stair].base_y).y + ROAD_PIECE_SUBGRADE_DEPTH;
        let stair_high_surface = stair_low_surface + ROAD_STAIR_RISE;
        let uphill_surface = cell_write_origin(uphill.cell, uphill.base_y).y + ROAD_PIECE_SUBGRADE_DEPTH;
        assert_eq!(stair_high_surface, uphill_surface);
    }

    /// `piece_for` is the one place a cell's ascent overrides its
    /// connection-derived shape — and the rotation it hands back is the one
    /// that actually points the canonical stair the right way.
    #[test]
    fn piece_for_turns_a_recorded_ascent_into_a_rotated_stair() {
        let straight = RoadConnections { north: true, south: true, ..RoadConnections::default() };
        assert_eq!(piece_for(straight, None), (RoadPieceKind::Straight, Rotation::Deg0));

        for ascent in Direction::ALL {
            let (kind, rotation) = piece_for(straight, Some(ascent));
            assert_eq!(kind, RoadPieceKind::Stair);
            assert_eq!(rotation, road::stair_rotation(ascent));
        }
    }

    /// A cell recorded as a stair is *written* as one, whatever its
    /// neighbours would otherwise have made it — the write-path half of
    /// `piece_for`.
    #[test]
    fn road_write_edit_writes_a_stair_for_a_cell_with_a_recorded_ascent() {
        let dir = temp_dir("write_stair");
        write_piece(&dir, "dirt", RoadPieceKind::Straight, &one_stone_piece());
        let mut stair_piece = one_stone_piece();
        stair_piece.palette[1] = state_named("minecraft:cobblestone");
        write_piece(&dir, "dirt", RoadPieceKind::Stair, &stair_piece);
        let (catalogue, _) = super::super::road_catalogue::load_road_catalogue_dir(&dir);

        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, -1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, Some(road::Direction::North), RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        let solid: std::collections::HashSet<&str> =
            edit.edits().iter().map(|e| e.state.name.as_str()).filter(|name| *name != "minecraft:air").collect();
        assert_eq!(
            solid,
            std::collections::HashSet::from(["minecraft:cobblestone"]),
            "a two-opposite-neighbour cell would be written as a Straight without its recorded ascent"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- tunnels (ticket 071) -----------------------------------------------

    /// [`one_stone_piece`], `height` blocks tall — the tunnel tests care
    /// about a piece's *height*, because that's what decides which layer
    /// [`cover_at`] reads (see [`piece_top_y`]), and the shipped flat pieces
    /// are five tall where `one_stone_piece` is one.
    fn piece_of_height(height: i32) -> Blueprint {
        let size = IVec3::new(ROAD_CELL_SIZE, height, ROAD_CELL_SIZE);
        let volume = (size.x * size.y * size.z) as usize;
        let mut blocks = vec![1u16; volume];
        blocks[0] = 0; // (0,0,0) stays air, so both palette entries are in use.
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state_named("minecraft:stone")],
            blocks,
            data_version: 4438,
            failed_columns: 0,
        }
    }

    /// Puts a solid block over `count` of `cell`'s 36 columns at world Y `y`
    /// — the hillside a tunnel is cut out of. Columns are roofed in
    /// [`state::road_cell_tiles`]' own order, which is arbitrary but stable;
    /// nothing about the rule cares *which* columns are covered, only how
    /// many.
    fn roof_cell(world: &mut DecodedWorld, cell: IVec2, y: i32, count: usize) {
        use crate::world::{BiomeRegistry, BlockRegistry, ChunkColumn, ChunkSection};

        let stone = world.registry.lock().unwrap().intern("minecraft:stone");
        let size = world::SECTION_SIZE as i32;
        for tile in state::road_cell_tiles(cell).take(count) {
            let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
            let column = world.columns.entry(chunk).or_insert_with(|| ChunkColumn {
                x: chunk.0,
                z: chunk.1,
                sections: Vec::new(),
                floor_y: world::WORLD_MIN_Y,
            });
            let section_y = y.div_euclid(size) as i8;
            let index = match column.sections.iter().position(|s| s.y == section_y) {
                Some(index) => index,
                None => {
                    column.sections.push(ChunkSection {
                        y: section_y,
                        blocks: Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]),
                        biomes: Box::new([BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]),
                    });
                    column.sections.len() - 1
                }
            };
            let (local_x, local_y, local_z) =
                (tile.x.rem_euclid(size) as usize, y.rem_euclid(size) as usize, tile.y.rem_euclid(size) as usize);
            column.sections[index].blocks[ChunkSection::index(local_x, local_y, local_z)] = stone;
        }
    }

    /// A `"dirt"` catalogue holding a five-tall straight and, if
    /// `with_tunnel`, a five-tall `straight-tunnel` beside it — paved in
    /// cobblestone so a write can be told apart from the surface piece's
    /// stone.
    fn tunnel_catalogue(dir: &std::path::Path, with_tunnel: bool) -> RoadCatalogue {
        write_variant_piece(dir, "dirt", RoadPieceKind::Straight, RoadPieceVariant::Surface, &piece_of_height(5));
        if with_tunnel {
            let mut tunnel = piece_of_height(5);
            tunnel.palette[1] = state_named("minecraft:cobblestone");
            write_variant_piece(dir, "dirt", RoadPieceKind::Straight, RoadPieceVariant::Tunnel, &tunnel);
        }
        let (catalogue, _skipped) = super::super::road_catalogue::load_road_catalogue_dir(dir);
        catalogue
    }

    /// A three-cell straight run at Y 64 with `cover` of the middle cell's
    /// 36 columns roofed over at the first Y its piece doesn't occupy, run
    /// through [`plan_drag`] and then [`plan_tunnels`]. Answers with the
    /// middle cell's planned variant.
    fn middle_variant(dir: &std::path::Path, cover: usize, with_tunnel: bool) -> RoadPieceVariant {
        let path: Vec<IVec2> = (0..3).map(|x| IVec2::new(x, 0)).collect();
        let mut world = world_with_cell_ground(&path.iter().map(|&c| (c, 64)).collect::<Vec<_>>());
        // A five-tall piece written from `64 - 1 - ROAD_PIECE_SUBGRADE_DEPTH`
        // tops out at 66, so 67 is the layer the rule reads.
        roof_cell(&mut world, path[1], 67, cover);

        let catalogue = tunnel_catalogue(dir, with_tunnel);
        let city = City::default();
        let mut plan = plan_drag(&path, &world, &city, false).unwrap();
        plan_tunnels(&mut plan, &path, &world, &city, Some(&catalogue), Some("dirt"));
        plan[1].variant
    }

    /// The layer the rule reads is the first one the *piece* doesn't occupy
    /// — pinned here rather than left implicit, because it's the one number
    /// that decides whether the whole feature looks at hillside or at sky.
    #[test]
    fn the_cover_reading_is_taken_directly_above_the_piece() {
        let dir = temp_dir("tunnel_top_y");
        let catalogue = tunnel_catalogue(&dir, true);
        // Origin is `64 - 1 - 1 = 62`; a five-tall piece occupies 62..=66.
        assert_eq!(cell_write_origin(IVec2::ZERO, 64).y, 62);
        assert_eq!(piece_top_y(IVec2::ZERO, 64, "dirt", RoadPieceKind::Straight, &catalogue), Some(67));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The user's rule, at its two neighbouring values: 18 of 36 is not a
    /// tunnel, 19 is. A strict majority, not "at least half" — the
    /// difference between a road that dips into a bank and one that goes
    /// under a hill.
    #[test]
    fn a_strict_majority_of_the_36_columns_being_covered_makes_a_tunnel() {
        assert_eq!(ROAD_TUNNEL_COVER_MAJORITY, 18);

        let dir = temp_dir("tunnel_threshold_18");
        assert_eq!(middle_variant(&dir, 18, true), RoadPieceVariant::Surface, "exactly half is not a majority");
        std::fs::remove_dir_all(&dir).ok();

        let dir = temp_dir("tunnel_threshold_19");
        assert_eq!(middle_variant(&dir, 19, true), RoadPieceVariant::Tunnel);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_uncovered_cell_is_never_a_tunnel() {
        let dir = temp_dir("tunnel_open_sky");
        assert_eq!(middle_variant(&dir, 0, true), RoadPieceVariant::Surface);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The [`stair_available`] rule, for tunnels: a style with no `-tunnel`
    /// export keeps building its surface pieces rather than recording cells
    /// whose blueprint can't be resolved — which [`road_write_edit`] would
    /// skip, leaving a hole in the road.
    #[test]
    fn a_style_with_no_tunnel_piece_never_records_a_tunnel_cell() {
        let dir = temp_dir("tunnel_missing_piece");
        assert_eq!(
            middle_variant(&dir, 36, false),
            RoadPieceVariant::Surface,
            "fully buried, but nothing to build it with"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The regression this ticket's design exists for. A tunnel piece carves
    /// away the cover that chose it, so a cell re-planned after its own write
    /// sees open sky. If the variant were re-derived, the next re-tile (a
    /// neighbour growing a connection) would write the surface piece back
    /// into the hillside and fill the bore in around the player.
    #[test]
    fn an_existing_tunnel_cell_keeps_its_variant_when_a_later_drag_recrosses_it() {
        let dir = temp_dir("tunnel_recross");
        let catalogue = tunnel_catalogue(&dir, true);

        let path: Vec<IVec2> = (0..3).map(|x| IVec2::new(x, 0)).collect();
        // No roof at all — this is the world *after* the tunnel was cut.
        let world = world_with_cell_ground(&path.iter().map(|&c| (c, 64)).collect::<Vec<_>>());

        let mut city = City::default();
        city.add_road_cell(path[1], "dirt", 64, None, RoadPieceVariant::Tunnel).unwrap();

        let mut plan = plan_drag(&path, &world, &city, false).unwrap();
        plan_tunnels(&mut plan, &path, &world, &city, Some(&catalogue), Some("dirt"));
        assert_eq!(plan[1].variant, RoadPieceVariant::Tunnel, "the recorded variant wins over a fresh reading");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The write-path half: a cell recorded as a tunnel resolves to the
    /// `-tunnel` blueprint, not to its kind's surface one.
    #[test]
    fn road_write_edit_writes_the_tunnel_piece_for_a_tunnel_cell() {
        let dir = temp_dir("write_tunnel");
        let catalogue = tunnel_catalogue(&dir, true);

        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, -1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Tunnel).unwrap();
        city.add_road_cell(IVec2::new(0, 1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        let solid: std::collections::HashSet<&str> =
            edit.edits().iter().map(|e| e.state.name.as_str()).filter(|name| *name != "minecraft:air").collect();
        assert_eq!(solid, std::collections::HashSet::from(["minecraft:cobblestone"]));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// [`cover_at`] counts one layer, and calls anything that isn't air
    /// cover — deliberately not `city::grid`'s clutter-skipping notion of
    /// ground, and deliberately not the terrain above or below that one
    /// layer.
    #[test]
    fn cover_at_counts_one_layer_of_not_air_and_nothing_else() {
        let cell = IVec2::new(0, 0);
        let mut world = world_with_cell_ground(&[(cell, 64)]);
        assert_eq!(cover_at(cell, 67, &world), 0);
        // The ground itself is at 63 and doesn't leak upward into the reading.
        assert_eq!(cover_at(cell, 63, &world), 36);

        roof_cell(&mut world, cell, 67, 7);
        assert_eq!(cover_at(cell, 67, &world), 7);
        assert_eq!(cover_at(cell, 68, &world), 0, "one layer up is its own question");
    }
}
