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
//! eviction needed). No real `.nbt` road pieces ship yet for any style (see
//! ticket 054's own "no real assets yet"), so in practice every cell falls
//! back to [`quad_mesh`]: one flat, unrotated plane per cell, tinted the same
//! green/red [`super::placement`] uses — proof the drag mechanic and its
//! validity signal work end to end before there's real geometry to show.
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
use super::road::{self, RoadConnections, RoadPieceKind};
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
    piece_meshes: HashMap<(String, RoadPieceKind, Rotation), Option<Handle<Mesh>>>,
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
/// sample in this module starts here.
fn cell_min_corner(cell: IVec2) -> IVec3 {
    IVec3::new(cell.x * ROAD_CELL_SIZE, 0, cell.y * ROAD_CELL_SIZE)
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

/// Resolves (and caches) the preview mesh for `(style, kind, rotation)`: the
/// real piece from `catalogue`, meshed via B2/B3's own path, if one has
/// loaded — [`quad_mesh`] otherwise (including when `style` is `None`: a
/// cell new to this drag with nothing selected yet still needs *some*
/// preview). The quad itself is cached once, not per style/kind — every
/// fallback is the same flat square.
#[allow(clippy::too_many_arguments)]
fn preview_mesh(
    preview: &mut RoadPreviewState,
    style: Option<&str>,
    kind: RoadPieceKind,
    rotation: Rotation,
    catalogue: Option<&RoadCatalogue>,
    atlas: &world::AtlasUvIndex,
    world: &DecodedWorld,
    color_maps: &world::ColorMaps,
    meshes: &mut Assets<Mesh>,
) -> Handle<Mesh> {
    let Some(style) = style else {
        return preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone();
    };
    let Some(piece) = catalogue.and_then(|c| c.get(style, kind)) else {
        return preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone();
    };

    let key = (style.to_string(), kind, rotation);
    if let Some(cached) = preview.piece_meshes.get(&key) {
        if let Some(handle) = cached {
            return handle.clone();
        }
        // A cached rotation failure for a *real* piece: still show
        // something rather than nothing.
        return preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone();
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
                println!("block_viewer: road preview: {style}/{kind:?} can't rotate to {rotation:?}: {err}");
                preview.piece_meshes.insert(key, None);
                return preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone();
            }
        }
    };

    let biome = plains_biome_colors(world, color_maps);
    let handle = blueprint::mesh_blueprint(blueprint, atlas, biome).map(|mesh| meshes.add(mesh));
    preview.piece_meshes.insert(key, handle.clone());
    handle.unwrap_or_else(|| preview.quad_mesh.get_or_insert_with(|| meshes.add(quad_mesh())).clone())
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

/// World-space transform for a cell's preview at Minecraft `(x, z)` origin
/// and world-Y `height` — the same `bevy.z = -mc.z` translation
/// `placement::ghost_transform` uses, centred on the cell rather than at its
/// minimum corner (the quad/piece mesh is centred on its own origin).
fn cell_transform(cell: IVec2, height: i32) -> Transform {
    let corner = cell_min_corner(cell);
    let half = ROAD_CELL_SIZE as f32 / 2.0;
    Transform::from_xyz(corner.x as f32 + half, height as f32, -(corner.z as f32 + half))
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
    for (index, &cell) in path.iter().enumerate() {
        let Some(height) = cell_height(cell, &world) else { continue };
        let valid = cell_valid(cell, &world, &city);
        let style = style_for_cell(cell, &city, selected_style);
        let (kind, rotation) = road::select_piece(connections_with_path(&city, &path, cell));
        let mesh = preview_mesh(
            &mut preview,
            style,
            kind,
            rotation,
            catalogue.as_deref(),
            &atlas.0,
            &world,
            &color_maps.0,
            &mut meshes,
        );
        let ghost_materials = ensure_materials(&mut preview, &terrain_material.0, &mut materials);
        let material = if valid { ghost_materials.valid.clone() } else { ghost_materials.invalid.clone() };
        let transform = cell_transform(cell, height);

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

    for &entity in preview.entities.iter().skip(path.len()) {
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
        let Some(style) = city.road_style_at(cell) else { continue };
        let (kind, rotation) = road::select_piece(road::connections_at(city, cell));
        let Some(piece) = catalogue.get(style, kind) else { continue };

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
                    println!("block_viewer: road build: {style}/{kind:?} can't rotate to {rotation:?}, skipping cell: {err}");
                    continue;
                }
            }
        };

        let corner = cell_min_corner(cell);
        let edit = blueprint_edit(blueprint, corner);
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

    let newly_added: Vec<IVec2> = path.iter().copied().filter(|&cell| !city.is_road_cell(cell)).collect();
    for &cell in &path {
        // Already validated above; `add_road_cell` only fails on occupancy,
        // which `cell_valid` just confirmed clear (or already-road, which is
        // idempotent) — see the module docs' "Committing".
        if let Err(err) = city.add_road_cell(cell, build_style.clone()) {
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
        // is nothing to mesh or write yet. See the module docs' "No real
        // assets" (ticket 054) note.
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
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();
        assert!(cell_occupancy_ok(IVec2::new(0, 0), &city), "already-road counts as ok, not blocked");
    }

    #[test]
    fn cell_occupancy_ok_is_false_over_a_building() {
        let mut city = City::default();
        city.place_building("house01", IVec3::new(0, 64, 0), crate::blueprint::Rotation::Deg0, IVec2::new(2, 2)).unwrap();
        assert!(!cell_occupancy_ok(IVec2::new(0, 0), &city));
    }

    #[test]
    fn affected_cells_includes_the_path_and_its_already_road_neighbours() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(-1, 0), "dirt").unwrap(); // west neighbour of (0, 0)
        city.add_road_cell(IVec2::new(5, 5), "dirt").unwrap(); // unrelated, far away

        let affected = affected_cells(&[IVec2::new(0, 0)], &city);
        assert!(affected.contains(&IVec2::new(0, 0)));
        assert!(affected.contains(&IVec2::new(-1, 0)));
        assert!(!affected.contains(&IVec2::new(5, 5)));
        assert_eq!(affected.len(), 2);
    }

    #[test]
    fn affected_cells_does_not_duplicate_a_neighbour_shared_by_two_path_cells() {
        let mut city = City::default();
        city.add_road_cell(IVec2::new(1, 1), "dirt").unwrap();

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
        city.add_road_cell(IVec2::new(0, -1), "dirt").unwrap(); // north of (0, 0)
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

    /// Writes `blueprint` at `piece_path(dir, style, kind)`, creating the
    /// style subdirectory first — same reason
    /// `road_catalogue::tests::write_piece` needs to.
    fn write_piece(dir: &std::path::Path, style: &str, kind: RoadPieceKind, blueprint: &Blueprint) {
        std::fs::create_dir_all(dir.join(style)).expect("should create style dir");
        write_structure_file(&piece_path(dir, style, kind), blueprint).unwrap();
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
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();

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
        city.add_road_cell(IVec2::new(0, 0), "paved").unwrap();

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
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0)], &catalogue, &city);
        assert!(edit.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn road_write_edit_offsets_each_cells_piece_by_its_own_corner() {
        let dir = temp_dir("write_edit_offset");
        let catalogue = catalogue_with_isolated(&dir);
        let mut city = City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();
        city.add_road_cell(IVec2::new(2, 0), "dirt").unwrap();

        let edit = road_write_edit(&[IVec2::new(0, 0), IVec2::new(2, 0)], &catalogue, &city);
        let positions: std::collections::HashSet<IVec3> = edit.edits().iter().map(|e| e.at).collect();
        // Cell (2, 0)'s corner is (12, 0, 0) — two cells over.
        assert!(positions.contains(&IVec3::new(0, 0, 0)));
        assert!(positions.contains(&IVec3::new(2 * ROAD_CELL_SIZE, 0, 0)));
        std::fs::remove_dir_all(&dir).ok();
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
        city.add_road_cell(IVec2::new(0, -1), "dirt").unwrap();
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();
        city.add_road_cell(IVec2::new(0, 1), "dirt").unwrap();

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
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();
        city.add_road_cell(IVec2::new(100, 100), "paved").unwrap();

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
        app.world_mut().resource_mut::<City>().add_road_cell(IVec2::new(0, 0), "dirt").unwrap();

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
            city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap(); // pre-existing neighbour, not newly added
            city.add_road_cell(IVec2::new(1, 0), "dirt").unwrap(); // this drag's own new cell
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
        city.add_road_cell(IVec2::new(0, 0), "dirt").unwrap();
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
}
