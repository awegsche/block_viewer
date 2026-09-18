//! Terraforming: dig and level (ticket 057, roadmap H1) — "reuses W4/W5
//! wholesale; it's the same write path with a different source of block
//! changes," per the roadmap's own one-line spec for this task. Originally
//! justified as the fix for a gap `city::grid`'s own docs left open — E2's
//! `fit_footprint` refused uneven ground under a footprint rather than
//! levelling it, and this ticket was named as "what would later let a
//! player fix a steeper site by hand." Ticket 058 removed that refusal
//! outright (a real world is inherently uneven; the game shouldn't block a
//! placement over it), so these tools are no longer load-bearing for
//! placement at all — they're a player's own choice to flatten a site they'd
//! rather not build into, through the write path this module deliberately
//! doesn't touch.
//!
//! ## A third tool, a drag rectangle, the same commit shape as the other two
//!
//! [`super::tool::ActiveTool::Terraform`] gates every system here exactly
//! the way [`super::tool::ActiveTool::Road`] gates `city::road_build` — see
//! that module's own docs for the shared reasoning. [`TerraformDragState`]
//! is `road_build::RoadDragState`'s shape (click starts it, release commits
//! and clears it) widened from an L-shaped cell path to a plain rectangle:
//! there's no piece catalogue or connectivity to keep cardinal here, so the
//! straightforward corner-to-corner rectangle [`rect_tiles`] walks is all a
//! dig or a level needs. [`try_commit_terraform`]/[`poll_terraform`] are the
//! same claim-nothing/apply-on-`AsyncComputeTaskPool`/poll-once shape
//! `commit`/`road_build` already use, reusing
//! [`super::commit::apply_building_edit`] directly rather than a third copy
//! of the region-cache dispatch.
//!
//! ## Not `WorldEdit::fill`
//!
//! [`crate::edit::WorldEdit::fill`]'s own doc comment already names
//! terraforming as an eventual user of "one block everywhere in a bounds" —
//! that shape fits a *uniform* edit. Neither tool here is one: a dig clears
//! a different Y at every tile (whatever that tile's own topmost block
//! happens to be), and a level writes air *or* fill depending on whether a
//! tile started above or below the target. [`dig_edit`]/[`level_edit`] build
//! a `WorldEdit` by hand, one `.set()` per position, for exactly that
//! reason.
//!
//! ## No city-state entry, no journal, no rollback to speak of
//!
//! Unlike a building or a road cell, dug/levelled terrain is not tracked in
//! [`super::state::City`] — there is nothing to claim before the write
//! starts and nothing to un-claim if it fails, so [`poll_terraform`]'s
//! failure arm is only a status line, not a rollback. No
//! [`super::journal::Journal`] entry either, so there's no undo for a
//! terraform edit yet — the same gap `road_build`'s own docs already accept
//! for a road cell ("Not journaled").
//!
//! ## Level's fill block is fixed, not chosen
//!
//! Iteration 1 has no material inventory (roadmap H2, "yields... inert" —
//! the same boundary C3 draws for production): [`FILL_BLOCK`] is always
//! `minecraft:dirt`, so raising a low tile costs nothing and looks the same
//! regardless of what was dug up to fill it. A per-material cost is H2's
//! job, once there's an inventory to spend from.
//!
//! ## No preview mesh
//!
//! `placement` and `road_build` both spawn a live translucent ghost;
//! terraforming here is console-only feedback, the same state ticket 035's
//! original paint/fill command shipped in before any tool grew a preview. A
//! rectangle is cheap to reason about without one; a follow-up ticket if
//! that turns out to be wrong in practice.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::blueprint::BlockState;
use crate::camera;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;
use crate::world;
use crate::DecodedWorld;

use super::drops::DropTable;
use super::inventory::Stock;
use super::journal::Baseline;
use super::picking::{HoveredBlock, PickingSet};
use super::tool::ActiveTool;
use super::write_status::{WriteKind, WriteStatus};

/// What a drag does — toggled by `Z` while [`ActiveTool::Terraform`] is
/// active.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerraformMode {
    #[default]
    Dig,
    Level,
}

/// The block [`level_edit`] fills a low tile with — see the module docs'
/// "Level's fill block is fixed, not chosen".
const FILL_BLOCK: &str = "minecraft:dirt";

/// Why [`level_edit`] refused a drag outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerraformError {
    /// The tile the drag *started* on isn't loaded — there's no target
    /// height to level the rest of the rectangle to. A tile elsewhere in the
    /// rectangle being unloaded is not this: [`level_edit`] just skips those
    /// (see its own docs), the same tolerant handling [`dig_edit`] gives an
    /// unloaded tile.
    NotLoaded { tile: IVec2 },
}

impl std::fmt::Display for TerraformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TerraformError::NotLoaded { tile } => {
                write!(f, "ground at ({}, {}) isn't loaded yet", tile.x, tile.y)
            }
        }
    }
}

impl std::error::Error for TerraformError {}

/// Which tile a left click started dragging from, if any — `None` between
/// drags. See the module docs' "A third tool, a drag rectangle".
#[derive(Resource, Default)]
struct TerraformDragState {
    anchor: Option<IVec2>,
}

/// A committed drag's write, in flight — the terraform counterpart of
/// `city::commit::PendingCommit`/`city::road_build::PendingRoadBuild`. Carries
/// only a tile count (for the status line): there's no `City` entry to
/// finish or roll back, so nothing else needs to survive to [`poll_terraform`].
struct PendingTerraform {
    tiles: usize,
    /// Kept for ticket 073's settlement: what the drag wrote is what it has
    /// to pay for (level's fill), and [`Baseline::capture`] needs the edit
    /// alongside the report to line that up with what it dug out. The same
    /// reason `city::commit::PendingCommit` holds its own edit.
    edit: WorldEdit,
    task: Task<Result<EditReport, EditRefusal>>,
}

/// One terraform write in flight at a time — the same single-slot
/// backpressure `city::commit::CommitState`/`city::road_build::RoadBuildState`
/// already use.
#[derive(Resource, Default)]
struct TerraformBuildState {
    pending: Option<PendingTerraform>,
}

pub struct TerraformPlugin;

impl Plugin for TerraformPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerraformMode>()
            .init_resource::<TerraformDragState>()
            .init_resource::<TerraformBuildState>()
            // Ticket 073 — same `init_resource` reasoning `WriteStatus`
            // below carries.
            .init_resource::<Stock>()
            .init_resource::<DropTable>()
            // Idempotent-either-order shape `city::commit`/`city::road_build`
            // already document for this resource.
            .init_resource::<WriteStatus>()
            .add_event::<ChunksEdited>()
            // After `PickingSet`, same reason every other per-frame reader of
            // `HoveredBlock` orders there.
            .add_systems(Update, (toggle_mode, update_drag_state, try_commit_terraform, poll_terraform).chain().after(PickingSet));
    }
}

// -----------------------------------------------------------------------------------------------
// ---- pure geometry/logic: tile rectangles, height reads, and the two edits --------------------
// -----------------------------------------------------------------------------------------------

/// Every tile in the axis-aligned rectangle `a` and `b` (inclusive) span —
/// either corner may be the greater one, and `a == b` is a single tile (a
/// plain click, no drag). Order within the rectangle isn't meaningful to any
/// caller.
fn rect_tiles(a: IVec2, b: IVec2) -> impl Iterator<Item = IVec2> {
    let (min_x, max_x) = (a.x.min(b.x), a.x.max(b.x));
    let (min_z, max_z) = (a.y.min(b.y), a.y.max(b.y));
    (min_x..=max_x).flat_map(move |x| (min_z..=max_z).map(move |z| IVec2::new(x, z)))
}

/// The world Y of the topmost non-air block at `tile`, or `None` if `tile`'s
/// chunk isn't decoded, or the column is air all the way down in what's
/// decoded. Deliberately [`world::ChunkColumn::topmost_non_air`], not
/// `city::grid`'s clutter-skipping `is_ground` — a dig or a level should
/// clear a tree or a fence post exactly like it clears stone, not read past
/// it the way a footprint fit does.
///
/// `pub(super)`: [`super::gatherer`]'s tick reuses this exact reading rather
/// than a second copy of the section/local-coordinate arithmetic — a
/// gatherer's dig is the same "what's on top right now" question this
/// module's own dig already answers.
pub(super) fn topmost_block_y(tile: IVec2, world: &DecodedWorld) -> Option<i32> {
    let size = world::SECTION_SIZE as i32;
    let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
    let column = world.columns.get(&chunk)?;
    let local_x = tile.x.rem_euclid(size) as usize;
    let local_z = tile.y.rem_euclid(size) as usize;
    column.topmost_non_air(local_x, local_z).map(|(y, _id)| y)
}

/// Clears the topmost block from every tile in `rect` that has one. A tile
/// whose chunk isn't decoded, or that's already air all the way down, is
/// silently skipped — a big drag reaching the streamed edge digs what it can
/// rather than refusing outright.
fn dig_edit(rect: &[IVec2], world: &DecodedWorld) -> WorldEdit {
    let mut edit = WorldEdit::new();
    for &tile in rect {
        if let Some(y) = topmost_block_y(tile, world) {
            edit.set(IVec3::new(tile.x, y, tile.y), BlockState::air());
        }
    }
    edit
}

/// Flattens every tile in `rect` to `anchor`'s own topmost-block height:
/// higher tiles are dug down to it (air), lower tiles are filled up to it
/// ([`FILL_BLOCK`]), and a tile already at that height is left untouched —
/// so a level of already-flat ground writes nothing. Refuses only when
/// `anchor` itself isn't loaded, since there is then no target height to
/// level to; any *other* unloaded tile in `rect` is skipped the same way
/// [`dig_edit`] skips one, not treated as a reason to refuse the whole drag.
fn level_edit(anchor: IVec2, rect: &[IVec2], world: &DecodedWorld) -> Result<WorldEdit, TerraformError> {
    let target = topmost_block_y(anchor, world).ok_or(TerraformError::NotLoaded { tile: anchor })?;
    let fill: BlockState = FILL_BLOCK.parse().expect("FILL_BLOCK is a valid block name");

    let mut edit = WorldEdit::new();
    for &tile in rect {
        let Some(top) = topmost_block_y(tile, world) else { continue };
        match top.cmp(&target) {
            std::cmp::Ordering::Greater => {
                for y in (target + 1)..=top {
                    edit.set(IVec3::new(tile.x, y, tile.y), BlockState::air());
                }
            }
            std::cmp::Ordering::Less => {
                for y in (top + 1)..=target {
                    edit.set(IVec3::new(tile.x, y, tile.y), fill.clone());
                }
            }
            std::cmp::Ordering::Equal => {}
        }
    }
    Ok(edit)
}

// -----------------------------------------------------------------------------------------------
// ---- input: mode toggle, the drag, and committing ------------------------------------------------
// -----------------------------------------------------------------------------------------------

/// `Z` flips [`TerraformMode`] while [`ActiveTool::Terraform`] is active —
/// same guard shape [`super::tool::toggle_tool`] uses for `T`, gated to the
/// terraform tool too so the key does nothing while it means nothing.
fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    egui_input: Res<camera::EguiInputCapture>,
    tool: Option<Res<ActiveTool>>,
    mut mode: ResMut<TerraformMode>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::Terraform)) || egui_input.keyboard {
        return;
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        *mode = match *mode {
            TerraformMode::Dig => TerraformMode::Level,
            TerraformMode::Level => TerraformMode::Dig,
        };
    }
}

/// Starts, tracks and clears [`TerraformDragState::anchor`] — the same
/// click/hold/release shape `road_build::update_drag_state` uses, see that
/// function's own docs.
fn update_drag_state(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_input: Res<camera::EguiInputCapture>,
    tool: Option<Res<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    mut drag: ResMut<TerraformDragState>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::Terraform)) || egui_input.pointer {
        drag.anchor = None;
        return;
    }

    if mouse.just_pressed(MouseButton::Left) {
        if let Some(hovered) = hovered.0 {
            drag.anchor = Some(IVec2::new(hovered.x, hovered.z));
        }
    }
}

/// Left-click release with the terraform tool active: builds the whole
/// drag's edit synchronously off *this* frame's [`DecodedWorld`], then
/// dispatches the write onto [`AsyncComputeTaskPool`] — see the module docs.
#[allow(clippy::too_many_arguments)]
fn try_commit_terraform(
    mouse: Res<ButtonInput<MouseButton>>,
    tool: Option<Res<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    mode: Res<TerraformMode>,
    world: Res<DecodedWorld>,
    mut build: ResMut<TerraformBuildState>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut drag: ResMut<TerraformDragState>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::Terraform)) || build.pending.is_some() || !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(anchor) = drag.anchor else { return };
    drag.anchor = None; // Cleared regardless of what happens below — see update_drag_state's docs.

    let Some(hovered) = hovered.0 else { return };
    let end = IVec2::new(hovered.x, hovered.z);
    let rect: Vec<IVec2> = rect_tiles(anchor, end).collect();

    let edit = match *mode {
        TerraformMode::Dig => dig_edit(&rect, &world),
        TerraformMode::Level => match level_edit(anchor, &rect, &world) {
            Ok(edit) => edit,
            Err(err) => {
                println!("block_viewer: level refused: {err}");
                return;
            }
        },
    };
    if edit.is_empty() {
        println!("block_viewer: terraform: nothing to change in the selected area");
        return;
    }

    let Some(region_cache) = region_cache else {
        println!("block_viewer: can't terraform: no save is loaded");
        return;
    };

    let tiles = rect.len();
    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    // `capture_replaced` (ticket 073): what a drag dug out is what it pays
    // the player, and the report is the only record of it — a terraform is
    // not journaled, so unlike a placement there is no second chance to ask.
    let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
    let task_edit = edit.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        super::commit::apply_building_edit(&mut cache, &task_edit, &policy)
    });

    build.pending = Some(PendingTerraform { tiles, edit, task });
}

/// Single non-blocking poll of the in-flight write, the same
/// `block_on(poll_once(..))` pattern `city::commit::poll_commit`/
/// `city::road_build::poll_road_build` use. There's no `City` entry to
/// finish or roll back either way — see the module docs.
fn poll_terraform(
    mut build: ResMut<TerraformBuildState>,
    mut write_status: ResMut<WriteStatus>,
    mut edited: EventWriter<ChunksEdited>,
    mut stock: ResMut<Stock>,
    drops: Res<DropTable>,
    capacity: Option<Res<super::warehouse::StorageCapacity>>,
) {
    let capacity = super::warehouse::storage_capacity(capacity.as_deref());
    let result = {
        let Some(pending) = &mut build.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still applying.
        };
        result
    };
    let PendingTerraform { tiles, edit, .. } = build.pending.take().expect("just matched Some above");

    match result {
        Ok(report) => {
            println!(
                "block_viewer: terraformed {tiles} tile(s) ({} block(s) across {} chunk(s), not yet saved to disk)",
                report.blocks_written,
                report.chunks.len()
            );
            // Ticket 073, both halves of the same rule in one place: a dig
            // writes air over stone (credit, nothing to debit), a level
            // writes dirt over air (debit, nothing to credit), and a level
            // that cuts one tile to fill another does both. `Baseline` is
            // reused purely as the "what was written where, and what was
            // there before" pairing — nothing here is journaled, the stock
            // is the only record a terraform leaves, exactly like the
            // terrain itself.
            if let Some(baseline) = Baseline::capture(&edit, &report) {
                let credited = drops.parcel_for(baseline.previous.iter().map(|(_, state)| state));
                let wanted = drops.parcel_for(baseline.written.iter().map(|(_, state)| state));
                // Ticket 079: what the city has no room for is reported, not
                // dropped in silence — a player digging out a hillside with a
                // full stock should be told the stone went nowhere.
                let overflow = stock.add_parcel_capped(&credited, capacity);
                stock.remove_parcel(&wanted);
                if !overflow.is_empty() {
                    println!("block_viewer: storage full — {overflow} could not be stored");
                }
            }
            write_status.record_success(WriteKind::Terraform, format!("{tiles} tile(s)"), &report);
            edited.send(ChunksEdited::from_report(&report));
        }
        Err(err) => {
            println!("block_viewer: terraform failed: {err}");
            write_status.record_failure(WriteKind::Terraform, format!("{tiles} tile(s)"), err.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::world::{BiomeRegistry, BlockId, BlockRegistry, ChunkColumn, ChunkSection};

    // --- rect_tiles ----------------------------------------------------------

    #[test]
    fn a_single_tile_click_is_a_one_tile_rectangle() {
        let tiles: Vec<IVec2> = rect_tiles(IVec2::new(3, 3), IVec2::new(3, 3)).collect();
        assert_eq!(tiles, vec![IVec2::new(3, 3)]);
    }

    #[test]
    fn a_rectangle_walks_every_tile_between_the_two_corners() {
        let tiles: Vec<IVec2> = rect_tiles(IVec2::new(0, 0), IVec2::new(1, 1)).collect();
        assert_eq!(tiles.len(), 4);
        for tile in [IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(0, 1), IVec2::new(1, 1)] {
            assert!(tiles.contains(&tile), "missing {tile:?}");
        }
    }

    #[test]
    fn a_rectangle_toward_negative_coordinates_still_orders_the_corners() {
        let tiles: Vec<IVec2> = rect_tiles(IVec2::new(2, 2), IVec2::new(0, 0)).collect();
        assert_eq!(tiles.len(), 9, "3x3, corners either way round");
        assert!(tiles.contains(&IVec2::new(0, 0)));
        assert!(tiles.contains(&IVec2::new(2, 2)));
    }

    // --- topmost_block_y / dig_edit / level_edit: a small synthetic world ---

    fn registry_with_stone() -> (BlockRegistry, BlockId) {
        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");
        (registry, stone)
    }

    /// A single chunk `(0, 0)`, flat stone at `y` across every tile
    /// `0..16`/`0..16` except whatever `raised`/`lowered` override.
    fn world_with_heights(base: i32, overrides: &[(IVec2, i32)]) -> DecodedWorld {
        let (registry, stone) = registry_with_stone();
        let size = world::SECTION_SIZE as i32;
        let mut column = ChunkColumn { x: 0, z: 0, sections: Vec::new(), floor_y: world::WORLD_MIN_Y };

        let mut set = |x: i32, z: i32, y: i32| {
            let section_y = y.div_euclid(size) as i8;
            let local_y = y.rem_euclid(size) as usize;
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
            column.sections[section].blocks[ChunkSection::index(x as usize, local_y, z as usize)] = stone;
        };

        for x in 0..16 {
            for z in 0..16 {
                let height = overrides.iter().find(|(tile, _)| *tile == IVec2::new(x, z)).map(|(_, y)| *y).unwrap_or(base);
                set(x, z, height);
            }
        }

        let mut columns: HashMap<(i32, i32), ChunkColumn> = HashMap::new();
        columns.insert((0, 0), column);
        DecodedWorld { registry: Arc::new(Mutex::new(registry)), biomes: Arc::new(Mutex::new(BiomeRegistry::new())), columns }
    }

    #[test]
    fn topmost_block_y_reads_the_stone_top() {
        let world = world_with_heights(64, &[]);
        assert_eq!(topmost_block_y(IVec2::new(5, 5), &world), Some(64));
    }

    #[test]
    fn topmost_block_y_is_none_for_an_undecoded_chunk() {
        let world = world_with_heights(64, &[]);
        assert_eq!(topmost_block_y(IVec2::new(500, 500), &world), None);
    }

    #[test]
    fn dig_edit_clears_the_topmost_block_at_every_tile() {
        let world = world_with_heights(64, &[]);
        let rect = vec![IVec2::new(1, 1), IVec2::new(2, 2)];
        let edit = dig_edit(&rect, &world);

        assert_eq!(edit.len(), 2);
        let by_pos: HashMap<IVec3, &BlockState> = edit.edits().iter().map(|e| (e.at, &e.state)).collect();
        assert_eq!(by_pos[&IVec3::new(1, 64, 1)].name, "minecraft:air");
        assert_eq!(by_pos[&IVec3::new(2, 64, 2)].name, "minecraft:air");
    }

    #[test]
    fn dig_edit_skips_a_tile_with_no_decoded_chunk() {
        let world = world_with_heights(64, &[]);
        let rect = vec![IVec2::new(1, 1), IVec2::new(500, 500)];
        let edit = dig_edit(&rect, &world);
        assert_eq!(edit.len(), 1, "only the loaded tile contributes an edit");
    }

    #[test]
    fn level_edit_digs_a_high_tile_down_to_the_anchor() {
        // Anchor at (0, 0), height 64; a raised tile at (1, 1), height 67.
        let world = world_with_heights(64, &[(IVec2::new(1, 1), 67)]);
        let rect = vec![IVec2::new(0, 0), IVec2::new(1, 1)];
        let edit = level_edit(IVec2::new(0, 0), &rect, &world).unwrap();

        let positions: std::collections::HashSet<IVec3> = edit.edits().iter().map(|e| e.at).collect();
        assert_eq!(positions.len(), 3, "y=65,66,67 cleared at (1,1); (0,0) untouched");
        for y in 65..=67 {
            assert!(positions.contains(&IVec3::new(1, y, 1)), "y={y} should be cleared");
            let state = edit.edits().iter().find(|e| e.at == IVec3::new(1, y, 1)).unwrap();
            assert_eq!(state.state.name, "minecraft:air");
        }
    }

    #[test]
    fn level_edit_fills_a_low_tile_up_to_the_anchor() {
        // Anchor at (0, 0), height 64; a lowered tile at (1, 1), height 61.
        let world = world_with_heights(64, &[(IVec2::new(1, 1), 61)]);
        let rect = vec![IVec2::new(0, 0), IVec2::new(1, 1)];
        let edit = level_edit(IVec2::new(0, 0), &rect, &world).unwrap();

        for y in 62..=64 {
            let state = edit.edits().iter().find(|e| e.at == IVec3::new(1, y, 1)).unwrap();
            assert_eq!(state.state.name, "minecraft:dirt", "y={y} should be filled with dirt");
        }
        assert_eq!(edit.len(), 3);
    }

    #[test]
    fn level_edit_leaves_a_tile_already_at_the_anchors_height_untouched() {
        let world = world_with_heights(64, &[]);
        let rect = vec![IVec2::new(0, 0), IVec2::new(1, 1), IVec2::new(2, 2)];
        let edit = level_edit(IVec2::new(0, 0), &rect, &world).unwrap();
        assert!(edit.is_empty(), "flat ground at the anchor's own height needs no changes");
    }

    #[test]
    fn level_edit_refuses_when_the_anchor_tile_is_unloaded() {
        let world = world_with_heights(64, &[]);
        let rect = vec![IVec2::new(500, 500)];
        let err = level_edit(IVec2::new(500, 500), &rect, &world).unwrap_err();
        assert_eq!(err, TerraformError::NotLoaded { tile: IVec2::new(500, 500) });
    }

    #[test]
    fn level_edit_skips_an_unloaded_tile_that_is_not_the_anchor() {
        let world = world_with_heights(64, &[]);
        let rect = vec![IVec2::new(0, 0), IVec2::new(500, 500)];
        let edit = level_edit(IVec2::new(0, 0), &rect, &world).unwrap();
        assert!(edit.is_empty(), "the anchor is flat and the far tile is just skipped, not a refusal");
    }

    // --- poll_terraform: the WriteStatus/ChunksEdited glue, through a real App -
    //
    // Same split `city::commit`/`city::road_build`'s own tests use: the pure
    // edit-building logic (above) is tested directly; `apply_building_edit`
    // itself is proven once by `city::commit`'s own fixture tests (this
    // module reuses that exact function); what's left to prove here is the
    // `WriteStatus`/`ChunksEdited` glue, through a task whose result is fixed
    // ahead of time.

    use bevy::tasks::TaskPool;

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn terraform_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(TerraformPlugin);
        app
    }

    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if app.world().resource::<TerraformBuildState>().pending.is_none() {
                return;
            }
        }
        panic!("terraform write never settled");
    }

    #[test]
    fn poll_terraform_success_fires_chunks_edited_and_records_the_write() {
        let mut app = terraform_test_app();
        let report = EditReport { blocks_written: 4, chunks: vec![(0, 0)], regions: vec![(0, 0)], replaced: None, ..Default::default() };
        let task = pool().spawn(async move { Ok(report) });
        app.world_mut().resource_mut::<TerraformBuildState>().pending = Some(PendingTerraform { tiles: 4, edit: WorldEdit::new(), task });

        run_until_settled(&mut app);

        assert!(app.world().resource::<TerraformBuildState>().pending.is_none());
        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].chunks(), vec![(0, 0)]);

        let write_status = app.world().resource::<WriteStatus>();
        assert!(matches!(write_status.last(), Some(super::super::write_status::LastWrite::Success(_))));
    }

    // --- ticket 073: a drag is paid for and paid out ------------------------

    /// A settled drag: `written` is what the edit put down, `replaced` what
    /// it took away, exactly as the write path reports them.
    fn run_terraform(app: &mut App, written: &[(IVec3, &str)], replaced: &[(IVec3, &str)]) {
        let mut edit = WorldEdit::new();
        for (at, name) in written {
            edit.set(*at, name.parse().unwrap());
        }
        let report = EditReport {
            blocks_written: written.len(),
            chunks: vec![(0, 0)],
            regions: vec![(0, 0)],
            replaced: Some(replaced.iter().map(|(at, name)| (*at, name.parse().unwrap())).collect()), ..Default::default()
        };
        let task = pool().spawn(async move { Ok(report) });
        app.world_mut().resource_mut::<TerraformBuildState>().pending =
            Some(PendingTerraform { tiles: written.len(), edit, task });
        run_until_settled(app);
    }

    #[test]
    fn digging_credits_what_it_dug_out() {
        let mut app = terraform_test_app();
        run_terraform(
            &mut app,
            &[(IVec3::new(0, 64, 0), "minecraft:air"), (IVec3::new(1, 64, 0), "minecraft:air")],
            &[(IVec3::new(0, 64, 0), "minecraft:dirt"), (IVec3::new(1, 64, 0), "minecraft:dirt")],
        );

        let stock = app.world().resource::<Stock>();
        assert_eq!(stock.count("minecraft:dirt"), 2);
        assert_eq!(stock.count("minecraft:air"), 0, "writing air costs nothing — air is not a material");
    }

    #[test]
    fn levelling_pays_for_the_dirt_it_fills_with() {
        let mut app = terraform_test_app();
        app.world_mut().resource_mut::<Stock>().add("minecraft:dirt", 5);

        // One tile cut down to the anchor, one filled up to it — the two
        // halves of a level drag, and the two halves of the rule.
        run_terraform(
            &mut app,
            &[(IVec3::new(0, 64, 0), "minecraft:air"), (IVec3::new(1, 63, 0), "minecraft:dirt")],
            &[(IVec3::new(0, 64, 0), "minecraft:stone"), (IVec3::new(1, 63, 0), "minecraft:air")],
        );

        let stock = app.world().resource::<Stock>();
        assert_eq!(stock.count("minecraft:dirt"), 4, "one dirt went into the ground");
        assert_eq!(stock.count("minecraft:stone"), 1, "and one stone came out of it");
    }

    #[test]
    fn a_fill_the_stock_cannot_cover_is_clamped_rather_than_refused() {
        // Same call `city::demolish` makes: the blocks are already in the
        // world by the time this settles, so the ledger follows the world
        // rather than the other way round.
        let mut app = terraform_test_app();
        run_terraform(&mut app, &[(IVec3::new(0, 64, 0), "minecraft:dirt")], &[(IVec3::new(0, 64, 0), "minecraft:air")]);

        assert!(app.world().resource::<Stock>().is_empty());
    }

    #[test]
    fn poll_terraform_failure_records_a_failure_and_fires_nothing() {
        let mut app = terraform_test_app();
        let task = pool().spawn(async { Err(EditRefusal::Empty) });
        app.world_mut().resource_mut::<TerraformBuildState>().pending = Some(PendingTerraform { tiles: 2, edit: WorldEdit::new(), task });

        run_until_settled(&mut app);

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "nothing changed in the world, so nothing needs re-meshing");

        let write_status = app.world().resource::<WriteStatus>();
        assert!(matches!(write_status.last(), Some(super::super::write_status::LastWrite::Failed { .. })));
    }
}
