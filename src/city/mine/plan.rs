//! The survey read and the per-slice plan (ticket 115, `MINES_DESIGN.md`).
//! No Bevy: a [`Box3`] in, a [`Blueprint`] out ([`survey`]); a
//! [`SliceGeometry`] plus what [`survey`] read, in, a [`WorldEdit`] out
//! ([`plan_slice`]). Ticket 116 is the only caller that touches Bevy or a
//! real region cache — everything here is tested against a synthetic
//! [`BlockSampler`].
//!
//! ## Why this reads through `blueprint::extract`, not `DecodedWorld`
//!
//! `city::gatherer` reads terrain from `DecodedWorld` because a hut only
//! ever digs at or above the citybuilder's render floor, which is exactly
//! what that resource holds decoded. A mine can't: everything below 030's
//! render floor is never decoded there. So [`survey`] is a thin wrapper
//! over [`extract_blueprint_locked`] — the same region-cache read
//! `ranvil-cli get-area` and the blueprint export tool already use, just
//! handed a `&mut RegionCache` instead of an `Arc<Mutex<..>>` so 116's job
//! (which already holds the lock for the write that follows) never has to
//! lock twice.
//!
//! ## Idempotence
//!
//! **A position whose sampled state already equals its target is not
//! written**, compared as a full [`BlockState`] (name and properties) —
//! [`write_if_needed`] is the one place that comparison happens, so every
//! rule below gets it for free. This is what makes a lost `mines.ron`
//! harmless (116 re-plans from the top and finds its own tunnels already
//! dug, at zero cost) and what makes [`SliceOutcome::Void`] mean "nothing
//! was ever here" rather than "already dug": a gallery slice's floor tiles
//! already reading [`GROUND`] is the tell that distinguishes the two (see
//! [`plan_level_slice`]).

use std::ops::RangeInclusive;

use bevy::math::IVec3;

use crate::blueprint::{extract_blueprint_locked, BlockState, Blueprint, ExtractError, ExtractProgress};
use crate::city::definition::Mine;
use crate::city::drops::AIR_NAMES;
use crate::city::grid::is_clutter_name;
use crate::edit::WorldEdit;
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;

use super::layout::{Box3, MineFrame, ShaftBlock, Side, SliceGeometry, TorchSpot, BAND, GROUND, PILLAR, STAIRS, TORCH, WALL_TORCH};
use super::progress::SliceOutcome;

// --- the survey ------------------------------------------------------------

/// Reads `bounds` out of the region cache — the mine's one and only read
/// path, `extract_blueprint_locked` under a synthetic [`ExtractProgress`]
/// nobody reads back (a slice survey is a handful of columns; there's
/// nothing worth a progress bar for). `Blueprint::block_at` — dead code
/// since ticket 022 — gets its first real caller through [`BlockSampler`].
pub fn survey(bounds: &Box3, cache: &mut RegionCache) -> Result<Blueprint, ExtractError> {
    let selection = SelectionBounds::from_corners(bounds.min, bounds.min, bounds.max);
    let progress = ExtractProgress::default();
    extract_blueprint_locked(selection, cache, &progress)
}

// --- classification ----------------------------------------------------

/// Fluid names a mine seals rather than digs through — the same three
/// `drops.ron`'s `nothing` list opens with, kept here as a small const
/// rather than a read of that file (see the module docs' "why this reads
/// through `blueprint::extract`" for the analogous call on the read side).
const FLUID_NAMES: [&str; 3] = ["minecraft:water", "minecraft:lava", "minecraft:bubble_column"];

const BEDROCK: &str = "minecraft:bedrock";

/// What one sampled block counts as for slice planning —
/// `MINES_DESIGN.md`'s rules read off one of these, never off a raw name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    Air,
    Fluid,
    Clutter,
    Solid,
    Ore,
    Bedrock,
    Unknown,
}

/// Classifies one sampled block. `None` (outside the surveyed box, or a
/// column the extraction couldn't read) is [`Material::Unknown`] — treated
/// as solid wherever a rule asks "is the wall solid?", and never written by
/// [`write_if_needed`] regardless of what a rule would otherwise target,
/// because a position nobody could actually read isn't safe to overwrite.
pub fn classify(state: Option<&BlockState>, mine: &Mine) -> Material {
    let Some(state) = state else {
        return Material::Unknown;
    };
    let name = state.name.as_str();
    if AIR_NAMES.contains(&name) {
        return Material::Air;
    }
    if FLUID_NAMES.contains(&name) {
        return Material::Fluid;
    }
    if name == BEDROCK {
        return Material::Bedrock;
    }
    if mine.is_valuable(name) {
        return Material::Ore;
    }
    let bare = name.strip_prefix("minecraft:").unwrap_or(name);
    if is_clutter_name(bare) {
        return Material::Clutter;
    }
    Material::Solid
}

/// A source of sampled blocks: [`Blueprint`] in real use, a
/// `HashMap<IVec3, BlockState>` in every test in this module.
pub trait BlockSampler {
    fn block(&self, at: IVec3) -> Option<&BlockState>;
}

impl BlockSampler for Blueprint {
    fn block(&self, at: IVec3) -> Option<&BlockState> {
        self.block_at(at)
    }
}

// --- the plan ----------------------------------------------------------

/// What [`plan_slice`] resolves one [`super::layout::Slice`] to.
pub struct SlicePlan {
    pub edit: WorldEdit,
    /// Blocks *removed*: every excavated/ore block that wasn't already air
    /// or a fluid. Duplicated inside `outcome`'s `Dug` case — the two are
    /// always equal — because the outcome is what `MineProgress::advance`
    /// consumes and the plain field is what a caller charging `dig_carry`
    /// wants without matching on the enum first.
    pub cost: u32,
    pub outcome: SliceOutcome,
}

fn empty_plan(outcome: SliceOutcome) -> SlicePlan {
    SlicePlan { edit: WorldEdit::new(), cost: 0, outcome }
}

/// Writes `target` at `pos` unless `sampled` already reads as `target` (the
/// idempotence rule the whole design leans on — see the module docs) or
/// `sampled` is `None` — a position the survey couldn't read is never
/// touched, whatever a rule would otherwise put there.
fn write_if_needed(edit: &mut WorldEdit, sampled: Option<&BlockState>, pos: IVec3, target: BlockState) {
    match sampled {
        Some(state) if *state == target => {}
        None => {}
        Some(_) => {
            edit.set(pos, target);
        }
    }
}

fn ground_state() -> BlockState {
    BlockState { name: GROUND.to_string(), properties: Vec::new() }
}

fn pillar_state() -> BlockState {
    BlockState { name: PILLAR.to_string(), properties: vec![("axis".to_string(), "y".to_string())] }
}

fn band_state() -> BlockState {
    BlockState { name: BAND.to_string(), properties: vec![("axis".to_string(), "y".to_string())] }
}

fn torch_state() -> BlockState {
    BlockState { name: TORCH.to_string(), properties: Vec::new() }
}

fn wall_torch_state(facing: Side) -> BlockState {
    BlockState { name: WALL_TORCH.to_string(), properties: vec![("facing".to_string(), side_name(facing).to_string())] }
}

fn stair_state(facing: Side) -> BlockState {
    BlockState {
        name: STAIRS.to_string(),
        properties: vec![
            ("facing".to_string(), side_name(facing).to_string()),
            ("half".to_string(), "bottom".to_string()),
            ("shape".to_string(), "straight".to_string()),
            ("waterlogged".to_string(), "false".to_string()),
        ],
    }
}

fn side_name(side: Side) -> &'static str {
    match side {
        Side::North => "north",
        Side::East => "east",
        Side::South => "south",
        Side::West => "west",
    }
}

/// Resolves one [`super::layout::Slice`]'s [`SliceGeometry`] to the
/// [`WorldEdit`] that digs it, against `sampler` — real terrain
/// ([`Blueprint`], via [`survey`]) or a synthetic one in tests.
///
/// `slice` picks which of the design's three rule sets applies (a Sink
/// resolves every survey position through [`MineFrame::shaft_target`]; a
/// Secondary or Gallery slice excavates/floors/scans); `frame` and
/// `new_bottom` are only read for a Sink (`new_bottom` is the bottom
/// *after* this flight there, and otherwise unused — a Secondary/Gallery
/// slice's level floor is already baked into `geometry.floor_y`).
pub fn plan_slice(
    slice: super::layout::Slice,
    geometry: &SliceGeometry,
    frame: &MineFrame,
    new_bottom: i32,
    mine: &Mine,
    sampler: &impl BlockSampler,
) -> SlicePlan {
    match slice {
        super::layout::Slice::Sink => plan_sink(geometry, frame, new_bottom, mine, sampler),
        super::layout::Slice::Secondary { .. } | super::layout::Slice::Gallery { .. } => {
            plan_level_slice(geometry, frame, mine, sampler)
        }
    }
}

/// A Secondary or Gallery slice: `MINES_DESIGN.md`'s "A mining level",
/// ticket 115's rules 1-6 (gallery) with rule 3's "always `GROUND`" and
/// rule 5's pillars swapped in for a secondary slice (`geometry.scan_for_ore`
/// tells the two apart — `true` only for a gallery).
fn plan_level_slice(geometry: &SliceGeometry, frame: &MineFrame, mine: &Mine, sampler: &impl BlockSampler) -> SlicePlan {
    // Rule 1: bedrock anywhere in `excavate` refuses the whole slice.
    if geometry.excavate.iter().any(|pos| classify(sampler.block(pos), mine) == Material::Bedrock) {
        return empty_plan(SliceOutcome::Bedrock);
    }

    let mut edit = WorldEdit::new();
    let mut cost = 0u32;
    let mut excavate_all_air = true;

    // Rule 2: excavate -> air.
    for pos in geometry.excavate.iter() {
        let sampled = sampler.block(pos);
        let mat = classify(sampled, mine);
        if mat != Material::Air {
            excavate_all_air = false;
        }
        match mat {
            Material::Air | Material::Unknown => {}
            Material::Fluid => write_if_needed(&mut edit, sampled, pos, BlockState::air()),
            Material::Solid | Material::Ore | Material::Clutter => {
                cost += 1;
                write_if_needed(&mut edit, sampled, pos, BlockState::air());
            }
            Material::Bedrock => unreachable!("checked above"),
        }
    }

    // Rule 3: floor tiles.
    let mut floor_already_ground = true;
    for f in &geometry.floor {
        let pos = IVec3::new(f.x, geometry.floor_y, f.y);
        let sampled = sampler.block(pos);
        if sampled != Some(&ground_state()) {
            floor_already_ground = false;
        }
        let mat = classify(sampled, mine);
        if geometry.scan_for_ore {
            // Gallery: solid stays; anything else (including ore) is sealed.
            match mat {
                Material::Solid | Material::Unknown => {}
                Material::Ore => {
                    cost += 1;
                    write_if_needed(&mut edit, sampled, pos, ground_state());
                }
                _ => write_if_needed(&mut edit, sampled, pos, ground_state()),
            }
        } else {
            // Secondary: floor is always ground.
            if mat == Material::Ore {
                cost += 1;
            }
            write_if_needed(&mut edit, sampled, pos, ground_state());
        }
    }

    // Rule 4: the scan shell (`survey` minus `excavate` minus the floor
    // tiles already resolved above).
    for pos in geometry.survey.iter() {
        if geometry.excavate.contains(pos) {
            continue;
        }
        if pos.y == geometry.floor_y && geometry.floor.iter().any(|f| f.x == pos.x && f.y == pos.z) {
            continue;
        }
        let sampled = sampler.block(pos);
        match classify(sampled, mine) {
            Material::Fluid => write_if_needed(&mut edit, sampled, pos, ground_state()),
            Material::Ore if geometry.scan_for_ore => {
                cost += 1;
                let target = if pos.y == geometry.floor_y { ground_state() } else { BlockState::air() };
                write_if_needed(&mut edit, sampled, pos, target);
            }
            _ => {}
        }
    }

    // Rule 5: lighting/ornament.
    if let Some(torch) = &geometry.torch {
        plan_gallery_torch(&mut edit, mine, sampler, torch, geometry.floor_y);
    }
    for &(pos, has_torch) in &geometry.pillars {
        write_if_needed(&mut edit, sampler.block(pos), pos, pillar_state());
        if has_torch {
            plan_pillar_torch(&mut edit, frame, sampler, pos);
        }
    }

    // Rule 6: void vs dug.
    let outcome = if cost == 0 && excavate_all_air && !floor_already_ground {
        SliceOutcome::Void
    } else {
        SliceOutcome::Dug { cost }
    };

    SlicePlan { edit, cost, outcome }
}

fn plan_gallery_torch(edit: &mut WorldEdit, mine: &Mine, sampler: &impl BlockSampler, torch: &TorchSpot, floor_y: i32) {
    let wall_mat = classify(sampler.block(torch.wall), mine);
    let backs_wall_torch = matches!(wall_mat, Material::Solid | Material::Unknown | Material::Fluid);
    if backs_wall_torch {
        let pos = IVec3::new(torch.fallback_floor.x, floor_y + 2, torch.fallback_floor.z);
        write_if_needed(edit, sampler.block(pos), pos, wall_torch_state(torch.facing));
    } else {
        let pos = torch.fallback_floor;
        write_if_needed(edit, sampler.block(pos), pos, torch_state());
    }
}

/// A secondary pillar's wall torch sits one block toward the corridor from
/// the pillar itself, facing back out at it — which side that is follows
/// from which wall (east or west of `frame`'s interior) the pillar sits on.
fn plan_pillar_torch(edit: &mut WorldEdit, frame: &MineFrame, sampler: &impl BlockSampler, pillar_pos: IVec3) {
    let interior_x = frame.interior_x();
    let (corridor_x, facing) = if pillar_pos.x == *interior_x.end() + 1 {
        (pillar_pos.x - 1, Side::West)
    } else {
        (pillar_pos.x + 1, Side::East)
    };
    let pos = IVec3::new(corridor_x, pillar_pos.y, pillar_pos.z);
    write_if_needed(edit, sampler.block(pos), pos, wall_torch_state(facing));
}

/// The primary shaft sinking one flight: every position in `geometry.survey`
/// resolves through [`MineFrame::shaft_target`] against `new_bottom` —
/// `MINES_DESIGN.md`'s "Sinking" table.
fn plan_sink(geometry: &SliceGeometry, frame: &MineFrame, new_bottom: i32, mine: &Mine, sampler: &impl BlockSampler) -> SlicePlan {
    if sink_has_bedrock(frame, geometry.survey, mine, sampler) {
        return empty_plan(SliceOutcome::Bedrock);
    }

    let mut edit = WorldEdit::new();
    let mut cost = 0u32;
    let torch_spacing = mine.torch_spacing as i32;

    for pos in geometry.survey.iter() {
        let sampled = sampler.block(pos);
        let mat = classify(sampled, mine);
        match frame.shaft_target(new_bottom, torch_spacing, pos) {
            ShaftBlock::Untouched => {}
            ShaftBlock::Air | ShaftBlock::Doorway => {
                if matches!(mat, Material::Solid | Material::Ore | Material::Clutter) {
                    cost += 1;
                }
                write_if_needed(&mut edit, sampled, pos, BlockState::air());
            }
            ShaftBlock::Ground => {
                if matches!(mat, Material::Solid | Material::Ore) {
                    cost += 1;
                }
                write_if_needed(&mut edit, sampled, pos, ground_state());
            }
            ShaftBlock::Stair { facing } => {
                if matches!(mat, Material::Solid | Material::Ore) {
                    cost += 1;
                }
                write_if_needed(&mut edit, sampled, pos, stair_state(facing));
            }
            ShaftBlock::Pillar => write_if_needed(&mut edit, sampled, pos, pillar_state()),
            ShaftBlock::Band => write_if_needed(&mut edit, sampled, pos, band_state()),
            ShaftBlock::SealIfNotSolid => {
                if matches!(mat, Material::Air | Material::Fluid | Material::Clutter) {
                    write_if_needed(&mut edit, sampled, pos, ground_state());
                }
            }
            ShaftBlock::WallTorch { facing } => {
                write_if_needed(&mut edit, sampled, pos, wall_torch_state(facing));
            }
        }
    }

    SlicePlan { edit, cost, outcome: SliceOutcome::Dug { cost } }
}

/// Bedrock anywhere in the ring or interior of `survey`'s box (never the
/// lining — rock behind a wall that's simply never excavated doesn't stop a
/// sink) refuses the whole flight.
fn sink_has_bedrock(frame: &MineFrame, survey: Box3, mine: &Mine, sampler: &impl BlockSampler) -> bool {
    let y_range: RangeInclusive<i32> = survey.min.y..=survey.max.y;

    for tile in frame.ring() {
        for y in y_range.clone() {
            if classify(sampler.block(IVec3::new(tile.tile.x, y, tile.tile.y)), mine) == Material::Bedrock {
                return true;
            }
        }
    }
    for x in frame.interior_x() {
        for z in frame.interior_z() {
            for y in y_range.clone() {
                if classify(sampler.block(IVec3::new(x, y, z)), mine) == Material::Bedrock {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bevy::math::{IVec2, IVec3};

    use crate::city::definition::ShaftAt;
    use crate::city::mine::layout::{Arm, GallerySide, Slice};

    use super::*;

    impl BlockSampler for HashMap<IVec3, BlockState> {
        fn block(&self, at: IVec3) -> Option<&BlockState> {
            self.get(&at)
        }
    }

    fn stone() -> BlockState {
        BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() }
    }

    fn ore(name: &str) -> BlockState {
        BlockState { name: format!("minecraft:{name}"), properties: Vec::new() }
    }

    fn water() -> BlockState {
        BlockState { name: "minecraft:water".to_string(), properties: Vec::new() }
    }

    fn bedrock() -> BlockState {
        BlockState { name: "minecraft:bedrock".to_string(), properties: Vec::new() }
    }

    fn cave_air() -> BlockState {
        BlockState { name: "minecraft:cave_air".to_string(), properties: Vec::new() }
    }

    fn mine_with(shaft_size: u32, first_level_depth: u32, min_level_y: i32) -> Mine {
        Mine {
            shaft: ShaftAt { x: 5, z: 5 },
            shaft_size,
            first_level_depth,
            min_level_y,
            level_reach: 100,
            gallery_length: 200,
            torch_spacing: 8,
            max_void_run: 6,
            blocks_per_minute: 60.0,
            buffer_stacks: 512,
            haul_at_stacks: Some(64),
            valuables: Vec::new(),
        }
    }

    fn frame() -> MineFrame {
        MineFrame { shaft_min: IVec2::new(100, 200), shaft_size: 6, floor_y: 64, first_level_depth: 12 }
    }

    /// Fills every position of `boxes` with `stone()` — the "solid rock
    /// everywhere" sampler most rule tests start from.
    fn solid_sampler(boxes: &[Box3]) -> HashMap<IVec3, BlockState> {
        let mut map = HashMap::new();
        for b in boxes {
            for pos in b.iter() {
                map.insert(pos, stone());
            }
        }
        map
    }

    // --- classify ----------------------------------------------------------

    #[test]
    fn classify_sorts_every_material() {
        let mine = mine_with(6, 12, 16);
        assert_eq!(classify(Some(&BlockState::air()), &mine), Material::Air);
        assert_eq!(classify(Some(&water()), &mine), Material::Fluid);
        assert_eq!(classify(Some(&bedrock()), &mine), Material::Bedrock);
        assert_eq!(classify(Some(&ore("iron_ore")), &mine), Material::Ore);
        assert_eq!(classify(Some(&BlockState { name: "minecraft:torch".to_string(), properties: Vec::new() }), &mine), Material::Clutter);
        assert_eq!(classify(Some(&stone()), &mine), Material::Solid);
        assert_eq!(classify(None, &mine), Material::Unknown);
    }

    #[test]
    fn classify_reads_mine_valuables_in_addition_to_ore_suffix() {
        let mut mine = mine_with(6, 12, 16);
        mine.valuables.push("ancient_debris".to_string());
        let state = BlockState { name: "minecraft:ancient_debris".to_string(), properties: Vec::new() };
        assert_eq!(classify(Some(&state), &mine), Material::Ore);
    }

    // --- gallery slice rules -------------------------------------------

    fn gallery_geometry(f: &MineFrame, mine: &Mine, distance: i32) -> (SliceGeometry, Slice) {
        let level = f.level(0);
        let slice = Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance };
        let geom = super::super::layout::slice_geometry(f, mine, f.floor_y, Some(&level), slice);
        (geom, slice)
    }

    #[test]
    fn bedrock_in_excavate_refuses_the_slice() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        let mut sampler = solid_sampler(&[geom.survey]);
        let excavate_pos = geom.excavate.iter().next().unwrap();
        sampler.insert(excavate_pos, bedrock());

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert_eq!(plan.outcome, SliceOutcome::Bedrock);
        assert!(plan.edit.is_empty());
        assert_eq!(plan.cost, 0);
    }

    #[test]
    fn solid_gallery_slice_costs_six_and_writes_six_airs() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        let sampler = solid_sampler(&[geom.survey]);

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert_eq!(plan.cost, 6);
        assert_eq!(plan.outcome, SliceOutcome::Dug { cost: 6 });
        let airs = plan.edit.edits().iter().filter(|e| e.state == BlockState::air()).count();
        assert_eq!(airs, 6);
    }

    #[test]
    fn three_ores_in_the_scan_shell_add_three_to_cost() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        let mut sampler = solid_sampler(&[geom.survey]);
        let mut placed = 0;
        for pos in geom.survey.iter() {
            if geom.excavate.contains(pos) {
                continue;
            }
            if placed < 3 {
                sampler.insert(pos, ore("iron_ore"));
                placed += 1;
            }
        }
        assert_eq!(placed, 3);

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert_eq!(plan.cost, 9);
    }

    #[test]
    fn ore_in_the_floor_becomes_ground_ore_in_the_ceiling_becomes_air() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        let mut sampler = solid_sampler(&[geom.survey]);
        let floor_pos = IVec3::new(geom.excavate.min.x, geom.floor_y, geom.floor.first().unwrap().y);
        let ceiling_pos = IVec3::new(geom.excavate.min.x, geom.floor_y + 4, geom.floor.first().unwrap().y);
        sampler.insert(floor_pos, ore("iron_ore"));
        sampler.insert(ceiling_pos, ore("iron_ore"));

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        let floor_write = plan.edit.edits().iter().find(|e| e.at == floor_pos).unwrap();
        assert_eq!(floor_write.state, ground_state());
        let ceiling_write = plan.edit.edits().iter().find(|e| e.at == ceiling_pos).unwrap();
        assert_eq!(ceiling_write.state, BlockState::air());
    }

    #[test]
    fn water_in_scan_is_sealed_free_water_in_excavate_is_cleared_free() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        let mut sampler = solid_sampler(&[geom.survey]);
        let excavate_pos = geom.excavate.iter().next().unwrap();
        sampler.insert(excavate_pos, water());
        let scan_pos = geom.survey.iter().find(|p| !geom.excavate.contains(*p) && p.y != geom.floor_y).unwrap();
        sampler.insert(scan_pos, water());

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert_eq!(plan.cost, 5); // 6 excavate cells minus the one water cell
        let excavate_write = plan.edit.edits().iter().find(|e| e.at == excavate_pos).unwrap();
        assert_eq!(excavate_write.state, BlockState::air());
        let scan_write = plan.edit.edits().iter().find(|e| e.at == scan_pos).unwrap();
        assert_eq!(scan_write.state, ground_state());
    }

    /// The state `edit` leaves at `pos` after every write is applied in
    /// order — [`WorldEdit`]'s own "last write wins" rule, since a torch's
    /// corridor cell is written once by the excavate rule (to air) and
    /// again by the torch rule, and only the second is what a real write
    /// would leave behind.
    fn resolved(edit: &WorldEdit, pos: IVec3) -> Option<BlockState> {
        edit.edits().iter().filter(|e| e.at == pos).next_back().map(|e| e.state.clone())
    }

    #[test]
    fn torch_falls_back_to_standing_when_the_wall_is_a_cave() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        // distance 4 is a torch distance (torch_spacing 8, mid-interval).
        let (geom, slice) = gallery_geometry(&f, &mine, 4);
        let torch = geom.torch.expect("distance 4 should carry a torch");
        let mut sampler = solid_sampler(&[geom.survey]);
        sampler.insert(torch.wall, cave_air());

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert_eq!(resolved(&plan.edit, torch.fallback_floor), Some(torch_state()));
        let wall_pos = IVec3::new(torch.fallback_floor.x, geom.floor_y + 2, torch.fallback_floor.z);
        assert_eq!(resolved(&plan.edit, wall_pos), Some(BlockState::air()));
    }

    #[test]
    fn torch_backs_onto_a_solid_wall_as_a_wall_torch() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 4);
        let torch = geom.torch.expect("distance 4 should carry a torch");
        let sampler = solid_sampler(&[geom.survey]);

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        let wall_pos = IVec3::new(torch.fallback_floor.x, geom.floor_y + 2, torch.fallback_floor.z);
        assert_eq!(resolved(&plan.edit, wall_pos), Some(wall_torch_state(torch.facing)));
        // Ticket 126: the block it hangs on is not one this slice writes to
        // air — `gallery_geometry` uses the north arm, where a torch spot
        // built from `row_z[0]` put the wall inside the gallery.
        assert!(!geom.excavate.contains(torch.wall));
        assert_eq!(resolved(&plan.edit, torch.wall), None, "the torch wall must stay rock");
    }

    // --- secondary slice rules -------------------------------------------

    fn secondary_geometry(f: &MineFrame, mine: &Mine, distance: i32) -> (SliceGeometry, Slice) {
        let level = f.level(0);
        let slice = Slice::Secondary { arm: Arm::North, distance };
        let geom = super::super::layout::slice_geometry(f, mine, f.floor_y, Some(&level), slice);
        (geom, slice)
    }

    #[test]
    fn secondary_slice_is_twelve_airs_four_ground_and_pillars_when_due() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        // distance 0 (stagger 0) is a pillar distance.
        let (geom, slice) = secondary_geometry(&f, &mine, 0);
        assert!(!geom.pillars.is_empty());
        let sampler = solid_sampler(&[geom.survey]);

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        let airs = plan.edit.edits().iter().filter(|e| e.state == BlockState::air()).count();
        assert_eq!(airs, 12);
        let grounds = plan.edit.edits().iter().filter(|e| e.state == ground_state() && e.at.y == geom.floor_y).count();
        assert_eq!(grounds, 4);
        let pillars = plan.edit.edits().iter().filter(|e| e.state == pillar_state()).count();
        assert_eq!(pillars, geom.pillars.len());
    }

    #[test]
    fn secondary_floor_becomes_ground_even_over_solid_stone() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = secondary_geometry(&f, &mine, 0);
        let sampler = solid_sampler(&[geom.survey]);

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        for tile in &geom.floor {
            let pos = IVec3::new(tile.x, geom.floor_y, tile.y);
            let write = plan.edit.edits().iter().find(|e| e.at == pos).unwrap();
            assert_eq!(write.state, ground_state());
        }
    }

    // --- sink slice rules -------------------------------------------------

    #[test]
    fn sink_from_floor_y_through_solid_rock_matches_the_design() {
        let f = frame();
        let mine = mine_with(6, 4, -60); // level_spacing 4, matches first flight
        let slice = Slice::Sink;
        let geom = super::super::layout::slice_geometry(&f, &mine, f.floor_y, None, slice);
        let new_bottom = f.floor_y - f.level_spacing();
        let sampler = solid_sampler(&[geom.survey]);

        let plan = plan_slice(slice, &geom, &f, new_bottom, &mine, &sampler);
        assert_eq!(plan.outcome, SliceOutcome::Dug { cost: plan.cost });
        assert!(plan.cost > 0);

        // The interior at y = new_bottom + 1 ..= floor_y - 1 goes to air.
        for y in (new_bottom + 1)..f.floor_y {
            for x in f.interior_x() {
                for z in f.interior_z() {
                    let pos = IVec3::new(x, y, z);
                    let write = plan.edit.edits().iter().find(|e| e.at == pos);
                    assert_eq!(write.map(|w| &w.state), Some(&BlockState::air()), "interior {pos:?}");
                }
            }
        }

        // Four pillars run the lining corners for every y in this flight.
        for l in f.lining().filter(|l| l.corner) {
            for y in (new_bottom + 1)..f.floor_y {
                let pos = IVec3::new(l.tile.x, y, l.tile.y);
                let write = plan.edit.edits().iter().find(|e| e.at == pos);
                assert_eq!(write.map(|w| &w.state), Some(&pillar_state()), "corner {pos:?}");
            }
        }
    }

    #[test]
    fn replanning_a_dug_slice_is_free_and_never_void() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        let mut sampler = solid_sampler(&[geom.survey]);
        // The floor starts out non-solid (a seam of water) so the first dig
        // marks it GROUND — the idempotence tell rule 6 leans on to tell a
        // resumed dig apart from a genuine void (`MINES_DESIGN.md`'s
        // "Idempotence"). A floor that started solid never gets written at
        // all (rule 3 keeps it), so it carries no such tell; that's a
        // known gap in the design's own check, not this planner's.
        for tile in &geom.floor {
            sampler.insert(IVec3::new(tile.x, geom.floor_y, tile.y), water());
        }
        let first = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert!(matches!(first.outcome, SliceOutcome::Dug { .. }));

        // Apply the plan's edits onto a fresh sampler standing in for the
        // now-dug terrain, then re-plan the same slice against it.
        let mut dug = HashMap::new();
        for e in first.edit.edits() {
            dug.insert(e.at, e.state.clone());
        }
        // Positions the plan didn't touch (already matching their target,
        // e.g. a survey cell that started as air) stay solid rock in this
        // synthetic sampler except where the plan wrote something — fill
        // the rest with whatever the original sampler had.
        for pos in geom.survey.iter() {
            dug.entry(pos).or_insert_with(stone);
        }

        let second = plan_slice(slice, &geom, &f, f.floor_y, &mine, &dug);
        assert_eq!(second.cost, 0);
        assert_eq!(second.outcome, SliceOutcome::Dug { cost: 0 });
        assert!(second.edit.is_empty());
    }

    #[test]
    fn a_genuine_cave_crossing_is_void() {
        let f = frame();
        let mine = mine_with(6, 12, 16);
        let (geom, slice) = gallery_geometry(&f, &mine, 3);
        // Open air everywhere, floor left natural (air, not ground) — a
        // cave the gallery breaks into rather than ground it dug itself.
        let mut sampler = HashMap::new();
        for pos in geom.survey.iter() {
            sampler.insert(pos, BlockState::air());
        }

        let plan = plan_slice(slice, &geom, &f, f.floor_y, &mine, &sampler);
        assert_eq!(plan.cost, 0);
        assert_eq!(plan.outcome, SliceOutcome::Void);
    }

    #[test]
    fn survey_matches_extracts_own_block_at_one_coordinate() {
        // No region cache/fixture available in this module's tests (those
        // live in `blueprint::extract`); this exercises `survey`'s bounds
        // conversion and `BlockSampler for Blueprint` against a Blueprint
        // built in memory instead of a real save.
        let bounds = Box3::new(IVec3::new(0, 0, 0), IVec3::new(1, 1, 1));
        let blueprint = Blueprint {
            size: IVec3::new(2, 2, 2),
            origin: IVec3::new(0, 0, 0),
            palette: vec![BlockState::air(), stone()],
            blocks: vec![1, 0, 0, 0, 0, 0, 0, 0],
            data_version: 1,
            failed_columns: 0,
        };
        assert_eq!(blueprint.block(bounds.min), Some(&stone()));
        assert_eq!(blueprint.block(IVec3::new(1, 1, 1)), Some(&BlockState::air()));
    }
}
