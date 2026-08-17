//! Rotating a [`Blueprint`] about Y (ticket 038, roadmap B3).
//!
//! The roadmap draws the line this module lives on: **the mesh part is a
//! transform** — spin the spawned entity's [`Transform`](bevy::prelude::Transform)
//! in 90° steps, no new geometry needed, [`crate::blueprint::mesh`] doesn't
//! change. **The blocks part is not** — a stair's `facing`, a log's `axis`,
//! a sign's 16-way `rotation`, and a fence/wall/pane's `north`/`south`/
//! `east`/`west` connections are strings in the palette that encode a
//! *world* direction. Turning the mesh without rewriting them leaves the
//! geometry pointing one way and the block data pointing the old way — wrong
//! the moment the rotated blueprint is written into a save.
//!
//! [`rotate_blueprint`] does both halves that *do* need rewriting: the block
//! grid (a size and index remap) and the palette (a property rewrite, once
//! per distinct [`BlockState`] rather than once per block instance, the same
//! "resolve the small palette, not the big grid" shape [`super::mesh`] uses).
//!
//! ## The rule for what's in the table
//!
//! Every property either has a rewrite rule below, is explicitly whitelisted
//! as direction-independent, or makes rotation fail with
//! [`RotationError::UnrotatableProperty`] naming the block, key and value.
//! There is deliberately no fourth option ("leave it as-is and hope") — a
//! rotated building with one silently-wrong property reads as correct until
//! someone notices a door swinging the wrong way, which is exactly the
//! failure the roadmap calls out.

use std::collections::HashMap;

use bevy::math::IVec3;
use serde::{Deserialize, Serialize};

use super::{BlockState, Blueprint};

/// One of the four Y-axis orientations a blueprint can be placed at.
///
/// `Deg0` is the identity — it clones the blueprint without touching the
/// palette at all, so a blueprint carrying a property this table doesn't
/// recognise still loads and previews at its as-authored orientation.
///
/// Derives `Serialize`/`Deserialize` for ticket 043 (roadmap D2): a placed
/// building's orientation on disk is exactly this type, not a mirror enum
/// invented in the persistence module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rotation {
    Deg0,
    Deg90,
    Deg180,
    Deg270,
}

impl Rotation {
    /// How many 90° clockwise (viewed from above: north -> east -> south ->
    /// west) turns this rotation is, for the geometry and property remaps
    /// below — both are written once against this count rather than three
    /// times against named cases.
    fn quarter_turns(self) -> u8 {
        match self {
            Rotation::Deg0 => 0,
            Rotation::Deg90 => 1,
            Rotation::Deg180 => 2,
            Rotation::Deg270 => 3,
        }
    }
}

/// Why [`rotate_blueprint`] refused a blueprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RotationError {
    /// A palette entry carries a property this table has no rewrite rule
    /// and no whitelist entry for. Detected rather than silently carried
    /// over unrotated (and therefore wrong) — see the module docs.
    UnrotatableProperty {
        block: String,
        key: String,
        value: String,
    },
}

impl std::fmt::Display for RotationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RotationError::UnrotatableProperty { block, key, value } => write!(
                f,
                "{block}: don't know how to rotate {key}={value} — add it to \
                 blueprint::rotate's table"
            ),
        }
    }
}

// --- cardinal directions -----------------------------------------------

/// The four horizontal directions, in clockwise order — `facing`'s
/// horizontal values, and the key set `north`/`south`/`east`/`west`
/// properties are named after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cardinal {
    North,
    East,
    South,
    West,
}

const CARDINAL_ORDER: [Cardinal; 4] = [Cardinal::North, Cardinal::East, Cardinal::South, Cardinal::West];

impl Cardinal {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "north" => Some(Cardinal::North),
            "east" => Some(Cardinal::East),
            "south" => Some(Cardinal::South),
            "west" => Some(Cardinal::West),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Cardinal::North => "north",
            Cardinal::East => "east",
            Cardinal::South => "south",
            Cardinal::West => "west",
        }
    }

    /// This direction after `turns` 90° clockwise turns.
    fn rotate(self, turns: u8) -> Self {
        let index = CARDINAL_ORDER.iter().position(|&c| c == self).unwrap();
        CARDINAL_ORDER[(index + turns as usize) % 4]
    }
}

// --- grid geometry -------------------------------------------------------

/// `size` after one 90° clockwise turn about Y — X and Z swap, Y is
/// untouched (rotation is about Y, not a general 3D rotation).
fn rotate90_size(size: IVec3) -> IVec3 {
    IVec3::new(size.z, size.y, size.x)
}

/// `pos` (valid inside `size`) after one 90° clockwise turn about Y.
/// `size` is the shape `pos` is valid in *before* the turn.
fn rotate90_pos(size: IVec3, pos: IVec3) -> IVec3 {
    IVec3::new(size.z - 1 - pos.z, pos.y, pos.x)
}

/// `size` and `pos` after `turns` 90° clockwise turns, composed rather than
/// hand-derived per turn count — one transform checked by construction
/// instead of three formulas (90/180/270) that could each be wrong in a
/// different way.
fn rotate_size_and_pos(mut size: IVec3, mut pos: IVec3, turns: u8) -> (IVec3, IVec3) {
    for _ in 0..turns {
        pos = rotate90_pos(size, pos);
        size = rotate90_size(size);
    }
    (size, pos)
}

/// Index into a size-`size` grid in [`Blueprint::blocks`]'s Y-outer /
/// Z-middle / X-inner order — matches `blueprint::mesh::block_at`'s layout.
fn grid_index(size: IVec3, pos: IVec3) -> usize {
    let (sx, sz) = (size.x as usize, size.z as usize);
    (pos.y as usize) * sz * sx + (pos.z as usize) * sx + (pos.x as usize)
}

// --- property rewrite ----------------------------------------------------

/// Rail `shape` values that curve or run diagonally-adjacent — these need a
/// full rewrite under rotation, unlike a stair's `shape` (see
/// [`STAIR_SHAPES`]). The two sets don't overlap, so the property is
/// disambiguated by its *value*, not by which block carries it.
const RAIL_STRAIGHT_SHAPES: [(&str, Cardinal, Cardinal); 2] =
    [("north_south", Cardinal::North, Cardinal::South), ("east_west", Cardinal::East, Cardinal::West)];
const RAIL_ASCENDING_PREFIX: &str = "ascending_";
const RAIL_CURVES: [(&str, Cardinal, Cardinal); 4] = [
    ("north_east", Cardinal::North, Cardinal::East),
    ("north_west", Cardinal::North, Cardinal::West),
    ("south_east", Cardinal::South, Cardinal::East),
    ("south_west", Cardinal::South, Cardinal::West),
];

/// A stair/slab-shape value — relative to the block's own `facing`, so it
/// passes through a rotation unchanged (rotating the whole structure keeps
/// "inner-left relative to facing" true, whatever `facing` rotates to).
const STAIR_SHAPES: [&str; 5] = ["straight", "inner_left", "inner_right", "outer_left", "outer_right"];

/// Properties that don't encode a horizontal direction, so a rotation
/// leaves them exactly as they were.
///
/// `hinge` and `type` (a chest's `left`/`right`/`single`) are here
/// deliberately rather than by accident: both are defined *relative to the
/// block's own `facing`*, the same way a stair's `shape` is. The roadmap
/// names `hinge` as a property "rotation" has to get right — rewriting
/// `facing` while leaving `hinge` alone is what gets it right, since a
/// whole-structure turn preserves left/right-relative-to-facing; only a
/// mirror (not offered here — see the ticket's out-of-scope) would need to
/// flip it.
const DIRECTION_INDEPENDENT_PROPERTIES: &[&str] = &[
    "half",
    "type",
    "hinge",
    "face",
    "waterlogged",
    "powered",
    "lit",
    "triggered",
    "enabled",
    "extended",
    "open",
    "locked",
    "disarmed",
    "attached",
    "conditional",
    "in_wall",
    "snowy",
    "persistent",
    "distance",
    "age",
    "stage",
    "moisture",
    "level",
    "layers",
    "note",
    "instrument",
    "delay",
    "occupied",
    "drag",
    "eye",
    "has_book",
    "has_record",
    "has_bottle_0",
    "has_bottle_1",
    "has_bottle_2",
    "cracked",
    "crafting",
    "count",
    "charges",
    "bites",
    "honey_level",
    "pickles",
    "eggs",
    "hatch",
    "short",
    "up",
    "down",
    "thickness",
    "tilt",
    "leaves",
    "segment_amount",
    "berries",
    "bottom",
    "shrieking",
    "can_summon",
    "vertical_direction",
    "sculk_sensor_phase",
];

/// The `north`/`south`/`east`/`west` property keys, present as a group on
/// fences, walls, glass panes, iron bars and redstone wire. A quarter turn
/// moves *which key* holds a given value — the value that was at `north` is
/// now the `east` key's value — the same cycle [`Cardinal::rotate`] applies
/// to a `facing` value.
const CONNECTION_KEYS: [&str; 4] = ["north", "south", "east", "west"];

/// Rewrites one `(key, value)` pair that isn't one of the grouped
/// `north`/`south`/`east`/`west` connection keys (those are handled
/// together by [`rotate_properties`], not here).
fn rotate_single_property(
    block: &str,
    key: &str,
    value: &str,
    turns: u8,
) -> Result<String, RotationError> {
    let unrotatable = || RotationError::UnrotatableProperty {
        block: block.to_string(),
        key: key.to_string(),
        value: value.to_string(),
    };

    match key {
        "facing" => match value {
            "up" | "down" => Ok(value.to_string()),
            _ => Cardinal::parse(value)
                .map(|c| c.rotate(turns).as_str().to_string())
                .ok_or_else(unrotatable),
        },
        "axis" => match value {
            "y" => Ok("y".to_string()),
            "x" if turns % 2 == 1 => Ok("z".to_string()),
            "z" if turns % 2 == 1 => Ok("x".to_string()),
            "x" | "z" => Ok(value.to_string()),
            _ => Err(unrotatable()),
        },
        "rotation" => value
            .parse::<u8>()
            .ok()
            .filter(|n| *n < 16)
            .map(|n| ((n as u16 + 4 * turns as u16) % 16).to_string())
            .ok_or_else(unrotatable),
        "shape" => rotate_shape(value, turns).ok_or_else(unrotatable),
        _ if DIRECTION_INDEPENDENT_PROPERTIES.contains(&key) => Ok(value.to_string()),
        _ => Err(unrotatable()),
    }
}

/// `shape`'s rewrite: stair-family values pass through (relative to
/// `facing`, see [`STAIR_SHAPES`]); rail straights/curves/ascents get the
/// same clockwise cycle as `facing`. `None` means the value is neither —
/// unrotatable.
fn rotate_shape(value: &str, turns: u8) -> Option<String> {
    if STAIR_SHAPES.contains(&value) {
        return Some(value.to_string());
    }
    if let Some(dir) = value.strip_prefix(RAIL_ASCENDING_PREFIX).and_then(Cardinal::parse) {
        return Some(format!("{RAIL_ASCENDING_PREFIX}{}", dir.rotate(turns).as_str()));
    }
    for &(name, a, b) in RAIL_STRAIGHT_SHAPES.iter().chain(RAIL_CURVES.iter()) {
        if value == name {
            return Some(named_pair(a.rotate(turns), b.rotate(turns)));
        }
    }
    None
}

/// The rail-shape name for an unordered pair of directions — matches
/// vanilla's naming (`north_east`, not `east_north`), by checking against
/// the known table rather than reconstructing the convention from scratch.
fn named_pair(a: Cardinal, b: Cardinal) -> String {
    for &(name, x, y) in RAIL_STRAIGHT_SHAPES.iter().chain(RAIL_CURVES.iter()) {
        if (a, b) == (x, y) || (a, b) == (y, x) {
            return name.to_string();
        }
    }
    unreachable!("rotating a valid direction pair always lands on another valid pair")
}

/// Rewrites every property on one [`BlockState`], grouping the
/// `north`/`south`/`east`/`west` connection keys (if present) rather than
/// rewriting them independently — see [`CONNECTION_KEYS`].
fn rotate_properties(state: &BlockState, turns: u8) -> Result<Vec<(String, String)>, RotationError> {
    let by_key: HashMap<&str, &str> =
        state.properties.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

    let mut out = Vec::with_capacity(state.properties.len());
    for (key, value) in &state.properties {
        if CONNECTION_KEYS.contains(&key.as_str()) {
            // The new value at `key` is whatever sat at the direction that
            // rotates *forward* into `key` — i.e. `key`'s source key turned
            // `turns` steps back.
            let dest = Cardinal::parse(key).expect("key is one of CONNECTION_KEYS");
            let source = dest.rotate((4 - turns % 4) % 4);
            if let Some(&value) = by_key.get(source.as_str()) {
                out.push((key.clone(), value.to_string()));
            }
            // A connection key absent from the source direction (shouldn't
            // happen in practice — vanilla always emits all four together)
            // is simply dropped, same as it would've been absent already.
            continue;
        }
        out.push((key.clone(), rotate_single_property(&state.name, key, value, turns)?));
    }
    out.sort();
    Ok(out)
}

/// Rewrites one palette entry for a `turns`-quarter-turn rotation.
fn rotate_block_state(state: &BlockState, turns: u8) -> Result<BlockState, RotationError> {
    Ok(BlockState {
        name: state.name.clone(),
        properties: rotate_properties(state, turns)?,
    })
}

// --- public entry point ---------------------------------------------------

/// Rotates a whole [`Blueprint`] `rotation` about Y: the block grid (size
/// and index remap) and every palette entry's rotation-sensitive
/// properties.
///
/// `Rotation::Deg0` is a plain clone — no palette rewrite, so a blueprint
/// with a property this table doesn't recognise still loads unrotated (see
/// the module docs).
///
/// Errors rather than guessing when a palette entry carries a property with
/// no rewrite rule and no whitelist entry — see [`RotationError`].
pub fn rotate_blueprint(blueprint: &Blueprint, rotation: Rotation) -> Result<Blueprint, RotationError> {
    let turns = rotation.quarter_turns();
    if turns == 0 {
        return Ok(blueprint.clone());
    }

    let mut new_palette = Vec::with_capacity(blueprint.palette.len());
    for state in &blueprint.palette {
        new_palette.push(rotate_block_state(state, turns)?);
    }

    let (new_size, _) = rotate_size_and_pos(blueprint.size, IVec3::ZERO, turns);
    let mut new_blocks = vec![0u16; blueprint.blocks.len()];
    for y in 0..blueprint.size.y {
        for z in 0..blueprint.size.z {
            for x in 0..blueprint.size.x {
                let pos = IVec3::new(x, y, z);
                let old_index = grid_index(blueprint.size, pos);
                let (_, new_pos) = rotate_size_and_pos(blueprint.size, pos, turns);
                new_blocks[grid_index(new_size, new_pos)] = blueprint.blocks[old_index];
            }
        }
    }

    Ok(Blueprint {
        size: new_size,
        origin: blueprint.origin,
        palette: new_palette,
        blocks: new_blocks,
        data_version: blueprint.data_version,
        failed_columns: blueprint.failed_columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(name: &str) -> BlockState {
        name.parse().unwrap()
    }

    fn blueprint_with(size: IVec3, palette: Vec<BlockState>, blocks: Vec<u16>) -> Blueprint {
        Blueprint {
            size,
            origin: IVec3::new(5, 6, 7),
            palette,
            blocks,
            data_version: 3953,
            failed_columns: 0,
        }
    }

    // --- geometry ---------------------------------------------------------

    #[test]
    fn deg0_is_a_plain_clone() {
        // A property this table can't rotate — Deg0 must not error on it.
        let unrotatable = BlockState {
            name: "minecraft:made_up_block".to_string(),
            properties: vec![("orientation".to_string(), "north_up".to_string())],
        };
        let blueprint = blueprint_with(IVec3::new(2, 1, 3), vec![unrotatable], vec![0; 6]);
        let rotated = rotate_blueprint(&blueprint, Rotation::Deg0).unwrap();
        assert_eq!(rotated, blueprint);
    }

    #[test]
    fn a_90_degree_turn_swaps_x_and_z_size() {
        let blueprint = blueprint_with(IVec3::new(2, 3, 5), vec![BlockState::air()], vec![0; 30]);
        let rotated = rotate_blueprint(&blueprint, Rotation::Deg90).unwrap();
        assert_eq!(rotated.size, IVec3::new(5, 3, 2));
    }

    #[test]
    fn a_180_degree_turn_keeps_size() {
        let blueprint = blueprint_with(IVec3::new(2, 3, 5), vec![BlockState::air()], vec![0; 30]);
        let rotated = rotate_blueprint(&blueprint, Rotation::Deg180).unwrap();
        assert_eq!(rotated.size, blueprint.size);
    }

    /// A marked corner block ends up where a 90° clockwise turn (viewed from
    /// above: north -> east -> south -> west) puts it. `size = (2, 1, 3)`
    /// (X=2, Z=3); the marker sits at the north-west corner (x=0, z=0).
    #[test]
    fn a_marked_corner_lands_where_a_90_degree_turn_puts_it() {
        let palette = vec![BlockState::air(), state("minecraft:stone")];
        let size = IVec3::new(2, 1, 3);
        let mut blocks = vec![0u16; 6]; // Y-outer/Z-middle/X-inner, all air.
        blocks[grid_index(size, IVec3::new(0, 0, 0))] = 1; // the marker, at NW.
        let blueprint = blueprint_with(size, palette, blocks);

        let rotated = rotate_blueprint(&blueprint, Rotation::Deg90).unwrap();
        assert_eq!(rotated.size, IVec3::new(3, 1, 2));
        // NW corner rotates to NE: new size is (3, 1, 2), NE is (x=2, z=0).
        let marker_index = grid_index(rotated.size, IVec3::new(2, 0, 0));
        assert_eq!(rotated.blocks[marker_index], 1);
        // Every other cell is still air.
        assert_eq!(rotated.blocks.iter().filter(|&&b| b == 1).count(), 1);
    }

    #[test]
    fn four_quarter_turns_are_the_identity() {
        let palette = vec![BlockState::air(), state("minecraft:oak_stairs[facing=north,half=bottom]")];
        let size = IVec3::new(2, 1, 3);
        let mut blocks = vec![0u16; 6];
        blocks[grid_index(size, IVec3::new(1, 0, 2))] = 1;
        let blueprint = blueprint_with(size, palette, blocks);

        let mut rotated = blueprint.clone();
        for _ in 0..4 {
            rotated = rotate_blueprint(&rotated, Rotation::Deg90).unwrap();
        }
        assert_eq!(rotated, blueprint);
    }

    // --- facing -------------------------------------------------------

    #[test]
    fn facing_cycles_clockwise() {
        let s = state("minecraft:oak_stairs[facing=north,half=bottom]");
        let r = rotate_block_state(&s, 1).unwrap();
        assert_eq!(r.properties, vec![("facing".to_string(), "east".to_string()), ("half".to_string(), "bottom".to_string())]);
    }

    #[test]
    fn facing_up_and_down_pass_through() {
        for value in ["up", "down"] {
            let s = BlockState { name: "minecraft:dropper".to_string(), properties: vec![("facing".to_string(), value.to_string())] };
            let r = rotate_block_state(&s, 1).unwrap();
            assert_eq!(r.properties[0].1, value);
        }
    }

    // --- axis -----------------------------------------------------------

    #[test]
    fn axis_x_and_z_swap_on_an_odd_turn() {
        let s = state("minecraft:oak_log[axis=x]");
        assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, "z");
        assert_eq!(rotate_block_state(&s, 3).unwrap().properties[0].1, "z");
    }

    #[test]
    fn axis_passes_through_on_an_even_turn() {
        let s = state("minecraft:oak_log[axis=x]");
        assert_eq!(rotate_block_state(&s, 2).unwrap().properties[0].1, "x");
    }

    #[test]
    fn axis_y_always_passes_through() {
        let s = state("minecraft:oak_log[axis=y]");
        for turns in 0..4 {
            assert_eq!(rotate_block_state(&s, turns).unwrap().properties[0].1, "y");
        }
    }

    // --- rotation (signs/banners) ----------------------------------------

    #[test]
    fn sixteen_way_rotation_advances_by_four_per_quarter_turn() {
        let s = state("minecraft:oak_sign[rotation=0]");
        assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, "4");
        assert_eq!(rotate_block_state(&s, 2).unwrap().properties[0].1, "8");
        assert_eq!(rotate_block_state(&s, 3).unwrap().properties[0].1, "12");
    }

    #[test]
    fn sixteen_way_rotation_wraps_around() {
        let s = state("minecraft:oak_sign[rotation=14]");
        assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, "2");
    }

    // --- connection keys (fences, walls, panes, bars, redstone) ----------

    #[test]
    fn connection_keys_rotate_as_a_group() {
        let s = BlockState {
            name: "minecraft:oak_fence".to_string(),
            properties: vec![
                ("east".to_string(), "false".to_string()),
                ("north".to_string(), "true".to_string()),
                ("south".to_string(), "false".to_string()),
                ("west".to_string(), "false".to_string()),
            ],
        };
        // north=true should move to east=true after a 90-degree turn.
        let r = rotate_block_state(&s, 1).unwrap();
        let get = |k: &str| r.properties.iter().find(|(key, _)| key == k).unwrap().1.clone();
        assert_eq!(get("east"), "true");
        assert_eq!(get("north"), "false");
        assert_eq!(get("south"), "false");
        assert_eq!(get("west"), "false");
    }

    #[test]
    fn connection_keys_survive_a_full_rotation_cycle() {
        let s = BlockState {
            name: "minecraft:iron_bars".to_string(),
            properties: vec![
                ("east".to_string(), "true".to_string()),
                ("north".to_string(), "false".to_string()),
                ("south".to_string(), "true".to_string()),
                ("west".to_string(), "false".to_string()),
            ],
        };
        let mut r = s.clone();
        for _ in 0..4 {
            r = rotate_block_state(&r, 1).unwrap();
        }
        assert_eq!(r, s);
    }

    // --- shape: stairs pass through, rails rewrite ------------------------

    #[test]
    fn stair_shapes_pass_through_unchanged() {
        for shape in STAIR_SHAPES {
            let s = BlockState {
                name: "minecraft:oak_stairs".to_string(),
                properties: vec![("shape".to_string(), shape.to_string())],
            };
            assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, shape);
        }
    }

    #[test]
    fn rail_straight_shapes_rotate() {
        let s = BlockState { name: "minecraft:rail".to_string(), properties: vec![("shape".to_string(), "north_south".to_string())] };
        assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, "east_west");
        assert_eq!(rotate_block_state(&s, 2).unwrap().properties[0].1, "north_south");
    }

    #[test]
    fn rail_curves_cycle_clockwise() {
        let s = BlockState { name: "minecraft:rail".to_string(), properties: vec![("shape".to_string(), "north_east".to_string())] };
        assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, "south_east");
        assert_eq!(rotate_block_state(&s, 2).unwrap().properties[0].1, "south_west");
        assert_eq!(rotate_block_state(&s, 3).unwrap().properties[0].1, "north_west");
    }

    #[test]
    fn rail_ascending_shapes_cycle_clockwise() {
        let s = BlockState { name: "minecraft:powered_rail".to_string(), properties: vec![("shape".to_string(), "ascending_north".to_string())] };
        assert_eq!(rotate_block_state(&s, 1).unwrap().properties[0].1, "ascending_east");
    }

    // --- direction-independent whitelist -----------------------------------

    #[test]
    fn hinge_passes_through_unchanged() {
        let s = BlockState {
            name: "minecraft:oak_door".to_string(),
            properties: vec![("facing".to_string(), "north".to_string()), ("hinge".to_string(), "left".to_string())],
        };
        let r = rotate_block_state(&s, 1).unwrap();
        let get = |k: &str| r.properties.iter().find(|(key, _)| key == k).unwrap().1.clone();
        assert_eq!(get("facing"), "east", "facing rotates");
        assert_eq!(get("hinge"), "left", "hinge is relative to facing, so it doesn't");
    }

    #[test]
    fn waterlogged_and_half_pass_through_unchanged() {
        let s = BlockState {
            name: "minecraft:oak_slab".to_string(),
            properties: vec![("type".to_string(), "top".to_string()), ("waterlogged".to_string(), "true".to_string())],
        };
        let r = rotate_block_state(&s, 1).unwrap();
        assert_eq!(r, s);
    }

    // --- unknown properties: detected, not silently kept -------------------

    #[test]
    fn an_unknown_property_is_a_rotation_error() {
        let s = BlockState { name: "minecraft:made_up_block".to_string(), properties: vec![("orientation".to_string(), "north_up".to_string())] };
        let err = rotate_block_state(&s, 1).unwrap_err();
        assert_eq!(
            err,
            RotationError::UnrotatableProperty {
                block: "minecraft:made_up_block".to_string(),
                key: "orientation".to_string(),
                value: "north_up".to_string(),
            }
        );
    }

    #[test]
    fn an_unrotatable_property_fails_the_whole_blueprint_rotation() {
        let palette = vec![BlockState { name: "minecraft:made_up_block".to_string(), properties: vec![("orientation".to_string(), "north_up".to_string())] }];
        let blueprint = blueprint_with(IVec3::ONE, palette, vec![0]);
        assert!(rotate_blueprint(&blueprint, Rotation::Deg90).is_err());
        // But Deg0 never even looks at the palette.
        assert!(rotate_blueprint(&blueprint, Rotation::Deg0).is_ok());
    }

    #[test]
    fn an_unrecognised_facing_value_is_a_rotation_error() {
        let s = BlockState { name: "minecraft:weird".to_string(), properties: vec![("facing".to_string(), "sideways".to_string())] };
        assert!(rotate_block_state(&s, 1).is_err());
    }
}
