//! The first-fit slot allocator (ticket 133, `MODEL_EXPORTER_ROADMAP.md`
//! "Coordinates and markers") — the one piece of genuinely new logic
//! `model-exporter` has. A pure function: candidates come from `world`'s own
//! grid/area/gap, conflicts are checked against `existing` slots under 131's
//! overlap rule ([`Footprint::expanded`] + [`Footprint::intersects`]), and
//! whether a candidate's ground is actually there to build on is answered by
//! a caller-supplied predicate rather than a save read — see [`allocate`]'s
//! doc comment for why.
//!
//! [`super::new`] is the thin command wrapped around this: it builds the
//! `chunk_generated` predicate from a real save and writes the `.ron`
//! [`allocate`] tells it to.

use std::fmt;

use bevy::math::{IVec2, IVec3};

use crate::blueprint::STRUCTURE_BLOCK_MAX_SIZE;

use super::registry::{Footprint, ModelSlot, ModelWorld};

/// Why [`allocate`] couldn't place a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocateError {
    /// `size` has an axis `< 1` or `> STRUCTURE_BLOCK_MAX_SIZE` — checked
    /// before anything else, so a typo'd size never triggers a save read.
    SizeTooLarge { size: IVec3 },
    /// Every candidate that fit inside `world.area` conflicted with an
    /// existing slot (or none fit at all) — `world.area` is full at this
    /// size, not a generation problem.
    AreaFull,
    /// At least one candidate fit and didn't conflict, but every such
    /// candidate's marker footprint touched a chunk column that isn't
    /// `minecraft:full` yet.
    NoGeneratedSpot { candidates_skipped: u32 },
}

impl fmt::Display for AllocateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocateError::SizeTooLarge { size } => write!(
                f,
                "size {}x{}x{} must have every axis in 1..={STRUCTURE_BLOCK_MAX_SIZE}",
                size.x, size.y, size.z
            ),
            AllocateError::AreaFull => {
                write!(f, "no free spot for that size inside world.area (with world.gap clearance)")
            }
            AllocateError::NoGeneratedSpot { candidates_skipped } => write!(
                f,
                "{candidates_skipped} candidate spot(s) fit but none had generated chunks around them \
                 — fly around the area in Minecraft (or pre-generate it, see docs/world-pregeneration.md) and retry"
            ),
        }
    }
}

impl std::error::Error for AllocateError {}

/// The chunk columns (in chunk coordinates) `footprint`'s blocks touch —
/// `footprint`'s `x`/`y` are world `x`/`z` (see [`Footprint`]'s own docs), so
/// this floor-divides both by 16, Anvil's chunk width.
fn chunk_columns(footprint: Footprint) -> impl Iterator<Item = IVec2> {
    let cx_min = footprint.min.x.div_euclid(16);
    let cx_max = footprint.max.x.div_euclid(16);
    let cz_min = footprint.min.y.div_euclid(16);
    let cz_max = footprint.max.y.div_euclid(16);
    (cz_min..=cz_max).flat_map(move |cz| (cx_min..=cx_max).map(move |cx| IVec2::new(cx, cz)))
}

/// Finds the first candidate origin that fits a box of `size` (x/y/z
/// extents) into `world`, or explains why none did.
///
/// Candidates are scanned `z` outer, `x` inner, both stepping by
/// `world.grid` from `world.area.min` — a human flying along a row sees
/// slots in creation order. A candidate `origin = (x, world.ground_y -
/// below, z)` is accepted at the first `x`/`z` that passes all three:
///
/// 1. its footprint lies entirely inside `world.area`;
/// 2. it doesn't conflict with any `existing` slot under 131's overlap rule
///    (the candidate's footprint, expanded by `world.gap`, must not
///    intersect an existing slot's footprint);
/// 3. every chunk column its *marker* footprint touches (the box expanded by
///    1 on every side, since 134's ring sits there) satisfies
///    `chunk_generated`.
///
/// `chunk_generated` is a parameter rather than a save read so this function
/// stays pure and its tests need no save: callers pass `|_| true`, or a
/// closure over a set of known-generated columns. [`super::new`] is the
/// caller that builds a real one.
pub fn allocate(
    world: &ModelWorld,
    existing: &[ModelSlot],
    size: IVec3,
    below: u32,
    chunk_generated: impl Fn(IVec2) -> bool,
) -> Result<IVec3, AllocateError> {
    let axis_in_range = |value: i32| (1..=STRUCTURE_BLOCK_MAX_SIZE).contains(&value);
    if !axis_in_range(size.x) || !axis_in_range(size.y) || !axis_in_range(size.z) {
        return Err(AllocateError::SizeTooLarge { size });
    }

    let origin_y = world.ground_y - below as i32;
    let gap = world.gap as i32;
    let grid = world.grid as i32;

    let mut candidates_skipped: u32 = 0;

    let mut z = world.area.min.z;
    while z + size.z - 1 <= world.area.max.z {
        let mut x = world.area.min.x;
        while x + size.x - 1 <= world.area.max.x {
            let footprint = Footprint {
                min: IVec2::new(x, z),
                max: IVec2::new(x + size.x - 1, z + size.z - 1),
            };

            let conflicts = existing
                .iter()
                .any(|slot| footprint.expanded(gap).intersects(&slot.footprint()));

            if !conflicts {
                let marker_footprint = footprint.expanded(1);
                if chunk_columns(marker_footprint).all(&chunk_generated) {
                    return Ok(IVec3::new(x, origin_y, z));
                }
                candidates_skipped += 1;
            }

            x += grid;
        }
        z += grid;
    }

    if candidates_skipped > 0 {
        Err(AllocateError::NoGeneratedSpot { candidates_skipped })
    } else {
        Err(AllocateError::AreaFull)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::*;
    use crate::blueprint::BlockState;
    use crate::model_exporter::registry::{Area, Point};

    fn test_world(area_max: (i32, i32), gap: u32, grid: u32) -> ModelWorld {
        ModelWorld {
            save: "models".to_string(),
            ground_y: -61,
            area: Area {
                min: Point { x: 0, z: 0 },
                max: Point { x: area_max.0, z: area_max.1 },
            },
            gap,
            grid,
            marker: BlockState::from_str("minecraft:orange_terracotta").expect("valid block state"),
            blueprints_dir: PathBuf::from("assets/city/blueprints"),
        }
    }

    fn slot(name: &str, origin: (i32, i32, i32), size: (i32, i32, i32)) -> ModelSlot {
        ModelSlot {
            name: name.to_string(),
            origin: IVec3::new(origin.0, origin.1, origin.2),
            size: IVec3::new(size.0, size.1, size.2),
            out: None,
        }
    }

    const ALWAYS: fn(IVec2) -> bool = |_| true;

    #[test]
    fn an_empty_registry_places_the_first_slot_at_area_min() {
        let world = test_world((63, 63), 3, 8);
        let origin = allocate(&world, &[], IVec3::new(5, 5, 5), 1, ALWAYS).expect("should allocate");
        assert_eq!(origin, IVec3::new(0, -62, 0));
    }

    #[test]
    fn a_second_same_size_slot_lands_at_the_next_grid_multiple_with_gap_clearance() {
        let world = test_world((63, 63), 3, 8);
        let existing = vec![slot("a", (0, -62, 0), (5, 5, 5))];
        let origin = allocate(&world, &existing, IVec3::new(5, 5, 5), 1, ALWAYS).expect("should allocate");
        // First footprint occupies x 0..=4; next grid multiple is x=8,
        // leaving x 5..=7 (3 blocks — exactly `gap`) clear between the boxes.
        assert_eq!(origin, IVec3::new(8, -62, 0));
    }

    #[test]
    fn a_slot_wider_than_the_rows_remaining_width_wraps_to_the_next_row() {
        // area x is 0..=15 (16 wide): only x=0 fits a 10-wide box on a
        // grid=8 step (x=8 would need up to x=17). A slot already at x=0
        // pushes the next allocation to the z=8 row.
        let world = test_world((15, 63), 3, 8);
        let existing = vec![slot("a", (0, -62, 0), (10, 5, 5))];
        let origin = allocate(&world, &existing, IVec3::new(10, 5, 5), 1, ALWAYS).expect("should allocate");
        assert_eq!(origin, IVec3::new(0, -62, 8));
    }

    #[test]
    fn a_slot_removed_from_the_middle_of_a_row_is_refilled_first_fit() {
        let world = test_world((63, 63), 3, 8);
        // "b" (which would have sat at x=8) has been removed; only "a" and
        // "c" remain.
        let existing = vec![
            slot("a", (0, -62, 0), (5, 5, 5)),
            slot("c", (16, -62, 0), (5, 5, 5)),
        ];
        let origin = allocate(&world, &existing, IVec3::new(5, 5, 5), 1, ALWAYS).expect("should allocate");
        assert_eq!(origin, IVec3::new(8, -62, 0));
    }

    #[test]
    fn a_footprint_that_would_straddle_area_max_is_rejected() {
        // area x is 0..=3: a 5-wide box can't fit anywhere (0+4=4 > 3), so
        // no candidate ever passes the area check.
        let world = test_world((3, 63), 3, 8);
        let err = allocate(&world, &[], IVec3::new(5, 5, 5), 1, ALWAYS).expect_err("should not fit");
        assert!(matches!(err, AllocateError::AreaFull));
    }

    #[test]
    fn ungenerated_chunks_in_the_first_row_push_allocation_to_the_second_row() {
        // grid=32 keeps each row's only candidate (area x is 16 wide) inside
        // one chunk column boundary; z=0's marker footprint touches chunk
        // row cz<=0, z=32's touches cz>=1.
        let world = test_world((15, 63), 3, 32);
        let generated_from_row_one = |c: IVec2| c.y >= 1;
        let origin = allocate(&world, &[], IVec3::new(5, 5, 5), 1, generated_from_row_one).expect("should allocate");
        assert_eq!(origin, IVec3::new(0, -62, 32));
    }

    #[test]
    fn ungenerated_chunks_everywhere_yield_no_generated_spot() {
        let world = test_world((15, 63), 3, 32);
        let err = allocate(&world, &[], IVec3::new(5, 5, 5), 1, |_| false).expect_err("should not fit");
        assert!(matches!(err, AllocateError::NoGeneratedSpot { candidates_skipped: 2 }));
    }

    #[test]
    fn an_oversize_axis_yields_size_too_large() {
        let world = test_world((511, 511), 3, 8);
        let size = IVec3::new(STRUCTURE_BLOCK_MAX_SIZE + 1, 5, 5);
        let err = allocate(&world, &[], size, 1, ALWAYS).expect_err("should reject");
        assert!(matches!(err, AllocateError::SizeTooLarge { .. }));
    }

    #[test]
    fn a_zero_size_axis_yields_size_too_large() {
        let world = test_world((511, 511), 3, 8);
        let err = allocate(&world, &[], IVec3::new(0, 5, 5), 1, ALWAYS).expect_err("should reject");
        assert!(matches!(err, AllocateError::SizeTooLarge { .. }));
    }
}
