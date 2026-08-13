//! The selected 3D region of the world: the data model and the coordinate
//! rules every later ticket in the group depends on (ticket 019), a wireframe
//! box around it ([`gizmo`], 019), and the click/keyboard interaction that
//! moves it ([`input`], ticket 020).
//!
//! This module holds no UI (ticket 021) and it never reads blocks (ticket
//! 022) — the data model exists on its own because
//! *four* later tickets read [`SelectionBounds`], and they must all agree on
//! inclusive-vs-exclusive bounds and on where the box's faces sit relative
//! to a block's cube. Those are the two things that silently produce
//! off-by-one and mirrored blueprints if each ticket decides for itself.
//!
//! ## Coordinates
//!
//! Bounds are in **Minecraft** block coordinates (the same convention the
//! block inspector's `Block: x, y, z` readout uses, so the two can be
//! cross-read) and are **inclusive** on both ends: `min == max` is a legal
//! 1x1x1 selection of a single block, never an empty one. There is no
//! representation of an empty selection — [`Selection`] holds `None` for
//! that.
//!
//! Conversion to Bevy space lives in exactly one place,
//! [`block_bevy_aabb`], and encodes the two facts from
//! [`crate::world::mesh`]'s module docs:
//!
//! 1. `bevy.x = mc.x`, `bevy.y = mc.y`, **`bevy.z = -mc.z`**;
//! 2. a block *occupies* the unit cube `x..x+1`, `y..y+1`, `z..z+1` from its
//!    coordinate — it is not centred on it.
//!
//! Taken together those mean the Z negation **swaps which end of the Z axis
//! is the minimum**: the Bevy-space AABB of an inclusive MC-space selection
//! runs from `-(max.z + 1)` to `-min.z`. Everything that needs a world-space
//! box goes through [`SelectionBounds::bevy_aabb`] rather than doing this
//! arithmetic again.

// Some of this module's API still has no caller: what's left is for tickets
// 021 (shows and edits the bounds) and 022 (walks them in `iter_blocks`
// order). Same call as `world/mod.rs`'s `allow(unused_imports)` — the point
// of doing 019 first is that those tickets find the coordinate rules already
// settled and tested, so trimming this down to what today's callers happen
// to use would defeat the ticket. Drop this once 022 lands.
#![allow(dead_code)]

mod gizmo;
mod input;

use bevy::prelude::*;

// `move_face` still has no caller outside `input` (ticket 021's typed bounds
// fields go through `SelectionBounds::from_corners` instead — they set both
// ends at once rather than pushing one face). It stays re-exported for the
// same reason as `world/mod.rs`'s re-exports: it's this module's public verb.
// `Face`/`CHUNK_STEP` are what 021's key legend is generated from.
#[allow(unused_imports)]
pub use input::{move_face, Face, SelectionInputSet, CHUNK_STEP};

/// The lowest and highest block Y a modern (1.18+) Minecraft world has.
///
/// Hardcoded rather than read from `level.dat`: `mc_anvil`'s `ZERO_OFFSET`
/// already bakes the same assumption into the *read* side, so hardcoding
/// here is consistent with what the app already does rather than a new
/// limitation. A pre-1.18 save (or a datapack changing the world height)
/// would need both of these and that constant sourced from
/// `level.dat`'s `WorldGenSettings`.
pub const WORLD_MIN_Y: i32 = -64;
/// See [`WORLD_MIN_Y`]. Inclusive — 319 is a placeable block, 320 is not.
pub const WORLD_MAX_Y: i32 = 319;

/// The currently selected region, or `None` when nothing is selected.
///
/// One selection at a time, deliberately: if several are ever wanted this
/// becomes a `Vec<SelectionBounds>` and everything downstream keeps working
/// against [`SelectionBounds`] unchanged.
#[derive(Resource, Default)]
pub struct Selection(pub Option<SelectionBounds>);

/// An inclusive box of blocks in Minecraft coordinates, plus the block it
/// started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionBounds {
    /// The block that was first clicked (ticket 020). Kept separately from
    /// `min`/`max` because it survives the box growing away from it — the
    /// anchor is not necessarily a corner once faces have been pushed
    /// inward — and ticket 021 shows it.
    pub anchor: IVec3,
    /// Componentwise minimum. Guaranteed `min <= max` on every axis: every
    /// constructor goes through [`normalized`], so no reader ever has to
    /// wonder whether this is really the minimum.
    pub min: IVec3,
    /// Componentwise maximum, **inclusive**.
    pub max: IVec3,
}

impl SelectionBounds {
    /// The 1x1x1 selection of a single block — what a click produces
    /// (ticket 020) before any keystroke has extended it.
    pub fn from_anchor(block: IVec3) -> Self {
        let block = clamp_y(block);
        Self {
            anchor: block,
            min: block,
            max: block,
        }
    }

    /// A selection spanning two arbitrary corners, keeping `anchor` as
    /// given. Corners are normalized and Y-clamped, so callers (ticket 021's
    /// typed bounds fields, ticket 020's face moves) can hand over whatever
    /// pair they have without sorting it first.
    pub fn from_corners(anchor: IVec3, a: IVec3, b: IVec3) -> Self {
        let (min, max) = normalized(clamp_y(a), clamp_y(b));
        Self {
            anchor: clamp_y(anchor),
            min,
            max,
        }
    }

    /// Block counts along each axis — `max - min + 1`, so always at least 1
    /// on every axis.
    pub fn size(&self) -> IVec3 {
        IVec3::new(
            axis_size(self.min.x, self.max.x),
            axis_size(self.min.y, self.max.y),
            axis_size(self.min.z, self.max.z),
        )
    }

    /// Total blocks in the selection.
    ///
    /// `u64`, not `usize`/`i32`: X and Z are unclamped, so a careless
    /// keystroke repeat can reach billions of blocks, and this is the number
    /// ticket 021 warns on and ticket 022 budgets against. Saturating rather
    /// than wrapping — a selection big enough to overflow `u64` is already
    /// far past any cap, and reporting `u64::MAX` is honest where a wrapped
    /// small number would not be.
    pub fn volume(&self) -> u64 {
        let size = self.size();
        (size.x as u64)
            .saturating_mul(size.y as u64)
            .saturating_mul(size.z as u64)
    }

    pub fn contains(&self, block: IVec3) -> bool {
        block.cmpge(self.min).all() && block.cmple(self.max).all()
    }

    /// Every block in the selection, in **Y outer, Z middle, X inner**
    /// order.
    ///
    /// That order is not arbitrary: it matches
    /// [`ChunkSection::index`](crate::world::decode::ChunkSection::index)'s
    /// layout and the vanilla structure format ticket 023 writes, so ticket
    /// 022 can fill its dense block array without buffering the whole box
    /// twice or transposing it. [`Self::index_of`] is the inverse, and the
    /// two are tested against each other — change one and the test catches
    /// the other.
    ///
    /// Takes `self` by value (it's `Copy`) so the returned iterator borrows
    /// nothing.
    pub fn iter_blocks(self) -> impl Iterator<Item = IVec3> {
        let (min, max) = (self.min, self.max);
        (min.y..=max.y).flat_map(move |y| {
            (min.z..=max.z).flat_map(move |z| (min.x..=max.x).map(move |x| IVec3::new(x, y, z)))
        })
    }

    /// Where `block` lands in [`Self::iter_blocks`]'s sequence, or `None` if
    /// it isn't in the selection at all.
    pub fn index_of(&self, block: IVec3) -> Option<usize> {
        if !self.contains(block) {
            return None;
        }
        let size = self.size();
        let local = block - self.min;
        Some(
            (local.y as usize) * (size.z as usize) * (size.x as usize)
                + (local.z as usize) * (size.x as usize)
                + (local.x as usize),
        )
    }

    /// The selection's world-space axis-aligned box in **Bevy** coordinates,
    /// as `(min, max)`.
    ///
    /// Built by unioning the unit cubes of the two corner blocks (see
    /// [`block_bevy_aabb`]) rather than by open-coding the arithmetic, so
    /// the `+1` on each max face and the Z flip are expressed exactly once.
    pub fn bevy_aabb(&self) -> (Vec3, Vec3) {
        let (a_min, a_max) = block_bevy_aabb(self.min);
        let (b_min, b_max) = block_bevy_aabb(self.max);
        (a_min.min(b_min), a_max.max(b_max))
    }
}

/// The Bevy-space unit cube block `block` occupies, as `(min, max)`.
///
/// The single place the MC→Bevy mapping and the "a block occupies `x..x+1`
/// from its coordinate" rule are written down — see the module docs. Note
/// the returned Z range is `-(z + 1) ..= -z`: negating Z turns the block's
/// *upper* MC-space Z face into its *lower* Bevy-space one.
pub fn block_bevy_aabb(block: IVec3) -> (Vec3, Vec3) {
    let min = Vec3::new(block.x as f32, block.y as f32, -((block.z + 1) as f32));
    let max = Vec3::new((block.x + 1) as f32, (block.y + 1) as f32, -(block.z as f32));
    (min, max)
}

/// Sorts two corners into `(min, max)` componentwise — each axis
/// independently, so a pair that is inverted on only one axis comes back
/// correct on all three.
pub fn normalized(a: IVec3, b: IVec3) -> (IVec3, IVec3) {
    (a.min(b), a.max(b))
}

/// Clamps a block's Y to the world's build limits, leaving X and Z alone
/// (the world is effectively unbounded horizontally — ticket 021 guards on
/// volume instead).
pub fn clamp_y(block: IVec3) -> IVec3 {
    IVec3::new(block.x, block.y.clamp(WORLD_MIN_Y, WORLD_MAX_Y), block.z)
}

/// `max - min + 1`, saturating rather than overflowing on a box spanning
/// most of the `i32` range. Callers rely on the result being `>= 1`.
fn axis_size(min: i32, max: i32) -> i32 {
    max.saturating_sub(min).saturating_add(1)
}

pub struct SelectionPlugin;

impl Plugin for SelectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Selection>()
            .add_plugins(gizmo::SelectionGizmoPlugin)
            .add_plugins(input::SelectionInputPlugin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(min: IVec3, max: IVec3) -> SelectionBounds {
        SelectionBounds::from_corners(min, min, max)
    }

    #[test]
    fn normalized_sorts_each_axis_independently() {
        // Inverted on Z only — a whole-vector swap would "fix" Z at the cost
        // of X and Y, which is exactly the bug this guards against.
        let a = IVec3::new(1, 2, 9);
        let b = IVec3::new(5, 7, 3);
        let (min, max) = normalized(a, b);
        assert_eq!(min, IVec3::new(1, 2, 3));
        assert_eq!(max, IVec3::new(5, 7, 9));
    }

    #[test]
    fn from_anchor_is_a_single_block() {
        let b = SelectionBounds::from_anchor(IVec3::new(-3, 64, 12));
        assert_eq!(b.min, b.max);
        assert_eq!(b.size(), IVec3::ONE);
        assert_eq!(b.volume(), 1);
        assert!(b.contains(IVec3::new(-3, 64, 12)));
        assert!(!b.contains(IVec3::new(-3, 65, 12)));
    }

    #[test]
    fn size_and_volume_of_a_known_box() {
        let b = bounds(IVec3::new(0, 0, 0), IVec3::new(3, 1, 7));
        assert_eq!(b.size(), IVec3::new(4, 2, 8));
        assert_eq!(b.volume(), 64);
    }

    /// Both axes that can go negative, since `max - min + 1` is where a sign
    /// error shows up as an off-by-one rather than as anything obvious.
    #[test]
    fn size_is_right_across_the_y_and_z_origins() {
        let across_y = bounds(IVec3::new(0, -2, 0), IVec3::new(0, 2, 0));
        assert_eq!(across_y.size(), IVec3::new(1, 5, 1));

        let across_z = bounds(IVec3::new(0, 0, -2), IVec3::new(0, 0, 2));
        assert_eq!(across_z.size(), IVec3::new(1, 1, 5));
    }

    #[test]
    fn a_single_block_occupies_a_unit_cube_offset_from_its_coordinate() {
        let (min, max) = block_bevy_aabb(IVec3::new(5, 64, 7));
        assert_eq!(min, Vec3::new(5.0, 64.0, -8.0));
        assert_eq!(max, Vec3::new(6.0, 65.0, -7.0));
    }

    /// The load-bearing test of this whole ticket: the `+1` lands on each
    /// max face, and negating Z swaps which end of that axis is the minimum.
    #[test]
    fn bevy_aabb_swaps_the_z_ends_and_covers_the_max_faces() {
        let b = bounds(IVec3::new(2, 10, 3), IVec3::new(4, 11, 6));
        let (min, max) = b.bevy_aabb();

        assert_eq!(min, Vec3::new(2.0, 10.0, -7.0));
        assert_eq!(max, Vec3::new(5.0, 12.0, -3.0));

        // Restating the rule the other way round: the box's Bevy extent on
        // each axis is exactly the block count along it.
        assert_eq!(max - min, b.size().as_vec3());
    }

    #[test]
    fn iter_blocks_covers_the_selection_exactly_once() {
        let b = bounds(IVec3::new(-1, 3, -5), IVec3::new(1, 4, -3));
        let blocks: Vec<IVec3> = b.iter_blocks().collect();

        assert_eq!(blocks.len() as u64, b.volume());
        assert!(blocks.iter().all(|&block| b.contains(block)));

        let unique: std::collections::HashSet<IVec3> = blocks.iter().copied().collect();
        assert_eq!(unique.len(), blocks.len());
    }

    #[test]
    fn iter_blocks_runs_y_outer_z_middle_x_inner() {
        let b = bounds(IVec3::new(0, 0, 0), IVec3::new(1, 1, 1));
        let blocks: Vec<IVec3> = b.iter_blocks().collect();
        assert_eq!(
            blocks,
            vec![
                IVec3::new(0, 0, 0),
                IVec3::new(1, 0, 0),
                IVec3::new(0, 0, 1),
                IVec3::new(1, 0, 1),
                IVec3::new(0, 1, 0),
                IVec3::new(1, 1, 0),
                IVec3::new(0, 1, 1),
                IVec3::new(1, 1, 1),
            ],
        );
    }

    /// `index_of` is what ticket 022 will write its dense block array with,
    /// so it has to agree with `iter_blocks` block for block.
    #[test]
    fn index_of_is_the_inverse_of_iter_blocks() {
        let b = bounds(IVec3::new(-4, 60, 8), IVec3::new(-2, 62, 11));
        for (expected, block) in b.iter_blocks().enumerate() {
            assert_eq!(b.index_of(block), Some(expected), "at {block}");
        }
        assert_eq!(b.index_of(IVec3::new(-5, 60, 8)), None);
    }

    #[test]
    fn y_is_clamped_to_the_world_build_limits() {
        let below = SelectionBounds::from_anchor(IVec3::new(0, -500, 0));
        assert_eq!(below.min.y, WORLD_MIN_Y);

        let above = SelectionBounds::from_anchor(IVec3::new(0, 5000, 0));
        assert_eq!(above.max.y, WORLD_MAX_Y);

        let spanning = bounds(IVec3::new(0, -500, 0), IVec3::new(0, 5000, 0));
        assert_eq!(spanning.min.y, WORLD_MIN_Y);
        assert_eq!(spanning.max.y, WORLD_MAX_Y);
    }

    /// X and Z deliberately aren't clamped — the world is effectively
    /// unbounded there and ticket 021 guards on volume instead.
    #[test]
    fn x_and_z_are_not_clamped() {
        let b = SelectionBounds::from_anchor(IVec3::new(-30_000_000, 64, 30_000_000));
        assert_eq!(b.min.x, -30_000_000);
        assert_eq!(b.min.z, 30_000_000);
    }

    #[test]
    fn volume_saturates_instead_of_overflowing() {
        let b = bounds(
            IVec3::new(i32::MIN, WORLD_MIN_Y, i32::MIN),
            IVec3::new(i32::MAX, WORLD_MAX_Y, i32::MAX),
        );
        assert!(b.volume() > 0, "an absurd box should still report a size");
    }
}
