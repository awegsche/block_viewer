//! Pure integer geometry for a mine (ticket 114, `MINES_DESIGN.md`). No
//! Bevy, no region cache, no `DecodedWorld` — this is the module ticket
//! 115's slice planner and ticket 116's tick both call into, and the one
//! where every coordinate rule the design doc states gets a test.
//!
//! Three layers, outside in:
//!
//! - [`MineFrame`]: the placement, resolved — where the shaft sits in world
//!   space, and the pure functions of `(x, y, z)` that say what belongs
//!   there ([`MineFrame::shaft_target`]) for the primary shaft's ring,
//!   lining and interior.
//! - [`LevelFrame`]: one mining level's row/gallery geometry, layered on
//!   top of a [`MineFrame`].
//! - [`Slice`]/[`SliceGeometry`]: the unit of work `city::mine::progress`
//!   walks through — one flight of the shaft, one step of a secondary arm,
//!   or one step of a gallery — resolved to the boxes and lists ticket
//!   115's planner reads blocks against and writes.

use std::ops::RangeInclusive;

use bevy::math::{IVec2, IVec3};
use serde::{Deserialize, Serialize};

use crate::city::definition::Mine;
use crate::city::state::{rotate_local_tile, PlacedBuilding};

// --- constants -------------------------------------------------------------

/// Given by the design: a tertiary gallery is 2 wide.
pub const GALLERY_WIDTH: i32 = 2;
/// Given by the design: a tertiary gallery (and the secondary shaft) is 3
/// tall.
pub const GALLERY_HEIGHT: i32 = 3;
/// Given by the design: the ore scan grows a gallery's cross-section by one
/// block on every side.
pub const SCAN_MARGIN: i32 = 1;
/// `GALLERY_WIDTH + 2 * SCAN_MARGIN` — derived, not an independent knob:
/// consecutive gallery slices' scan boxes tile the level's slab with no
/// gaps and no double-scanning only because the gallery advances exactly
/// as far as its own scan box is wide. See `MINES_DESIGN.md`'s constants
/// table.
pub const GALLERY_PITCH: i32 = GALLERY_WIDTH + 2 * SCAN_MARGIN;
/// Given by the design: the secondary shaft is 3 tall, same as a gallery.
pub const SECONDARY_HEIGHT: i32 = 3;

pub const GROUND: &str = "minecraft:cobblestone";
pub const STAIRS: &str = "minecraft:oak_stairs";
pub const PILLAR: &str = "minecraft:oak_log";
pub const BAND: &str = "minecraft:stripped_oak_log";
pub const TORCH: &str = "minecraft:torch";
pub const WALL_TORCH: &str = "minecraft:wall_torch";

// --- Side --------------------------------------------------------------

/// A ring or lining side — also a stair flight's *descent* direction's
/// side, and a doorway/pillar wall's compass direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    North,
    East,
    South,
    West,
}

impl Side {
    fn from_index(s: i32) -> Side {
        match s {
            0 => Side::North,
            1 => Side::East,
            2 => Side::South,
            _ => Side::West,
        }
    }

    /// The direction facing back into the shaft from a lining wall on this
    /// side — where a wall torch on that wall faces.
    pub fn opposite(self) -> Side {
        match self {
            Side::North => Side::South,
            Side::South => Side::North,
            Side::East => Side::West,
            Side::West => Side::East,
        }
    }

    /// A stair flight running along this ring side faces the direction of
    /// *ascent* — `MINES_DESIGN.md`'s "The primary shaft": a north-side
    /// flight descends eastward (walking east means walking down), so its
    /// stairs face west.
    fn ascent_facing(self) -> Side {
        match self {
            Side::North => Side::West,
            Side::East => Side::North,
            Side::South => Side::East,
            Side::West => Side::South,
        }
    }
}

// --- the square perimeter walk, shared by the ring and the lining ------

/// The tile at `index` walking a `size`-square's perimeter clockwise from
/// its NW corner (`min`), 4·(size-1) tiles total — [`MineFrame::ring`]'s own
/// walk (`size = shaft_size`) and [`MineFrame::lining`]'s (`size =
/// shaft_size + 2`) are the same function at two different sizes.
fn perimeter_tile(min: IVec2, size: i32, index: usize) -> (IVec2, Side, bool) {
    let side_len = size - 1;
    let idx = index as i32;
    let s = idx / side_len;
    let p = idx % side_len;
    let corner = p == 0;
    let tile = match s {
        0 => IVec2::new(min.x + p, min.y),
        1 => IVec2::new(min.x + side_len, min.y + p),
        2 => IVec2::new(min.x + side_len - p, min.y + side_len),
        _ => IVec2::new(min.x, min.y + side_len - p),
    };
    (tile, Side::from_index(s), corner)
}

/// The inverse of [`perimeter_tile`]: which perimeter index (if any) `tile`
/// is, for a `size`-square with min corner `min`. `None` for a tile off the
/// perimeter entirely (outside the square, or strictly inside it).
fn perimeter_index(min: IVec2, size: i32, tile: IVec2) -> Option<usize> {
    let side_len = size - 1;
    let rel = tile - min;
    let (rx, rz) = (rel.x, rel.y);
    if rx < 0 || rx > side_len || rz < 0 || rz > side_len {
        return None;
    }
    let idx = if rz == 0 && rx < side_len {
        rx
    } else if rx == side_len && rz < side_len {
        side_len + rz
    } else if rz == side_len && rx > 0 {
        2 * side_len + (side_len - rx)
    } else if rx == 0 && rz > 0 {
        3 * side_len + (side_len - rz)
    } else {
        return None; // strictly inside the square
    };
    Some(idx as usize)
}

/// One tile of [`MineFrame::ring`] — the `S²` perimeter around the shaft's
/// light well.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingTile {
    pub index: usize,
    pub tile: IVec2,
    pub side: Side,
    pub corner: bool,
}

/// One tile of [`MineFrame::lining`] — the `(S+2)²` perimeter around the
/// ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiningTile {
    pub index: usize,
    pub tile: IVec2,
    pub side: Side,
    pub corner: bool,
}

/// What belongs at one block position of the primary shaft — the target
/// [`MineFrame::shaft_target`] resolves to. Ticket 115 turns each variant
/// into an actual block (and, for the `SealIfNotSolid`/`Doorway` pair, a
/// read of what's already there); this module only says which one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaftBlock {
    /// Open air: the interior between platforms, or a ring tile between
    /// stair passes.
    Air,
    /// A full [`GROUND`] block: a landing, a level's floor, a level's
    /// interior platform, or a stair's support.
    Ground,
    /// A [`STAIRS`] block, `facing` the direction of ascent.
    Stair { facing: Side },
    /// A lining corner, [`PILLAR`], full depth.
    Pillar,
    /// The lining at a mining level's floor, [`BAND`].
    Band,
    /// The lining's doorway into a level's secondary arms — open air, on
    /// the north/south lining sides only.
    Doorway,
    /// Lining that isn't a corner, a level band or a doorway: [`GROUND`],
    /// but only if what's already there isn't solid (a cave or a lake
    /// behind the wall gets sealed; plain rock is left alone).
    SealIfNotSolid,
    /// A wall torch on the lining, facing into the shaft.
    WallTorch { facing: Side },
    /// Below the bottom, or at `floor_y` on the lining (the blueprint's own
    /// floor is authoritative there) — nothing is written.
    Untouched,
}

/// The placement of a mine's primary shaft, resolved into world
/// coordinates — `MINES_DESIGN.md`'s "Coordinates and constants" and "The
/// primary shaft", made queryable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MineFrame {
    /// World `(x, z)` of the ring square's (the `S²` stair perimeter's)
    /// min corner — its NW tile.
    pub shaft_min: IVec2,
    pub shaft_size: i32,
    /// `origin.y + ground_level` — the world Y of the blueprint's own
    /// ground layer, the same value `city::gatherer`'s `DigSite::floor_y`
    /// reads.
    pub floor_y: i32,
    pub first_level_depth: i32,
}

impl MineFrame {
    /// Resolves `mine`'s placeholder-free geometry (`shaft`, `shaft_size`,
    /// `first_level_depth`) against where `placed` actually sits.
    ///
    /// **Rotating `shaft`.** [`Mine::shaft`] is in the *unrotated*
    /// blueprint's local `(x, z)`. The shaft is a square, so rotating just
    /// its stored corner through [`rotate_local_tile`] would land on the
    /// rotated square's *opposite* corner under a 90°/270° turn, not its
    /// new minimum (the same reason [`super::super::state::footprint_extent`]
    /// exists instead of rotating a single footprint corner) — both
    /// corners are rotated and the componentwise minimum taken.
    pub fn from_placement(placed: &PlacedBuilding, mine: &Mine, ground_level: u32) -> Self {
        let shaft_size = mine.shaft_size as i32;
        let local_min = IVec2::new(mine.shaft.x, mine.shaft.z);
        let local_max = local_min + IVec2::splat(shaft_size - 1);
        let rotated_min = rotate_local_tile(local_min, placed.footprint, placed.rotation);
        let rotated_max = rotate_local_tile(local_max, placed.footprint, placed.rotation);
        let world_origin = IVec2::new(placed.origin.x, placed.origin.z);
        MineFrame {
            shaft_min: world_origin + rotated_min.min(rotated_max),
            shaft_size,
            floor_y: placed.origin.y + ground_level as i32,
            first_level_depth: mine.first_level_depth as i32,
        }
    }

    /// One flight of stairs runs along each side of the ring between two
    /// corner landings; a side has `shaft_size - 2` non-corner tiles, so
    /// that's exactly how far one flight descends — `MINES_DESIGN.md`'s
    /// "`level_spacing` is derived, not declared".
    pub fn level_spacing(&self) -> i32 {
        self.shaft_size - 2
    }

    /// The world Y of mining level `level`'s floor: `floor_y -
    /// first_level_depth - level * level_spacing`.
    pub fn level_floor(&self, level: u32) -> i32 {
        self.floor_y - self.first_level_depth - level as i32 * self.level_spacing()
    }

    /// The inverse of [`level_floor`](Self::level_floor): which level (if
    /// any) has its floor exactly at `y`.
    pub fn level_at_floor(&self, y: i32) -> Option<u32> {
        let spacing = self.level_spacing();
        let diff = self.floor_y - self.first_level_depth - y;
        if diff < 0 || diff % spacing != 0 {
            return None;
        }
        Some((diff / spacing) as u32)
    }

    /// A [`LevelFrame`] for mining level `level` of this shaft.
    pub fn level(&self, level: u32) -> LevelFrame<'_> {
        LevelFrame { frame: self, level, floor: self.level_floor(level) }
    }

    pub fn interior_x(&self) -> RangeInclusive<i32> {
        self.shaft_min.x + 1..=self.shaft_min.x + self.shaft_size - 2
    }

    pub fn interior_z(&self) -> RangeInclusive<i32> {
        self.shaft_min.y + 1..=self.shaft_min.y + self.shaft_size - 2
    }

    /// The ring's `4(shaft_size - 1)` tiles, clockwise from the NW corner.
    pub fn ring(&self) -> impl Iterator<Item = RingTile> {
        let min = self.shaft_min;
        let size = self.shaft_size;
        let count = (4 * (size - 1)) as usize;
        (0..count).map(move |i| {
            let (tile, side, corner) = perimeter_tile(min, size, i);
            RingTile { index: i, tile, side, corner }
        })
    }

    /// The lining's `4(shaft_size + 1)` tiles, one block outside the ring.
    pub fn lining(&self) -> impl Iterator<Item = LiningTile> {
        let min = self.shaft_min - IVec2::ONE;
        let size = self.shaft_size + 2;
        let count = (4 * (size - 1)) as usize;
        (0..count).map(move |i| {
            let (tile, side, corner) = perimeter_tile(min, size, i);
            LiningTile { index: i, tile, side, corner }
        })
    }

    /// The height of ring tile `index` on its `revolution`-th pass down —
    /// `MINES_DESIGN.md`'s two landing/stair formulas, applied directly.
    pub fn ring_height(&self, index: usize, revolution: u32) -> i32 {
        let side_len = self.shaft_size - 1;
        let s = index as i32 / side_len;
        let p = index as i32 % side_len;
        let spacing = self.level_spacing();
        let base = self.floor_y - s * spacing - 4 * revolution as i32 * spacing;
        if p == 0 {
            base
        } else {
            base - (p - 1)
        }
    }

    /// The inverse of [`ring_height`](Self::ring_height): which revolution
    /// (if any) puts ring tile `index` at exactly `y`.
    pub fn ring_revolution_at(&self, index: usize, y: i32) -> Option<u32> {
        let side_len = self.shaft_size - 1;
        let s = index as i32 / side_len;
        let p = index as i32 % side_len;
        let spacing = self.level_spacing();
        let base0 = self.floor_y - s * spacing - if p == 0 { 0 } else { p - 1 };
        let step = 4 * spacing;
        let diff = base0 - y;
        if diff < 0 || diff % step != 0 {
            return None;
        }
        Some((diff / step) as u32)
    }

    /// The design's pure target function for the primary shaft: what block
    /// belongs at `at`, for a shaft currently excavated down to `bottom`.
    /// Defined for every `(x, z)` in the lining square and every `y` in
    /// `bottom..=floor_y`; [`ShaftBlock::Untouched`] outside (including
    /// below `bottom` — "below the bottom nothing is touched",
    /// `MINES_DESIGN.md`'s "The primary shaft").
    pub fn shaft_target(&self, bottom: i32, torch_spacing: i32, at: IVec3) -> ShaftBlock {
        let tile = IVec2::new(at.x, at.z);
        let y = at.y;
        if y < bottom || y > self.floor_y {
            return ShaftBlock::Untouched;
        }

        let lining_min = self.shaft_min - IVec2::ONE;
        let lining_size = self.shaft_size + 2;
        if let Some(index) = perimeter_index(lining_min, lining_size, tile) {
            return self.lining_target(tile, index, lining_size, bottom, torch_spacing, y);
        }
        if let Some(index) = perimeter_index(self.shaft_min, self.shaft_size, tile) {
            return self.ring_target(index, bottom, y);
        }
        if self.interior_x().contains(&tile.x) && self.interior_z().contains(&tile.y) {
            return self.interior_target(bottom, y);
        }
        ShaftBlock::Untouched
    }

    fn lining_target(
        &self,
        tile: IVec2,
        index: usize,
        lining_size: i32,
        bottom: i32,
        torch_spacing: i32,
        y: i32,
    ) -> ShaftBlock {
        if y == self.floor_y {
            return ShaftBlock::Untouched; // the blueprint owns its own floor
        }
        let side_len = lining_size - 1;
        let s = index as i32 / side_len;
        let p = index as i32 % side_len;
        if p == 0 {
            return ShaftBlock::Pillar; // lining corner, full depth
        }
        if self.level_at_floor(y).is_some() {
            return ShaftBlock::Band;
        }
        let side = Side::from_index(s);
        if matches!(side, Side::North | Side::South) && self.interior_x().contains(&tile.x) {
            for dy in 1..=3 {
                let level_floor = y - dy;
                if level_floor >= bottom && self.level_at_floor(level_floor).is_some() {
                    return ShaftBlock::Doorway;
                }
            }
        }
        // A torch spot: the ring step this lining cell sits directly behind
        // (one block outward of ring position `p - 1` on the same side —
        // the lining is the ring's square grown by one on every side, so a
        // non-corner lining position `p` sits behind ring position `p - 1`)
        // is a multiple of `torch_spacing` steps from the top, counted
        // across revolutions.
        let ring_index = (s * (self.shaft_size - 1) + (p - 1)) as usize;
        let step_y = y - 2;
        if step_y >= bottom
            && let Some(k) = self.ring_revolution_at(ring_index, step_y)
        {
            let ring_len = 4 * (self.shaft_size - 1);
            let step_number = ring_index as u32 + ring_len as u32 * k;
            if torch_spacing > 0 && step_number.is_multiple_of(torch_spacing as u32) {
                return ShaftBlock::WallTorch { facing: side.opposite() };
            }
        }
        ShaftBlock::SealIfNotSolid
    }

    fn ring_target(&self, index: usize, bottom: i32, y: i32) -> ShaftBlock {
        if y >= bottom {
            if self.ring_revolution_at(index, y).is_some() {
                let corner = index as i32 % (self.shaft_size - 1) == 0;
                return if corner {
                    ShaftBlock::Ground
                } else {
                    let s = index as i32 / (self.shaft_size - 1);
                    ShaftBlock::Stair { facing: Side::from_index(s).ascent_facing() }
                };
            }
            if self.ring_revolution_at(index, y + 1).is_some() {
                return ShaftBlock::Ground; // the support under that step
            }
        }
        if y == bottom {
            return if self.level_at_floor(bottom).is_some() { ShaftBlock::Ground } else { ShaftBlock::Untouched };
        }
        ShaftBlock::Air
    }

    fn interior_target(&self, bottom: i32, y: i32) -> ShaftBlock {
        if y == bottom {
            return if self.level_at_floor(bottom).is_some() { ShaftBlock::Ground } else { ShaftBlock::Untouched };
        }
        if self.level_at_floor(y).is_some() {
            return ShaftBlock::Ground; // a level's junction platform
        }
        ShaftBlock::Air
    }
}

// --- a mining level ------------------------------------------------------

/// Which way a secondary arm runs off the primary shaft.
///
/// Derives `Serialize`/`Deserialize` (ticket 114) since [`super::progress::LevelCursor`]
/// — which 116 persists — carries one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arm {
    North,
    South,
}

impl Arm {
    pub fn index(self) -> usize {
        match self {
            Arm::North => 0,
            Arm::South => 1,
        }
    }
}

/// Which way a tertiary gallery runs off the secondary shaft. See [`Arm`]
/// for why this derives `Serialize`/`Deserialize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GallerySide {
    East,
    West,
}

impl GallerySide {
    pub fn index(self) -> usize {
        match self {
            GallerySide::East => 0,
            GallerySide::West => 1,
        }
    }

    pub fn opposite(self) -> GallerySide {
        match self {
            GallerySide::East => GallerySide::West,
            GallerySide::West => GallerySide::East,
        }
    }
}

/// Whether the wall between two gallery mouths carries a pillar, and
/// whether that pillar also carries a wall torch — [`LevelFrame::pillar_at`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pillar {
    pub torch: bool,
}

/// One mining level's row/gallery geometry, layered on a [`MineFrame`] —
/// `MINES_DESIGN.md`'s "A mining level".
#[derive(Debug, Clone, Copy)]
pub struct LevelFrame<'a> {
    frame: &'a MineFrame,
    pub level: u32,
    pub floor: i32,
}

impl LevelFrame<'_> {
    /// Odd levels shift every row by half a pitch so a gallery never sits
    /// directly over the one below it.
    pub fn stagger(&self) -> i32 {
        if self.level % 2 == 1 {
            2
        } else {
            0
        }
    }

    pub fn rows_per_arm(&self, level_reach: i32) -> u32 {
        (level_reach / GALLERY_PITCH) as u32
    }

    /// `z` of the arm's slice `distance` blocks past the lining wall (`0` =
    /// the first block outside it).
    pub fn secondary_z(&self, arm: Arm, distance: i32) -> i32 {
        match arm {
            Arm::North => self.frame.shaft_min.y - 2 - distance,
            Arm::South => self.frame.shaft_min.y + self.frame.shaft_size + 1 + distance,
        }
    }

    /// The two `z` tiles of row `k`: distances `4k + 2 + stagger` and
    /// `4k + 3 + stagger` from the lining.
    pub fn row_z(&self, arm: Arm, row: u32) -> [i32; 2] {
        let d0 = 4 * row as i32 + 2 + self.stagger();
        [self.secondary_z(arm, d0), self.secondary_z(arm, d0 + 1)]
    }

    /// The secondary distance the arm must reach before row `k`'s mouths
    /// are open: `4k + 4 + stagger`.
    pub fn row_reach(&self, row: u32) -> i32 {
        4 * row as i32 + 4 + self.stagger()
    }

    /// `x` of gallery slice `distance` (`0` = the secondary's wall block,
    /// the mouth).
    pub fn gallery_x(&self, side: GallerySide, distance: i32) -> i32 {
        match side {
            GallerySide::East => *self.frame.interior_x().end() + 1 + distance,
            GallerySide::West => *self.frame.interior_x().start() - 1 - distance,
        }
    }

    /// Whether the secondary wall at `distance` carries a pillar (`distance
    /// ≡ stagger mod 4` — the nearer of the two rock blocks between
    /// mouths, never a row's own `z`), and whether that pillar also
    /// carries a wall torch (every `ceil(torch_spacing / GALLERY_PITCH)`-th
    /// one).
    pub fn pillar_at(&self, distance: i32, torch_spacing: i32) -> Option<Pillar> {
        let stagger = self.stagger();
        if distance < 0 || distance.rem_euclid(4) != stagger {
            return None;
        }
        let k = (distance - stagger) / 4;
        let pitch = ((torch_spacing + GALLERY_PITCH - 1) / GALLERY_PITCH).max(1);
        Some(Pillar { torch: k % pitch == 0 })
    }
}

// --- slices ----------------------------------------------------------------

/// An inclusive block box in Minecraft world coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Box3 {
    pub min: IVec3,
    pub max: IVec3,
}

impl Box3 {
    pub fn new(min: IVec3, max: IVec3) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, pos: IVec3) -> bool {
        pos.cmpge(self.min).all() && pos.cmple(self.max).all()
    }

    /// Every position in the box, Y outer / Z middle / X inner — ticket
    /// 115's planner is the first caller, walking `excavate` and `survey`
    /// one block at a time.
    pub fn iter(&self) -> impl Iterator<Item = IVec3> + '_ {
        let (min, max) = (self.min, self.max);
        (min.y..=max.y)
            .flat_map(move |y| (min.z..=max.z).map(move |z| (y, z)))
            .flat_map(move |(y, z)| (min.x..=max.x).map(move |x| IVec3::new(x, y, z)))
    }
}

/// One unit of mining work — `MINES_DESIGN.md`'s "Order of work": one
/// flight of the primary shaft, one 1-block step of a secondary arm, or
/// one 1-block step of a tertiary gallery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slice {
    Sink,
    Secondary { arm: Arm, distance: i32 },
    Gallery { arm: Arm, row: u32, side: GallerySide, distance: i32 },
}

/// A wall torch with a fallback for when the wall behind it isn't solid —
/// `MINES_DESIGN.md`'s "a standing torch on the north floor tile when it
/// isn't (a cave wall can't hold a torch)".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TorchSpot {
    pub wall: IVec3,
    pub fallback_floor: IVec3,
    pub facing: Side,
}

/// What one [`Slice`] resolves to — the boxes and lists ticket 115's
/// planner reads blocks against (`survey`) and writes (everything else).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceGeometry {
    /// The box this slice clears to air (informational for `Sink`; see
    /// [`slice_geometry`]'s doc comment).
    pub excavate: Box3,
    /// What 115 must read to plan this slice: `excavate` grown by
    /// [`SCAN_MARGIN`] on its two cross-section sides (the travel axis is
    /// never grown — consecutive slices already tile it with no gaps).
    pub survey: Box3,
    /// Whether this slice's `survey` box is scanned for ore — gallery
    /// slices only.
    pub scan_for_ore: bool,
    /// `(x, z)` tiles at `y = floor_y` that get [`GROUND`] — always for a
    /// secondary slice, only if what's there isn't solid for a gallery
    /// slice (see `MINES_DESIGN.md`'s "A mining level").
    pub floor: Vec<IVec2>,
    pub torch: Option<TorchSpot>,
    /// Secondary slice only: wall positions that become [`PILLAR`], each
    /// tagged with whether a wall torch also belongs there.
    pub pillars: Vec<(IVec3, bool)>,
    /// The level floor `L` this slice belongs to — for classifying ore
    /// found in the floor layer (`GROUND`, not air) during 115's scan.
    /// For `Sink`, the shaft's new bottom.
    pub floor_y: i32,
}

/// Resolves one [`Slice`] to its [`SliceGeometry`] — `level` is required
/// for `Secondary`/`Gallery`, ignored for `Sink`.
///
/// For `Sink`, `excavate` is the `S²` column `bottom - spacing + 1 ..=
/// bottom - 1` (the old platform at `bottom` stays if `bottom` was a
/// level; the new bottom's own platform, if any, is written by
/// [`MineFrame::shaft_target`], not here) — informational for a cost
/// estimate, since the plan (115) resolves every position in `survey`
/// (the `(S+2)² × [bottom - spacing, bottom]` box `MINES_DESIGN.md`'s
/// "Sinking" names) through `shaft_target` with the *new* bottom anyway.
pub fn slice_geometry(frame: &MineFrame, mine: &Mine, progress_bottom: i32, level: Option<&LevelFrame>, slice: Slice) -> SliceGeometry {
    match slice {
        Slice::Sink => {
            let bottom = progress_bottom;
            let new_bottom = bottom - frame.level_spacing();
            let ring_min = frame.shaft_min;
            let ring_max = frame.shaft_min + IVec2::splat(frame.shaft_size - 1);
            let excavate = Box3::new(
                IVec3::new(ring_min.x, new_bottom + 1, ring_min.y),
                IVec3::new(ring_max.x, bottom - 1, ring_max.y),
            );
            let lining_min = frame.shaft_min - IVec2::ONE;
            let lining_max = frame.shaft_min + IVec2::splat(frame.shaft_size);
            let survey = Box3::new(
                IVec3::new(lining_min.x, new_bottom, lining_min.y),
                IVec3::new(lining_max.x, bottom, lining_max.y),
            );
            SliceGeometry {
                excavate,
                survey,
                scan_for_ore: false,
                floor: Vec::new(),
                torch: None,
                pillars: Vec::new(),
                floor_y: new_bottom,
            }
        }
        Slice::Secondary { arm, distance } => {
            let level = level.expect("Secondary slice requires a level");
            let z = level.secondary_z(arm, distance);
            let ix_min = *frame.interior_x().start();
            let ix_max = *frame.interior_x().end();
            let excavate = Box3::new(IVec3::new(ix_min, level.floor + 1, z), IVec3::new(ix_max, level.floor + 3, z));
            let survey = Box3::new(IVec3::new(ix_min - 1, level.floor, z - 1), IVec3::new(ix_max + 1, level.floor + 4, z + 1));
            let floor = frame.interior_x().map(|x| IVec2::new(x, z)).collect();
            let pillars = match level.pillar_at(distance, mine.torch_spacing as i32) {
                Some(pillar) => [ix_max + 1, ix_min - 1]
                    .into_iter()
                    .flat_map(|wall_x| {
                        (1..=3).map(move |dy| (IVec3::new(wall_x, level.floor + dy, z), pillar.torch && dy == 2))
                    })
                    .collect(),
                None => Vec::new(),
            };
            SliceGeometry {
                excavate,
                survey,
                scan_for_ore: false,
                floor,
                torch: None,
                pillars,
                floor_y: level.floor,
            }
        }
        Slice::Gallery { arm, row, side, distance } => {
            let level = level.expect("Gallery slice requires a level");
            let x = level.gallery_x(side, distance);
            let z = level.row_z(arm, row);
            let (z_min, z_max) = (z[0].min(z[1]), z[0].max(z[1]));
            let excavate = Box3::new(IVec3::new(x, level.floor + 1, z_min), IVec3::new(x, level.floor + 3, z_max));
            let survey = Box3::new(IVec3::new(x, level.floor, z_min - 1), IVec3::new(x, level.floor + 4, z_max + 1));
            let floor = vec![IVec2::new(x, z[0]), IVec2::new(x, z[1])];
            let torch_spacing = mine.torch_spacing as i32;
            let torch = if torch_spacing > 0 && distance.rem_euclid(torch_spacing) == torch_spacing / 2 {
                Some(TorchSpot {
                    wall: IVec3::new(x, level.floor + 2, z[0] - 1),
                    fallback_floor: IVec3::new(x, level.floor + 1, z[0]),
                    facing: Side::South,
                })
            } else {
                None
            };
            SliceGeometry {
                excavate,
                survey,
                scan_for_ore: true,
                floor,
                torch,
                pillars: Vec::new(),
                floor_y: level.floor,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::ShaftAt;

    fn mine_with(shaft: (i32, i32), shaft_size: u32, first_level_depth: u32, min_level_y: i32) -> Mine {
        Mine {
            shaft: ShaftAt { x: shaft.0, z: shaft.1 },
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

    fn placed_at(origin: IVec3, footprint: IVec2, rotation: Rotation) -> PlacedBuilding {
        PlacedBuilding {
            catalogue_id: "mine01".to_string(),
            definition_id: Some("mine01".to_string()),
            origin,
            rotation,
            footprint,
            work_area: None,
        }
    }

    fn frame(shaft_size: i32, first_level_depth: i32) -> MineFrame {
        MineFrame { shaft_min: IVec2::new(100, 200), shaft_size, floor_y: 64, first_level_depth }
    }

    // --- ring walk -----------------------------------------------------

    #[test]
    fn ring_walk_yields_4s_minus_1_tiles_clockwise_from_nw() {
        let f = frame(6, 12);
        let tiles: Vec<_> = f.ring().collect();
        assert_eq!(tiles.len(), 20); // 4 * (6 - 1)
        assert_eq!(tiles[0].tile, f.shaft_min);
        assert!(tiles[0].corner);
        assert_eq!(tiles[0].side, Side::North);
        // Corners at index % (S - 1) == 0.
        for t in &tiles {
            assert_eq!(t.corner, t.index % 5 == 0);
        }
        // NE corner is index 5, at shaft_min + (5, 0).
        assert_eq!(tiles[5].tile, f.shaft_min + IVec2::new(5, 0));
        assert!(tiles[5].corner);
        assert_eq!(tiles[5].side, Side::East);
    }

    #[test]
    fn ring_height_matches_the_design_worked_example() {
        let f = frame(6, 12);
        // S = 6: indices 0..=5 at floor_y, floor_y, -1, -2, -3, -4.
        let expected = [0, 0, -1, -2, -3, -4];
        for (i, offset) in expected.iter().enumerate() {
            assert_eq!(f.ring_height(i, 0), f.floor_y + offset, "index {i}");
        }
        // NE corner (index 5) is floor_y - 4.
        assert_eq!(f.ring_height(5, 0), f.floor_y - 4);
        // Index 0 on revolution 1 is floor_y - 16 (4 * (S - 2) per turn).
        assert_eq!(f.ring_height(0, 1), f.floor_y - 16);
    }

    #[test]
    fn every_level_floor_lands_on_exactly_one_corners_landing() {
        let f = frame(6, 12);
        let corners: Vec<usize> = f.ring().filter(|t| t.corner).map(|t| t.index).collect();
        assert_eq!(corners.len(), 4);
        for level in 0..10 {
            let floor = f.level_floor(level);
            let matches = corners.iter().filter(|&&c| f.ring_revolution_at(c, floor).is_some()).count();
            assert_eq!(matches, 1, "level {level} floor {floor}");
        }
    }

    // --- from_placement / rotation --------------------------------------

    #[test]
    fn from_placement_at_deg0_places_the_shaft_at_origin_plus_local() {
        let mine = mine_with((5, 5), 6, 12, 16);
        let placed = placed_at(IVec3::new(1000, 64, 2000), IVec2::new(16, 16), Rotation::Deg0);
        let f = MineFrame::from_placement(&placed, &mine, 2);
        assert_eq!(f.shaft_min, IVec2::new(1005, 2005));
        assert_eq!(f.floor_y, 66);
        assert_eq!(f.shaft_size, 6);
    }

    #[test]
    fn from_placement_rotates_the_shaft_with_the_footprint() {
        let mine = mine_with((5, 5), 6, 12, 16);
        let footprint = IVec2::new(16, 16);
        let origin = IVec3::new(1000, 64, 2000);
        // Local shaft spans x/z 5..=10 inside a 16x16 footprint. Under each
        // rotation, compare the *rotated* min/max corner (via
        // `rotate_local_tile`, ticket 114's own helper) against
        // `from_placement`'s result — the two must agree since one is
        // built directly from the other.
        for rotation in [Rotation::Deg0, Rotation::Deg90, Rotation::Deg180, Rotation::Deg270] {
            let placed = placed_at(origin, footprint, rotation);
            let f = MineFrame::from_placement(&placed, &mine, 2);
            let local_min = IVec2::new(5, 5);
            let local_max = IVec2::new(10, 10);
            let r_min = rotate_local_tile(local_min, footprint, rotation);
            let r_max = rotate_local_tile(local_max, footprint, rotation);
            let expected = IVec2::new(origin.x, origin.z) + r_min.min(r_max);
            assert_eq!(f.shaft_min, expected, "{rotation:?}");
        }
    }

    // --- shaft_target ----------------------------------------------------

    #[test]
    fn lining_at_floor_y_is_untouched() {
        let f = frame(6, 12);
        for l in f.lining() {
            let target = f.shaft_target(f.floor_y - 100, 8, IVec3::new(l.tile.x, f.floor_y, l.tile.y));
            assert_eq!(target, ShaftBlock::Untouched);
        }
    }

    #[test]
    fn lining_corners_are_pillars_down_to_bottom() {
        let f = frame(6, 12);
        let bottom = f.floor_y - 40;
        for l in f.lining().filter(|l| l.corner) {
            for y in bottom..f.floor_y {
                let target = f.shaft_target(bottom, 8, IVec3::new(l.tile.x, y, l.tile.y));
                assert_eq!(target, ShaftBlock::Pillar, "corner {:?} y {y}", l.tile);
            }
        }
    }

    #[test]
    fn interior_at_a_level_floor_above_bottom_is_ground() {
        let f = frame(6, 12);
        let level0 = f.level_floor(0);
        let bottom = f.level_floor(1); // below level 0
        let at = IVec3::new(*f.interior_x().start(), level0, *f.interior_z().start());
        assert_eq!(f.shaft_target(bottom, 8, at), ShaftBlock::Ground);
    }

    #[test]
    fn doorway_spans_exactly_the_interior_x_range_and_l_plus_1_to_3() {
        let f = frame(6, 12);
        let level0 = f.level_floor(0);
        let bottom = level0;
        let north_z = f.shaft_min.y - 1;
        for x in (f.shaft_min.x - 1)..=(f.shaft_min.x + f.shaft_size) {
            for dy in 1..=3 {
                let target = f.shaft_target(bottom, 8, IVec3::new(x, level0 + dy, north_z));
                let expected_doorway = f.interior_x().contains(&x);
                assert_eq!(target == ShaftBlock::Doorway, expected_doorway, "x {x} dy {dy}");
            }
        }
    }

    #[test]
    fn torch_spots_repeat_every_torch_spacing_steps_and_face_inward() {
        let f = frame(6, 12);
        let bottom = f.floor_y - 200;
        let mut found = Vec::new();
        for l in f.lining().filter(|l| !l.corner) {
            for y in bottom..f.floor_y {
                if let ShaftBlock::WallTorch { facing } = f.shaft_target(bottom, 8, IVec3::new(l.tile.x, y, l.tile.y)) {
                    assert_eq!(facing, l.side.opposite());
                    found.push((l.tile, y));
                }
            }
        }
        assert!(!found.is_empty());
    }

    #[test]
    fn nothing_below_bottom_is_anything_but_untouched() {
        let f = frame(6, 12);
        let bottom = f.floor_y - 40;
        for y in (bottom - 5)..bottom {
            for l in f.lining() {
                assert_eq!(f.shaft_target(bottom, 8, IVec3::new(l.tile.x, y, l.tile.y)), ShaftBlock::Untouched);
            }
        }
    }

    // --- level rows --------------------------------------------------

    #[test]
    fn row_z_shifts_by_stagger_on_odd_levels() {
        let f = frame(6, 12);
        let level0 = f.level(0);
        let level1 = f.level(1);
        let row0_l0 = level0.row_z(Arm::North, 0);
        let row0_l1 = level1.row_z(Arm::North, 0);
        assert_eq!(level0.stagger(), 0);
        assert_eq!(level1.stagger(), 2);
        assert_eq!(row0_l1[0], row0_l0[0] - 2);
        assert_eq!(row0_l1[1], row0_l0[1] - 2);
        // North lining wall is at shaft_min.y - 1; row 0 distance 2 is the
        // first block past the 2-block rock margin.
        assert_eq!(row0_l0[0], f.shaft_min.y - 1 - 1 - 2);
        assert_eq!(row0_l0[1], f.shaft_min.y - 1 - 1 - 3);
        // No z collision between the same row on two consecutive levels.
        assert_ne!(row0_l0[0], row0_l1[0]);
        assert_ne!(row0_l0[1], row0_l1[1]);
    }

    #[test]
    fn row_reach_grows_by_4_per_row_plus_stagger() {
        let f = frame(6, 12);
        let level1 = f.level(1);
        assert_eq!(level1.row_reach(0), 4 + level1.stagger());
        assert_eq!(level1.row_reach(1), 8 + level1.stagger());
    }

    #[test]
    fn pillar_at_is_one_per_4_blocks_never_at_a_rows_z_and_torches_every_second() {
        let f = frame(6, 12);
        let level0 = f.level(0);
        assert_eq!(level0.stagger(), 0);
        let mut pillar_distances = Vec::new();
        for d in 0..40 {
            if let Some(p) = level0.pillar_at(d, 8) {
                pillar_distances.push((d, p.torch));
            }
        }
        // One every 4 blocks.
        for w in pillar_distances.windows(2) {
            assert_eq!(w[1].0 - w[0].0, 4);
        }
        // Never at a row's own z-distance (4k+2, 4k+3).
        for &(d, _) in &pillar_distances {
            assert_ne!(d.rem_euclid(4), 2);
            assert_ne!(d.rem_euclid(4), 3);
        }
        // Every second pillar (torch_spacing 8, GALLERY_PITCH 4 -> pitch 2)
        // carries a torch.
        let torches: Vec<bool> = pillar_distances.iter().map(|&(_, t)| t).collect();
        for (i, &t) in torches.iter().enumerate() {
            assert_eq!(t, i % 2 == 0);
        }
    }

    // --- slice_geometry ------------------------------------------------

    #[test]
    fn gallery_slice_geometry_matches_the_design() {
        let f = frame(6, 12);
        let mine = mine_with((5, 5), 6, 12, 16);
        let level = f.level(0);
        let z = level.row_z(Arm::North, 0);
        let slice = Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance: 3 };
        let geom = slice_geometry(&f, &mine, f.floor_y, Some(&level), slice);
        let x = level.gallery_x(GallerySide::East, 3);

        // excavate: 2x3x1.
        assert_eq!(geom.excavate.max.x - geom.excavate.min.x + 1, 1);
        assert_eq!(geom.excavate.max.y - geom.excavate.min.y + 1, 3);
        assert_eq!(geom.excavate.max.z - geom.excavate.min.z + 1, 2);
        assert_eq!(geom.excavate.min.x, x);

        // survey: 4x5x1.
        assert_eq!(geom.survey.max.x - geom.survey.min.x + 1, 1);
        assert_eq!(geom.survey.max.y - geom.survey.min.y + 1, 5);
        assert_eq!(geom.survey.max.z - geom.survey.min.z + 1, 4);

        assert!(geom.scan_for_ore);
        assert_eq!(geom.floor.len(), 2);
        assert!(geom.floor.contains(&IVec2::new(x, z[0])));
        assert!(geom.floor.contains(&IVec2::new(x, z[1])));
        assert_eq!(geom.floor_y, level.floor);

        // torch exactly at mid-interval distances (torch_spacing 8 -> distance % 8 == 4).
        let torch_distances: Vec<i32> = (0..20)
            .filter(|&d| {
                slice_geometry(&f, &mine, f.floor_y, Some(&level), Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance: d })
                    .torch
                    .is_some()
            })
            .collect();
        assert_eq!(torch_distances, vec![4, 12]);
    }

    #[test]
    fn round_trip_ring_and_lining_index_are_inverses() {
        let f = frame(6, 12);
        for t in f.ring() {
            assert_eq!(perimeter_index(f.shaft_min, f.shaft_size, t.tile), Some(t.index));
        }
        for t in f.lining() {
            assert_eq!(perimeter_index(f.shaft_min - IVec2::ONE, f.shaft_size + 2, t.tile), Some(t.index));
        }
    }
}
