# 114 - Mine: layout geometry and the progress cursor (pure, no IO)

Design: `MINES_DESIGN.md` ("The primary shaft", "A mining level", "Order of
work"). Depends on 113 (`definition::Mine`). Nothing here touches Bevy, the
region cache or `DecodedWorld` — this is the module 115 and 116 call, and
the one where every coordinate rule in the design gets a test.

## Module

`city::mine` becomes a directory module now rather than in 116, so the
three tickets don't fight over one file:

```
src/city/mine/mod.rs        // 116: the plugin. For now: `pub mod layout; pub mod progress;` and the docs
src/city/mine/layout.rs     // this ticket: frames, ring walk, shaft target fn, level rows, slice geometry
src/city/mine/progress.rs   // this ticket: MineProgress, cursor, next_slice / advance
```

## Constants (`layout`)

The table from the design, plus the materials:

```rust
pub const GALLERY_WIDTH: i32 = 2;
pub const GALLERY_HEIGHT: i32 = 3;
pub const SCAN_MARGIN: i32 = 1;
pub const GALLERY_PITCH: i32 = GALLERY_WIDTH + 2 * SCAN_MARGIN;   // 4
pub const SECONDARY_HEIGHT: i32 = 3;
pub const GROUND: &str = "minecraft:cobblestone";
pub const STAIRS: &str = "minecraft:oak_stairs";
pub const PILLAR: &str = "minecraft:oak_log";          // axis=y
pub const BAND: &str = "minecraft:stripped_oak_log";   // axis=y
pub const TORCH: &str = "minecraft:torch";
pub const WALL_TORCH: &str = "minecraft:wall_torch";
```

`GALLERY_PITCH` is *derived* in code exactly as the design derives it; a
doc comment carries the "every block scanned once" argument.

## `MineFrame` — the placement, resolved

```rust
pub struct MineFrame {
    pub shaft_min: IVec2,   // world (x, z) of the ring square's min corner
    pub shaft_size: i32,
    pub floor_y: i32,       // origin.y + ground_level — the same value gatherer's DigSite::floor_y is
    pub first_level_depth: i32,
}
impl MineFrame {
    pub fn from_placement(placed: &PlacedBuilding, mine: &Mine, ground_level: u32) -> Self;
    pub fn level_spacing(&self) -> i32;                 // shaft_size - 2
    pub fn level_floor(&self, level: u32) -> i32;       // floor_y - first_level_depth - level * spacing
    pub fn level_at_floor(&self, y: i32) -> Option<u32>;
    pub fn interior_x(&self) -> RangeInclusive<i32>;    // shaft_min.x + 1 ..= shaft_min.x + shaft_size - 2
    pub fn interior_z(&self) -> RangeInclusive<i32>;
    pub fn ring(&self) -> impl Iterator<Item = RingTile>;   // 4(S-1) tiles, clockwise from NW
    pub fn lining(&self) -> impl Iterator<Item = LiningTile>; // 4(S+1) tiles, with `corner: bool`
}
pub struct RingTile { pub index: usize, pub tile: IVec2, pub side: Side, pub corner: bool }
pub enum Side { North, East, South, West }   // the ring side; also a stair's *descent* direction's side
```

**Rotating `shaft`.** `Mine::shaft` is in unrotated blueprint coordinates.
`footprint_extent` already says how the footprint's axes swap; the square's
min corner under `Rotation::Deg90/180/270` follows the same corner math
`blueprint::rotate` applies to block positions — if there's a shared helper
in `rotate.rs`/`placement.rs` for "rotate a local (x, z) within a footprint"
use it, otherwise add `state::rotate_local_tile(tile, footprint, rotation)`
next to `footprint_extent` and give it the four-rotation test. The shaft is
a square, so only its min corner needs the treatment.

## The ring walk and `shaft_target`

```rust
/// Height of ring tile `index` on its `revolution`-th pass — the design's
/// two formulas. `revolution` 0 starts at `floor_y` (NW landing).
pub fn ring_height(&self, index: usize, revolution: u32) -> i32;

/// Which revolution (if any) puts ring tile `index` at exactly `y`.
pub fn ring_revolution_at(&self, index: usize, y: i32) -> Option<u32>;

pub enum ShaftBlock {
    Air,                    // interior between platforms, ring tiles between passes
    Ground,                 // landings, floors, platforms, stair supports
    Stair { facing: Side }, // `facing` = direction of ascent
    Pillar, Band,           // lining corners; lining at a level floor
    Doorway,                // lining, interior-width span, L+1..=L+3 at a level: air
    SealIfNotSolid,         // lining otherwise: write GROUND only if what's there isn't solid
    WallTorch { facing: Side },
    Untouched,              // below the bottom; at floor_y on the lining
}

/// The design's pure target function: what block belongs at `at`, for a
/// shaft whose bottom is `bottom`. Defined for every (x, z) in the lining
/// square and every `y` in `bottom ..= floor_y`; `Untouched` outside.
pub fn shaft_target(&self, bottom: i32, torch_spacing: i32, at: IVec3) -> ShaftBlock;
```

Rules `shaft_target` encodes, in order (each is a test):

1. Lining at `y == floor_y` → `Untouched` (the blueprint owns its floor).
2. Lining corner → `Pillar`. Lining at a level floor `y == L` → `Band`.
   Lining north/south side, `x` in interior range, `L+1 ..= L+3` for a level
   `L >= bottom` → `Doorway`. Lining with a torch spot → `WallTorch` (see
   below). Other lining → `SealIfNotSolid`.
3. Ring tile whose `ring_revolution_at(index, y)` is `Some` and `y >= bottom`
   → `Ground` if corner else `Stair { facing }`, where facing is the
   ascent: north side → `West`, east → `North`, south → `East`, west →
   `South`.
4. Ring tile one below such a step (`ring_revolution_at(index, y + 1)`) →
   `Ground` (the support).
5. Any `(x, z)` of the ring or interior at `y == bottom` when `bottom` is a
   level floor → `Ground`. Interior at `y == L` for any level `L > bottom`
   → `Ground` (the platform).
6. Ring or interior otherwise, `bottom < y <= floor_y` → `Air`; `y == bottom`
   not a level → `Untouched` (the rock the next flight lands on).

**Torch spots**: every `torch_spacing`-th ring step (`index + 4(S-1)·k`
counted from the top, so spacing carries across revolutions) gets a wall
torch on the lining block *behind* it (outward normal of its side) at
`step_y + 2`, `facing` = into the shaft (the side's inward direction).
Rule 2 checks this before `SealIfNotSolid`; a torch spot on a corner
pillar or a doorway loses to those (rare; fine).

## Levels: rows, arms, galleries

```rust
pub enum Arm { North, South }
pub enum GallerySide { East, West }

pub struct LevelFrame<'a> { frame: &'a MineFrame, pub level: u32, pub floor: i32 }
impl LevelFrame<'_> {
    pub fn stagger(&self) -> i32;                              // 2 on odd levels, else 0
    pub fn rows_per_arm(&self, level_reach: i32) -> u32;       // level_reach / GALLERY_PITCH
    /// z of the arm's slice `distance` blocks past the lining (0 = first block outside the lining wall).
    pub fn secondary_z(&self, arm: Arm, distance: i32) -> i32;
    /// The two z tiles of row `k`: distances 4k+2+stagger and 4k+3+stagger from the lining.
    pub fn row_z(&self, arm: Arm, row: u32) -> [i32; 2];
    /// Secondary distance the arm must reach before row `k`'s mouths are open: 4k+4+stagger.
    pub fn row_reach(&self, row: u32) -> i32;
    /// x of gallery slice `distance` (0 = the secondary's wall block, i.e. the mouth).
    pub fn gallery_x(&self, side: GallerySide, distance: i32) -> i32;
    /// Whether the wall at secondary `distance` carries a pillar (distance ≡ stagger mod 4 — the
    /// nearer of the two rock blocks between mouths) and whether that pillar carries a torch
    /// (every `ceil(torch_spacing / GALLERY_PITCH)`-th pillar).
    pub fn pillar_at(&self, distance: i32, torch_spacing: i32) -> Option<Pillar>;  // Pillar { torch: bool }
}
```

## Slices and their geometry

```rust
pub enum Slice {
    Sink,                                                        // one flight: bottom → bottom - spacing
    Secondary { arm: Arm, distance: i32 },                       // one z step, 4x3
    Gallery { arm: Arm, row: u32, side: GallerySide, distance: i32 },  // one x step, 2x3 + scan
}

pub struct SliceGeometry {
    pub excavate: Box3,              // inclusive block box cleared to air (Sink: the interior L-1..; see below)
    pub survey: Box3,                // what 115 must read: excavate grown by SCAN_MARGIN on every side (the 4x5 for a gallery); the whole lining column span for Sink
    pub scan_for_ore: bool,          // Gallery only
    pub floor: Vec<IVec2>,           // (x, z) tiles at y = floor whose block is GROUND-if-not-solid (gallery) or GROUND-always (secondary)
    pub torch: Option<TorchSpot>,    // TorchSpot { wall: IVec3, fallback_floor: IVec3, facing: Side }
    pub pillars: Vec<(IVec3, bool)>, // secondary: wall blocks to make PILLAR, and whether a torch goes on the corridor side
    pub floor_y: i32,                // the level floor L (for "ore in the floor layer → GROUND")
}
pub fn slice_geometry(frame: &MineFrame, mine: &Mine, progress_bottom: i32, level: Option<&LevelFrame>, slice: Slice) -> SliceGeometry;
```

For `Sink`, `excavate` is the `S²` column `bottom - spacing + 1 ..= bottom - 1`
plus the ring tiles at `bottom` that the next flight turns into stairs (the
plan (115) resolves every position in the `(S+2)²` × `[bottom - spacing,
bottom]` box through `shaft_target` with the *new* bottom anyway, so the
box is what matters; `excavate` is informational for the cost estimate).

Torch for a gallery slice: `Some` when `distance % torch_spacing == torch_spacing / 2`
(mid-interval, so the mouth itself isn't lit twice by the secondary's
pillar torch), on the north wall `(x, L+2, row_z[0] - 1)` facing `South`,
fallback a standing torch at `(x, L+1, row_z[0])`.

## `progress` — the cursor

```rust
pub enum Phase {
    Sinking { target: i32 },
    Mining(LevelCursor),
    MinedOut,
}
pub struct MineProgress {
    pub bottom: i32,        // the shaft's current bottom (a level floor while Mining)
    pub phase: Phase,
}
pub struct LevelCursor {
    pub level: u32,
    pub row: u32,
    pub arm: Arm,
    pub step: RowStep,
    pub arm_reach: [i32; 2],      // secondary distance dug, per Arm
    pub arm_closed: [bool; 2],    // refused/bedrock: no further rows on that arm
}
pub enum RowStep {
    Secondary,                                   // extend `arm` until arm_reach >= row_reach(row)
    Galleries { faces: [Face; 2], next: GallerySide },
}
pub struct Face { pub distance: i32, pub void_run: u32, pub closed: bool }

pub enum SliceOutcome {
    Dug { cost: u32 },   // something was removed (cost > 0) — resets void_run
    Void,                // nothing solid to excavate — void_run += 1
    Bedrock,             // the slice would touch bedrock — close the face
    Refused,             // EditRefusal — close the face (Secondary/Gallery) or retry (Sink)
}

impl MineProgress {
    pub fn new(frame: &MineFrame) -> Self;    // bottom = floor_y, Sinking { target: level_floor(0) }
    pub fn next_slice(&self, mine: &Mine, frame: &MineFrame) -> Option<Slice>;   // None iff MinedOut
    pub fn advance(&mut self, mine: &Mine, frame: &MineFrame, slice: Slice, outcome: SliceOutcome);
    pub fn level(&self) -> Option<LevelFrame>;   // while Mining
}
```

`next_slice` walks the design's order of work: `Sinking` → `Sink`;
`Mining` with `RowStep::Secondary` → `Secondary { arm, distance:
arm_reach[arm] }` while `arm_reach < row_reach(row)` (a closed arm skips
straight to the next row/arm); then `Galleries` → the open face named by
`next`, alternating; both faces closed → next row (other arm first); rows
exhausted or both arms closed → `Sinking { target: floor - spacing }`, or
`MinedOut` when `target < mine.min_level_y`.

`advance` applies the outcome: `Dug` resets `void_run`; `Void` bumps it
and closes the face at `max_void_run`; `Bedrock`/`Refused` close the face
(for `Secondary`, close the arm); every gallery slice bumps `distance` and
closes at `gallery_length`; a `Sink` `Dug`/`Void` lowers `bottom` by
`spacing` and flips to `Mining` when it reaches `target`; a `Sink`
`Refused` changes nothing (116 logs and retries).

`MineProgress` derives `Serialize`/`Deserialize` (116 persists it);
`BuildingId` stays outside it, the `SavedProducer` way.

## Tests (the point of the ticket)

Layout:
- `ring()` yields `4(S-1)` tiles clockwise from NW; corners at `index %
  (S-1) == 0`; for `S = 6` the NE corner is index 5 at `shaft_min + (5, 0)`.
- `ring_height`: `S = 6` → indices 0..=5 at `floor_y, floor_y, -1, -2, -3,
  -4`; index 5 (NE) is `floor_y - 4`; index 0 on revolution 1 is `floor_y -
  16`. **Every level floor coincides with some corner's landing** — for
  levels 0..10, `ring_revolution_at(corner, level_floor(n))` is `Some` for
  exactly one corner.
- `shaft_target` per rule above, including: interior at a level floor above
  the bottom is `Ground`; lining at `floor_y` is `Untouched`; doorway spans
  exactly the interior's x range and `L+1..=L+3`; a torch spot every
  `torch_spacing` steps and its `facing` points inward; nothing below
  `bottom` is anything but `Untouched`.
- `from_placement` under all four rotations puts the shaft where the rotated
  footprint puts that corner (compare against `footprint_tiles`).
- Rows: level 0 row 0 north is at `z = lining_north - 1 - 2` and `- 3`;
  level 1 is shifted by 2; `row_reach(k) = 4k + 4 + stagger`; galleries at
  the same row on levels 0 and 1 don't share a `z`.
- `pillar_at`: exactly one pillar per 4 blocks of secondary, in the rock
  between mouths (never at a row's `z`); torches on every second pillar
  for `torch_spacing: 8`.
- `slice_geometry` for a gallery: `excavate` is 2×3×1 at the right `x`,
  `survey` is 4×5×1 around it, `floor` has two tiles, torch `Some` exactly
  at the mid-interval distances.

Progress:
- A fresh mine with `first_level_depth: 12`, `S = 6` yields three `Sink`
  slices then `Secondary { North, 0 }`.
- First row: four `Secondary { North, 0..4 }`, then `Gallery { North, 0,
  East, 0 }`, `Gallery { North, 0, West, 0 }`, `East, 1`, … (stagger 0).
- A `Void` × `max_void_run` closes a face; the other face keeps going; both
  closed → the *south* arm's `Secondary` comes next.
- `Refused` on a `Secondary` closes the arm and no later row names it.
- `gallery_length` reached → closed.
- A level whose rows are all done → `Sinking { target: floor - 4 }`; the
  target below `min_level_y` → `MinedOut`, `next_slice` is `None`.
- Round-trip through RON.
- `cargo check`, `cargo test --lib`, `cargo clippy` clean.
