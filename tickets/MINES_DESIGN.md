# Mines — design (tickets 113–119)

Not a work item; the shared design the numbered mine tickets implement.
Read this first, then the ticket. Where a ticket and this document disagree,
the ticket wins — it's the one that was written against the code.

## What a mine is

A **surface complex** placed like any other building — a headframe over the
shaft, a storage building (the set-dressing for its buffer, same role the
Gatherer's Hut's chests play), and decoration — plus everything it digs
underneath itself over time:

- a **primary shaft**: vertical, square, "considerable diameter", housing a
  spiral staircase with a landing at every corner, lit and ornamented;
- at the current **mining level**, a **secondary shaft**: horizontal,
  4 wide × 3 high, running north and south from the primary shaft;
  cobblestone floor throughout, torches on both walls, ornamental wooden
  pillars;
- off the secondary, **tertiary galleries**: 2 wide × 3 high, running east
  and west, one row every 4 blocks along the secondary, each slowly extended
  out to `gallery_length` (200). Everything in the 2×3 is excavated; the 4×5
  cross-section around it is **scanned for ores**, and any found are dug too.
  Torches on one wall every 8 blocks. Where the gallery breaks into a cave,
  the two floor blocks are laid in cobblestone.

Once a level is worked out the primary shaft is **sunk one flight** (4
blocks) and the next level starts. Tiers differ by how deep they may go:
`min_level_y` is the tier knob — a tier-1 Mine stops in the iron/coal band,
a Deepslate Mine reaches the diamond band.

Like the Gatherer's Hut, a mine has **no recipe**: what comes out is whatever
was actually there, resolved through `drops.ron` (stone → cobblestone, ores →
raw items). Its buffer is a `production::Producer` like every other
producer's, so warehouses, haulage and both UI panels already work — and it
is enormous next to a hut's (`buffer_stacks: 512`, i.e. 4096 blocks at the
debug `stack_size` of 8).

## The rule everything below follows

> **The mine's geometry is a pure function of its definition, its placement
> and its progress cursor.** The tick never "adds" blocks; it computes what
> the next few slices *should* look like, reads what is actually there, and
> writes the difference.

Consequences, all deliberate:

- **Idempotent.** Re-running the generator over already-dug rock costs
  nothing (air is free) and converges. A mine whose progress file is lost
  (demolish → undo, a crash before save) fast-forwards through its own
  tunnels at zero cost rather than corrupting them.
- **Testable without a world.** Layout (114) is pure integer geometry; slice
  planning (115) takes a synthetic block sampler. Only 116's tick touches
  Bevy and the region cache, and it's the same dispatch/poll shape as
  `city::gatherer`.
- **No journal, no `WriteStatus`**, for exactly the reasons `city::gatherer`
  gives: undo undoes builds, not the passage of time, and a background write
  every few seconds would bury the panel's "Last edit" line.

## Coordinates and constants

All in Minecraft world coordinates, `y` up, north = `-z`, east = `+x`. The
mine's placement gives `floor_y = origin.y + ground_level` (the world Y of
the blueprint's own ground layer, the same value `city::gatherer` reads as
its dig floor) and the shaft square's min corner, `shaft_min: IVec2`,
from the definition's `shaft: (x, z)` rotated with the footprint.

Fixed by the cross-sections the design asks for (constants, not `.ron`
fields — changing one changes the others):

| name              | value | why                                                          |
|-------------------|-------|--------------------------------------------------------------|
| `GALLERY_WIDTH`   | 2     | given                                                        |
| `GALLERY_HEIGHT`  | 3     | given                                                        |
| `SCAN_MARGIN`     | 1     | 2×3 grown by one on every side is the given 4×5              |
| `GALLERY_PITCH`   | 4     | width + 2·margin: every block of a level's slab is scanned exactly once — no gaps, no double work. `200 / 4 = 50` rows per 200 blocks of secondary. |
| `SECONDARY_WIDTH` | 4     | given — and equal to `shaft_size - 2`, the shaft interior    |
| `SECONDARY_HEIGHT`| 3     | given                                                        |

Per-definition parameters (`Building::mine`, ticket 113):

```ron
mine: Some((
    shaft: (x: 5, z: 5),      // min corner of the shaft square, unrotated blueprint coords
    shaft_size: 6,            // outer edge of the stair ring; interior is shaft_size - 2
    first_level_depth: 12,    // floor_y - first level's floor; a multiple of level_spacing
    min_level_y: 16,          // THE TIER KNOB: lowest level floor allowed
    level_reach: 100,         // each secondary arm's length (north 100 + south 100 = 200)
    gallery_length: 200,      // each tertiary gallery's length
    torch_spacing: 8,
    max_void_run: 6,          // consecutive all-air slices that end a gallery
    blocks_per_minute: 60.0,
    buffer_stacks: 512,
    haul_at_stacks: Some(64),
    valuables: [],            // extra block names scanned for; every `*_ore` is built in
)),
```

**`level_spacing` is derived, not declared: `shaft_size - 2`.** One flight
of stairs runs along each side of the ring between two corner landings; a
side has `shaft_size - 2` non-corner tiles, so one flight descends exactly
that far, and the corner landings sit at `floor_y - k·level_spacing`. Making
levels the same distance apart is what puts **every level's landing on a
corner**, which is the whole reason the stairs and the levels agree. With
the default `shaft_size: 6` that's the requested 4.

## The primary shaft

Viewed from above, for `shaft_size = 6` (`.` interior, `#` ring, `L` lining):

```
 L L L L L L L L      lining: the (S+2)² perimeter — sealed and ornamented
 L # # # # # # L      ring:   the S² perimeter — stairs and landings
 L # . . . . # L      interior: (S-2)² — the light well, and at every
 L # . . . . # L                mining level a platform / junction floor
 L # . . . . # L
 L # . . . . # L
 L # # # # # # L
 L L L L L L L L
```

**Ring walk.** Index the ring `r ∈ 0 .. 4(S-1)` clockwise from the NW
corner: side `s = r / (S-1)` (north, east, south, west), position
`p = r % (S-1)`, `p = 0` being the corner. Tile heights, for revolution `k`:

```
landing (p = 0):  y = floor_y - s·(S-2) - 4k(S-2)          — a full ground block
stair   (p ≥ 1):  y = floor_y - s·(S-2) - (p-1) - 4k(S-2)  — a stair block
```

Check, `S = 6`: NW landing at `floor_y`; north flight stairs at `floor_y`,
`-1`, `-2`, `-3`; NE landing at `floor_y - 4`. A stair's back face is flush
with the surface of a full block at the same Y, so the first stair of a
flight shares its landing's block Y. Descent per revolution `4(S-2) = 16`.

- **Stairs** face the direction of *ascent* (Minecraft's `facing` for
  stairs): a north-side flight descends eastward so its stairs are
  `facing=west`; east side `facing=north`; south `facing=east`; west
  `facing=south`. `half=bottom`, `shape=straight` (the game re-derives
  `shape` at corners on its first block update; not worth modelling).
- **Under every stair and landing**, one `ground` block: the staircase reads
  as a solid ribbon rather than floating steps.
- **The interior** is air from `floor_y` (the blueprint's own floor over it
  is opened — the well through the headframe) down to the current bottom.
  **At every mining level `L`**, the interior at `y = L` is a `ground`
  platform: it is that level's junction, and the secondary shaft's two arms
  connect across it. The stairs wrap around the stack of platforms.
- **The current bottom `B`** (a mining level while mining, an intermediate
  flight end while sinking): ring tiles are landing/stair per the formula
  where the formula puts them at `y ≥ B`; below `B` nothing is touched.
  While `B` is a mining level, the whole `S²` at `y = B` is `ground`.
- **Lining** (`y < floor_y` only — at `floor_y` the blueprint is
  authoritative): the four corners are `pillar` (`oak_log[axis=y]`) the full
  depth; at every mining level `y = L` the lining is a band of `band`
  (`stripped_oak_log`); everything else is written only if it is *not
  solid* (air, fluid, clutter) — a cave or a lake behind the wall gets
  `ground`, plain rock is left as it is. On the north and south sides at a
  mining level, the interior-width span (`x` in the interior's range,
  `y = L+1 ..= L+3`) is the **doorway** into that level's secondary arms.
- **Torches**: on the lining wall, facing into the shaft, one every
  `torch_spacing` ring steps, at head height (`stair_y + 2`) above the step
  they belong to. Wall torches need a solid block behind them; the lining
  guarantees one.

**Sinking.** The shaft descends one flight (`level_spacing`) per job: bottom
`B → B - level_spacing`. The interior is excavated `B-1 ..= B-spacing+1`
(the old platform at `B` stays if `B` was a level), the ring's next flight
is written, the lining continued, and if the new bottom is a level, its
floor. The initial descent from `floor_y` to `floor_y - first_level_depth`
is `first_level_depth / level_spacing` such jobs.

## A mining level

Level floor `L`. Air at `L+1 ..= L+3`; the block below that is the floor,
and the block above is the level above's floor (levels are 4 apart and
galleries 3 high — one block of rock between stacked galleries, hence the
row stagger below).

**Secondary shaft.** `x` range = the shaft interior's `x` range (4 wide);
`y = L+1 ..= L+3`; floor `y = L` **always `ground`**. Two arms: north from
the lining outward `level_reach` blocks, south likewise. It grows in
4-block steps as rows are opened (see "order of work").

**Rows.** Along each arm, row `k` occupies the two `z` tiles at distance
`4k + 2` and `4k + 3` from the lining (distance 0 is the first block past
the lining wall), leaving 2 blocks of rock between the lining and row 0 and
between successive rows. **Odd-numbered levels shift every row by 2** (half
a pitch) so a gallery never sits directly over the one below it — the rock
between galleries staggers into a checkerboard instead of 1-block floors
stacked 4 apart. Rows per arm: `level_reach / GALLERY_PITCH` (25 at the
default; 50 rows, 100 galleries per level).

**Galleries.** From each row, one gallery east and one west. The mouth is
the secondary's wall block (`x = interior_max + 1` east, `interior_min - 1`
west); the gallery runs `gallery_length` blocks outward from there. Per
`x` step ("slice"): excavate the 2×3 (`z` = the row's two tiles,
`y = L+1..=L+3`), lay `ground` under either floor tile that isn't solid,
and **scan** the 4×5 (`z ± 1`, `y = L ..= L+4`) — every block whose name
matches `*_ore` or is in `valuables` is dug too. Ore *in the floor layer*
(`y = L`) is replaced with `ground`, not air; ore anywhere else with air.

**Fluids.** Any water or lava in the one-block shell around an excavation
(for a gallery, exactly the 4×5 scan box; for the secondary its 6×5 shell;
for the shaft its lining and the layer under the bottom) is replaced with
`ground`. Not in the request, but a Minecraft mine that isn't sealed floods
the moment it crosses an aquifer, and the shell is already being read.

**Lighting and ornament.**
- Gallery: one torch every `torch_spacing` blocks from the mouth, on the
  north wall only (`z = row_min - 1`), at `y = L+2`: `wall_torch[facing=south]`
  when that wall block is solid, a standing `torch` on the north floor tile
  when it isn't (a cave wall can't hold a torch).
- Secondary: the 2-wide rock between mouths carries a **pillar** on each
  wall — `pillar` at `y = L+1..=L+3` in the wall block nearest the shaft of
  each pair (distance `4k` from the lining, staggered with the rows). Every
  other pillar (`8`-block pitch, i.e. `torch_spacing / GALLERY_PITCH`
  pillars apart, rounded up) carries a wall torch on the corridor side of
  it, on both walls.

**Where a face stops early.** A gallery ends before `gallery_length` when:
`max_void_run` consecutive slices had nothing solid to excavate (a cave
crossing is a few blocks; a breakout to the surface never ends); its next
slice would touch bedrock (`never_dig`); or its write was **refused**
(`EditRefusal`, in practice an ungenerated or non-`full` chunk — the edge of
the explored world). A secondary arm stops on refusal or bedrock the same
way, and no rows beyond it are opened. A **shaft** job refused is a real
error (its chunk is the building's own): log, refund, retry next tick, like
a gatherer dig.

## Order of work

A level is a deterministic sequence of **slices**; the progress cursor is a
small struct, not a list. Rows are worked nearest-first, alternating arms:
row 0 north, row 0 south, row 1 north, … For a row:

1. **Extend that arm's secondary** 4 more blocks (four 1-block `z` slices,
   each 4×3 + floor), so the arm reaches past the row's mouths.
2. **Advance the row's two galleries alternately**, one slice east, one
   slice west, until both have ended.

Then the next row. When every row is done (or every remaining arm has
stopped), the level is finished: cursor → `Sinking` toward
`L - level_spacing`, unless that is below `min_level_y` (or within 4 of
`WORLD_MIN_Y`), in which case the mine is **mined out** — a terminal
`ProducerState::MinedOut` ("mined out"), distinct from `Depleted`'s "site
levelled".

```
Sinking { target }  ──bottom == target──▶  Mining { level, cursor }
      ▲                                            │ every row done
      └──── next level ≥ min_level_y ──────────────┤
                                                   └── else ──▶ MinedOut
```

## The tick (ticket 116), in one paragraph

Per placed mine, per frame with clock time: accrue `blocks_per_minute ·
minutes` into the producer's `dig_carry`; skip if the buffer is at capacity
(`BufferFull`, no carry), if a job is already in flight for this building,
or if `floor(carry) < 1`. Otherwise spawn one task with a **budget** of
`min(floor(carry), MAX_BLOCKS_PER_JOB)` blocks and at most
`MAX_SLICES_PER_JOB` slices. The task, under one region-cache lock: reads
the survey box for the next slice (`extract_blueprint`), plans it (115),
adds it to the edit, charges its cost (non-air blocks removed), repeats
while budget remains, then applies the merged `WorldEdit` via
`commit::apply_building_edit` with `capture_replaced` and returns the edit,
the report, the advanced cursor and the cost. Settling on success: credit
`drops.parcel_for(baseline.previous)` to the buffer, `carry -= cost`, store
the cursor. On refusal: refund, and close the face or retry per "where a
face stops early". Blocks the mine *places* (ground, stairs, torches,
pillars) are free in this iteration — ticket 119 revisits.

**Heightmaps and the render floor.** Every mine write is applied with
`HeightmapPolicy::Leave`, unconditionally. Two reasons, one of them a hard
requirement: (1) **what a mine does underground must stay invisible to the
citybuilder.** 030's render floor is computed from a chunk's `OCEAN_FLOOR`
heightmap when the chunk is decoded, and falls back to whole-world decoding
for a chunk with *no* heightmaps — so the default `Delete` would have the
citybuilder re-decode every chunk a 200-block gallery touches in full,
forever, and `Recompute` would lower the floor under the well's mouth to
the shaft's bottom. `Leave` keeps the heightmap, and therefore the floor,
exactly as it was before the mine existed; the `ChunksEdited` re-decode
that follows a write sees the same floor and decodes nothing new below it.
(2) Almost every mine write genuinely can't move a surface. The one that
can — opening the well through the headframe's floor — leaves a stale
`MOTION_BLOCKING`/`OCEAN_FLOOR` reading for those `(S-2)²` columns, which
Minecraft repairs itself the first time any block in that column changes
in-game; until then rain draws over the well's mouth. Accepted. The blocks
themselves are in the region file regardless: `set_blocks` already clears
`isLightOn`, so the game relights the tunnels on load and a player walks
them exactly as generated.

## Tiers

Three definitions sharing one blueprint, the `warehouse01`/`warehouse02`
pattern (`tier` + `requires`, nothing new):

| file         | name            | tier | requires | `min_level_y` | band                   |
|--------------|-----------------|------|----------|---------------|------------------------|
| `mine01.ron` | Mine            | 1    | —        | `16`          | coal, iron, copper     |
| `mine02.ron` | Deep Mine       | 2    | mine01   | `-24`         | + gold, lapis, redstone|
| `mine03.ron` | Deepslate Mine  | 3    | mine02   | `-56`         | + diamond, emerald     |

`-56` keeps the deepest scan layer (`L`) above the bedrock band
(`-64 ..= -60`); `never_dig` guards it anyway. Higher tiers may also dig
faster and reach further; a Deep Mine is its own building, not an upgrade
applied to a placed Mine (there is no upgrade mechanic to reuse).

## What the citybuilder shows

The RTS camera sees the surface complex and the well's mouth through the
headframe floor; everything below is behind 030's render floor, **which
mines never move** (see "Heightmaps and the render floor"). The mine is
*walked* in Minecraft; the inspect panel (117) is where its progress is read
in the citybuilder. A debug "show me the underground" view is out of scope.

## Ticket map

```
113  definition schema: Building::mine, validation, three tier .ron files   (schema, placeholder geometry)
      |
114  layout: pure geometry — ring walk, flights, landings, lining, rows,    (no IO)
     slices, cursor, order of work
      |
115  survey + slice planning: region-cache read, block classification,      (no Bevy)
     the WorldEdit for one slice / one sink job
      |
116  simulation: MinePlugin tick, budgeted jobs, level transitions,          (the game)
     MinedOut, mines.ron persistence, is_producer + panel capacities
      |
117  inspect panel section + manual-verification to-dos
118  a real model: mine.nbt via ranvil-cli struct
119  (stretch) consumables: timber and torches out of the city's stock
```

113 → 114 → 115 → 116 are sequential; 117/118/119 follow 116 in any order.
