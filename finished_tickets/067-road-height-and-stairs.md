# 067 - a road's height is decided per placement, and stairs bridge levels

## Report

User, on the state ticket 065 left things in:

> currently building height for roads follows terrain. while a sensible
> default, it diverges from my idea. the problem is that now adjacent tiles
> can (and will) have different Y positions which makes the road
> discontinuous. please stick to the rule: Y of first tile dictates Y of the
> entire road (per placement).
>
> stairs: I created a new tile "stair" that can bridge different building Y
> positions. can we adapt the algorithm to draw three dimensionally? then the
> Y rule becomes: Y of first and Y of last are fix, the road inbetween uses
> stair tiles to create a contiguous piece of road. NOTE: in this case, Y
> position needs to lock to a grid with 4-blocks cell width, since my stairs
> can only fill a 4 block gap.

## The problem with per-cell terrain fitting

Ticket 065 made every cell remember the height it was fitted at, which fixed
roads being buried at Y=0 and fixed re-tiling drift. But the height each cell
was fitted *to* is still `grid::fit_footprint` over that cell's own 6x6 patch
of ground — so a drag across a gentle slope produces a staircase of flat
pieces at six different Y values, each one a hard 1-block cliff against its
neighbour. Every piece is individually correct and the road as a whole is
discontinuous, which is the opposite of what a road is for.

## The rule

A **placement** (one drag) is the unit that owns a height, not a cell:

- The drag's **first** cell fixes the road's starting level.
- The drag's **last** cell fixes its ending level, *snapped* to the first's
  level plus a whole number of `ROAD_STAIR_RISE` = 4 block steps — one stair
  piece's worth of climb each, since that's all the shipped stair geometry
  can bridge. The 4-block grid is anchored to the first cell, not to absolute
  world Y: the road starts flush with the real terrain wherever the drag
  began, and every later level is a whole number of stair pieces from it.
- The cells **in between** run flat at the level they're on, except for the
  `|steps|` of them chosen to be **stair** cells, which each climb one step.
- Where the drag doesn't change level at all — or the style has no
  `stair.nbt` — every cell in it is flat at the first cell's level. That is
  the user's first rule, and it falls out of this one rather than being a
  separate mode.

### Joining an existing road

A drag's first/last level is read off the *city*, not the terrain, whenever
the city already has an answer:

1. the cell's own recorded `base_y`, if it's already a road cell (this is
   what makes re-crossing existing road a no-op rather than a re-levelling);
2. otherwise a cardinally-adjacent existing road cell's `base_y`, so a new
   drag starting next to an existing road meets it flush instead of at
   whatever the ground happens to be;
3. otherwise `grid::fit_footprint`'s reading of the terrain.

### Where the stair cells go

Spread as evenly as the path allows. A cell is eligible to be a stair only
if it is neither the first nor the last cell of the drag *and* it is a
straight run — the L-shaped path's corner cell can't be a bend and a ramp at
once, and neither can a cell that a third road branches into (a T or a
cross). If there aren't enough eligible cells for the steps the drag needs,
the whole drag is refused with a message saying so, the same all-or-nothing
shape `try_commit_drag` already applies to validity.

## Scope

- **`road::RoadPieceKind::Stair`** — a seventh kind, loaded from
  `stair.nbt`. Never returned by `select_piece` (a cell's *connections* can't
  tell you it's a ramp); it's chosen from the cell's recorded ascent instead.
  Canonical orientation: connects north + south, ascending toward **north**,
  consistent with the ticket-066 convention that every piece opens south.
- **`state::RoadCell::ascent: Option<Direction>`** — `None` for a flat cell;
  `Some(d)` for a stair whose *low* end is at `base_y` and whose high end,
  on side `d`, is at `base_y + ROAD_STAIR_RISE`. Stored, not derived: the
  same "the city state is authoritative, and a height is decided once"
  reasoning `base_y` itself carries (ticket 065).
- **`road_build::plan_drag`** — the whole rule above as one pure function
  from (path, world, city, "does this style have a stair piece") to a
  per-cell `(base_y, ascent)` plan, or a refusal. Shared by the preview and
  the commit so the ghost can't disagree with what gets written, the same
  way ticket 065 made them share `cell_write_origin`.
- **`road_build::road_write_edit`** — a cell with an `ascent` writes the
  `Stair` piece rotated to that direction, instead of the connection-derived
  kind.
- **`road_build::update_drag_preview`** — previews the plan's heights; a
  refused plan previews flat at the start level, tinted invalid.
- **`persistence`** — `SavedRoadCell` gains `ascent`; `CURRENT_VERSION`
  4 -> 5, refused-not-guessed like every bump before it.
- **`assets/city/roads/dirt/README.md`** — how to author `stair.nbt`.

## Out of scope

- **The `stair.nbt` asset itself.** It isn't in the repo (the user has one in
  Minecraft; nothing has been exported). A missing piece is skipped, not
  fatal — same tolerance every other kind gets — so until it lands, a stair
  cell is recorded in `City` and writes no blocks. Everything else in this
  ticket works without it: with no stair piece loaded, `plan_drag` puts the
  whole drag flat at the first cell's level.
- **Re-tiling a stair.** A cell that was placed as a stair stays a stair even
  if a later drag connects a third road into its side; that branch simply
  won't meet it. Demolishing and re-dragging is the workaround. Roads still
  aren't journaled (ticket 055's own note), so there's no undo either way.
- **Terraforming under the road.** A road held at one level across rising
  ground will be buried where the terrain is higher than the road. The dig/
  level tools (ticket 057) are the existing answer; auto-cutting a cutting is
  a project of its own.

## Done when

- A drag across sloped ground produces one contiguous road: every cell at
  the first cell's level, or stepping in whole 4-block steps through stair
  cells.
- The last cell's level snaps to the first's plus a multiple of 4.
- A stair cell's high end is exactly level with its uphill neighbour's
  surface course.
- A drag needing more steps than it has eligible cells is refused whole.
- A road cell round-trips through `city.ron` with its ascent; a version-4
  file is refused, not silently read as flat.
- `cargo check --all-targets` and `cargo test --lib` clean.
- Whether the road *looks* continuous is a human-at-the-window check, noted
  in `todo.md`.

## Resolution

Landed as scoped, across five files.

**`road.rs`** — `RoadPieceKind::Stair` (a seventh kind, loaded from
`stair.nbt`, never returned by `select_piece`), `canonical_pattern`'s arm
for it (north + south, ascending north — consistent with ticket 066's
"every piece opens south"), `stair_rotation`, and `Direction::opposite` /
`rotation_from_north` built on a single shared `CLOCKWISE` table so a stair's
rotation can't disagree with `Direction::rotated`. `Direction` gained
`Serialize`/`Deserialize` so `persistence` uses the real type rather than a
mirror enum.

**`state.rs`** — `RoadCell::ascent: Option<Direction>`, kept by
`add_road_cell`'s existing idempotence (a drag re-crossing a stair doesn't
flatten it), and `base_y` re-documented as a stair's *low* end.

**`road_build.rs`** — the new "height" section: `ROAD_STAIR_RISE`,
`CellPlan`, `PlanRefusal`, `level_on_edge`, `anchor_level`, `step_direction`,
`stair_eligible`, `spread_evenly`, `plan_drag`, `stair_available`,
`piece_for`. `try_commit_drag` and `update_drag_preview` both go through
`plan_drag` (replacing per-cell `cell_height`), and `road_write_edit` goes
through `piece_for`. A refused plan still previews — flat at the start level,
tinted invalid throughout — rather than the ghost blinking out unexplained.

**`road_catalogue.rs`** — `filename_for(Stair) = "stair"`; the tests that
counted "six kinds" now count `RoadPieceKind::ALL.len()`.

**`persistence.rs`** — `SavedRoadCell::ascent`, `CURRENT_VERSION` 4 -> 5.
Two fixtures that had version numbers hard-coded for reasons unrelated to
versioning now interpolate `CURRENT_VERSION`, so the next bump doesn't break
them for the wrong reason.

### Tests

Fourteen new ones. The load-bearing ones:

- `a_drag_across_a_slope_is_flat_at_the_first_cells_level_when_no_stair_exists`
  — the user's original rule, directly.
- `a_four_block_climb_becomes_one_stair_with_flat_runs_either_side` and
  `a_descending_drag_records_the_stairs_low_end_and_an_uphill_ascent` — a
  stair's `ascent` always points *uphill*, which on a descending drag means
  back the way the path came.
- `the_end_level_snaps_to_a_whole_number_of_stair_steps` — the 4-grid,
  including the halves-away-from-zero rounding (6 blocks is two steps, not
  one) and that the *start* level is never snapped, since it's the anchor.
- `a_stairs_high_end_is_level_with_its_uphill_neighbours_surface` — the
  arithmetic everything rests on, checked through `cell_write_origin` rather
  than restated.
- `the_corner_of_an_l_shaped_drag_is_never_a_stair`,
  `a_climb_with_nowhere_to_put_the_stairs_is_refused_whole`.
- `a_drag_starting_on_an_existing_road_cell_continues_at_its_recorded_level`,
  `a_drag_starting_beside_an_existing_road_cell_meets_it_flush`,
  `joining_a_stair_reads_the_level_of_the_edge_being_joined`.
- `road_write_edit_writes_a_stair_for_a_cell_with_a_recorded_ascent` — the
  write-path half: a cell with two opposite neighbours would be a `Straight`
  without its ascent.
- `persistence::an_old_ascentless_road_cell_save_is_refused_not_silently_defaulted`.

`cargo check --all-targets` clean; `cargo test --lib` — 591 passed, 0 failed.
Visual confirmation, and the stair-specific half of it once `stair.nbt`
exists, are in `todo.md`.
