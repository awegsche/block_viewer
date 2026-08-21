# 065 - roads are written at Y=0 instead of ground level

## Report

User: "I tried building roads in the citybuilder. but I cant see anything."
Then, after the diagnosis: "I still don't see anything appearing after
building the road. the log shows: N road tiles built. but nothing is
visible. I tried saving, closing and reopening. but still nothing visible."

## Root cause

`road_build::road_write_edit` builds each cell's edit with

```rust
let corner = cell_min_corner(cell);   // IVec3::new(cell.x * 6, 0, cell.y * 6)
let edit = blueprint_edit(blueprint, corner);
```

`cell_min_corner`'s Y is a hardcoded `0`, and `commit::blueprint_edit`
offsets every block in the piece by that origin. So every road piece is
written into the deepslate at Y=0..piece_height, ~60 blocks under the
terrain. The write genuinely succeeds — hence "N road tiles built" and the
survival across save/reload — it just lands somewhere nobody looks.

The *preview* never had this problem: `cell_height` reads
`grid::fit_footprint`'s `base_y` and `cell_transform` puts the ghost there.
The ghost path and the write path simply never shared a height.

Buildings are unaffected: `commit::try_commit_placement` passes an `origin`
that already carries `placement::resolve_placement`'s `base_y`.

## Why the fix isn't just `0` -> `cell_height(...)`

`road_write_edit` also rewrites *already-placed* neighbours whose piece
shape changed (`affected_cells` — a dead end that grew a neighbour becomes
a straight). Once a road piece has been written, `grid::ground_height_at`
samples the road surface as ground, so re-deriving a neighbour's fit
returns `base_y + 1` and each re-tile walks that cell one block upward.

The height has to be decided once, when the cell is first placed, and
remembered from then on.

## Second cause: the piece's own subgrade, and the preview's anchoring

Fixing the Y alone still wouldn't have looked right. Probing the shipped
`dirt` pieces (`assets/city/roads/dirt/*.nbt`, ticket 063) shows they are
6x5x6 and laid out:

```
y=0: 36x dirt                                          <- subgrade
y=1: dirt_path / grass_block / cobblestone_stairs      <- surface course
y=2..4: air                                            <- clearance
```

So the piece's *surface* is its layer 1, not layer 0, and the three air
layers above it are deliberate — they mow whatever grew over the road. That
only works if layer 1 lands on the terrain surface (`base_y - 1`, `base_y`
being `fit_footprint`'s "one *above* the ground"). Anchoring the piece's
bottom at `base_y` would have left the paving a block proud of the grass with
the clearance wasted on empty sky.

`cell_write_origin` is that offset, via `ROAD_PIECE_SUBGRADE_DEPTH = 1`. A
constant rather than a `RoadType` field: `assets/city/road_types/*.ron` is
game data (travel speed, capacity) and isn't threaded into the write path at
all — if a style ever ships a different subgrade depth, that's where it goes.

A third bug surfaced with it. `cell_transform` centred the preview mesh in
`(x, z)` — correct for `quad_mesh` (built centred on its own origin), wrong
for a real piece, whose `mesh_blueprint` geometry starts at its minimum
corner like `placement`'s building ghost. Harmless while every cell fell back
to the quad; a 3-block diagonal slide the moment ticket 063's assets started
resolving. `PreviewAnchor` now distinguishes the two, and both the ghost and
the write go through `cell_write_origin`, so they cannot drift apart again.

## Scope

- `state::City`: `road_cells: HashMap<IVec2, String>` becomes
  `HashMap<IVec2, RoadCell>` carrying `style` + `base_y`. `add_road_cell`
  takes the height; idempotent re-add keeps the original, same as it
  already does for `style`.
- `road_build::try_commit_drag`: resolves each cell's height (the same
  `cell_height` the preview uses) and hands it to `add_road_cell`.
- `road_build::road_write_edit`: origin comes from `cell_write_origin`
  (stored `base_y`, offset by the piece's subgrade depth).
- `road_build`: `PreviewAnchor` splits the corner-anchored piece mesh from
  the centred fallback quad, and the preview shares `cell_write_origin` with
  the write.
- `persistence`: `SavedRoadCell` gains `y`; `CURRENT_VERSION` 3 -> 4,
  refused-not-guessed like the 1->2 and 2->3 bumps before it.

## Not fixed by this

Roads built with an earlier build are still sitting at Y=0 in the world's
region files. Nothing here goes back and removes them — they'd have to be
dug out in-game or the affected chunks regenerated. And an existing
`city.ron` is version 3, so it's refused on load (`city save is version 3,
this build reads version 4`) and its road cells are lost; the buildings in it
go with it. Deleting `<save>/citybuilder/city.ron` clears the message.

## Done when

- A committed road cell's blocks land at the cell's own fitted ground, with
  the piece's surface course flush with the terrain — the same place its
  ghost previewed at.
- Re-tiling an existing neighbour reuses that neighbour's recorded height
  rather than resampling terrain that now includes the road itself.
- A road cell round-trips through `city.ron` with its height.
- `cargo check` and `cargo test --lib` clean (bar the two pre-existing
  save-dependent failures noted in ticket 062).

## Verification

- `cargo check --all-targets` — clean.
- `cargo test --lib` — 570 passed, 0 failed (including the two ticket 062
  flagged as environment-dependent, which pass on this machine's saves).
- New tests:
  - `state::re_adding_an_existing_road_cell_keeps_its_original_base_y` — the
    drift guard: re-adding a cell at a *higher* Y keeps the original.
  - `state::road_cell_at_reports_the_style_and_height_a_cell_was_built_with`
  - `state::road_cells_with_data_iterates_every_cell_with_its_style_and_height`
  - `road_build::road_write_edit_writes_relative_to_the_cells_recorded_base_y_not_zero`
  - `road_build::cell_write_origin_puts_the_surface_course_at_the_terrain_surface`
  - `road_build::road_write_edit_uses_each_cells_own_base_y` — two cells on
    different ground don't flatten to one shared Y.
  - `road_build::cell_transform_stands_a_piece_ghost_exactly_where_the_write_lands`
  - `persistence::an_old_heightless_road_cell_save_is_refused_not_silently_defaulted`
- Manual/visual verification (does the road actually appear, flush with the
  terrain, with the ghost standing where the blocks land) is noted in
  `todo.md` — needs a human watching the window, per this repo's convention.
