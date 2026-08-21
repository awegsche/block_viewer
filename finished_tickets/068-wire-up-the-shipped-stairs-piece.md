# 068 - the real `stairs.nbt` lands, and the loader has to actually find it

## Report

Ticket 067 shipped the stair *algorithm* with the asset explicitly out of
scope — there was no stair piece in the repo, and a missing piece is skipped
rather than fatal, so every drag stayed flat. The user exported one into
`assets/city/roads/dirt/` while 067 was being written. (It landed in 067's
own commit as an untracked file swept up by `git add -A`, which is why this
is a separate ticket rather than part of that one.)

## What actually shipped, probed

`assets/city/roads/dirt/stairs.nbt`, `6x8x6`:

- `y=0` — solid dirt subgrade across the whole cell, same as the flat pieces
- `y=1` — the road surface (`dirt_path` between cobblestone-stair kerbs) on
  the **south** edge: the low end
- `y=2..5` — one cobblestone-stair step per layer, walking north, with solid
  dirt fill behind each
- `y=5` — the road surface again on the **north** edge: the high end
- `y=6..7` — air clearance

So it connects **north + south**, ascends toward **north**, and rises
`5 - 1 = 4` blocks — exactly `ROAD_STAIR_RISE`, and exactly the canonical
orientation ticket 067 defined and 066's convention predicts ("every piece
opens south"; a stair's south edge is its low end). Nothing about the
algorithm needs changing.

## What does need changing

1. **The filename.** `road_catalogue::filename_for` expects `stair.nbt`; the
   export is `stairs.nbt`. The asset wins — `stairs` is what the Minecraft
   block family is called, and renaming a file the user exported to satisfy
   a table is the wrong way round.
2. **The orientation guard test.** Ticket 066's
   `the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations`
   reads each piece's open edges off *one* layer (`ROAD_PIECE_SUBGRADE_DEPTH`).
   That's the whole story for a flat piece and meaningless for a ramp, whose
   two ends are four layers apart — at `y=1` the stair's north edge is solid
   fill, so the flat-piece rule would read it as "opens north, closed south",
   the exact opposite of the truth. `Stair` is skipped there and gets its own
   test instead, which is stronger: it pins the low surface to the south edge
   at `y = ROAD_PIECE_SUBGRADE_DEPTH`, the high surface to the north edge at
   `y = ROAD_PIECE_SUBGRADE_DEPTH + ROAD_STAIR_RISE`, and therefore both the
   ascent direction *and* the rise the whole plan arithmetic assumes.
3. **A rotation guard.** `blueprint::rotate_blueprint` refuses a palette
   entry it has no rule for (`RotationError::UnrotatableProperty`), and
   `road_write_edit` handles that by printing a line and **skipping the
   cell** — an invisible hole in a road, one console line deep. The stair
   piece is the most property-dense one yet (cobblestone stairs with
   `outer_left` shapes, oak fences with four connection booleans), so every
   shipped piece now gets rotated through all four rotations in a test.

## Done when

- `stairs.nbt` loads into `RoadCatalogue` under `RoadPieceKind::Stair`, so a
  drag with a 4-block level change actually builds a ramp.
- The stair's geometry is pinned by a test, not just by the README.
- Every shipped piece rotates through all four rotations without error.
- `cargo check --all-targets` and `cargo test --lib` clean.
- Whether the ramp *looks* right in-game is the human check already queued in
  `todo.md` under 067.

## Resolution

Landed as scoped.

- `road_catalogue::filename_for(Stair)` -> `"stairs"`. The `RoadPieceKind`
  variant stays singular; only the filename follows the asset.
- `the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations` skips
  `Stair` alongside `Isolated`, with the reason written down.
- New `the_shipped_stair_climbs_north_by_exactly_one_stair_rise`: reads the
  real file, asserts the surface course paves the **south** edge at
  `y = ROAD_PIECE_SUBGRADE_DEPTH` and the **north** edge at
  `y = ROAD_PIECE_SUBGRADE_DEPTH + ROAD_STAIR_RISE`, and that the north edge
  is *not* road at the low layer (which is what tells a ramp from a flat
  straight). Pins the ascent direction and the rise together.
- New `every_shipped_piece_rotates_through_all_four_rotations`: every style,
  every kind, all four rotations through `blueprint::rotate_blueprint`,
  asserting `Ok` and a preserved (square) size. Passes — the stair's
  `outer_left`/`outer_right` cobblestone stairs and four-way oak fences are
  all covered by `blueprint::rotate`'s existing table.
- `README.md`, `road`'s module docs, `road_build`'s module docs and the
  roadmap updated to `stairs.nbt` and to the piece's real `6x8x6` shape.
- `todo.md`'s 067 entry item 4 rewritten from "once the asset exists" to a
  live check, including the two-ramp and refusal cases.

`cargo check --all-targets` clean; `cargo test --lib` — 593 passed, 0
failed (also clean under `--test-threads=1`).
