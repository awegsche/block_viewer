# 052 - Terrain fit shouldn't see trees/clutter as "the ground"

## Status
Open — not yet picked up. Filed after manually testing placement (tickets
048–050) surfaced the gap; not previously covered by any ticket. Roadmap
group E (`tickets/CITYBUILDER_ROADMAP.md`) is the parent.

## The gap

`city::grid::fit_footprint` (ticket 046, roadmap E2) samples ground height
via `ChunkColumn::topmost_non_air` — literally the highest non-air block in
each footprint column, using `world::is_solid` (not-air) as its only notion
of "ground." E4's commit (ticket 048) then writes every position in the
blueprint, air included ("air is a block," `blueprint_edit`'s own docs) —
so at commit time the building happily overwrites whatever was there. The
two don't agree: **eligibility is computed from blocks the write path is
already prepared to bulldoze.**

Concretely: a tree, a fence post, tall grass, or a player-built shed
anywhere in a footprint reads as "ground" `MAX_FOOTPRINT_STEP` (1 block)
away from the real terrain, or ten blocks away for a tree — so
`fit_footprint` refuses the placement as `TooSteep` even though the actual
terrain underneath is flat and the building would clear the obstruction
without complaint if the fit check weren't in the way. The player has to
manually clear decorative/vegetation blocks by hand in Minecraft before the
game will let them build on ground that was never actually a problem.

The module's own docs already flag the adjacent half of this ("ground" not
distinguishing a lake surface from a hilltop) as deliberately deferred to
I3's block-classification table — but I3 is about *damage detection*,
scanning a building's volume after it exists, and has no eligibility-check
caller. Nothing in the roadmap currently connects a block taxonomy to
`fit_footprint` itself.

## Proposed fix

`ground_height_at` needs a narrower predicate than `is_solid` for what
counts as *terrain* worth fitting against — a per-block-name table (the same
shape `world::tint::build_block_tint_table` and I3's planned classifier both
already use elsewhere) that separates:

- **Terrain** — stone, dirt, grass block, sand, the block families a surface
  is actually made of. This is what `fit_footprint` samples.
- **Clutter** — logs/leaves, flowers, tall grass, fences, signs, anything a
  footprint's own "air is a block" write already intends to clear. Skipped
  when sampling height: `topmost_non_air` becomes "topmost *terrain*
  block," so a tree standing in an otherwise flat clearing no longer reads
  as a 10-block cliff.

Two things this fix must *not* do, both already decided elsewhere and worth
repeating so a future implementer doesn't re-litigate them:

- **No auto-clearing during the fit check.** E2's own "No auto-level" rule
  already ruled out writing blocks from a read-only fit function; the same
  reasoning applies here. The classification only changes what height is
  *sampled*, never writes anything — clearing the clutter is still E4's
  `blueprint_edit` doing exactly what it already does.
- **Don't reach for I3's damage classifier and widen its scope.** I3
  classifies *structural vs. volatile vs. interaction-state* for diffing a
  built building against its baseline — a different question (what counts
  as damage) asked at a different time (after placement). A terrain/clutter
  table for E2 is smaller and simpler than I3's, and conflating the two
  would make E2 wait on I3's tuning, which the roadmap explicitly doesn't
  want ("I2 needs no write path... so group I can land early if it's the fun
  part").

## Scope (when picked up)

- A `TerrainClass { Ground, Clutter }` table in `city::grid` or `world`,
  built once against `BlockRegistry` the way `world::tint`'s table is.
- `ground_height_at`/`topmost_non_air`'s caller in `grid.rs` skips `Clutter`
  blocks rather than treating them as the topmost surface. Likely needs a
  new `ChunkColumn` walk (or a `topmost_non_air`-style method parameterized
  by a predicate) rather than reusing `topmost_non_air` verbatim, since that
  method is also `world::mesh`'s own primitive and shouldn't grow a
  citybuilder-specific notion of "solid."
- Tests: a tree/fence standing on otherwise-flat ground fits; a real slope
  (terrain height varying past tolerance) still refuses; a footprint that's
  clutter all the way down to bedrock (a floating structure with nothing
  else under it) doesn't crash or fit at a nonsensical height.

## Done when

N/A — split into a real ticket, with its own resolution, when picked up.
