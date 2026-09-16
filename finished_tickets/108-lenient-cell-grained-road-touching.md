# 108 - Road connectivity: cell-grained, not block-grained, touching

## Why

Requested directly: the "not connected to a warehouse" check (ticket 107's
display of `Coverage`, itself built on `road::touching_road_cells`) was too
strict. It required a road cell to sit in the exact block immediately next
to a footprint tile — zero tolerance for any gap, and no forgiveness for a
building whose footprint doesn't reach all the way to the edge of the
[`ROAD_CELL_SIZE`]-block cell it falls in. A player who visibly surrounded a
building with road still saw "not connected" whenever the building's own
footprint didn't happen to touch the exact cell boundary.

## Scope

- `city::road::touching_road_cells`: resolve every footprint tile to the
  road cell it falls in (fully or partially covering that cell), then check
  *that* cell's own cardinal neighbours for a road, instead of checking each
  footprint tile's single block-adjacent neighbour tile. A building now only
  needs a road on any cell adjacent to one it covers — it no longer has to
  be flush against the building's actual blocks.
- Updated the function's and the module's doc comments to describe the new
  cell-grained algorithm.
- Added a test (`touching_road_cells_finds_a_road_on_the_far_side_of_a_partially_covered_cell`)
  pinning the new leniency: a footprint that only clips a corner of a cell,
  nowhere near that cell's own edge, still counts as touching a road on the
  far side of it.

## Out of scope

Not touching `Coverage`/`compute_coverage`'s own radius-and-travel-time
logic, or `is_connected`/`reachable_from` — this only widens what counts as
a building "touching" a road cell in the first place; everything downstream
of `touching_road_cells` is unchanged.

## Status

Done — `cargo test` green (978 tests, including the existing F4 suite and
the new leniency test).
