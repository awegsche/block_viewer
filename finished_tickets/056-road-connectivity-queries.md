# 056 - Road connectivity queries

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 509
tests, up from 496). See the Resolution.

## Part of

Roadmap group F (`tickets/CITYBUILDER_ROADMAP.md`): F4 ("connectivity
queries") — the last piece of group F named in the roadmap that doesn't need
a real `.nbt` road piece to exist, per the ordering advice ("F is more
independent than it looks... nearly free once F1 exists").

## Problem

`city::road` (053/054) already has `reachable_from` (BFS over connected road
cells) and `is_connected` (the two-cell membership test), but both carry
`#[allow(dead_code)]` — nothing outside their own tests calls them. There is
also no bridge from a road cell to a *building*: `state::City` tracks
buildings and road cells in the same occupancy grid, but nothing answers "is
this building's footprint actually touching a road cell" or "which buildings
does this road network reach."

## Goal

Per the roadmap: "'Is this building on the road network', 'what does this
road segment reach'." No consumer yet in iteration 1 (same as the roadmap
says) — this is substrate for a later UI affordance (an unconnected-building
warning) or for logistics (a later iteration), proven by tests the way
042/053/054 landed their own primitives.

## Scope

- A tile-coordinate -> road-cell mapping shared between `road` and
  `road_build` (the latter already has a private `cell_of`; this needs the
  same math from a building's `(x, z)` footprint tile, not a block's).
- Which road cells are orthogonally adjacent to a building's footprint.
- "Is this building on the road network" — a building touching at least one
  road cell.
- "Are these two buildings connected" — via the existing `reachable_from`/
  `is_connected` primitives, not a second BFS.
- "What buildings does a road segment reach" — the buildings-oriented view of
  `reachable_from`.

## Out of scope

- Any UI or consumer — same "proven, not yet used" state F1-F3 themselves
  landed in.
- Logistics/production reading any of this — C3/H2 are still inert.

## Done when

- `cargo build`/`cargo test --lib` clean.
- Tests cover: a building with no adjacent road cell reads as unconnected; a
  building actually touching a road cell (including a large footprint
  touching more than one cell along one edge) reads as connected; two
  buildings on the same network read as connected, two on disconnected
  islands don't; a road segment's reachable buildings match the network it
  actually spans, not every building in the city.
- `reachable_from`/`is_connected` lose their `#[allow(dead_code)]` — this
  ticket is their first real caller.

## Resolution

Landed as scoped, entirely inside `city::road` plus one shared helper moved
to `city::state`.

**`state::cell_of(tile: IVec2) -> IVec2`** (new): the inverse of
`road_cell_tiles`'s `cell * ROAD_CELL_SIZE` corner math, via `div_euclid` so
it stays correct for negative coordinates — plain `/` truncates toward zero,
which would put tile `(-1, -1)` in cell `(0, 0)` instead of `(-1, -1)`.
`road_build::cell_of` (private since ticket 055, working from a 3D block
coordinate) became a one-line wrapper around this rather than keeping its own
copy of the same arithmetic.

**`city::road`** gained four functions, all composed from the two
primitives 053/054 already wrote and left `#[allow(dead_code)]` for exactly
this (`reachable_from`, `is_connected`) rather than a second BFS or a second
occupancy walk:

- `touching_road_cells(city, building: &PlacedBuilding) -> HashSet<IVec2>` —
  every road cell orthogonally adjacent to a building's footprint. Walks
  every footprint tile's four block-adjacent neighbours through `state::cell_of`
  and keeps whichever are road cells. Checking every tile rather than just
  the footprint's outer edge is redundant for interior tiles (an interior
  neighbour is always another footprint tile, never a road cell, so it never
  matches) but correct, and simple beats fast for a query with no caller yet.
  Returns every touching cell, not just the nearest — a footprint wider than
  one `ROAD_CELL_SIZE` cell can border more than one along a single edge.
- `is_building_connected(city, id: BuildingId) -> Option<bool>` — "is this
  building on the road network," the roadmap's own phrasing. `None` for an id
  that isn't currently placed, distinct from `Some(false)` (placed, but
  nothing adjacent is a road cell) — the same `Option`-vs-`bool` distinction
  `city::demolish`'s `occupant_at` already draws elsewhere in this crate. A
  building next to a single isolated road cell (no neighbours of its own)
  still reads as connected — "touches a road cell," not "touches a road cell
  that goes anywhere."
- `buildings_connected(city, a, b: BuildingId) -> Option<bool>` — whether two
  buildings share a network. Deliberately doesn't short-circuit on an empty
  adjacency set; it hands both buildings' touching cells straight to
  `is_connected` and lets that function's own `is_road` checks return
  `false`, so the two-building question is answered by composing 053/054's
  primitives rather than re-deriving their edge cases here.
- `buildings_reachable_from(city, start: IVec2) -> HashSet<BuildingId>` —
  "what does this road segment reach," widened from cells (`reachable_from`)
  to the buildings sitting next to them: every currently-placed building
  whose `touching_road_cells` intersects the reachable set.

All four, plus `reachable_from`/`is_connected` now that they have real (if
still unconsumed) callers, stay `#[allow(dead_code)]` — nothing outside this
module's own tests calls any of it yet, the same "proven, not yet used"
state F1-F3 themselves landed in. A later unconnected-building UI warning, or
logistics in a later iteration, are the eventual readers.

13 new tests: `state::cell_of`'s negative-coordinate correctness (1), plus
12 across the four new `road` functions covering the done-when list above,
including the large-footprint-touches-two-cells case and the
isolated-road-island-still-counts-as-connected case. 496 to 509 per
`cargo test --lib`; `cargo build` stays warning-clean.
