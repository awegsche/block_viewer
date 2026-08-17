# 053 - Road graph on the grid

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 465
tests, up 15 from this ticket's own suite). See the Resolution.

## Part of

Roadmap group F (`tickets/CITYBUILDER_ROADMAP.md`): F1, "the road graph."
First task of the streets group (milestone M5), following G's build
menu/city panel (ticket 050). Depends on D1's `City` (ticket 042), which
already tracks road tiles in its occupancy grid via `add_road`/`roads()`.

## Problem

`state::City` already has a `HashSet<IVec2>` of road tiles
(`add_road`/`remove_road`/`roads()`, landed with D1) but no notion of which
road tiles are *adjacent* to which others. F3 (auto-tiling) needs to know,
per tile, which of its four neighbours are also road, to pick a straight/
corner/T/cross/end piece. F4 (connectivity queries — "is this building on
the road network", "what does this segment reach") needs to walk that
adjacency to answer reachability. Neither exists yet, and both read the same
underlying structure, so the roadmap calls it out as its own task ahead of
either: "iteration 1 needs the graph even without logistics, because F3 and
F4 both read it."

## Goal

Per the roadmap: "Tiles plus adjacency, on the same grid as E2."

- A per-tile connection query: given a road tile, which of its four cardinal
  neighbours (north/south/east/west, Minecraft's own x/z convention) are
  also road tiles. This is what F3's auto-tiling will pick a piece from.
- A reachability query over that adjacency: every tile reachable from a
  given road tile through an unbroken chain of road tiles. This is the
  primitive F4's "what does this segment reach" and "is this building on
  the network" (test the tile(s) adjacent to the building's footprint for
  membership in the reachable set) will both be built on.

## Scope

- A new `city::road` module, following D1's own precedent: no new stored
  state — the road graph is *derived* from `City::roads()`/`City::occupant_at`
  on demand, the same "the city state is authoritative" rule the whole
  citybuilder follows (`state.rs`'s own module docs). No caching, no
  invalidation to get wrong: `City`'s occupancy grid is already the single
  source of truth for which tiles are road.
- Cardinal direction offsets in Minecraft's own convention (north = `-z`,
  south = `+z`, east = `+x`, west = `-x`) — this module works in the same
  Minecraft `(x, z)` tile space `state::footprint_tiles` does, not a Bevy
  transform, so no `bevy.z = -mc.z` flip applies here.
- `RoadConnections`: which of a tile's four neighbours are road, plus a
  count (0 = isolated tile, 1 = dead end, 2 = straight or corner depending
  on which two, 3 = a T, 4 = a cross) — exactly the classification F3's
  auto-tiling will switch on.
- `reachable_from`: BFS over road tiles from a starting tile.
- `is_connected`: two road tiles in the same reachable set.

## Out of scope

- F2 (drag-to-build), F3 (auto-tiling pieces), F4's own building-facing API
  (this ticket is the primitive, not the "is this building on the network"
  question itself — that's F4's job, once buildings' adjacent tiles are the
  ones being tested).
- Any rendering of roads — no mesh, no pieces, no material. Roads currently
  render as nothing at all; that's unchanged here.

## Done when

- `cargo build`/`cargo test` clean.
- Tests: direction offsets are mutually opposite and sum to the four
  cardinal points; `RoadConnections` correctly reports 0/1/2/3/4-neighbour
  configurations including which specific neighbours, off tiles that are
  road, non-road, and off the grid entirely; `reachable_from` covers a
  straight run, a branch, a loop (must not infinite-loop or double-count),
  and two disconnected road islands (each reaches only itself); a tile
  that isn't a road reaches nothing (not even itself); `is_connected` true
  within one island, false across two, false when either endpoint isn't a
  road tile at all.
- No manual verification needed — this ticket adds no rendering and no
  input; F2/F3 are what a human will eventually see.

## Resolution

Landed as scoped: a new `city::road` module, no new resource and no plugin —
`connections_at`/`reachable_from`/`is_connected` all take `&City` and
recompute their answer against its occupancy grid on every call, the same
"derived, never stored" choice `city::grid::fit_footprint` already made for
terrain fit. `Direction`'s four offsets follow Minecraft's own convention
(north `-z`, south `+z`, east `+x`, west `-x`) rather than inventing a
screen-relative one — this module never touches a Bevy `Transform`, so the
`bevy.z = -mc.z` flip other modules carry doesn't apply here at all.

`connections_at` deliberately doesn't require `tile` itself to be a road —
asking "what would connect here" before a tile is placed is exactly what
F2's drag-to-build preview will need, the same way E3's ghost preview asks
`fit_footprint` before `City::place_building` ever runs. It also reads
`Occupant::Road` specifically, not "is this tile occupied" — a building
standing where a road would connect must not read as a road neighbour.

`reachable_from` is a plain BFS over `is_road` neighbours, `HashSet`-visited
so a loop can't enqueue a tile twice or spin forever. Its empty-set-for-a-
non-road-start behaviour is deliberate, not a shortcut: it means "reaches
only itself" (a one-tile road island, `HashSet` of size 1) and "isn't a
road at all" (empty set) are distinguishable results rather than the same
one-element answer, which matters once F4 asks "is this building on the
network" by testing a footprint's adjacent tiles against a reachable set.
`is_connected` is the two-tile read of the same primitive, with its own
short-circuit: both endpoints must themselves be road tiles, not merely
present in some `reachable_from` set (which they trivially would be for
`a` itself if `a` starts the walk).

No caller yet — every public item is behind an `#[allow(dead_code)]` with a
comment naming the future caller, the same "proven, not yet used" pattern
tickets 039/040/042 left their own resources in until a later ticket in
this same group picked them up. F2 (drag-to-build) is the first one due.

Testing: 15 new tests in `city::road`, covering direction-offset symmetry,
`RoadConnections` against 0/1/2/4-neighbour tiles and a building-occupied
neighbour, and `reachable_from`/`is_connected` across a straight run, a
branch, a loop, a single-tile island, and two disconnected islands.
`cargo test --lib` — 465 passed, 0 failed.
