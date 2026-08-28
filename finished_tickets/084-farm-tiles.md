# 084 - Farm tiles: Anno-style radius production scaling

## Status
Done — `cargo test --lib` green (788 passed).

Landed as designed below, with two deliberate deviations:

- **Picked up out of order, ahead of 083.** 083 (inspect mode) hadn't
  landed when this was picked up; the "Place Farm Tile" button that was
  meant to live in its inspect panel is deferred to whenever 083 does land
  (see "Not in scope here", below) — everything else in this ticket needed
  nothing from 083. The build menu's ordinary catalogue row already places a
  tile once its hub has unlocked it (`requires: ["lumber"]`), so a player
  isn't blocked, just one click further from the hub than 083 will make it.
- **Field names are `radius_blocks`/`tiles_for_full_rate`, matching this
  ticket's own schema below** — the geometry/blueprint file that prompted
  picking this up (`lumber_farm_01.nbt`) was added to the repo before this
  ticket was read, and a first pass implemented a similar-but-not-identical
  schema (`radius`/`tiles_for_full_output`, only scaling outputs, and
  letting a tile double-count toward two overlapping hubs) before this file
  was found and the implementation was brought in line with it.

## Depends on
083 (the button lives in the inspect panel it adds) — soft dependency; see
"Status" above for what shipped without it.

## Why
The tree farm's design: production isn't a flat `per_minute` the way
farm01's wheat is — it scales with how many "farm tile" buildings sit within
a radius of the hub, 0 at none up to 100% at some cap, linearly in between.
This is the first building relationship that's neither the tech tree
(`requires`) nor the road network (warehouse coverage) — straight-line
distance between two placed buildings.

## Schema
`definition::Building` gains:

```
farm: Some(Farm(
    radius_blocks: 24,       // straight-line, not road cells: Anno-style
                              // tile placement, not haulage
    tiles_for_full_rate: 8,  // linear ramp: tiles_in_range / this, capped
                              // at 1.0
    tile: "tree_plot",       // definition id of the tile building this hub
                              // accepts
)),
```

A building named by `tile` is otherwise an ordinary `Building` — its own
blueprint, footprint, cost, `requires: ["lumber"]` (or whichever hub) so
it's locked until the hub exists, reusing C3's unlock rule verbatim. No
`production` of its own; no `farm` of its own (a tile that's also a hub is
out of scope).

Shipped as `lumber.ron` (the hub, `farm.tile: "lumber_farm_01"`) and
`lumber_farm_01.ron`/`lumber_farm_01.nbt` (the tile) rather than the
placeholder `tree_plot` name above — real geometry for both landed together,
so there was no placeholder-`.nbt` stage to go through.

## Coverage
Built the same shape as `city::warehouse`'s coverage but simpler: no road
graph, straight-line distance from a tile's footprint to a hub's footprint
(Chebyshev, matching the square-grid feel of `city::grid`), **nearest hub
wins** on overlap (ties broken by `BuildingId` order — an
arbitrary-but-deterministic tie the codebase doesn't sweat elsewhere
either). Result: `tiles_in_range(hub) -> u32` (shipped as
`FarmCoverage::tiles_near`), read by `production::tick` (via
`scale_production`, ahead of `advance_producer`) and multiplied into both
`outputs` and `inputs` per-minute rates before they're integrated — an
under-tiled hub slows down rather than stalling outright the way a full
buffer does.

Recomputed on demand — landed as change-detected (`City`/`BuildingDefinitions`
`is_changed()`) rather than literally every producer tick, the same
`Coverage`-vs-every-tick call `city::warehouse` already made; a city's
buildings change on a click, not a tick, so this is cheaper than "every
producer tick" for the identical end result and follows the precedent
already set right next to it.

## UI
The hub's inspect panel (083) grows a "Place Farm Tile" button when
`Building::farm` is `Some` — same effect as clicking the tile's own
build-menu row (`PlacementSelection` set, `ActiveTool::Building`), just
reachable from the hub you're already looking at. **Not shipped** — waits on
083 itself; see "Status".

Shipped instead, since 083's panel doesn't exist yet: the build menu's own
catalogue row for a farm-linked hub names what it needs
(`Scales with <tile name> nearby (N needed, within R blocks)`), and the city
panel's per-producer line appends a live `count/needed` tile fraction next
to its running/starved/buffer-full state — both read straight off
`FarmCoverage`, so 083's button will have real data the moment it lands.

## lumber.ron / tree_plot.ron
`lumber.ron` gains a `farm` block once real numbers are picked (placeholder
to start — guessed, not measured, like every other tuning knob here).
`tree_plot.ron` needs its own blueprint; until `tree_plot.nbt` is exported
this points at `house01.nbt`, per the established placeholder-geometry
convention (farm01/warehouse01/02 already do this).

Landed with real geometry for both instead: `lumber.ron` (`farm.tile:
"lumber_farm_01"`, `radius_blocks: 16`, `tiles_for_full_rate: 3`,
`production: Some(...)` producing `minecraft:oak_log` at 8.0/min full rate)
and `lumber_farm_01.ron` (`requires: ["lumber"]`, no production or farm of
its own, cost `4x oak_log + 16x dirt`).

## Not in scope here
Whatever mechanic actually consumes/regrows a tile's trees over time (a real
Anno farm cycles plots) — this ticket is the count-and-scale relationship
only, not a resource on the tile itself. The "Place Farm Tile" inspect-panel
button (needs 083). Chest-based output — `lumber.ron`'s pre-ticket-085
comment floated scanning a blueprint's chests directly rather than going
through 078's abstract per-building buffer; superseded, not deferred, once
this ticket settled on the tile-count mechanic instead.
