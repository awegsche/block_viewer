# 084 - Farm tiles: Anno-style radius production scaling

## Status
Open — after 083.

## Depends on
083 (the button lives in the inspect panel it adds).

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

## Coverage
New, small — built the same shape as `city::warehouse`'s coverage but
simpler: no road graph, straight-line distance from a tile's footprint to a
hub's footprint (Chebyshev, matching the square-grid feel of `city::grid`),
**nearest hub wins** on overlap (ties broken by `BuildingId` order — an
arbitrary-but-deterministic tie the codebase doesn't sweat elsewhere
either). Result: `tiles_in_range(hub) -> u32`, read by `advance_producer`
and multiplied into both `outputs` and `inputs` per-minute rates before
they're integrated — an under-tiled hub slows down rather than stalling
outright the way a full buffer does.

Recomputed on demand (every producer tick — cheap at this building count)
rather than cached and invalidated, the same "nothing here can go stale"
call `city::road`'s connectivity queries already made.

## UI
The hub's inspect panel (083) grows a "Place Farm Tile" button when
`Building::farm` is `Some` — same effect as clicking the tile's own
build-menu row (`PlacementSelection` set, `ActiveTool::Building`), just
reachable from the hub you're already looking at.

## lumber.ron / tree_plot.ron
`lumber.ron` gains a `farm` block once real numbers are picked (placeholder
to start — guessed, not measured, like every other tuning knob here).
`tree_plot.ron` needs its own blueprint; until `tree_plot.nbt` is exported
this points at `house01.nbt`, per the established placeholder-geometry
convention (farm01/warehouse01/02 already do this).

## Not in scope here
Whatever mechanic actually consumes/regrows a tile's trees over time (a real
Anno farm cycles plots) — this ticket is the count-and-scale relationship
only, not a resource on the tile itself.
