# 109 - Dirt road: travel_speed 10x for faster haul feedback

## Why

Requested directly: hauling speed increased by a factor of 10, the same
"faster visual feedback while iterating on haulage" motive ticket 107 already
applied to `economy.ron`'s `stack_size`.

## Scope

- `assets/city/road_types/dirt.ron`: `travel_speed` 1.0 -> 10.0.
  `warehouse::travel_times` charges `1.0 / travel_speed` minutes per road
  cell entered, so a shipment now crosses the same road 10x faster. `dirt`
  is the only shipped road style, so this covers every road in the game.

## Out of scope

Not touching `handling_minutes` (warehouse dwell time) or
`DEFAULT_TRAVEL_SPEED` (the fallback for a road style with no `.ron`,
`warehouse.rs`) — the ask was road travel speed specifically.

## Status

Done — `cargo test` green (978 tests; nothing pins the real `dirt.ron`
value, only self-contained fixtures).
