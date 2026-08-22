# 079 - Warehouses: working radius, road coverage, and storage capacity

## Status
Open

## Why

Roadmap H2: "a producer's output has to reach a warehouse, and how long that
takes comes from the road distance". This ticket is the warehouse itself and
the coverage question — *which* producers a warehouse serves and how far each
one is along the road. Ticket 080 is the haul that uses the answer.

It is also the first consumer of roadmap F4 (ticket 056), whose
`touching_road_cells`/`reachable_from`/`is_connected`/`buildings_reachable_from`
have carried `#[allow(dead_code)]` and the note "a later unconnected-building
UI warning, or **logistics in a later iteration**, are the eventual readers"
since they landed. This is that reader.

## Tiers are the tier tree, not a new concept

A "tier 2 warehouse" is a second `.ron` with bigger numbers and a `requires`
on the first — `definition::Building.tier`/`requires` and
`resolve_requirements` (ticket 041) already are the tier mechanism, and
`ui::build_menu` already groups by it. Nothing here adds a parallel notion of
tier.

## Schema

`definition::Building.warehouse: Option<Warehouse>` — the four knobs a tier
moves:

```
Warehouse(
    radius_cells: 8,        // how far along the road it reaches
    concurrent_hauls: 2,    // how many stacks it can have in flight at once
    handling_minutes: 0.5,  // fixed load/unload cost added to every haul
    storage: 4096,          // what it adds to the city's storage capacity
)
```

`radius_cells` and `concurrent_hauls` must be `> 0` — a warehouse that
reaches nowhere or can carry nothing is a definition error, not a value
downstream code should defend against, same call `road_definition` makes for
`travel_speed`/`capacity`.

## Coverage: one BFS, then one Dijkstra

`city::warehouse::compute_coverage(&City, &BuildingDefinitions, &RoadTypes)`
-> `Coverage`, rebuilt each tick from `City` and never stored — the same "no
new stored state, so it can never go stale" rule `city::road`'s module docs
set for the road graph itself.

Per placed warehouse:

1. **Coverage** is a BFS over road cells from the warehouse's own
   `touching_road_cells`, to a depth of `radius_cells` hops. Hops, not time —
   "working radius" is a distance, and a fast road should make a haul quicker,
   not make the warehouse reach further.
2. **Travel time** is then a Dijkstra *restricted to the covered set*,
   weighting each cell entered by `1.0 / road_type.travel_speed` minutes per
   cell. A style with no `.ron` in `assets/city/road_types` falls back to
   `travel_speed: 1.0` rather than being impassable — a road with no
   definition is undescribed, not broken, which is the same call
   `road_definition`'s module docs already make.
3. A producer is served by the covered warehouse it can reach in the **least
   time**, ties broken by the lower `BuildingId` so the answer is stable
   frame to frame.

A producer that touches no road cell, or whose road island holds no
warehouse, is **unserved** — the deliberate consequence of measuring the
radius along the road. This is what makes the road network the thing the
economy runs on rather than decoration, and it is what `travel_speed` has
been waiting for since ticket 060.

Restricting the Dijkstra to the BFS-covered set (rather than pruning by hop
count inside one pass) is what makes "within the radius" and "the fastest
route" two separate, exactly-answerable questions instead of one
approximation of both.

## Storage capacity

`StorageCapacity(u64)` — `economy.base_storage + sum(storage)` over placed
warehouses, recomputed alongside coverage. `base_storage` (default 2048)
exists so a city with no warehouse can still hold its founding grant; without
it a fresh save would be over capacity before its first click.

This does cut against ticket 072's "one global pile, unbounded", and the
reconciliation is that a cap is a *warehouse* property being enforced against
the pile, not per-warehouse storage: goods still don't live anywhere.
`inventory::Stock` gains `add_parcel_capped(&parcel, capacity) -> Parcel`
(the overflow), and the callers that credit the stock — `demolish`,
`terraform`, `undo`, and 080's deliveries — go through it. Overflow is
**reported, never silent**: `WriteStatus` and the city panel say how much
was lost. `Stock::add`/`add_parcel` stay uncapped as the primitive, since a
placement's refund is settling a spend that already fit.

## Assets

Ship three definitions, all reusing `house01.nbt` as placeholder geometry
(nothing else exists on disk), with `footprint: Explicit` sized to what they
should eventually be:

- `warehouse01.ron` — tier 1, the numbers above.
- `warehouse02.ron` — tier 2, `requires: ["warehouse01"]`, roughly double.
- `farm01.ron` — tier 1, a producer (`outputs: [(item: "minecraft:wheat",
  per_minute: 12.0)]`), so 078 and 080 have something to actually run.

`todo.md` gets the note that these three want real `.nbt` exports, and that a
human has to look at the result.

## UI

The city panel's Production section (078) gains, per producer, the warehouse
serving it and the one-way travel time — or **"unserved"** in red, which is
the whole diagnostic for "why has my farm stopped".
