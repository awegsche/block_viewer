# 080 - Haulage: a stack travels the road to its warehouse

## Status
Open

## Why

The last of roadmap H2's "still open": production reaching the pile. 078 fills
a buffer, 079 says which warehouse serves it and how far away it is; this
carries the goods between them, in stacks, over game time.

## The model

`Shipment`, held in 078's `ProductionState` (they share a file and a tick, and
splitting them would mean two resources mutating each other's state in one
frame):

```
struct Shipment {
    from: BuildingId,      // the producer
    to: BuildingId,        // the warehouse
    parcel: Parcel,        // one stack, one item type
    remaining: f32,        // game minutes still to travel
    blocked: bool,         // arrived, but the city has no room (see below)
}
```

## Dispatch

Each tick, per producer with a serving warehouse (079's `Coverage`):

- The warehouse must have a free hauler slot — in-flight shipments `to` it,
  counted against its `concurrent_hauls`. This is the "transport per time"
  throughput knob: one warehouse serving six farms delivers them one stack at
  a time and falls behind, and a tier upgrade is how that gets fixed.
- **What ships is one stack of one item.** Dispatch when any single item in
  the buffer has reached `economy.stack_size`; take exactly that many.
- **A stalled producer ships its largest partial stack** even below
  `stack_size`. Without this a building whose outputs are mixed can fill its
  buffer to the cap without any one item reaching a full stack, and deadlock
  there forever.
- `remaining = travel_minutes + warehouse.handling_minutes`, both from 079.
  One-way: the cart coming back empty is not modelled, and
  `concurrent_hauls` is what stands in for that occupancy.

## Arrival

`remaining` counts down by `clock.delta_minutes()`. At zero the parcel goes
into `Stock` via `add_parcel_capped` (079). If the city is at capacity the
shipment sets `blocked` and **waits at the warehouse holding its goods**
rather than dropping them.

That is what closes the loop the storage cap opens: stock full -> hauls block
-> hauler slots stay occupied -> buffers fill -> producers stall. Every step
is visible in the city panel, and every one of them is fixed by building
another warehouse. A blocked shipment retries every tick and delivers the
instant room appears.

## Persistence

`logistics.ron` -> version 2, adding `shipments`. In-flight hauls survive a
quit with their `remaining` intact — a stack does not teleport home because
the player closed the window, and it does not evaporate either. On load, a
shipment whose `from` or `to` is no longer a placed building is dropped and
its goods with it; the warehouse it was going to no longer exists.

Same "log and start empty" rule 078 set for this file, and for the same
reason.

## `RoadType::capacity` stays inert

Ticket 060's other field is *not* read here. Congestion — several hauls
sharing a road cell and slowing each other — is a real mechanic and a
different ticket; wiring `capacity` in as a per-warehouse limit instead would
put a road's number on a building and make the eventual real thing harder to
add. Noted rather than quietly used.

## UI

The city panel's Production section gains an in-flight line per warehouse:
`2/2 hauls, 1 blocked (storage full)`, and each producer's row shows whether
its stack is waiting on a slot or on the road.
