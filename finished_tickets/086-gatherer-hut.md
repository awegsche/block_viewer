# 086 - Gatherer's Hut: schema and simulation

## Status
Done — `cargo test --lib` green (834 passed). Schema, definition, and the
extraction tick all land in this ticket.

## History

Landed in two passes within the same ticket. The first pass shipped the
schema and the `.ron` definition only, deliberately deferring the tick —
mirroring how `production`/`cost` sat schema-only between tickets 040 and
073/078. Asked directly "why schema-first? remove that rule" — the answer is
that this repo's schema-first split exists for *iteration boundaries a real
ticket sequence already crossed* (040 needed 076's per-instance definition
lookup and 077's game clock before production could tick at all); it isn't a
default to reach for on a single ticket that has everything it needs already
in hand. This one did, so the second pass added the simulation in the same
ticket rather than opening a follow-up.

## Why

Roadmap H2 asks for a low-radius, low-speed building distinct from a
specialised quarry/mine (neither built): something that helps a player level
a build site and get a trickle of early materials, rather than run a real
extraction economy.

## Design

- **It slowly removes blocks and stores the drops in its own inventory** —
  the same shape ticket 078's `Production::buffer`/`buffer_stacks` already
  gives a producer, reused rather than invented fresh: a gatherer's output is
  hauled out exactly like a producer's, through the *same*
  `production::ProductionState` map. That one decision is what makes
  everything downstream free — `warehouse::compute_coverage` and
  `production::dispatch_hauls`/`deliver_arrivals` already iterate every
  entry in that map without caring how it got filled, so a gatherer hub is
  served by a warehouse and shows up in both UI panels' producer lines
  without either learning what a gatherer *is*, beyond the two call sites
  that had to (`warehouse::is_producer`, and the two panels' buffer-capacity
  lookups, which previously assumed every buffer belonged to a `production`
  block).
- **A rather low radius and slow extraction speed, compared to a specialised
  quarry/mine** — `radius_blocks` (straight-line/Chebyshev, `farm::
  rect_distance` — the same measure `Farm::radius_blocks` uses and for the
  same reason: there's no road for a gathering radius to be routed along)
  and `blocks_per_minute`, both deliberately small in the shipped file.
- **Removes every block top to bottom until it reaches its own ground level,
  and no deeper** — this needed no new field at all.
  `Building::ground_level` (ticket 085) already names which layer of the
  blueprint is the building's own foundation surface; `placement::
  resolve_placement`'s own arithmetic (`origin.y = base_y - ground_level +
  y_offset`) is exactly what makes `placed.origin.y + ground_level` the
  world Y that layer landed at, once placed — the same value `commit` and
  the ghost preview already read back. `next_gather_target` treats that as
  the floor: strictly above it is diggable, at or below it is not.
- **No `outputs` list.** Unlike `Production`, what a gatherer produces isn't
  a chosen recipe — it's whatever block was actually there. The tick
  resolves that through `super::drops::DropTable`, the same table
  `city::terraform`'s dig already resolves a drop through.

## Schema

`definition::Building` gains an optional `gatherer` block:

```ron
gatherer: Some((
    radius_blocks: 6,
    blocks_per_minute: 2.0,
    buffer_stacks: 4,   // defaults to 4, same as Production::buffer_stacks
)),
```

Validated at load (`validate`): `radius_blocks > 0`, `blocks_per_minute >
0.0`, `buffer_stacks > 0`. `DefinitionError::InvalidGatherer(&'static str)`
carries which rule broke, same shape `InvalidWarehouse`/`InvalidFarm` use.

## Simulation (`city::gatherer`)

- **No resource of its own for the buffer.** `production::Producer` grows one
  field, `dig_carry: f32` — the same fractional-carry shape `partial`/`owed`
  already use, one direction further over: a `blocks_per_minute` rate is a
  count of blocks, not of one chosen item, so it can't share `partial`'s
  per-item map. `production::ProductionState` grows one method, `entry(id) ->
  &mut Producer` (create-on-first-touch), so `city::gatherer` can share one
  buffer/state per building instance rather than a second per-building map.
- **`next_gather_target`** scans every tile within `radius_blocks`
  (Chebyshev, never the building's own footprint) and picks whichever
  still-diggable one is nearest. A `claimed: HashMap<IVec2, i32>` is a
  per-dispatch scratchpad: once a tile is picked, its entry remembers the
  *next* Y down, so digging several blocks in one tick empties the nearest
  column before moving outward, rather than reading the tile's stale
  pre-edit height a second time.
- **`plan_dig`/`settle_dig`** are the pure halves — factored out of the two
  Bevy systems the same way `production::advance_producer` and
  `terraform::dig_edit`/`level_edit` are, so the decision logic is testable
  without a real `App` or task pool. `plan_dig` accrues carry, checks the
  buffer cap first (mirroring `advance_producer`'s "a full buffer stops
  everything, including the carry"), and returns the `WorldEdit` to dispatch
  or `None`. `settle_dig` credits a successful dig's drops into the buffer,
  or refunds the carry a failed one claimed but never spent — the same
  "unaffordable is refunded, not lost" shape ticket 073 gives a placement's
  cost.
- **One write in flight per building, not one shared slot.** Unlike
  `CommitState`/`TerraformBuildState`'s single pending slot (only one
  placement or terraform drag ever happens at once), several gatherer huts
  can legitimately be digging on the same tick; `GathererDigState::pending`
  is keyed per `BuildingId` so a hub with a write already in flight skips
  dispatch (and doesn't accrue more carry) until its own settles, without
  slowing any other hub down.
- **`ProducerState::Depleted`** ("site levelled") — reached when nothing is
  left within radius above the floor. Distinct from `Starved` (a missing
  input, not a fact about the ground) and from `BufferFull` (clears itself
  the moment a haul empties the buffer). The carry resets to `0.0` rather
  than growing forever with nothing to spend it on; the hub keeps cheaply
  re-checking every tick after, so a neighbouring edit putting material back
  in range is picked up rather than latching permanently.
- **No journal entry, ever** — same reasoning `production::tick` already
  gives: undo undoes *builds*, not the passage of time.
- **No `WriteStatus` line, deliberately, unlike every other write path in
  this crate.** That resource drives the city panel's "Last edit" line, and
  a background dig landing every few seconds per hub would mean it never
  gets a chance to show what the player actually just did. `ChunksEdited`
  still fires every successful dig, so the block disappears from the mesh
  without a restart — the panel's silence, not the world's.
- **`warehouse::is_producer`** now also counts a `gatherer` block, not only
  `production` — without this a gatherer hub would never be served by a
  warehouse (`Coverage::served` would return `None` forever) and its buffer
  would fill and stall permanently with no haul to empty it.
- **Both UI panels' buffer-capacity lookups** (`city_panel::producer_lines`,
  `inspect_panel::producer_capacity`) now check `building.gatherer` when
  `building.production` is absent — otherwise a gatherer-only building's
  capacity read as `0`, showing `x/0` instead of `x/256`. A definition
  declaring both reads as `production`'s own cap; the shipped set never
  declares both on one building.

## Shipped: `gatherer_hut.ron`

`name: "Gatherer's Hut"`, `tier: 1`, `category: Production`,
`ground_level: 2`, `radius_blocks: 6`, `blocks_per_minute: 2.0`,
`buffer_stacks: 4`. Cost kept cheap and early-game: 12 oak planks, 8 dirt.

**Placeholder geometry**: `blueprint: "lumber.nbt"`, borrowed rather than
modelled — same pattern `warehouse01`/`warehouse02` already use against
`house01.nbt`. `ground_level: 2` is copied from `lumber.ron` because it
describes the *borrowed* blueprint's own foundation, not a property of the
gatherer hut's design. A real `gatherer_hut.nbt` is still to be exported —
noted in `todo.md`.

## Not in scope

- **A real model.** `lumber.nbt` is a stand-in; a dedicated model will be
  provided later, per the request that started this ticket.
- **A specialised quarry/mine.** Named in the roadmap and in this ticket
  only as the *comparison* a gatherer hub is deliberately weaker than;
  neither is built.
- **Congestion, obstruction, or anything else roadmap H2's own "still open"
  list already names** — a gatherer's dig is an ordinary write through the
  same `apply_building_edit` path every other write in this crate uses, and
  inherits exactly what that path already does and doesn't model.
