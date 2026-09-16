# 119 - Mine (stretch): timber and torches out of the city's stock

Design: `MINES_DESIGN.md` ("The tick" — "blocks the mine *places* … are
free in this iteration"). Depends on 116. Optional; write it only once
mines have been played and the free supports feel like a hole.

## The problem

116 conjures every block a mine *places* — cobblestone floors and seals,
oak stairs, oak-log pillars, torches — from nothing. Cobblestone is a
non-issue (a mine makes far more than it lays), but a level's secondary
carries a pillar every 4 blocks and a torch every 8, and a torch is coal
plus a stick: the one building that produces coal never spends any. The
economy already has the shape for "a producer that needs inputs" —
`Production::inputs`, `Producer::owed`, `ProducerState::Starved` with
`short_of` — and it has never been exercised against a real recipe graph
(H2's "still open" list).

## Design

- `Mine` gains `consumes: Vec<ProductionItem>`-shaped **per-slice** costs,
  keyed by what the slice places — not a rate per minute:
  ```ron
  consumes: (
      torch:  [(item: "coal", count: 1), (item: "stick", count: 1)],   // per torch placed
      pillar: [(item: "oak_log", count: 3)],                          // per pillar
      stairs: [(item: "oak_planks", count: 1)],                       // per stair block (6 planks → 4 stairs, rounded against the player)
  ),
  ```
  Cobblestone is deliberately **not** listed: it comes out of the mine's
  own buffer first (`Parcel::take`), and only what the buffer lacks is
  free — the miners lay the rubble they just dug.
- The plan (115) already knows what a slice places; `SlicePlan` grows
  `places: Parcel`. The job sums it; settling **charges the city's stock**
  (`Stock::take`) through the same conversions/interchangeables a
  placement's payment uses (`economy.ron`: a birch forest's logs are a
  pillar too). Nothing is charged for a refused job.
- **Starvation.** If the stock can't cover a job's `places`, the job is
  *still applied* (the geometry is in the world; refusing it would leave a
  half-lit corridor) and the shortfall goes into `Producer::owed`; while
  `owed` is non-empty the mine's `blocks_per_minute` is halved and its
  state reads `Starved` with `short_of` naming the item — the same shape
  and label a starved production building shows. Each tick first tries to
  pay `owed` down out of the stock. Never a hard stop: a mine without
  torches digs in the dark, slowly, rather than not at all.
- `haul` direction is unchanged: inputs are taken from the city's stock
  directly (no delivery mechanic exists — 080 is one-way), which is
  consistent with how a placement is paid.

## Not in scope

- A second haulage direction (warehouse → producer).
- Charging the Gatherer's Hut for anything (it places nothing).

## Tests

- `places` for a gallery slice with a torch is one torch; a secondary
  slice with a pillar and torch is 3 logs + 1 torch; cobblestone never
  appears in `places` when the buffer covers it, and appears only as the
  shortfall when it doesn't.
- Settling charges stock through a conversion (planks from logs) and
  through an interchangeable (birch for oak).
- A shortfall lands in `owed`, halves the rate, labels `Starved` / `short_of`;
  restocking clears it on the next tick.
