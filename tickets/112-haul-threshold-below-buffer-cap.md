# 112 - Haul threshold below the buffer cap; bigger Gatherer's Hut buffer

## The problem

Seen in play: a Gatherer's Hut fills its buffer, goes `buffer full`, and
*only then* does hauling start — so the hut spends most of its life stalled.

The cause is in `production::ready_stack`: a haul is dispatched when one
item has reached a full `stack_size`, or — as ticket 080's deadlock guard —
when the producer is `BufferFull`, in which case its largest partial stack
goes. A hut's drops are mixed (dirt, grass, stone, gravel, sand…), so with
`buffer_stacks: 4` at `stack_size: 8` (32 items) no single item reliably
reaches 8 before the combined total reaches 32. The "stalled" guard is
therefore the *normal* haul trigger for a gatherer, not the fallback.

Two things wanted, and one hard rule:

1. **A bigger hut buffer**, so a haul round-trip doesn't stall it.
2. **A hauling threshold separate from the buffer cap**: hauling must be
   triggered while the building is still able to produce.
3. **The two must never be the same number.** A threshold equal to the cap
   is exactly today's behaviour, so the definition loader refuses it.

## Design

### `haul_at_stacks` on `Production` and `Gatherer`

One new optional field on both structs (the same shape `buffer_stacks`
already shares between them): the fill level, in stacks, at which the
producer ships its **largest partial stack** even though no item has
reached a full one. Full stacks still dispatch the moment they exist,
below or above the threshold — the threshold only adds a second trigger.

- Left out: defaults to `buffer_stacks / 2`. A `buffer_stacks: 1`
  definition gets a threshold of 0, i.e. "ship whatever you have" — fine.
- Given: `validate` refuses `haul_at_stacks >= buffer_stacks`
  (`InvalidProduction`-style error, `"haul_at_stacks must be below
  buffer_stacks"`), for both blocks.

### `ready_stack` reads the threshold, not the state

`dispatch_hauls` takes `&EconomyConfig` (it needs `stack_size` *and* the
threshold in items) and resolves each producer's threshold off its
definition — `production::haul_threshold` / `gatherer::gatherer_haul_threshold`,
mirrors of the two `*_buffer_capacity` helpers. `ready_stack` ships the
largest partial when `buffer.total() >= threshold`. The `BufferFull` check
goes: a cap strictly above the threshold means "full" always implies "past
the threshold", so it's subsumed.

Single-output producers (farm, lumber) behave exactly as before: a full
stack always exists before the combined total can pass any threshold.

### The hut

`gatherer_hut.ron`: `buffer_stacks: 12`, `haul_at_stacks: 4`. At
`stack_size: 8` that is a haul from 32 items with 64 of headroom above it.

## Out of scope

- Showing the threshold in the inspect panel (buffer "x/y" is unchanged).
- Dispatching more than one partial stack per producer at a time.
