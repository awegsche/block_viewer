# 078 - Production: outputs accumulate into a buffer, and stall when it fills

## Status
Done — `cargo test` green (722 passed).

Landed as described, with two additions the work turned up:

- **`Producer::short_of`**, not persisted: which input a starved producer is
  waiting for, so the panel reads "starved (needs coal)" rather than sending
  the player looking. Re-derived on the first tick after a load.
- **Two new definition/economy validations**, both cases that would otherwise
  read in-game as a building that silently doesn't work: `stack_size: 0`
  (`EconomyError::ZeroStackSize`) and `production.buffer_stacks: 0`
  (`DefinitionError::ZeroBufferStacks`).

`tick` returns early on a zero delta rather than running with one, which is
what makes "paused produces nothing" true by construction rather than by
arithmetic that happens to come out at zero.

## Why

Roadmap H2's "still open" list opens with production. `definition::Production`
(`outputs`/`inputs`, both `per_minute`) has been loaded and inert since ticket
040; this is what finally reads it. Needs 076 (a placed building can find its
own `.ron`) and 077 (a game clock to integrate a rate against).

## The model

One `city::production` module, one resource: `ProductionState`, a
`HashMap<BuildingId, Producer>` alongside `City` rather than inside it —
`City` is the *placement* record, and a buffer is not a placement. Same
relationship `journal::Journal` already has to it.

```
struct Producer {
    /// Fractional carry per output item, always in [0, 1). The whole units
    /// it rolls over into land in `buffer`.
    partial: BTreeMap<String, f32>,
    /// Whole items produced and waiting for a haul (ticket 080).
    buffer: Parcel,
    /// Fractional input debt, same shape as `partial` — an input is taken
    /// from the global `Stock` a whole unit at a time, not 0.017 at a time.
    owed: BTreeMap<String, f32>,
    state: ProducerState,
}

enum ProducerState { Running, Starved, BufferFull }
```

Per tick, for every placed building whose definition has a `production`:

1. **Inputs first.** Accrue `rate * clock.delta_minutes()` into `owed`. Any
   input owing a whole unit or more is taken from the global `Stock`,
   **all-or-nothing across every input at once** — a building that can pay
   for its planks but not its cobblestone must not eat the planks. If it
   can't be paid, the producer is `Starved` and produces nothing this tick;
   the debt stays owed, so it resumes the moment materials arrive rather
   than losing the partial run.
2. **Then outputs.** Accrue `rate * delta_minutes` into `partial`, roll whole
   units into `buffer`.
3. **The cap.** `buffer.total() >= buffer_capacity` -> `BufferFull`, and
   nothing accrues at all: not `partial`, not `owed`, not the inputs. This is
   the answer to "what happens when hauls can't keep up" — the building
   *stops*, visibly, the way Anno's do, rather than quietly evaporating its
   output. Capacity is `production.buffer_stacks * economy.stack_size`,
   `buffer_stacks` defaulting to 4.

A building with no `production`, or with no `definition_id` (076 — the
keyboard stand-in's placements), simply never gets a `Producer`.

## Schema additions

- `definition::Production.buffer_stacks: u32` (default 4).
- `economy.ron`: `stack_size: u64` (default 64) — the unit both this buffer
  cap and 080's hauls are measured in. It belongs with the economy's other
  tunable numbers rather than in each building's file, since a "stack" is a
  property of the world's items, not of the farm holding them.

## Persistence

`<save>/citybuilder/logistics.ron`, its own file for the same reason ticket
072 gave `stock.ron` one: folding it into `city.ron` would put a buffer's
fractional carry behind that file's strict version check and discard a whole
city to add it. Version 1 holds producers; ticket 080 bumps it to 2 for
in-flight shipments.

Unlike `city.ron`, a version this build doesn't know is **logged and started
empty**, not refused — `run()` already does exactly this for a `city.ron`
that fails to load. The justification is specific to this file and worth
stating so it isn't cited as precedent: a producer's buffer regenerates
within minutes of play, where a placement or an as-built baseline (roadmap
I1) is gone for good.

Producer entries for ids `City` no longer holds are dropped on load — a
building demolished in a previous session leaves no buffer behind.

## No journal entry, ever

Production changes `Stock` without writing a single block, so it takes no
`journal::Ledger` and no `JournalEntry`. Undo undoes *builds*, not the passage
of time; an "Undo" that clawed back a farm's output would be a different
mechanic wearing the same button.

## UI

The city panel grows a Production section: one row per producer — name, state
(`Running`/`Starved`/`Buffer full`), buffer fill as `n/cap`. `Starved` names
the input it is short of.

## Assets

Nothing to place yet: the only shipped blueprint is `house01.nbt` and the only
definition is a `House` with no `production`. Ticket 079 ships the placeholder
farm/warehouse definitions (both reusing `house01.nbt`'s geometry) that make
this and 080 playable, and files the real `.nbt` authoring in `todo.md`.
