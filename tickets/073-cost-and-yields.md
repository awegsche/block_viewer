# 073 - buildings cost materials, cleared blocks yield them

Second half of the iteration-2 economy's foundation, on top of ticket 072's
`Stock`/`DropTable`. Production, warehouses and road haulage are ticket 074+;
this ticket is the ledger every one of those will settle against.

## The problem

072 built a stockpile nothing puts anything into or takes anything out of.
`Building::cost` is still decoration, and the roadmap's H2 ("dug blocks
become resource counts", "charging build cost for blocks a placement
clears") is still open.

## The rule

> **The world and the ledger stay in sync.** A write that *removes* blocks
> credits their drops. A write that *restores* blocks debits them. A
> building's own blocks are neither — they are what `cost` buys.

Applied to the four write paths that exist:

| path | credit | debit |
|---|---|---|
| placement (`commit`) | drops of `baseline.previous` (the terrain it cleared) | the definition's `cost` |
| terraform dig/level | drops of every replaced block | drops of every written block (level's fill; dig writes air, so nothing) |
| demolish | *nothing* — no salvage (see below) | drops of `baseline.written` (the terrain put back) |
| undo | the entry's own `debited` | the entry's own `credited` |

**No salvage on demolition, and it still charges for the backfill.** Without
that second half the loop `place -> demolish -> place` is a material farm:
each cycle clears the same hillside again and hands back its stone. Charging
the restored terrain closes it, and it falls straight out of the rule above
rather than being a special case invented to plug the hole. Salvaging a
demolished building's own blocks is a real mechanic, but it's a mechanic
(a *fraction*, a rubble state), not a rounding decision — later.

**Undo is settled from the journal, not recomputed.** Each entry gains a
`Ledger { credited: Parcel, debited: Parcel }` — what the action actually
moved. Undo adds back what it debited and removes what it credited, exactly,
which stays correct even when the definition's `cost` was edited (hot
reload!) between the placement and the undo. Recomputing the cost at undo
time would refund a number that was never paid.

That makes `journal.ron` version **2**, and — unlike `city.ron`'s equality
check — it's read as a **band** `1..=2`, ticket 069's precedent: refusing a
version-1 journal would throw away every as-built baseline in it (roadmap
I1's whole point, and unrecoverable after the fact) to add a field whose
correct value for those entries is *empty* — a placement made before the
economy existed moved nothing.

**A debit that can't be paid is clamped, never refused.** `Stock::remove`
already clamps at zero. Demolishing or undoing is a world-state operation;
blocking one because the player is short of dirt would mean the city state
and the world can't be brought back into agreement until they go mine some.
Costs are the opposite — `Stock::spend` is all-or-nothing, and a placement
that can't be paid for never happens.

## The definition-id problem this exposes

`PlacementSelection` carries a **catalogue** id (`house01`, the `.nbt` stem),
and `PlacedBuilding::definition` stores that same catalogue id — but `cost`
lives on the **definition** (`assets/city/buildings/<id>.ron`), and a `.ron`
is free to name a blueprint with a different stem. So there is currently no
way to get from a placement to its cost.

Narrow fix, in this ticket: `PlacementSelection::definition_id: Option<String>`,
set by the build menu (the only real selector). A selection made with the
keyboard stand-in (`1`-`9`, catalogue entries directly) has no definition and
is therefore free — the same "no definition, no game data" hole that already
makes such a placement have no requirements and no production.

The deeper fix — `PlacedBuilding` storing the definition id, so a *placed*
building can find its own data — is a `city.ron` version bump and belongs to
the ticket that first needs it per-instance, which is production (074+). Noted
in `todo.md` and the roadmap rather than smuggled in here.

## Work

1. `journal`: `Ledger`, `JournalEntry::{Placed, Demolished}::ledger`,
   `record_placement`/`record_demolition` take one, `CURRENT_VERSION` 1 -> 2
   with `MIN_READABLE_VERSION` 1 and `#[serde(default)]` on the saved field,
   `UndoStep` carries the ledger to settle.
2. `commit`: `definitions` + `stock` in `try_commit_placement`; refuse (with
   a `WriteStatus` failure line, so the city panel says so) when the cost
   can't be paid; `spend` at the same instant `place_building` claims the
   tiles; keep the spent `Parcel` on `PendingCommit`; on write failure refund
   it beside the existing `remove_building` rollback; on success credit the
   drops of `baseline.previous` and journal the ledger.
3. `demolish`: debit the drops of the restoring edit's `written`, journal it.
4. `undo`: settle the popped entry's ledger in reverse.
5. `terraform`: `EditPolicy::capture_replaced` on (it isn't today), then
   credit replaced / debit written off the report. Not journaled, so no
   ledger entry — the stock is the only record, same as the terrain itself.
6. `build_menu`: mark rows the player can't afford, and say what's short.
7. Roadmap H2 rewritten as done, with the salvage/definition-id follow-ups
   named; `todo.md` gets the visual checks.

## Done when

- `cargo check` and `cargo test` pass.
- Placing a house costs its planks and stocks the dirt it displaced; undoing
  it returns the ledger to exactly where it stood before, both directions.
- `place -> demolish -> place` on the same hillside is not a way to make
  stone.
- A version-1 `journal.ron` still loads, with empty ledgers.
