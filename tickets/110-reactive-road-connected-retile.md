# 110 - Reactive re-tile: a building placed/removed next to a road repaints it

## The problem

`state::RoadCell::variant` is decided once, when a cell is written or
re-tiled by a road drag (`road_build::plan_connections`), and then just
remembered — `RoadCell::variant`'s own doc comment already names the gap:

> nothing yet re-tiles a road cell when a building is placed or removed next
> to it after the fact. A building that shows up beside an already-built road
> doesn't retroactively repaint it; only a cell this drag actually writes or
> re-tiles picks the variant up.

Same gap in reverse: demolish the building next to a `Connected` cell and the
cell stays `Connected` forever, pointing at a building that's gone.

`road::touches_building` is explicitly safe to call again at any time (unlike
`RoadPieceVariant::Tunnel`, which destroys the evidence it was computed
from — see the same doc comment) — this ticket is the thing that actually
calls it again.

## Design

Event-driven, not a full re-derive. `warehouse::recompute` already rebuilds
`Coverage` from scratch on `city.is_changed()` (cheap: an in-memory graph
walk, no I/O), but this ticket's rewrite is a real world write — re-scanning
every road cell on every unrelated `City` mutation would mean a write task
per production tick's neighbor or per road-drag cell for no reason. Scope the
trigger to exactly "a building's footprint appeared or disappeared":

1. **New event**, e.g. `city::road_build::BuildingFootprintChanged(PlacedBuilding)`
   (one event for both directions — placed or removed — since the recompute
   below is the same either way: re-read `touches_building` fresh, it isn't
   told which direction the change went).

2. **Fired at every point that already mutates `City` for a building and
   fires `ChunksEdited` on success** — same success branch, same data already
   in scope:
   - `city::commit::poll_commit`, `Ok(report)` arm, after
     `journal.record_placement(...)` — send with `placement.clone()`.
   - `city::demolish::poll_demolish`, `Ok(report)` arm, after
     `city.remove_building(building)` — send with `placement.clone()`
     (`placement` is still the pre-removal snapshot; `touching_road_cells`
     only needs the geometry, not a live `City` entry).
   - `city::undo::poll_undo`, `Ok(report)` arm — undoing a placement removes
     a building, undoing a demolition re-adds one; either way the affected
     cells need the same recompute. **Requires plumbing**:
     `journal::UndoStep` doesn't currently carry the `PlacedBuilding` (only
     `building: BuildingId`) — add a `placement: PlacedBuilding` field,
     filled from `JournalEntry::Placed.placement` /
     `JournalEntry::Demolished.placement` in `Journal::undo_last`, threaded
     through `undo::PendingUndo` into `poll_undo`.
   - Do **not** fire it from `commit::poll_commit`'s `Err` arm (rollback of a
     placement that never actually happened) or from
     `journal.rs`'s replay paths — only a change that also produced a real
     `ChunksEdited` write.

3. **New system, consuming the event** (`city::road_build` is the natural
   home — it already owns `RoadCatalogue`, `road_write_edit`, and the
   async-write/`ChunksEdited` plumbing every other write path here uses):
   for each `BuildingFootprintChanged(placement)`, `road::touching_road_cells(&city, &placement)`
   gives the candidate cells. For each:
   - Skip if `city.road_cell_at(cell)` is `None` (not a road cell — shouldn't
     happen since `touching_road_cells` only returns road cells, but a
     defensive read beats a panic) or its `variant` is already `Tunnel`
     (tunnel always wins — same rule `plan_connections` already follows, see
     `road_build`'s "Connected" module docs).
   - Recompute the wanted variant: `Connected` if `road::touches_building(&city, cell)`
     is now true *and* the catalogue actually has `(style, kind, Connected)`
     for that cell's kind (same catalogue-gating `plan_connections` does —
     reuse `piece_for` to get `kind` from the cell's `ascent`/connections),
     else `Surface`.
   - Skip if that equals the cell's current recorded variant — nothing to
     rewrite (this is what keeps two buildings sharing one cell, or an
     already-correct cell, a no-op).
   - Otherwise: record the change.

4. **New `state::City` mutator** — `add_road_cell` no-ops on a cell that
   already exists (that's what protects `base_y`/`ascent`/`variant` from a
   drag that merely crosses an existing cell), so it can't be reused here.
   Add something like `City::set_road_cell_variant(&mut self, cell: IVec2, variant: RoadPieceVariant) -> Option<RoadPieceVariant>`
   (returns the old value, so the caller can roll back on write failure) —
   the affected `RoadCell`s in place.

5. **Write it**, same shape as every other write path in this crate
   (`try_commit_drag`/`poll_road_build` is the closest cousin — reuse
   `road_write_edit(&changed_cells, &catalogue, &city)` directly, since it
   already reads each cell's *current* recorded style/kind/variant off
   `City`, which step 4 just updated): spawn the edit on
   `AsyncComputeTaskPool` via `commit::apply_building_edit`, poll it next
   frame the same `block_on(poll_once(..))` way, and on success fire
   `ChunksEdited` so the mesh updates live. On failure, roll every changed
   cell's variant back to what step 4's return value recorded — the same
   "don't leave city state ahead of the world" shape `poll_road_build`'s own
   failure arm already follows for `newly_added` cells.

## Out of scope

- Any change to how a cell is *initially* decided (`plan_connections` is
  untouched) — this only adds the retroactive path.
- Moving/rotating an already-placed building (doesn't exist yet).
- Road cell removal reacting to anything (`remove_road_cell` has no caller
  yet, still `#[allow(dead_code)]`) — a building never invalidates a road
  cell's existence, only its variant.
- A style with no `-connected.nbt` for some kind: stays `Surface`, exactly
  like `plan_connections` already tolerates (see `assets/city/roads/dirt/README.md`,
  which currently ships `-connected.nbt` for every kind except `cross` and
  `stairs` — see that file's "What's shipped").

## Done when

- `cargo build`/`cargo test --lib` clean.
- A building placed next to an already-built `Surface` cell rewrites it to
  `Connected` in both `City` and the world (assert the recorded `variant`
  *and* the written blocks, the way `road_write_edit_writes_the_connected_piece_for_a_connected_cell`
  already does for the write-time path).
- Demolishing that building reverts the cell back to `Surface`, rewritten in
  the world to match.
- A cell already `Tunnel` never flips to `Connected` (or back), whatever
  buildings appear or disappear next to it.
- A style/kind with no `-connected.nbt` shipped stays `Surface` when a
  building shows up next to it — no panic, no phantom variant recorded
  pointing at a piece the catalogue doesn't have.
- Undoing a placement that had triggered a neighbour's `Surface -> Connected`
  flip reverts that neighbour to `Surface` too; undoing a demolition that had
  triggered `Connected -> Surface` re-flips it to `Connected`.
- Two buildings bordering the same cell: removing one leaves it `Connected`
  (the other still touches it) — not a spurious rewrite down to `Surface`
  and back.
- A simulated write failure leaves every candidate cell's recorded `variant`
  at what it was before this event, not stuck at the new value with the old
  blocks still on disk/in the region cache.
