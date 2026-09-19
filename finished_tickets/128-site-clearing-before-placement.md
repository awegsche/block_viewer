# 128 - Building on non-air ground takes time: the site is cleared first (buildings and roads)

## The problem

A placement is instant. `city::commit::try_commit_placement` claims the
tiles, spends the cost, and dispatches one `blueprint_edit` write that
overwrites *everything* in the blueprint's volume — a hillside, a forest,
the top of a mountain — in a single apply. Ticket 073 already credits the
drops of whatever that write removed (`poll_commit` reads the baseline's
`previous` through `DropTable` into the `Stock`), so the player is *paid*
for the terrain; they just never *wait* for it. A Gatherer's Hut needs
minutes to level a yard at `blocks_per_minute: 20.0`, while a house
dropped on the same slope levels it for free in one click. Roads are
worse: `road_build::try_commit_drag` writes every piece with
`capture_replaced: false`, so a tunnel bored through a hill is both
instant *and* unpaid — the hillside simply vanishes. Terrain a placement
displaces should cost the same thing the gatherer charges for it: time,
and it should land in the stock as the drops it was.

## Design

### What gets cleared, and at what rate

The **site** of a placement is the rotated blueprint's box at `origin`
(`origin .. origin + rotated size`, the same volume `blueprint_edit`
writes). Every position in it that currently holds a non-air block
(`ChunkColumn::get` on `DecodedWorld`, the read `gatherer::next_gather_target`
and `terraform::topmost_block_y` already make; a column whose chunk isn't
decoded counts as nothing to clear — the blueprint write still overwrites
and credits it, exactly as today) has to be dug to air before the
building's own blocks go in. The foundation layers below
`Building::ground_level` count too: they replace real ground, and the
gatherer would have charged for that ground. So a Gatherer's Hut (9x9,
`ground_level: 1`) on flat grass is 81 blocks — about 40 seconds at the
rate below — before the hut appears; the same hut against a hillside is
that plus the hill. A road cell (6x6, its subgrade layer replacing the
topmost ground block) on flat ground is 36 blocks, 18 seconds; a ten-cell
drag on the flat is three minutes, a tunnel cell several times that. If
any of that is wrong in play, the rate is the knob, not the rule.

The rate is one global number, not per definition — nothing in the world
does the clearing yet (no crew, no builder), so there's nothing to hang a
per-building rate on:

```ron
// assets/city/economy.ron
    // ---- how fast a build site is cleared -----------------------------
    // Blocks per minute of game time dug out of a placement's volume
    // (a building's or a road cell's) before its blueprint is written. Six
    // times the gatherer hut's 20: a build site is worked by a whole crew,
    // not one hut's worth of hands, and a road has to keep up with a drag.
    site_clearing_blocks_per_minute: 120.0,
```

`EconomyConfig::site_clearing_blocks_per_minute: f32`, `#[serde(default)]`
to 120.0 so an `economy.ron` without the line loads; the loader refuses a
value `<= 0.0` the way it refuses the other knobs' nonsense.

### The flow: a placement becomes a site first

`try_commit_placement` is unchanged up to and including the synchronous
claim and payment (`place_building`, the conversion, `Stock::spend`): a
second click still can't afford the last forty planks, and an
unaffordable click still refuses before anything is claimed. What changes
is what happens next:

- **Nothing to clear** (the volume is all air, or all undecoded): exactly
  today's path — the blueprint write is dispatched into
  `CommitState::pending` at once. The keyboard stand-in, the existing
  commit tests, and a placement over a previously demolished site all
  keep behaving as they do now.
- **Something to clear**: the placement is entered as a **site**.
  `PlacedBuilding` gains `under_construction: bool` (on the placement, not
  in a side map, for the same reason `work_area` lives there), the journal
  entry is recorded **now** — `Journal::record_placement` with an *empty*
  baseline (`written`/`previous` both empty, `data_version` from the
  blueprint) and a ledger of `debited: spent`, `credited: gained` — and no
  world write is dispatched. The journal entry existing from the first
  tick is what makes cancel, undo and persistence below all fall out of
  machinery that already exists.

### The clearing tick: `city::construction` (new module)

`gatherer`'s dispatch/poll shape, per site (`ConstructionState {
carry: HashMap<SiteId, f32>, pending: HashMap<SiteId, PendingClear> }`,
`enum SiteId { Building(BuildingId), RoadCell(IVec2) }` — one write in
flight per site, sites don't queue behind each other; cite `gatherer`'s
module docs rather than re-arguing it). The tick below is written for a
building; "Roads" further down says what a road-cell site does
differently, which is only the settle and the completion.

- Each tick with `GameClock::delta_minutes() > 0`, for every site without
  a pending write: `carry += site_clearing_blocks_per_minute * minutes`;
  `floor(carry) >= 1` builds a `WorldEdit` of that many positions set to
  air, **topmost layer first, then column order within the layer** (the
  site is peeled from the top down — a hill shrinks, it doesn't get
  hollowed), from a fresh scan of `DecodedWorld` each tick (the volume is
  a few thousand lookups at most; scanning fresh is what makes a
  neighbouring edit safe, the same argument `plan_dig` makes). The carry
  is charged at dispatch and refunded on a refused write, as `plan_dig`/
  `settle_dig` do.
- Same `EditPolicy` as the commit (`capture_replaced: true`,
  `allow_dirty_regions: true`); `ChunksEdited` fires on success so the
  blocks vanish from the mesh. No `WriteStatus` line for a dig — the
  completed placement records `Placed` as today, and the gatherer's reason
  for keeping background digs off the "Last edit" line applies verbatim.
- **Settling a dig**: `Baseline::capture`, credit the drops through
  `DropTable` into `Stock::add_parcel_capped` (overflow printed, as
  `poll_commit` does), and **extend the site's journal entry** — new
  `Journal::extend_placement_baseline(building, more: Baseline, credited: Parcel)`:
  merges by position, a position already recorded keeps its `previous`
  (what stood there before the *first* write) and takes the new `written`;
  both vectors stay `(y, z, x)`-sorted so they line up index-for-index as
  `Baseline`'s docs require; `ledger.credited.add_all(&credited)`.
- **Completion**: a tick whose scan finds no non-air left dispatches the
  blueprint write — `try_commit_placement`'s tail, extracted into a
  `pub(super) fn dispatch_blueprint_write(..)` both callers use — into
  `CommitState::pending`. The slot is single; a site that finds it busy
  simply tries again next tick. `poll_commit`'s `Ok` arm for a site
  extends the entry (blueprint baseline merged in, `credited` from what
  the write replaced — almost nothing by now) instead of
  `record_placement`, clears `under_construction`, and then does what it
  does today: `WriteStatus` `Placed`, `ChunksEdited`,
  `BuildingFootprintChanged` (a road only wants its connected piece once
  the building is really there). The `Err` arm for a site does **not**
  roll back the way a fresh placement's does — the hole is real and
  journaled — it logs, leaves the site standing, and the next tick retries;
  the player's way out is cancel, below.

### Cancel: `Delete` on a site

`demolish::resolve_demolition_target` finds a site the same way it finds a
building (occupancy + `placement_baseline`, which now exists from the first
tick). For a site, the demolition is the entry's **reversal**, not a
demolition: the restore edit is the partial `previous` (puts the dug
terrain back), and on success `poll_demolish` removes the building,
refunds `ledger.debited` into the stock and takes `ledger.credited` back
out (clamped, like every debit — the drops may already have been spent),
and **removes the site's journal entry** (`Journal::remove_site_entry`)
rather than recording a demolition: a cancelled site never happened, and
there is nothing to undo. Ordering is demolish's, not undo's — the entry
and the `City` row go only after the restoring apply succeeds. A site with
a dig in flight refuses `Delete` with a message, the way an occupied undo
refuses, rather than racing the write. `write_status` records it as
`Demolished` with a "cancelled" message.

`undo::start_undo` gains the same precondition — the last entry's building
has no clearing write in flight. With that, `undo_last` on a site needs no
special case: the partial baseline restores, the ledger reverses, the row
goes.

### Roads: every newly added cell is a site of its own

`try_commit_drag` is unchanged through planning and the synchronous
`add_road_cell` loop — heights, tunnels and connections are all decided
before anything is written, exactly as 065/071 require, and a second drag
still sees the cells as taken at once. What changes is the write:

- **Cells this drag newly added** (`newly_added`, the list the rollback
  already keeps) are entered as sites: `RoadCell` gains
  `under_construction: bool`, and each cell's volume is the piece's box at
  `cell_write_origin(cell, base_y)` — the same volume `road_write_edit`
  would write, tunnel piece included. No world write is dispatched for
  them; a drag that only re-crosses existing road (no new cell) writes
  nothing and behaves exactly as today.
- **Already-road neighbours a drag re-tiles** are not sites: their volume
  holds the *old piece*, not terrain, and re-tiling is a swap, not a
  clearing. Likewise the reactive re-tile (110) stays instant — but it
  must **skip a cell under construction** (`wanted_variant` returns `None`
  for one), because that cell's write is the completion below, not a
  re-tile.
- **The tick** peels each cell's volume top-down at the same rate, one
  write in flight per cell, cells cleared in parallel like buildings. The
  drops go straight into the stock (`add_parcel_capped`, overflow printed)
  and **nowhere else**: roads aren't journaled — "there is no undo or
  demolish for a road cell yet" — and this ticket doesn't change that.
  A road's clearing is the gatherer's kind of write, not a placement's:
  credited, unrecorded, not undoable. The old `capture_replaced: false`
  becomes `true` for these digs so the drops can be read off the report.
- **Completion**, per cell: when a cell's scan finds nothing left, its
  variant is refreshed through `wanted_variant` (a tunnel stays a tunnel;
  a building that appeared beside it *while it was clearing* was skipped
  by the re-tile, so this is where it catches up), `under_construction`
  is cleared, and `road_write_edit(&affected_cells(&[cell]), ..)` — this
  cell's piece plus whatever already-built neighbours' pieces change
  because of it — is dispatched through `RoadBuildState::pending`, one at
  a time; a cell that finds the slot busy tries again next tick. The road
  therefore *advances* along the drag, cell by cell from the start, rather
  than appearing all at once when the last cell clears. A neighbour still
  under construction already counts as road for `connections_at` (it's in
  `City`), so the piece written now already points at it. A refused
  completion write logs and retries next tick; the cells stay recorded,
  as a no-catalogue drag's cells do today.
- **Persistence**: `SavedRoadCell` gains `#[serde(default)]
  under_construction: bool`, same no-bump argument as the building field.
  On load every such cell is re-entered as a site with carry 0.
- **No cancel** for a road site — nothing can remove a road cell yet, and
  a cancel mechanic for roads belongs to the ticket that adds road
  removal.
- **Marker**: `preview_mesh`'s piece mesh (made `pub(super)`, with
  `cell_transform`) at the cell in the drag preview's valid tint, spawned
  and despawned like a building's marker.

### What a site is not, yet

A building site is claimed ground with a journal entry, not a building;
a road site is a claimed cell, not a road. `City::buildings()` yields
**only completed placements**; a new
`City::placements()` yields everything. Audit every `buildings()` caller
and move the ones that must see sites: `persistence::save_city`, the
construction tick, the site marker below, `journal::reconcile`'s row
lookup. Everything else keeps `buildings()` and so ignores a site for
free: production/gatherer/mine ticks, `warehouse::compute_coverage` and
`StorageCapacity` (a warehouse site adds no storage), the build menu's
`requires` unlocking (a site unlocks nothing), integrity, the city panel's
counts. Road cells have fewer readers: `warehouse` coverage's
`touching_road_cells`/road-network walk must treat a cell under
construction as **not road** (a producer isn't connected by a road that
doesn't exist yet), while occupancy, `connections_at`/`select_piece` and
the drag's own re-crossing keep seeing it as road (it's claimed, and the
next cell's piece must point at it). `City::road_cells()` keeps yielding
everything; the coverage walk filters on the flag.

### Seeing it

- **Site marker**: one translucent entity per site, `placement::ghost_mesh`'s
  cached mesh for `(catalogue_id, rotation)` at the site's origin with the
  ghost's *valid* material, spawned when a site is entered/loaded and
  despawned on completion or cancel. Without it a site is a hole with
  nothing to say a building is coming. Road cells get the same treatment
  with the piece mesh (see "Roads").
- **Inspect panel**: for a site, above the producer lines, `Clearing site:
  N of M blocks (~x min left)` — `M` counted once when the site is entered
  and kept on `ConstructionState`, `N` = `M` minus the entry's `written`
  count, minutes from the rate. Producer lines are not shown for a site
  (it isn't producing).

### Persistence

`SavedBuilding` gains `#[serde(default)] under_construction: bool`; no
`CURRENT_VERSION` bump, on 111's argument — every existing file's
buildings are complete, so `false` is the truth about that file. The
journal already persists the entry and its partial baseline. On load a
site's carry starts at 0 (at most one block lost) and `M` is recounted
from the world plus the entry's `written` (both are needed: the cleared
blocks are air now).

## Out of scope

- Re-tiles of existing road pieces (a drag's neighbours, 110's reactive
  re-tile) — a swap of one piece for another, instant as today.
- Cancelling or undoing a road site — no road removal exists to hang it
  on.
- An estimate before the click (blocks/minutes on the ghost preview) —
  no on-screen text near the cursor exists yet.
- Who does the clearing (a crew, builders, a per-building rate).
- Terraform's dig/level tools — the player's own hand, already instant by
  design.

## Files

- `assets/city/economy.ron`, `src/city/economy.rs` — the knob + refusal.
- `src/city/construction.rs` (new) + `src/city/mod.rs` — state, tick,
  poll, site marker; `ConstructionPlugin` after `PickingSet` in
  `GameplaySet` like the others.
- `src/city/commit.rs` — site entry, `dispatch_blueprint_write`, both
  `poll_commit` arms for a site.
- `src/city/journal.rs` — `extend_placement_baseline`,
  `remove_site_entry`; tests in `journal/tests.rs`.
- `src/city/state.rs` — `PlacedBuilding::under_construction`,
  `City::placements()`, `buildings()` filtering, `complete_building(id)`.
- `src/city/road_build.rs` — `newly_added` cells become sites, per-cell
  completion write, `wanted_variant` skips a site, `preview_mesh`/
  `cell_transform` visibility; `RoadCell::under_construction` in
  `state.rs` beside the building flag.
- `src/city/warehouse.rs` — coverage ignores a cell under construction.
- `src/city/demolish.rs` — site reversal path, in-flight refusal.
- `src/city/undo.rs` — in-flight precondition.
- `src/city/persistence.rs` — both fields + round-trips + "no field loads
  as `false`" tests.
- `src/city/ui/inspect_panel.rs` — the clearing line.
- `tickets/CITYBUILDER_ROADMAP.md` — E4's and F2's "instant" wording.

## Tests

- Site scan: counts every non-air block in the rotated volume, foundation
  layers included; an undecoded column counts zero; the dig order is
  top layer first.
- Tick: carry accrues at the rate, `floor(carry)` blocks per dispatch,
  refunded on refusal, nothing dispatched while a write is pending; the
  tick that finds nothing left dispatches the blueprint write, and retries
  next tick when the commit slot is busy.
- Placement over pure air is unchanged: journaled once, on the write's
  success, with today's baseline (the existing commit tests keep passing
  untouched).
- Placement over stone: the entry exists from the claim with an empty
  baseline and the cost ledger; after two digs its `previous` holds the
  stone and `credited` the cobblestone; after completion `written` holds
  the blueprint and `under_construction` is `false`.
- `extend_placement_baseline`: a repeated position keeps the first
  `previous`, takes the last `written`; vectors stay sorted and aligned.
- Cancel: restores the partial `previous`, refunds `debited`, removes
  `credited` (clamped), removes the row and the entry, records nothing;
  refused with a dig in flight.
- Undo of a site as the last entry does the same through `undo_last`.
- `City::buildings()` skips a site; `placements()` includes it; coverage,
  storage capacity and the build menu's `requires` ignore a site.
- Roads: a drag over stone enters its new cells as sites and writes
  nothing; a drag that only re-crosses existing road is written at once as
  today; a cell's clearing credits drops to the stock and the journal
  stays empty; the cell whose scan finds nothing left is written with
  `affected_cells(&[cell])` and its flag cleared while its drag-neighbour
  is still clearing; `wanted_variant` is `None` for a site and the
  completion write picks `Connected` for a building that appeared
  meanwhile; coverage doesn't reach a producer whose only road is under
  construction.
- Persistence: both `under_construction` flags round-trip; a file without
  the fields loads as `false`.
- `cargo check`, `cargo test`, `cargo clippy` clean.

## Manual check (todo.md)

Place a house against a hillside: the ghost-coloured marker appears at
once, the hill peels away from the top at about 120 blocks/min (watch the
inspect panel's count fall), the stock gains the dirt/stone, and the house
appears only once the volume is clear. Place one on flat grass: the grass
layer under it goes first. `Delete` mid-clearing puts the hill back and
refunds the planks. Drag a road into a hillside: the piece markers appear
along the whole drag at once, the cells clear in parallel, each piece
lands the moment its own cell is clear (the tunnel cells last), and the
stock gains the stone. Save and reload mid-clearing: both kinds of site
resume.
