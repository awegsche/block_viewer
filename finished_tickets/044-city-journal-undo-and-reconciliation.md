# 044 - city journal, undo and reconciliation

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 369 tests,
up from 351). See the Resolution.

## Part of
Roadmap D3 (`tickets/CITYBUILDER_ROADMAP.md`) — the third and last ticket in
group D (city state), closing out that group. D1 (042) gave the game a
`City`; D2 (043) made it survive closing the window; this is the record that
makes undo, demolish's terrain restore (E5) and world reconciliation all the
same mechanism instead of three.

## Depends on
- 031's `EditPolicy::capture_replaced` / `EditReport::replaced` — the as-built
  baseline (roadmap I1) was already half-built in ticket 031; this ticket is
  the other half, the record that keeps it.
- 032's `RegionSource`/`apply_routed` — reconciliation reads the world
  through the same trait the write path already uses.
- 042's `City` — `place_building`/`remove_building`/`insert_loaded`, and the
  `PlacedBuilding`/`BuildingId` types a journal entry snapshots.
- 043's persistence shape (`SavedBuilding`-style mirror types, `as_u64`/
  `from_u64`, the `<save>/citybuilder/` directory) — the journal file sits
  next to `city.ron` and follows the same conventions.

## Problem

`City` (042) records *what's placed now*. Nothing records *how it got that
way* — what blocks a placement actually wrote, and what stood there before.
Without that record:

- Undo has nothing to undo to. A demolition or a mistaken placement is
  permanent the moment it's committed.
- Demolish (E5) has no terrain to restore — "put back what was there before"
  requires having kept it.
- A rebuild path ("the world's blocks and the city list disagree, recompute
  and fix it") has no source of truth to recompute *from* other than the
  blueprint, which the roadmap already rejects for I1: the blueprint has air
  in it, and doesn't know what terrain used to be under that air.

The roadmap is explicit that this is one record, not three: "Four features,
one record; see I1 for why they're the same record and not four."

## Goal

A journal: an append-only log of placement and demolition entries, each
carrying an as-built baseline (roadmap I1) — the blocks it wrote, and what
those positions held immediately before. Built on top so that:

- **Undo** pops the most recent entry, reverses its `City`-side effect, and
  hands back the world edit that restores what was there.
- **Reconciliation** recomputes what the world *should* hold from every
  currently-placed building's own placement baseline, diffs it against a
  `RegionSource`, and can produce the edit that repairs the difference.

No caller yet for either — E4 (commit) and E5 (demolish) are what will
eventually call `record_placement`/`record_demolition`, and no UI calls
`undo_last`/`reconcile` yet either. Same "proven, not yet used" state 039-042
landed their own resources in; `city::run()` gives persistence a real caller
regardless, the same way ticket 043 did for `City` itself.

## Scope

- `city::journal::Baseline` — `written: Vec<(IVec3, BlockState)>`,
  `previous: Vec<(IVec3, BlockState)>`, `data_version: Option<i32>`.
  `Baseline::capture(edit: &WorldEdit, report: &EditReport) -> Option<Baseline>`
  reads `report.replaced` as `previous` (031's own baseline capture) and
  `edit.edits()` as `written`, deduped and sorted the same way so the two
  line up position-for-position. `None` when `report.replaced` is `None` —
  the edit was applied with `capture_replaced` off.
- `city::journal::JournalEntry` — `Placed { building, placement, baseline }`
  / `Demolished { building, placement, baseline }`. `placement` is a full
  `PlacedBuilding` snapshot (which gained `Clone` this ticket), not a lookup
  key, because `City::insert_loaded` needs the whole thing to put a
  demolished building back and, by the time undo runs on a *placement*
  entry, `City` may no longer have it.
- `city::journal::Journal` (a `Resource`) — the append-only `Vec<JournalEntry>`,
  plus:
  - `record_placement`/`record_demolition`
  - `entries`/`len`/`is_empty`
  - `placement_baseline(building) -> Option<&Baseline>` — the most recent
    *placement* record for a building, read in reverse so a re-placement
    after a demolition supersedes the earlier one.
  - `undo_last(&mut self, city: &mut City) -> Result<UndoStep, UndoError>` —
    pops the last entry, applies its `City`-side reversal (`remove_building`
    for a placement, `insert_loaded` for a demolition), and returns an
    `UndoStep { building, edit }` where `edit` restores `baseline.previous`.
- `city::journal::reconcile(journal, city, source: &mut impl RegionSource) -> ReconcileReport` —
  recomputes the expected block at every position any currently-placed
  building's placement baseline covers, groups by region (the same shape
  `edit::route` uses for the write side), and diffs against what `source`
  actually holds. `ReconcileReport { mismatched: Vec<Mismatch>, unknown: Vec<(IVec3, String)> }` —
  `unknown` is a distinct bucket, not a mismatch, for a region/chunk that
  can't be read at all (I2's own "destroyed" vs "unknown" line, one
  mechanic early since the read path is the same either way).
- `city::journal::repair_edit(&ReconcileReport) -> Option<WorldEdit>` — the
  write half: an edit that writes every mismatch's expected state back.
  `None` when there's nothing to repair.
- Persistence, same split as 043: `SavedJournal`/`SavedEntry`/`SavedPlacement`/
  `SavedBaseline` mirror the in-memory types rather than deriving serde on
  them directly (consistent with `BuildingId` staying deliberately awkward
  to serialize outside `as_u64`/`from_u64`). `save_journal`/`load_journal`
  read and write `<save_root>/citybuilder/journal.ron`, versioned
  (`CURRENT_VERSION`) the same way `city.ron` is.
- `blueprint::BlockState` gained `Serialize`/`Deserialize` directly (unlike
  `PlacedBuilding`, which kept its mirror type) — both its fields were
  already plain serde-able data, and a baseline can carry thousands of block
  states per building; a mirror type for each would be a lot of copying for
  nothing.
- `city::state::PlacedBuilding` gained `Clone` — what a journal entry needs
  to hold its own snapshot.
- `city::run()` wiring: `Journal::default()` (or `load_journal`'d) inserted
  as a resource alongside `City`, and a `save_journal_on_exit` system added
  to `Last` next to `save_city_on_exit` — same `CitySavePath`-gated,
  `AppExit`-triggered shape.

## Watch out

- **Undo is all-or-nothing against the journal and `City`, not against the
  world.** `undo_last` pops the entry and mutates `City` in memory before
  returning; the returned `edit` is left for the caller to commit through the
  write path, exactly the same "apply doesn't save" contract every other
  edit entry point in this crate keeps. A caller that can't commit afterwards
  (locked world, unreadable region) is left with `City`/the journal ahead of
  the world — the same risk that already exists between any `apply` and its
  `save`, not a new one this ticket introduces.
- **Undoing a demolition can fail; undoing a placement can't.** Putting a
  demolished building back can collide with something placed on its tile
  since (`UndoError::Occupied`) — checked *before* the entry is popped, so a
  failed undo leaves the journal exactly as it was and can be retried once
  the conflict clears. Removing a placed building never fails this way —
  `City::remove_building` treats a missing id as a no-op, not an error, by
  042's own contract.
- **Reconciliation reads the *placement* baseline, not the demolition one.**
  A building still in `City::buildings()` is, by definition, not demolished;
  its most recent `Placed` record is the one as-built baseline that answers
  "what should be here." `Journal::placement_baseline` is written to search
  in reverse for exactly this reason — a building placed, demolished, and
  placed again would otherwise match its *first* placement's baseline
  instead of its current one, except that path can't happen today since a
  demolished building's id never gets a second `Placed` entry (a
  re-placement mints a fresh id). Still worth the reverse search rather than
  the first match, since nothing prevents a future ticket from reusing ids.
- **A building with no placement record at all is skipped, not flagged.** A
  save from before this ticket (or a hand-built one) can have a `City`
  building with nothing in the journal. `reconcile` treats that as "nothing
  to check" rather than "everything about it is unknown" — there's no
  baseline to compare against, so saying anything more specific would be a
  guess.
- **`unknown` is a `String` reason, not a typed `RegionUnavailable`.** A
  whole-region failure and a single unreadable chunk/section inside an
  otherwise-fine region are different failure shapes (`edit::route`'s
  `RegionUnavailable` only models the first); rather than inventing a second
  enum to unify them, `reconcile` renders the region-level case through
  `edit::route::refusal_for`'s existing `Display` and writes the
  position-level case by hand. Same reasoning `EditRefusal::Rejected`
  already uses for a case that doesn't fit its other variants.

## Out of scope

- **Wiring `record_placement`/`record_demolition`/`undo_last`/`reconcile` to
  anything that actually places or demolishes a building.** That's E4/E5 —
  this ticket builds the record and its two operations, proven by its own
  tests, the same "no caller yet" state 039-042 landed in.
- **A UI for undo or a repair button.** G2/I6's job.
- **Redo.** The roadmap says "gives undo for free," not redo; adding a redo
  stack would mean deciding what happens to it when a *new* action is
  recorded after an undo, which nothing here needs an answer to yet.
- **Automatic/scheduled reconciliation.** I5's scan-scheduling problem
  (timestamp gates, `ranvil` 020) is untouched; `reconcile` is a function
  that runs when called, not a system that runs on its own.
- **The damage mechanic itself (I2-I7).** `reconcile`/`repair_edit` are the
  plumbing I2/I3 will eventually classify (structural vs. interaction-state
  vs. volatile) before turning a mismatch into a health number; this ticket
  reports raw mismatches with no classification at all.

## Done when

- `city::journal::{Baseline, JournalEntry, Journal, reconcile, repair_edit}`
  exist with the API scoped above; `BlockState` derives `Serialize`/
  `Deserialize`; `PlacedBuilding` derives `Clone`; `city::run()` loads and
  saves the journal alongside the city save.
- Tests: a baseline captured from an edit+report lines up `written`/
  `previous` by position; capture is `None` without `capture_replaced`;
  recording and undoing a placement removes the building and returns the
  restoring edit; the same for a demolition, reinserting under the original
  id; undoing an empty journal is `UndoError::Empty`; undoing a demolition
  onto an occupied tile is `UndoError::Occupied` and touches neither the
  journal nor `City`; `placement_baseline` returns the most recent record;
  reconciliation reports no mismatch when the world matches, a mismatch when
  a block was changed directly in a region, `unknown` for a region the
  source doesn't have and for an ungenerated chunk inside one it does, and
  skips a building with no placement record; `repair_edit` builds an edit
  from mismatches and is `None` when there are none; the journal file
  round-trips (empty, and with both entry kinds), a missing file loads
  empty, a version mismatch and garbage RON are both refused rather than
  panicking.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as scoped. `Baseline`, `JournalEntry`, `Journal`, `UndoStep`/
`UndoError`, `Mismatch`/`ReconcileReport`, `reconcile`, `repair_edit`, and
the `SavedJournal`-family persistence types all live in one new file,
`city/journal.rs` — unlike `City`, the journal has no derived state (an
occupancy grid) that justified splitting 042/043 into `state.rs` +
`persistence.rs`; a flat append-only log doesn't need that split, the same
call `city::definition` made for bundling its schema and loader together.

`Journal::undo_last` turned out to want the edit built *before* matching on
which city-side reversal to apply, since `entry.baseline().restore_edit()`
needs the entry that `self.entries.last()` is about to be matched on, and
matching first would have meant either cloning the entry or fighting the
borrow checker over `self.entries.pop()` happening inside the match. Building
the edit first, then mutating `city`, then popping only on success is what
makes the failed-undo-leaves-nothing-touched property (the Watch out section
above) fall out of the code's own order rather than needing a separate
rollback step.

`reconcile`'s region grouping reuses `crate::edit::address_of` and, for the
whole-region failure case, `crate::edit::route::refusal_for` — called
directly by its `pub(crate)` path (`route` is a `pub mod`, so `refusal_for`
is reachable from `city::journal` even though it isn't re-exported at
`edit`'s top level) rather than rendering a second "region unavailable"
message. That reuse is also why `ReconcileReport::unknown` ended up as
`Vec<(IVec3, String)>` instead of carrying `RegionUnavailable` directly — the
position-level failure (a chunk that won't read inside an otherwise-present
region) has no `RegionUnavailable` variant of its own, and unifying the two
under one typed enum would have meant inventing a variant for a case
`edit::route` never needed.

Testing reconciliation needed a real `ChunkRegion` to read blocks out of, so
`journal/tests.rs` carries a trimmed copy of `edit::tests`' `RegionFixture`
technique (one region, one finished chunk, one all-stone section) rather than
sharing test code across the two `#[cfg(test)]` modules — consistent with how
`edit::tests` itself already keeps `RegionFixture` and `SaveFixture` as two
separate, slightly-overlapping fixtures in one file rather than unifying
them.

`city::run()` wiring mirrors 043 exactly: `load_journal` (the `city::mod.rs`
wrapper, distinct from `journal::load_journal`) is silent when the loaded
journal is empty and logs an entry count otherwise, and
`save_journal_on_exit` is a second system on `Last`, gated by the same
`CitySavePath` and triggered by the same `AppExit` reader `save_city_on_exit`
uses — two independent `EventReader<AppExit>`s over the one event, which Bevy
supports natively (each system's reader has its own cursor).

18 new tests in `city::journal::tests` (369 total, up from 351): baseline
capture (position line-up, and the `None`-without-capture case); record +
undo for both a placement and a demolition; the empty-journal and
occupied-tile undo failures; `placement_baseline`'s most-recent-wins lookup;
four reconciliation cases (clean, one changed block, an absent region, an
ungenerated chunk inside a present one) plus the no-baseline-skip case;
`repair_edit`'s edit-building and `None`-on-empty; and five persistence
round-trip/failure cases mirroring 043's own.

`todo.md`'s existing 043 entry gets a short addendum rather than a new
checklist item — the journal save/load is the same "does `AppExit` really
fire and land the file where expected" question against the same window
close, so it rides along with that pass instead of asking for the app to be
run a second time.
