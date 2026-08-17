# 051 - Defer world writes to a manual Save

## Status
Done — implemented and tested. See the Resolution.

## Problem

Every placement (E4, ticket 048), demolition (E5, ticket 049) and undo
(ticket 050) opens its own [`edit::session::WriteSession`] and calls
`.commit(...)`, which backs up and saves the touched region file(s) to disk
**immediately** — one region-file rewrite per building, even though 009's
writer rewrites the whole `.mca` (roadmap's own words: "the cost is that a
single-block edit rewrites 512×512 blocks' worth of file"). Placing a street
of houses means a street's worth of full-region rewrites, one click apart,
which is both slow and pointless: nothing reads the disk copy again until
the player re-opens the world in Minecraft.

Reported directly after manually testing placement (ticket 048/049/050).

## Goal

Placing, demolishing and undoing still mutate the shared [`RegionCache`]
synchronously (so the ghost/live re-mesh and the occupancy grid stay
correct), but **stop opening a `WriteSession` and stop touching disk** on
every one of them. Edited regions stay dirty — and, per `RegionCache::evict_lru`'s
existing rule (ticket 032), a dirty region is never evicted by the LRU no
matter how far the camera roams, so every block the citybuilder has changed
stays resident in memory regardless of load distance. A new **Save** button
in the city panel (G2) is what actually opens a `WriteSession` and writes
every dirty region to disk (`WriteSession::flush`, already built by ticket
033 and unused since); once flushed, a region is clean again and the
existing LRU is free to purge it like any other.

Because closing the app already saves `city.ron`/`journal.ron` unconditionally
on `AppExit` (ticket 043/044), leaving world edits unflushed at that same
moment would leave the city state and the journal describing buildings that
were never actually written to the save — so `AppExit` also flushes the
region cache, synchronously, before those two saves. This isn't "another
autosave on every placement"; it's the one safety net that keeps the three
on-disk records (world, `city.ron`, `journal.ron`) from silently diverging
at the one moment there's no more "later" to click Save.

## Scope

- `city::commit`'s `commit_building` stops opening a `WriteSession`; it
  becomes `apply_building_edit`, calling `edit::apply_routed` directly against
  the cache with `EditPolicy::allow_dirty_regions = true` (a placement in a
  region another placement already dirtied must not be refused — that's the
  whole point of batching). Returns `Result<EditReport, EditRefusal>`, not
  `Result<WriteSummary, WriteError>` — there is no disk write here to report.
  `city::demolish` and `city::undo` call the same function; both already
  shared `commit_building`, so this is one change with three callers.
- `city::write_gate::WriteGate` is removed. It existed only to stop two
  independent `WriteSession::open` calls from racing over the same
  `session.lock` (`city::commit` vs `city::demolish`); with neither opening a
  session per edit any more, the race it guarded against no longer exists.
  The shared `Arc<Mutex<RegionCache>>` still serializes concurrent edits
  correctly on its own.
- A new `city::save` module: `SaveCommand` (the same `request`/`busy`/`state`
  shape `city::undo::UndoCommand` already uses), `start_save` opens a
  `WriteSession` and calls `.flush(&cache)` on `AsyncComputeTaskPool`,
  `poll_save` records the result into `WriteStatus`.
- `city::write_status::WriteStatus` splits into what it already was (the
  last placement/demolition/undo *applied to memory*, not written to disk —
  reworded, and its `WriteRecord` drops the `backups` field, which no longer
  applies to an in-memory apply) and a new `last_save` slot for the Save
  button's own outcome (regions written, backups taken).
- `city::ui::city_panel` gains a "World save" section: how many regions are
  currently dirty (read straight off `RegionCache::dirty_regions`), a "Save
  world" button disabled while a save is already in flight, and the last
  save's own result.
- `city::run` flushes the region cache on `AppExit`, synchronously, before
  `save_city_on_exit`/`save_journal_on_exit` — see the Goal section for why.

## Out of scope

- Any change to the write path itself (`edit`, `edit::route`,
  `edit::session`) — ticket 033 already built exactly the batching primitives
  this ticket needed (`EditPolicy::allow_dirty_regions`,
  `RegionCache::dirty_regions`, `WriteSession::flush`); this ticket is wiring
  the citybuilder up to call them instead of `commit`.
- Autosave on a timer, or warning the player about unsaved changes beyond the
  city panel's own dirty-region count — a later G2 refinement if wanted.
- Roadmap group I (damage/reconciliation) — unaffected; `reconcile` already
  diffs the world against the journal's baselines regardless of when the
  world was last flushed.

## Done when

- `cargo build`/`cargo test` clean.
- Placing, demolishing and undoing no longer touch disk, no longer take the
  save's `session.lock`, and still update the live mesh (`ChunksEdited` fires
  off the in-memory apply, same as before).
- A region a second placement lands in, that an earlier unsaved placement
  already dirtied, is accepted rather than refused.
- Clicking "Save world" writes every dirty region to disk, backs each one up
  once per session, and clears the dirty set; the city panel's dirty-region
  count drops to zero.
- Quitting with unsaved placements still on the region cache flushes them
  before `city.ron`/`journal.ron` are written.
- Manual verification (place several buildings without saving, confirm the
  save's `.mca` files are untouched on disk; click Save world and confirm
  they update; quit with something unsaved and confirm it landed anyway) goes
  in `../todo.md`.

## Resolution

Landed as scoped. `city::commit::apply_building_edit` replaces
`commit_building`, calling `crate::edit::apply_routed` directly with
`allow_dirty_regions: true` — no `WriteSession`, no `SaveMeta` parameter, no
disk I/O. `city::demolish::try_demolish`/`poll_demolish` and
`city::undo::start_undo`/`poll_undo` were updated to the same
`Result<EditReport, EditRefusal>` shape; all three still dispatch onto
`AsyncComputeTaskPool` since `RegionCache::get_or_load_mut` can still do a
first-touch disk read for a region not yet resident.

`city::write_gate` is deleted outright — grep confirmed its only callers were
the three modules above, all of which stopped needing session-lock
serialization the moment they stopped opening sessions.

`city::save::SavePlugin` follows `city::undo::UndoPlugin`'s shape exactly:
`SaveCommand::request`/`busy`/`state`, `start_save` (no save loaded → a
`Failed` state, same guard the other three commands use), `poll_save`. The
one thing worth noting: `flush` is called with the whole `RegionCache` behind
the same `Arc<Mutex<_>>` commit/demolish/undo already share, so a save that
lands mid-placement simply blocks on the mutex like any other contended
access — no new coordination needed.

`WriteStatus` gained `last_save`/`record_save_success`/`record_save_failure`,
and `record_success`/`record_failure` (the placement/demolition/undo side)
now take `&EditReport` instead of `&WriteSummary` — `WriteRecord` dropped its
`backups` field, since an in-memory apply has none. The city panel's "Write
status" heading is now "Last edit" (queued-to-memory language, explicitly
says "not yet saved to disk"), with a new "World save" heading below it
showing the live dirty-region count via a `try_lock` on the shared cache
(skipped gracefully if the lock is contended that frame — a UI count one
frame stale is harmless).

`city::run` adds `flush_world_on_exit` to the same `Last`-schedule `AppExit`
handler as `save_city_on_exit`/`save_journal_on_exit`, ordered before both —
a synchronous `WriteSession::open` + `flush` (blocking is fine; the app is
already exiting). A failure is logged, not fatal, the same tone every other
exit-time save uses.

Testing: `city::save` has its own test module (request/busy state machine,
`poll_save` success/failure through a fixed task, and one real-fixture
end-to-end test: two edits applied in memory, `flush` writes them, a fresh
`RegionCache` over the same directory reads them back). `commit`/`demolish`
tests updated for the new return type; the `WriteGate`-specific tests were
deleted with the module. `write_status` tests split across the two record
kinds.

No manual verification recorded as done — `../todo.md` carries the
checklist from "Done when" above.
