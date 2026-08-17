# 049 - Demolish

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 418 tests,
up from 408). See the Resolution.

## Part of

Roadmap E5 (`tickets/CITYBUILDER_ROADMAP.md`), the fifth and last ticket in
group E (placement). Depends on E4's commit (ticket 048, `city::commit`,
`state::City::place_building`/`remove_building`), D1's city state (ticket
042), D3's journal (ticket 044, `journal::Baseline`, `Journal::record_demolition`,
already built but uncalled), and the whole write path (W1-W7).

## Problem

A building placed via E4 can't be removed. `Journal::record_demolition`
exists (ticket 044) but has no caller; `City::remove_building` exists but is
only called as E4's own rollback, never deliberately.

## Goal

A key press on a hovered, placed building demolishes it: removes it from
`City`, writes the terrain its own *placement* baseline says stood there
back through the write path, and journals a `Demolished` entry recording
what the building's blocks actually were at demolition time (not
re-derived from the blueprint, so a damaged building demolishes as what it
actually was).

## Scope

- **Target resolution**: no G1 build menu, so `Delete` on
  `picking::HoveredBlock`'s tile — `City::occupant_at` finds the building id,
  `Journal::placement_baseline` finds what to restore. A building with no
  recorded baseline (an older save) refuses rather than guessing.
- **The restoring edit is the placement baseline's own `previous` half** —
  no re-derivation from the blueprint or the footprint, `Baseline::restore_edit`
  (already built for undo) is reused directly.
- **The demolition's own baseline** comes from `EditPolicy::capture_replaced`
  on the restoring write, the same way commit's baseline comes from its own
  write — `written` = the restored terrain, `previous` = the building's
  actual blocks read off the world at the moment of demolition.
- **Ordering, deliberately the mirror of commit's**: commit claims the tile
  in `City` *before* its write starts (so a second click can't race it).
  Demolish frees the tile only *after* its write succeeds — freeing it first
  would let a placement land on the same tile while the restore was still in
  flight, and whichever write reached disk last would clobber the other.
- **Cross-module write serialization.** `CommitState`/`DemolishState` each
  already refuse a second operation of their own kind while one is pending.
  What neither alone prevents is *each other* — two independent
  `WriteSession::open` calls, on unrelated tiles, at the same moment. Given
  `SessionLock::acquire` takes a mandatory lock on a freshly-opened file
  handle rather than on the process, a second concurrent open from this same
  app fails exactly like Minecraft already having the world open. Needs a
  small shared gate between `city::commit` and `city::demolish`.

## Out of scope

- G1's build menu — `Delete` is the whole UI, the same interim state number
  keys/`R`/`Escape`/height keys are already in.
- Undo of a demolition — `Journal::undo_last` already handles it (ticket
  044); wiring a key to it is separate.
- Damage detection (I2-I7) — this only lands the `Demolished` journal entry
  and the write; nothing here scans for player-caused damage.

## Done when

- `cargo build`/`cargo test` clean.
- Tests: resolving a demolition target on empty ground/a road tile is a
  no-op; on a placed-but-unjournaled building it refuses with a named id; on
  a journaled building it returns the right baseline. A successful demolition
  (against a real write-session fixture) removes the building from `City`,
  frees its tile, journals a `Demolished` entry whose baseline lines up with
  what was actually written and what was actually there before, and fires
  `ChunksEdited`. A failed write leaves `City` and the journal exactly as
  they were — no rollback needed, since nothing was mutated before the write.
  The write gate refuses a second acquire while one is held and frees it on
  release.
- Manual verification (place a building, demolish it, open the world in
  Minecraft and confirm the original terrain is back) goes in `../todo.md`.

## Resolution

Landed as scoped, in a new `city/demolish.rs` (declared alongside the rest of
`city`'s modules in `city/mod.rs`) plus a small new `city/write_gate.rs`.

**`resolve_demolition_target` is the pure decision function**, mirroring
`city::commit`'s split between plain functions (`blueprint_edit`,
`commit_building`) and the ECS systems that call them — testable directly
against a bare `City`/`Journal`, no `App` involved. It reads
`City::occupant_at` to find the id under the hovered tile, then
`Journal::placement_baseline` to find what to restore; a `Some(Occupant::Building)`
with no baseline comes back as its own `DemolitionTarget::NoBaseline` variant
rather than being folded into "nothing here," so `try_demolish` can print a
useful refusal (an older save, predating ticket 048's journal) instead of
silently doing nothing indistinguishable from hovering empty ground.

**The restoring edit needed no new code to build.** `Journal::Baseline::restore_edit`
already existed for `undo_last` (ticket 044) — widened from private to
`pub(super)` since `city::demolish` is a second, sibling caller — and is
exactly the edit a demolition needs: the placement baseline's own `previous`
positions and states, not a re-derivation from the blueprint or a second walk
of the footprint. `city::commit::commit_building` (the `WriteSession::open` +
`.commit` pair) also widened to `pub(super)` and is reused verbatim; opening a
session and committing an edit through it doesn't care which direction the
edit is going.

**`City::remove_building` moved to the very end of the transaction — the
mirror image of commit's own ordering, not an oversight.** Commit claims the
tile in `City` *before* its write starts, because a tile has to read occupied
the instant a click lands. Demolish has the opposite race to guard against:
freeing the tile before the restoring write finishes would let a placement
land on the same tile mid-write, and whichever of the two writes reached disk
last would silently clobber the other's blocks. So `try_demolish` never
touches `City` at all — only `poll_demolish`, and only in the `Ok` branch,
after the write has actually succeeded. This also means demolish's failure
path is shorter than commit's: nothing was mutated synchronously, so a failed
write needs no rollback, just a console line.

**`write_gate.rs`, the one piece the roadmap ticket didn't originally name.**
`CommitState`/`DemolishState` each already serialize themselves (a single
`pending` slot, same shape both tickets use), but neither alone stops a
commit and a demolition on *different* tiles from each opening their own
`WriteSession` in the same frame. That turned out not to be a theoretical
race: `ranvil`'s `SessionLock::acquire` takes a mandatory lock
(`File::try_lock`, `LockFileEx` on Windows) on a freshly-opened file handle,
not on the process — a second concurrent open from this same app fails
exactly like Minecraft already having the world open
(`WriteError::WorldIsOpen`), and there's no retry built into that error, so
it would have failed one of the two operations outright rather than merely
delaying it. `write_gate::WriteGate` is a `bool`-backed resource both plugins
`init_resource` (idempotent — whichever plugin builds first wins, the other's
call is a no-op), checked-and-set with `try_acquire` immediately before
spawning a task and released in each module's own poll system on both the
success and failure paths. Commit's own acquire point sits *after* its
synchronous `City::place_building`, so a failed acquire there rolls that
placement back — the same rollback shape commit's own write failure already
used, just triggered synchronously instead of through the task.

Testing: 10 new tests. `write_gate` (3): a second acquire fails while the
first is held, releasing frees it again, releasing an already-free gate is a
no-op. `demolish` (7): `resolve_demolition_target` on empty ground, on a road
tile, on an unjournaled building, and on a journaled one (4); `poll_demolish`
success (removes the building, frees its tile, journals a `Demolished` entry
whose `written`/`previous` line up with the restoring edit and its captured
report, fires `ChunksEdited`) and failure (leaves `City` and the journal
untouched, fires nothing) through a real `App` with a task result fixed ahead
of time, the same split `city::commit`'s own `poll_commit` tests use (2); and
one end-to-end test against a real fixture region file — place, capture the
placement baseline, demolish via `restore_edit`, confirm the original terrain
is back and the demolition's own baseline captured the building's actual
block as `previous` (1).

No manual verification recorded as done — `../todo.md` carries the checklist
(place a building, press `Delete` over it, confirm it disappears and the
original terrain is back both in-game and after reopening in Minecraft; try
demolishing while a commit is mid-write and confirm it's refused rather than
racing it), same as every other real-world check in this project.
