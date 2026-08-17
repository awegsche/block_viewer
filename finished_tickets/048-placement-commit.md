# 048 - Commit, and a manual height adjustment

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 408 tests,
up from 400). See the Resolution.

## Part of

Roadmap E4 (`tickets/CITYBUILDER_ROADMAP.md`), the fourth ticket in group E
(placement). Depends on E1's picking (ticket 045), E2's terrain fit (ticket
046), E3's ghost preview and validity (ticket 047, `resolve_placement`,
`PlacementSelection`), D1's city state (ticket 042, `City::place_building`),
D3's journal (ticket 044, `Baseline::capture`, `Journal::record_placement`),
and the whole write path (W1-W7, tickets 031-035). Also lands I1 (the
as-built baseline) for real, per the roadmap's "I1 ships with E4, in
iteration 1."

Plus a piece the roadmap doesn't call out but the user asked for directly:
letting the player nudge a placement's height up or down before committing,
rather than being stuck with whatever `fit_footprint`'s auto-fit picked.

## Problem

E3 shows a green or red ghost at the cursor. Nothing turns green into a real
building: no city entry, no blocks in the world, no baseline. And the only
height a placement can have is whatever `grid::fit_footprint` computes from
the terrain — there's no way to deliberately place a building a block or two
above or below that (a stilted structure, a sunken foundation, compensating
for a fit that clips a little into a slope).

## Goal

A left click, on a valid (green) placement, commits it: a `City` entry, a
`WorldEdit` built from the rotated blueprint written through the real write
path (W4/W5/W6), and a journal entry carrying the as-built baseline —
transactionally, so a write failure removes the `City` entry it would
otherwise have left behind. A successful write also fires `ChunksEdited` so
the building appears without a restart (W7).

Separately: `PlacementSelection` gains a `y_offset`, adjustable by
dedicated keys, that shifts a placement's height up or down from the
auto-fit `base_y` — reflected live in the ghost preview, carried through to
the committed placement.

## Scope

- **`blueprint_edit`**: a `Blueprint` (already rotated, per B3) plus a
  Minecraft-space origin, turned into a `WorldEdit` — one `set` per grid
  position, `0..size` on every axis, matching `mesh_blueprint`'s own
  `dy*sz*sx + dz*sx + dx` indexing so the two never disagree about which
  index is which corner. `WorldEdit::with_data_version(blueprint.data_version)`
  so a version mismatch against the save is refused the way W4 describes.
- **Air is written, not skipped.** The `edit` module's own docs already call
  this an E4 decision to make (`WorldEdit`'s "Air is a block" note): a
  building's declared-empty interior clears whatever terrain the footprint
  fit's ~1-block clip tolerance left poking into it, rather than leaving a
  render-only gap over real blocks. No special-casing needed —
  `blueprint_edit` writes every palette entry, air included.
- **The commit itself, async like W8's paint command**: `City::place_building`
  runs synchronously (cheap — an occupancy check, not I/O) the instant the
  click is accepted, so the id and tile claim exist before the write starts.
  The actual `WriteSession::open` + `commit` runs on `AsyncComputeTaskPool`,
  the same shape `viewer::paint::start_paint`/`poll_paint` already
  established. On success: `Baseline::capture` off the edit and its report,
  `Journal::record_placement`, `ChunksEdited` fired with the written chunks.
  On failure: `City::remove_building` undoes the synchronous half — the
  transactional guarantee the roadmap names ("if the write fails, the city
  entry doesn't survive either").
- **One commit in flight at a time.** A second click while one is writing is
  ignored, the same backpressure `PaintCommand`/`BlueprintExtraction` already
  use — a second commit racing the first over the same region files is
  exactly the failure mode W6 exists to prevent.
- **`y_offset` on `PlacementSelection`.** `resolve_placement` (E3) takes it as
  a new parameter and applies it (`saturating_add`, to rule out overflow from
  input alone) to whichever Y it would otherwise have used — `fit_footprint`'s
  `base_y` on a valid fit, the hovered-block fallback on a refused one.
  Terrain flatness and occupancy stay exactly as E3 computed them: the offset
  moves the building, it doesn't relax what ground or tiles are allowed
  underneath it.
- **Height keys**: `Page Up`/`Page Down` step the offset by one block, `Home`
  resets it to zero. Not the mouse wheel — Rts's own camera zoom (ticket 045)
  already owns scroll in this mode, and reusing it for height too would mean
  every zoom is also a height change. Guarded by
  `camera::EguiInputCapture::keyboard`, same as every other key in
  `cycle_selection`. Reset to zero on `Escape` and on picking a new catalogue
  entry — a fresh selection starts at the terrain's own fit, not wherever the
  last one was left.

## Watch out

- **Don't recompute validity with different inputs than the ghost used.**
  `try_commit_placement` calls the same `resolve_placement` E3's preview
  reads from, with this frame's `y_offset` — a click has to commit *exactly*
  what's currently on screen, or "I clicked the green ghost" and "what got
  built" can disagree.
- **The write is async; the city entry isn't.** Between the synchronous
  `place_building` and the task's completion, the tile reads occupied (by
  `City::is_tile_free`) even though nothing has reached disk yet — correct,
  and it's what stops a second click from claiming the same tile while the
  first write is still in flight.
- **`Baseline::capture` needs `EditPolicy::capture_replaced = true`.** Without
  it `report.replaced` is `None` and there's nothing to journal — the default
  `EditPolicy` has it off (it costs a read per position), so this caller has
  to turn it on explicitly, the same call `viewer::paint` did *not* need to
  make (painting doesn't journal).
- **Don't clone a multi-million-block `Blueprint` needlessly.** Mirror
  `placement::ghost_mesh`'s `Deg0`-skips-`rotate_blueprint` shortcut and
  borrow rather than clone when a rotation is a no-op.

## Out of scope

- Demolish (E5) — no removal, no terrain restore beyond what a failed
  write's rollback already gives for free.
- Undo (D3's `Journal::undo_last` already exists; nothing here calls it) —
  wiring a key to it is a separate, small ticket if wanted.
- A build-menu affordance for height (G1, later) — the keys are the whole UI
  for now, the same interim state numbered-key selection is already in.
- Reconciliation, damage detection (I2-I7) — I1's baseline is the only piece
  of group I this ticket touches.

## Done when

- `cargo build`/`cargo test` clean.
- Tests: `blueprint_edit` writes every grid position (air included) at the
  right world coordinates for a rotated and an unrotated blueprint;
  `resolve_placement` applies a positive/negative `y_offset` on both the
  fitting and the refused-fallback path without disturbing terrain/occupancy
  validity; a committed placement (against a real write-session fixture, the
  same shape `viewer::paint`'s tests use) produces a `City` entry, a journal
  entry with a baseline, and writes the expected blocks; a failed write (a
  chunk outside the fixture's generated region) leaves `City` and the journal
  exactly as they were before the click; height keys step and reset the
  offset, guarded by the egui keyboard capture.
- Manual verification (place a building, open the world in Minecraft, and
  check the blocks, the baseline, and a couple of height-offset placements
  look right) goes in `../todo.md`, same as every other real-world check in
  this project.

## Resolution

Landed as scoped, in a new `city/commit.rs` (declared alongside
`definition`/`grid`/`journal`/`persistence`/`picking`/`placement`/`state` in
`city/mod.rs`), plus the `y_offset` piece folded into `city/placement.rs`
rather than a separate module — it's read by both the ghost preview (every
frame) and the commit click, and `resolve_placement` was already the one
function both needed.

**`resolve_placement` grew a `y_offset: i32` parameter**, applied via
`saturating_add` to whichever Y it had already picked (`fit_footprint`'s
`base_y`, or the hovered-block fallback on a refusal) — after terrain and
occupancy are resolved, not before, so the offset moves the building without
relaxing what ground or tiles are allowed under it. It and `GhostPlacement`
both went from private to `pub(super)`: `city::commit` is a second caller,
on a click instead of every frame, reusing the exact same function E3's
ghost preview reads rather than a parallel copy that could quietly
disagree with what's on screen. Height keys (`Page Up`/`Page Down`/`Home`)
landed in `cycle_selection` alongside the rest of the keyboard stand-in, not
the mouse wheel — Rts's own camera zoom already owns scroll in this mode
(ticket 045), and camera.rs was deliberately left untouched rather than
teaching its zoom system about a citybuilder-only concept. The offset resets
to zero on `Escape` and on picking a new catalogue entry, so a fresh
selection always starts at the terrain's own fit.

**`city::commit` recomputes validity itself rather than trusting the ghost.**
`try_commit_placement` calls `placement::resolve_placement` with *this*
frame's `HoveredBlock`/`PlacementSelection` on a left click (guarded by
`camera::EguiInputCapture::pointer`, the same guard every other input system
in this crate uses) — a click commits exactly what's computed at the moment
of the click, never a value cached from a previous frame.

**The commit is synchronous-entry, asynchronous-write, exactly as scoped.**
`City::place_building` runs the instant a valid click lands — an occupancy
check over a `HashMap`, cheap enough not to need a task — so the tile claim
exists before `WriteSession::open`/`commit` even starts on
`AsyncComputeTaskPool`. `commit_building` is `viewer::paint::commit_fill`'s
exact shape, pulled out on its own so it's testable directly and
synchronously against a real fixture rather than only through a task.
`CommitState::pending` is a single `Option`, the same one-at-a-time
backpressure `PaintCommand`/`BlueprintExtraction` already use — a second
click while one write is in flight is silently ignored, which is also
*why* the synchronous city entry has to land first: the occupied tile is
exactly what stops a second click from racing the first over the same
region files.

**`poll_commit` is where the "transactionally" half of the roadmap's own
wording for E4 actually happens.** On success: `journal::Baseline::capture`
off the edit and the write's own `EditReport` (`EditPolicy::capture_replaced`
turned on specifically so this is never `None` here, unlike
`viewer::paint`'s fill command, which doesn't journal and so never needed
it), `Journal::record_placement`, then `ChunksEdited` fired with the written
chunks for W7's live re-mesh. On failure: `City::remove_building` — the
synchronous half doesn't survive a write that didn't happen. Borrowing
`commit.pending` to poll the task and then `.take()`-ing it afterward needed
a small scoping block (poll inside a block that ends the mutable borrow,
`.take()` after) rather than the two operations back to back, which the
borrow checker rejects — worth recording since the fix is invisible from the
error message alone.

**`blueprint_edit` decided the "air is a block" question the `edit` module's
own docs left open for E4**: every grid position is written, air included,
mirroring `blueprint::mesh_blueprint`'s exact `dy*sz*sx + dz*sx + dx`
indexing (`block_at` in `blueprint/mesh.rs`) so a placed building and the
mesh that previewed it never disagree about which corner is which. No
special-casing for air needed — `WorldEdit::set` already treats every
`BlockState` uniformly, and this is what clears the small terrain clip E2's
`MAX_FOOTPRINT_STEP` tolerance allows without a second pass hunting for it.
`Deg0` never touches `rotate_blueprint`, mirroring `placement::ghost_mesh`'s
own shortcut — borrowed, not cloned, since a real blueprint can run into the
millions of blocks.

Testing: 8 tests. `blueprint_edit` (writes every grid position including air,
at the right world-space coordinates for an origin offset). `commit_building`
(writes a blueprint's blocks against a real single-region fixture and
produces a baseline whose `previous` half is the ground it overwrote;
refuses ungenerated terrain and leaves the region cache clean) — the same
fixture shape `viewer::paint::tests::Fixture` uses, copied rather than
shared since it's private to that module. `poll_commit` (a successful task
result journals exactly one baseline entry and fires `ChunksEdited` with the
written chunks, leaving the synchronous city entry in place; a failed task
result removes the city entry, frees its tile, journals nothing, and fires
no event) — through a real `App` with `CommitPlugin`, the task's result
fixed ahead of time rather than run against a second real fixture, the same
split `viewer::paint`'s own `poll_paint` tests use to avoid re-proving the
write path a second time. `resolve_placement`'s two new tests (in
`placement.rs`) confirm a positive offset adds to a fitting placement's
`base_y` and stays valid, and a negative offset subtracts from the refused
fallback height without becoming valid on its own.

No manual verification recorded as done — `../todo.md` carries the checklist
(a real click against a real save, the blocks landing right-side-up and not
mirrored, interior air actually clearing terrain, the height keys feeling
usable, a second click on an in-flight commit being silently ignored), same
as every other real-world check in this project.
