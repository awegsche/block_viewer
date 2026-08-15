# 035 - W8: a paint/fill command in the viewer

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 249
tests). See the Resolution.

## Part of
Roadmap W8 (`tickets/CITYBUILDER_ROADMAP.md`) — the gate the rest of the
roadmap waits on. W1-W7 are all "done" per the roadmap table, but nothing in
the app has actually called any of it yet; this ticket is the first thing
that does.

## Depends on
- 031 (the edit model), 032 (boundary routing/region batching), 033 (write
  safety: lock, backup, atomic), 034 (live re-mesh on edit)
- 021's selection panel, which this is driven from per the roadmap's
  instruction ("driven from the existing selection panel" — no new window)

## Problem

Every ticket from 031 through 034 built a piece of the write path and proved
it against synthetic fixtures. None of them are reachable from the running
app: there is no button, keystroke or command anywhere that turns a
[`crate::selection::Selection`] into a call to
[`crate::edit::session::WriteSession`]. The roadmap calls this out
explicitly — W8 exists to prove W1-W7 end to end *before* any game code is
written, because a bug in the write path is far cheaper to find here than
after the citybuilder's placement/demolish/terraform features are all built
on top of it.

## Goal

A "Fill" command in the existing selection panel: type a block name (with
optional properties), click "Fill", and every position in the current
selection is written into the save's region files for real — refused if
Minecraft has the world open, backed up first, written atomically, and the
affected chunks re-mesh on screen without a restart.

## Scope

- `WorldEdit::fill(bounds, state)` (`edit/mod.rs`) — builds a fill edit from
  a [`SelectionBounds`] and one [`BlockState`], reused later by terraforming
  (H1), which is the same write path with a different source of block
  changes.
- `BlockState: FromStr` (`blueprint/extract.rs`) — the inverse of the
  `Display` impl 022 already has, so a text field is a way *in* as well as
  a way out. `name[key=value,...]`, with a bare name defaulting to the
  `minecraft:` namespace.
- `viewer::paint` (new, viewer-only) — `PaintCommand`/`PaintState`, the same
  one-slot state-machine-plus-task shape `blueprint::export` already
  established: `start_paint` opens a `WriteSession` and commits on
  `AsyncComputeTaskPool`, `poll_paint` reports the result and fires
  `ChunksEdited` with the written chunks on success, handing off into 034's
  reload queue.
- The selection panel grows a "Paint" section: a block-name field, a "Fill"
  button (disabled on a parse error, an over-cap selection, or a paint
  already running — the same [`VOLUME_CAP`]/[`VOLUME_WARN`] thresholds the
  export button already uses), and a status line.

## Watch out

- **The edit's regions stay locked for the whole commit**, unlike
  extraction's per-chunk-column locking. A commit is one transaction —
  plan every region, then apply every region, then back up and save every
  one — and releasing the shared region-cache lock partway through would
  let a streaming load or another edit observe or evict a region mid-edit,
  breaking 032's all-or-nothing guarantee. Documented as a deliberate
  trade-off rather than an oversight: painting is an occasional deliberate
  click, not continuous background work.
- **Deliberately not "stamp a loaded blueprint"**, the roadmap's other
  suggested command. There is no blueprint *reader* yet (B1) — the only
  in-memory `Blueprint` today is whatever the last extraction produced, and
  building a blueprint-to-`WorldEdit` converter now would be scoped by B1's
  future rotation/property rules (B3) rather than this ticket's. Fill alone
  already exercises every layer W1-W7 built; stamping is better done once
  there's a real blueprint to stamp.
- **No baseline capture** (`EditPolicy::capture_replaced` stays off). The
  as-built baseline (I1) is a city-game concern tied to a placed building's
  record; a manual viewer fill has no such record to attach it to. Nothing
  here stops a future undo command from turning it on.

## Out of scope

- Stamping a blueprint (see above) — B1/B2/B3 first.
- Undo for a paint command — no baseline is captured; see above.
- Manual verification that the write is correct against a real save opened
  in Minecraft — noted in `./todo.md` per this repo's rule on visual
  checks, not something this ticket can confirm itself.

## Done when

`cargo build`/`cargo test` clean, including coverage for: `WorldEdit::fill`
building one edit per block in the bounds and landing them all (`edit`'s
tests), `BlockState::from_str` round-tripping through `Display` and
refusing malformed input (`extract`'s tests), and `viewer::paint`'s state
machine — busy-guards a second request, reports "no save loaded" without a
region cache, and (via a `commit_fill` helper pulled out of the task body
the same way `chunk_pipeline::load_and_mesh_chunk` is) writes every block
in a synthetic fixture and refuses ungenerated terrain — plus the
`poll_paint` plumbing: a finished commit becomes `Done` and fires
`ChunksEdited` with the written chunks, a failed one becomes `Failed` and
fires nothing.

## Resolution

Landed as designed. `WorldEdit::fill` is an inherent method on `WorldEdit`
in `edit/mod.rs`, built with `bounds.iter_blocks().map(...).collect()`
through the existing `FromIterator<BlockEdit>` impl — no new type.

`BlockState::from_str` lives in `blueprint/extract.rs` next to `Display`,
the two format definitions kept together. It trims whitespace throughout
(the same forgiveness the coordinate fields already give), sorts properties
by key (matching `from_palette_entry`'s convention, so the palette dedupe
rule holds for typed-in states too), and prefixes a bare name with
`minecraft:` — every `Display` output round-trips through it unchanged
(tested).

`viewer::paint` is a new private module (`mod paint;` in `viewer/mod.rs`,
sibling of `ui`), not shared with the citybuilder — the roadmap's own
phrasing ("viewer: a paint/fill command in the viewer") and 027's docs
already draw that line for viewer-only interaction code. `PaintCommand`
mirrors `BlueprintExport`'s `request`/`busy`/`state` surface exactly,
`PaintPlugin` registers `ChunksEdited` itself (idempotent, and keeps the
plugin from silently depending on `ChunkLoadPipelinePlugin` having run
first) alongside `poll_paint`/`start_paint` chained the same order
`export`'s systems are.

The task body is a standalone function, `commit_fill(save, cache, edit,
policy)`, called both from inside the spawned task and directly from tests
— the same split `chunk_pipeline::load_and_mesh_chunk` uses, and for the
same reason: a real `AsyncComputeTaskPool` task's completion timing isn't
something a test should depend on, so the actual write is tested
synchronously and the task/poll plumbing is tested separately with
manually-constructed tasks spawned directly onto the real task pool
(`export.rs`'s own tests do the same). Those tasks' completion turned out
to be a genuine race rather than a hypothetical one — one early run saw
`poll_paint` observe the task still in flight after a single `app.update()`
— so the two poller tests tick the app in a bounded loop
(`run_until_settled`) until `PaintCommand` stops being busy, rather than
assuming one frame is enough. `paint.rs` carries its own minimal
single-chunk fixture rather than reaching into `edit::tests`' private
`RegionFixture`/`SaveFixture`.

The selection panel's new "Paint" section sits between the export button's
readout and the export status line — `paint_controls` (the field and
button) inside the `Some(bounds)` arm, `paint_status` alongside
`export_status` outside it, so a paint keeps reporting even if the
selection it started from is cleared or moved while it's writing, same as
an export. A `PaintDraft` local holds the typed block name; unlike
`BoundsDraft` it never re-syncs from anything, since the block to fill with
has nothing to do with where the selection currently is.

`viewer/ui/mod.rs`'s module doc, which claimed every panel but the
selection panel's bounds fields was read-only, is updated to say what's now
true: the Fill button is a real write, gated by the same safety rules W6
built.

New tests: `WorldEdit::fill` (builds the right edit, lands every block, in
`edit/tests.rs`), `BlockState::from_str` (bare names, properties, sorting,
whitespace, round-trip through `Display`, malformed input — in
`extract.rs`), and `viewer::paint`'s own module (`commit_fill` writing and
refusing directly, `PaintCommand`'s busy guard, the no-save-loaded fast
path, and `poll_paint`'s Done/Failed transitions with the `ChunksEdited`
event). 249 tests total (was 234 before this ticket).

A `todo.md` entry records the manual check this ticket can't do itself:
open the world in Minecraft after a fill and confirm the blocks, their
properties, and the surrounding terrain are what they should be.
