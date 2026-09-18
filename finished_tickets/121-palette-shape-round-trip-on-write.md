# 121 — palette shape round-trip on write (savegame `01`, buildings come out as air)

## Symptom

Every building the citybuilder places into savegame `01` (`DataVersion` 5023)
comes out as **pure air across its entire footprint** once Minecraft has
loaded and re-saved the world — reported by the user as "chunks modified by
the citybuilder seem to not get loaded by minecraft."

Confirmed directly against the real save (not the game — file inspection
only, per this repo's manual-verification rule):

- The write-session backups `WriteSession` leaves under
  `<save>/block_viewer_backups/` show real terrain at `house01`'s exact
  footprint in the oldest backup (pre-edit), and total air in every backup
  taken after it — i.e. the footprint reads as air even in the citybuilder's
  own before/after snapshots, not just "after Minecraft touched it."
- `blueprint_edit` (`src/city/commit.rs`) was tested directly against the
  real `house01.nbt` asset: it produces a correct `WorldEdit` (282 non-air
  blocks of 900, real block names at the right positions).
- The full write path — `blueprint_edit` → `apply_building_edit` →
  `WriteSession::flush` — was tested against a **copy of the actual live
  region file** and round-tripped correctly (100/100 non-air blocks in the
  test layer, verified after a save-and-reload from a fresh cache).

So the bug is not in blueprint loading, `blueprint_edit`'s indexing, or
`ranvil`'s write path — all three are individually correct. The only
untested link left is the real Minecraft client itself loading and
re-saving an edited chunk, which this repo's rules don't let Claude verify
directly (needs a human at the game).

## Root cause (hypothesis, backed by the evidence above, not yet confirmed in-game)

Ticket `../ranvil/finished_tickets/120-chunk-palette-shape-drift.md` fixed
*reading* the three newer `block_states.palette` shapes `DataVersion` 5023
writes (bare string, single-unnamed-field compound, `id`/`properties`
compound) back into the legacy `{Name, Properties}` shape, so this crate's
own readers never see them. It deliberately left *writing* alone:
`ChunkRegion`'s `edit_section` always re-emits the legacy shape, on the
stated assumption ("already the crate's established migration policy") that
a newer game build keeps reading a save's older layout.

That assumption is the thing this ticket doubts. A chunk's `DataVersion` tag
is left untouched by an edit — still 5023 — so when the real game loads it
back, it has no reason to run its data-fixer upgrade path (that only fires
for a *lower* stored `DataVersion` than the game's current one; here they're
equal). If the 5023 palette codec expects the compact shapes it itself
writes and doesn't also accept the legacy one when the tag claims the
chunk is already current, the most plausible failure mode is exactly what's
observed: the edited section comes back unreadable and gets replaced with
air once the game re-saves the chunk.

## Fix

Make `edit_section` round-trip the palette in whichever family it was
originally written in, instead of always downgrading to legacy:

- `ChunkRegion::load_chunks` records, per section (keyed by the section's
  `Y` tag, alongside the existing legacy-normalization pass), whether that
  section's `block_states.palette` was already legacy or one of the three
  compact shapes — a new `palette_shapes` field, same lifetime as `chunks`/
  `raw` (reset on the next load).
- `edit_section` takes that section's recorded shape and, for a
  *compact*-shaped section, re-emits every entry (both untouched ones and
  any newly-appended palette entry a batch introduces) as a `TAG_COMPOUND`
  using the two compact variants ticket 120 actually found together in the
  real save: a single unnamed string field when the block has no
  properties, `id`/`properties` when it does. A *legacy*-shaped section is
  unchanged — still written the old way, which is what a chunk from an
  older game build needs.
- This is a per-section decision, not a per-chunk or per-save one — ticket
  120 already established that one chunk can mix sections written by
  different game builds.

## Explicitly not attempted

- Reproducing the bare-string-list compact shape ticket 120 also found. A
  `TAG_LIST` element type is homogeneous, so a palette list can only be a
  list of strings when *every* entry has no properties at all — the moment
  one entry needs `properties`, the whole list has to be `TAG_COMPOUND`
  entries. The single-unnamed-field compound is the no-properties encoding
  used here instead, for exactly that reason: it mixes freely with
  `id`/`properties` entries in the same compound list, and ticket 120 found
  both forms present together in the real save.
- Confirming this against the actual game. That's this ticket's manual-
  verification item in `todo.md` — place a building, save, quit, reopen in
  Minecraft, quit again, and check the footprint survived. Nothing here is
  "done" in the sense the roadmap otherwise uses that word until a human has
  done that.

## Done when

- `ranvil`'s `cargo test` passes, including new coverage: editing a
  compact-shaped section writes the compact shape back (not legacy), a
  brand-new palette entry added to a compact section is also written
  compact, and a legacy-shaped section's write is unchanged (regression).
- `block_viewer`'s `cargo test`/`cargo check` pass.
- A `todo.md` manual-verification entry exists pointing here, since the
  load-bearing question — does the real game actually keep the blocks now —
  can only be answered by a human opening Minecraft.

## Resolution

Implemented as designed. `ranvil/src/chunkregion.rs`:

- `PaletteShape` (`Legacy` / `Compact`) — two variants, not four: the three
  compact shapes ticket 120 found all decode identically once normalized,
  and there's no attempt to remember which of the three a given *entry*
  started as (see `render_compact_entry`'s doc comment for why the
  bare-string-list variant specifically can't always be reproduced — a
  `TAG_LIST`'s element type is uniform, so it's only valid when *every*
  entry in that palette has no properties).
- `ChunkRegion` gained a `palette_shapes: Option<Vec<BTreeMap<i32,
  PaletteShape>>>` field, same lifetime as `chunks`/`raw`, populated by
  `load_chunks` in the same pass that normalizes each section's palette to
  the in-memory legacy shape (`normalize_palettes`/`normalize_palette_field`
  now return what they saw instead of discarding it).
- `edit_section` takes the section's recorded shape and, for `Compact`,
  re-emits every entry (untouched and newly-appended alike) via
  `render_compact_entry`: a single-unnamed-field compound for a
  no-properties block, `id`/`properties` for a stateful one — the two
  compound shapes ticket 120 found together in `01`'s real save, chosen
  over the bare-string-list shape specifically because they mix freely in
  one palette. `Legacy` sections are untouched.
- `ChunkRegion::palette_shape` looks the shape up by (chunk slot, section
  `Y`), defaulting to `Legacy` for a slot `set_blocks` reaches before ever
  loading (not reachable in practice — `set_blocks` always loads first).

Five new tests in `chunkregion.rs`'s `palette_drift_tests`: a compact
section's edit stays compact (plain block), the same with a stateful block
(asserts the `id`/`properties` shape specifically, since the no-properties
shape can't carry one), a legacy section's edit stays legacy (regression),
and `load_chunks` actually records the shape it read off a real
round-tripped region file.

`ranvil`: `cargo test` — 148 passed, 0 failed, 1 ignored (a helper process,
not a test). `block_viewer`: `cargo check` clean (pre-existing dead-code
warnings only, unrelated); `cargo test --lib` — 1125 passed, 1 failed
(`city::road_build::tests::a_building_placed_beside_an_existing_surface_cell_rewrites_it_connected`,
an `AsyncComputeTaskPool`-not-initialized test-ordering flake — reproduced
identically against the unmodified tree via `git stash`, so unrelated to
this change and the same flake ticket 120's own resolution notes already
named).

### What's still open

The root cause is a hypothesis, not a confirmed one — nothing here can be
checked against the real game (per this repo's manual-verification rule),
only against this crate's own read/write round trip, which was already
self-consistent before this fix in the sense that it never threw an error.
The `todo.md` entry this ticket added is the actual test: does a building
placed in `01`, saved, and then loaded and re-saved by real Minecraft still
have its blocks afterward. If it does, this is done. If it doesn't, the
hypothesis was wrong (or incomplete — e.g. the real codec might be pickier
about the bare-string-list case this fix deliberately doesn't reproduce,
or the failure might not be palette-shape-related at all), and that check
is where the next look starts, not back here.
