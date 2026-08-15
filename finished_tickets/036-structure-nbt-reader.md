# 036 - structure `.nbt` reader

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 263
tests). See the Resolution.

## Part of
Roadmap B1 (`tickets/CITYBUILDER_ROADMAP.md`) — the first ticket in group B
(blueprints as building models).

## Depends on
- 022's `Blueprint`/`BlockState` (`blueprint::extract`)
- 023's structure writer (`blueprint::structure`) — this is its inverse,
  and the round-trip target

## Problem

023 can write a `Blueprint` out as a vanilla structure file. Nothing reads
one back. B2-B4 (blueprint meshing, rotation, the asset catalogue) all need
a `Blueprint` in memory to work with, and today the only way to produce one
is a fresh extraction from a save — there's no way to load a building that
was authored in Minecraft itself (via a structure block's Save) or that
came from anywhere but this app's own export button.

## Goal

`blueprint::structure::read_structure_file` / `read_structure`: a gzipped
vanilla structure file in, a `Blueprint` out. The inverse of 023's writer,
proven by round-tripping through it.

## Scope

- `read_structure<R: Read>` / `read_structure_file(&Path)`, mirroring the
  writer's own split (a `Read`-taking function under a path-taking
  convenience wrapper) so tests can round-trip through a `Vec<u8>`.
- Reuses `BlockState::from_palette_entry` (already `pub(crate)`, written
  for extraction) for the palette — one sort-and-dedupe definition, not a
  second copy that could drift from the first.
- A dedicated `StructureReadError`, shaped like `world::DecodeError`'s two
  catch-alls (`MissingField`/`UnexpectedType`) rather than one variant per
  malformed shape, plus `PaletteTooLarge`/`PaletteIndexOutOfRange`/
  `TooLarge` for the cases those two don't cover. `TooLarge` reuses
  `blueprint::MAX_BLOCKS`, checked before allocating, so a corrupt or
  hostile `size` field can't be used to exhaust memory.
- Round-trip tests against 023's own writer (synthetic fixtures, byte-level
  edge cases: single block, oversized structure, properties), plus a
  real-save round trip mirroring 023's `writes_a_real_box_from_the_save`:
  extract → write → read → compare against the extracted blueprint.
- Malformed-input tests built by hand-assembling NBT (not through the
  writer, which can't produce these shapes): missing `DataVersion`, a
  palette index out of range, a position outside the declared `size`, a
  `blocks` list shorter than the volume, a position claimed twice, and more
  than 65,536 distinct palette entries.

## Watch out

- **`Blueprint::origin` isn't recoverable.** 023 deliberately never writes
  it (a structure is position-independent) — the reader has nothing to read
  it back from. Comes back `IVec3::ZERO`; a caller placing the building
  (E-group) supplies its own world position. Don't try to infer one.
- **`Blueprint::failed_columns` means something specific elsewhere**
  (`blueprint::extract`'s per-column walk, where one bad chunk doesn't lose
  the whole extraction). Reading a structure file is one parse,
  all-or-nothing — it comes back `0` rather than being repurposed.
- **The format technically allows a sparse `blocks` list**; the writer
  never produces one, and neither does an in-game Save. Rather than
  guessing "air" for an unlisted position (which would silently carve holes
  in a building nobody asked for), a gap is treated as a malformed file.
  That's a deliberate strictness call, not an oversight — see the module
  docs' "The reader's decisions".

## Out of scope

Everything else in group B: meshing a `Blueprint` into a Bevy `Mesh` (B2),
rotation (B3), the asset catalogue that will actually call this reader
(B4). This ticket only gets a `Blueprint` back into memory from a file.

## Done when

- `read_structure`/`read_structure_file` exist and round-trip every
  synthetic fixture 023's writer already covers, plus a real box extracted
  from the save.
- Malformed input (missing fields, out-of-range indices/positions, a
  non-gzipped file, a missing file) errors rather than panics.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed — `blueprint::structure::{read_structure,
read_structure_file, StructureReadError}`, re-exported from `blueprint` the
same way the writer's own public surface is.

`StructureReadError` ended up with six variants: `Io` (open/gunzip/NBT
parse failure — folds in `NbtError`'s non-I/O variants the same way
`write_field` already collapses them in the other direction),
`MissingField`/`UnexpectedType` (the `world::DecodeError`-shaped
catch-alls), `PaletteTooLarge`, `PaletteIndexOutOfRange`, and `TooLarge`.

The one design decision worth recording: `read_blocks` validates the
`blocks` list in two passes over the same `Vec<Option<u16>>` — fill by
position while checking for duplicates, then check nothing was left
unfilled. The length check (`entries.len() == volume`) alone isn't enough:
a file with the right *count* but a duplicated position and a missed one
would pass it while still being wrong, and the earlier draft of this
(`.expect()`-ing every slot to be `Some` after the loop, reasoning that the
length+duplicate checks together guaranteed full coverage) would have been
a panic waiting for exactly that input. The explicit post-loop check
(`slots.iter().any(Option::is_none)`) is the fix, and it's covered by
`a_position_claimed_twice_is_an_error`, which is built precisely to have
the right length while still missing a position.

`Blueprint` gained `PartialEq` (previously `Debug, Clone` only) so the
round-trip tests could compare structurally instead of field-by-field;
nothing else in the codebase depended on its absence.

14 new tests in `blueprint::structure::tests` (263 total, up from 249):
round trips (whole-blueprint, single-block, oversized, properties, real
save via `reads_back_a_real_box_extracted_from_the_save`), the
path-taking wrapper end to end, and the seven malformed-input cases above.
All pass, including against the real save (`nbt_test`).

No manual/in-game check is needed for this ticket specifically — nothing
here writes to a world, and the real-save test already proves the read
side against real chunk NBT. The in-game half of "authored in Minecraft"
(saving a building with a structure block and loading it through this
reader) is B4's concern, once there's a catalogue to load it into.
