# 100 - ranvil-cli: `struct get`, `struct set`, `struct fill`

The first commands that edit a `Blueprint` in memory outside of extraction
and rotation — nothing today needs to poke a single block into an
already-loaded structure file, so this ticket adds that, scoped tightly to
exactly what `get`/`set`/`set-area`'s world-side equivalents already do.

## Scope

All three take `<file.nbt>` as their first argument, `read_structure_file`
it, operate on the in-memory `Blueprint`, and — for `set`/`fill` — write the
result back with `write_structure_file`.

- **`ranvil-cli struct get <file.nbt> <x,y,z>`** — `Blueprint::block_at`
  (already exists, used by `extract.rs`'s own tests). Position is relative
  to the structure's own `0..size` space, not a world coordinate — say so
  in `--help`, since every other `get`-shaped command in this plan reads
  world coordinates and this is the one exception. Out-of-bounds is
  `CliError::Usage` (a bad request against this file's known size), not
  `CliError::Data` — unlike a live save, a structure file's extent is fully
  known up front from `struct info`, so there's no "maybe it's just
  ungenerated" ambiguity to preserve.
- **`ranvil-cli struct set <file.nbt> <x,y,z> <blockstate> [--out
  <file2.nbt>] [--force]`** — sets one position's palette index (inserting
  a new palette entry if `<blockstate>` isn't already in it — same "does
  the palette need to grow" logic `ranvil::chunkregion::set_blocks`
  performs for a live section, mirrored here in `Blueprint`-space rather
  than reused directly, since `Blueprint`'s dense array and a section's
  packed-bits array are different representations of the same idea).
  Without `--out`, overwrites `<file.nbt>` in place (still refuses without
  `--force` unless `--out` is given — editing a file "in place" is itself a
  destructive default worth gating the same way `struct new`'s overwrite
  is). With `--out`, the original is untouched and the edited copy is
  written fresh.
- **`ranvil-cli struct fill <file.nbt> <x1,y1,z1> <x2,y2,z2> <blockstate>
  [--out <file2.nbt>] [--force]`** — the same in-bounds box-fill as `set`,
  batched (one palette-growth pass, not one per block, mirroring why
  `set_blocks` exists over calling `set_block` in a loop). This is the
  command that answers "carve a chest well" / "clear a floor" from the
  roadmap's motivating jobs.

## Done when

- `struct get`/`struct set` round-trip: setting a position and reading it
  back (same file, or via `--out` then reading the new file) agree.
- `struct set` on a blockstate not already in the palette grows the palette
  by exactly one entry; setting to a state already present does not
  duplicate it.
- `struct fill` over a sub-box changes only that sub-box — verified against
  `struct info`'s block count before/after (unchanged, since fill replaces,
  doesn't add) and a couple of sampled `struct get` calls outside the
  filled box.
- Out-of-bounds coordinates on any of the three exit 2 with the file's
  actual size named in the message.
- `cargo check` and `cargo test` pass.
