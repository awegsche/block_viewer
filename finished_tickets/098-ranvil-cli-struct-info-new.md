# 098 - ranvil-cli: `struct info`, `struct new`

First tickets in group E (structure files). Per `RANVIL_CLI_ROADMAP.md`'s
ordering advice, group E needs only 087 — nothing here opens a save.

## Scope

- **`ranvil-cli struct info <file.nbt>`** — `blueprint::structure::
  read_structure_file`, then report `Blueprint`'s own fields: `size`,
  `origin` (always `IVec3::ZERO` per `structure`'s doc comment — report it
  anyway rather than omitting it, so the field is present and consistent
  with `get-area`/`struct export`'s output shape), block count
  (`Blueprint::volume()`), `data_version`, palette (size, and the full
  listing — unlike `get-area`, a structure file's palette is small enough
  that `text` mode lists it by default, capped the same way
  `blueprint/mod.rs::log_blueprint`'s `LOGGED_PALETTE_ENTRIES` already caps
  the viewer's own console log — reuse that constant rather than picking a
  new number). A malformed or oversized file is `CliError::Data` carrying
  `StructureReadError`'s own message, not a panic.
- **`ranvil-cli struct new --size <x,y,z> --out <file.nbt> [--fill
  <blockstate>]`** — builds a `Blueprint` of the given size, every position
  set to `--fill` (default `minecraft:air`, `BlockState::AIR`), and writes
  it with `write_structure_file`. `--size` components must each be at least
  1 and within `STRUCTURE_BLOCK_MAX_SIZE` (the same cap `structure.rs`
  already enforces on write — check it here too, before allocating a dense
  array for a size a later write would reject anyway). Refuses to overwrite
  an existing file without `--force`, the same convention every other
  file-writing `struct` command (100–102) follows.

## Done when

- `struct info` against a real shipped blueprint (`assets/city/blueprints/
  house01.nbt` or similar) reports a size/palette matching what the
  citybuilder's own catalogue load logs at startup for the same file.
- `struct new --size 5,5,5 --out <tmp>.nbt` produces a file `struct info`
  reports as `5x5x5`, all-air, palette size 1; `--fill minecraft:stone`
  produces palette size 1 of stone.
- `struct new` onto an existing path without `--force` refuses; with
  `--force` overwrites.
- `cargo check` and `cargo test` pass.
