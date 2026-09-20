# 135 - model-exporter: `export [<name>...]`

Design: `MODEL_EXPORTER_ROADMAP.md`. Depends on 131, 132 (not on 133/134
— a slot written by hand exports fine). The command the tool is named
for: every registered model's world box → its `.nbt`, in one call, with no
corners typed and no selection dragged.

## Scope

- **`export [<name>...] [--force] [--dry-run]`** in
  `src/model_exporter/export.rs`. No names → every slot, in registry
  order; names → those (an unknown one is `CliError::Usage` before any
  export runs).
- Per slot, in order:
  1. `extract_blueprint(slot.bounds(), &cache, &progress)` — the exact
     call `ranvil_cli::structure::export` makes, with the `RegionCache`
     sized by the same `region_span` arithmetic. Factor that read into a
     `pub fn` in `ranvil_cli::structure` (or `blueprint::extract`) taking
     `(&SaveMeta, SelectionBounds)` so this ticket calls it rather than
     copying the cache-sizing code. One cache for the whole run, sized to
     the union of every slot's regions, rather than one per slot.
  2. `failed_columns > 0` → status `failed` for this slot (ungenerated
     chunks inside the box), file untouched.
  3. `blueprint::catalogue::run_checks(&blueprint, splat(STRUCTURE_BLOCK_MAX_SIZE))`
     — the air-only check failing means nobody has built here yet →
     status `empty`, file untouched. (Because `--below 1` puts a dirt
     layer in `y=0`, "empty" is also true when the *only* non-air blocks
     are the untouched ground layer: compare against the flat world's
     ground, i.e. every non-air block is at `y < ground_y + 1`. Spell the
     rule out in the code comment; a registered slot nobody built in must
     never overwrite a real `.nbt`.)
  4. If `slot.out_path()` exists, `read_structure_file` it and compare
     `size` plus every position's state (the comparison `ranvil_cli::
     structure::diff` already does — reuse its walk); identical → status
     `unchanged`, file untouched (keeps the file's mtime and git status
     clean); different → `updated`; no file → `new`.
  5. `write_structure_file` (unless `--dry-run`), creating the parent
     directory if needed.
- **Lock gate**: if the save `is_locked()` and not `--force`, the whole
  command refuses up front (`CliError::Data`, "models world is open in
  Minecraft — save & quit, or pass --force to read what's on disk
  anyway"). A region that Minecraft hasn't flushed is the one way to
  export something other than what the human sees; the gate is the same
  one `run_write` applies to writes, applied here to a read for that
  reason.
- **Output**: one line per slot in `text`/`compact`
  (`house02  updated  12x10x12  38 states -> assets/city/blueprints/house02.nbt`),
  and a trailer `3 exported (1 new, 2 updated), 4 unchanged, 1 empty, 0
  failed`. `json`: an array of per-slot objects (`name`, `status`, `out`,
  `size`, `blocks`, `palette_size`, `failed_columns`) plus the counts.
  Exit code: `0` if nothing `failed`, `1` otherwise — like `struct
  validate`, a partial failure is a printable answer, not an error
  envelope, so the other slots' results are still reported.

## Done when

- Against a fixture save with two slots registered — one containing
  planted blocks, one untouched ground — `export` writes the first's
  `.nbt` (`status: new`), reports the second as `empty` and writes
  nothing for it; a second run reports the first as `unchanged` with the
  file's bytes and mtime identical; `struct set` on the world (or a
  `set` through the fixture) then makes it `updated`.
- `struct import` of the exported file back at `slot.origin` on a copy of
  the fixture reproduces the box exactly (the 099 round-trip, now driven
  from a `.ron`).
- `--dry-run` writes no file, prints the same statuses.
- `todo.md` gains the manual check: build something in a `new` slot in
  Minecraft, save & quit, `model-exporter export <name>`, open the
  citybuilder (or `ranvil-cli struct info`) and confirm the model is the
  build, ring not included, foundation layer at `y=0`.
- `cargo check` and `cargo test --lib` pass.
