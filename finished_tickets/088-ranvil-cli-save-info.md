# 088 - ranvil-cli: `info`, `regions`, `lock`

Save-level survey commands. All three read `SaveMeta` and its region
listing; none open a chunk.

## Scope

All three resolve `--save` via 087's save-resolution path and fail with a
`CliError::Usage` (exit 2) naming the save that couldn't be found — not a
`CliError::Data`, since an unresolvable save name is a bad argument, not a
read that failed against a real save.

- **`ranvil-cli lock`** — `SaveMeta::is_locked()`. Simplest of the three;
  build first as a second proof of the format/error contract before the
  other two's larger output shapes.
- **`ranvil-cli regions [--format grid|list|json]`** — the region files
  `resolve_region_dir` finds for this save: `(rx, rz)`, file size in bytes,
  chunk count (non-empty sector-table entries — `Region`'s own count, not a
  full chunk decode). `grid` renders `ranvil::save::GridView` as-is (the
  ASCII map the crate already builds); `list` is one line per region sorted
  by `(rx, rz)`; `json` is an array of objects. `--format grid` only makes
  sense as `text`-shaped output — reject `--format json --format-as grid`
  combinations that don't exist rather than silently picking one; `grid` is
  its own value of a `--layout` flag *within* `text`/`compact`, not a fourth
  `OutputFormat` variant, so it composes with 087's format contract instead
  of fighting it.
- **`ranvil-cli info`** — the save-level summary: name, path, `SaveMeta`'s
  DataVersion (read the same way ticket 069's compatibility bands do — reuse
  `edit::data_versions_compatible`'s notion of what a "version" even is
  rather than inventing a second reader), dimensions present (does
  `dimensions/minecraft/the_end` etc. exist, alongside the overworld
  `resolve_region_dir` already resolves), region count, locked. Level-level
  fields beyond DataVersion (seed, spawn point, gamerules) are **out of
  scope for this ticket** — `SaveMeta` doesn't parse `level.dat` today, and
  adding that is its own scoped ticket if a later job actually needs seed/
  spawn (nothing in `RANVIL_CLI_ROADMAP.md`'s three motivating jobs asks for
  it yet). `info`'s `text` output says as much rather than silently omitting
  the fields.

## Done when

- `ranvil-cli lock`, `ranvil-cli regions`, `ranvil-cli regions --format
  grid`, `ranvil-cli regions --format json`, and `ranvil-cli info` all run
  against a real save and a real `.minecraft/saves` instance.
- `ranvil-cli info --save does-not-exist` exits 2 and names the save it
  couldn't find, in both `text` and `json`.
- `cargo check` and `cargo test` pass.
