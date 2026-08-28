# 095 - ranvil-cli: `set`, `set-area`

First real write commands, both thin builders on top of 094's `run_write`.

## Scope

- **`ranvil-cli set <x>,<y>,<z> <blockstate> [--dry-run] [--force]`** —
  `<blockstate>` parsed via `blueprint::BlockState`'s existing `FromStr`
  (ticket 035's paint-tool parser — reused, not reimplemented). Builds a
  `WorldEdit::new().set(pos, state)` and hands it to `run_write`.
- **`ranvil-cli set-area <x1>,<y1>,<z1> <x2>,<y2>,<z2> <blockstate>
  [--dry-run] [--force]`** — `WorldEdit::fill(bounds, state)`, the exact
  function `viewer::paint` already calls for the same job (ticket 035) —
  this command is that feature's headless twin, not a new fill
  implementation.

Both report through `EditReport`: blocks written, regions touched, (on
success) the backup directory `run_write` created. `text` is a short
summary line; `json` includes the full region list; `--dry-run`'s report is
shaped identically but labeled as a plan rather than a completed write, in
every format, so a script can't mistake a dry run's JSON for a real one by
skimming.

## Done when

- `ranvil-cli set <pos> <blockstate>` against a scratch/fixture save writes
  the block, and `ranvil-cli get <pos>` immediately after reads it back
  identically (including properties on a multi-property state like a
  stair).
- `ranvil-cli set-area` over a small box, followed by `ranvil-cli scan
  --block <that name>` (093), finds exactly the filled positions.
- `--dry-run` on both leaves the save's files byte-identical (checksum
  before/after in the test).
- A `set`/`set-area` into an ungenerated chunk refuses with the same
  `Status`-gate message 094's policy raises, exit 1.
- `cargo check` and `cargo test` pass.
