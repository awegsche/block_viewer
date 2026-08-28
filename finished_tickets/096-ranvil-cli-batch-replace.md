# 096 - ranvil-cli: `set-batch`, `replace`

Two more write commands on 094's substrate; both assemble a single
`WorldEdit` with many entries rather than issuing 095's `set` in a loop, so
they get 094's plan-before-apply all-or-nothing guarantee across the whole
batch, not per line.

## Scope

- **`ranvil-cli set-batch <file|-> [--dry-run] [--force]`** — reads
  `x,y,z blockstate` lines (same coordinate and block-state syntax as
  every other command; blank lines and `#`-prefixed lines ignored) from a
  file path or, given `-`, stdin — the shape an agent's own generated diff
  arrives in without needing a temp file. Every line is parsed *before* any
  edit is attempted (a malformed line anywhere in the batch is
  `CliError::Usage`, reported with its line number, and nothing is
  written) — parse-then-apply, matching the "plan every region before
  applying any" discipline `edit::route` already holds writers to.
  Duplicate positions in the batch: last write wins, same as calling
  `WorldEdit::set` twice on one `WorldEdit` already behaves, not a rejected
  batch — a caller regenerating a batch that includes a correction doesn't
  need to dedupe first.
- **`ranvil-cli replace <x1,y1,z1> <x2,y2,z2> --from <name> --to
  <blockstate> [--dry-run] [--force]`** — one pass: extract the box (092's
  `get-area` primitive), find every position whose current block's `name()`
  matches `--from` (properties ignored on the match side, same convention
  093's `scan` uses — replacing every orientation of a block is the common
  case), build one `WorldEdit` setting each matched position to `--to`, run
  it through `run_write`. Reports how many positions matched/were written,
  distinctly from `set-area`'s "every position in the box" count.

Both share `set`/`set-area`'s output shape (`EditReport` plus a
batch-specific count: lines applied for `set-batch`, positions
matched/replaced for `replace`).

## Done when

- `set-batch` against a small fixture file writes every line correctly and
  reads back via `get`; a batch with one malformed line writes nothing and
  reports the offending line number.
- `set-batch -` reads from piped stdin.
- `replace` over a box with a mix of blocks changes only the matching
  positions, verified against `get-area`'s palette before/after.
- `cargo check` and `cargo test` pass.
