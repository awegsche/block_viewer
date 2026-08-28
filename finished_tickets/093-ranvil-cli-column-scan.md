# 093 - ranvil-cli: `column`, `scan`

Two more block-read commands, both built on 092's `get-area` primitive
rather than a fresh box walk.

## Scope

- **`ranvil-cli column <x>,<z> [--from <y>] [--to <y>]`** — a single-column
  `get-area` (`x1=x2=x`, `z1=z2=z`), defaulting `--from`/`--to` to the
  save's actual world Y bounds (`selection::WORLD_MIN_Y`/`WORLD_MAX_Y` —
  reuse, don't hardcode `-64`/`320`, which stopped being universal the
  moment 069's DataVersion-band work had to start caring about older
  worlds). Output is one row per Y, bottom to top or top to bottom
  (`--top-down` flag, default bottom-to-top matching how a player reads a
  cave profile): `y block_name[properties]`. `text`/`compact` collapse
  consecutive identical blocks into one range line (`-58..-1 minecraft:
  stone`) rather than 379 duplicate lines of stone — this is the one place
  in the block-read group where the "compact" format earns a genuinely
  different shape from "text", not just a shorter summary. `json` keeps the
  full per-Y array; a caller wanting the collapsed ranges in machine form
  gets them as a second field (`"ranges": [{"from": -58, "to": -1, "block":
  "minecraft:stone"}, ...]`) alongside the flat `"blocks"` array, not
  instead of it.
- **`ranvil-cli scan <x1,y1,z1> <x2,y2,z2> --block <name> [--limit N]`** —
  runs `get-area`'s extraction, then filters the resulting `Blueprint` for
  positions whose palette entry's `name()` matches `--block` (exact match on
  the namespaced name; properties are not part of the match — `scan --block
  minecraft:oak_door` finds every door regardless of open/closed/hinge).
  Reports absolute world positions (the `Blueprint`'s `origin`-relative
  indices translated back), capped at `--limit` (default a few hundred,
  overridable, never unbounded by default — a `scan` over a whole loaded
  region for `minecraft:air` would otherwise print millions of positions).
  `json` includes whether the result was capped (`"truncated": true/false`)
  so an agent knows to narrow the box rather than trust an incomplete list
  silently.

Both commands are read-only, no write-safety concerns, and share 092's
`MAX_BLOCKS` ceiling and `failed_columns` reporting unchanged.

## Done when

- `ranvil-cli column` over a real column shows the same blocks `get` reports
  at each of a handful of sampled Y values, and its range-collapsing merges
  runs of identical blocks correctly (unit test: a synthetic column with
  three distinct runs collapses to exactly three range lines).
- `ranvil-cli scan` finds a known block (plant a chest via `set` once 095
  lands, or use a fixture with a known block today) and respects `--limit`,
  setting `truncated` correctly at the boundary (limit exactly met vs.
  exceeded by one).
- `cargo check` and `cargo test` pass.
