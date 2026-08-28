# 092 - ranvil-cli: `get-area` — the shared box-scan primitive

The load-bearing ticket the roadmap's ordering advice calls out: `get-area`
is `blueprint::extract_blueprint` under a CLI wrapper, and `097` (`copy`)
and `099` (`struct export`) are both meant to reuse whatever this ticket
builds rather than growing their own box-walking loop.

## Problem

`extract_blueprint` takes a `SelectionBounds` (`src/selection/mod.rs`),
which is a Bevy `Resource`-adjacent type living in a module that also pulls
in `bevy::prelude::*` for its ECS plugin half. `ranvil-cli` has no ECS world
and shouldn't need one to call a function that's already pure `IVec3` math
underneath (`extract.rs`'s own doc comment: "No Bevy beyond `IVec3`, and no
threading"). Confirm `SelectionBounds::from_corners`/`::iter_blocks`/
`::contains` — the parts `extract_blueprint` actually calls — don't
themselves reach into ECS (a quick read of `selection/mod.rs` says they
don't; the `Resource`/`Component` derives are the only Bevy-specific parts,
and a derive costs nothing to leave in place unused). If that holds,
`ranvil-cli` constructs a `SelectionBounds` directly with no plugin, no
`App`, no `Commands` — exactly the boundary `RANVIL_CLI_ROADMAP.md`'s
architecture section already commits to.

## Scope

`ranvil-cli get-area <x1>,<y1>,<z1> <x2>,<y2>,<z2>` — builds a
`SelectionBounds::from_corners`, calls `extract_blueprint(bounds,
&region_cache, &ExtractProgress::default())` synchronously (no task pool —
that machinery exists in `blueprint/mod.rs` to keep the *viewer's* frame
from freezing; a CLI process blocking on I/O is exactly what it should do),
and formats the resulting `Blueprint`.

Output is the palette-plus-index shape the roadmap commits to, not one line
per block:
- `json`: `{"origin": [x,y,z], "size": [x,y,z], "palette": ["minecraft:...",
  ...], "blocks": [0, 0, 1, ...]}` — `blocks` indexes into `palette`, row-
  major in the same `(x, y, z)` order `Blueprint`'s own dense array already
  uses, documented rather than reinvented.
- `text`: a summary line (size, block count, palette size) plus the palette
  listing — the full index array is not dumped to a terminal; use
  `--format json` for that.
- `compact`: the summary line alone, no palette listing — this is the form
  an agent asks for when it only needs "how big / how many distinct blocks"
  before deciding whether to pull the full `json`.

A selection reaching outside generated chunks is not refused outright —
`extract_blueprint` already tolerates unreadable columns by treating them as
air and reporting `failed_columns`; surface that count in every format
rather than hiding it, since a caller building on top of this needs to know
its palette might read "mostly air" for the wrong reason.

`MAX_BLOCKS` (`blueprint::extract`'s existing cap) applies unchanged — a
selection over the limit is `CliError::Usage` naming the cap and the
requested volume, checked before extraction starts.

## Done when

- `ranvil-cli get-area` over a small known box (a single chunk corner with a
  few distinct blocks) matches `get`'s output for each corner individually.
- A selection with unreadable columns reports `failed_columns` in every
  format rather than silently going missing.
- A selection over `MAX_BLOCKS` exits 2 before touching the region cache.
- `cargo check` and `cargo test` pass.
