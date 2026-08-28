# 099 - ranvil-cli: `struct export`, `struct import`

The bridge between a live save and a structure file — the headless form of
the viewer's selection-and-Export flow (tickets 019–024). Per the roadmap's
ordering note, this ticket's two halves sit at different tiers: `export`
needs only 092's read primitive, `import` needs 094's write session. If
that split matters for scheduling, ship them as two tickets; documented
together here because they're each a handful of lines around functions that
already exist.

## Scope

- **`ranvil-cli struct export <x1>,<y1>,<z1> <x2>,<y2>,<z2> --out
  <file.nbt> [--force]`** — 092's exact `extract_blueprint` call, then
  `blueprint::structure::write_structure_file` instead of (or alongside)
  `get-area`'s stdout formatting. Reports the same summary `struct info`
  would report on the freshly-written file (size, block count, palette
  size, `failed_columns`) — this is the CLI's answer to "extract this
  building and tell me about it" in one command, rather than `struct
  export` followed by a separate `struct info` call.
- **`ranvil-cli struct import <file.nbt> --at <x>,<y>,<z> [--rotate
  90|180|270] [--dry-run] [--force]`** — `read_structure_file`, then (if
  `--rotate` given) `rotate_blueprint` (102's function — this ticket calls
  it, doesn't reimplement it; if 102 hasn't landed yet, `--rotate` can ship
  as accepted-but-unimplemented returning a clear "not yet supported"
  `CliError::Usage` rather than blocking this ticket on 102, since `Deg0`
  needs no rotation logic at all), then a `WorldEdit` writing every
  position (including air, matching `city::commit`'s own
  `blueprint_edit` convention of writing air explicitly so a placement
  clears whatever terrain poked into the footprint — this is the same
  choice E4 made for the game, reused rather than re-litigated) offset by
  `--at`. Runs through 094's `run_write`.

`--at` names the structure's own `origin` (always `ZERO`) mapped to that
world position — position `p` inside the structure lands at `--at + p`, the
same convention `copy`'s `--to` uses for its source's minimum corner.

## Done when

- `struct export` over a box, followed by `struct import` of the resulting
  file back at the *same* coordinates on a copy of the fixture save,
  reproduces the original blocks exactly (round-trip test).
- `struct import --at <elsewhere>` places the structure at the new location
  only, leaving the original untouched.
- `struct import --rotate 90` (once 102 exists) matches what `struct
  rotate`'s own output file placed with no rotation would produce — i.e.
  the two paths agree.
- `--dry-run` on `struct import` leaves the destination save untouched.
- `cargo check` and `cargo test` pass.
