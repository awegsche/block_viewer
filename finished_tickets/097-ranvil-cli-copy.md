# 097 - ranvil-cli: `copy`

Extract a box and re-apply it translated elsewhere in the same save — the
one command that visibly composes 092's read primitive with 094's write
substrate rather than adding new box-walking logic.

## Scope

`ranvil-cli copy <x1>,<y1>,<z1> <x2>,<y2>,<z2> --to <x>,<y>,<z> [--dry-run]
[--force]` — `--to` names the destination's minimum corner, matching how
the source box's own minimum corner is read (`normalized(a, b)`'s existing
convention from `selection`, reused rather than reinvented). Steps:

1. `blueprint::extract_blueprint` over the source box (092's exact call).
2. Build a `WorldEdit` writing the extracted `Blueprint`'s blocks at
   `dest = to + (block_pos - source_min)` for every non-air entry — air
   positions inside the source box are **not** written at the destination
   by default (copying a tree-shaped selection shouldn't punch an
   air-shaped hole through whatever already stands at the destination);
   `--include-air` opts into overwriting the destination with the source's
   air too, for the "I want an exact clone, blank spots included" case.
3. `run_write` as usual.

Source and destination boxes overlapping is allowed (a same-region nudge is
a legitimate use) — `extract_blueprint` finishes its read before `apply`
starts any write, so an overlap reads the *original* blocks throughout, not
a partially-copied one. State this explicitly in `--help` since it's the
kind of thing worth confirming rather than assuming.

Rotation is **not** a flag here — the roadmap places `struct rotate` (102)
as the one place rotation logic lives (`blueprint::rotate_blueprint`), and
`copy` composing with it (`struct export` the source, `struct rotate` it,
`struct import` at the destination) is the fallback if a rotated in-place
copy turns out to be wanted before this ticket grows one itself. Keeping
`copy` translation-only keeps this ticket small; note the fallback path in
`--help`'s long description so it isn't a dead end.

## Done when

- `ranvil-cli copy` on a small known structure (a few distinct blocks, one
  with properties) reproduces it exactly at the destination, verified by
  `get-area`'s palette matching at both locations.
- Overlapping source/destination reads the original blocks (a fixture with
  a checkerboard pattern shifted by one, confirming no self-clobbering).
- `--include-air` overwrites the destination's prior contents where the
  source was air; without it, those positions are untouched.
- `cargo check` and `cargo test` pass.
