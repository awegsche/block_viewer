# 101 - ranvil-cli: `struct resize`

The command the roadmap's second motivating job names directly: "adding
ground and underground layers, adding headroom above" to an existing
building model.

## Scope

`ranvil-cli struct resize <file.nbt> --out <file2.nbt> [--pad-y-top N]
[--pad-y-bottom N] [--pad-x-neg N] [--pad-x-pos N] [--pad-z-neg N]
[--pad-z-pos N] [--fill <blockstate>] [--force]` — six independent padding
amounts, one per face of the bounding box, each defaulting to `0`. Positive
values add margin (filled with `--fill`, default air); **negative values
crop** that face instead — resize is one command for both directions rather
than a `pad` command plus a separate `crop` command, since both are "change
where one face of the box sits" and a caller (especially an agent) reaching
for "make this 2 blocks shorter" shouldn't have to know it's a different
command from "make this 2 blocks taller".

Mechanics: compute the new size from the six pads, allocate a new dense
block array of that size pre-filled with `--fill`, then copy every position
from the source `Blueprint` that lands inside the new bounds at its shifted
coordinate (a crop simply never copies the part that falls outside). A crop
that removes every block on some axis (`--pad-y-top` more negative than the
structure is tall) is `CliError::Usage`, refused before writing anything,
naming the resulting non-positive dimension.

`--pad-y-bottom`/`--pad-x-neg`/`--pad-z-neg` shift every remaining block's
coordinate by the pad amount (a new bottom layer at `y=0` pushes the old
`y=0` to `y=N`); `ground_level` in a building's own `.ron` (the citybuilder
field this directly serves — `city::definition`'s `ground_level`,
"which 0-indexed Y layer of the blueprint is its own ground surface") is
**not** touched by this command since structure files carry no such field
themselves; note in `--help` and in the roadmap-facing docs that
`--pad-y-bottom` on a building already in `assets/city/buildings` likely
needs its companion `.ron`'s `ground_level` bumped by the same amount by
hand, same as any other structural edit to a building's blueprint would.

## Done when

- `--pad-y-top 3` on a known structure produces a file 3 taller, with the
  top 3 layers filled per `--fill` and everything below identical to the
  source (`struct diff`, once 102 lands, against a manually-shifted
  comparison; until then, spot-checked with `struct get`).
- `--pad-y-bottom 2` shifts every existing block up by 2 and fills the new
  bottom 2 layers.
- A negative pad crops correctly, including cropping away part of a
  non-air region (data loss is allowed — this is a deliberate crop, not a
  refusal) but refused outright when a whole axis would collapse to zero
  or less.
- Combining pads on multiple faces in one call produces the same result as
  applying them one at a time in any order (an order-independence unit
  test — the six pads must not interact).
- `cargo check` and `cargo test` pass.
