# 001 - `block_states` bit width is wrong for small palettes (upstream, in `../ranvil`)

## Status
Open

## Depends on
Nothing.

## Where
`../ranvil/src/chunkregion.rs`, `ChunkRegion::get_block` (line ~119).

This is an upstream fix in the sibling `ranvil` crate, tracked here because
it blocks every rendering ticket in this repo (`CLAUDE.md`: "changes may
require editing those repos too"). Consider opening the matching ticket in
`../ranvil/tickets/` and doing the work there.

## Problem

`get_block` derives the bit width of a packed block-state index purely from
the palette size:

```rust
let bit_size = (palette.len() as u32 - 1).ilog2() + 1;
```

Minecraft enforces a **minimum of 4 bits per entry** for `block_states`
(biome containers use a minimum of 1 — hence the confusion). For any section
whose palette holds 2..=8 entries, the formula yields 1-3 bits where the file
actually uses 4, so every index is read at the wrong offset and the decoded
blocks are garbage.

## Evidence

Probed against the real save `nbt_test` (DataVersion 4438), region chunk
list index 0 (`xPos=-32, zPos=-32`). `data.len()` tells us the true width,
since a section always packs 4096 indices at `64 / bits` per long:

| section | palette | actual `data.len()` | ranvil's bits (→ len) | true bits (→ len) |
|---------|---------|---------------------|------------------------|-------------------|
| Y=-4    | 7       | **256**             | 3 (→ 196) ❌           | 4 (→ 256) ✅      |
| Y=-3    | 9       | 256                 | 4 (→ 256) ✅           | 4 (→ 256) ✅      |
| Y=-1    | 15      | 256                 | 4 (→ 256) ✅           | 4 (→ 256) ✅      |
| Y=0     | 17      | 342                 | 5 (→ 342) ✅           | 5 (→ 342) ✅      |

The Y=-4 row is the bug: bedrock/deepslate/tuff layers decode incorrectly
today. Palettes of 9+ happen to agree with the formula, which is why this
wasn't caught by the existing fixtures — the fixture palettes are large
enough to mask it.

## Suggested fix

- Clamp: `let bit_size = ((palette.len() as u32 - 1).ilog2() + 1).max(4);`
- Add a regression fixture with a small palette (2-8 entries, e.g. 3) whose
  packed `data` is written at 4 bits/entry, and assert a known coordinate
  resolves to the expected block. `../ranvil/examples/gen_fixtures.rs` is
  where fixtures are generated — make sure the generator itself writes the
  4-bit-minimum encoding, or the test will pass against a wrong-but-matching
  file.

## Also worth fixing while in there (same function)

`get_block` indexes the `sections` list positionally
(`sections.get(section_index)`) assuming list index `i` == section
`Y = i - 4`. That holds for the probed save (24 sections, `Y` running -4..19,
no gaps), but it is an assumption about file layout, not a guarantee — chunks
can carry lighting-only sentinel sections at `Y=-5`/`Y=20`, which would shift
every lookup by one. Prefer selecting the section whose `Y` byte equals
`y.div_euclid(16)`. See ticket 002, which does exactly this on our side.

## Done when

- A section with a 2..=8 entry palette decodes to the correct blocks.
- Regression test covering a small palette passes in `../ranvil`.
- `cargo test` in `../ranvil` and `cargo build` here are both clean.
