# 023 - Structure NBT writer: a Blueprint to a vanilla `.nbt` file

## Status
Open

## Depends on
022 (`Blueprint` — this ticket serializes it).

## Goal

Write a `Blueprint` as a gzipped vanilla **structure** file: the format
`/structure` blocks save and load, so a blueprint captured here can be
placed back into a world by the game itself. Path in, `io::Result` out — the
filename comes from 024.

## Why this format

Picked over the alternatives:

- **Sponge schematic (`.schem`)** — what WorldEdit and Litematica read, so
  broader third-party support. But it needs varint-encoded block data, a
  palette *map* (name → index rather than a list), and offset/metadata
  conventions that differ between v2 and v3. More format code, more to get
  wrong, and the tooling isn't needed to prove the pipeline works. A good
  follow-up ticket once this one has established the writer.
- **A JSON/RON format of our own** — trivial to write and diff, useless
  everywhere else. Only worth it if blueprints were purely for this app.

The deciding factor is that `rnbt` already has both halves — `write_nbt` and
the `NbtField::new_*` constructors — so the structure format costs almost
nothing beyond assembling the right tags.

## The format

Root is an unnamed compound:

```text
DataVersion : TAG_Int      -- from Blueprint::data_version
size        : TAG_List of TAG_Int, 3 entries [x, y, z]
palette     : TAG_List of TAG_Compound
                Name       : TAG_String   ("minecraft:oak_stairs")
                Properties : TAG_Compound (optional; all values TAG_String)
blocks      : TAG_List of TAG_Compound
                state : TAG_Int                        (index into palette)
                pos   : TAG_List of TAG_Int, 3 entries (x, y, z, 0-based,
                                                        relative to the
                                                        structure's corner)
                nbt   : TAG_Compound (optional; block-entity data — omitted,
                                      see 022's out-of-scope)
entities    : TAG_List (empty)
```

The whole thing is **gzip**-compressed. `flate2` is already in `Cargo.lock`
(pulled in by `ranvil` for region decompression), so adding it as a direct
dependency of `block_viewer` costs no new download or compile —
`flate2::write::GzEncoder` around the file handle, `write_nbt` into that.

Details that are easy to get wrong:

- **Air is written.** Unlike some formats, the vanilla structure format's
  `blocks` list is sparse-by-omission-capable but structure blocks expect
  air to be present for the "replace the volume" behaviour to work. Emit
  every block including `minecraft:air`, and note the decision in a comment
  — dropping air is a legitimate size optimisation but changes placement
  semantics, so it's a deliberate choice, not an oversight.
- **`pos` is 0-based and relative** to the selection's min corner, not world
  coordinates. `Blueprint::origin` is *not* written.
- **All property values are strings** in NBT, including numeric-looking ones
  (`"level": "3"`) and booleans (`"waterlogged": "false"`). 022 already
  keeps them as `String`s; don't be clever and re-type them here.
- **Property order** — emit 022's sorted order, so the same blueprint
  written twice is byte-identical. Makes the round-trip test meaningful and
  makes files diffable.
- **Empty `Properties`** — omit the tag entirely rather than writing an
  empty compound. Vanilla does, and some parsers dislike the empty form.

### Size

The format has no size cap; **structure blocks do** (48x48x48). A blueprint
larger than that is a valid file the game can't load through a structure
block. Don't refuse to write it — 021 already warns on volume — but log the
limit once when exceeding it, so the user finds out here rather than
in-game.

## rnbt caveat

`rnbt`'s writer is far less exercised than its reader — nothing in this repo
has ever called `write_nbt`. Assume it has bugs (empty lists, list-of-
compound headers, and the root-tag name are the usual suspects) and expect
this ticket to include a fix in `../rnbt` (sibling crate, see `CLAUDE.md` —
changes there may need committing separately). The round-trip test below is
what finds them, so write it first.

## Tests

- **Round trip.** Build a small `Blueprint` (2x2x2, two palette entries, one
  with properties), write it to a `Vec<u8>` through the gzip encoder, read
  it back with `flate2::read::GzDecoder` + `rnbt::read_nbt`, and assert every
  field: `DataVersion`, `size`, palette length/names/properties, `blocks`
  length, and a specific `(state, pos)` pair.
- Determinism: writing the same blueprint twice gives identical bytes.
- A blueprint whose palette has an entry with no properties omits the
  `Properties` tag rather than writing an empty compound.
- `pos` is relative: a blueprint with a nonzero `origin` still writes
  `pos: [0,0,0]` for its min-corner block.
- A 1x1x1 blueprint (the degenerate case that shakes out empty-list and
  single-entry handling).

## Done when

- A written file re-reads as an equal blueprint.
- `cargo test` passes, including whatever `../rnbt` needed.
- `todo.md` gets a manual check: this is the one ticket whose real
  verification is *in Minecraft*. Export a small, recognisable structure,
  drop the file into a world's `generated/minecraft/structures/` folder, and
  load it with a structure block. Confirm the size matches, the blocks are
  in the right places (not mirrored or rotated — the `bevy.z = -mc.z` flip
  from 019 is exactly the kind of thing that shows up as a mirrored build
  and nowhere else), and that stairs/logs/doors kept their orientation.
