# 023 - Structure NBT writer: a Blueprint to a vanilla `.nbt` file

## Status
Done

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

---

## Resolution

Landed as `src/blueprint/structure.rs`, one module: the format, the writer,
and the round trips. Nothing else in the app calls it yet — ticket 024 is
what supplies a filename, and `poll_extraction` still drops the blueprint it
just built.

### What was built

- `write_structure_file(&Path, &Blueprint) -> io::Result<()>` — the entry
  point 024 will call.
- `write_structure(&mut W, &Blueprint) -> io::Result<()>` — the same thing
  into any sink, which is what the tests write to a `Vec<u8>` through.
- `STRUCTURE_BLOCK_MAX_SIZE` (48), pub because ticket 021's panel had its own
  copy of that number.

Every format decision the ticket listed went the way it asked: air is
written, `pos` is 0-based and relative (`origin` is not written at all),
property values stay strings in 022's sorted order, and an empty
`Properties` is omitted rather than written empty. The oversize case logs
one line and writes the file anyway.

### The rnbt caveat didn't materialise

`../rnbt` needed no changes. The ticket assumed its writer was unexercised,
but `rnbt/tests/nbt_spec_tests.rs` is a 107-test sweep of the whole spec —
every tag type, empty and single-element collections, binary spot-checks —
plus 9 round trips in `simple_tests.rs`. All of it passes. The round-trip
test here was still written first, as the ticket asked, and passed on its
first run; the usual suspects it names (empty lists, list-of-compound
headers, the root tag name) are all handled correctly.

### One departure: the `blocks` list is streamed, not built

The ticket's plan is "assemble the right tags and hand them to `write_nbt`",
which is what everything except `blocks` does. `blocks` can't: one block
costs ~250 bytes as an `NbtField` tree (a compound of two fields, each with a
heap-allocated name, plus a three-element `Vec<i32>` for `pos`) against 2
bytes in `Blueprint::blocks`. At `MAX_BLOCKS` — a size the extractor will
hand over happily, and the panel's export button allows — that tree is
several GB, i.e. an out-of-memory abort on a path a user can reach by
clicking. Ticket 008's "nothing user-triggerable panics" rule made that not
worth shipping for the sake of a simpler writer.

So `write_blocks` emits the list header itself and writes one entry at a
time, building and dropping each entry's fields as it goes; peak memory is
the blueprint's own. The price is six bytes of wire format written by hand
(the `TAG_List` header and the per-entry `TAG_End`s) plus the unnamed root
compound's three — the only NBT encoding in this repo that doesn't go
through `rnbt`, and all of it pinned down by round trips that read back
through `rnbt`'s own reader.

The alternative considered and rejected was widening `rnbt`'s API with a
streaming writer: its `TagWrite` trait is already the right shape but its
module is private, and exporting it is a bigger change to a shared crate than
this ticket needs.

### Measured

`cargo test measure_large_write -- --nocapture`:

```
wrote 2097152 blocks in 1.2505585s — 4857510 bytes gzipped
```

The same 2,097,152 blocks ticket 022 measured its extraction against (~175
ms), so the **write is the expensive half by a factor of seven**, and the
whole export of that volume is ~1.4 s. Extrapolated to the 16,000,000-block
`MAX_BLOCKS` cap: ~10 s and ~37 MB, with memory flat.

That finally settles what 022 left open: ticket 021's `VOLUME_WARN` of
1,000,000 is *not* an order of magnitude off once the write is counted — a
million blocks is most of a second — so it stays, and its doc comment now
cites both measurements instead of promising one.

### Also touched

- `ui/selection_panel.rs`'s `STRUCTURE_BLOCK_LIMIT` is now
  `STRUCTURE_BLOCK_MAX_SIZE.pow(3)` rather than a second `48 * 48 * 48`, the
  same reasoning that made `VOLUME_CAP` literally `MAX_BLOCKS`. The hover
  text interpolates the constant too.
- `flate2` is now a direct dependency. It was already in `Cargo.lock` via
  `ranvil`, so naming it added no download and no compile.

### Still out of scope

Block entities, entities and biomes — the `entities` tag is written empty
(it isn't optional) and there's no `blocks[].nbt` on any entry, exactly as
022 left them. The Sponge `.schem` writer the ticket sketches as a follow-up
is untouched; this module's split (tag assembly for the small parts,
streaming for the block array) is the shape that one would reuse.

12 new tests; `cargo test` is 175 passing.
