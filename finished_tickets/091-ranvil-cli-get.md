# 091 - ranvil-cli: `get`

The simplest block-level read, and the first command in the roadmap's own
motivating example (`ranvil-cli get 200,60,150`).

## Scope

`ranvil-cli get <x>,<y>,<z>` — resolves the block's chunk/region through the
`RegionCache` + `ChunkRegion::get_chunk_or_load` path 089 already
established, then calls `ChunkRegion::get_block(x_in_region, y, z_in_region)`
and formats the palette entry it returns.

`get_block` hands back a raw `&NbtField` (a `Name` + optional `Properties`
compound) rather than a `ranvil::BlockState` — parse it the same way
`palette_states`/`chunkregion`'s own internals do rather than re-deriving
NBT-to-BlockState decode logic here; if the crate has no standalone
"one palette entry to `BlockState`" function yet, extracting one from
`chunkregion`'s existing per-entry logic (used by `palette_states`) is
in scope for this ticket — it doesn't currently need to exist for anything
outside a full palette scan.

Output:
- `text`/`compact`: `minecraft:oak_stairs[facing=east,half=top]` — the same
  string `BlockState`'s `Display` already produces, so this is copy-pasteable
  straight into a `set` command.
- `json`: `{"pos": [x, y, z], "name": "minecraft:oak_stairs", "properties":
  {"facing": "east", "half": "top"}}`.

A position outside any loaded/generated chunk is `CliError::Data` naming the
chunk (`MCLoadError::ChunkNotFound`/`SectionNotFound` pass through, wrapped
with position context) — not a silent "air".

## Done when

- `ranvil-cli get <x,y,z>` against a real save matches what the viewer's
  block inspector (ticket 007) reports for the same coordinate, including a
  block with properties (a stair, a log, a door half).
- `ranvil-cli get` on a coordinate in an ungenerated chunk exits 1 with a
  clear message, in both `text` and `json`.
- `cargo check` and `cargo test` pass; a fixture chunk NBT (shared with 089)
  covers a no-properties block, a with-properties block, and a
  single-entry-palette section (`get_block`'s documented fast path).
