# 089 - ranvil-cli: `chunk`, `chunks`

The first commands that decode a chunk. Both read through
`ranvil::chunkregion::ChunkRegion` (`get_chunk_or_load`) against a
`RegionCache` built the same way `citybuilder`'s startup builds one, just
without the streaming/unload machinery around it — a one-shot cache with
enough capacity for whatever the command touches, dropped at the end of the
process.

## Scope

- **`ranvil-cli chunk <cx>,<cz> [--print]`** — one chunk:
  - `Status` (the chunk's own `Status` NBT field — `"minecraft:full"` etc.,
    the same field ticket 031's `EditPolicy` gates writes on).
  - DataVersion.
  - Section Y-range present (min/max `Y` among `sections`, not assumed
    `-4..19` — a chunk can have fewer sections than the world's configured
    height, which is exactly what `chunks`' "how much is generated" survey
    below needs to not get wrong).
  - `InhabitedTime`.
  - `block_entities` count, entity count (if `entities` lives in this
    chunk's NBT rather than a separate entities region — check both; older
    saves keep them together, `1.17+` splits them into `entities/`).
  - Biome list — the *distinct* biome names across the chunk's `biomes`
    palettes, not a per-block breakdown (that's what `get`/`get-area`'s
    block-level data covers; a chunk-level command should stay chunk-level).
  - A heightmap peek: min/max/average `WORLD_SURFACE` height across the
    column, from `ranvil::heightmap::read_heightmap` — cheap, and answers
    "is this chunk flat" without a full `heightmap` ticket-090 call.
  - `--print` is accepted as the roadmap's example spelled it
    (`chunk 0,0 --print --format json`) but changes nothing about the
    output — a `chunk` command *always* prints its result; `--print` is
    kept as a no-op accepted flag only so the exact invocation the roadmap
    and any earlier planning discussion used keeps working verbatim. Note
    this plainly in the flag's `--help` text so it doesn't look like a
    silently-broken option.
- **`ranvil-cli chunks <cx1>,<cz1> <cx2>,<cz2>`** — bulk survey over a
  chunk-coordinate rectangle (inclusive both ends, matching 019's inclusive-
  bounds convention). Walks every chunk in the range through the same
  `RegionCache`, without decoding sections beyond what "generated y/n" and
  the section-range/entity-count fields need. Reports: total chunks in
  range, counts grouped by `Status`, chunks with no NBT at all (ungenerated
  — `MCLoadError::ChunkNotFound` from the region's sector table, not an
  error for this command, just a tally), aggregate block-entity/entity
  counts, and (in `text`/`compact`) the list of ungenerated chunk
  coordinates capped at a sane count (`--format json` includes the full
  list). A large range (hundreds of chunks) should still finish in seconds —
  this reuses `chunk`'s per-chunk read, it does not additionally decode
  block_states.

## Done when

- `ranvil-cli chunk 0,0` and `ranvil-cli chunk 0,0 --format json` run
  against a real save's spawn chunk and an out-of-range chunk (reports "not
  generated" as a `CliError::Data`, exit 1 — a missing chunk is a real
  answer about the save, not a bad request).
- `ranvil-cli chunks -2,-2 2,2` finishes promptly and its status counts sum
  to the range's total chunk count.
- `cargo check` and `cargo test` pass; a fixture chunk NBT (already used by
  `blueprint::extract`'s own tests — reuse rather than hand-build a new one)
  covers `chunk`'s field extraction.
