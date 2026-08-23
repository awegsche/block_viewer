# 081 - warn once per process for missing resources, not once per chunk

## Report

User: "there is a lot of logging happening. please suppress the 'failed to
load resource' kind of logs."

## Root cause

The console noise is not Bevy's — this app never routes anything through
`AssetServer` (see `sky::bodies`' module docs), so every line comes from the
crate's own `println!`s. The spam is one specific family: the
missing-resource warnings that fall back to a default.

Each of them dedupes against a `HashSet` created **fresh inside the function
that builds a per-chunk table**:

- `world::atlas::build_block_uv_table` — `let mut warned = HashSet::new()`
- `world::tint::build_block_tint_table` — same
- `world::tint::build_biome_tint_table` — same
- `world::decode::decode_chunk` — `warned_missing_biomes`

All four tables are rebuilt **per background chunk task**
(`chunk_pipeline.rs:826-828` calls all three table builders; `decode_chunk`
is per chunk by definition), because block/biome ids are interned lazily as
chunks stream in and there is no fixed set to resolve up front. So "warn
once" meant *once per chunk*: streaming a few hundred chunks reprinted

```
block_viewer: no texture mapping for 'minecraft:<name>' — using fallback checker
block_viewer: unknown biome 'minecraft:<name>' — falling back to plains tint values
block_viewer: section has no usable biome data — defaulting to minecraft:plains
block_viewer: 'minecraft:<name>' looks like it should be tinted ...
```

a few hundred times each. `blueprint::mesh::resolve_palette` has the same
per-call sets, so re-meshing a blueprint reprints them too.

Two genuine "failed to load" lines repeat for a different reason:
`region_cache` re-logs a failing region every time its 10s retry cooldown
expires (a permanently unreadable `.mca` therefore prints forever), and
`chunk_pipeline` re-logs an undecodable chunk every time it is re-streamed.

## Fix

New `world::warn` module with a `WarnLedger` — the same "keys already warned
about" set, but living in a `static` instead of a local, so it outlives the
task that reads it. `WarnLedger::new()` is a `const fn` and the ledger is
still passed by reference, so it is a shared value rather than a hidden
global: tests hold their own.

Ledgers, each keyed by the thing that is missing so distinct gaps stay
enumerable:

| ledger | key | declared in |
| --- | --- | --- |
| `MISSING_TEXTURE` | block name | `world::atlas` |
| `MISSING_TINT` | block name / overlay stem | `world::tint` |
| `UNKNOWN_BIOME` | biome name | `world::tint` |
| `MISSING_BIOME_DATA` | `"biomes"` (one line per run) | `world::decode` |
| `UNDECODABLE_CHUNK` | chunk coord | `chunk_pipeline` |
| `UNREADABLE_REGION` | region coord | `region_cache` |

The retry behaviour of the last two is unchanged — only the console line is
deduped. `region_cache` still retries after the cooldown; the chunk pipeline
still returns `None`.

Deliberately left alone: `streaming`'s "camera entered chunk" line (already
suppressed when the diff is empty, and not a load failure), the one-time
startup lines (atlas packed, save loaded, catalogues loaded), and the
city/edit action feedback — those are per-user-action, not per-chunk.

## Verification

- `cargo check`, `cargo test --lib`.
- Ledger unit tests: repeat keys warn once, distinct keys each warn, a fresh
  ledger is independent.
- The existing `unmapped_name_falls_back_to_the_checker_and_warns_once` and
  its tint sibling now assert against a local `WarnLedger`.
- Manual: does a full streaming session's console stay readable — noted in
  `todo.md`, needs a human watching the window.
