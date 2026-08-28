# 090 - ranvil-cli: `heightmap`

## Scope

`ranvil-cli heightmap <cx>,<cz> [--kind world-surface|motion-blocking|
motion-blocking-no-leaves|ocean-floor]` (default `world-surface`) — the full
16×16 grid for one chunk, wrapping `ranvil::heightmap::read_heightmap` and
`HeightmapKind`'s existing variants directly (no new decode logic; this
ticket is output shaping only).

- `text`/`compact`: a 16-row ASCII grid, each cell the height (or a fixed-
  width placeholder for "no value" — a heightmap entry can be absent below
  the world's lowest possible surface). `compact` drops row/column headers
  `text` includes.
- `json`: `{"chunk": [cx, cz], "kind": "...", "heights": [[...16 values per
  row...], ...16 rows...]}`, row-major matching `heightmap::column_index`'s
  own `(dx, dz)` ordering so a caller cross-referencing `get`/`get-area`
  output doesn't have to guess the axis order.

A chunk with no heightmap of the requested kind (089's `Status` isn't
`"minecraft:full"` yet, or the kind was stripped) is a `CliError::Data`
naming which kind and chunk, not a grid of zeroes — a wrong-looking zero is
worse than a stated absence for anything downstream automating on this.

## Done when

- All four `--kind` values round-trip against a real chunk and visibly
  differ from each other on terrain with trees/leaves nearby (that's the
  whole reason `motion-blocking` and `motion-blocking-no-leaves` are
  separate maps).
- `--format json`'s row/column order matches `heightmap::column_index`
  exactly — a unit test builds a small synthetic `ColumnHeights`, prints it,
  and checks a known `(dx, dz)` lands at the expected `heights[dz][dx]` (or
  documents the order chosen if it's the other way — pick one and pin it in
  the test, since this is exactly the kind of axis convention 019 warns
  five later tickets would each decide differently if it weren't fixed
  once).
- `cargo check` and `cargo test` pass.
