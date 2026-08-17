# 054 - Road cells and the piece catalogue

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 474
tests, up 9 from this ticket's own suite). See the Resolution.

## Part of

Roadmap group F (`tickets/CITYBUILDER_ROADMAP.md`): revises F1 ("the road
graph," ticket 053) and lands the pure-logic half of F3 ("auto-tiling the
pieces") ahead of it, because F3's piece *selection* has no dependency on
F3's rendering or F2's drag input — the same "prove the primitive before the
input/rendering that drives it" shape 046/053 already used.

## Problem

053 modeled a road as a set of single-block tiles with no shape at all —
fine for the adjacency graph, but roads were never going to render as
1x1-block dots. The design decision (from conversation, not the roadmap
text): a road is a **6x6-block cell**, laid out cross-section as 1 shoulder +
1 kerb + 2 road surface + 1 kerb + 1 shoulder, and — like buildings — gets a
real blueprint. Unlike buildings, one road "type" isn't one blueprint: it's
up to six, one per connection shape, because a road cell's neighbours decide
which piece it has to be.

This is a breaking change to 053's shipped API (`City::add_road`/`remove_road`/
`roads()` and `city::road`'s tile-space `Direction`/`connections_at`/
`reachable_from`), not an addition next to it — see the roadmap ticket's own
open question about corners: a straight piece rotated cannot stand in for a
corner, so the block-tile model (which had no piece concept to select) had
nothing to get right or wrong here to begin with. Two grids that could
disagree (block-tile roads and cell roads) is worse than one migration.

## Goal

- `state::City`'s road half moves from single-block tiles to 6x6 cells:
  `add_road_cell(cell)`/`remove_road_cell(cell)`/`road_cells()`/
  `is_road_cell(cell)`, in cell coordinates. Placing a cell marks all 36
  underlying block tiles `Occupant::Road` in the existing occupancy grid —
  the same all-or-nothing shape `place_building` already uses, so a road
  cell overlapping a building (or vice versa) is refused, never partial.
- `city::road`'s adjacency (`Direction`, `RoadConnections`, `connections_at`,
  `reachable_from`, `is_connected`) moves to cell space. The four cardinal
  offsets don't change value (still ±1) — they now mean "one cell," not "one
  block."
- A pure piece-selection function: `RoadConnections` -> `(RoadPieceKind,
  Rotation)`, where `RoadPieceKind` is `Isolated | DeadEnd | Straight |
  Corner | T | Cross`. Every piece is authored once at a canonical
  connection pattern and matched by rotating that pattern through
  `blueprint::rotate::Rotation`'s own clockwise convention (`Deg90` = north
  -> east) until it equals the actual connections — the same rotation
  semantics a placed blueprint already uses, so a selected piece and rotation
  are what `blueprint::rotate_blueprint` would actually need to reproduce the
  right shape once F3 calls it.
- A road piece catalogue (`city::road_catalogue`), loading a fixed set of
  `.nbt` files — `isolated.nbt`, `dead_end.nbt`, `straight.nbt`,
  `corner.nbt`, `t.nbt`, `cross.nbt` — from `assets/city/roads`, each
  validated to be exactly 6 blocks on `x` and `z` (the roadmap's own
  building-catalogue validation shape, narrowed to this module's own size
  rule instead of B4's `STRUCTURE_BLOCK_MAX_SIZE` ceiling).

## Out of scope

- **Art assets.** No real `.nbt` road pieces ship with this ticket — unlike
  039's `house01.nbt`, these need an actual structure-block export from
  Minecraft, which isn't something to fabricate. `load_road_catalogue_dir`
  against a missing/empty `assets/city/roads` is exercised (an empty
  catalogue, not an error, the same contract 039 gives a missing
  `assets/city/blueprints`); the loader's own tests build synthetic
  structures the same way `blueprint::catalogue`'s tests do. Authoring the
  six real pieces is a to-do, not this ticket.
- **F2, drag-to-build.** Nothing here takes player input or places a cell in
  response to it.
- **F3's rendering.** Nothing here spawns a mesh, calls
  `blueprint::rotate_blueprint`, or writes a road's blocks into the world.
  `select_piece` is the answer F3's rendering will call; wiring it to an
  actual mesh/write is F3's own job.
- **F4's connectivity queries.** Unaffected in shape by this ticket beyond
  the coordinate-space change — still not built.

## Done when

- `cargo build`/`cargo test --lib` clean.
- `state::City` tests: placing a road cell marks exactly its 36 tiles,
  refuses (all-or-nothing) when any of them collide with a building or
  another road cell, `remove_road_cell` frees them, re-adding the same cell
  is a no-op.
- `city::road` tests, re-based on cells: direction offsets, `connections_at`
  over 0/1/2/3/4-neighbour cells (including a building neighbour not
  counting), `reachable_from`/`is_connected` — same coverage 053 had, in cell
  space.
- `select_piece` tests: every one of `RoadConnections`' 16 possible
  neighbour combinations resolves to the right `RoadPieceKind`, and for the
  five oriented kinds, the returned `Rotation` actually reproduces the
  original connections when the canonical pattern is rotated by it — the
  round-trip property, not just an example per kind. `Cross`/`Isolated`
  always come back `Deg0`.
- `city::road_catalogue` tests: a missing directory is an empty catalogue,
  not an error; a synthetic 6x6xN structure loads under its kind; a wrong
  size (not 6 on `x` or `z`) is skipped and reported.
- `city::persistence`'s `CitySave` schema updates (`roads` -> `road_cells`,
  `CURRENT_VERSION` bumped) with a round-trip test, and a version-mismatch
  test proving an old-format file is refused rather than silently
  reinterpreted as cell coordinates.
- No manual verification needed — this ticket adds no rendering and no
  input, same as 053.

## Resolution

Landed as scoped, across four files.

**`state.rs`**: `ROAD_CELL_SIZE = 6` and `road_cell_tiles(cell)` (the
cell-space counterpart of `footprint_tiles`), plus `add_road_cell`/
`remove_road_cell`/`road_cells()`/`is_road_cell` replacing `add_road`/
`remove_road`/`roads()` outright — not added alongside them, per the design
decision. `road_cells: HashSet<IVec2>` (cell coordinates) is the source of
truth, the same role `buildings` plays for buildings; `add_road_cell` marks
all 36 underlying block tiles `Occupant::Road` in the existing `occupancy`
map with the same all-or-nothing "check every tile before marking any"
shape `place_building` already used, so a cell overlapping a building (or
another cell) is refused as a unit.

**`road.rs`**: rewritten to cell space — `Direction`'s offsets are
unchanged in value (still ±1) but now mean "one cell." Two new pieces:
`Direction::rotated`/`RoadConnections::rotated` (a quarter-turn walk using
the same north→east→south→west clockwise convention
`blueprint::rotate::Rotation`/`Cardinal` already use, so the result composes
directly with `blueprint::rotate_blueprint`), and `RoadPieceKind`
(`Isolated | DeadEnd | Straight | Corner | T | Cross`) with `select_piece`.
`select_piece` classifies by `RoadConnections::count()` first (0→Isolated,
4→Cross with no rotation search — neither has an orientation), then for
count 1/3/2 rotates a `canonical_pattern` for the matching kind through 0..4
turns via `matching_rotation` until it equals the actual connections. Count
2 splits on adjacency (`north && south || east && west` → opposite →
Straight; otherwise Corner) — the corner-vs-straight distinction the
conversation called for, confirmed by `select_piece_round_trips_every_
connection_pattern`, which checks all 16 `RoadConnections` values resolve
to the expected kind *and* that rotating that kind's canonical pattern by
the returned `Rotation` reproduces the exact input, not just one example
per kind.

**`road_catalogue.rs`** (new): `load_road_catalogue_dir` loads a fixed set
of six `.nbt` files by name (`isolated.nbt`, `dead_end.nbt`, `straight.nbt`,
`corner.nbt`, `t.nbt`, `cross.nbt`) rather than scanning a directory the way
`blueprint::catalogue` does for buildings — there are exactly six kinds,
not an open-ended set. A missing file is `RoadCatalogueError::Missing`,
reported per kind alongside whatever *did* load, the same per-file
tolerance 039 established. Validation is narrower than 039's: `x`/`z` must
be exactly `ROAD_CELL_SIZE`, `y` only bounded by `STRUCTURE_BLOCK_MAX_SIZE`.
No real `.nbt` pieces ship with this ticket (see Out of scope); the loader's
own tests build synthetic structures the same way `blueprint::catalogue`'s
tests do, and it isn't wired into `city::run()` yet — same "proven, not yet
used" state 039/040/042 landed their own resources in, `#[allow(dead_code)]`
throughout with a comment naming the eventual caller.

**`persistence.rs`**: `CitySave::roads: Vec<(i32,i32)>` (block tiles)
became `road_cells: Vec<(i32,i32)>` (cell coordinates) — same on-disk shape,
different meaning, which is exactly the silent-misinterpretation risk the
module's own "no migration path" policy exists to catch. `CURRENT_VERSION`
bumped `1` → `2` rather than trusting the field rename alone to break old
files: a new test (`an_old_block_tile_road_save_is_refused_not_silently_
reinterpreted`) writes a version-1 file with the old `roads` field name and
confirms it fails to deserialize into the version-2 shape (a `Parse` error
— RON's missing-required-field failure, encountered before the version
check even runs) rather than loading and placing every road cell six blocks
off from where it was.

`demolish.rs` and `ui/city_panel.rs` updated to the new API (a cell
covering the same block position in the one demolish test that cared, and
"N cell(s)" replacing "N tile(s)" in the city panel — F2/F3 are what will
make that count mean something a player placed).

No caller for `select_piece`/`road_catalogue` beyond their own tests —
F2 (drag-to-build) is the first one due, and is what would call
`select_piece` per road cell and `road_catalogue::RoadCatalogue::get` to
actually mesh and write one.

