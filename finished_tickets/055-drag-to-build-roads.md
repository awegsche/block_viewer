# 055 - Drag-to-build roads

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 496
tests, up from 474). See the Resolution.

## Part of

Roadmap group F (`tickets/CITYBUILDER_ROADMAP.md`): F2 ("drag-to-build"),
wired straight through to the rendering/write half of F3 ("auto-tiling the
pieces") that ticket 054 explicitly left open — 054's own resolution named
this as the next ticket due, and what would call `road::select_piece` and
`road_catalogue::RoadCatalogue::get` to actually mesh and write a cell.

## Problem

053/054 built the whole road *model* — cell occupancy, adjacency queries,
piece selection, a piece catalogue loader — with no caller beyond their own
tests. There was no way to actually build a road in the game: no input, no
preview, and nothing wiring a cell's selected piece to a mesh or to the world.

## Goal

Per the roadmap: "Click-drag from A to B, routed over the grid, with a live
preview of the cells it would claim and their cost," plus F3's remaining
"wiring that to a spawned/rotated mesh... and to the write path... once F2
gives it something to place."

## Scope

- A road tool, switchable from building placement (`T`), so a click means one
  thing at a time.
- Click-drag from a start cell to the hovered cell, routed as an L-shape.
- A live preview per cell in the path — a real, rotated piece mesh when the
  catalogue has one for the resolved kind, a flat tinted quad otherwise (no
  real `.nbt` road pieces exist yet, per ticket 054).
- Committing a drag: all-or-nothing validation, `City::add_road_cell` for
  every cell, and a batched world write for every affected cell (the path
  plus already-road neighbours whose own piece might now need to change).
- Wiring `road_catalogue::load_road_catalogue_dir` into `city::run()`.

## Out of scope

- Real `.nbt` road pieces — still nobody's authored `isolated.nbt` etc.
  (ticket 054's own "no real assets yet" stands).
- A journal entry / undo / demolish for a road cell — `City`'s own
  persistence still saves `road_cells` regardless.
- F3's slope handling and F4's connectivity queries.

## Done when

- `cargo build`/`cargo test --lib` clean.
- Tests: `drag_path`'s L-shape and its edge cases; `cell_of`; per-cell
  occupancy/validity; `affected_cells`' neighbour inclusion and dedup;
  `connections_with_path`'s mid-drag hypothetical connectivity;
  `road_write_edit`'s per-cell piece resolution, offsetting, and its
  "no matching piece, no edit" tolerance; `poll_road_build`'s success/failure
  transition, including that a failed write only rolls back cells *this*
  drag newly added.
- Manual verification (drag out a road, watch the preview switch pieces as
  neighbours change, release and confirm it lands) goes in `../todo.md` —
  moot until real `.nbt` pieces exist to actually see.

## Resolution

Landed as scoped, across two new modules plus small edits to five existing
ones.

**`city::tool`** (new): `ActiveTool { Building, Road }`, toggled by `T`,
guarded by `camera::EguiInputCapture` the same way every other keyboard
stand-in in this game is. Read through `Option<Res<ActiveTool>>` everywhere,
defaulting to `Building` when absent — so every existing test app that never
adds `ToolPlugin` (there are many: `placement`'s `selection_test_app`,
`commit`'s `commit_test_app`, `demolish`'s `demolish_test_app`, ...) keeps
behaving exactly as it did before this ticket, with no changes needed to any
of them. Deliberately *not* wired to auto-switch back to `Building` on a
number-key or build-menu pick — a `T` press to get back is an acceptable,
easy-to-explain rough edge next to threading a second `ResMut` into two more
call sites for it.

**`placement::resolve_ghost`** and **`commit::try_commit_placement`** both
gained the same guard: hidden/no-op unless `ActiveTool::Building` (or the
resource is absent). `resolve_ghost` takes the tool as a plain value
parameter, computed by its caller off the resource — the same "pure function
takes already-unwrapped values" shape the rest of that module already uses,
so every existing test calling it directly needed one added argument
(`ActiveTool::Building`), plus a new test proving the road tool hides it.

**`city::road_build`** (new) is the rest of the ticket:

- `drag_path(start, end)`: an L-shape, `start`'s row then `end`'s column, not
  a diagonal — every consecutive pair stays a cardinal neighbour of the next,
  which `road::connections_at`/`select_piece` assume throughout. A diagonal
  path would need pieces this catalogue has no concept of.
- The drag itself: `update_drag_state` sets `RoadDragState::start` on a left
  click (road tool active, nothing captured by egui), cleared on release
  regardless of outcome. `current_path` is the single hovered cell when
  nothing is being dragged — a plain click is a valid one-cell placement.
- The preview: `connections_with_path` is `road::connections_at`'s own logic,
  but treating every cell in the *current* path as road too — so a straight
  run previews as a run of `Straight` pieces mid-drag, not disconnected dead
  ends. `preview_mesh` mirrors `placement::ghost_mesh` for a real catalogue
  piece (rotated via B3, meshed via B2, cached by `(RoadPieceKind, Rotation)`
  — six kinds, four rotations, no eviction needed) and falls back to
  `quad_mesh`, a flat unrotated plane built the same low-level way
  `sky::bodies::quad_mesh` builds its billboard, when the catalogue has
  nothing for the resolved kind — which is every kind today, since no real
  `.nbt` pieces exist yet. The pool of preview entities (`RoadPreviewState::entities`)
  grows to the longest path shown and only ever hides its tail, never
  despawns — the same "update in place" reasoning `placement::GhostPreview`
  already gives for its one entity, generalised to a pool.
- Terrain fit reuses E2 directly rather than a second height walk: a road
  cell is exactly a `ROAD_CELL_SIZE`-square footprint at `Rotation::Deg0`, so
  `cell_fit` calls `grid::fit_footprint` with that footprint. Slopes are the
  acknowledged gap the roadmap's own F3 entry names as an iteration-2
  deferral "if it bites" — this ticket doesn't touch that, only reuses
  whatever E2 already decided about a 6x6 patch of ground.
- Committing (`try_commit_drag`/`poll_road_build`): the whole path is
  validated before any of it is touched — the same "plan every tile before
  marking any of them" shape `state::City::place_building`/`add_road_cell`
  already use, lifted one level to a multi-cell drag. `City::add_road_cell`
  runs synchronously for every path cell before the write starts (mirroring
  `city::commit`'s own ordering, and for the same reason: a second drag over
  the same cells must see them taken immediately, not once an async task
  gets around to it). `affected_cells` is `path` plus any *already-road*
  neighbour — a pre-existing dead end that just grew a neighbour may need to
  become a straight or a corner, and this ticket re-resolves and re-writes it
  too, not just the newly-added cells. `road_write_edit` folds every affected
  cell's own `commit::blueprint_edit` (now `pub(super)`, reused verbatim
  rather than a second copy) into one merged `WorldEdit`, so a multi-cell
  drag is one routed transaction, not one per cell; a cell with no matching
  catalogue piece contributes nothing to the edit but keeps its `City` entry
  regardless — the state is authoritative whether or not there's an asset to
  render it with yet. On a failed write, only the cells *this* drag actually
  added are rolled back (`newly_added`, tracked before the mutation) — a
  neighbour that merely needed re-tiling was never newly claimed by this drag
  and must not be un-built by its failure.
- `write_status::WriteKind` gained a third-plus-one variant, `Road`
  (`city_panel::kind_verb` reads it as "Built"), rather than borrowing
  `Placed` — a road commit's `WriteRecord::building` names a cell count
  (`"N road cell(s)"`), not a `BuildingCatalogue`/`BuildingDefinitions` id.

**`city::run`** wires `road_catalogue::load_road_catalogue_dir("assets/city/roads")`
in (the same synchronous-before-`App::new()` shape ticket 039 established for
the building catalogue), always inserting whatever loaded — even an entirely
empty catalogue — so `road_build` reads it unconditionally rather than
through a second layer of `Option`. `ToolPlugin` and `RoadBuildPlugin` are
added alongside the existing picking/placement/commit/demolish plugins.

**Dead-code cleanup**: `road::RoadConnections`, `RoadConnections::count`,
`connections_at`, `RoadPieceKind` (and `::ALL`), `select_piece`, and their
private helpers (`Direction::rotated`, `RoadConnections::rotated`,
`canonical_pattern`, `matching_rotation`) all lost their `#[allow(dead_code)]`
markers — this ticket is their first real, non-test caller, the whole chain
now reachable from `city::run()`. Same for `road_catalogue::RoadCatalogueError`,
`RoadCatalogue::get`, and `load_road_catalogue_dir`. `RoadCatalogue::is_empty`,
`road::reachable_from`, and `road::is_connected` keep theirs — F4's
connectivity queries are still what's missing to call them for real.

18 new tests in `road_build` (`drag_path`'s shape and edge cases, `cell_of`,
per-cell occupancy/validity, `affected_cells`, `connections_with_path`,
`road_write_edit`'s piece resolution/offsetting/no-match tolerance,
`poll_road_build`'s success/rollback), 3 in `tool`, 1 in `placement` — 474 to
496 per `cargo test --lib`.

No manual verification recorded as done — `../todo.md` carries the checklist,
though it's moot until real `.nbt` road pieces exist to actually watch land
(today, every drag falls back to the flat quad preview and records cells in
`City` with nothing written to the world, exactly as designed).
