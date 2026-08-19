# 059 - Multiple road styles

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 540
tests, all passing). User-directed: tickets 053/054/055 (roadmap F1/F1b/F2/F3)
landed exactly one road style — a single fixed six-piece set loaded from a
flat `assets/city/roads/*.nbt` directory. `Occupant::Road` and
`state::City`'s `road_cells: HashSet<IVec2>` carry no style data at all, so
there was no way for a placed cell to remember which of several styles it
was built as, even if several `.nbt` sets existed on disk. This ticket adds
that dimension. Roadmap group F (`tickets/CITYBUILDER_ROADMAP.md`) is the
parent; recorded there as a revision to F1b/F3 once done.

## The gap

- `road_catalogue::RoadCatalogue` is keyed by `RoadPieceKind` alone — one
  piece per shape, system-wide.
- `assets/city/roads/{isolated,dead_end,straight,corner,t,cross}.nbt` is a
  flat, fixed six-file layout — no room for a second style alongside it.
- `state::City`'s road cells are a bare `HashSet<IVec2>` ("is this tile
  road"), not "is this tile road, *and which kind*."
- `CitySave.road_cells` (persistence) is `Vec<(i32, i32)>` — same gap on
  disk.
- Nothing in the road tool (`city::road_build`, `city::tool::ActiveTool`)
  lets a player choose which style to place.

## The change

**Directory layout**: `assets/city/roads/<style>/{isolated,dead_end,
straight,corner,t,cross}.nbt` — one subdirectory per style, its name (like a
building's filename stem) *is* the style id. `road_catalogue::
load_road_catalogue_dir` scans `assets/city/roads` for subdirectories
instead of loading six fixed files directly out of it; each subdirectory is
loaded the same six-fixed-name way as before. A missing/empty
`assets/city/roads`, or a stray non-directory file next to the style
folders, is still an empty catalogue, not an error — same tolerance ticket
054 established for a missing piece.

**`RoadCatalogue`**: keyed by `(style, RoadPieceKind)` (a nested
`HashMap<String, HashMap<RoadPieceKind, Blueprint>>`, not a flat tuple key —
`get(style, kind)` avoids allocating a lookup key). New `styles()` returns
every style id that has at least one piece loaded, sorted for a
deterministic cycle order.

**`state::City`**: `road_cells: HashSet<IVec2>` becomes
`HashMap<IVec2, String>` (cell -> style id). `add_road_cell` takes a style;
`road_style_at(cell)` is the new lookup; `road_cells()` keeps returning cell
coordinates only (`.keys()`), so call sites that only ever wanted "is this a
road cell" (the city panel's count, `city::road`'s connectivity queries) are
untouched. Re-adding an already-road cell stays idempotent and keeps its
*existing* style — changing a cell's style is out of scope here (a future
"repaint" tool, not this ticket).

**Persistence**: `CitySave.road_cells` becomes `Vec<SavedRoadCell { x, z,
style }>`; version bumps 2 -> 3, refusing an old file rather than
misinterpreting its shape — same call ticket 054 made for the 1 -> 2 bump.

**The write path is style-agnostic beyond the lookup**: `road::
connections_at`/`reachable_from`/`select_piece` don't change at all — a road
cell's *shape* (which of six pieces) depends only on which neighbours are
road, never on style, so a dirt path and a paved street connect to each
other exactly like two dirt cells would. Style only decides *which*
`.nbt` gets meshed/written for a given `(kind, rotation)` once the shape is
already known — `road_build::road_write_edit` reads each affected cell's
style back off `City` itself (every affected cell is either already in
`City`, or was just added there by this same commit, by the time it runs) —
"the city state is authoritative" extends to road style the same way it
already covers footprint occupancy.

**Choosing a style, no build menu yet**: `road_build` gets a small
`RoadStyleSelection` resource (`current: Option<String>`), the same
keyboard-stand-in role `placement::PlacementSelection`/`cycle_selection`
play for buildings — `[`/`]` cycle through `RoadCatalogue::styles()`
(unused keys, gated on `ActiveTool::Road` unlike the building keys, since
they're new). The first loaded style is auto-selected the moment the
catalogue has one, so a single-style setup needs no keypress at all. A drag
that would place a *new* cell with nothing selected (no styles loaded, or a
catalogue that failed to load anything) is refused with a console message,
the same tone `try_commit_drag`'s existing "no catalogue"/"no matching
piece" refusals use.

## Explicitly not in scope here

- **Repainting an existing road cell to a different style.** Idempotent
  re-add keeps the original style; changing it needs demolish/rebuild for
  now.
- **A real style picker UI.** `[`/`]` is the same stand-in tier ticket 047
  shipped for buildings — G1's eventual build menu (or a road-specific
  panel) is the real fix, not scoped here.
- **Per-style cost/tier/unlock data**, mirroring `city::definition`'s
  buildings schema. A road style today is exactly a `.nbt` set, nothing
  more — same "just geometry" scope the original six pieces had.
- **Real `.nbt` assets.** Same as 054/055: no style ships with real
  Minecraft structure-block exports; this ticket is proven against
  synthetic fixtures.

## Scope

- `city::road_catalogue`: style-keyed `RoadCatalogue`, subdirectory scan,
  `piece_path(dir, style, kind)`, updated module docs, updated tests.
- `city::state`: `road_cells` -> `HashMap<IVec2, String>`,
  `add_road_cell(cell, style)`, `road_style_at`, `road_cells_with_styles`.
- `city::persistence`: `SavedRoadCell`, version 3, updated tests (including
  the version-2 fixtures embedded as RON literals).
- `city::road_build`: `RoadStyleSelection` + `cycle_road_style`, style-aware
  preview cache key and `preview_mesh`, `road_write_edit` reading style off
  `City`, `try_commit_drag`'s new style/refusal wiring, updated tests.
- Every other `add_road_cell` call site (tests in `city::road`,
  `city::demolish`, `city::persistence`) gets a style argument.
- `CITYBUILDER_ROADMAP.md`: record this as a revision to F1b/F3.

## Done when

- `cargo build`/`cargo test --lib` clean.
- Two styles loaded side by side in a test fixture resolve independently
  (`RoadCatalogue::get("dirt", kind)` vs `get("paved", kind)`), and a road
  network mixing both styles still tiles correctly (shape from
  connectivity, asset from each cell's own recorded style).
- A city save round-trips a road cell's style.

## Resolution

Landed as scoped.

**`city::road_catalogue`**: `RoadCatalogue` is now a nested
`HashMap<String, HashMap<RoadPieceKind, Blueprint>>` (`get(style, kind)`,
`styles()`) rather than a flat `HashMap<RoadPieceKind, Blueprint>`.
`load_road_catalogue_dir` scans `assets/city/roads` for subdirectories
(`style_dirs`, sorted, non-directory entries ignored) and loads each one's
six fixed filenames the same way the old flat loader did; the skip list
grew a `style` field (`Vec<(String, RoadPieceKind, RoadCatalogueError)>`).
`piece_path` gained a `style` parameter.

**`city::state`**: `City`'s `road_cells` field is now
`HashMap<IVec2, String>` (was `HashSet<IVec2>`). `add_road_cell(cell,
style)` — idempotent re-add keeps the *original* style, exactly as scoped
(no repaint). New `road_style_at(cell) -> Option<&str>` and
`road_cells_with_styles() -> impl Iterator<Item = (&IVec2, &str)>`;
`road_cells()` unchanged in shape (`.keys()`), so `city::road`'s
connectivity queries and the city panel's count needed no changes at all.

**`city::persistence`**: `CURRENT_VERSION` 2 → 3. `SavedRoadCell { x, z,
style }` replaces the bare `(i32, i32)` tuple. A version-2 file (no
`style` field) is refused, not guessed at — same call as the 1→2 bump.

**`city::road_build`**: new `RoadStyleSelection` resource (`current:
Option<String>`) and `cycle_road_style` system — `[`/`]`, gated on
`ActiveTool::Road`, sorted cycle order via `RoadCatalogue::styles()`,
auto-picking the first loaded style the moment one exists so a
single-style setup needs no keypress. The actual design decision: a
placed cell's style lives on `City` itself (`road_style_at`), not threaded
through as a parameter — `road_write_edit` and the preview both read a
cell's style back off `City` once it's recorded there, and `RoadStyleSelection`
only matters for a cell that's brand new to the current drag. `preview_mesh`'s
cache key grew from `(RoadPieceKind, Rotation)` to `(String, RoadPieceKind,
Rotation)`; `try_commit_drag` refuses a drag up front if it would place a
*new* cell with no style selected, but a drag that only re-crosses existing
road commits regardless (nothing there needs a style).

Every other `add_road_cell` call site across the crate (tests in
`city::road`, `city::demolish`, `city::persistence`, `city::state`) picked
up a style argument — 15 test failures from `write_structure_file` not
creating the new style subdirectory itself surfaced immediately on the
first `cargo test` pass; fixed by a small `write_piece` test helper in
both `road_catalogue` and `road_build` that creates the style dir before
writing.

New tests: two independent styles loading side by side and resolving
correctly (`road_catalogue`, `road_build::road_write_edit`), a cell mixing
two different styles in one write, `road_style_at`/idempotent-keeps-
original-style/`road_cells_with_styles` on `City`, a style round-tripping
through save/load, and `cycle_road_style`'s auto-pick/wrap/tool-gating —
alongside updating every pre-existing road test for the new signatures.

Summarized on `CITYBUILDER_ROADMAP.md` under F1b/F3.
