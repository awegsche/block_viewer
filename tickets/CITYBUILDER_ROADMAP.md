# Citybuilder — design, current state, and remaining plan

Not a work item: the design document for the citybuilder game and the shared
world-edit infrastructure it uses. High-level tasks here get split into
numbered tickets in this directory when picked up (**next free number: 086**).
Companion to `ROADMAP.md`, which covers the viewer 001–029.

## The goal

A citybuilder played on a **real Minecraft save**. Buildings are blueprints
extracted with the existing 019–024 mechanism; placing one writes its blocks
into the save's region files, so you can close the game, open the world in
Minecraft, and walk the city you built.

The trip is meant to run both ways: blocks a player breaks in Minecraft are
damage to the building they belonged to, dropping its health and output until
repaired (group I, not built yet). The world isn't an export target, it's the
other half of the game state.

Iteration 1 (the current scope) places non-functional buildings and streets.
No production, no resources, no spending — but the blocks land in the world
for real, and the definition format has room for production rates and an
Anno-style tier tree.

## The architectural rule everything follows from

> **The city state is authoritative. The blocks in the world are a
> projection of it.**

The game owns the list of placed buildings and road cells — ids, positions,
rotations, styles — and saves it alongside the world. Blocks written into
region files are the *rendering* of that list, not the source of truth.

Consequences that later work depends on:

- **Undo/demolish** is a city-state edit plus a re-projection, not an attempt
  to remember what blocks used to be there.
- **Reconciliation** is possible: recompute what the world *should* look like
  from the city list, diff against the region files, write the delta. That is
  also the repair path when someone edits the world in Minecraft between
  sessions.
- **Chunk and region boundaries stop being a placement problem.** A building
  is one entry regardless of how many chunks its blocks land in; only the
  write path deals with the split.
- Corrupting a save is recoverable — city list plus terrain backup rebuilds it.
- **Damage (group I) is that same diff read as a game signal.** Intent minus
  reality is an error when we caused it and a gameplay event when the player
  did.

## Crate layout

One package, one lib, two three-line `[[bin]]` shims. Both games live *inside*
the lib so `pub(crate)` keeps working across shared modules. A workspace split
(`mc_core` / `block_viewer` / `citybuilder`) stays available as a later move.

```
src/
  lib.rs              pub mod world, blueprint, edit, sky, camera,
                      region_cache, streaming, chunk_pipeline, unload,
                      viewer, city; shared startup in lib.rs::world_app()
  world/              decode, mesh, atlas, tint, biome, block
  blueprint/          extract, structure (read+write), mesh, rotate, catalogue
  edit/               shared write path: policy, route, session
  selection/          crate root — blueprint::extract and I2's scan both use it
  viewer/             the viewer app, selection UI, ui/, paint
  city/               the game
  bin/block_viewer.rs fn main() { block_viewer::viewer::run() }
  bin/citybuilder.rs  fn main() { block_viewer::city::run() }
```

Upstream siblings: `ranvil` (`../ranvil`, Anvil read **and** write) and `rnbt`
(`../rnbt`, NBT parse + mutation API). Both are local path dependencies;
changes here often require editing those repos too.

---

# Current state

Groups W/B/C/D/E/F/G/H/I/R below are the labels tickets and code comments use.
Everything in W, B, C, D, E, F, G, R, H1 and H2 is built; all of I is not.

## W — The write path (`edit/`, plus `ranvil`/`rnbt` upstream)

Shared by viewer and game; it's what turns `block_viewer` into a general
explore-*and-modify* app.

**Upstream (`ranvil`)** provides: region file writer (sector table, timestamps,
zlib, atomic replace), mutable chunk access with dirty tracking, `set_blocks`
(palette insertion + `block_states` re-packing, batched per section), section
creation, `Heightmaps` pack/unpack/recompute, `isLightOn` clearing, orphaned
`block_entities` removal, `session.lock` detection, `.mcc` oversized-chunk
*reads*. Still open upstream: **ticket 020** (per-chunk timestamps — the
change-detection gate I5 needs) and **027** (writing oversized chunks).

`ranvil`'s writer **rewrites the whole region file** rather than reusing freed
sectors — a chunk grows when a placement adds a palette entry, so everything
after it moves anyway. A single-block edit therefore costs a 512×512-column
file rewrite, which is why edits batch by region and why the game defers
writes to an explicit Save.

**`edit` owns policy — what a well-formed edit is; `ranvil` owns the format.**

- **Edit model** (`edit`, ticket 031): order is preflight → `set_blocks` →
  heightmap invalidation, the last once per chunk at the **end of the
  transaction**, not per `set_blocks` call. Block-entity cleanup and
  `isLightOn` clearing are folded into `ranvil`'s `set_blocks`. `Status` gate:
  only `minecraft:full` chunks are edited. A `DataVersion` *incompatibility*
  between blueprint and save refuses loudly — compared by band, not by
  equality (ticket 069): a save played across updates is a patchwork of
  versions, and only the ones a block migration separates are a problem.
- **Routing** (`edit::route`): block coordinate → `(region, chunk, section,
  local index)`, batched by region file (a 20×20 building can straddle 4
  chunks or 4 region files). **Plans every region before applying any** —
  neither `set_blocks` nor `apply` is all-or-nothing across files, so a
  write-phase failure discards already-applied regions from the cache. A
  region carrying an earlier transaction's unsaved changes is refused
  (rollback would take those too): **one transaction at a time, saved before
  the next**. Placements reaching an ungenerated chunk are refused, not
  panicked on.
- **Write safety** (`edit::session::WriteSession`, ticket 033):
  `SessionLock::acquire` **holds** the lock for the session (released on drop)
  rather than probing — a probe can't close the check/write race, and holding
  keeps Minecraft out mid-edit. `commit` backs up *every* touched file before
  saving *any*, so a backup failure leaves the save untouched. A failure
  during saves can't be rolled back: `WriteError::WriteFailed` names what was
  and wasn't written plus the backup dir, and those regions stay dirty
  (retryable) rather than discarded. Dry run supported.
- **Live re-mesh**: an edit marks its chunks dirty via `EditReport::chunks`;
  `chunk_pipeline::ChunksEdited` drives the re-mesh, reusing 005-f's frontier
  dirty-chunk mechanism (do not invent a second one). The region cache holds
  the mutated region, so *reads* already see post-edit blocks — what goes
  stale is `DecodedWorld`/meshes.
- **Viewer paint** (`viewer::paint`, ticket 035): fills the current 019
  selection with a block. `WorldEdit::fill(bounds, state)` (shared with H1),
  `BlockState: FromStr` (inverse of 022's `Display`). Holds the region-cache
  lock for the whole commit — releasing partway breaks route's
  all-or-nothing. Blueprint stamping is not wired up here.

**Coordinate rules**, inherited from ticket 019 and fixed once in `edit` for
every caller: `bevy.z = -mc.z`, bounds are inclusive.

## B — Blueprints as building models (`blueprint/`)

- **`structure::{read_structure, read_structure_file}`** — vanilla structure
  `.nbt` → `Blueprint`, the inverse of 023's writer, mirroring its `Read`/path
  split. Stricter than the format requires: `blocks` must be dense (a gap is a
  malformed file, not implicit air). `Blueprint::origin` reads back
  `IVec3::ZERO`, `failed_columns` always `0`. Lets buildings be authored in
  Minecraft itself.
- **`mesh_blueprint`** — `Blueprint` → Bevy `Mesh`. A separate entry point
  from `mesh_chunk_column`: a blueprint is a `BlockState` palette with
  properties and no `BlockRegistry`. Reuses `world::atlas::resolve_faces` and
  `world::tint::resolve_block_tint` (both name-keyed) — no duplicated logic.
  Faces at the outer boundary are always emitted; air in the palette stays
  air; every biome-tinted block resolves against one caller-supplied
  `BiomeColors`. Mesh coordinates run `0..size` — a blueprint has no absolute
  world height until placed.
- **`rotate::rotate_blueprint(&Blueprint, Rotation) -> Result<Blueprint,
  RotationError>`** — 90/180/270 about Y. The mesh half is a transform; the
  block half rewrites palette properties or a rotated building is visibly
  wrong once written. `facing`, `axis`, 16-way `rotation` and the N/S/E/W
  connection-key group rewrite; `shape` disambiguates **by value, not by
  block** (stairs vs. rails share the property name with disjoint value sets);
  `hinge` and a chest's `type` pass through (they're relative to the block's
  own `facing`). Anything else → `RotationError::UnrotatableProperty`, except
  `Deg0`, which is identity and never inspects the palette. `Rotation` derives
  `Hash` + `Serialize`/`Deserialize`.
- **`catalogue::load_catalogue_dir(&Path) -> (BuildingCatalogue,
  Vec<(PathBuf, CatalogueError)>)`** — non-recursive scan of `*.nbt`
  (case-insensitive) under `assets/city/blueprints`, validated for size limit
  and a non-air-only palette. Failure is per-file: a missing dir is an empty
  catalogue (logged, not fatal); a bad file is skipped. The **filename stem is
  the id**. `assets/city/blueprints/house01.nbt` is the fixture.

## C — Definitions (`city::definition`, `city::road_definition`, `city::hot_reload`)

**Data files, not a scripting language.** Iteration 1's values are numbers and
references. RON is the format (serde, comments, real enums, no whitespace
significance). Reach for Rhai or Lua only when a *behaviour* must vary per
building rather than a number.

One `.ron` per building under `assets/city/buildings`; the **filename stem is
the id** (no inline `id` field), and `blueprint`'s stem must resolve to a real
catalogue entry or the definition is rejected (`UnknownBlueprint`).

```ron
Building(
    name: "Lumberjack's Hut",
    blueprint: "lumberjack.nbt",
    tier: 1,
    requires: [],                       // other building ids
    footprint: FromBlueprint,           // or explicit (x, z)
    production: Some(Production(        // parsed and displayed; not simulated
        outputs: [(item: "wood", per_minute: 4.0)],
        inputs:  [],
        radius:  Some(24),
    )),
    cost: [(block: "minecraft:oak_planks", count: 40)],
    integrity: Integrity(               // group I; per-building because a
        pristine_above: 0.95,           // warehouse should shrug off what
        ruined_below:   0.60,           // wrecks a lighthouse
    ),                                  // between the two, output scales
)                                       // linearly. Named curves later.
```

- **`ground_level`** (ticket 085, `#[serde(default)]` -> `0`) — which
  0-indexed Y layer of the blueprint is its own ground surface. Without it, a
  placement always lines terrain height up against the blueprint's `y=0`,
  which is wrong for a building whose blueprint buries a foundation below
  its visible surface (`lumber.ron`'s two solid dirt layers before its
  grass/path at `y=2`, hence `ground_level: 2`). Validated against the
  matched blueprint's own height in `load_entry`, not `validate` — a
  file-only check can't know how tall the blueprint it names actually is.
  `placement::resolve_placement` subtracts it from `grid::fit_footprint`'s
  `base_y` before `y_offset`'s manual nudge, and `BuildingDefinitions::
  ground_level` is how both the ghost preview and `commit` read it off the
  selected *definition* — a keyboard-stand-in selection with no
  `definition_id` behind it reads as `0`, the same gap `cost`/`requires`/
  `production` already leave one in.
- **`resolve_requirements`** — tiers and tech tree. A **loop** (not a single
  pass) alternating a dangling-reference check with DFS cycle detection
  (`find_cycle`) until a pass removes nothing: removing one bad entry can
  dangle another. The result is the maximal subset of loaded buildings whose
  `requires` graph resolves acyclically, order-independent. Errors:
  `DanglingRequirement`, `CyclicRequirement` (carries the whole loop). A cycle
  is unwinnable and fails silently ("button greyed out forever") if uncaught.
- **Production/cost fields are read but inert** — `city::ui::build_menu`'s
  `cost_line`/`production_line` display them on every catalogue row. Nothing
  simulates either. The point of exercising the schema is that a schema
  nothing reads drifts from reality.
- **`hot_reload::DefinitionHotReloadPlugin`** — polls a `(path, mtime)`
  snapshot of `assets/city/buildings` and `assets/city/road_types` once a
  second (deliberately no filesystem-watcher dependency) and replaces
  `BuildingDefinitions`/`RoadTypes` wholesale on change. Scoped to the two RON
  *data* directories; the `.nbt` geometry catalogues are **not** watched.
- **`city::ui::definition_errors`** — a window listing every currently-skipped
  `.ron`, seeded from startup's own load so a bad file is visible from the
  first frame rather than only after a live edit.

## D — City state (`city::state`, `city::persistence`, `city::journal`)

- **`City`** — placed buildings keyed by `BuildingId` (distinct from a
  definition's id), road cells, and a `HashMap<IVec2, Occupant>` occupancy grid
  (`Occupant::Building(BuildingId) | Road`) for fast "is this tile free".
  `place_building` plans the full footprint's tiles, checks all are free, then
  mutates — the same plan-before-apply shape as `edit::route`. Rotation swaps
  the occupied rectangle's (x,z) extent via `footprint_extent`/
  `footprint_tiles`, the horizontal counterpart of B3's block-grid axis swap.
  `state::cell_of` maps a tile to its road cell (`div_euclid`-correct for
  negative coordinates).
- **`persistence::{save_city, load_city}`** — `<save>/citybuilder/city.ron`,
  storing `CitySave`/`SavedBuilding`. **Occupancy is derived, not stored**:
  load rebuilds it through `City::insert_loaded`/`add_road_cell`, so a corrupt
  or overlapping file fails with `PersistenceError::Corrupt` rather than
  producing an inconsistent `City`. `next_id` is persisted verbatim (never
  recomputed) so a removed id is never reissued. **Save version 6**
  (1→2 when roads became cells, 2→3 when cells gained a `style`, 3→4 a `y`,
  4→5 a stair's `ascent`, 5→6 a cell's tunnel `variant`); an old save is
  refused rather than misplaced — including at 4→5, where defaulting the new
  field to "flat" *would* have been safe, because one quietly-defaulted field
  is the precedent the next unsafe one argues from, and 5→6 is exactly the
  case it was kept for: defaulting a cut tunnel to `Surface` re-fills it. `city::run()` skips persistence for ticket
  008's `empty_save` placeholder, otherwise loads synchronously before
  `App::run()` and saves on `AppExit`.
- **`journal::{Baseline, JournalEntry, Journal, reconcile, repair_edit}`** —
  every placement and demolition appended, carrying the **as-built baseline**
  (I1). One record, four consumers: undo, demolish's terrain restore,
  reconciliation, and the eventual damage diff. `Baseline::capture(edit,
  report)` reuses `EditPolicy::capture_replaced` (what an edit overwrote)
  rather than computing anything new. An entry snapshots a full
  `PlacedBuilding` (not a lookup key) because undo removes the building from
  `City` before the entry is read. `undo_last` is all-or-nothing against
  journal + `City` (`UndoError::Occupied` on conflict) and returns the
  restoring `WorldEdit` **uncommitted** — the "apply doesn't save" contract
  holds everywhere. `reconcile` recomputes expected world state from each
  building's own placement baseline (not by re-deriving the blueprint),
  grouped by region like `edit::route`. `BlockState` derives
  `Serialize`/`Deserialize`.

## E — Placement (`camera`, `city::picking`, `city::grid`, `city::placement`, `city::commit`, `city::demolish`)

- **RTS camera** — `camera::CameraMode::Rts`, a third mode alongside
  `Fly`/`Orbit` in the same enum, because `streaming`/`unload`/`sky` all find
  "the camera" via `Query<&Transform, With<CameraRig>>`. WASD pans on the
  yaw-relative ground plane, Q/E rotates yaw, scroll zooms, right-drag
  free-looks *ungrabbed* — left mouse is reserved for picking.
  `CameraStartMode` lets `city::run()` start in `Rts`.
- **Picking** — `city::picking::HoveredBlock`, built on
  `camera::block_under_cursor` (no second raycast). `PickingSet` orders it.
- **`grid::fit_footprint(origin, footprint, rotation, &DecodedWorld) ->
  FootprintFit`** — samples ground against already-decoded
  `DecodedWorld.columns` (no I/O) through a `city::grid`-local
  `is_ground`/`is_clutter_name` predicate on `ChunkColumn::topmost_matching`,
  so trees and fence posts don't read as a cliff. `base_y` is the footprint's
  lowest sampled point and is **purely an initial suggested height** feeding
  `PlacementSelection::y_offset`; **nothing refuses a steep placement**
  (`FitError::TooSteep` was removed in 058 — a hard 1-block step cap
  restricted buildable ground far more than it was worth on real terrain).
  `FitError::NotLoaded` is the only refusal left, which also means H1's
  terraforming is not load-bearing for placement.
- **`placement::resolve_placement`** — combines the terrain fit with
  `City::is_tile_free` into one green/red signal. Ghost mesh plus two
  `AlphaMode::Blend, unlit: true` materials cached by `(catalogue id,
  Rotation)`. Keyboard stand-ins where a menu would be: number keys pick a
  catalogue entry, `R` rotates, `Page Up`/`Page Down`/`Home` nudge
  `y_offset`, `Escape` clears.
- **`commit::CommitPlugin`** — `try_commit_placement` recomputes validity off
  *this* frame's inputs (never trusts the ghost's last-frame result).
  `City::place_building` runs synchronously the instant a click is accepted,
  claiming the tile before any write; a single `CommitState::pending` slot
  backpressures. `blueprint_edit` writes **every grid position including air**,
  clearing whatever terrain the fit tolerance left poking into the building.
  On success: `Baseline::capture` + `Journal::record_placement`, then
  `ChunksEdited`. On failure: `City::remove_building` — the transactional half.
- **`demolish::DemolishPlugin`** — `Delete` on the hovered tile.
  `City::occupant_at` finds the id, `Journal::placement_baseline` finds what
  to restore (refuses when there is none — an older-save edge case), and
  `Baseline::restore_edit` writes it back. **Ordering matters and is
  deliberate:** commit claims the tile *before* its write; demolish frees the
  tile only *after* its restoring write succeeds — the opposite order, which
  avoids a new placement landing on the same tile mid-write.
- **World writes are deferred to a manual Save** (ticket 051). Commit,
  demolish and undo apply straight to the shared `RegionCache`
  (`EditPolicy::allow_dirty_regions`) and stop — no session, no disk write.
  `city::save` is the only place that reaches disk: "Save world" in the city
  panel opens a real `WriteSession` and calls `WriteSession::flush`. `AppExit`
  flushes synchronously *before* `city.ron`/`journal.ron` are written, so
  those files never describe buildings the world never got. (Before 051 every
  edit rewrote a whole region file; there is no per-edit `WriteGate` any more.)

## F — Streets (`city::road`, `city::road_catalogue`, `city::road_definition`, `city::road_build`)

- **A road cell is `ROAD_CELL_SIZE` = 6 blocks square**, not a 1×1 tile — a
  rotated straight piece cannot stand in for a corner. `City::add_road_cell`/
  `remove_road_cell`/`road_cells()`/`is_road_cell`/`road_style_at` is the whole
  API; there are no dual grids.
- **`road`** — `connections_at`/`reachable_from`/`is_connected`, all
  recomputed off `City`'s occupancy grid on every call, so nothing here can go
  stale. `Direction`'s four offsets follow Minecraft's x/z convention (this
  module never touches a Bevy `Transform`). `reachable_from` on a non-road
  start returns the **empty** set, not a one-element set.
- **Auto-tiling** — `RoadPieceKind` (`Isolated | DeadEnd | Straight | Corner |
  T | Cross | Stair`) and `select_piece`, which rotates a canonical connection
  pattern until it matches a cell's actual connections; re-picked when a
  neighbour changes. `Stair` is the exception `select_piece` never returns —
  a ramp and a straight have identical connections, so it comes off the
  cell's own recorded `ascent` instead (see "Height", below).
- **The authoring convention** (ticket 066) — `canonical_pattern` is not a
  guess about how a style's `.nbt` files were exported, it's the **contract**
  they have to be exported to, because `select_piece` is style-blind and
  rotates every style's pieces through the same table. The rule: **every
  piece opens to the south** — `dead_end` south, `straight` north+south,
  `corner` south+west, `t` north+south+east, `cross` all four,
  `stairs` north+south ascending north. Read off ticket 063's shipped `dirt`
  pieces rather than imposed on them; the table originally disagreed with
  them, which rotated every dead end and corner 180° and every T 90°
  (straight and cross are symmetric under exactly the turn they were wrong
  by, which is what hid it). `road_catalogue`'s
  `the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations` reads
  the real `.nbt` files and fails if code and assets ever drift apart again.
- **Styles** — `assets/city/roads` is a directory *of styles*:
  `assets/city/roads/<style>/{isolated,dead_end,straight,corner,t,cross,stairs}.nbt`,
  each piece required to be exactly `ROAD_CELL_SIZE` on x/z; the subdirectory
  name is the style id. `RoadCatalogue` is keyed by
  `(style, RoadPieceKind, RoadPieceVariant)`;
  a missing piece is skipped, not fatal, and `RoadCatalogue::styles()` only
  counts a style once it has at least one piece **loaded**, not once its
  directory exists. Connectivity and shape selection are entirely
  **style-blind** — style only picks which `.nbt` is meshed/written once the
  shape is chosen, read back off `City` at write time.
  `road_build::RoadStyleSelection` (`[`/`]`, gated on the road tool,
  auto-picking the first loaded style) is the keyboard stand-in for choosing
  one, reading geometry rather than the type definitions.
- **Road types** — one RON per style under `assets/city/road_types`, carrying
  `name`, `travel_speed`, `capacity`; the filename stem **is** the style id
  (no separate field, unlike a building's `.ron`/`.nbt` pair). Validated
  against the geometry catalogue (`RoadDefinitionError::UnknownStyle`).
  **Schema only, inert** — there is no traffic/logistics system to hang a
  simulation off, and building one is a project of its own. Loaded into a
  `RoadTypes` resource next to `BuildingDefinitions`. This mirrors buildings'
  geometry/data split (`BuildingCatalogue` vs `BuildingDefinitions`).
- **Shipped assets**: `dirt` is currently the only style — six `.nbt` pieces
  under `assets/city/roads/dirt/` plus `assets/city/road_types/dirt.ron`
  (`travel_speed: 1.0`, `capacity: 4`). See that directory's `README.md` for
  the fixed filenames, the orientation convention and the cross-section the
  loader and the write path expect. `stairs.nbt` landed with ticket 068 —
  `6x8x6`, ascending north, rising exactly `ROAD_STAIR_RISE` = 4, which is
  what makes 067's stair planning do anything. Two gaps remain, both needing
  Minecraft rather than code: `isolated.nbt` is a byte-identical copy of
  `dead_end.nbt`, so a lone road cell renders as a south-pointing stub rather
  than an island; and **no `-tunnel.nbt` piece is checked in yet** (ticket
  071's code is complete and inert until one is, exactly the state 067's stair
  planning sat in until 068).
- **`road_build`** — drag-to-build, gated by `city::tool::ActiveTool`
  (`Building | Road | Terraform`, `T` cycles; placement/commit no-op unless
  `Building`). A drag is an **L-shaped `drag_path`** (row then column) so every
  cell stays a cardinal neighbour, which `select_piece` assumes. Preview uses
  a rotated catalogue piece where one exists, a flat translucent quad
  otherwise. Commit validates the whole path first, claims every cell in
  `City` synchronously, then batches **one merged `WorldEdit`** across the path
  plus any already-road neighbour needing re-tiling. **Road cells are not
  journaled** — no undo or demolish for them yet.
- **Height is a property of the placement, not the cell** (tickets 065/067).
  065 established that a cell's Y is resolved *once* and remembered on
  `state::RoadCell::base_y`, never re-derived — after a piece is written
  `grid::ground_height_at` samples the road surface as ground, so a re-tiled
  neighbour would climb a block per rewrite. (Before 065 the write origin's Y
  was a hardcoded `0` and every road ever built went into the deepslate.)
  067 changed *what* gets resolved once: fitting each cell to its own 6x6
  patch produced a run of individually-correct pieces separated by one-block
  cliffs. `road_build::plan_drag` gives the whole drag one profile — the
  first cell's level is the anchor, the last cell's snaps to it plus a whole
  number of `ROAD_STAIR_RISE` = 4 steps, and the `Stair` cells that bridge
  them are spread evenly over the path's straight interior (never an end,
  never the L's corner, never a cell something else branches into). A drag
  needing more steps than it has room for is refused whole. With no
  `stairs.nbt` loaded, every drag is flat at its first cell's level — the
  simple "one Y per road" rule as the degenerate case rather than a second
  mode. Ends prefer an existing road cell's recorded level over the terrain
  (`anchor_level`), so consecutive drags join flush; a stair is read on the
  *edge* being joined, since its two ends are four blocks apart. Preview and
  write share `plan_drag` and `cell_write_origin` so the ghost cannot stand
  anywhere the blocks won't land.
- **Tunnels are a variant, not a kind** (ticket 071). A road that has to run
  *under* terrain needs a bore, and 065/067's height planning has nothing to
  say about it: the profile happily puts the surface course under a hillside
  and the piece's three clearance layers mow three layers of it. The rule, the
  user's own: **more than 18 of the 36 columns in the single layer directly
  above a cell's piece are not air** -> that cell is a tunnel
  (`road_build::cover_at`/`ROAD_TUNNEL_COVER_MAJORITY`, the layer located by
  `piece_top_y` off the *surface* piece's own height, because a stair is
  `6x8x6` where the flat pieces are `6x5x6`). It resolves to
  `<kind>-tunnel.nbt` — the same kind, the same rotation, the same canonical
  orientation, a different `.nbt` — so `select_piece` and everything
  connectivity-shaped is untouched. `plan_tunnels` runs as a second pass over
  `plan_drag`'s output (it needs that pass's heights *and* the catalogue that
  pass deliberately doesn't see), shared by preview and commit like
  everything else here. Two gates: the cover majority, and the catalogue
  actually holding the piece — the same shape `stair_available` gives ramps,
  so a style with no `-tunnel` exports builds exactly what it built before
  rather than leaving unresolvable cells as holes. And, like `base_y` and
  `ascent`, the variant is **decided once and stored**
  (`state::RoadCell::variant`, `city.ron` version 6): here that isn't 065's
  subtle drift but a straight contradiction, since writing the tunnel carves
  away the very cover that chose it, and a re-tiled neighbour would fill the
  bore back in with hillside.
- **Connectivity queries** — four functions built entirely on
  `reachable_from`/`is_connected` (no second BFS).
  `touching_road_cells(city, building)` bridges a footprint to adjacent road
  cells via `state::cell_of`; `is_building_connected`/`buildings_connected`
  return `Option<bool>` (`None` = not currently placed). All still
  `#[allow(dead_code)]` — substrate for later logistics or an
  unconnected-building warning.

## G — UI (`city::ui`)

`UiPlugin` registers its **own** `EguiPlugin`, separate from the viewer's —
this game doesn't use `crate::selection`-based panels. Four windows:

- **Build menu** — catalogue grouped by tier, locked entries visible but
  disabled and showing what unlocks them, plus each row's cost and production
  lines. Bridges `PlacementSelection::catalogue_id` against
  `BuildingDefinitions` via `LoadedBuilding::catalogue_id`. **"Unlocked" is
  defined as: a `requires` id is met once at least one building of that type
  has been placed** — there is no separate "researched techs" resource.
- **City panel** — building counts, road length, a `WriteStatus` resource
  (last write, dirty regions, backup location, recorded by
  commit/demolish/road_build), the "Save world" button, and an Undo button via
  `city::undo` (`Journal::undo_last`'s caller).
- **Definition errors** — see C above.
- **Inspect panel** (ticket 083, roadmap G3) — the reason `ActiveTool::Inspect`
  can be `#[default]` at all: a left-click that no longer commits a placement
  needs something else to mean, and selecting a placed building to see what
  it's doing is that something. `city::picking::SelectedBuilding` is where the
  click lands (`City::occupant_at`, the same lookup `city::demolish` uses for
  `Delete`'s target; a road tile or empty ground clears the selection, a
  missed click leaves it alone — ticket 020's own "a missed click does
  nothing" precedent). The panel itself draws nothing at all with nothing
  selected, rather than every other panel's placeholder-line-in-an-empty-window
  shape — a resting-state panel visible constantly would be exactly the
  clutter making `Inspect` the default was trying to avoid. Shows the
  building's name (falling back to its catalogue id the same way
  `requirement_label` does), position, rotation, and — if it has a
  `ProductionState` entry — its running/starved/buffer-full state and buffer
  contents, one line per item against the buffer's shared cap. **No health
  field**: Group I doesn't exist yet, and a stub number now is something to
  rip out later rather than fill in — it lands when I4 does. `T` still cycles
  the other three tools; `Inspect` isn't a stop on that cycle, entered instead
  by clearing a placement (`Escape`, which now also resets `*tool`) or, at
  first, by never having left it.

**Which save it opens** (ticket 064): up to two positional CLI arguments,
parsed in the crate root (`resolve_selection_in`/`pick_named_save`) so
`block_viewer` honours them as its startup default too. A **single** argument
is classified against the filesystem — a directory that lists saves is ticket
008's saves directory, unchanged; anything else is the save, by name under
`.minecraft/saves` or by path (`citybuilder nbt_test`). Two arguments are
directory then save. Name matching is exact first, then case-insensitive, and
an unmatched name is a ticket-008-style startup issue listing the names that
do exist, not a silent fall back to the first save. The empty-first-argument
form this shipped with (`-- "" nbt_test`) is still accepted but is no longer
required: Windows PowerShell 5.1 doesn't pass an empty argument to a native
executable intact, so requiring one made the feature unusable on the shell
this repo is developed on. The city panel's **World**
section shows the resolved save, its region count, and that issue in red —
the citybuilder's stand-in for the viewer's save picker, which it can't
reuse; see "Deliberately not in this iteration".

## H — Terraforming (`city::terraform`)

`ActiveTool::Terraform` reuses `road_build`'s click/hold/release drag shape,
widened to a plain rectangle (`rect_tiles`) — no piece catalogue or adjacency
to keep straight. **Dig** clears the topmost block via
`ChunkColumn::topmost_non_air` (deliberately *not* the clutter-skipping
`is_ground`: a dig clears a tree the same as stone). **Level** reads the drag's
start tile as the target height, digging high tiles down and filling low ones
up with fixed `minecraft:dirt` (no material inventory). Neither reuses
`WorldEdit::fill` — both write a different value per position. Commits reuse
`city::commit::apply_building_edit`. No `City` entry and **no journal record**
for terrain edits, so a failed write has nothing to roll back beyond its
`WriteStatus` line. No preview mesh; console-only feedback.

## R — Render depth (`city::run()` only)

Below the terrain surface, ~7 sections per chunk column were decoded, meshed
and drawn for nothing. A per-chunk floor from **`min(OCEAN_FLOOR)`** over the
chunk's 256 heightmap columns (`ranvil` 013) cuts them. **Minimum, not
average**, is what keeps it safe: a ravine or cave mouth anywhere in the chunk
drags the floor down with it, so the cutoff never slices into a visible hole.
`block_viewer` still renders everything; the floor is switched on by
`city::run()` only. Digging is what later moves the floor — a re-*decode*, not
a re-mesh, since blocks under the floor were never decoded at all.

**Render distance and retention (ticket 070).** The floor is what pays for the
citybuilder's `RenderDistance(16)` — an RTS camera looks across a city, so the
viewer's default 10 ran out of terrain inside the frustum, and
`region_cache::recommended_capacity` is flat from 10 to 32 so the region cache
doesn't grow with it. Separately, and for both binaries, *what to load* and
*what to keep* stopped being the same square: `streaming::ChunkRetention` keeps
columns out to `render_distance + margin` (hysteresis, kills boundary thrash),
then holds anything past that as a timestamped *lingering* candidate for a
`grace` before unloading (going somewhere and coming back is free), with
`max_lingering` bounding the trail a long straight flight leaves behind by
evicting the farthest first. Defaults 2 / 30s / 256. Two consequences worth
remembering: `unload`'s in-flight cancellation is computed against the retain
square, not the load square, and a lingering entry deliberately survives being
emitted into `PendingChunkWork::to_unload` (nothing drains that list).

## L — Console logging (`world::warn`, ticket 081)

Nothing here goes through Bevy's `AssetServer` — every console line is one of
this crate's own `println!`s. The fallback warnings ("no texture mapping for
X", "unknown biome X", "section has no usable biome data") used to dedupe
against a `HashSet` built *inside* `build_block_uv_table` /
`build_block_tint_table` / `build_biome_tint_table` / `decode_chunk`, all of
which run **per background chunk task** — so "warn once" was once per chunk
and a few hundred streamed chunks reprinted each line a few hundred times.

`world::warn::WarnLedger` is that set declared as a `static` instead: keyed by
the missing thing (block name, biome name, chunk/region coordinate) so
distinct gaps stay enumerable, and printed exactly once per process.
`WarnLedger::new()` is a `const fn` and callers take one by reference, so it
is a shared value rather than a hidden global and tests hold their own.
Ledgers live next to the code that prints: `atlas::MISSING_TEXTURE` (also used
by `blueprint::mesh`, which re-meshes on every ghost rebuild),
`tint::MISSING_TINT`, `tint::UNKNOWN_BIOME`, `decode::MISSING_BIOME_DATA`,
`chunk_pipeline::UNDECODABLE_CHUNK`, `region_cache::UNREADABLE_REGION`.

The last two are the "failed to load" lines proper, and only their *logging*
is deduped — `region_cache` still retries after `FAILED_RETRY_COOLDOWN` and
the pipeline still returns `None`. Anything per-user-action (placement
refusals, save/load summaries, catalogue counts) is deliberately untouched.

---

# Remaining plan

## H2 — Materials, cost, production and haulage (iteration 2, done)

**Materials are Minecraft item ids** — the stock is keyed by
`minecraft:oak_planks`, not by an invented `"wood"`, so a building's existing
`cost` field became spendable without a schema change and the blocks the
world is made of are the economy's own units.

- **`city::inventory` (ticket 072, done)** — `Stock`, one global pile, `u64`
  counts (fractions belong to a producer's own accumulator, not to the
  ledger), and `Parcel`, the "some quantity of some materials" bundle that a
  spend, a yield, a journal delta and later a shipment all are. `spend` is
  all-or-nothing (a half-paid building must not exist); `remove` clamps at
  zero (a debit settles a world change that has already happened, and a
  negative stock is a debt no mechanic can discharge). Persisted to its own
  `<save>/citybuilder/stock.ron` rather than a `city.ron` field — that file's
  loader refuses any version but its own, so folding the stock in would have
  discarded every existing city to add an empty ledger to it.
- **`city::drops` (ticket 072, done)** — `assets/city/drops.ron`, a *table*
  rather than a one-file-per-id catalogue, because drops are a mapping over
  block names rather than a set of entities. Four rules in order: air drops
  nothing (hardcoded — a blueprint is mostly air and no asset file should be
  able to get that wrong), `nothing` drops nothing, `replaced` drops what it
  says, and **anything else drops itself**, so a missing table is a crude
  economy rather than a dead one. Keyed on `BlockState::name` alone;
  properties, tools and randomness are all deliberately out of scope.
- **Charging and crediting (ticket 073, done)** — one rule: *a write that
  removes blocks credits their drops; a write that restores blocks debits
  them; a building's own blocks are what `cost` buys.* Applied to all four
  write paths: a placement spends its definition's `cost` the instant
  `place_building` claims the tiles (refused before anything is claimed if
  it can't be paid, refunded verbatim if the apply fails) and credits the
  drops of `baseline.previous` on success; a terraform drag credits what it
  replaced and debits what it wrote (dig and level are the same rule, not
  two modes); a **demolition pays for its own backfill** and salvages
  nothing, which is what stops `place -> demolish -> place` from being a
  stone farm; undo settles the entry's own recorded numbers in reverse.
  - **The ledger is recorded, not recomputed** — `journal::Ledger` on every
    entry, `journal.ron` version **2**, read as a band `1..=2` (ticket 069's
    precedent, and the one place in this crate where a band beats
    `persistence`'s equality check: refusing a version-1 journal would
    discard every as-built baseline in it — roadmap I1, unrecoverable after
    the fact — to avoid defaulting a field whose correct value for those
    entries is provably empty). Recomputing a cost at undo time would refund
    a number nobody paid, since definitions hot-reload.
  - **Debits clamp, costs don't.** `Stock::remove_parcel` takes what's there:
    a demolition or an undo settles a world change that has already landed,
    and blocking one for want of dirt would leave city state and the world
    unable to agree. `Stock::spend` is the opposite — all-or-nothing, because
    a half-paid building must not exist.
  - **`PlacementSelection::definition_id`** — the build menu now names the
    *definition* alongside the catalogue id, because `cost` lives on the
    `.ron` and the two stems are free to differ. A selection made through
    `city::placement`'s keyboard stand-in has no definition behind it and is
    placed **free**, the same hole that already leaves it with no
    requirements and no production.
  - **Known gap, deliberately not smuggled in**: `PlacedBuilding::definition`
    is a *catalogue* id, so a **placed** building still can't find its own
    definition (this also means `build_menu`'s requirement check only works
    while the two stems coincide). Production is the first mechanic that
    needs it per-instance, and it's a `city.ron` version bump — it belongs to
    that ticket.
- **`city::economy` (ticket 074, done)** — `assets/city/economy.ron`, the
  economy's two tunable tables in one hand-editable file (`drops.ron` stays
  separate: hundreds of *block* names is a different kind of thing from a
  dozen knobs).
  - **The founding grant** closes 073's bootstrap gap: `start_stock` is
    handed out when a save has **no** `stock.ron` — not when it has one that
    happens to be empty, because a player who spent everything hasn't founded
    a new city. `inventory::load_stock` returns `Option<Stock>` precisely to
    keep those apart, and a stock file that fails to *load* isn't granted
    either (a corrupt file must not become free materials). Ships 512 dirt,
    256 cobblestone, 128 oak planks, 64 oak logs.
  - **Trivially transformable materials** — `1 oak_log -> 4 oak_planks` and
    friends, applied automatically when a placement is paid for, and only
    then: a demolition's backfill and a terraform's fill *debit*, and
    converting a player's logs to satisfy a clamped debit would be the game
    spending their materials to fill a hole. Conversions cover a shortfall
    rather than running over the pile, in whole runs (needing 2 planks with
    one log converts the log and leaves 2 planks behind). Chains work
    (logs -> planks -> sticks) via a recursion with a depth cap and a
    visited set, so a hand-written `A -> B, B -> A` gives up instead of
    hanging the frame; `from == to` is refused at load.
  - **One planner, two callers** — `economy::plan_payment` prices a cost
    against a stock and answers with *both* the conversions needed and the
    shortfall that survives them. `city::ui::build_menu` shows it (a row
    that can only be afforded by converting says what it will eat) and
    `city::commit` pays with it, so the menu can't read red above a click
    that succeeds.
  - **The conversion is part of the placement's ledger** — consumed joins
    `debited`, produced joins `credited`, so undo hands back the logs rather
    than the planks they became, and a failed apply reverses it the same way.
  - **Interchangeable groups (ticket 075)** — wood type survives the whole
    trip (`drops.ron`'s "a block drops itself" keeps `birch_log` as
    `birch_log`), which left a birch forest unable to build anything priced
    in oak. `economy.ron`'s `interchangeable` lists are the answer: every
    member converts to every other **1:1**, so the *payment* stops caring
    while the pile goes on saying what the map actually gave. Ratios stay in
    `conversions` — a ratio means the two aren't the same material after
    all, and an explicit ratio is tried before a synonym. Groups are
    expanded at lookup (`routes_to`), not at load: the shipped 44-name log
    group would otherwise become ~1,900 `Conversion` structs, and the build
    menu prices every visible row every frame. A name in two groups is a
    load error, since "which group wins" isn't a question the file should be
    able to ask. Collapsing wood to one id in `drops.ron` was the
    alternative, and it would have made a building that genuinely wants
    birch impossible to express.
- **A placed building knows its definition (ticket 076, done)** — 073's
  "known gap" closed, because everything below is a *definition* property
  read per placed instance. `PlacedBuilding::definition` had held a
  **catalogue** id since 042 while its name claimed otherwise; the two
  keyspaces coincided only because the one shipped `.nbt`/`.ron` pair shares
  a stem. Renamed to `catalogue_id`, with `definition_id: Option<String>`
  beside it and `City::definition_of` as the lookup. `city.ron` -> **7**,
  refused rather than defaulted (a version-6 file's field holds a *wrong*
  answer, not an absent one); `journal.ron` -> **3**, where the same field
  *is* defaulted inside its existing band, because a placement journalled
  before this genuinely has no definition. `None` stays the honest value for
  `city::placement`'s keyboard stand-in, which picks geometry and has no game
  data behind it.
- **The game clock (ticket 077, done)** — `city::clock`, the decision above
  turned into `GameSpeed` (`|| 1x 2x 4x`, pause as a *speed* rather than a
  second flag) and `GameClock` (`elapsed` plus this frame's `delta`, zero
  while paused), advanced in `First`. `delta_minutes()` is the unit every
  `per_minute` in a definition is already written in, so nothing downstream
  divides by 60 itself, and it is the **only** time the economy reads — a
  producer's accumulator and a cart's remaining travel cannot disagree about
  what "4x" meant. A per-frame clamp drops what a hitch would otherwise
  deliver at once; the remainder is dropped, not banked. Not persisted:
  nothing reads an absolute time, and a save that reopened paused would look
  broken.
- **Production (ticket 078, done)** — `city::production` reads
  `definition::Production`, inert since 040. Output goes into the *building's*
  own buffer, not the city stock, and a full buffer stops **everything**,
  inputs included. That stall is the mechanic: a farm that has visibly
  stopped is a fixable problem, one producing into nowhere loses half its
  output to a mistake nobody can see. Inputs are all-or-nothing across the
  whole list and paid a whole unit at a time (a `u64` stock cannot be charged
  0.003 of a plank); a starved producer keeps its debt and resumes from it.
  **No journal entry, ever** — production writes no blocks, and an "Undo"
  that clawed back a farm's output would be a different mechanic wearing the
  same button. State lives in `<save>/citybuilder/logistics.ron`, its own
  file for the reason 072 gave `stock.ron` one, and the **one** save file
  here whose version mismatch starts empty rather than refusing: a buffer
  regenerates in minutes, where a placement or an as-built baseline never
  does.
- **Warehouses (ticket 079, done)** — `definition::Warehouse`:
  `radius_cells`, `concurrent_hauls`, `handling_minutes`, `storage`. A
  warehouse **tier is not a new concept** — it is `tier` + `requires`, which
  041's tech tree and the build menu already implement, so warehouse02 is
  warehouse01 with bigger numbers.
  - **The working radius is measured along the road**, in cells: a BFS to
    `radius_cells` hops for coverage, then a Dijkstra *restricted to what it
    found* for the haul time, charging `1.0 / travel_speed` per cell entered.
    Two passes rather than one pruned Dijkstra, because the hop count along a
    *fastest* path can exceed the radius while a slower path stays inside it
    — one pass has to approximate one of the two questions. Hops for range
    and time for speed keeps a fast road from silently buying reach.
  - **A producer off the road is unserved**, and fills up and stops. That is
    the point rather than a limitation: it is what makes the road network the
    thing the economy runs on rather than decoration, and it is the first
    consumer of F4's connectivity queries (ticket 056), unused since they
    landed. An undescribed road style travels at 1.0 rather than being
    impassable.
  - **Storage is capped** — `economy.base_storage` plus every placed
    warehouse's `storage`. This cuts against 072's unbounded pile, and the
    reconciliation is that the cap is a *warehouse* property enforced against
    the pile, not per-warehouse storage: goods still don't live anywhere.
    `Stock::add_parcel_capped` is where it bites, on the three paths that
    credit the stock, and overflow is **reported, never dropped in silence**.
    `base_storage` exists so a fresh city can hold its own founding grant.
- **Haulage (ticket 080, done)** — one stack of one item, dispatched to the
  serving warehouse, arriving `travel_minutes + handling_minutes` later.
  `concurrent_hauls` is the "transport per time" knob; one warehouse serving
  six farms delivers them a stack at a time and falls behind, and that is
  what a tier upgrade fixes. One stack in flight per producer, so one full
  farm can't take every slot. One way only — the empty cart home isn't
  modelled, and `concurrent_hauls` stands in for its occupancy.
  - **A stalled producer ships its largest partial stack.** Without it a
    building with *mixed* outputs can fill its buffer without any one item
    reaching a full stack, and deadlock there forever.
  - **A delivery the city has no room for blocks at the warehouse holding its
    goods.** That closes the loop the storage cap opens: stock full -> hauls
    block -> slots stay occupied -> buffers fill -> producers stall, every
    step visible in the city panel and every step fixed by another warehouse.
    A partial unload keeps only the remainder, or the next tick would
    duplicate the stack.
  - **`RoadType::capacity` stays inert, deliberately.** Congestion — several
    hauls sharing a cell and slowing each other — is a real mechanic and a
    different ticket; using a *road's* number as a per-warehouse limit would
    put it on the wrong thing and make the eventual real one harder to add.
  - `logistics.ron` -> **2**, carrying in-flight stacks across a quit with
    their `remaining` intact: a stack does not teleport home because the
    player closed the window, and does not evaporate either.
- **Farm tiles (ticket 084, done)** — closes the open design question
  `lumber.ron` carried since it landed: whether a producer's footprint widens
  to reserve ground for Anno-style tiles placed around it, or those tiles are
  separate buildings placed near it. It's the latter, per 084's own design.
  `definition::Farm` (`Building::farm`) names another building by its
  *definition* id — the same keyspace `requires` uses — plus a search
  `radius_blocks` and `tiles_for_full_rate`. `city::farm::FarmCoverage`
  counts, per placed hub, how many placed instances of that tile are both
  within `radius_blocks` and *nearest* to it: **plain straight-line distance
  between the two buildings' own footprint rectangles, not the road
  network** — unlike a warehouse's `radius_cells`, a field doesn't need a
  road to reach its farmhouse, so the count is a Chebyshev gap (0 when the
  rectangles touch or overlap) rather than a road-cell BFS. Two overlapping
  catchments don't double-spend a tile: 084 calls for **nearest hub wins**,
  ties broken by the lower `BuildingId` — the same kind of arbitrary-but-
  deterministic tie-break `warehouse::compute_coverage` already uses for a
  producer equidistant from two warehouses. `production::tick` reads the
  count and scales *every* rate — outputs **and inputs alike** — by
  `tiles / tiles_for_full_rate` (clamped to `1.0`) before handing the spec to
  `advance_producer`: a half-tiled hub runs its whole recipe at half
  throughput rather than producing less while still paying full price for
  what it consumes. Zero tiles is zero rate, not a starved/stalled state,
  since nothing is owed and nothing is missing; the building simply has
  nothing to scale yet. A tile is placed through the *exact same* catalogue,
  footprint, occupancy, ghost-preview, commit, journal, undo and demolish
  machinery every other building already uses; nothing about placement
  changed to add this. `FarmCoverage` is derived every frame from `City` and
  `BuildingDefinitions`, never stored, the same rule `warehouse::Coverage`
  follows and for the same reason. Surfaced in both UI panels: the build
  menu's catalogue row for a farm-linked hub names what it needs before a
  player commits to building one (`Scales with <tile name> nearby (N needed,
  within R blocks)`), and the city panel's per-producer line appends its
  live `count/needed` tile fraction next to its running/starved/buffer-full
  state — 084's own "Place Farm Tile" inspect-panel button is still not
  built: it waited on inspect mode existing at all, and 083 (which shipped
  that) deliberately didn't add it either, since it needs the hub schema 084
  owns — a gap between the two tickets, open for whichever picks it up; the
  build menu's ordinary row already places one once the hub unlocks it.
  Shipped as the first real farm-tile
  pair: `lumber.ron`'s Lumberjack's Hut (now producing `minecraft:oak_log`,
  `farm.tile: "lumber_farm_01"`, radius 16 blocks, 3 tiles for the full rate)
  and `lumber_farm_01.ron`/`lumber_farm_01.nbt` (`requires: ["lumber"]`, no
  production of its own). The chest-based-output idea `lumber.ron`'s comment
  used to float — scanning a blueprint's chests directly rather than going
  through 078's abstract per-building buffer — is superseded, not merely
  deferred: this building's output goes through the ordinary `production`
  buffer like every other producer, and its blueprint's chests are set
  dressing.
- **Still open in H2**: production *chains* have never been played — the
  shipped set is farms with no inputs, so `Production::inputs`, the starved
  state and `resolve_requirements`' tier tree are all implemented and
  untested against a real recipe graph. Balance is a guess throughout.
  `Production::radius` is still inert (078 had no use for it). Warehouses and
  the wheat farm still run on **placeholder geometry** — `house01.nbt` is the
  only *other* structure export on disk, now that lumber and its farm tile
  are real — so a city with two warehouses and a wheat farm looks like three
  identical houses next to a genuinely distinct lumberjack's hut and its
  fields; see `../todo.md`.

## I — Damage: the world diffing back (iteration 2, except I1)

Blocks a player alters in Minecraft, inside a building's volume, count as
damage; past a threshold the building stops working. This is the architectural
rule turned into gameplay — the same intent-vs-reality diff reconciliation
already needs, read as a game signal instead of an error.

**I1. The as-built baseline — the one part that cannot wait.** At placement,
record what was actually written: positions, `BlockState` at each, and the
save's `DataVersion`/chunk timestamps at the time. A building placed without a
baseline can never be diffed later: "what it should look like" isn't
recoverable after the fact, and re-deriving from the blueprint alone is wrong
because the blueprint contains air and the terrain fit may have adjusted the
placement. *Positions and states already ship — `Baseline::capture` runs on
every commit. `DataVersion`/timestamps are the part still missing, and
timestamps are blocked on `ranvil` 020.*

**I2. The scan.** Per building, read its volume out of the save and diff
against the baseline. Reuse `blueprint::extract`'s bounds-walking read path —
no new walk needed. Two failure modes that must **not** read as damage:

- **An unreadable/missing chunk is *unknown*, not *destroyed*** —
  `Blueprint::failed_columns` already separates real failures from ungenerated
  chunks; inherit that.
- **A `DataVersion` change** — every block could mismatch after a Minecraft
  version bump through no fault of the player. Refuse to scan and offer a
  re-baseline. One comparison prevents the single most damaging false positive
  available.

**I3. The comparison policy — the hard one, and the actual mechanic.** Naive
`BlockState` equality does not work. Classify per block:

- **Structural** (walls, floors, roof, stairs, logs) — full equality including
  properties. This is the denominator.
- **Interaction state** (`open`, `powered`, `lit`, a lectern's `has_book`) —
  ignore. Opening a door is not vandalism.
- **Volatile** (leaves, grass/dirt spread, water/lava flow, snow, fire, crops,
  ice, copper oxidation) — exclude entirely; these change with no player
  involved.
- **Blueprint air** — a block placed there is *obstruction*, a separate later
  mechanic: excluded from the denominator, counted separately.

Build a per-block-name table once against the registry, following
`world::tint::build_block_tint_table`. **When unsure, classify as volatile** —
under-reporting damage is a mild disappointment; a false positive that ruins a
city because it snowed is a bug report and a reload from backup. Budget real
time for tuning this against a real world.

**I4. Health, thresholds, malus.** `health = 1 - damaged_weight /
structural_weight`, uniform weights to start but written as weighted from the
outset. State: `Pristine | Damaged(health) | Ruined | Unknown`. Thresholds and
the malus curve come from the definition's `integrity` block, per-building.
**The scan is stateless** — rebuilding a wall by hand heals the building;
damage describes the world right now, not an accumulated counter. Production
is inert, so malus is a displayed number first.

**I5. Scan scheduling and the timestamp gate.** Gate on a region file's
per-chunk last-written timestamp: if nothing under a building has been
rewritten since the last scan, nothing changed. Needs `ranvil` 020. Compare
against the baseline's own recorded timestamp, or our own writes make every
building look changed the instant it's placed. Schedule: full scan on load
behind the gate on the async task pool; re-scan on chunk re-read; on demand on
inspect. **Never on the main thread.**

**I6. Repair.** Re-project the baseline onto the volume through the existing
write path — nearly free. A free button in iteration 1; costs materials in
iteration 2.

**I7. Display.** Per building: health percent, state, output multiplier, in an
inspect panel off picking. City-wide: counts by state in the city panel.
In-world at-a-glance (a mesh tint via the vertex colour channel, or a gizmo
reusing `selection::gizmo`) beats making the player click to find out.

## Standing decisions for future work

- **Lighting is Minecraft's job — decided, not deferred.** This project never
  computes light. `set_blocks` clears the chunk's `isLightOn` byte and the
  game relights when it gets round to it; a building wrongly lit until
  something touches its chunk is an accepted cost. The flag is confirmed
  present as a root `TAG_Byte` on the real save (`DataVersion` 4438) and
  cleared on every edit. If the game turns out not to honour it, the fallback
  is deleting the affected sections' `BlockLight`/`SkyLight` arrays — **not**
  writing a lighting engine.
- **`Heightmaps` follow lighting: let the game rebuild them.** An edited
  chunk's compound is deleted (`ChunkRegion::remove_heightmaps`), not
  recomputed. The **read** half stays useful regardless: chunk-root heightmaps
  say how deep terrain goes before a section is decoded, which is what R's
  render-depth cutoff keys off.
- **One transaction at a time**, saved before the next — `edit::route`'s
  rollback contract depends on it.
- **Round-trip tests before encoders.** Ticket 001 was palette/bit-width
  arithmetic in the read direction and shipped broken upstream; the same
  arithmetic in the write direction got its round-trip property test first.

## Deliberately not in this iteration

- **Terrain generation.** Placements reaching ungenerated chunks are refused,
  not generated.
- **Lighting computation of any kind.** We clear `isLightOn` and accept stale
  light.
- **Writing block entities.** Blueprints with chests/signs place their blocks;
  the entities are dropped with a warning. Removing *existing* ones we
  overwrite is in scope — corruption avoidance, not a feature.
- **Entities and mobs.** Structure files can carry them. Ignored.
- **Production simulation.** Landed in full: resources and spending (072-075),
  then production, warehouses and haulage (076-080). I4's malus now has
  something to multiply — a damaged producer's rate — which makes M7 a
  better-connected target than it was when this line was written. What is
  still *not* in scope: congestion (`RoadType::capacity`), obstruction, and
  anything that models where goods physically sit.
- **Damage detection (I2–I7).** Iteration 2 — but I1's baseline belongs to
  iteration 1, since it can't be added retroactively.
- **Obstruction as a distinct mechanic** (blocks placed in a building's
  declared air). I3 counts it separately; nothing reads it yet.
- **Multiplayer or any concurrent access to the save.** The write session
  refuses instead.
- **Switching saves at runtime.** The viewer's save picker (ticket 007) tears
  down streaming state and swaps `LoadedSave`; the citybuilder would also have
  to save and swap `City`/`Journal`/`CitySavePath` and flush E's deferred
  world writes to the save being left. Ticket 064 chose the save at launch
  instead.

## Milestones

M1 ("two binaries"), M2 ("the world can be changed"), M3 ("a building has a
shape"), M4 ("a city exists"), M5 ("streets") and M6 ("and it's mine" —
demolish, undo, terraform) are all done. Iteration 1 is delivered.

M-H2 ("the economy runs") is done as of ticket 080: a farm produces, a cart
carries a stack down the road you built, and a warehouse's radius and tier
decide whether it arrives. Roads stopped being decoration.

**M7 — "the world answers back"**: I2–I7 plus `ranvil` 020. Break a wall in
Minecraft, come back, and the building says so. The first mechanic that makes
the round trip *matter* rather than just work, and the right first target for
iteration 2 — ahead of production, because it needs no economy to be
interesting. Note that damage *detection* needs no write path at all (it's
pure reading on `blueprint::extract`); only repair does.

Everything after that is production, resources, and the tier tree coming
alive.
