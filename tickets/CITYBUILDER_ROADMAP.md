# Citybuilder — design, current state, and remaining plan

Not a work item: the design document for the citybuilder game and the shared
world-edit infrastructure it uses. High-level tasks here get split into
numbered tickets in this directory when picked up (**next free number: 065**).
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
Everything in W, B, C, D, E, F, G, R and H1 is built; H2 and all of I are not.

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
  only `minecraft:full` chunks are edited. A `DataVersion` mismatch between
  blueprint and save refuses/warns loudly.
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
  recomputed) so a removed id is never reissued. **Save version 3**
  (1→2 when roads became cells, 2→3 when cells gained a `style`); an old save
  is refused rather than misplaced. `city::run()` skips persistence for ticket
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
  T | Cross`) and `select_piece`, which rotates a canonical connection pattern
  until it matches a cell's actual connections; re-picked when a neighbour
  changes.
- **Styles** — `assets/city/roads` is a directory *of styles*:
  `assets/city/roads/<style>/{isolated,dead_end,straight,corner,t,cross}.nbt`,
  each piece required to be exactly `ROAD_CELL_SIZE` on x/z; the subdirectory
  name is the style id. `RoadCatalogue` is keyed by `(style, RoadPieceKind)`;
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
  the fixed filenames the loader expects.
- **`road_build`** — drag-to-build, gated by `city::tool::ActiveTool`
  (`Building | Road | Terraform`, `T` cycles; placement/commit no-op unless
  `Building`). A drag is an **L-shaped `drag_path`** (row then column) so every
  cell stays a cardinal neighbour, which `select_piece` assumes. Preview uses
  a rotated catalogue piece where one exists, a flat translucent quad
  otherwise. Commit validates the whole path first, claims every cell in
  `City` synchronously, then batches **one merged `WorldEdit`** across the path
  plus any already-road neighbour needing re-tiling. **Road cells are not
  journaled** — no undo or demolish for them yet.
- **Connectivity queries** — four functions built entirely on
  `reachable_from`/`is_connected` (no second BFS).
  `touching_road_cells(city, building)` bridges a footprint to adjacent road
  cells via `state::cell_of`; `is_building_connected`/`buildings_connected`
  return `Option<bool>` (`None` = not currently placed). All still
  `#[allow(dead_code)]` — substrate for later logistics or an
  unconnected-building warning.

## G — UI (`city::ui`)

`UiPlugin` registers its **own** `EguiPlugin`, separate from the viewer's —
this game doesn't use `crate::selection`-based panels. Three windows:

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

**Which save it opens** (ticket 064): two positional CLI arguments, parsed in
the crate root (`saves_directory_from`/`pick_named_save`) so `block_viewer`
honours them as its startup default too. `argv[1]` keeps ticket 008's meaning
(which saves *directory* to scan) but now treats an empty value as unset;
`argv[2]` picks a save within it by name (exact, then case-insensitive), or
is taken as the save directory itself when it names an existing one. An
unmatched name is a ticket-008-style startup issue listing the names that do
exist, not a silent fall back to the first save. The city panel's **World**
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

---

# Remaining plan

## H2 — Yields (open)

Dug blocks become resource counts in city state. Inert in iteration 1, like
the production fields. `city::terraform`'s dig currently records nothing
anywhere production could later read. Likely shares plumbing with the noted-
but-unscheduled idea of **charging build cost/time for blocks a placement
clears** (instead of the old refusal); worth picking up together.

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
- **Production simulation, resources, spending.** The definition fields, H2
  and I4's malus carry the data; nothing consumes it.
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

**M7 — "the world answers back"**: I2–I7 plus `ranvil` 020. Break a wall in
Minecraft, come back, and the building says so. The first mechanic that makes
the round trip *matter* rather than just work, and the right first target for
iteration 2 — ahead of production, because it needs no economy to be
interesting. Note that damage *detection* needs no write path at all (it's
pure reading on `blueprint::extract`); only repair does.

Everything after that is production, resources, and the tier tree coming
alive.
