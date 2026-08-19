# Roadmap — from "explore a save" to "build a city in one"

Not a work item; the plan for the citybuilder game and the shared world-edit
infrastructure it needs. High-level tasks here get split into numbered
tickets in this directory when they're picked up (next free number: 061).
Companion to `ROADMAP.md`, which covers the viewer 001–029.

## The goal

A citybuilder played on a **real Minecraft save**. Buildings are blueprints
extracted with the existing 019–024 mechanism; placing one writes its blocks
into the save's region files, so you can close the game, open the world in
Minecraft, and walk the city you built.

And the trip runs both ways: blocks a player breaks in Minecraft are damage
to the building they belonged to, dropping its health and its output until
it's repaired (group I). The world isn't an export target, it's the other
half of the game state.

Iteration 1: place a handful of non-functional buildings and build streets.
No production, no resources, no spending. But the blocks land in the world
for real, and the definitions are written in a format that has room for
production rates and an Anno-style tier tree from day one.

## Two decisions already made

**In-crate, `lib.rs` + two binary shims.** One package, one lib, two
`[[bin]]` targets that are three lines each. Both games live *inside* the
lib, so `pub(crate)` keeps working across the shared modules and no
visibility churn is needed now. The workspace split (`mc_core` /
`block_viewer` / `citybuilder`) stays available as a later move, once the
shared core stops changing shape.

```
src/
  lib.rs              pub mod world, blueprint, edit, sky, camera,
                      region_cache, streaming, chunk_pipeline, unload,
                      viewer, city
  world/              unchanged — decode, mesh, atlas, tint, biome, block
  blueprint/          + structure reader, + blueprint mesher
  edit/               NEW, shared: the write path
  viewer/             today's main.rs body, selection/, ui/
  city/               NEW: the game
  bin/block_viewer.rs fn main() { block_viewer::viewer::run() }
  bin/citybuilder.rs  fn main() { block_viewer::city::run() }
```

**The map is a real save, and edits are written back.** Which makes the
write path the spine of the whole project rather than a feature at the end.

## The architectural rule everything else follows from

> **The city state is authoritative. The blocks in the world are a
> projection of it.**

The game owns a list of placed buildings and roads — ids, positions,
rotations, tiers, whatever production state later iterations add — and saves
that alongside the world. The blocks written into the region files are the
*rendering* of that list, not the source of truth.

This is worth stating up front because six later tasks depend on it:

- **Undo/demolish** is a city-state edit plus a re-projection, not an
  attempt to remember what blocks used to be there.
- **Reconciliation** becomes possible: recompute what the world *should*
  look like from the city list, diff against what's actually in the region
  files, write the delta. That's also the repair path when someone edits the
  world in Minecraft between sessions.
- **Chunk and region boundaries stop being a placement problem.** A building
  is one entry in the city list regardless of how many chunks its blocks
  land in; only the write path deals with the split.
- Corrupting a save is recoverable — the city list plus the original terrain
  backup rebuilds it.
- **Damage (group I) is that same diff read as a game signal.** Intent minus
  reality is an error when we caused it and a gameplay event when the player
  did. One mechanism, two readings — which is why the mechanic costs so
  little to add and why it belongs in the design rather than bolted on.

## What the code can and can't do today

Reusable as-is: `world::{atlas, block, biome, tint, decode}`, the quad/UV/
tint emission in `world::mesh`, `blueprint::{Blueprint, BlockState}`,
`region_cache`, `chunk_pipeline`/`streaming`/`unload`, `sky`, `camera`'s rig
and ray-march.

Gaps found while planning, each of which is real work:

| Gap | Where |
|---|---|
| ~~`ranvil` cannot write~~ — done: sector allocation, header, zlib framing and an atomic replace (009), with dirty tracking (010) | upstream, `ranvil` 009–010 |
| ~~`rnbt` has almost no mutation API~~ — done: `get_mut`, `insert`, `remove`, `as_*_mut` | upstream `../rnbt` (no ticket system there; W1) |
| ~~No `block_states` **encoder**~~ — done: `set_blocks` packs, widens and re-packs a whole section per batch | upstream, `ranvil` 011 |
| ~~Nothing recomputes `Heightmaps`, clears `isLightOn`, or cleans up `block_entities` after an edit~~ — all three now exist upstream (013, 014, 015); sequencing them is W4's job | upstream, `ranvil` 013–015 |
| `blueprint::structure` only **writes** `.nbt`. No reader | `blueprint/structure.rs` |
| `mesh_chunk_column` meshes a `ChunkColumn` via `BlockId`; a `Blueprint` is a `BlockState` palette + `Vec<u16>` with no registry | `world/mesh.rs` |
| ~~`RegionCache::get_or_load` hands out `&ChunkRegion`. No mutation, no invalidation~~ — done: `get_or_load_mut`, an eviction guard for dirty regions, `discard` and `dirty_regions` (032) | `region_cache.rs` |

## The shape of the work

```
L1  lib.rs + two bin shims                       <- DONE (ticket 027)
     |
     +-- W  the write path (shared infrastructure)
     |    W1  rnbt: mutation API             upstream ../rnbt      DONE
     |    W2  ranvil: the whole Anvil write  upstream ../ranvil    DONE (009-017)
     |    W3  bulk encode / section batching  DONE (ranvil's set_blocks)
     |         |
     |    W4  the chunk edit model            <- DONE (ticket 031)
     |         |
     |    W5  boundary routing + region batching  <- DONE (ticket 032)
     |    W6  write safety: lock, backup, atomic  <- DONE (ticket 033)
     |    W7  live re-mesh: edits mark chunks dirty  <- DONE (ticket 034)
     |    W8  viewer: a paint/fill command proving W1-W7  <- DONE (ticket 035)
     |
     +-- B  blueprints as building models
     |    B1  structure .nbt reader (inverse of 023)  <- DONE (ticket 036)
     |    B2  Blueprint -> Bevy mesh                  <- DONE (ticket 037)
     |    B3  rotation, incl. block-state properties  <- DONE (ticket 038)
     |    B4  the building asset catalogue             <- DONE (ticket 039)
     |
     +-- C  definitions & scripting
     |    C1  building definition schema + loader  <- DONE (ticket 040)
     |    C2  tiers and tech tree                  <- DONE (ticket 041)
     |    C3  production fields (data only in iteration 1)
     |    C4  hot reload + a definition-error panel
     |
     +-- D  city state (authoritative)
          D1  the City resource: buildings, footprints, occupancy  <- DONE (ticket 042)
          D2  city save/load next to the world  <- DONE (ticket 043)
          D3  journal, undo, world reconciliation  <- DONE (ticket 044)
               |
               +-- E  placement          +-- F  streets
               |    E1  RTS camera, picking    <- DONE (ticket 045)
               |                                F1  road graph on the grid
               |    E2  grid + footprint fit        <- DONE (053, revised 054)
               |         <- DONE (ticket 046)   F1b road cells + piece
               |    E3  ghost + validity           selection <- DONE (054)
               |         <- DONE (ticket 047)   F2  drag-to-build, wired to
               |    E4  commit                      F3's mesh+write
               |         <- DONE (ticket 048)        <- DONE (ticket 055)
               |    E5  demolish                 F3  auto-tiling the pieces
               |         <- DONE (ticket 049)        <- DONE (054 selection,
               |                                      055 mesh+write; real
               |                                      .nbt pieces still
               |                                      absent)
               |                                 F4  connectivity queries
               |                                      <- DONE (ticket 056)
               |
               +-- G  UI: build menu, city panel   <- DONE (ticket 050)
               +-- H  terraforming: dig and level
               |    H1  level and dig tools        <- DONE (ticket 057)
               |    H2  yields (inert)
               |    (H's digging is what makes R1's floor move — ticket 030)
               |
               +-- I  damage: the world diffing back
                    I1  the as-built baseline        <- ITERATION 1
                    I2  the scan: baseline vs world
                    I3  the comparison policy        <- the hard one
                    I4  health, thresholds, malus
                    I5  scan scheduling + timestamp gate
                    I6  repair
                    I7  display: per-building and city-wide

R  the city view's render depth   (independent of everything above)
     R1  don't mesh below the terrain surface   <- DONE (ticket 030)
```

---

## L — Layout

**L1. `lib.rs` and two binary shims. — done, ticket 027.** Mechanical:
`src/lib.rs` declares the module tree, `main.rs`'s body moved to
`viewer::run()`, two `[[bin]]` shims added. `ui` moved under `viewer`;
**`selection` stayed at the crate root** — `blueprint::extract` takes its
`SelectionBounds`, and I2's damage scan will read it too. Shared startup
lives in `lib.rs::world_app()`, so `city::run()` is not a copy of
`viewer::run()`.

---

## W — The write path

The largest and most valuable group, and the one that isn't really about the
citybuilder at all: it's what turns `block_viewer` into the general
explore-*and-modify* app. Every task here is shared.

**W1. `rnbt`: a mutation API.** `get_mut`, `set`/`insert`, `remove`, and
list mutation on `NbtField`/`NbtValue`. Today's accessors are all `&self`.
Expect to fix things — 023 was the first code in this repo ever to call
`write_nbt` and found bugs; this is the second such expedition.

**W2. `ranvil`: a region file writer, and everything under it.** The `.mca`
format in reverse, plus the chunk-NBT surgery that goes with it. **Already
written up as tickets in `../ranvil/tickets/` — 009 through 019** — because
it's that crate's subject matter, not ours:

| | |
|---|---|
| 009 | region file writer: sector table, timestamps, zlib, atomic replace |
| 010 | mutable chunk access + dirty tracking on `ChunkRegion` |
| 011 | `set_block`: palette insertion and `block_states` re-packing |
| 012 | creating sections that don't exist yet (building above terrain) |
| 013 | `Heightmaps` pack/unpack and recompute — **done**; reading them is what a render-depth cutoff would use |
| 014 | relight-on-load (`isLightOn`) — **done**; the game does the lighting |
| 015 | removing orphaned `block_entities` on overwrite |
| 016 | `session.lock` detection: is the world open right now |
| 017 | oversized chunks (`.mcc`) — **read done**; writing them is ranvil 027 |
| 018 | compression types beyond zlib (robustness, low priority) |
| 019 | stale fixtures README (cleanup found while planning) |
| 020 | expose per-chunk timestamps — the change-detection gate for I5 |
| 027 | writing an oversized chunk back out (fallout of 017) |

The decision worth repeating here because it shapes W5: 009 **rewrites the
whole region file** rather than allocating freed sectors in place. A chunk
grows when a placement adds a palette entry, so everything after it moves
anyway. The cost is that a single-block edit rewrites 512×512 blocks' worth
of file — which is exactly why W5 batches by region.

**W3. Encoding lives upstream; the `block_viewer` side is the bulk path.**
The palette/bit-width encoder is ranvil 011, next to its mirror image
`get_block`. What stays here is `world::decode`'s *bulk* decode for meshing
and whatever the edit model needs to hand ranvil section-sized changes
rather than 4000 individual `set_block` calls — see 011's note on a batch
entry point. The rules either side must agree on (4-bit floor, indices never
spanning longs, single-entry palettes omitting `data`) are written out in
011; ticket 001 was a bug in that same arithmetic in the read direction, so
the round-trip property test comes before the encoder.

**W4. The chunk edit model — the keystone. — done, ticket 031.** Given ranvil
provides the primitives, this is the *policy* layer: what a well-formed edit
is, and in what order the pieces happen. The split is deliberate — ranvil owns
the Anvil format, `edit` owns what we're allowed to do to a world.

Sequencing: block changes (011/012) → block-entity cleanup (015) →
heightmap invalidation (013) → relight flag (014) → write (009), with two of
those (block-entity cleanup, `isLightOn`) folded into ranvil's `set_blocks`
itself now. `Status` gate: only edit chunks that are fully generated
(`minecraft:full`). `DataVersion` mismatch between blueprint and save
refuses/warns loudly.

*How it came out:* `edit` module. Order is preflight → `set_blocks` → heightmap
invalidation, once per chunk at the *end of the transaction* rather than once
per `set_blocks` call. Coordinate rules (`bevy.z = -mc.z`, inclusive bounds)
inherited from ticket 019, fixed here once for every later caller.

**W5. Boundary routing and region batching. — done, ticket 032.** Block
coordinate → `(region, chunk, section, local index)`, batched by region file
(a 20×20 building can straddle up to 4 chunks, or 4 region files near a
corner); refuse (not panic) any placement reaching an ungenerated chunk;
`RegionCache` needs a mutable, invalidating path.

*How it came out:* `edit::route` splits by region and **plans every region
before applying any** — neither `set_blocks` nor W4's `apply` gives
all-or-nothing across files, so a write-phase failure discards the
already-applied regions from the cache (full rollback, since nothing saves
yet). A region carrying an earlier transaction's unsaved changes is refused
(rollback would take those too) — one transaction at a time, saved before the
next (W6's contract). Invalidation was the small half: the cache holds the
mutated region, so reads already see post-edit blocks; what goes stale is
`DecodedWorld`/meshes, off `EditReport::chunks` (W7).

**W6. Write safety. — done, ticket 033.** Refuse to write to a world
Minecraft has open (ranvil 016's `session.lock`); back up before the first
write of a session, per region file; write atomically (temp/fsync/rename);
support a dry run.

*How it came out:* `edit::session::WriteSession` **holds** the session lock
(`SessionLock::acquire`, released on drop) rather than probing it — a probe
can't close the check/write race, and holding keeps the game out mid-edit.
Scoped to the write session, not the app's lifetime. `commit` backs up
*every* touched file before saving *any* of them, so a backup failure leaves
the save untouched (rollback via W5's discard); a failure during saves can't
be rolled back and is reported via `WriteError::WriteFailed` (names what
was/wasn't written and the backup dir) — those regions stay dirty rather than
discarded, since a disk error is retryable.

**W7. Live re-mesh.** An edit marks its chunks dirty; the pipeline re-meshes
them, including the neighbours whose faces the edit exposed — 005-f already
solved exactly this problem for the loading frontier, so follow it rather
than inventing a second dirty-chunk mechanism.

**W8. A paint/fill command in the viewer. — done, ticket 035.** Fill the
current 019 selection with a block, or stamp a loaded blueprint at it. This
exists to *prove W1–W7 end to end before any game code is written*, and it's
directly the first feature of the general explore-and-modify app. Manual
verification (write, open in Minecraft, confirm blocks/lighting/no
corruption) is the gate the rest of the roadmap waits on.

*How it came out:* fill only, not blueprint stamping (no reader yet — B1).
`WorldEdit::fill(bounds, state)` (shared with H1), `BlockState: FromStr`
(inverse of 022's `Display`), `viewer::paint` module holding the region-cache
lock for the whole commit (can't release partway without breaking W5's
all-or-nothing). Fires `chunk_pipeline::ChunksEdited` so W7's re-mesh shows
results live. Manual open-in-Minecraft check recorded in `../todo.md`, not
done here.

---

## B — Blueprints as building models

**B1. Structure `.nbt` reader. — done, ticket 036.** The inverse of 023:
vanilla structure file → `Blueprint`. Round-trip test against 023's writer.
Lets buildings be authored in Minecraft itself.

*How it came out:* `blueprint::structure::{read_structure, read_structure_file}`,
mirroring the writer's `Read`/path-taking split, reusing
`BlockState::from_palette_entry`. Stricter than the format requires:
`blocks` must be dense (no gaps/dupes) — a gap is a malformed file, not
implicit air. `Blueprint::origin` comes back `IVec3::ZERO`, `failed_columns`
always `0`. No caller yet.

**B2. `Blueprint` → Bevy `Mesh`. — done, ticket 037.** Not the same entry
point as `mesh_chunk_column`: a blueprint has a `BlockState` palette with
properties and no `BlockRegistry`. Faces at the blueprint's outer boundary
are always emitted; air in the palette stays air.

*How it came out:* `blueprint::mesh_blueprint` reuses `world::atlas::resolve_faces`
/ `world::tint::resolve_block_tint` directly (both already name-keyed) — only
visibility widened to `pub(crate)`, no duplicated logic. Every biome-tinted
block resolves against one caller-supplied `BiomeColors` (no per-block biome
on a `Blueprint`). Mesh coordinates run `0..size` on all axes — a blueprint
has no absolute world height until placed. No caller yet.

**B3. Rotation. — done, ticket 038.** 90/180/270 about Y. The mesh part is a
transform; the blocks part rewrites `facing`/`axis`/`hinge`/etc. in the
palette, or a rotated building is visibly wrong once written to the world.

*How it came out:* `blueprint::rotate::rotate_blueprint(&Blueprint, Rotation)
-> Result<Blueprint, RotationError>`. `facing`, `axis`, 16-way `rotation`,
and the N/S/E/W connection-key group all rewrite; `shape` disambiguates by
*value* not by block (stairs vs. rails share the property name with disjoint
value sets). `hinge` and a chest's `type` are pass-through (relative to the
block's own `facing`, already correct once `facing` rotates). Anything else →
`RotationError::UnrotatableProperty`, except `Deg0` (identity, never inspects
the palette). No caller yet.

**B4. The building asset catalogue. — done, ticket 039.**
`assets/city/blueprints/*.nbt`, loaded and validated at startup: size
limits, palette sanity, footprint derived from the blueprint's own
dimensions.

*How it came out:* `blueprint::catalogue::load_catalogue_dir(&Path) ->
(BuildingCatalogue, Vec<(PathBuf, CatalogueError)>)`, non-recursive scan of
`*.nbt` (case-insensitive), validated against B1's own size limit and a
non-air-only palette check. Failure is per-file — a missing dir is an empty
catalogue (logged, not fatal), a bad file is skipped. `city::run()` loads
`assets/city/blueprints` into the `BuildingCatalogue` resource;
`assets/city/blueprints/house01.nbt` is the first fixture. Id collisions
(`DuplicateId`) can only be constructed directly in tests — Windows' default
case-insensitive filesystem already prevents two same-stem files coexisting.

---

## C — Definitions and scripting

**Data files now, not a scripting language.** Iteration 1's values are
numbers and references — production rate, inputs, outputs, tier, unlock
dependencies, blueprint file, cost. RON (serde, comments, real enums, no
whitespace significance) is the format. Reach for Rhai or Lua only when a
*behaviour* needs to vary per building rather than a number.

**C1. Building definition schema and loader. — done, ticket 040.** One file
per building. Sketch:

```ron
Building(
    id: "lumberjack",
    name: "Lumberjack's Hut",
    blueprint: "lumberjack.nbt",
    tier: 1,
    requires: [],                       // other building ids
    footprint: FromBlueprint,           // or explicit (x, z)
    // iteration 1 parses and displays these; nothing simulates them yet
    production: Some(Production(
        outputs: [(item: "wood", per_minute: 4.0)],
        inputs:  [],
        radius:  Some(24),
    )),
    cost: [(block: "minecraft:oak_planks", count: 40)],
    // group I. Per-building because a warehouse should shrug off what
    // wrecks a lighthouse.
    integrity: Integrity(
        pristine_above: 0.95,   // full output at or above this health
        ruined_below:   0.60,   // non-functional below it
        // between the two, output scales linearly. Named curves later.
    ),
)
```

*How it came out:* `city::definition::{Building, load_definitions_dir}`, one
`.ron` per building under `assets/city/buildings`, validated the same way B4
validates blueprints. Two deviations from the sketch: no inline `id` — the
filename stem is the id (same call as B4); `blueprint`'s stem must resolve
to a real catalogue entry or the definition is rejected
(`DefinitionError::UnknownBlueprint`). `requires` is parsed but deliberately
unvalidated — C2's job.

**C2. Tiers and the tech tree. — done, ticket 041.** Anno-style: a `tier`
per building, plus `requires` edges. Needs cycle detection and
dangling-reference checks at load — a cycle is unwinnable and fails silently
("button greyed out forever") if uncaught.

*How it came out:* `city::definition::resolve_requirements` — a loop (not a
single pass) alternating a dangling-reference check with DFS cycle detection
(`find_cycle`) until a pass removes nothing; removing one bad entry can
dangle another. Result is the maximal subset of loaded buildings whose
`requires` graph resolves with no cycle, order-independent. New errors:
`DanglingRequirement`, `CyclicRequirement` (carries the whole loop). No
consumer yet — G1's build menu is the eventual reader.

**C3. Production fields, parsed but inert.** Iteration 1 shows rates and
costs in the build menu and does not simulate them. The point is that the
schema is exercised — a schema that nothing reads drifts from reality.

**C4. Hot reload and an error panel.** Definition files reload on change;
errors go to an egui panel rather than a panic or a console line nobody
sees. This is what makes balancing tolerable later.

---

## D — City state

**D1. The `City` resource. — done, ticket 042.** Placed buildings (id,
origin, rotation), roads, and a footprint occupancy grid for fast
"is this tile free" queries. The authoritative state from the rule above.

*How it came out:* `city::state::City`, keyed by `BuildingId` (distinct from
a building's definition id). `HashMap<IVec2, Occupant>` occupancy grid
(`Occupant::Building(BuildingId) | Road`). `City::place_building` plans the
full footprint's tiles, checks all are free, then mutates — same
"plan-before-apply" shape as `edit::route` (W5). Rotation swaps the occupied
rectangle's (x,z) extent via `footprint_extent`/`footprint_tiles` — the
horizontal counterpart of B3's block-grid axis swap. `city::run()` inserts an
empty `City`; no consumer yet.

**D2. City save/load. — done, ticket 043.** RON next to the world —
`<save>/citybuilder/city.ron`. Versioned from the first write.

*How it came out:* `city::persistence::{save_city, load_city}`, storing
`CitySave`/`SavedBuilding` — occupancy is derived, not stored, so load
rebuilds it through `City::insert_loaded`/`add_road` (a corrupt/overlapping
file fails with `PersistenceError::Corrupt` rather than producing an
inconsistent `City`). `next_id` is persisted verbatim (not recomputed) so a
removed id is never reissued. `blueprint::rotate::Rotation` got
`Serialize`/`Deserialize` directly. `city::run()` skips persistence for
ticket 008's `empty_save` placeholder; otherwise loads synchronously before
`App::run()`, saves on `AppExit`.

**D3. Journal, undo, reconciliation. — done, ticket 044.** Every placement
and demolition as an appended entry, carrying the **as-built baseline**
(I1). Gives undo for free, demolish's terrain restore (E5), a rebuild path
when the world's blocks and city list disagree, and what the damage
mechanic (I) diffs against — one record, four consumers.

*How it came out:* `city::journal::{Baseline, JournalEntry, Journal,
reconcile, repair_edit}`. `Baseline::capture(edit, report)` reuses W4's
`EditPolicy::capture_replaced` (what an edit overwrote) rather than
computing anything new. `JournalEntry` snapshots a full `PlacedBuilding` per
entry (not a lookup key) since undo removes the building from `City` before
the entry is read. `Journal::undo_last` is all-or-nothing against
journal+`City` (`UndoError::Occupied` on conflict) but returns the restoring
`WorldEdit` **uncommitted** — the same "apply doesn't save" contract
everywhere else. `reconcile` recomputes expected world state from each
building's own placement baseline (not blueprint re-derivation), grouped by
region like `edit::route`. Persistence mirrors D2's shape;
`blueprint::BlockState` got `Serialize`/`Deserialize` directly. No caller for
`record_placement`/`record_demolition`/`undo_last`/`reconcile` until E4/E5.

---

## E — Placement

**E1. RTS camera and picking. — done, ticket 045.** Pan/zoom/rotate over the
terrain, plus screen ray → block coordinate.

*How it came out:* `camera::CameraMode::Rts`, a third mode alongside
`Fly`/`Orbit` in `camera.rs` (kept in the same enum since `streaming`/
`unload`/`sky` all find "the camera" via `Query<&Transform, With<CameraRig>>`).
WASD pans on the yaw-relative ground plane, Q/E rotates yaw, scroll zooms,
right-drag free-looks (ungrabbed — left mouse is reserved for E3/E4
picking). `CameraStartMode` resource lets `city::run()` start in `Rts`.
Picking: `city::picking::HoveredBlock`, built on `camera::block_under_cursor`
(no second raycast). No consumer yet — E2/E3 are the eventual readers.

**E2. Grid and footprint fit. — done, ticket 046.** Tile grid, footprint
occupancy, and the terrain rules: sample heights under the footprint, decide
what to do about slope.

*How it came out:* `city::grid::fit_footprint(origin, footprint, rotation,
&DecodedWorld) -> FootprintFit`, sampling ground via `ChunkColumn::
topmost_non_air` against already-decoded `DecodedWorld.columns` (no I/O).
Resolved to **refuses** steep ground rather than auto-levelling — levelling
means writing blocks, which is W4/W5's job, kept out of a read-only check
paid for every frame. Ground reuses `world::is_solid`. No caller yet.

Ticket 052 fixed a gap: `is_solid` treated trees/fence posts like terrain, so
clutter on flat ground read as a cliff. `ground_height_at` now walks a
`city::grid`-local `is_ground`/`is_clutter_name` predicate off a new
`ChunkColumn::topmost_matching`.

**Ticket 058 removed the slope refusal entirely** — a hard 1-block step cap
restricted buildable ground far more than it was worth on a real, uneven
Minecraft world. `base_y` is still the footprint's lowest sampled point, but
it's now purely the *initial suggested* height (feeds
`PlacementSelection::y_offset`); nothing refuses a steep placement any more,
and H1's terraforming is no longer load-bearing for placement.
`FitError::TooSteep` is gone; `NotLoaded` is the only refusal left.

Noted, not scheduled: charging build cost/time for blocks a placement
clears, instead of refusing — same shape as H2's open "yields" task.

**E3. Ghost preview and validity. — done, ticket 047.** The B2 mesh at the
cursor with a translucent material, tinted by validity, snapped to the grid.

*How it came out:* `city::placement::resolve_placement` combines E2's
`fit_footprint` (terrain) and D1's `City::is_tile_free` (occupancy) into one
green/red signal. No G1 build menu yet, so `PlacementSelection`/
`cycle_selection` (number keys pick catalogue entry, R rotates, Escape
clears) stands in for it. Ghost mesh + two `AlphaMode::Blend, unlit: true`
materials are cached by `(catalogue id, Rotation)` — `Rotation` gained
`Hash`. `city::picking` gained a `PickingSet` ordering label. No commit yet —
E4 calls `City::place_building` off this same signal.

**E4. Commit. — done, ticket 048.** City state entry + W4/W5 write + W7
re-mesh, transactionally: a failed write means the city entry doesn't
survive either.

*How it came out:* `city::commit::CommitPlugin`. `try_commit_placement`
recomputes validity itself off *this* frame's inputs (doesn't trust the
ghost's last-frame result). `City::place_building` runs synchronously the
instant a click is accepted (claims the tile before the write starts); a
single `CommitState::pending` slot backpressures, like `PaintCommand`.
`blueprint_edit` writes every grid position including air — clears whatever
terrain E2's fit tolerance left poking into the building. On success:
`Baseline::capture` + `Journal::record_placement` (I1's baseline, for real)
and `ChunksEdited` fires. On failure: `City::remove_building` — the
transactional half. Also added: `PlacementSelection::y_offset` (Page
Up/Down, Home) to nudge height off the auto-fit before committing.

**E5. Demolish. — done, ticket 049.** City state removal, then re-project —
needs the pre-placement blocks kept in the journal (a D3 decision).

*How it came out:* `city::demolish::DemolishPlugin`. `Delete` on
`picking::HoveredBlock`'s tile is the whole UI (no G1 yet) —
`City::occupant_at` finds the id, `Journal::placement_baseline` finds what to
restore (refuses if none — an older-save edge case). Restoring edit reuses
D3's `Baseline::restore_edit` (widened to `pub(super)`). Ordering: commit
claims the tile *before* its write; demolish frees the tile only *after* its
restoring write succeeds — the opposite order, avoiding a race where a new
placement lands on the same tile mid-write. New `city::write_gate::WriteGate`
resource, shared by commit/demolish, covers two independent
`WriteSession::open` calls on unrelated tiles in the same frame — `ranvil`'s
session lock is per-file-handle, not per-process, so a second concurrent
open from this app would fail outright (no retry) rather than just delay.

**E4/E5/G2 addendum — ticket 051, "defer world writes to a manual Save."**
Manual testing showed every commit/demolish/undo opened its own
`WriteSession` and saved immediately — one full region-file rewrite per
building. `city::commit::apply_building_edit` now applies straight to the
shared `RegionCache` (`EditPolicy::allow_dirty_regions`) and stops — no
session, no disk write, no `WriteGate` (removed — nothing opens a per-edit
session any more). `city::save` is now the only place that reaches disk:
"Save world" in the city panel opens a real `WriteSession` and calls
`WriteSession::flush`. `AppExit` also flushes synchronously before
`city.ron`/`journal.ron` are written, so those files never describe
buildings the world never actually got.

---

## F — Streets

**F1. The road graph. — done, ticket 053; revised to cell space and given
F3's piece selection by ticket 054.** Tiles plus adjacency, on the same
grid as E2.

*How 053 came out:* `city::road` — `connections_at`/`reachable_from`/
`is_connected`, all recomputed off `&state::City`'s occupancy grid on every
call (nothing here can go stale). `Direction`'s four offsets follow
Minecraft's x/z convention (this module never touches a Bevy `Transform`).
`reachable_from` on a non-road start returns the empty set, not a
one-element set (needed by F4).

*How 054 revised it:* a road became a real `ROAD_CELL_SIZE` (6-block) cell,
not a 1×1 tile — a rotated straight piece can't stand in for a corner.
`City::add_road_cell`/`remove_road_cell`/`road_cells()`/`is_road_cell`
replaced the old tile-based API outright (no dual grids). `CitySave.roads` →
`road_cells`, save version bumped 1→2 (refuses an old save rather than
misplacing it). Also landed early: `road::RoadPieceKind` (`Isolated |
DeadEnd | Straight | Corner | T | Cross`) and `road::select_piece`, rotating
a canonical connection pattern until it matches a cell's actual connections.
`city::road_catalogue` loads six fixed `.nbt` files from **`assets/city/roads`**
(`isolated.nbt`, `dead_end.nbt`, `straight.nbt`, `corner.nbt`, `t.nbt`,
`cross.nbt`), each required to be exactly `ROAD_CELL_SIZE` on x/z —
**no real pieces ship yet**, tested against synthetic fixtures only.

**Revised by ticket 059: multiple road styles.** The six-file layout above
was system-wide — one style, period. `assets/city/roads` is now a directory
*of styles*: `assets/city/roads/<style>/{isolated,dead_end,straight,corner,
t,cross}.nbt`, one subdirectory per style, its name serving as the style id
the same way a building's filename stem already does. `RoadCatalogue` is
keyed by `(style, RoadPieceKind)`; a road cell records which style it was
built as directly on `state::City` (`add_road_cell`'s own argument,
`road_style_at`), not carried around separately — the city-state-is-
authoritative rule extended one field further. Connectivity and shape
selection (`road::connections_at`/`select_piece`) stay entirely style-blind;
style only picks which `.nbt` gets meshed/written once the shape is already
chosen, read back off `City` at write time. `road_build::RoadStyleSelection`
(`[`/`]`, gated on the road tool, auto-picking the first loaded style) is
the keyboard stand-in for choosing a style, the same role ticket 047's
number keys play for buildings. `CitySave.road_cells` gained a `style`
field, version bumped 2→3. Still no real `.nbt` pieces for any style — see
`finished_tickets/059-multiple-road-styles.md`.

**Revised by ticket 060: a road type schema.** 059's `RoadCatalogue` is
geometry only; a style has no *properties*. `city::road_definition` adds the
missing half, mirroring buildings' own geometry/data split
(`blueprint::catalogue::BuildingCatalogue` vs `city::definition::BuildingDefinitions`):
one RON file per style under `assets/city/road_types` (new directory,
alongside `assets/city/roads` the way `assets/city/buildings` sits alongside
`assets/city/blueprints`), carrying `name`, `travel_speed` and `capacity`.
The filename stem **is** the style id — no separate field to name it, since
`RoadCatalogue` is already keyed by that same name, unlike a building's
`.ron`/`.nbt` pair which need one. Validated against the geometry catalogue
(`RoadDefinitionError::UnknownStyle` for a type file naming a style with no
`.nbt` pieces) the same way a building definition's `blueprint` reference is
checked. **Schema only, inert** — per the user's own framing and this
project's C3 precedent: nothing simulates travel speed or capacity: there is
no traffic/logistics system anywhere in this codebase to hang a real
simulation off, and building one is a project of its own. `city::run()`
loads and logs a `RoadTypes` resource the same way it does
`BuildingDefinitions`; `RoadStyleSelection`'s `[`/`]` cycle keeps reading
geometry (`RoadCatalogue::styles()`), not this — placing a road only needs
its shape, not its (still-unused) numbers. No consumer yet — same
"proven, not yet used" state every other data-only resource in this crate
has landed in. See `finished_tickets/060-road-type-schema.md`.

**F2. Drag-to-build. — done, ticket 055.** Click-drag from A to B, routed
over the grid, with a live preview of the cells it would claim and their
cost.

**F3. Auto-tiling. — selection done, ticket 054; mesh+write done, ticket
055.** Each cell's `select_piece` answer is the kind/rotation to render,
re-picked when a neighbour changes.

*How 055 came out:* `city::road_build`, gated by `city::tool::ActiveTool`
(`Building | Road`, `T` switches — placement/commit no-op unless `Building`).
A drag is an L-shaped `drag_path` (row then column) so every cell stays a
cardinal neighbour, which `select_piece` assumes. Preview uses a rotated
catalogue piece where one exists, a flat translucent quad otherwise (still
no real `.nbt` pieces). Commit validates the whole path first, claims every
cell in `City` synchronously, then batches one merged `WorldEdit` across the
path plus any already-road neighbour that needs re-tiling. Not journaled
yet — no undo/demolish for road cells.

**F4. Connectivity queries. — done, ticket 056.** "Is this building on the
road network", "what does this road segment reach". No consumer in
iteration 1 — substrate for later logistics.

*How it came out:* four functions in `city::road`, built entirely on
`reachable_from`/`is_connected` (no second BFS). `touching_road_cells(city,
building)` bridges a building's footprint to adjacent road cells via a new
`state::cell_of` (`div_euclid`-correct for negative coords).
`is_building_connected`/`buildings_connected` return `Option<bool>` (`None` =
not currently placed). All new fns stay `#[allow(dead_code)]` — no caller
yet; a later unconnected-building warning or logistics feature is the
eventual reader.

---

## G — UI

**G1. Build menu. — done, ticket 050.** Catalogue grouped by tier, locked
entries visible but disabled, showing what unlocks them.

**G2. City panel. — done, ticket 050; write/save split by 051.** Building
counts, road length, write status (last write, dirty regions, backup
location).

*How it came out:* `city::ui::UiPlugin`, its own `EguiPlugin` registration
(separate from the viewer's — this game doesn't use `crate::selection`-based
panels). Build menu bridges `PlacementSelection::catalogue_id` against
`BuildingDefinitions` via a new `LoadedBuilding::catalogue_id` field.
"Unlocked" is defined for the first time: a `requires` id is met once at
least one building of that type has been placed (no separate "researched
techs" resource). City panel adds a `WriteStatus` resource (recorded by
commit/demolish, later road_build) and an Undo button via `city::undo`
(`Journal::undo_last`'s promised caller). See
`finished_tickets/050-build-menu-and-city-panel.md` for the
manual-verification checklist it left in `../todo.md`.

---

## H — Terraforming

**H1. Level and dig tools. — done, ticket 057.** Reuses W4/W5 wholesale —
same write path, different source of block changes.

*How it came out:* `city::tool::ActiveTool::Terraform`, a third mode (`T`
cycles three ways). `city::terraform` reuses `road_build`'s
click/hold/release drag shape, widened to a plain rectangle (`rect_tiles`) —
no piece catalogue or adjacency to keep straight. Dig clears the topmost
block (`ChunkColumn::topmost_non_air`, not the clutter-skipping `is_ground`
— a dig clears a tree same as stone) per tile. Level reads the drag's start
tile as target height, digs high tiles down / fills low ones up with fixed
`minecraft:dirt` (no material inventory yet). Neither reuses `WorldEdit::fill`
(both write a *different* value per position). Commits reuse
`city::commit::apply_building_edit`. No `City` entry, no journal record for
terrain edits — a failed write has nothing to roll back beyond its
`WriteStatus` line. No preview mesh, console-only feedback.

**H2. Yields.** Dug blocks become resource counts in city state. Inert in
iteration 1, like C3 — still open; `city::terraform`'s dig doesn't record
what it removed anywhere production could later read. Likely shares plumbing
with E2's "noted for later" placement-clearing-cost idea; worth picking up
together.

---

## I — Damage: the world diffing back

Blocks a player alters in Minecraft, inside a building's volume, count as
damage. Past a threshold the building stops working. This is the
architectural rule turned into gameplay: the plan already says the city
state is authoritative and the blocks are a projection of it, and already
needs a reconciliation pass (D3) that diffs intent against reality. Damage
is that same diff, read as a game signal instead of an error.

**I1. The as-built baseline. — do this in iteration 1.** At placement,
record what was actually written: positions, `BlockState` at each, and the
save's `DataVersion`/chunk timestamps at the time. **This is the one part of
group I that cannot wait** — a building placed without a baseline can never
be diffed later ("what it should look like" isn't recoverable after the
fact; re-deriving from the blueprint alone is wrong because the blueprint
has air in it, and E2's terrain fit may have adjusted the placement). Same
record E5 (demolish) and D3 (undo) already need.

**I2. The scan.** Per building, read its volume out of the save and diff
against the baseline. Reuse `blueprint::extract`'s existing bounds-walking
read path — no new walk needed. Two failure modes that must **not** read as
damage:

- **An unreadable/missing chunk** is *unknown*, not *destroyed* —
  `Blueprint::failed_columns` already distinguishes real failures from
  ungenerated chunks; inherit that.
- **A `DataVersion` change** — every block could mismatch after a Minecraft
  version bump, through no fault of the player. Refuse to scan and offer a
  re-baseline instead. Costs one comparison to prevent the single most
  damaging false positive available.

**I3. The comparison policy. — the hard one.** Naive `BlockState` equality
does not work. Classify per block:

- **Structural** (walls, floors, roof, stairs, logs) — full equality,
  including properties. This is the denominator.
- **Interaction state** (`open`, `powered`, `lit`, a lectern's `has_book`) —
  ignore. Opening a door is not vandalism.
- **Volatile** (leaves, grass/dirt spread, water/lava flow, snow, fire,
  crops, ice, copper oxidation) — exclude entirely; these change with no
  player involved.
- **Blueprint air** — a block placed there is *obstruction*, a different,
  later mechanic; excluded from the denominator but counted separately.

Build a per-block-name table once against the registry, the same pattern
`world::tint`'s `build_block_tint_table` establishes. When unsure, classify
as volatile — under-reporting damage is a mild disappointment; a false
positive that ruins a city because it snowed is a bug report and a reload
from backup.

**I4. Health, thresholds, malus.**
`health = 1 - damaged_weight / structural_weight`, uniform weights to start
but written as weighted from the outset. State: `Pristine | Damaged(health)
| Ruined | Unknown`. Thresholds/malus curve come from C1's `integrity`
block, per-building. Production is inert until iteration 2, so malus is a
displayed number first. **The scan is stateless** — rebuilding a wall by
hand heals the building; damage describes the world right now, not an
accumulated counter.

**I5. Scan scheduling and the timestamp gate.** Gate on a region file's
per-chunk last-written timestamp — if nothing under a building has been
rewritten since the last scan, nothing changed. Not exposed by `ranvil`
today (`ranvil` ticket 020). Compare against the baseline's own recorded
timestamp, or our own writes make every building look changed the instant
it's placed. Schedule: full scan on load behind the gate on the async task
pool; re-scan on chunk re-read; on demand on inspect. Never on the main
thread.

**I6. Repair.** Re-project the baseline onto the volume through W4/W5.
Almost free once the write path exists. Costs materials in iteration 2;
free (a button) in iteration 1.

**I7. Display.** Per-building: health percent, state, output multiplier, in
an inspect panel off E1's picking. City-wide: counts by state in the G2
panel. In-world at-a-glance (a mesh tint via the vertex colour channel, or a
gizmo/icon reusing `selection::gizmo`) beats making the player click to find
out.

---

## R — The city view's render depth

**R1. Don't mesh below the terrain surface. — done, ticket 030.** The
citybuilder's camera looks at the surface from above and never goes under
it, so the ~7 sections per chunk column below the terrain were decoded,
meshed and drawn for nothing.

*How it came out:* a per-chunk floor from `min(OCEAN_FLOOR)` over the
chunk's 256 heightmap columns (`ranvil` 013) cuts them. Minimum (not
average) is what keeps it safe — a ravine or cave mouth anywhere in the
chunk drags the floor down with it, so the cutoff never slices into a hole
you can see. `block_viewer` still renders everything; the floor is switched
on by `city::run()` only. H (digging) is what later moves the floor — a
re-*decode*, not a re-mesh, since blocks under the floor were never decoded
to begin with.

---

## Ordering advice

- **L1 first, alone.** It's mechanical and it touches everything.
- **W is the critical path, and W8 is the gate.** Nothing in D/E/F is worth
  building until blocks provably land in a world Minecraft opens without
  complaint.
- **Lighting is Minecraft's job — decided, not deferred.** This project
  never computes light. `set_blocks` clears the chunk's `isLightOn` byte
  (ranvil 014) and the game relights when it gets round to it; a building
  wrongly lit until something touches its chunk is an accepted cost. The
  flag is confirmed present as a root `TAG_Byte` on the real save
  (`DataVersion` 4438) and cleared on every edit; whether the game honours it
  is still worth ten minutes with the world open (manual check in
  `../todo.md`) — if it doesn't, the fallback is deleting the affected
  sections' `BlockLight`/`SkyLight` arrays, not writing a lighting engine.
- **ranvil 011's round-trip test before its encoder.** Ticket 001 was this
  exact arithmetic, in the other direction, and it shipped broken upstream.
- **`Heightmaps` follow lighting: let the game rebuild them** — still the
  default. W4 deletes the compound on an edited chunk
  (`ChunkRegion::remove_heightmaps`) rather than recomputing it; the in-game
  check (grass/snow/rain landing correctly, mobs not spawning on lit ground)
  is worth doing alongside the relight check. `ranvil` 013's **read** half is
  useful regardless — chunk-root heightmaps say how deep terrain goes before
  a section is decoded, which a render-depth cutoff keys off (see R1).
- **B and C can run in parallel with W** — different files, no shared types
  beyond `Blueprint`.
- **D1 before E and F, always.** Both write to city state; if they land
  first they'll each invent their own.
- **F is more independent than it looks.** Roads need D1 and the write path,
  not placement.
- **I1 ships with E4, in iteration 1.** Three fields on a journal entry
  written at placement time, unrecoverable afterwards. The rest of group I
  can wait indefinitely; this cannot.
- **I2 needs no write path.** Damage *detection* is pure reading, on
  machinery that already exists (`blueprint::extract`). Only repair (I6)
  needs W — group I can land early if it's the fun part.
- **I3 is the ticket to spend real time on.** Everything else in the group
  is plumbing; the classification table is the mechanic. Budget for tuning
  it against a real world.

## Deliberately not in this iteration

- **Terrain generation.** Placements reaching ungenerated chunks are refused
  (W5), not generated.
- **Lighting computation of any kind.** We clear `isLightOn`, never compute
  a light level; stale light is accepted (see ordering advice for fallbacks).
- **Writing block entities.** Blueprints with chests/signs place their
  blocks; the entities are dropped, with a warning. Removing *existing* ones
  we overwrite is in scope (corruption avoidance, not a feature).
- **Entities and mobs.** Structure files can carry them. Ignored.
- **Production simulation, resources, spending.** C3, H2 and I4's malus
  carry the data; nothing consumes it.
- **Damage detection itself (I2–I7).** Iteration 2 — but **I1's baseline is
  in iteration 1**, since it can't be added retroactively.
- **Obstruction as a distinct mechanic** (blocks placed in a building's
  declared air). I3 counts it separately; nothing reads it yet.
- **Multiplayer, or any concurrent access to the save.** W6 refuses instead.

## Suggested milestones

1. **M1 — "two binaries"**: L1. The citybuilder binary opens a window on a
   real save with the existing streaming and camera. Nothing else.
2. **M2 — "the world can be changed"**: W1–W8. Fill a selection in the
   viewer, open the world in Minecraft, walk into the change. The single
   highest-value milestone here, and the one that stands on its own even if
   the game never ships.
3. **M3 — "a building has a shape"**: B1–B4, C1. A blueprint loads,
   meshes, and shows in a catalogue.
4. **M4 — "a city exists"**: D1, D2, E1–E4, **I1**. Place buildings; they
   persist, they're in the world, and each one records what it built.
5. **M5 — "streets"**: F1–F3, G1. The iteration-1 deliverable.
6. **M6 — "and it's mine"**: E5, D3, G2, H1. Demolish, undo, terraform. —
   done (ticket 057 was the last piece).
7. **M7 — "the world answers back"**: I2–I7, plus `ranvil` 020. Break a
   wall in Minecraft, come back, and the building says so. The first
   mechanic that makes the round trip *matter* rather than just work — a
   good first target for iteration 2, ahead of production, because it needs
   no economy to be interesting.

Everything after that is production, resources, and the tier tree coming
alive.
