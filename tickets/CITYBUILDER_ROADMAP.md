# Roadmap — from "explore a save" to "build a city in one"

Not a work item; the plan for the citybuilder game and the shared world-edit
infrastructure it needs. High-level tasks here get split into numbered
tickets in this directory when they're picked up (next free number: 058).
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

**L1. `lib.rs` and two binary shims.** — **done, ticket 027.** Mechanical:
add `src/lib.rs` declaring the module tree, move `main.rs`'s body to
`viewer::run()`, add the two shims to `Cargo.toml`. `pub(crate)` stays as-is
throughout. The one judgement call is which modules the citybuilder shares —
start with everything except `viewer::{selection, ui}`, and let the split
fall out.

Do this first and alone. It touches every file's module path; overlapping it
with real work means every later diff is unreadable.

How the split actually fell out: `ui` moved under `viewer`, **`selection`
stayed at the crate root**. `blueprint::extract` already takes a
`SelectionBounds`, so a shared module would have had to depend on a
viewer-only one — and `SelectionBounds` is where 019 fixed the coordinate
rules W4 inherits, and what I2's damage scan reads. Selection's
*interaction* half (`gizmo`, `input`) is genuinely viewer-flavoured;
splitting the module along that line is a later refactor if it earns itself.
The shared startup lives in `lib.rs::world_app()`, so `city::run()` is not a
copy of `viewer::run()`.

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

- **Sequencing.** Block changes (011/012) → block-entity cleanup (015) →
  heightmap invalidation (013) → relight flag (014) → write (009). Getting
  the order wrong means recomputing heightmaps from pre-edit blocks — or,
  under the "let the game rebuild them" default below, deleting a compound
  that a later step in the same edit would have wanted to read.

  **Three of those five moved upstream**: `ranvil`'s `set_blocks` does the
  block-entity cleanup and clears `isLightOn` itself, in its own passes, only
  on an accepted batch. So W4's own order is preflight → `set_blocks` →
  heightmaps, with the heightmap step running once per chunk at the *end of
  the transaction* rather than once per `set_blocks` call. See ticket 031.
- **Block classification.** *Only if* heightmaps end up being recomputed
  here rather than deleted: 013's `recompute_heightmaps` takes the "does
  this block motion-block / count as a leaf" taxonomy from the caller, and
  that caller is here, because that's game data and this is where the block
  registry lives. The default is not to need it.
- **`Status`.** Only edit chunks that are fully generated
  (`minecraft:full`); writing into a partially-generated chunk invites the
  generator to overwrite it later.
- **`DataVersion`.** Compare the blueprint's against the save's and refuse,
  or warn loudly, on a mismatch — block names and properties are not stable
  across versions.
- **The coordinate rules.** This is where they get fixed once, the way 019
  fixed them for selection. Six later tasks would otherwise each decide for
  themselves what "the block at (x,y,z)" means, and that is how mirrored and
  off-by-one buildings happen. This codebase already carries `bevy.z = -mc.z`
  and inclusive bounds from 019 — inherit them, don't reinvent them.

**W5. Boundary routing and region batching. — done, ticket 032.** Block
coordinate → `(region, chunk, section, local index)`, and the batching that
makes it sane. A 20×20 building straddles up to 4 chunks; placed near a region
corner it straddles up to 4 region files. Requirements:

- group all changes by region file, apply every one of them, write each file
  **once**
- refuse (with a clear message, not a panic) any placement reaching into an
  ungenerated chunk — iteration 1 does not generate terrain
- `RegionCache` needs a mutable path and invalidation; the streaming
  pipeline must not go on serving pre-edit bytes

How it came out: `edit::route` splits the edit by region and then **plans every
region before applying any of them**, because neither `set_blocks` nor 031's
`apply` gives all-or-nothing across four files, and three quarters of a building
is worse than none. If the write phase fails anyway — the preflight can't model
every way the Anvil format can be malformed — the regions already applied are
*discarded* from the cache, which is a complete rollback exactly because nothing
here saves. That's also why a region carrying an earlier transaction's unsaved
changes is refused: the rollback would take those with it. One transaction at a
time, saved before the next, which is the contract W6 implements.

The invalidation turned out to be the small half: the cache *holds* the mutated
region, so reads already see post-edit blocks. What goes stale is `DecodedWorld`
and its meshes, and that's W7, off `EditReport::chunks`.

**W6. Write safety. — done, ticket 033.** The "sound way to save a modified
world" this plan rests on:

- **Refuse to write to a world Minecraft has open.** ranvil 016 reports it
  (via `session.lock`'s OS lock — note the file's *existence* proves
  nothing, it survives every clean exit); deciding to refuse is this layer's
  call. Writing under a running game loses the edit at best and corrupts at
  worst.
- **Back up before the first write of a session**, per region file.
- **Write atomically**: temp file, fsync, rename. A crash mid-write must
  leave the old region file intact.
- **Dry run**: report which regions and chunks would change, and how many
  blocks, without touching disk. This is also the test harness.

How it came out: `edit::session::WriteSession` **holds** the lock rather than
probing it (`SessionLock::acquire`, released on drop), because a probe cannot
close the gap between the check and the write — and holding also keeps the game
out *mid-edit*, which a probe never could. It is a write session, not the app's
lifetime: a viewer that never edits never locks the world.

Atomicity was already upstream (009's temp/fsync/rename), so what this ticket
actually decided is where the guarantee *stops*: per region file. `commit`
applies in memory (all-or-nothing, 032), then backs up **every** touched file
before saving **any** of them, so a backup failure always happens with the save
untouched and can be rolled back by 032's discard. A failure during the saves
can't be, and says so — `WriteError::WriteFailed` names what was written, what
wasn't, and the backup directory. Leaving those regions dirty rather than
discarding them is deliberate: a disk error is retryable, and discarding would
turn it into lost work.

**W7. Live re-mesh.** An edit marks its chunks dirty; the pipeline re-meshes
them, including the neighbours whose faces the edit exposed — 005-f already
solved exactly this problem for the loading frontier, so follow it rather
than inventing a second dirty-chunk mechanism.

**W8. A paint/fill command in the viewer. — done, ticket 035.** Fill the
current 019 selection with a block, or stamp a loaded blueprint at it,
driven from the existing selection panel. This exists to *prove W1–W7 end to
end before any game code is written*, and it's directly the first feature of
the general explore-and-modify app. Its manual verification — write, open
the world in Minecraft, confirm the blocks are there, the lighting is right,
and nothing is corrupt — is the gate the rest of the roadmap waits on.

How it came out: fill only, not blueprint stamping — there's no blueprint
*reader* yet (B1), and the only in-memory blueprint today is whatever the
last extraction produced, which isn't what "a loaded blueprint" means. Fill
alone already exercises W1–W7 end to end: `WorldEdit::fill(bounds, state)`
(shared with H1 later), a `BlockState: FromStr` parser as the inverse of
022's `Display`, and a `viewer::paint` module built to the same
state-machine-plus-task shape as `blueprint::export`, holding the shared
region-cache lock for the whole commit (a routed transaction can't release
it partway through without breaking 032's all-or-nothing guarantee, unlike
extraction's per-column locking). A successful commit fires
`chunk_pipeline::ChunksEdited`, so 034's live re-mesh is what makes the
result visible without a restart — this ticket is the first thing that
actually calls it. The manual open-in-Minecraft check is recorded in
`../todo.md`, not done here.

---

## B — Blueprints as building models

**B1. Structure `.nbt` reader. — done, ticket 036.** The inverse of 023:
vanilla structure file → `Blueprint`. Round-trip test against 023's writer.
This is what lets buildings be authored in Minecraft itself and pulled in
with the existing extraction UI.

How it came out: `blueprint::structure::{read_structure, read_structure_file}`,
the mirror of the writer's own split (a `Read`-taking function under a
path-taking convenience wrapper), reusing `BlockState::from_palette_entry`
for the palette rather than a third copy of the sort-and-dedupe logic. The
reader is stricter than the format technically requires — `blocks` must be
dense (one entry per position in `size`, no gaps, no duplicates) — because
that's what `write_blocks` and an in-game Save both actually produce; a gap
reads as a malformed file rather than something to guess "air" for.
`Blueprint::origin` comes back `IVec3::ZERO` (never written, per 023) and
`failed_columns` comes back `0` (no partial-column walk on this path — it's
one parse, all-or-nothing). No caller yet; B2-B4 are what will load a `.nbt`
file into the catalogue.

**B2. `Blueprint` → Bevy `Mesh`. — done, ticket 037.** Not the same entry
point as `mesh_chunk_column`: a blueprint has a `BlockState` palette with
properties and no `BlockRegistry`, so UV and tint resolution differs. Share
the quad emission; add a palette-based front end. Two decisions: faces at
the blueprint's outer boundary are always emitted (it's a free-standing
object, not a chunk with neighbours), and air in the palette stays air —
blueprints are not solid boxes.

How it came out: `resolve_faces` (`world::atlas`) and `resolve_block_tint`
(`world::tint`) were already name-keyed rather than `BlockId`-keyed, so
`blueprint::mesh_blueprint` reuses both directly — only their visibility
(and `world::mesh`'s `Face` enum and quad-push functions) had to widen to
`pub(crate)`, no logic duplicated. No per-block biome exists on a
`Blueprint`, so every biome-dependent tint on the mesh resolves against one
`BiomeColors` the caller passes in — a reasonable default for a preview, not
a claim about where the building will stand. Mesh coordinates run
`0..size` on all three axes including Y, since a blueprint has no absolute
world height until something places it. No caller yet, same as 036 — B3/B4
are what will call this.

**B3. Rotation. — done, ticket 038.** 90/180/270 about Y. The mesh part is a
transform; the *blocks* part is not — a stair's `facing`, a log's `axis`, a
door's `hinge` all have to be rewritten in the palette, or a rotated
building is visibly wrong the moment it's written to the world. Iteration 1
can ship with a table covering the properties the chosen starter buildings
actually use, as long as unrotatable properties are *detected* rather than
silently kept.

How it came out: `blueprint::rotate::rotate_blueprint(&Blueprint, Rotation)
-> Result<Blueprint, RotationError>`, resolving each palette entry once
(not once per block) the same way 037's mesher resolves the palette rather
than the grid. The grid remap composes a single 90°-turn transform `turns`
times instead of hand-deriving 180°/270° separately. `facing`, `axis`,
16-way `rotation` (signs/banners) and the `north`/`south`/`east`/`west`
connection-key group (fences, walls, panes, bars, redstone) all rewrite;
`shape` turned out to need disambiguating by *value* rather than by block —
a stair's `shape` values and a rail's are the same property name with
disjoint value sets, and only the rail set encodes an absolute direction.
`hinge` (and a chest's `type`) landed in the pass-through whitelist rather
than a rewrite rule, contrary to how this paragraph reads: both are defined
relative to the block's own `facing`, so rotating `facing` consistently
already keeps them correct. Anything outside the table is
`RotationError::UnrotatableProperty`, except at `Rotation::Deg0`, which
never inspects the palette at all — the identity case can't fail on a
property nothing here recognises yet. No caller yet, same as 036/037 —
B4 and E3 are what will call this.

**B4. The building asset catalogue. — done, ticket 039.** `assets/city/blueprints/*.nbt`,
loaded and validated at startup: size limits, palette sanity, footprint
derived from the blueprint's own dimensions.

How it came out: `blueprint::catalogue::load_catalogue_dir(&Path) ->
(BuildingCatalogue, Vec<(PathBuf, CatalogueError)>)` — a non-recursive scan
of `*.nbt` files (matched case-insensitively), each read through 036's
`read_structure_file` and checked against two rules beyond what the reader
already guarantees: no size axis may be zero or exceed
`STRUCTURE_BLOCK_MAX_SIZE` (036's own constant, reused rather than
inventing a second "how big is too big"), and the palette must have more
than just `minecraft:air` in it. Failure is per-file, not per-directory —
a missing directory is an empty catalogue (logged, not fatal, the same call
008 made for a missing saves directory) and a bad file is skipped and
reported alongside whatever did load, never taking the rest of the
directory down with it. `city::run()` calls it against
`assets/city/blueprints`, logs one line per loaded/skipped entry, and
inserts `BuildingCatalogue` as a resource — the first real caller of
036/037/038's primitives, though nothing reads the resource back out yet;
G1's build menu is what will. `assets/city/blueprints/house01.nbt` is the
first fixture, a real structure-block export rather than a synthetic one.

The one thing worth recording for later callers: two on-disk files can
collide on id (their shared filename stem) only via `DuplicateId`, detected
rather than one silently overwriting the other — but on a case-insensitive
filesystem (Windows' default) that collision can only be observed by
constructing it directly against the loader's inner path-list function,
since the OS itself won't let two differently-cased filenames coexist in
one directory to begin with.

---

## C — Definitions and scripting

The user-facing question was how to script and define values. The
recommendation:

**Data files now, not a scripting language.** Iteration 1's values are
numbers and references — production rate, inputs, outputs, tier, unlock
dependencies, blueprint file, cost. That's a schema, not a program. RON
(serde, comments, real enums, no whitespace significance) beats TOML for
nested/tagged data and beats JSON for hand-editing. Reach for Rhai or Lua
only when a *behaviour* needs to vary per building rather than a number —
and by then the data schema will tell us which hook points it needs.

**C1. Building definition schema and loader. — done, ticket 040.** One file
per building, or one per category. Sketch:

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

How it came out: `city::definition::{Building, load_definitions_dir}`, one
`.ron` per building under `assets/city/buildings`, `BuildingCatalogue`-
validated the same way 039 validates blueprints. Two deviations from the
sketch above: no inline `id` field — the filename stem is the id, same call
039 made for blueprints, so a file and its id can't drift apart — and
`blueprint`'s stem must resolve to a real catalogue entry or the whole
definition is rejected (`DefinitionError::UnknownBlueprint`), which the
sketch doesn't show because 039 had nothing to cross-check against yet.
`requires` is parsed and carried, deliberately unvalidated — C2 is what
needs every definition loaded first to check dangling references and
cycles against.

**C2. Tiers and the tech tree. — done, ticket 041.** Anno-style: a `tier` per
building, plus `requires` edges. Needs cycle detection and dangling-reference
checks at load — a tech tree with a cycle is unwinnable and the failure mode
is "button greyed out forever" if it isn't caught.

How it came out: `city::definition::resolve_requirements` runs as a second
pass after 040's per-file loop, alternating a dangling-reference check
against the whole loaded set with a DFS-based cycle check (`find_cycle`)
until a pass removes nothing. The loop, not a single pass, is the part worth
noting — removing a cyclic or dangling entry can turn some *other* entry's
`requires` into a fresh dangling reference, and a single pass would leave
that dependent looking fine when it can never unlock either. The surviving
set is the maximal subset of loaded buildings whose `requires` graph,
restricted to that subset, resolves and has no cycle — order-independent,
so it doesn't matter which problem gets found first. Two new
`DefinitionError` variants, `DanglingRequirement` and `CyclicRequirement`
(carrying the whole loop, not just one id on it), reported through the same
per-file `skipped` list 039/040 established. No consumer yet — same
"proven, not yet used" state the rest of group C is in; G1's build menu is
what will eventually grey out a locked entry using this.

**C3. Production fields, parsed but inert.** Iteration 1 shows rates and
costs in the build menu and does not simulate them. The point is that the
schema is exercised — a schema that nothing reads drifts from reality.

**C4. Hot reload and an error panel.** Definition files reload on change;
errors go to an egui panel rather than a panic or a console line nobody
sees. This is what makes balancing tolerable later.

---

## D — City state

**D1. The `City` resource. — done, ticket 042.** Placed buildings (id, origin,
rotation), roads, and a footprint occupancy grid for fast "is this tile free"
queries. The authoritative state from the rule above.

How it came out: `city::state::City`, keyed by a `BuildingId` distinct from
a building's definition id (many placed instances share the second, none
share the first). One `HashMap<IVec2, Occupant>` is the occupancy grid,
`Occupant` being `Building(BuildingId) | Road`; `City::place_building`
computes a footprint's full tile list, checks every tile is free, and only
then mutates — the same "plan every region before applying any of them"
shape `edit::route::apply_routed` (W5) already uses, applied here to avoid a
placement refused partway through leaving a phantom building's tiles marked
occupied. Rotation matters to occupancy, not just the mesh: a 90°/270°
placement's occupied rectangle has its `(x, z)` extent swapped from the
stored (always-unrotated) footprint, via `footprint_extent`/
`footprint_tiles` — the horizontal counterpart of the axis swap B3's
`rotate_blueprint` already performs on the block grid itself. `city::run()`
inserts an empty `City`; no consumer yet — E1/E2 are what will call
`place_building`, the same "proven, not yet used" state 039-041 landed
their own resources in.

**D2. City save/load. — done, ticket 043.** RON next to the world —
`<save>/citybuilder/city.ron` — so a save and its city travel together.
Versioned from the first write.

How it came out: `city::persistence::{save_city, load_city}`, reading and
writing a `CitySave`/`SavedBuilding` pair rather than `City` itself —
`occupancy` is derived, not stored, so `load_city` rebuilds it through
`City`'s own `insert_loaded`/`add_road`, which means a corrupt or
hand-edited file with two overlapping buildings fails the load
(`PersistenceError::Corrupt`) instead of producing a `City` that disagrees
with itself. `next_id` is the one field that *is* persisted verbatim rather
than recomputed: removing the highest-numbered building before saving
doesn't roll it back (`BuildingId`s are never reissued, D1's own rule), so
recomputing it from only the survivors would have reissued a removed id on
the next placement after a reload. `blueprint::rotate::Rotation` picked up
`Serialize`/`Deserialize` directly rather than a mirror enum — a placement's
orientation on disk is exactly that type. `city::run()` reads the loaded
save's root before inserting anything else and skips persistence entirely
when it's ticket 008's `empty_save` placeholder (nowhere to read from or
write to); otherwise it loads synchronously before `App::run()`, the same
shape 039/040 use, and a `Last`-schedule system saves on `AppExit`. Unlike
D1's own landing, this ticket gives `City` a real caller on both ends —
`buildings()`/`roads()`/`add_road()` lost their "no caller yet"
`#[allow(dead_code)]` markers because of it, though `place_building` and
friends still wait on E1-E4.

**D3. Journal, undo, reconciliation. — done, ticket 044.** Every placement
and demolition as an appended entry, each carrying the **as-built baseline**
(I1): what blocks we wrote, and what was there before. Gives undo for free,
gives demolish its terrain restore (E5), gives a rebuild path if the world's
blocks and the city list disagree — and gives the damage mechanic (I) the
thing it diffs against. Four features, one record; see I1 for why they're
the same record and not four.

How it came out: `city::journal::{Baseline, JournalEntry, Journal, reconcile,
repair_edit}`, in one new file rather than split state/persistence the way
042/043 were — the journal has no derived state (an occupancy grid) that
justified that split, so a flat append-only log stayed in one place, the
same call `city::definition` made for its schema and loader. `Baseline`
turned out to already be half-built: 031's `EditPolicy::capture_replaced`
records what an edit overwrote, so `Baseline::capture(edit, report)` only
had to line that up against the edit's own `written` positions, deduped and
sorted the same way, rather than computing anything new. `JournalEntry`
snapshots a full `PlacedBuilding` (which picked up `Clone` for this) on
every entry rather than a lookup key, because undoing a placement removes
the building from `City` before the entry is read, and undoing a demolition
needs the whole thing to hand `City::insert_loaded`. `Journal::undo_last`
is all-or-nothing against the journal and `City` — a failed undo (putting a
demolished building back onto a tile something else now occupies,
`UndoError::Occupied`) touches neither — but deliberately *not*
all-or-nothing against the world: it returns the restoring `WorldEdit`
uncommitted, the same "apply doesn't save" contract every other edit entry
point in this crate keeps, for the future E4/E5 caller to run through the
write path. `reconcile` recomputes the expected world state from every
currently-placed building's own *placement* baseline (not blueprint
re-derivation — I1's whole argument), grouped by region the same way
`edit::route` batches the write side, and reuses `edit::route::refusal_for`'s
`Display` for a whole-region failure rather than inventing a second message;
`repair_edit` is the write half, an edit nobody commits automatically.
Persistence follows 043's mirror-type shape (`SavedJournal`/`SavedEntry`/
`SavedPlacement`/`SavedBaseline`), with one departure: `blueprint::BlockState`
picked up `Serialize`/`Deserialize` directly rather than a third mirror type,
since a baseline can carry thousands of block states per building and both
its fields were already plain serde-able data. `city::run()` loads and saves
the journal next to `city.ron`, the same "proven by the app lifecycle, not
yet fed by gameplay" state 043 landed `City` in — no caller for
`record_placement`/`record_demolition`/`undo_last`/`reconcile` until E4/E5
exist.

---

## E — Placement

**E1. RTS camera and picking. — done, ticket 045.** Pan/zoom/rotate over the
terrain, plus screen ray → block coordinate. `camera.rs` already has a
ray-march for orbit targeting (006) to build on. Note the existing
controller owns WASD/QE/Shift/Tab and both mouse buttons — 020 hit this and
its ticket lists what's free; the citybuilder has more freedom since it
needn't keep the viewer's flight controls, but it *does* need to keep
egui's input claim ordering (`main.rs:172`).

How it came out: a third `camera::CameraMode::Rts`, living in `camera.rs`
next to `Fly`/`Orbit` rather than a parallel camera stack — `streaming`,
`unload` and `sky` all find "the camera" via `Query<&Transform,
With<CameraRig>>`, so a second component would have meant teaching three
shared systems about it. WASD pans `orbit_target` on the yaw-relative ground
plane (pitch ignored, so panning never drifts vertically), `Q`/`E` rotate
yaw, scroll zooms (reusing `Orbit`'s radius math), and right-mouse-drag free
looks — ungrabbed, unlike `Fly`'s right-drag, since left mouse is reserved
for E3/E4's picking and the cursor needs to stay visible. A new
`CameraStartMode` resource (default `Fly`) lets `city::run()` start the rig
in `Rts` the same way it already overrides `RenderFloor`, so `setup_world`
stays the one place the camera entity is spawned for both games. Picking
landed as `city::picking::HoveredBlock`, a per-frame resource built on the
existing `camera::block_under_cursor` — no second raycast — with no
consumer yet, the same "proven, not yet used" state ticket 042's `City`
landed in; E2's grid fit and E3's ghost preview are the eventual readers.

**E2. Grid and footprint fit. — done, ticket 046.** Tile grid, footprint
occupancy, and the terrain rules: sample heights under the footprint, define
what slope is buildable, decide whether the game auto-levels or refuses. The
answer here feeds H directly.

How it came out: `city::grid::fit_footprint(origin, footprint, rotation,
&DecodedWorld) -> FootprintFit`, sampling ground through
`ChunkColumn::topmost_non_air` against `DecodedWorld.columns` directly —
no `RegionCache`, no I/O, the same already-decoded data E1's picking reads
— and walking D1's own `footprint_tiles` rather than a second rotation-aware
tile walk. The roadmap's open question ("auto-levels or refuses") resolved
to **refuses**: levelling means writing blocks, which is W4/W5's job through
a real `WorldEdit`, and folding that into a read-only fit check would make
every future caller (E3's every-frame ghost preview included) secretly pay
for the write path. Ground level within `MAX_FOOTPRINT_STEP` (1 block) fits
at its *lowest* sampled point — a low corner clips a little into the
building's own foundation rather than leaving a gap floating under it, given
nothing here fills terrain in — and anything steeper is `Refused` outright,
for H1's terraforming to fix later by hand. "Ground" reuses `world::is_solid`
(not air), the same predicate the mesher already culls faces against, not a
new fluid-aware classifier — a lake surface counts as ground for now,
deliberately, per the module's own docs; that refinement is roadmap I3's job.
No caller yet — E3's ghost preview and E4's commit are the eventual readers,
the same "proven, not yet used" state ticket 045's `HoveredBlock` landed in.

`is_solid` turned out to have a second gap, found once E3/E4 actually had a
building to place near one: it treats a tree or a fence post exactly like
terrain, so a footprint with an obstruction standing on otherwise-flat
ground reads as a cliff and is refused, even though E4's own "air is a
block" policy would clear the obstruction without complaint if the fit
check weren't in the way. Not fixed inline — filed as ticket 052, done
separately (see `finished_tickets/052-terrain-fit-ignores-clutter.md`):
`ground_height_at` now walks a `city::grid`-local `is_ground`/`is_clutter_name`
predicate (mirroring `world::mesh::is_solid_name`/`is_solid`'s shape, not
`world::tint`'s per-`BlockId` table — a footprint fit samples too few tiles
a frame for a table to earn its keep) instead of plain not-air, off a new
general `ChunkColumn::topmost_matching` that `topmost_non_air` itself now
just specialises.

**Revised by ticket 058: the "refuses past `MAX_FOOTPRINT_STEP`" half is
gone.** User-directed: a real Minecraft world is inherently uneven, and a
hard 1-block cap on how much a footprint's ground could vary restricted
where a player could build far more than it was worth — refusing outright is
a much bigger cost to the player than the mild clipping/floating a steep
site produces. `fit_footprint` still samples ground and still returns
`base_y` at the footprint's lowest point exactly as before; what's gone is
the `max_y - min_y > MAX_FOOTPRINT_STEP` check and `FitError::TooSteep`
itself — `NotLoaded` is the only refusal reason left. `base_y` is now purely
the placement's *initial suggested* height, same as it always fed into
`PlacementSelection::y_offset` (ticket 048, `Page Up`/`Page Down`/`Home`);
nothing here stops a steep placement, it just isn't auto-levelled either.
H1's terraforming is no longer load-bearing for placement at all — it's a
player's own choice to flatten a site rather than build into the slope. See
`finished_tickets/058-remove-footprint-slope-refusal.md`.

Noted for later, not scoped or scheduled: rather than a hard block, an
extreme placement (heavily embedded in a hillside, mostly floating over a
low spot) could cost something instead — build time, or a future resource,
scaling with the number of solid blocks the placement needs to clear. That's
a genuinely new mechanic (counting blocks cleared under/around a footprint,
and *some* notion of cost to charge it against — neither exists; see C3/H2's
own "inert" state) and sits naturally next to H2's already-open "yields"
task, which has the same "count blocks a terraform/placement action
disturbs" shape. No ticket yet — it needs a real cost system to hang off
first.

**E3. Ghost preview and validity. — done, ticket 047.** The B2 mesh at the
cursor with a translucent material, tinted by validity, snapped to the grid.
Needs a second material — the terrain's is `AlphaMode::Mask(0.5)`
(`main.rs:225`), which is a cutout, not translucency.

How it came out: `city::placement`, combining E2's `fit_footprint` (terrain)
and D1's `City::is_tile_free` (occupancy) into one green/red signal
(`resolve_placement`) rather than either check growing a second
responsibility. G1's build menu doesn't exist yet, so a small keyboard
stand-in (`PlacementSelection`/`cycle_selection` — number keys pick a
catalogue entry, `R` rotates, `Escape` clears) drives *which* building is
selected, explicitly not G1 itself, the same role ticket 035's paint command
played for the write path before any UI did. The ghost mesh and its two
translucent materials (`AlphaMode::Blend`, `unlit: true` so the tint reads
the same day or night) are both cached — `(catalogue id, Rotation)` for the
mesh (needing `Rotation: Hash`, added directly to the type per ticket 043's
own precedent), a rotation the palette can't support cached as a failure
right alongside successes rather than retried and re-logged every frame.
`city::picking` gained a `PickingSet` label so the ghost orders after this
frame's `HoveredBlock` rather than last frame's. No commit yet — E4 is what
will call `City::place_building` off the same validity signal this ticket
computes for the preview.

**E4. Commit. — done, ticket 048.** City state entry + W4/W5 write + W7
re-mesh, in that order, transactionally: if the write fails, the city entry
doesn't survive either.

How it came out: `city::commit::CommitPlugin`, recomputing validity itself
(`try_commit_placement` calls the same `placement::resolve_placement` E3's
ghost reads, with *this* frame's inputs) rather than trusting whatever the
ghost displayed last frame — a click commits exactly what's on screen at the
moment of the click. `state::City::place_building` runs synchronously the
instant a click is accepted (cheap — an occupancy check, no I/O), so the tile
claim exists before `WriteSession::open`/`commit` even starts on
`AsyncComputeTaskPool`; a single `CommitState::pending` slot is the same
one-at-a-time backpressure `PaintCommand`/`BlueprintExtraction` already use.
`blueprint_edit` decided the "air is a block" question `WorldEdit`'s own docs
left open for this ticket: every grid position is written, air included,
clearing whatever sliver of terrain E2's footprint-fit tolerance left poking
into the building rather than leaving it standing. On a successful write:
`journal::Baseline::capture` (with `EditPolicy::capture_replaced` turned on
specifically for this caller) and `Journal::record_placement` land I1's
as-built baseline for real, and `ChunksEdited` fires for W7's live re-mesh.
On failure: `City::remove_building` — the transactional half the roadmap
names. A separate, adjacent piece the roadmap didn't call out:
`PlacementSelection` gained a `y_offset`, stepped by `Page Up`/`Page Down`
and reset by `Home`, so a placement's height can be nudged off the terrain's
own auto-fit before committing — not the mouse wheel, which Rts's camera
already owns for zoom. See `finished_tickets/048-placement-commit.md`.

**E5. Demolish. — done, ticket 049.** City state removal, then re-project —
restoring the terrain that was there needs the pre-placement blocks kept in
the journal, which is a D3 decision to make deliberately rather than
discover.

How it came out: `city::demolish::DemolishPlugin`. No G1 build menu yet, so
`Delete` on `picking::HoveredBlock`'s tile is the whole UI — `City::occupant_at`
finds the id, `Journal::placement_baseline` finds what to restore, refusing
(rather than guessing) when a placed building has no recorded baseline, an
older-save edge case. The restoring edit needed no new code:
`journal::Baseline::restore_edit`, already built for D3's undo, is exactly
the placement baseline's own `previous` half, widened from private to
`pub(super)` for this second caller. The interesting decision was ordering:
commit claims the tile in `City` *before* its write starts, so a second click
can't race it; demolish does the opposite, freeing the tile only *after* its
restoring write succeeds — freeing it first would let a new placement land on
the same tile mid-write, and whichever write reached disk last would clobber
the other. A new `city::write_gate::WriteGate` resource, shared between
`CommitPlugin` and `DemolishPlugin`, closes a gap neither module's own
single-pending-slot backpressure covered on its own: two independent
`WriteSession::open` calls, on unrelated tiles, in the same frame. `ranvil`'s
session lock is a mandatory lock on a freshly-opened file handle, not on the
process, so a second concurrent open from this same app fails exactly like
Minecraft already having the world open — and unlike that case, there's no
retry built in, so it would have failed one of the two writes outright rather
than merely delayed it. See `finished_tickets/049-demolish.md`.

**E4/E5/G2 addendum — ticket 051, "defer world writes to a manual Save."**
Manually testing placement surfaced that every commit, demolition and undo
above opened its own `WriteSession` and saved to disk immediately — one
full region-file rewrite per building, per 009's own "a single-block edit
rewrites 512×512 blocks' worth of file" note. `city::commit::apply_building_edit`
(the renamed `commit_building`) now applies straight to the shared
`RegionCache` with `EditPolicy::allow_dirty_regions` and stops — no session,
no disk write, no `WriteGate` (removed; nothing opens a session per edit any
more, so the race it prevented no longer exists). A new `city::save`
module is the one place any of it reaches disk: "Save world" in the city
panel opens a real `WriteSession` and calls `WriteSession::flush` (built by
033, unused until now) to write every dirty region at once. `AppExit` also
flushes, synchronously, before `city.ron`/`journal.ron` are written — the
one safety net that keeps those two files from describing buildings the
world never actually got. See `finished_tickets/051-defer-world-writes-to-a-save-button.md`.

---

## F — Streets

**F1. The road graph. — done, ticket 053; revised to cell space and given
F3's piece selection by ticket 054.** Tiles plus adjacency, on the same
grid as E2. Iteration 1 needs the graph even without logistics, because F3
and F4 both read it.

How 053 came out: `city::road`, no new resource — `connections_at`/
`reachable_from`/`is_connected` all take `&state::City` and recompute their
answer off its existing occupancy grid on every call, the same "derived,
never stored" choice `city::grid::fit_footprint` already made for terrain
fit, so there's nothing here that can itself go stale. `Direction`'s four
offsets follow Minecraft's own `x`/`z` convention (north `-z`, south `+z`,
east `+x`, west `-x`), not a screen-relative one — this module never
touches a Bevy `Transform`, so the `bevy.z = -mc.z` flip other modules
carry doesn't apply. `connections_at` deliberately works from a tile that
isn't itself a road yet (F2's drag-to-build preview will want "what would
this connect to" before committing) and reads `Occupant::Road` specifically,
not "is this tile occupied" — a building must not read as a road neighbour.
`reachable_from`'s BFS returns the empty set for a non-road start rather
than a one-element set, so "reaches only itself" (a real one-tile road
island) and "isn't a road at all" stay distinguishable — the distinction
F4's "is this building on the network" will need.

How 054 revised it: a road became a real `ROAD_CELL_SIZE` (6-block) cell —
2 road-surface blocks flanked by a kerb and a shoulder on each side — not a
1x1-block tile, because a rotated straight piece can't stand in for a
corner and 053's tile model had no piece concept to get that right or
wrong. `state::City::add_road`/`remove_road`/`roads()` became
`add_road_cell`/`remove_road_cell`/`road_cells()`/`is_road_cell` outright
(not added alongside the old ones — two road grids that could disagree is
worse than one migration), each cell marking all 36 underlying block tiles
`Occupant::Road` with the same all-or-nothing shape `place_building` uses.
`city::road`'s whole coordinate space moved with it — `Direction`'s offsets
are unchanged in *value* but now mean "one cell." `persistence::CitySave`'s
`roads` field (block tiles) became `road_cells` (cell coordinates) with a
version bump (`1` → `2`), refusing rather than silently misplacing an old
save's roads six blocks off.

054 also lands F3's *selection* half early, since it has no dependency on
F3's rendering or F2's input: `road::RoadPieceKind` (`Isolated | DeadEnd |
Straight | Corner | T | Cross`) and `road::select_piece`, which rotates a
canonical connection pattern per kind through `blueprint::rotate::Rotation`'s
own clockwise convention until it matches a cell's actual connections —
so the returned kind and rotation are exactly what
`blueprint::rotate_blueprint` would need to reproduce the right shape.
`city::road_catalogue` loads the fixed six `.nbt` files (`isolated.nbt`,
`dead_end.nbt`, `straight.nbt`, `corner.nbt`, `t.nbt`, `cross.nbt`) that
`select_piece`'s kinds pick between, validated to be exactly
`ROAD_CELL_SIZE` on `x`/`z`. No real pieces ship with 054 — like 039's
`house01.nbt`, they need an actual structure-block export, not something to
fabricate — so the catalogue is proven against synthetic fixtures and isn't
wired into `city::run()` yet. No caller for any of this beyond its own
tests — same "proven, not yet used" state tickets 039/040/042 landed their
own resources in; F2 is the first one due, and what will call `select_piece`
and `road_catalogue::RoadCatalogue::get` to actually mesh and write a cell.

**F2. Drag-to-build. — done, ticket 055.** Click-drag from A to B, routed
over the grid, with a live preview of the cells it would claim and their
cost.

**F3. Auto-tiling. — selection done, ticket 054; mesh+write done, ticket
055.** Each cell's `select_piece` answer is the kind and rotation to render,
re-picked when a neighbour changes. Wired to a spawned/rotated mesh (the same
`blueprint::rotate_blueprint` + `blueprint::mesh_blueprint` path E3's ghost
preview already uses) and to the write path (W4/W5) once F2 gave it
something to place. Slopes are the hard part and remain a legitimate
iteration-2 deferral if they bite — 055 reuses E2's `fit_footprint` as-is
rather than relaxing or working around it.

How 055 came out: `city::road_build`, gated by a new `city::tool::ActiveTool`
(`Building | Road`, `T` to switch — `placement`'s ghost and `commit`'s click
handler both no-op unless `Building`, so the two tools never react to the
same input). A drag is an L-shaped `drag_path` (start's row, then end's
column) rather than a diagonal — every consecutive cell stays a cardinal
neighbour of the next, which `select_piece` assumes throughout. The preview
uses a real, rotated catalogue piece where one exists and a flat translucent
quad otherwise (no real `.nbt` road pieces exist yet — see 054's own "no real
assets"), tinted green/red the same way E3's ghost preview is; mid-drag, a
`connections_with_path` variant treats the rest of the current path as road
too, so a straight run previews as a run of `Straight` pieces rather than
disconnected dead ends. Committing validates the whole path before touching
any of it (the same "plan before apply" shape `place_building`/
`add_road_cell` already use, lifted to a multi-cell drag), claims every cell
in `City` synchronously before the write starts, and batches one merged
`WorldEdit` across every *affected* cell — the path plus any already-road
neighbour whose own piece might now need to change (a dead end that grew a
neighbour becomes a straight or a corner). A cell with no matching catalogue
piece keeps its `City` entry but contributes nothing to the write — the
state is authoritative regardless of whether there's an asset to render it
with. A failed write rolls back only the cells *this* drag newly added, not
a pre-existing neighbour that merely needed re-tiling. Not journaled — no
undo or demolish for a road cell yet, only `City`'s own persistence.
`city::run` now loads `assets/city/roads` the same way it loads the building
catalogue (039's own shape), always inserting whatever loaded even if empty.

**F4. Connectivity queries. — done, ticket 056.** "Is this building on the
road network", "what does this road segment reach". No consumer in
iteration 1 — it's the substrate every later logistics feature needs, and
it's nearly free once F1 exists.

How it came out: four functions in `city::road`, built entirely on
`reachable_from`/`is_connected` — the two primitives 053/054 had already
written and left `#[allow(dead_code)]` for exactly this — rather than a
second BFS. The missing piece was a bridge from a *building*'s footprint to
the road cells next to it, since `state::City` had no such query:
`touching_road_cells(city, building)` walks every footprint tile's four
block-adjacent neighbours through a new shared `state::cell_of` (the inverse
of `road_cell_tiles`'s corner math, `div_euclid` so it's correct for
negative coordinates — `road_build::cell_of`, private until now, became a
one-line wrapper around it rather than a second copy) and collects whichever
resolve to a road cell. Redundant over a footprint's interior tiles (an
interior neighbour is always another footprint tile, never a road cell, so
it simply never matches) but correct, and simple beats fast for a query nothing
calls yet. `is_building_connected`/`buildings_connected` take a `BuildingId`
and return `Option<bool>` — `None` for an id that isn't currently placed,
distinct from `Some(false)`, the same distinction `city::demolish`'s
`occupant_at` already draws elsewhere. `buildings_connected` deliberately
doesn't short-circuit on an empty adjacency set; it hands both buildings'
touching cells straight to `is_connected` and lets that function's own
`is_road` checks return `false`, so this ticket's connectivity questions are
answered by *composing* 053/054's primitives, not reimplementing their edge
cases. `buildings_reachable_from(city, start)` is the buildings-oriented view
of `reachable_from` itself. All four, plus `reachable_from`/`is_connected`
now that they have real (if still unconsumed) callers, stay
`#[allow(dead_code)]` — nothing outside this module's own tests calls any of
it yet, the same "proven, not yet used" state F1-F3 themselves landed in; a
later unconnected-building UI warning or logistics feature is the eventual
reader. 13 new tests (`state::cell_of`'s negative-coordinate correctness,
plus 12 across the four new functions) — 496 to 509 per `cargo test --lib`.

---

## G — UI

**G1. Build menu. — done, ticket 050.** Catalogue grouped by tier, with
locked entries visible but disabled and showing what unlocks them — the Anno
affordance that makes a tier tree readable. Shows costs and production from
C1 even while inert.

**G2. City panel. — done, ticket 050; write/save split by 051.** Building
counts, road length, and the write status — last write, dirty regions,
backup location. That last part matters more than it sounds: the user needs
to know whether what they see has actually reached the world.

How it came out: `city::ui::UiPlugin`, a second `EguiPlugin` registration
independent of the viewer's own (which carries panels this game has no use
for, several built around `crate::selection` that `city` deliberately
doesn't use). The build menu bridges `PlacementSelection::catalogue_id`
(a `BuildingCatalogue` key) against `BuildingDefinitions` (tier/cost/
production/`requires`) via a new `LoadedBuilding::catalogue_id` field, and
defines "unlocked" for the first time in the project — a `requires` id is met
once at least one building of that type has actually been placed in `City`,
the smallest rule that uses data already on hand rather than inventing a
"researched techs" resource C3's own docs say iteration 1 has no use for. The
city panel adds a `WriteStatus` resource (recorded by `commit`/`demolish`,
later `road_build`) and an "Undo" button through a new `city::undo` module,
`Journal::undo_last`'s promised caller since ticket 044. See
`finished_tickets/050-build-menu-and-city-panel.md` for the full account,
including the manual-verification checklist it left in `../todo.md`.

---

## H — Terraforming

**H1. Level and dig tools. — done, ticket 057.** Reuses W4/W5 wholesale —
it's the same write path with a different source of block changes.

How it came out: a third `city::tool::ActiveTool::Terraform`, `T` now
cycling three ways instead of flipping two, gated the same way
`city::road_build` already gates its own drag on `ActiveTool::Road`. The new
`city::terraform` module reuses `road_build`'s click/hold/release drag shape
(`TerraformDragState::anchor`) but widened from an L-shaped cell path to a
plain rectangle (`rect_tiles`) — there's no piece catalogue or cardinal
adjacency to keep straight here, so a rectangle is the natural shape for "an
area of ground." Dig clears the topmost block (`world::ChunkColumn::
topmost_non_air`, not `city::grid`'s clutter-skipping `is_ground` — a dig
clears a tree exactly like it clears stone) from every tile in the
rectangle; Level reads the drag's starting tile as a target height and digs
the high tiles down to it or fills the low ones up to it with a fixed
`minecraft:dirt` (no material inventory yet to spend from, the iteration-1
boundary C3/H2 already draw around production) — closing the gap
`city::grid`'s own docs left open, where E2's `fit_footprint` refuses uneven
ground rather than levelling it and names this ticket as the fix. Neither
tool reuses `WorldEdit::fill` despite that function's own forward reference
to terraforming — both write a *different* value at every position, which
`fill`'s one-block-everywhere shape doesn't cover. Commits reuse
`city::commit::apply_building_edit` directly, the same region-cache dispatch
`commit`/`road_build` already use; unlike either of those, there's no
`state::City` entry and no journal record for dug or levelled terrain at
all, so a failed write has nothing to roll back beyond its own `WriteStatus`
line. No preview mesh — console-only feedback, the same state ticket 035's
original paint/fill command shipped in before any tool grew a ghost; a named
follow-up if that turns out to matter in practice.

**H2. Yields.** Dug blocks become resource counts in city state. Inert in
iteration 1, like C3, but the plumbing is the same shape production will
need — still open; `city::terraform`'s dig doesn't record what it removed
anywhere production could later read. E2's own "noted for later" addendum
(ticket 058) is the same shape read the other direction: a *placement's*
own footprint clearing solid blocks, charged as build time or a resource
cost rather than refused outright. Whichever lands first (a dig's yield, or
a placement's clearing cost) is likely most of the plumbing the other one
needs too — worth picking up together rather than twice.

---

## I — Damage: the world diffing back

Blocks a player alters in Minecraft, inside a building's volume, count as
damage. Past a threshold the building stops working. Health and the
resulting production malus are shown in the game.

This is the best-fitting mechanic in the whole design, because it is the
architectural rule turned into gameplay. The plan already says *the city
state is authoritative and the blocks are a projection of it*, and already
needs a reconciliation pass that diffs intent against reality (D3). Damage
is that same diff, read as a game signal instead of an error. Nothing new
has to be invented to detect it — only decided.

It also closes the loop the whole project is about: the world you can walk
into is not a read-only export, it answers back.

**I1. The as-built baseline. — do this in iteration 1.** At placement, record
what was actually written: the positions, the `BlockState` at each, and the
save's `DataVersion` and chunk timestamps at the time.

This is the one part of group I that cannot wait. Everything else here is
iteration 2 — but a building placed *without* a baseline can never be
diffed later, because "what it should look like" is not recoverable after
the fact. Re-deriving it from the blueprint alone is wrong for two reasons:
the blueprint has air in it (what was under that air before? terrain we
didn't write and mustn't blame the player for), and E2's terrain fit may
have adjusted the placement. So: **iteration 1 places buildings and writes
the baseline, iteration 2 reads it.** If nothing else from group I lands
now, land this.

It is also the same record E5 (demolish) needs to restore terrain and D3
needs for undo. One structure, four consumers — which is the argument for
getting it right rather than treating it as bookkeeping.

**I2. The scan.** For each building, read its volume out of the save and
diff against the baseline. The read side already exists and needs no new
code: `blueprint::extract` walks a bounds through `SharedRegionCache` →
`ChunkRegion::get_block`, off the main thread, with progress reporting. A
building is a small extraction. Reuse it rather than writing a second walk.

Two failure modes that must **not** read as damage:

- **An unreadable or missing chunk.** If the player deleted a region file or
  the chunk won't parse, that is *unknown*, not *destroyed*. `Blueprint`
  already distinguishes these: `failed_columns` counts only real failures
  and deliberately excludes ungenerated chunks. Inherit that distinction —
  a building whose scan failed shows as "unknown", not 0%.
- **A `DataVersion` change.** If the player opened the world in a newer
  Minecraft, block names and properties may have been migrated wholesale and
  *every* block will mismatch — instantly ruining the entire city through no
  fault of the player. Compare against the baseline's `DataVersion` and, on
  a mismatch, refuse to scan and offer a re-baseline instead. This is the
  single most damaging false positive available and it costs one comparison
  to prevent.

**I3. The comparison policy. — the hard one.** Naive `BlockState` equality
does not work, and this is where the mechanic lives or dies. A building with
a door in it is "damaged" the moment somebody opens the door. Classify
instead, per block:

- **Structural** — walls, floors, roof, stairs, logs. Full equality
  including properties. This is what damage means, and it's the denominator.
- **Interaction state** — same block, different property because a player
  used it: `open`, `powered`, `lit`, `triggered`, a lectern's `has_book`.
  Ignore. Opening a door is not vandalism.
- **Volatile** — leaves (decay), grass/dirt/mycelium (spread), water and
  lava (flow), snow layers, fire, crops, ice, copper oxidation. Exclude from
  the denominator entirely. These change on their own with no player
  involved, and counting them means every building rots quietly overnight.
- **Blueprint air** — positions the building declares as empty. A block
  placed there is *obstruction*, not damage; a different thing, and
  legitimately a later mechanic. Excluded from the denominator by default,
  but counted separately so the option stays open.

The shape to use is a per-block-name table built once against the registry
— the same pattern `world::tint`'s `build_block_tint_table` /
`BlockTint` already establishes for "a property of every block id".
Follow it rather than inventing a second one.

Get this wrong in the harmless direction: when unsure, classify as volatile
and ignore it. A mechanic that under-reports damage is a mild
disappointment; one that ruins a player's city because it snowed is a bug
report and a reload from backup.

**I4. Health, thresholds, malus.**
`health = 1 - damaged_weight / structural_weight`, uniform weights to start
— but write the function as weighted from the outset so per-block importance
is a data change later, not a rewrite.

State is `Pristine | Damaged(health) | Ruined | Unknown`. Thresholds and the
malus curve come from C1's `integrity` block: full output above
`pristine_above`, linear ramp down to `ruined_below`, zero underneath.
Per-building, because a warehouse and a lighthouse shouldn't have the same
tolerance.

Production is inert until iteration 2, so the malus is a displayed number
first and a simulated one later — which is the same deal C3 and H2 already
take, and it means the curve gets tuned against real numbers before anything
depends on it.

Worth stating as intentional: **the scan is stateless**, so a player who
rebuilds a wall by hand heals the building. Damage is a description of the
world right now, not an accumulated counter. That's the better mechanic and
it falls out for free.

**I5. Scan scheduling and the timestamp gate.** Scanning every building at
every startup is fine at 100 buildings and not at 10,000. The gate is nearly
free: a region file's header carries a **per-chunk last-written timestamp**,
and if no chunk under a building has been rewritten since the last scan,
nothing in it changed. Four bytes, no decompression, no NBT parsing.

`ranvil` doesn't expose that table today — it reads bytes 0..4096 and never
touches 4096..8192 — so it's now **`ranvil` ticket 020**. Note 009's writer
maintains the same table, which means our *own* writes bump it; the
comparison must be against the baseline's recorded timestamp, or every
building looks changed the instant it's placed.

Schedule: full scan on load behind the gate, on the async task pool; re-scan
a building when a chunk intersecting it is re-read; on demand when the
player inspects one. Never on the main thread — this is I/O and lock-bound,
exactly the reason `blueprint` runs one extraction at a time on
`AsyncComputeTaskPool`.

**I6. Repair.** Re-project the baseline onto the volume through W4/W5. Almost
free once the write path exists, and it's the obvious counterpart to a damage
number the player can see. Costs materials in iteration 2; costs nothing in
iteration 1 beyond a button.

**I7. Display.** Per-building: health percent, state, and the resulting
output multiplier, in an inspect panel off E1's picking. City-wide: counts
by state in the G2 panel, so a player who's been away knows something
happened without clicking twenty buildings.

At-a-glance in the world is the part worth thinking about rather than
defaulting: a tint on the building's mesh reusing the vertex colour channel
(011) is cheap and readable; a gizmo or icon over damaged buildings reuses
`selection::gizmo`. Either beats making the player click to find out.

---

## R — The city view's render depth

**R1. Don't mesh below the terrain surface. — done, ticket 030.** The citybuilder's
camera looks at the surface from above and never goes under it, so the ~7
sections per chunk column below the terrain are decoded, meshed and drawn for
nothing — and the caves in them are where the invisible face count really is.
A per-chunk floor from `min(OCEAN_FLOOR)` over the chunk's 256 heightmap
columns (`ranvil` 013, done) cuts them, and taking the *minimum* is what makes
it safe: a ravine or cave mouth anywhere in the chunk drags the floor down with
it, so the cutoff never slices into a hole you can see.

**`block_viewer` keeps rendering everything** — the explorer needs the
underground, and the floor is switched on by `city::run()` rather than shared.

Independent of W, B, C and D: it's a change to the decode/mesh path and nothing
in the game logic touches it. H (digging) is what later makes the floor move,
and that half is written up in the ticket as a re-*decode* rather than 005-f's
re-mesh, because the blocks under the floor were never decoded to begin with.

---

## Ordering advice

- **L1 first, alone.** It's mechanical and it touches everything.
- **W is the critical path, and W8 is the gate.** Nothing in D/E/F is worth
  building until blocks provably land in a world Minecraft opens without
  complaint. W8 puts that proof one ticket after the infrastructure instead
  of at the end of the project.
- **Lighting is Minecraft's job — decided, not deferred.** This project
  never computes light. `set_blocks` clears the chunk's `isLightOn` byte
  (ranvil 014) and the game relights when it gets round to it; lazily is
  fine, and a building that is wrongly lit until something touches its chunk
  is an accepted cost, not a bug to be fixed here.

  This used to be written up as the assumption to verify before anything
  else. It isn't a gate any more, because the decision no longer depends on
  the answer. The flag is confirmed present as a root `TAG_Byte` on the real
  save (`DataVersion` 4438) and is cleared on every edit; whether the game
  honours it is still worth ten minutes with the world open (the manual check
  in `../todo.md`), and if it doesn't, the fallback is deleting the affected
  sections' `BlockLight`/`SkyLight` arrays. If *that* doesn't work either,
  the answer is still not a lighting engine — it's living with stale light.
  Nothing in W waits on this.
- **ranvil 011's round-trip test before its encoder.** Ticket 001 was this
  exact arithmetic, in the other direction, and it shipped broken upstream.
- **`Heightmaps` follow lighting: let the game rebuild them** — still the
  default, but no longer the only option. W4 deletes the compound on an edited
  chunk (`ChunkRegion::remove_heightmaps`) rather than recomputing it, and the
  in-game check (grass, snow and rain landing correctly, mobs not spawning on
  lit ground) is still worth doing in the same sitting as the relight check
  above. What changed is that ranvil 013 got written anyway, so a bad answer
  costs one call swapped for `recompute_heightmaps(x, z, classify)` rather than
  a day of 9-bit packing code — and the classifier it would need is the block
  taxonomy W4 already has to own. W4 doesn't wait for the check either way.

  013 got written because its **read** half is useful regardless of any of
  that: the heightmaps sit at the chunk root and say how deep a chunk's terrain
  goes before a single section is decoded, which is what a render-depth cutoff
  would key off. See ranvil's `finished_tickets/013-...md`.
- **B and C can run in parallel with W** — different files, no shared types
  beyond `Blueprint`. If two things are being worked at once, that's the
  split.
- **D1 before E and F, always.** Both write to city state; if they land
  first they'll each invent their own.
- **F is more independent than it looks.** Roads need D1 and the write path,
  not placement — they can be built before E if that's more appealing.
- **I1 ships with E4, in iteration 1.** It is three fields on a journal
  entry when written at placement time, and unrecoverable afterwards. The
  rest of group I can wait indefinitely; this cannot.
- **I2 needs no write path.** Damage *detection* is pure reading, on
  machinery that already exists (`blueprint::extract`). Only repair (I6)
  needs W. So group I can land early if it's the fun part — the usual
  reason to reorder a roadmap.
- **I3 is the ticket to spend real time on.** Everything else in the group
  is plumbing; the classification table is the mechanic. Budget for tuning
  it against a real world rather than expecting to get it right on paper.

## Deliberately not in this iteration

- **Terrain generation.** Placements reaching ungenerated chunks are refused
  (W5), not generated. Generating vanilla-compatible terrain is a project.
- **Lighting computation of any kind** — and not just in this iteration.
  We clear `isLightOn`, we never compute a light level, and stale light in
  the meantime is accepted. See the ordering advice for the fallbacks if the
  game turns out to be lazier about relighting than hoped; none of them is
  "write a lighting engine".
- **Writing block entities.** Blueprints with chests and signs place their
  blocks; the entities are dropped, with a warning. Removing *existing* ones
  we overwrite is in scope — that's corruption avoidance, not a feature.
- **Entities and mobs.** Structure files can carry them. Ignored.
- **Production simulation, resources, spending.** C3, H2 and I4's malus
  carry the data; nothing consumes it. This is the explicit iteration-1
  boundary.
- **Damage detection itself (I2–I7).** Iteration 2. But **I1's baseline is
  in iteration 1** — see the ordering advice; it's the one piece that can't
  be added retroactively.
- **Obstruction as a distinct mechanic** (blocks placed in a building's
  declared air). I3 counts it separately and nothing reads it yet.
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
