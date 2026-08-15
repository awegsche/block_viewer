# Roadmap — from "explore a save" to "build a city in one"

Not a work item; the plan for the citybuilder game and the shared world-edit
infrastructure it needs. High-level tasks here get split into numbered
tickets in this directory when they're picked up (next free number: 038).
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
     |    B3  rotation, incl. block-state properties
     |    B4  the building asset catalogue
     |
     +-- C  definitions & scripting
     |    C1  building definition schema + loader
     |    C2  tiers and tech tree
     |    C3  production fields (data only in iteration 1)
     |    C4  hot reload + a definition-error panel
     |
     +-- D  city state (authoritative)
          D1  the City resource: buildings, footprints, occupancy
          D2  city save/load next to the world
          D3  journal, undo, world reconciliation
               |
               +-- E  placement          +-- F  streets
               |    E1  RTS camera, picking    F1  road graph on the grid
               |    E2  grid + footprint fit   F2  drag-to-build routing
               |    E3  ghost + validity       F3  auto-tiling the pieces
               |    E4  commit                 F4  connectivity queries
               |    E5  demolish
               |
               +-- G  UI: build menu, city panel
               +-- H  terraforming: dig and level
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
     R1  don't mesh below the terrain surface   <- ticket 030
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

**B3. Rotation.** 90/180/270 about Y. The mesh part is a transform; the
*blocks* part is not — a stair's `facing`, a log's `axis`, a door's `hinge`
all have to be rewritten in the palette, or a rotated building is visibly
wrong the moment it's written to the world. Iteration 1 can ship with a
table covering the properties the chosen starter buildings actually use, as
long as unrotatable properties are *detected* rather than silently kept.

**B4. The building asset catalogue.** `assets/city/blueprints/*.nbt`,
loaded and validated at startup: size limits, palette sanity, footprint
derived from the blueprint's own dimensions.

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

**C1. Building definition schema and loader.** One file per building, or one
per category. Sketch:

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

**C2. Tiers and the tech tree.** Anno-style: a `tier` per building, plus
`requires` edges. Needs cycle detection and dangling-reference checks at
load — a tech tree with a cycle is unwinnable and the failure mode is
"button greyed out forever" if it isn't caught.

**C3. Production fields, parsed but inert.** Iteration 1 shows rates and
costs in the build menu and does not simulate them. The point is that the
schema is exercised — a schema that nothing reads drifts from reality.

**C4. Hot reload and an error panel.** Definition files reload on change;
errors go to an egui panel rather than a panic or a console line nobody
sees. This is what makes balancing tolerable later.

---

## D — City state

**D1. The `City` resource.** Placed buildings (id, origin, rotation),
roads, and a footprint occupancy grid for fast "is this tile free" queries.
The authoritative state from the rule above.

**D2. City save/load.** RON next to the world — `<save>/citybuilder/city.ron`
— so a save and its city travel together. Versioned from the first write.

**D3. Journal, undo, reconciliation.** Every placement and demolition as an
appended entry, each carrying the **as-built baseline** (I1): what blocks we
wrote, and what was there before. Gives undo for free, gives demolish its
terrain restore (E5), gives a rebuild path if the world's blocks and the
city list disagree — and gives the damage mechanic (I) the thing it diffs
against. Four features, one record; see I1 for why they're the same record
and not four.

---

## E — Placement

**E1. RTS camera and picking.** Pan/zoom/rotate over the terrain, plus
screen ray → block coordinate. `camera.rs` already has a ray-march for orbit
targeting (006) to build on. Note the existing controller owns WASD/QE/
Shift/Tab and both mouse buttons — 020 hit this and its ticket lists what's
free; the citybuilder has more freedom since it needn't keep the viewer's
flight controls, but it *does* need to keep egui's input claim ordering
(`main.rs:172`).

**E2. Grid and footprint fit.** Tile grid, footprint occupancy, and the
terrain rules: sample heights under the footprint, define what slope is
buildable, decide whether the game auto-levels or refuses. The answer here
feeds H directly.

**E3. Ghost preview and validity.** The B2 mesh at the cursor with a
translucent material, tinted by validity, snapped to the grid. Needs a
second material — the terrain's is `AlphaMode::Mask(0.5)` (`main.rs:225`),
which is a cutout, not translucency.

**E4. Commit.** City state entry + W4/W5 write + W7 re-mesh, in that order,
transactionally: if the write fails, the city entry doesn't survive either.

**E5. Demolish.** City state removal, then re-project — restoring the
terrain that was there needs the pre-placement blocks kept in the journal,
which is a D3 decision to make deliberately rather than discover.

---

## F — Streets

**F1. The road graph.** Tiles plus adjacency, on the same grid as E2.
Iteration 1 needs the graph even without logistics, because F3 and F4 both
read it.

**F2. Drag-to-build.** Click-drag from A to B, routed over the grid, with a
live preview of the tiles it would claim and their cost.

**F3. Auto-tiling.** Each tile picks its piece — straight, corner, T,
cross, end — from its neighbours in the graph, and re-picks when a neighbour
changes. Pieces can be tiny blueprints (consistent with everything else) or
generated block patterns; blueprints are the better default since it makes
roads authorable in Minecraft like buildings. Slopes are the hard part and
are a legitimate iteration-2 deferral if they bite.

**F4. Connectivity queries.** "Is this building on the road network", "what
does this road segment reach". No consumer in iteration 1 — it's the
substrate every later logistics feature needs, and it's nearly free once
F1 exists.

---

## G — UI

**G1. Build menu.** Catalogue grouped by tier, with locked entries visible
but disabled and showing what unlocks them — the Anno affordance that makes
a tier tree readable. Shows costs and production from C1 even while inert.

**G2. City panel.** Building counts, road length, and the write status —
last write, dirty regions, backup location. That last part matters more than
it sounds: the user needs to know whether what they see has actually reached
the world.

---

## H — Terraforming

**H1. Level and dig tools.** Reuses W4/W5 wholesale — it's the same write
path with a different source of block changes.

**H2. Yields.** Dug blocks become resource counts in city state. Inert in
iteration 1, like C3, but the plumbing is the same shape production will
need.

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

**R1. Don't mesh below the terrain surface. — ticket 030.** The citybuilder's
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
6. **M6 — "and it's mine"**: E5, D3, G2, H1. Demolish, undo, terraform.
7. **M7 — "the world answers back"**: I2–I7, plus `ranvil` 020. Break a
   wall in Minecraft, come back, and the building says so. The first
   mechanic that makes the round trip *matter* rather than just work — a
   good first target for iteration 2, ahead of production, because it needs
   no economy to be interesting.

Everything after that is production, resources, and the tier tree coming
alive.
