# 047 - Ghost preview and validity

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 400 tests,
up from 385). See the Resolution.

## Part of
Roadmap E3 (`tickets/CITYBUILDER_ROADMAP.md`), the third ticket in group E
(placement). Depends on E1's picking (ticket 045, `HoveredBlock`), E2's
terrain fit (ticket 046, `fit_footprint`), D1's occupancy (ticket 042,
`City::is_tile_free`), and B2/B3's blueprint meshing/rotation (tickets 037,
038). Feeds E4 (commit), which needs the same rotated blueprint and the same
validity signal this ticket computes, just on a click instead of every frame.

## Problem

Nothing shows the player what they're about to build. `HoveredBlock` knows
what tile the cursor is over, `fit_footprint` knows whether the ground under
a footprint is buildable, and `City` knows whether its tiles are free — but
nothing draws a building there, and nothing combines those two very
different "can't build here" answers (terrain, occupancy) into one signal.

## Goal

At the hovered tile, when a building is selected, show the B2 mesh of that
building (rotated per B3 if the player has rotated it) with a translucent
material tinted by validity — green if it would actually be placeable there
(terrain fits *and* every tile is free), red otherwise. No world writes; E4
is the only thing that commits.

## Scope

- **A minimal interim selection.** G1's build menu (roadmap group G) doesn't
  exist yet, so there's currently no way to say "I want to place a
  lumberjack's hut." A `PlacementSelection` resource (`catalogue_id`,
  `rotation`) plus a small keyboard system — number keys 1-9 pick the *n*th
  catalogue entry (sorted by id, for a stable mapping), `R` rotates 90°
  clockwise, `Escape` clears the selection — is the stand-in. This is
  explicitly not G1; it's the least that lets E3 (and later E4) be exercised
  and demoed before a real menu exists, the same way W8 proved the write
  path with a paint command before any UI did.
- **A ghost entity, spawned once, updated in place.** Not despawned/respawned
  every frame — its `Mesh3d`/`MeshMaterial3d`/`Transform`/`Visibility` are
  overwritten, so a held selection doesn't churn entities 60 times a second.
- **Mesh caching keyed by `(catalogue id, Rotation)`.** `rotate_blueprint` +
  `mesh_blueprint` only run once per combination the player actually visits,
  not once per frame at the cursor — `Rotation` needs `Hash` for this, which
  it doesn't derive yet.
- **Two translucent materials, not a per-vertex tint.** A `valid` (green)
  and an `invalid` (red) `StandardMaterial`, both sharing the atlas texture
  cloned off `chunk_pipeline::TerrainMaterial`, `AlphaMode::Blend` instead of
  the terrain's `Mask(0.5)` cutout. The ghost entity's material handle swaps
  between them; the mesh itself (and its vertex-colour biome tint from B2)
  is untouched.
- **Validity combines two independently-owned answers.** `fit_footprint`
  (terrain) and `City::is_tile_free` (occupancy) each answer one question;
  this ticket is what ANDs them together for the ghost, not a new third
  check.
- **Placement height.** `Fits { base_y }` from E2 wins when the fit
  succeeds; a refused fit still shows a (red) ghost, at the hovered block's
  height, so the player sees *something* rather than nothing when they're
  somewhere unbuildable.

## Watch out

- **A rotation failure must not spam or panic.** Not every blueprint's
  palette rotates cleanly at every angle (B3's `UnrotatableProperty`) — cache
  the failure alongside successes (`Option<Handle<Mesh>>` in the cache, not
  just `Handle<Mesh>`) so a bad combination is logged once and then quietly
  hides the ghost, not re-attempted and re-logged every frame.
- **Don't rebuild the biome tint table every frame.** `mesh_blueprint` needs
  a `BiomeColors` (B2 has no per-block biome to look one up against — see
  its module docs); resolving one means walking `world::build_biome_tint_table`
  against the whole interned biome registry. That only belongs inside the
  mesh-cache-miss path, not the per-frame system body, or every idle frame
  with the cursor still pays for it.
- **Two `Res`/`ResMut` of the same `Assets<T>` in one system is a panic at
  app-build time**, not a runtime one — reading a handle's current value
  (e.g. the terrain material's texture, to build the ghost materials off it)
  has to go through the one `ResMut<Assets<StandardMaterial>>` the system
  already takes, not a second `Res` alongside it.
- **Order after `city::picking`'s system**, not just after `camera::CameraSet`
  — otherwise the ghost reads last frame's `HoveredBlock`, a frame of lag
  that's invisible most of the time and exactly the kind of thing that's
  obvious the one time it isn't. `picking` has no exported ordering label
  today; add one rather than reaching for an implicit/accidental order.

## Out of scope

- A real build menu (G1) — the numbered-key selection here is a stand-in,
  not the feature.
- Committing a placement (E4) — nothing here calls `City::place_building` or
  touches `WorldEdit`.
- Auto-levelling or otherwise reacting to a refused fit beyond showing red
  (H1, later, and opt-in — same call E2 already made).
- Any UI beyond the ghost mesh itself (no cost/name label, no tooltip — G1's
  job once it exists).

## Done when

- `cargo build`/`cargo test` clean.
- Tests: validity is true on flat, free ground and false when the terrain is
  too steep or a tile is already occupied; a refused fit still produces a
  ghost, positioned at the hovered height; the mesh cache reuses the same
  `Handle<Mesh>` on a second call with the same id/rotation rather than
  rebuilding; an unrotatable property is cached as a failure and doesn't
  panic on a second lookup; the keyboard selection picks entries by sorted
  index, rotates, and clears on Escape.
- Manual verification (window open, a real save, real buildings) goes in
  `../todo.md` — whether the ghost actually reads as "translucent and
  tinted" against real terrain, and whether the ~1-block clip E2's fit
  already allows looks acceptable with a mesh sitting on top of it, aren't
  things a unit test can judge.

## Resolution

Landed as scoped, in a new `city/placement.rs` (private module, declared
next to `definition`/`grid`/`journal`/`persistence`/`picking`/`state` in
`city/mod.rs`), plus small, justified additions to three files it depends on
rather than working around their gaps locally:

- `blueprint::rotate::Rotation` picked up `Hash` (the ghost mesh cache's key
  is `(String, Rotation)`) and `Default`/`#[default] Deg0` (`PlacementSelection`
  needs a starting rotation before the player rotates anything) — both land
  directly on the type, the same call ticket 043 already made adding
  `Serialize`/`Deserialize` to it rather than a mirror enum.
- `city::grid::{ground_height_at, fit_footprint}` and
  `city::state::City::is_tile_free` lost their `#[allow(dead_code)]`
  markers — this ticket is their first caller reachable from `city::run`
  (see the note in `city/grid.rs`/`city/state.rs`'s own module docs on why
  everything under a private `city::*` submodule counts as dead until
  something on that path actually calls it, `pub` or not).
- `city::picking` gained a `PickingSet` `SystemSet` label so
  `update_ghost_preview` can order `.after(PickingSet)` rather than an
  implicit/accidental ordering — `HoveredBlock` needs to be *this* frame's
  answer, not last frame's, and picking previously exported no label to
  order against.

**The interim selection** (`PlacementSelection`, `cycle_selection`) is
exactly the keyboard stand-in the ticket scoped: number keys 1-9 (sorted
catalogue ids, not `HashMap` iteration order — the same "don't depend on
that" call `blueprint::catalogue::build_catalogue` already made for its own
duplicate-id/sort-by-path rule) pick an entry, `R` rotates clockwise through
the same four-way cycle `blueprint::rotate`'s own geometry does (a small
local `rotate_clockwise` rather than exposing a "next rotation" method on
`Rotation` itself, since nothing else needs one), `Escape` clears. Guarded
by `camera::EguiInputCapture::keyboard`, the same guard `camera.rs`'s own
input systems use, even though `city::run()` doesn't add `EguiPlugin` today
— cheap now, and correct the moment it does.

**Validity is `resolve_placement` ANDing two independently-owned answers**,
per the ticket's central decision: `grid::fit_footprint` (terrain, E2) and
`state::City::is_tile_free` (occupancy, D1) are each asked, never
reimplemented. A refused fit still returns a placement — at the hovered
block's height (`hovered.y + 1`, the same "+1" `ground_height_at` already
applies), invalid — so the ghost always has *something* to show rather than
vanishing exactly when the player most wants feedback.

**Caching turned out to need two different shapes.** The mesh cache
(`GhostState::meshes: HashMap<(String, Rotation), Option<Handle<Mesh>>>`)
caches failure alongside success — an `UnrotatableProperty` blueprint at a
given rotation is logged once and then quietly returns `None` from the
cache on every later frame the player leaves that combination selected,
rather than re-attempting `rotate_blueprint` and re-printing the warning 60
times a second. `Deg0` never touches `rotate_blueprint` at all (mirrors B3's
own identity shortcut in `rotate_blueprint` itself), which is exactly what
the "caches a rotation failure" test's last assertion exercises: the same
blueprint that fails to rotate at `Deg90` still meshes fine at `Deg0`, as
two separate cache entries. The material cache (`GhostState::materials`) is
simpler — built once, lazily, the first frame anything needs to be shown,
off whatever texture `chunk_pipeline::TerrainMaterial` currently uses via a
single `ResMut<Assets<StandardMaterial>>` read-then-add (a second `Res`
alongside it, to read the terrain material's texture handle before adding
the new ones, is a conflicting-access panic at app-build time, not a
borrow-checker error caught before running — worth naming since it's not
obvious from the error alone).

**Biome resolution stayed out of the per-frame system body** — the ticket's
other "watch out." `plains_biome_colors` (`world::build_biome_tint_table`
against the whole interned biome registry, indexed at
`BiomeRegistry::PLAINS`) only runs inside `ghost_mesh`'s cache-miss branch,
so an idle frame with the cursor sitting still costs nothing beyond the
`HashMap` lookup that says "already built."

**The ghost entity is spawned once and updated in place**
(`apply_ghost_update`), not despawned/respawned — `GhostState::entity`
tracks it, `Mesh3d`/`MeshMaterial3d`/`Transform`/`Visibility` are computed
first (by `resolve_ghost`, entirely `Commands`-free) and then either applied
to the existing entity via a `Query` or used to spawn it the first time,
which sidesteps the one-frame-late problem a spawn-then-update-next-frame
split would have had. The world-space transform
(`ghost_transform`) is a plain `Transform::from_xyz(x, y, -z)` — B2's mesh
vertices already bake the `bevy.z = -mc.z` flip per-vertex
(`world::mesh::face_geometry`), the same reason `chunk_pipeline`'s own chunk
mesh spawn only translates and never flips a second time.

**The two materials are `AlphaMode::Blend`, green/red, `unlit: true`** —
translucent rather than the terrain's own `Mask(0.5)` cutout (a ghost is
meant to be seen through), and unlit so the tint reads the same regardless
of the sky's current light level (ticket 011's day/night cycle) rather than
going dark and unreadable at night. Both share the terrain material's
current texture handle rather than loading the atlas a second time.

Testing: 15 tests in `city::placement::tests`. `resolve_placement` (valid on
flat free ground; invalid when a tile is occupied; falls back to the hovered
height, invalid, when the fit is refused for unloaded ground). `ghost_mesh`
(builds and caches a handle; reuses the same `Handle<Mesh>` on a second call
rather than rebuilding — asserted via `Handle` equality and the cache's own
length; caches an `UnrotatableProperty` failure as `None` without panicking,
reusing `blueprint::rotate`'s own `orientation=north_up` fixture, and
confirms the same blueprint still meshes at `Deg0` as a second, distinct
cache entry). `resolve_ghost` (hidden with no selection; hidden with no
hovered block; shows the valid material on buildable free ground; shows the
invalid material when a tile is occupied) — built against a real
`BuildingCatalogue` loaded through `blueprint::write_structure_file`/
`load_catalogue_dir` over a temp directory, the same fixture shape
`blueprint::catalogue`'s own tests use, since `BuildingCatalogue`'s fields
are private and going through the public load path was simpler than adding
a test-only constructor. `rotate_clockwise` (cycles through all four and
wraps). `cycle_selection` (number key picks the sorted *n*th entry; a number
past the catalogue's size does nothing; `Escape` clears; `R` rotates) —
against a bare `App` with `ButtonInput<KeyCode>` `init_resource`'d directly
rather than `bevy::input::InputPlugin` added, mirroring `camera.rs`'s own
`rts_test_app`: `InputPlugin` clears `just_pressed`/`just_released` as part
of its real-event-processing systems, which raced a manual `.press()` called
immediately before `app.update()` and produced three failures (selection
never updating) before the fix — worth recording since it isn't obvious from
the panic message alone, only from comparing against `camera.rs`'s working
pattern.

No manual verification recorded as done — `../todo.md` carries the checklist
(does the ghost actually read as translucent and validity-tinted against
real terrain and real lighting, does rotation look right, does an
unrotatable-property combination hide cleanly) since none of that is
something a unit test can judge, per the ticket's own "Done when."
