# 042 - city state resource

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 343
tests). See the Resolution.

## Part of
Roadmap D1 (`tickets/CITYBUILDER_ROADMAP.md`) — the first ticket in group D
(city state), and the start of milestone M4 ("a city exists"). Group W (the
write path) and group B/C (blueprints, definitions) are both done; this is
the first ticket to give the game itself a state to hold.

## Depends on
- 027's `city` module — the wiring point, same as 039/040.
- 038's `blueprint::Rotation` — a placed building's orientation is this type,
  not a new one invented here.
- 039/040's `CatalogueEntry`/`LoadedBuilding` footprint convention
  (`IVec2 = (size.x, size.z)`, Y doesn't factor into a footprint) — the
  occupancy grid inherits it rather than reinventing what a footprint means.

## Problem

Nothing in the game owns any city state yet. `BuildingCatalogue` (039) and
`BuildingDefinitions` (040) describe what *could* be built; nothing describes
what *has* been. The roadmap's rule this whole project follows — "the city
state is authoritative, the blocks in the world are a projection of it" —
needs a place to live before E (placement) or F (streets) can be built on top
of it, and the roadmap says so explicitly: "D1 before E and F, always."

## Goal

A `City` resource: placed buildings (a definition id, an origin, a rotation),
a set of road tiles, and a footprint occupancy grid so "is this tile free"
is a `HashMap` lookup rather than a scan over every building. Pure data plus
the invariants that keep it consistent — no picking (E1), no terrain fit
(E2), no ghost preview (E3), no world-write glue (E4), no save/load (D2), no
journal/undo (D3). Those all build on this; none of them are this ticket.

## Scope

- `city::state::BuildingId` — a `u64` newtype identifying one *placed*
  building instance, distinct from the `String` definition id
  (`BuildingDefinitions`/`BuildingCatalogue` key) many instances can share.
  Assigned monotonically by `City::place_building`.
- `city::state::PlacedBuilding { definition: String, origin: IVec3, rotation:
  Rotation, footprint: IVec2 }` — `footprint` is the *unrotated* footprint
  (what the catalogue/definition already resolved), stored alongside the
  placement so removal doesn't need the caller to look it back up.
- `city::state::Occupant` — `Building(BuildingId)` or `Road`, what a tile in
  the occupancy grid holds.
- `city::state::PlacementError::TileOccupied { tile: IVec2, by: Occupant }`.
- `footprint_extent(footprint: IVec2, rotation: Rotation) -> IVec2` — 0°/180°
  keep the footprint's axes, 90°/270° swap them. The horizontal counterpart
  of the axis swap `blueprint::rotate::rotate_blueprint`'s grid remap already
  performs on the block grid itself.
- `footprint_tiles(origin: IVec3, footprint: IVec2, rotation: Rotation) ->
  impl Iterator<Item = IVec2>` — every Minecraft `(x, z)` tile a footprint
  covers at `origin`, rotation-aware.
- `City` (a `Resource`, `Default`):
  - `place_building(definition, origin, rotation, footprint) ->
    Result<BuildingId, PlacementError>` — computes every tile first, checks
    all of them are free, and only then mutates. All-or-nothing, the same
    "plan before apply" shape `edit::plan`/`edit::route` already use for the
    write path — a placement refused on tile 40 of 40 must not have already
    occupied the first 39.
  - `remove_building(id) -> Option<PlacedBuilding>` — frees exactly the tiles
    the removed building's own record covers.
  - `add_road(tile) -> Result<(), PlacementError>` — idempotent on an
    existing road tile, refused against a building or against nothing free
    ... refused against anything else occupied.
  - `remove_road(tile) -> bool`.
  - `is_tile_free`, `occupant_at`, `building(id)`, `buildings()`, `roads()`,
    `len`, `is_empty` — read-only queries, the same surface shape
    `BuildingCatalogue`/`BuildingDefinitions` already establish.
- `city::run()` wiring: `.insert_resource(City::default())`. No consumer
  yet — same "proven, not yet used" state 039/040 landed in; E1-E4 are what
  will call `place_building`.
- Unit tests: a building occupies exactly its footprint's tiles; a 90°/270°
  rotation swaps the occupied extent; an overlapping placement is refused
  and leaves the grid exactly as it was before the attempt (no partial
  occupation); a road can't be placed on a building's tile and vice versa;
  placing the same road tile twice is a no-op; removing a building frees its
  tiles for reuse; removing an unknown id is `None`, not a panic.

## Watch out

- **Check every tile before occupying any of them.** A footprint spans many
  tiles; refusing at tile *k* after already marking `1..k` occupied leaves
  the grid holding a phantom building nobody placed. Collect the tile list,
  check it whole, then mutate — mirrors W5's "plan every region before
  applying any of them."
- **Rotation swaps the footprint's axes, not just the mesh.** `footprint`
  on `PlacedBuilding`/`CatalogueEntry` is always the *unrotated* size; every
  tile computation goes through `footprint_extent`/`footprint_tiles` rather
  than reading `footprint.x`/`footprint.y` directly, or a rotated building
  occupies the wrong rectangle.
- **`origin` is Minecraft world coordinates, `(x, z)` horizontal.** Same
  convention `edit`/`selection` already settled on — no `bevy.z = -mc.z`
  flip anywhere in this module. That flip belongs at the render boundary
  only.
- `BuildingId` is not the same thing as a definition id (`"house01"`). Two
  houses placed side by side share a definition id and must not share a
  `BuildingId` — that's what the monotonic counter is for.

## Out of scope

- D2 (save/load) — `City` lives in memory only; nothing persists it next to
  the world yet.
- D3 (journal, undo, reconciliation) — no record of *why* a tile changed,
  only its current state.
- E1-E5, F1-F4, G1-G2 — picking, terrain fit, ghost preview, commit,
  demolish, the road graph, drag-to-build, auto-tiling, connectivity
  queries, and both UI panels. All future callers of `City`; none of them
  exist yet.
- I1's as-built baseline — a `PlacedBuilding` records what the city *intends*
  is there, not what was actually written to the save (there is no writer
  wired to `City` yet).

## Done when

- `city::state::City` exists with the API scoped above, and `city::run()`
  inserts it as a resource.
- Placement is all-or-nothing across a multi-tile footprint, verified by a
  test that provokes a refusal partway through and checks the grid is
  unchanged.
- Rotation-aware tile computation is verified for at least one 90° case.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed, in a new `city::state` module (private, same visibility
as `city::definition` — nothing outside `city` reaches either type yet).

`BuildingId(u64)` is a newtype distinct from a definition id, assigned
monotonically by `City::place_building` and never reused. `PlacedBuilding`
carries its *unrotated* `footprint` alongside `origin`/`rotation` so
`City::remove_building` can recompute exactly the tiles to free without the
caller re-resolving it against a catalogue. `Occupant` is `Building(BuildingId)
| Road`; `PlacementError::TileOccupied { tile, by }` is the one way a write
is refused.

`footprint_extent`/`footprint_tiles` are the rotation-aware tile
computation: 0°/180° keep a footprint's `(x, z)` axes, 90°/270° swap them —
the horizontal counterpart of the axis swap `blueprint::rotate_blueprint`'s
grid remap already performs on the block grid. Every occupancy operation
goes through `footprint_tiles` rather than reading `footprint.x`/`.y`
directly.

`City` holds `buildings: HashMap<BuildingId, PlacedBuilding>`, `roads:
HashSet<IVec2>`, and one `occupancy: HashMap<IVec2, Occupant>` both are
checked against. `place_building` computes the full tile list first, checks
every tile is free, and only then mutates — all-or-nothing, mirroring
`edit::route::apply_routed`'s "plan every region before applying any of
them." `add_road` is idempotent on an existing road tile and refused
against anything else occupied; `remove_building`/`remove_road` free exactly
what they cover. Read-only queries (`is_tile_free`, `occupant_at`,
`building`, `buildings`, `roads`, `len`, `is_empty`) round out the surface,
matching `BuildingCatalogue`/`BuildingDefinitions`' shape.

`city::run()` inserts `City::default()` as a resource. No consumer yet —
same "proven, not yet used" state 039/040 landed in — and unlike those two,
there's nothing to load from disk or log at startup (a fresh city is always
empty), so the wiring is a bare `insert_resource` rather than a summary
print. Every write/query method past `Default` carries `#[allow(dead_code)]`
with a one-line pointer to the module docs' "No caller yet" section, rather
than a blanket module-level allow — matches the per-item convention already
used in `blueprint::structure`/`city::definition` rather than introducing a
new one.

10 new tests in `city::state::tests` (343 total, up from 333): exact
footprint occupation, a 90° rotation swapping the occupied extent
(`footprint_extent` checked directly too), an overlapping placement refused
with the grid left completely unchanged (the all-or-nothing property, not
just the refusal), a road refused against a building and vice versa, adding
the same road tile twice as a no-op, a removed building's tiles becoming
placeable again, removing an already-removed id returning `None` rather than
panicking, `remove_road`'s before/after report, and `buildings()`/`roads()`
iterating exactly what was added.

No manual/in-game check needed — nothing is spawned, rendered, or read from
disk; `City` is pure in-memory state with no world interaction yet. Nothing
added to `todo.md` for this ticket for the same reason 039/040's later
entries only ask for a console glance: there's no new console output here to
glance at, since the resource starts empty and logs nothing.
