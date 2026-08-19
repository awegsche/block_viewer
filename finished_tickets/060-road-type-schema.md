# 060 - Road type schema: travel speed and capacity

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 547
tests, all passing). User-directed follow-up to ticket 059: a road *style*
(`city::road_catalogue::RoadCatalogue`) is geometry only — six `.nbt`
pieces per style, nothing about the style's *properties*. Buildings already
have this split (`blueprint::catalogue::BuildingCatalogue` = geometry,
`city::definition::BuildingDefinitions` = game data); roads are missing
their half of it. Roadmap group F (`tickets/CITYBUILDER_ROADMAP.md`) is the
parent.

## Scope, per the user's own framing

**Schema only, inert — mirrors the project's own C3 boundary.** Add
`travel_speed`/`capacity` as data on a road type, loaded and validated, with
nothing in the game reading or simulating them yet. There is no
traffic/logistics system anywhere in this codebase to hang a real simulation
off, and building it is a project of its own, well past what "add two
fields" means — same reasoning the roadmap already uses for C3's production
rates and H2's yields. If a simulation is wanted later, this ticket is the
data half of it existing first, same as C1 was for buildings.

## The change

New `city::road_definition` module, mirroring `city::definition`'s shape
(schema + loader + per-file validation, RON, filename-is-the-id) but trimmed
to what roads actually need — no tech tree, no integrity/damage (group I is
building-only), no footprint resolution (a road cell's footprint is fixed at
`ROAD_CELL_SIZE`, not per-style):

```ron
// assets/city/road_types/dirt.ron
RoadType(
    name: "Dirt Path",
    travel_speed: 1.0,
    capacity: 4,
)
```

- **Filename stem is the style id, unifying with `road_catalogue`'s own
  convention** — `dirt.ron` describes the `"dirt"` style, the same directory
  name `assets/city/roads/dirt/` uses. No separate `style:` field to name
  it, for the same reason C1 dropped buildings' inline `id:` field: a
  filename is its own id, nothing for it to disagree with.
- **Validated against the geometry catalogue**: a road type's id must be one
  of `RoadCatalogue::styles()`, or it's `RoadDefinitionError::UnknownStyle`
  — the same call C1 makes for a building's `blueprint` reference. A style
  with geometry but no type file is just undescribed (not an error); a type
  file with no geometry is the error, since it can never be placed.
- `travel_speed` must be `> 0.0`, `capacity` must be `> 0` — both checked the
  same shape `city::definition::validate` already uses for cost/production.
- Loaded from `assets/city/road_types` (new directory, alongside
  `assets/city/roads`), the same "missing dir is empty, not an error, one
  bad file doesn't take down the rest" contract every other loader in this
  crate uses.
- `city::run()` loads it after the road piece catalogue (needs it for
  validation) and inserts a `RoadTypes` resource, logged the same way
  `load_building_catalogue`/`load_building_definitions` are. No consumer
  yet — same "proven, not yet used" state ticket 040 itself landed in; a
  later road-type picker (G1's build-menu equivalent for roads) and any
  actual simulation are the eventual readers. `RoadStyleSelection`'s
  `[`/`]` cycle keeps reading `RoadCatalogue::styles()` (geometry), not this
  — you need geometry to place a road regardless of whether its data file
  exists yet, same as a building's catalogue entry not depending on its
  definition existing.

## Explicitly not in scope here

- **Any simulation.** Nothing computes travel time, throughput, or routes
  differently based on these fields. See the roadmap's own iteration-1
  boundary ("Production simulation... nothing consumes it").
- **Cost, tier, or a tech tree for road types.** Not asked for; roads don't
  currently have any of C1/C2's building-tier machinery, and adding it
  without a use is exactly the kind of unread field this crate's own
  conventions warn against.
- **Wiring into the build menu or a road-type picker.** No UI reads this
  yet, same as C1's own landing.

## Done when

- `cargo build`/`cargo test --lib` clean.
- A road type definition round-trips: loads, resolves against a real style
  in the catalogue, and is rejected (not silently accepted) when it names a
  style the catalogue doesn't have, or an invalid `travel_speed`/`capacity`.

## Resolution

Landed as scoped. New `city::road_definition` module: `RoadType { name,
travel_speed, capacity }`, `RoadDefinitionError` (`Read`/`Parse`/
`UnknownStyle`/`InvalidTravelSpeed`/`InvalidCapacity`/`NoFilenameStem`/
`DuplicateId`), `LoadedRoadType`, `RoadTypes` (`get`/`len`/`is_empty`/
`iter`), `load_road_types_dir`/`build_road_types`, mirroring
`city::definition`'s shape closely — same per-file validation, same
filename-is-the-id convention, same "missing dir is empty, one bad file
doesn't take the rest down" contract.

The one real design decision beyond mirroring: the RON filename stem *is*
the style id outright, with no separate field to name it — simpler than
buildings' `.ron`/`blueprint:` indirection, since `RoadCatalogue` was
already keyed by style name (a directory name), so there was nothing left
for a filename and an inline field to disagree about.

`city::run()` wires it in: loads `assets/city/road_types` right after the
road piece catalogue (needed for the `UnknownStyle` cross-check), inserts a
`RoadTypes` resource, and logs it the same way building definitions are
logged. `RoadStyleSelection`'s `[`/`]` cycle was deliberately left reading
`RoadCatalogue::styles()` (geometry) rather than `RoadTypes` — placing a
road only needs its shape to exist, not a data file, the same way a
building's catalogue entry doesn't depend on its definition.

Hit the same test-fixture gap ticket 059 did on first `cargo test`: the
fixture catalogue builder didn't create the `dirt` style subdirectory
before writing pieces into it (`write_structure_file` doesn't create parent
directories) — one `fs::create_dir_all` fixed all seven failures.

No manual-verification checklist item existed for road-catalogue startup
logging before this ticket even though 054/055/059 wired it into `run()` —
added retroactively alongside 060's own entry in `todo.md`, mirroring
039/040's existing "startup loads and logs without panicking" checks.

Summarized on `CITYBUILDER_ROADMAP.md` under F1b/F3.
