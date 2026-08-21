# 063 - Dirt road type: folder structure and RON definition

## Status
Done. Follow-up asset addition to tickets 059 (multiple road styles) and
060 (road type schema): add the first real road style, `"dirt"`, in both
halves the schema needs — the geometry directory `road_catalogue` scans,
and the `road_definition` RON file describing it.

## The change

- `assets/city/roads/dirt/` — the style directory `road_catalogue::
  load_road_catalogue_dir` scans for. Per 054/059's own scope notes, no real
  Minecraft structure-block `.nbt` exports ship in this repo yet, so the
  directory is scaffolded with a `README.md` naming the six fixed filenames
  it expects (`isolated.nbt`, `dead_end.nbt`, `straight.nbt`, `corner.nbt`,
  `t.nbt`, `cross.nbt`) rather than placeholder geometry — an empty/partial
  style directory is already tolerated (missing pieces are skipped, not
  fatal), so this doesn't block the loader.
- `assets/city/road_types/dirt.ron` — a `RoadType` for `"dirt"` (filename
  stem is the style id, per 060), matching the module doc's own example
  values (`travel_speed: 1.0`, `capacity: 4`).

## Done when

- `assets/city/roads/dirt/` and `assets/city/road_types/dirt.ron` exist.
- `dirt.ron` parses as a valid `RoadType` (name/travel_speed > 0/capacity > 0).

## Resolution

Landed as scoped. `assets/city/roads/dirt/README.md` names the six fixed
`.nbt` filenames the style directory expects (`isolated`, `dead_end`,
`straight`, `corner`, `t`, `cross`); no real structure-block export exists
yet, so the directory itself has no geometry in it — same "no real assets"
state 054/059 shipped in. `assets/city/road_types/dirt.ron` carries the
`RoadType` values from `road_definition`'s own module-doc example
(`travel_speed: 1.0`, `capacity: 4`).

One consequence worth flagging: `RoadCatalogue::styles()` only counts a
style once it has at least one piece *loaded*, not once its directory
exists — so until real `.nbt` pieces land under `assets/city/roads/dirt/`,
`dirt.ron` fails `road_definition`'s own `UnknownStyle` cross-check at
startup (logged as a `skipped` line, not fatal, same contract every other
loader in this crate uses). Updated `todo.md`'s existing 060 manual-
verification item (renamed 060/063) to describe the console output this
now actually produces, and to note what it should look like once real
geometry is added.

`cargo check` clean; `cargo test --lib road_definition` (all 7 tests) pass.
`cargo build` hit an unrelated `Access is denied` removing a locked
`block_viewer.exe` — a stale process holding the file, not caused by this
change (no Rust source was touched, only new files under `assets/`).
