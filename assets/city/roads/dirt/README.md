# `dirt` road style — geometry

`city::road_catalogue::load_road_catalogue_dir` scans this directory for six
fixed-name `.nbt` structure-block exports, one per `RoadPieceKind`:

- `isolated.nbt` — no neighbours
- `dead_end.nbt` — exactly one neighbour
- `straight.nbt` — two opposite neighbours
- `corner.nbt` — two adjacent neighbours (90° bend)
- `t.nbt` — three neighbours
- `cross.nbt` — all four neighbours

No real Minecraft structure-block export ships in this repo yet (same scope
note as tickets 054/059) — a missing piece here is skipped, not fatal, so
this directory can stay partially (or entirely) empty until real geometry
is exported and dropped in under the filenames above.

See `assets/city/road_types/dirt.ron` for this style's game data
(travel speed, capacity).
