# `dirt` road style — geometry

`city::road_catalogue::load_road_catalogue_dir` scans this directory for
fixed-name `.nbt` structure-block exports, one per `RoadPieceKind`.

## Orientation: the authoring convention (ticket 066)

`city::road::select_piece` is style-blind — it answers with a piece *kind*
and the `Rotation` that turns **that kind's canonically-authored blueprint**
into the shape a cell's neighbours call for. So every piece has to be
exported at the orientation `city::road::canonical_pattern` names, or every
cell of that kind comes out rotated wrong in the world.

The rule, in one line: **every piece opens to the south.** Per file:

| file           | road surface reaches   |
|----------------|------------------------|
| `isolated.nbt` | nothing (no neighbours)|
| `dead_end.nbt` | south                  |
| `straight.nbt` | north + south          |
| `corner.nbt`   | south + west           |
| `t.nbt`        | north + south + east   |
| `cross.nbt`    | all four               |
| `stairs.nbt`   | north + south, ascending toward **north** (tickets 067/068) |

Every row above ships **twice** (ticket 071): `<name>.nbt` and
`<name>-tunnel.nbt`. See "Tunnel pieces", below.

`city::road_catalogue`'s
`the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations` reads
these files and checks exactly that, so a re-export at a different
orientation fails a test rather than showing up as a bent corner in-game.

## Cross-section and height

Each piece is `6` blocks on x and z (`ROAD_CELL_SIZE`) — that's enforced by
the loader — laid out cross-section as shoulder / kerb / surface / surface /
kerb / shoulder. Vertically the six flat pieces are `6x5x6`:

- `y=0` — subgrade (solid), written one block *below* the terrain surface
- `y=1` — the surface course a player walks on, flush with the terrain
- `y=2..4` — air, deliberate clearance that mows whatever grew over the road

`city::road_build::ROAD_PIECE_SUBGRADE_DEPTH` is the `1` in "the surface
course is one layer up"; see that module's docs.

### `stairs.nbt` (tickets 067/068)

The piece that bridges two road levels. Same `6` x/z footprint, `6x8x6` as
shipped, climbing `ROAD_STAIR_RISE = 4` blocks across the cell:

- `y=0` — subgrade under the whole cell, as above
- `y=1` — the surface course on the **south** edge: the ramp's *low* end
- `y=2..5` — one step per layer walking north, with solid fill behind each
  (not air — a player has to be able to walk up it)
- `y=5` — the surface course again on the **north** edge: the *high* end
- `y=6..7` — air clearance

A road cell records its stair's **low** end as its `base_y`, and the flat
cell on the high side records `base_y + 4` — see `city::road_build`'s
"Height" docs. The four-block rise isn't decoration: it's
`ROAD_STAIR_RISE`, and every level in a planned drag is a multiple of it, so
a re-export with a different rise leaves a lip at every ramp.
`city::road_catalogue`'s `the_shipped_stair_climbs_north_by_exactly_one_stair_rise`
checks all three of those properties against the real file.

## Tunnel pieces: `<name>-tunnel.nbt` (ticket 071)

A road cell whose terrain closes over it needs a piece with a **bore** rather
than open sky above the paving. `city::road_build` measures that as: more than
half of the cell's 36 columns are not air, in the single layer directly above
the piece that would otherwise be written. When that holds, the cell resolves
to `<kind>-tunnel.nbt` instead of `<kind>.nbt`.

What a tunnel piece has to keep identical to its surface twin:

- **the same `6` x/z footprint** — the loader enforces it either way;
- **the same canonical orientation**, per the table above. It is the same
  `RoadPieceKind`; `select_piece` picks the shape and rotation without ever
  knowing which variant will be used, and
  `the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations`
  checks the tunnel files against exactly the same rule;
- **the same subgrade and surface layers** (`y=0`, `y=1`) — the write origin
  is computed from `ROAD_PIECE_SUBGRADE_DEPTH` for both, so a tunnel whose
  paving sat on a different layer would step down at every portal;
- for `stairs-tunnel.nbt`, **the same `ROAD_STAIR_RISE = 4`** between its low
  and high surface, for the same reason.

What it should differ in: everything above the surface course. A tunnel piece
is expected to be *taller* than its five-block twin — the clearance layers
become a walled and roofed bore, and whatever is written there is what
replaces the hillside. The blocks it does *not* write are left as they were,
so a tunnel piece that stops short leaves stone hanging.

A missing `-tunnel.nbt` is not an error: `RoadCatalogue::get` never falls back
from one variant to the other, and `city::road_build` asks the catalogue
*before* recording a cell as a tunnel, so a style with none simply keeps
building its surface pieces the way it did before this existed.

## Game data

See `assets/city/road_types/dirt.ron` for this style's travel speed and
capacity.
