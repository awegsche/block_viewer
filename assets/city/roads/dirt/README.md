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

Every row above can ship up to **three** files: `<name>.nbt`, `<name>-tunnel.nbt`
(ticket 071) and `<name>-connected.nbt`. See "Tunnel pieces" and "Connected
pieces", below.

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

## Connected pieces: `<name>-connected.nbt`

A road cell that touches a currently-placed building (`city::road::touches_building`
— any of its four cardinal neighbour cells has a building's footprint in it)
resolves to `<kind>-connected.nbt` instead of `<kind>.nbt`, where one exists —
a visual cue that this stretch of road actually reaches a building rather than
just running past open ground. There's no ticket behind this one; see
`city::road::RoadPieceVariant::Connected` and `city::road_build`'s module docs
("Connected") for the exact rule.

Same contract as a tunnel piece: same `6` x/z footprint, same canonical
orientation, same subgrade/surface layers — only what's built *around* the
piece (a kerb, a lamp, paving stones, whatever signals "connected" for this
style) should differ. A missing `-connected.nbt` is not an error, for the
same reason a missing `-tunnel.nbt` isn't: the cell just keeps its plain
surface piece.

One thing this variant does **not** do, unlike a tunnel: if a building goes
up *after* the road next to it is already built, the existing cell isn't
retroactively repainted — nothing re-tiles a road cell when a building is
placed or removed nearby yet. Only a cell a drag actually writes or re-tiles
(because a neighbour's own connections changed) picks up `-connected.nbt`.

### What's shipped

`isolated-connected.nbt`, `dead_end-connected.nbt`, `straight-connected.nbt`,
`corner-connected.nbt` and `t-connected.nbt` exist, built with `ranvil-cli
struct fill` on top of a copy of each kind's plain piece. `cross-connected.nbt`
doesn't: a cross cell's four neighbours are all road cells, and a road cell
can never overlap a building's footprint tiles, so `road::touches_building`
can never be true for one — nothing could ever select it.
`stairs-connected.nbt` doesn't exist either — the ramp's side faces are
fenced (a fall hazard along the slope) and its walkable height changes with
`z`, so a side exit would need a per-row height and a fence gap rather than
one flat cut; deferred rather than guessed at blind.

Each shipped `-connected` piece adds a **`minecraft:gravel` exit** — two
blocks wide, matching the surface course's own width — cut through the
shoulder (and, on `isolated`/`dead_end`, through the two-deep grass verge
behind their curb cap too) on every side [`canonical_pattern`] does *not*
mark as open for that kind. Gravel rather than `dirt_path`: the automated
`the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations` check
reads the exact edge cells and would misread a `dirt_path` exit on a
non-canonical side as a real road connection. Gravel reads as neither, so it
survives that check, and it doubles as the piece's cosmetic "connected"
marker — the visual cue this section already asks for.

Since a road cell can't have a building on a side that's also a road
connection, every side left closed by [`canonical_pattern`] is the *only*
kind of side that could ever face a building — so cutting an exit on
every closed side (rather than guessing which one) covers every rotation
`select_piece` could ever apply the piece at.

## Game data

See `assets/city/road_types/dirt.ron` for this style's travel speed and
capacity.
