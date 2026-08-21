# 071 - tunnel tiles where a road runs under the terrain

## The problem

Ticket 067 gave a drag one height profile — one anchor level, stairs spread
over the interior to reach the other end. That's right for a road running
*over* terrain, but it says nothing about terrain running over the *road*.
Drag across the foot of a hill and the plan happily puts the surface course
four blocks under the hillside: the piece writes its own three layers of
clearance, and the player walks into a dirt wall one block later. The road is
there, it just has no hole to be in.

The user's rule for spotting that, and the one this ticket implements:

> When more than **18 of the 36** blocks in the layer **just above a cell's
> piece** are not air, the cell needs a **tunnel** tile instead of the
> surface one.

36 is a cell's own footprint (`ROAD_CELL_SIZE` squared), so this is "is the
majority of this cell roofed over" — a single layer, read at the first Y the
piece itself doesn't occupy.

## The design

**A variant, not a seventh kind.** The user is supplying `xxx-tunnel.nbt` per
existing tile, so a tunnel straight is still a *straight* — same connections,
same rotation, same `select_piece` answer. What changes is which file that
answer resolves to. So `RoadPieceKind` is untouched and a second, orthogonal
axis appears next to it:

- `road::RoadPieceVariant { Surface, Tunnel }`.
- `RoadCatalogue` is keyed `(style, kind, variant)`; filenames gain a
  `-tunnel` suffix (`straight-tunnel.nbt`, `stairs-tunnel.nbt`, ...).
- A missing tunnel file is skipped exactly like any other missing piece.

**Decided once, at placement — never re-derived.** This is the same trap
`base_y` fell into in ticket 065 and `ascent` in 067, and it bites harder
here: writing the tunnel piece *carves the cover away*, so a cell re-sampled
after its own write reads back as open sky, flips to `Surface`, and the next
re-tile fills the tunnel back in with hillside. The variant is therefore
recorded on `state::RoadCell` and kept by `add_road_cell` for a cell that
already exists, alongside the other two.

That makes it `city.ron` schema **6** (a version-5 file has no `variant`, and
the version check is an equality test — refused, not defaulted).

**Only when the style can actually render it.** Mirroring
`stair_available`: a cell is only marked `Tunnel` if the catalogue really
holds `(style, kind, Tunnel)`. A style with no tunnel exports yet keeps
building the surface pieces it has — the pre-071 behaviour as the degenerate
case, not a second mode and not a hole in the road where an unresolvable
piece used to be.

**Where the sample is taken.** The layer above a cell's piece is
`cell_write_origin(cell, base_y).y + piece.size.y` — read off the *surface*
piece's own blueprint rather than a constant, because a stair is `6x8x6`
where the flat pieces are `6x5x6` and hardcoding either would test the wrong
layer for the other. "Not air" is literal (`BlockRegistry::AIR`), not
`grid::is_ground`'s clutter-skipping notion: a cell roofed by 36 leaves is
as unwalkable as one roofed by 36 stone.

## Work

1. `road::RoadPieceVariant`, `Serialize`/`Deserialize` like `road::Direction`.
2. `road_catalogue`: key on `(kind, variant)`, `filename_for(kind, variant)`,
   loader loops both axes, `get` takes a variant.
3. `state::RoadCell::variant` + `add_road_cell`'s fifth argument, kept on a
   re-add like `base_y`/`ascent`.
4. `persistence`: `SavedRoadCell::variant`, `CURRENT_VERSION` 5 -> 6.
5. `grid::block_at` — a decoded-world block read at an explicit Y (`None`
   only for an undecoded column; an absent section is air).
6. `road_build`: `CellPlan::variant`, a `plan_tunnels` pass after
   `plan_drag`, `ROAD_TUNNEL_COVER_MAJORITY`, and the variant threaded
   through `road_write_edit`, the preview cache key and `cell_transform`'s
   callers.
7. `assets/city/roads/dirt/README.md` and the citybuilder roadmap's F
   section.

## Done when

- `cargo check` and `cargo test` pass.
- A drag through a hill records `Tunnel` cells and writes the `-tunnel`
  pieces, once the user drops the files in; with none present it builds
  exactly what it built before.
- The "carve, then re-tile, then fill the tunnel back in" regression is
  covered by a test, not just by this document.
