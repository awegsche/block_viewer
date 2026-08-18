# 057 - Terraforming: dig and level tools

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 523
tests, up from 509). See the Resolution.

## Part of

Roadmap group H (`tickets/CITYBUILDER_ROADMAP.md`): H1 ("Level and dig
tools. Reuses W4/W5 wholesale — it's the same write path with a different
source of block changes"). The last piece milestone M6 ("and it's mine":
E5, D3, G2, H1) is still waiting on — E5, D3 and G2 are all done.

Also closes a gap `city::grid`'s own docs left open: E2's `fit_footprint`
refuses uneven ground rather than levelling it, and names H1 as "what would
later let a player fix a steeper site by hand."

## Problem

There is no way to change terrain shape in the citybuilder short of placing
a building on top of it (which only ever writes the building's own
blueprint, air included). A player who wants to clear a hill, fill a
hollow, or flatten a lumpy building site before E2 will refuse to fit a
footprint there has no tool for it.

## Goal

Two left-click-drag tools, gated the same way `city::road_build`'s drag
already is (a third `ActiveTool`, active only while selected):

- **Dig**: clears the topmost block from every tile in the dragged
  rectangle, one layer per commit. Repeated drags over the same area dig a
  pit; run over a tree or a fence, it clears it same as a building's own
  write already does.
- **Level**: flattens the dragged rectangle to the height of the tile the
  drag started on — digging down the high tiles, filling the low ones.

## Scope

- A new `ActiveTool::Terraform` (extends `city::tool`'s existing
  Building/Road cycle to three).
- A `city::terraform` module: drag-rectangle tracking (mirroring
  `road_build::RoadDragState`'s click/hold/release shape, but a full
  rectangle rather than an L-path — there's no piece catalogue or
  connectivity here to keep a path cardinal for), the two edit-building
  functions, and the same claim-nothing/apply-async/poll shape
  `commit`/`road_build` already use — reusing `commit::apply_building_edit`
  directly rather than a third copy of the region-cache dispatch.
- A key to flip Dig/Level while the tool is active.
- `WriteKind::Terraform` so the city panel's existing status line picks it
  up for free.

## Out of scope

- **No city-state entry, no journal record.** Unlike a building or a road
  cell, dug/levelled terrain isn't tracked in `state::City` — there's
  nothing to claim, nothing to roll back on a failed write beyond the write
  itself never having happened. No undo for a terraform edit either, the
  same gap `road_build`'s own docs already accept for roads ("Not
  journaled").
- **No preview mesh.** `road_build` and `placement` both spawn a live ghost;
  this ticket ships console-only feedback the same way W8's original
  paint/fill command did before any tool grew a preview. A follow-up ticket
  if it's worth the mesh/material code a translucent rectangle would need.
- **No brush radius, no held-key continuous digging.** One rectangle per
  drag, one commit per release.
- **H2 (yields)** — dug/filled blocks don't become resource counts. Still
  inert, like C3.

## Done when

- `cargo build`/`cargo test --lib` clean.
- Tests cover: the rectangle both corners produce (including a
  single-tile/no-drag click, and a drag toward negative coordinates); dig
  clearing the topmost block per tile and skipping a tile with nothing to
  clear; level raising a low tile with fill and lowering a high tile to the
  anchor's own height, leaving an already-level tile untouched; level
  refusing when the anchor tile itself isn't loaded; the poll glue recording
  a `WriteStatus` success/failure and firing `ChunksEdited` on success.
- `city::tool`'s three-way cycle (`Building -> Road -> Terraform ->
  Building`) is tested the same way its two-way cycle already is.

## Resolution

Landed as scoped, as one new module plus the small cross-cutting touches the
scope section already named.

**`ActiveTool::Terraform`** (`city::tool`): `T` now cycles three ways
(`Building -> Road -> Terraform -> Building`) instead of flipping two. No
other tool's own gating needed to change — `placement`/`commit` already only
act on `Some(ActiveTool::Building)` (or `None`), `road_build` only on
`Some(ActiveTool::Road)`; a third variant they don't match on just falls
through the same "not my tool" path both already had.

**`city::terraform`**, mirroring `road_build`'s click/hold/release shape
(`TerraformDragState::anchor`, set on press, read and cleared on release) but
widened from an L-shaped cell path to a plain axis-aligned rectangle
(`rect_tiles`) — there's no piece catalogue or cardinal-adjacency constraint
here to keep a path straight for, so the natural shape for "an area of
ground" is the rectangle both drag corners describe, not a path between them.
Two pure edit-builders, both against `topmost_block_y`
(`world::ChunkColumn::topmost_non_air`, not `city::grid`'s clutter-skipping
`is_ground` — a dig or a level clears a tree exactly like it clears stone,
unlike a footprint fit reading past it):

- `dig_edit(rect, world)`: clears the topmost block at every tile that has
  one, skipping a tile whose chunk isn't decoded rather than refusing the
  whole rectangle — a drag reaching the streamed edge digs what it can.
- `level_edit(anchor, rect, world)`: reads `anchor`'s own topmost-block
  height as the target, then per tile either digs down to it (air) or fills
  up to it (`minecraft:dirt`, fixed — there's no material inventory yet to
  spend from, the same iteration-1 boundary C3/H2 already draw around
  production), leaving an already-level tile untouched so a level of flat
  ground writes nothing. Refuses only when `anchor` itself is unloaded (no
  target height to level to); any other unloaded tile in the rectangle is
  skipped the same tolerant way `dig_edit` skips one.

Neither reuses `WorldEdit::fill` despite that function's own doc comment
naming terraforming as an eventual user — `fill` is one block everywhere in
a bounds, and both tools here write a *different* value at every position
(whatever height that tile's own terrain happens to need), so a hand-built
`WorldEdit` was the actual fit.

Committing reuses `city::commit::apply_building_edit` directly — the same
region-cache dispatch `commit`/`road_build` already use, `EditPolicy {
allow_dirty_regions: true, ..default() }` since an earlier building/road/
terraform edit may well have already dirtied the same region. **No `City`
entry, no journal record**: unlike a building or a road cell, dug or
levelled terrain isn't tracked in `state::City` at all, so there's nothing
to claim before the write starts and nothing to roll back if it fails —
`poll_terraform`'s failure arm is only a `WriteStatus` line, simpler than
`poll_commit`/`poll_road_build`'s own rollback logic because there's
genuinely nothing here to undo. Also, deliberately, no undo/journal entry —
the same gap `road_build`'s own docs already accept for a road cell ("Not
journaled").

**No preview mesh** — console output only, the same state ticket 035's
original paint/fill command shipped in before any tool grew a translucent
ghost. Left as a named follow-up in the ticket's own "Out of scope" rather
than attempted here.

**`WriteKind::Terraform`** (`city::write_status`) and its city-panel verb
("Shaped") slot into the existing status line and undo/save sections with no
further changes — the panel's own `write_status_section`/`kind_verb`
already dispatch on `WriteKind` generically. `build_menu`'s key legend
gained `T` (previously undocumented anywhere in the UI) and `Z`/drag rows
for the new tool.

`ChunkColumn::topmost_non_air` loses the `#[allow(dead_code)]` ticket 052
left on it when `city::grid` moved to the clutter-aware `topmost_matching` —
`city::terraform` is a real caller again, and wants the literal predicate
`topmost_non_air` already was.

14 new tests (`city::terraform`: 3 for `rect_tiles`, 2 for `topmost_block_y`,
2 for `dig_edit`, 5 for `level_edit`, 2 for the `poll_terraform`/
`WriteStatus`/`ChunksEdited` glue) plus one `city::tool` test rewritten for
the three-way cycle. 509 to 523 per `cargo test --lib`; `cargo build` stays
warning-clean.
