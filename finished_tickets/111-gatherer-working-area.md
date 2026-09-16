# 111 - Gatherer's Hut: player-drawn working area, never digs roads or buildings

## The problem

Seen in play: a Gatherer's Hut removed the blocks of a dirt-road tile next
to it. `gatherer::next_gather_target` only reads `DecodedWorld` — it walks
every tile within `Gatherer::radius_blocks` of the footprint and digs
whatever column is highest above `floor_y`. The one exclusion is the hut's
*own* footprint; `City` occupancy is never consulted, so a road cell's 36
tiles and any neighbouring building's footprint are ordinary terrain to it.
A road piece sits one block above ground (`RoadCell::base_y`), so it is
always the topmost block on its tile and therefore always the first thing
the hut reaches for.

Two things are wrong, and this ticket fixes both:

1. **Nothing the city has claimed may ever be dug** — roads, other
   buildings, tunnels. This is a hard rule, independent of the area below.
2. **Where the hut digs is the player's decision, not an automatic radius.**
   Today the hut levels a fixed 6-block ring around itself the moment it's
   placed, with no way to say "not there". Ticket 111 replaces that with a
   working area the player draws; a hut with no area drawn does nothing.

## Design

### 1. Occupancy guard (the bug fix)

`next_gather_target` gains a `city: &City` parameter and skips any tile
where `city.occupant_at(tile).is_some()`. The existing "never the building's
own footprint" check becomes a special case of this (its own footprint is an
`Occupant::Building(id)` entry) and can go, keeping the existing test
`never_targets_the_buildings_own_footprint` as the regression test for the
general rule. `gather_edit`/`plan_dig` thread `&City` through; `dispatch_digs`
already has `city: Res<City>` in scope.

Tunnels need no special case: a `RoadPieceVariant::Tunnel` cell's tiles are
in `occupancy` like any other road cell's, so the hillside above a tunnel is
left alone too (the tunnel's evidence survives — see `RoadCell::variant`'s
own docs on why re-deriving it is destructive).

### 2. The working area: one rectangle per hut, drawn by the player

`state::WorkArea { min: IVec2, max: IVec2 }` — an inclusive axis-aligned
rectangle of block tiles in Minecraft `(x, z)`, stored on
`PlacedBuilding` as `work_area: Option<WorkArea>`. On the placement itself,
not in a side map, for the same reason `footprint` and `RoadCell::base_y`
live where they do: everything that already clones/saves/restores a
`PlacedBuilding` — the journal's demolish-undo copy, `persistence`,
`City::remove_building` — carries it for free, so demolish-then-undo brings
the area back without a new journal entry kind. Drawing an area is *not*
itself journalled/undoable (it's a setting, like a road's style, not a
build).

- `City::set_work_area(id, Option<WorkArea>)` — the one mutator.
  `place_building` always starts it as `None`.
- **Clamped to the hut's reach.** `Gatherer::radius_blocks` stops meaning
  "the area the hut digs" and becomes "how far from the footprint a drawn
  area may extend". A `WorkArea::clamp_to_reach(footprint_min,
  footprint_max, radius)` pure fn intersects the drawn rectangle with the
  footprint's `radius`-expanded box; an empty intersection is refused (the
  drag just does nothing, the old area is kept). The footprint's own tiles
  may lie inside the rectangle — the occupancy guard already skips them,
  so the player can draw straight across the hut.
- **No area, no digging.** `plan_dig` returns `None` and sets a new
  `ProducerState::NoWorkArea` (label: `"no working area"`) when
  `work_area` is `None`. Distinct from `Depleted` (which says "the area is
  done") — a fresh hut is not levelled, it hasn't been told where to work.
  The carry does not accrue while in this state, same as `BufferFull`.
- `next_gather_target` iterates the work area's tiles instead of the
  radius box; `rect_distance` is kept only as the nearest-first ordering
  key (walk out from the footprint, same as today). The `distance > radius`
  check is dropped — the clamp at draw time already guarantees it, and a
  saved area is trusted the way `base_y` is.

### 3. `radius_blocks` 6 -> 18

`assets/city/buildings/gatherer_hut.ron`: `radius_blocks: 18` (3x today's
6) with the comment rewritten — it's the cap on a drawn area now, and the
old "tighter than a warehouse's reach because it levels the site right
around it" reasoning no longer applies: the player decides how much of the
reach to use. Same rewrite on `definition::Gatherer::radius_blocks`'s doc
comment and in `CITYBUILDER_ROADMAP.md`'s gatherer entry.

### 4. The tool: "Draw working area" from the inspect panel

- `ActiveTool::DrawWorkArea` — a fifth variant, and like `Inspect` **not**
  a stop on `T`'s cycle: it's only ever entered from the inspect panel's
  button and left by committing a drag or pressing `Escape`, both of which
  return to `Inspect` with `SelectedBuilding` untouched (so the panel is
  still open showing the new area). Pressing `T` in it enters the cycle at
  `Building`, same as from `Inspect`. Every other tool's systems are
  already gated on their own variant, and `picking::update_selected_building`
  on `Inspect`, so nothing else reacts to clicks while drawing.
- **Inspect panel**: for a building whose definition has `gatherer`,
  a section after the producer lines:
  - `Working area: (x1, z1) - (x2, z2), N tiles` or a red
    `⚠ No working area - draw one` (same colour/shape as the warehouse
    warning).
  - `[Draw working area]` button -> `ActiveTool::DrawWorkArea`. While the
    tool is active the button reads `Drawing... (Escape to cancel)` and is
    disabled.
  - `[Clear working area]` (only when one is set) -> `set_work_area(None)`.
  The panel needs `ResMut<ActiveTool>` and `ResMut<City>`; mutation happens
  after the `Window::show` closure, the way `clear_buffer` already does.
- **`city::work_area` module** (new) owns the drag and the preview:
  - `WorkAreaDragState { anchor: Option<IVec2> }` — `terraform::
    update_drag_state`'s shape verbatim: anchored on left-press with
    something hovered, cleared when the tool changes or egui owns the
    pointer. The panel button's own click frame has `egui_input.pointer`
    set, so it can never start a drag by itself.
  - On left-release: `rect` from `terraform::rect_tiles` (make it
    `pub(super)`, or move it to `state` next to `footprint_tiles` — its
    natural home now that two modules want it), clamp, `set_work_area`,
    tool back to `Inspect`.
  - `Escape` while active: drop the anchor, tool back to `Inspect`, area
    unchanged.
  - **Preview via `Gizmos`** — the reason `selection::gizmo` gives for
    gizmos over a mesh applies exactly (changes every frame, no entity
    lifecycle). Its own `GizmoConfigGroup` so the depth bias stays scoped:
    - while dragging: the clamped candidate rectangle's outline, drawn per
      edge tile at that tile's `topmost_block_y + 1` so it hugs the
      terrain (a flat rectangle at one Y is unreadable on a hill — the
      same lesson 026 learned for the selection box), plus the reach box
      (footprint expanded by `radius_blocks`) faintly, so the clamp is
      visible rather than surprising;
    - whenever a gatherer is the `SelectedBuilding` (any tool): its
      committed area's outline, so the panel's numbers have a picture.
  Wired into `CityPlugin` after `PickingSet`, same chain the other tools
  use.

### 5. Persistence

`SavedBuilding` gains `#[serde(default)] work_area: Option<((i32, i32), (i32, i32))>`
and `CURRENT_VERSION` stays at 7 — decided against the module's seven-bump
precedent, on the argument its own docs ask for: a version-7 file's huts
genuinely never had an area drawn, so `None` is the truth about that file,
not a guess at it. Every earlier bump defaulted something that could be
*wrong* (a tunnel read as surface, a catalogue id read as a definition id);
this is the first field whose default is exactly right for every file that
lacks it. A v7 file must load with every hut at `None`, and a test says so.

## Out of scope

- A gizmo/ghost for the *reach box* outside of drawing mode.
- Reacting to a road/building placed *inside* an existing work area — the
  occupancy guard handles it at dig time, nothing needs re-deriving.
- Farm `radius_blocks` — untouched; a farm's radius is about tile
  buildings, not terrain.

## Files

- `src/city/gatherer.rs` — occupancy guard, work-area iteration,
  `NoWorkArea`; tests.
- `src/city/state.rs` — `WorkArea`, `PlacedBuilding::work_area`,
  `City::set_work_area`, `clamp_to_reach`; tests.
- `src/city/production.rs` — `ProducerState::NoWorkArea` + label.
- `src/city/tool.rs` — `ActiveTool::DrawWorkArea`; `T`-cycle tests.
- `src/city/work_area.rs` (new) + `src/city/mod.rs` — drag, commit, Escape,
  gizmo preview.
- `src/city/terraform.rs` — `rect_tiles` visibility (or move to `state`).
- `src/city/ui/inspect_panel.rs` — section + two buttons.
- `src/city/persistence.rs` — field, version, round-trip + refusal tests.
- `src/city/journal.rs` — check `insert_loaded`/replay paths construct
  `PlacedBuilding` with the new field (the compiler will find them).
- `assets/city/buildings/gatherer_hut.ron`, `src/city/definition.rs`,
  `tickets/CITYBUILDER_ROADMAP.md` — radius 18 + reworded docs.

## Tests

- `next_gather_target` skips a road tile, another building's tile, and the
  hut's own footprint (the existing test, now via occupancy).
- A tile inside the area is dug; one outside — even within radius — is not.
- `plan_dig` with `work_area: None` -> `None`, state `NoWorkArea`, carry
  untouched.
- `clamp_to_reach`: inside untouched; overlapping the reach edge is cut;
  fully outside -> `None`; drawn across the footprint is allowed.
- `ActiveTool::DrawWorkArea`: `T` enters the cycle at `Building`; `Escape`
  returns to `Inspect`; a release with an anchor commits and returns to
  `Inspect`; egui pointer capture never anchors.
- Persistence: an area round-trips; `None` round-trips; a file with no
  `work_area` field loads with `None`.
- `cargo check`, `cargo test`, `cargo clippy` clean.

## Manual check (todo.md)

Place a hut beside a dirt road, draw an area covering the road and a
neighbouring building, let it run: the road piece and the building must
stay intact while the ground around them is levelled; the outline must
follow the terrain; Escape mid-drag must leave the old area.
