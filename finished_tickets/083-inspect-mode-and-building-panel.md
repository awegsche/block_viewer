# 083 - Inspect mode: default tool, per-building panel

## Status
Done — `cargo test --bin citybuilder --lib` green (456 `city::` tests, 0
failed). Landed as designed below; one implementation choice worth naming:
[`building_name`](../src/city/ui/inspect_panel.rs) reads through
`City::definition_of` rather than `PlacedBuilding::definition_id` directly,
which finally gives that lookup (dead code since ticket 076) a real caller.

Manual/visual verification (window opens, click-to-select actually feels
right) is a to-do in `../todo.md` — see CLAUDE.md's "Manual/visual
verification".

## Depends on
082 (the menu this wires into; not a hard code blocker).

## Why
Left-click currently always means "commit the selected building" while
`ActiveTool::Building` is active — the historical default. 082 makes "pick a
category, then a building" how you *enter* placement, so the resting state
needs to be something else: click a placed building, see what it's doing.
That resting state is `ActiveTool::Inspect`.

## `ActiveTool` changes
New variant, becomes the `#[default]`:

```rust
pub enum ActiveTool {
    #[default]
    Inspect,
    Building,
    Road,
    Terraform,
}
```

`T` keeps cycling the other three (`Building -> Road -> Terraform ->
Building`) — `Inspect` isn't in that cycle; reached only by clearing a
placement or finishing one. Existing `defaults_to_building` test becomes
`defaults_to_inspect`; `t_cycles_...` starts asserting from `Building`, not
from the default.

Entering `Building`/`Road`: the build menu's row click handlers (082) also
set `*tool` to match. Leaving them: `Escape` (already wired to clear
`PlacementSelection`) also resets `*tool = ActiveTool::Inspect`. A successful
commit does **not** auto-revert — placing several of the same building in a
row is the point of picking it once.

## Selection
New resource, `city::picking::SelectedBuilding(Option<BuildingId>)`. A
system gated on `ActiveTool::Inspect` (mirroring how `commit`/`road_build`/
`terraform` already gate on their own tool) reads a left-click plus
`HoveredBlock`: `City::occupant_at` resolves the id (same lookup `demolish`
uses). A building tile selects it; a road tile or empty ground clears it; a
missed click (nothing hovered at all) leaves the current selection alone —
same "a missed click does nothing" precedent as the viewer's selection box
(ticket 020).

## Inspect panel
New `city::ui::inspect_panel`, a third window alongside Build/City. Nothing
shown when `SelectedBuilding` is `None`. Otherwise, for the selected
`PlacedBuilding`:

- name (via `City::definition_of`, falling back to the catalogue id the same
  way `build_menu::requirement_label` already does), position, rotation.
- if `ProductionState::get(id)` is `Some`: the `ProducerState` label
  (`Running`/`Starved`/`BufferFull`) and the buffer's contents, one line per
  item (`count / capacity`).
- **no health field** — Group I doesn't exist yet. A stub number now is
  something to rip out later rather than fill in; revisit when I4 lands.

Doesn't add the "Place Farm Tile" button — that's 084, since it needs the
hub schema this ticket has no reason to touch. (084 landed ahead of this
ticket and left that button as an open gap between the two — still open.)
