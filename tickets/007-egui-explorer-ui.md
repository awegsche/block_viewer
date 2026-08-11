# 007 - egui UI: pick a save, jump to a coordinate, inspect a block

## Status
Open

## Depends on
005 (jumping somewhere requires streaming), 002 (block readout).

## Problem

`bevy_egui` is in `Cargo.toml` and used nowhere. The save is chosen by
`saves.into_iter().next()` — literally whichever the filesystem lists first —
with no way to pick another, no way to see where the world's regions are, and
no feedback about what's loaded.

## Scope

Add `EguiPlugin` and a UI module (`src/ui/`) with:

- **Save picker**: list saves from `mc_anvil::get_saves()` (each `SaveMeta`
  has `name`, `path`, `regions`), let the user switch; switching tears down
  spawned chunks and re-streams. `SaveMeta::get_grid_view()` already renders
  an ASCII map of which regions exist — either show it monospaced or, better,
  draw the same data as a clickable 2D region grid that teleports the camera
  to a region on click.
- **Coordinate jump**: X/Y/Z input + "go", and a readout of the camera's
  current block coords, chunk coords, and region coords. This is the single
  most useful feature for "explore a save" — everything else is optional.
- **Block inspector**: the block under the cursor (or at the crosshair),
  showing `Name` and decoded `Properties`. Use
  `ChunkRegion::get_block(x, y, z)` for this — the single-block API is
  exactly right here, and it means a bug in either path is caught by
  disagreeing with the decode layer. Note its coordinates are
  **region-local** (`x`,`z` as `usize` in 0..512) and world-Y as `i32`, so
  convert from world coords carefully.
- **Status panel**: FPS, loaded/queued chunk counts, render distance slider
  (drives 005), memory in use if cheap to get.

## Watch out

- `bevy_egui` 0.31.1 is pinned against Bevy 0.15 — check the version pairing
  before bumping either.
- egui wants pointer input when the cursor is over a panel; the camera (006)
  must not also consume it, or dragging a slider will spin the view.

## Out of scope

- Editing/saving the world. This is a viewer — everything is read-only.
- Dimension switching (nether/end), entity/block-entity browsing.

## Done when

- A save can be chosen at runtime without restarting, from a list of all
  local saves.
- Typing coordinates moves the camera there and the terrain streams in.
- Pointing at a block shows its real name and properties, matching what the
  terrain shows.
