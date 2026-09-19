# 126 - Gallery torch floats mid-corridor on the north arm

Reported from walking a mine in Minecraft: gallery torches hang in the
middle of the corridor instead of on the wall.

## Cause

`slice_geometry`'s `Slice::Gallery` branch builds the `TorchSpot` from
`row_z(arm, row)[0]`, assuming that's the north tile. `row_z` is
`[secondary_z(d0), secondary_z(d0 + 1)]`, and `secondary_z` *decreases*
with distance on the north arm, so there `z[0]` is the **south** tile.
`wall = z[0] - 1` is then the gallery's own north tile: the planner
samples pre-dig rock, decides the wall is solid, and writes
`wall_torch[facing=south]` on the south tile — attached to a block the
same edit turns to air. South-arm galleries are correct.

The tests in `plan.rs` use `Arm::North` but assert relative to
`torch.wall` / `torch.fallback_floor`, so they never noticed the wall was
inside `excavate`.

## Fix

Use the row's `z_min` (already computed for `excavate`) as the north
tile, per `MINES_DESIGN.md` ("north wall only, `z = row_min - 1`"):
`wall = (x, L+2, z_min - 1)`, `fallback_floor = (x, L+1, z_min)`.

Tests: a layout test that the torch wall is outside `excavate` and the
fallback floor tile is inside it, for both arms; and the plan tests gain
the same invariant.
