# 127 - Shaft torch sat in the lining wall, hanging on raw rock

Follow-up to 126. `MineFrame::lining_target` returned
`ShaftBlock::WallTorch` for the *lining cell* behind a torch step, so the
lining block itself became the torch: a one-block niche in the shaft
wall, with the torch hanging on whatever the world has behind the lining
- `SealIfNotSolid` never runs for that cell, so behind a cave it floats
in a hole. `MINES_DESIGN.md` ("Torches: on the lining wall ... Wall torches
need a solid block behind them; the lining guarantees one") wants the
torch *in front of* the lining, not instead of it.

## Fix

The torch spot moves one block inward to the ring cell above the step it
belongs to (`stair_y + 2`, the air column over ring tile `index`), facing
into the shaft; `lining_target` falls through to `SealIfNotSolid` there,
so the block it hangs on is guaranteed solid. `ring_target` takes
`torch_spacing`, and the spot is dug (costed) like any air cell before
the torch is written. Same step selection as before: ring step number
(counted across revolutions) a multiple of `torch_spacing`.

Shafts already sunk keep their niche torches - flights aren't re-run.
