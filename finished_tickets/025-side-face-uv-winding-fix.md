# 025 - Side-face texture UVs were inconsistently wound

## Status
Done (see Resolution)

## Origin

Not a planned ticket — raised as a side observation while implementing 018
("side textures seem rotated by 90 degrees. is there a problem with
Minecraft's y is up?"). Investigated and fixed in the same session.

## The bug

Short answer: no, not a Y-up/axis-mapping problem. `src/world/mesh.rs`'s
`push_quad_offset` applied one fixed UV corner order —
`(u0,v1),(u0,v0),(u1,v0),(u1,v1)` — to every block face, but
`face_geometry`'s four corners are listed starting from a *different*
physical corner per face (whatever keeps the shared
CCW-from-outside/`(0,1,2,0,2,3)` triangulation rule true without
special-casing each direction — a deliberate, correct choice for geometry,
just not one that also happens to line up with a single UV order).

Worked out by hand (viewer standing on the normal's side, looking at the
face, `up` = Bevy `+Y`):

- **East/South**: corners come out bottom-right -> top-right -> top-left ->
  bottom-left. The fixed UV order happens to map that correctly — it's a
  pure horizontal mirror of the naive expectation, but internally
  consistent (V is never wrong).
- **West/North**: corners come out bottom-left -> bottom-right -> top-right
  -> top-left. The *same* fixed UV order maps this as a diagonal
  **transpose** instead — right on two corners (the BL/TR diagonal), wrong
  on the other two (BR and TL get swapped, so V is wrong there too, not
  just U).

A mirror on one pair of opposite-ish faces and a transpose on the other
differ by exactly a 90° rotation — which is exactly "side textures look
rotated" on any side texture with directional detail (wood grain, an
asymmetric bevel, baked shading, ...). Confirmed by a regression test
before the fix would have failed it (`world::mesh::tests::every_side_faces_top_corners_sample_v0_and_bottom_corners_sample_v1`).

## Resolution

Fixed in `src/world/mesh.rs`: `Face::corner_uvs` gives each face the UV
order that actually matches its own corner list (East/South share one
order, West/North share a different one) instead of one order shared by
all four. `Up`/`Down` were left on the original order — nothing reported a
problem there, and unlike the four side faces (which at minimum have to
agree with *each other*), top/bottom orientation has no single "correct"
answer without also matching Minecraft's real north-aligned UV convention,
which is out of scope here.

Added two regression tests in `world::mesh::tests`:
- every side face's geometrically-highest corners sample the texture's
  `v0` and lowest sample `v1` (catches the West/North vertical swap
  directly);
- East/South share one UV corner order, West/North share a different one,
  and the two differ (catches a regression back to one shared order).

`cargo test` passes (104/104, all pre-existing tests unaffected).

## Manual verification still needed

Not run — see CLAUDE.md's "Manual/visual verification": this needs a human
at the window. Added to `todo.md`.
