# 003 - Chunk mesher: block grids → Bevy meshes

## Status
Open

## Depends on
002.

## Goal

Replace the hardcoded 8×8×8 placeholder in `create_block_mesh()` with a
mesher that turns a decoded `ChunkColumn` into real Bevy meshes, and render
one region's worth of them.

## Scope

Move meshing into `src/mesh/` (or `src/world/mesh.rs`) and generalise the
existing code rather than rewriting it — `create_block_mesh` already has the
right bones: iterate blocks, emit a quad per face whose neighbour is empty,
with the six `get_*_tris` helpers and matching normals/UVs. What changes:

- Source blocks from a `ChunkColumn` instead of the local `blocks` vec.
- "Neighbour is air" replaces "neighbour == 0" — needs a notion of which
  `BlockId`s are non-solid (air, cave_air, void_air at minimum; water and
  leaves are ticket 010's problem, treat them as solid for now).
- Emit **one mesh per chunk column** (16×16×384), spawned as its own entity
  with a `Transform` at the chunk's world origin. Do not emit one mesh for a
  whole region — that's a single 100M-block draw call and kills culling.
- Chunk-boundary faces: looking up a neighbour outside the chunk requires the
  adjacent column. Start by treating out-of-chunk neighbours as **air**
  (visible seams between chunks, acceptable for a first landing), then close
  the seams by passing the 4 horizontal neighbour columns into the mesher.
  Do it in that order — the seam version is a working checkpoint.
- Vertical: nothing above/below a column, so top/bottom of the world are
  boundaries; don't emit the underside of the world.
- Skip fully-air sections entirely (002 makes these cheap to detect).

## Watch out

- `create_block_mesh`'s winding is inconsistent between faces today (compare
  the index order for left/front vs. right/back/top/bottom — two different
  patterns are in use). Verify winding against backface culling before
  scaling up, or half the world will be invisible from one side.
- The current code centres cubes on the block coordinate (`±0.5`). Minecraft
  block `(x,y,z)` occupies `x..x+1`; pick one convention and apply it
  consistently, or blocks will be offset by half a unit from where picking
  and the coordinate readout (007) say they are.
- Bevy is Y-up and Minecraft is Y-up, but Minecraft is **Z-south / X-east**
  and left-handed relative to Bevy's convention. Decide the axis mapping once
  and write it down in the module docs, otherwise the world renders mirrored
  and every later ticket inherits the confusion.

## Out of scope

- Per-block textures (004) — use the single `stone.png` currently loaded for
  every face.
- Greedy meshing / quad merging, LOD, transparency sorting (009).
- Loading more than the already-loaded first region (005).

## Done when

- The app renders real terrain from the local save instead of the
  placeholder, one entity per chunk column.
- Terrain is recognisably correct: a flat-ish surface at plausible heights,
  solid from the outside, no inside-out faces.
- `create_block_mesh` and its now-unused helpers are gone or folded into the
  new module — no dead placeholder left behind.
