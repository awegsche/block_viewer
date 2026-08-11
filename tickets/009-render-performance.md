# 009 - Performance pass

## Status
Open

## Depends on
003, 005. **Do not start before both have landed** — every item here needs a
real profile to justify it, and guessing at this stage will optimise code
that's about to be rewritten.

## Goal

A usable frame rate at a render distance worth exploring at.

## Scope — measure first, then pick from this list

Start by profiling: where does the time actually go — decode (002), mesh
build (003), mesh upload, or draw calls? Fix in that order.

- **Frustum culling**: one mesh entity per chunk column (003) gives Bevy this
  for free via `Aabb`; verify the AABBs are actually computed and that
  off-screen chunks are being culled.
- **Greedy meshing / quad merging**: merging coplanar same-texture faces cuts
  vertex counts dramatically on flat terrain. Interacts with the atlas (004)
  — merged quads need tiled UVs, which conflicts with a naive atlas. Decide
  the approach before committing to either.
- **Vertical splitting**: a full 384-block column is a tall AABB that's
  almost always partly on screen. Consider one mesh per 16-block section
  instead, trading draw calls for culling granularity — measure both.
- **Opaque/transparent split**: separate meshes and materials so alpha
  blending (water, glass, leaves) doesn't force sorting on everything.
- **Occlusion**: the vast majority of a save is solid stone that is never
  visible. A cheap "is this section fully enclosed by solid neighbours" test
  can skip whole sections at decode time.
- **Vertex format**: positions/normals/UVs as f32 arrays are generous for
  block-aligned geometry; packing into a smaller vertex layout is a real win
  but only worth it once vertex count is the proven bottleneck.

## Also

- `[profile.dev] opt-level = 1` with deps at 3 is already set in
  `Cargo.toml`; check whether the mesher/decoder want to be in the
  `opt-level = 3` bucket during development too, since they're the hot path.
- Record before/after numbers in this ticket's resolution. "Feels faster"
  isn't a result.

## Done when

- A stated target is met and written down (e.g. 60 fps at render distance 12
  on this machine), with before/after numbers.
- No single frame spike above a stated budget while flying.
