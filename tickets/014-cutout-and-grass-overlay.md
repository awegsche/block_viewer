# 014 - Alpha-masked cutouts and the grass side overlay

## Status
Open

## Depends on
013 (biome tint). Small on its own; finishes the job 013 starts.

## Two defects, one mechanism

**Leaves are solid bricks.** `oak_leaves.png` has 172 of 256 pixels opaque —
the rest is meant to be see-through. The terrain material is
`AlphaMode::Opaque` (the `StandardMaterial::default()` in
`main.rs::setup`), so the alpha channel is discarded and leaves render as
solid cubes. 013 makes this worse, not better: green solid cubes read as
more obviously wrong than grey ones.

**Grass blocks have brown sides.** Measured: `grass_block_side.png` averages
(126, 107, 65) — it is plain dirt. The green fringe lives in
`grass_block_side_overlay.png` (a greyscale mask, 45 of 256 pixels opaque)
which vanilla tints and composites over the dirt. 013 deliberately leaves
this alone; without it, a grass field viewed from ground level is a field of
dirt blocks with green lids.

Both are fixed by letting the terrain material discard transparent
fragments.

## Approach

**Set `AlphaMode::Mask(0.5)` on the one existing terrain material, and emit
the grass overlay as an extra quad in the same mesh, offset outward along
its normal by a small epsilon.**

That is one line for the material and a contained change in the mesher. It
needs no second material, no second mesh per chunk, and therefore no change
to `SpawnedChunkEntities`, `unload`, or the re-mesh swap in
`poll_completed_chunk_remeshes` — all of which assume one entity per chunk
coordinate.

### Why not a separate cutout mesh + material?

That is the textbook answer and it is what water/glass will eventually
need (010's transparency item, which requires real alpha *blending* and
back-to-front sorting — genuinely a second material). It is the wrong
trade here: it doubles the per-chunk entity bookkeeping across four
systems, for geometry that is a rounding error in vertex count. Revisit it
when transparency lands, and expect the overlay quads to move into that
pass then.

### Epsilon offset

The overlay quad is coplanar with the dirt side face it sits on, so it
z-fights without an offset. Push it out along the face normal by ~`0.001`
blocks. Bevy uses a reversed-Z depth buffer, which concentrates precision
near the camera, so a fixed world-space epsilon is well-behaved out to the
far plane at these distances — but this is the number to revisit if
shimmering shows up at the render-distance edge (see the manual check).

Alternative if it does misbehave: `StandardMaterial::depth_bias` on a
second material, which is the properly-supported mechanism but reintroduces
the second-material cost above.

### Which faces get an overlay

Only where the base side quad was emitted (so it inherits face culling for
free), and only for blocks whose tint table says so. Extend 013's
`BlockTint` with an optional overlay rather than special-casing
`grass_block` by name in the mesher:

```rust
struct BlockTint {
    top: TintSource,
    bottom: TintSource,
    side: TintSource,
    /// Extra tinted quad over the side faces: atlas tile + its tint source.
    side_overlay: Option<(UvRect, TintSource)>,
}
```

`grass_block` is the only entry that uses it today. `podzol` and `mycelium`
have no overlay in vanilla (their side textures are pre-coloured), and
snowy variants (`grass_block_snow`) depend on a `snowy=true` block-state
property that the decoder discards — out of scope, note it.

## Alpha-mask side effects to check

- **The atlas's 1px padding ring.** `atlas::blit_padded` extrudes edge
  pixels, including their alpha. A tile whose edge pixels are transparent
  extrudes transparency, which is correct. No change needed, but confirm
  the leaves' silhouette doesn't gain a fringe.
- **Cost.** Alpha testing defeats early-Z for the whole terrain material,
  including the ~99% of it that is fully opaque. At this project's scale
  that is very unlikely to matter, but if 009's profiling later shows
  fragment-bound frames, splitting opaque from cutout is the first thing to
  try — which is the same split transparency will want. Leave a pointer in
  009.
- **Leaves still occlude.** `mesh::NON_SOLID` doesn't list leaves, so a
  leaf block still culls its neighbours' faces. With holes now visible you
  will see through a leaf into missing geometry behind it. Vanilla's
  "fancy" leaves render every leaf face; matching that means adding leaves
  to a new "transparent, doesn't cull same-type neighbours" category, which
  is 010's transparency rule and a meaningful vertex-count increase.
  **Out of scope here** — but say so in the ticket resolution, because it
  is the first thing someone will notice after this lands.

## Tests

- a lone grass block emits 6 base quads **+ 4 overlay quads** (index count
  goes from 36 to 60);
- the overlay quads' positions are offset outward from the base side quads
  by the epsilon, along the correct normal for each of the four sides;
- the overlay quads carry the grass tint colour while the base side quads
  stay white;
- a block with `side_overlay: None` emits exactly what it did before —
  every existing `mesh.rs` index-count assertion still passes;
- an overlay is not emitted for a side face that was culled by a neighbour.

## Done when

- Leaves have holes you can see through.
- Grass blocks are green on top *and* around their sides, biome-tinted, with
  brown dirt below the fringe.
- `cargo test` passes.
- `todo.md` gets a manual check: stand at ground level in a grassy biome and
  confirm the sides look right and don't shimmer; fly to the render-distance
  edge and watch the overlay quads on distant grass for z-fighting; look at
  a tree and confirm the leaves read as foliage, not a solid block.
