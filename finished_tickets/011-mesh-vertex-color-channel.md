# 011 - Mesher: a per-vertex colour channel

## Status
Open

## Depends on
003 (mesher), 004 (atlas). Nothing else.

## Why this exists as its own ticket

Three separate features want to multiply a colour into a block face:

- **biome tint** (013) — grass/foliage/water colour per biome,
- **baked light** (010's "Lighting" item) — `SkyLight`/`BlockLight` nibbles,
- **ambient occlusion** (010's "Ambient occlusion" item).

They all write the same channel and they *compose by multiplication*. If each
one adds the attribute itself, the second one to land has to reconcile a
vertex format and a combination rule that was never designed. This ticket
adds the channel once, with the semantics written down, so 013 and the 010
items are each a small change that only computes a factor.

It is also the only change here that touches the vertex layout, so getting
it out of the way first means no other ticket has to invalidate mesh memory
numbers twice.

## Scope

`src/world/mesh.rs` only.

1. `push_quad` takes a `color: [f32; 4]` and pushes it four times (once per
   corner) into a new `colors` vec.
2. `mesh_chunk_column` inserts it as `Mesh::ATTRIBUTE_COLOR` alongside
   position/normal/UV.
3. Every call site passes opaque white — `[1.0, 1.0, 1.0, 1.0]`. This ticket
   changes nothing visually.

### Emit the attribute unconditionally

Even for chunks with nothing to tint. Two reasons: Bevy specialises the
render pipeline on the mesh's vertex layout, so a mix of with-colour and
without-colour chunk meshes means two pipelines and two draw-call batches
for the same material; and 013/010 would otherwise have to add the attribute
to a mesh mid-flight depending on contents.

### Semantics to write into the module docs

> The vertex colour channel is a **multiplicative modulation** of the
> sampled atlas texel, in **linear** (not sRGB) space. White = unmodified.
> Contributors multiply into it independently:
>
> ```text
> vertex_color = biome_tint (013) * baked_light (010) * ao (010)
> ```
>
> Anything writing this channel converts from sRGB itself — colours read out
> of a PNG or written as a hex literal are sRGB and must go through
> `Color::srgb_u8(..).to_linear()` before they land here.

The sRGB note is the part that bites. Bevy's `StandardMaterial` multiplies
vertex colour into an already-linearised base colour; feeding it sRGB
values makes tinted surfaces visibly too bright and washed out. There is no
error, just a wrong-looking world, so state the rule where the next person
will read it.

## Material

None needed. `StandardMaterial` picks up `ATTRIBUTE_COLOR` automatically
(the `VERTEX_COLORS` shader def is driven off the mesh layout), so
`main.rs::setup`'s terrain material is untouched. Confirm this holds in
Bevy 0.15 rather than assuming it — if it doesn't, the fallback is a tiny
`ExtendedMaterial`, but check first.

## Vertex format

Use `Float32x4` (the default for `ATTRIBUTE_COLOR`) for now. That is
+16 bytes/vertex on top of the current 32 (pos 12 + normal 12 + uv 8) — a
50% increase in chunk mesh memory, which is real but not the current
bottleneck.

`VertexAttributeValues::Unorm8x4` is a drop-in 4-byte alternative that the
shader still reads as `vec4<f32>`. Don't do it here — do it in **009**,
alongside the other vertex-packing work, with a measurement. Add a line to
009's "Vertex format" bullet pointing at this channel so it isn't forgotten.

## Tests

Extend `src/world/mesh.rs`'s existing test module:

- the returned mesh has `ATTRIBUTE_COLOR`, and its length equals
  `ATTRIBUTE_POSITION`'s;
- every entry is `[1.0, 1.0, 1.0, 1.0]`;
- the existing index-count assertions still pass unchanged (they should —
  this adds no geometry).

## Done when

- `cargo check` and `cargo test` pass.
- Chunk meshes carry a colour attribute that is white everywhere.
- The composition rule and the sRGB→linear requirement are documented in
  `src/world/mesh.rs`'s module docs.
- 009's vertex-format bullet mentions the `Unorm8x4` packing opportunity.
- A note in `todo.md`: run the app and confirm the world looks **exactly**
  as it did before (this ticket is a no-op visually; if anything changed
  colour, the vertex colour is being applied in the wrong colour space).
