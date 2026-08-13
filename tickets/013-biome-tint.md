# 013 - Biome tint: colormaps into the vertex colour channel

## Status
Open

## Depends on
011 (vertex colour channel), 012 (biome grids). Both are hard blockers.

## Why the world looks grey right now

Measured from the resource pack checked into `assets/` (average RGB of the
opaque pixels of frame 0):

| texture | avg RGB | meaning |
|---|---|---|
| `grass_block_top.png` | (147, 147, 147) | greyscale mask — **must** be tinted |
| `oak_leaves.png` | (144, 144, 144) | greyscale mask, 172/256 px opaque |
| `water_still.png` | (177, 177, 177) | greyscale mask |
| `grass_block_side_overlay.png` | (154, 154, 154) | greyscale mask, 45/256 px opaque |
| `grass_block_side.png` | (126, 107, 65) | plain dirt, **not** tinted |

Every "green" texture in modern vanilla ships neutral and is coloured at
render time. That is the whole of this ticket. The last row is 014's
problem: the green fringe on a grass block's *side* is a separate overlay
texture composited on top of dirt, so this ticket leaves grass sides brown.

## Design

Two lookups, resolved independently and multiplied at the point a face is
emitted:

```text
face colour = tint_source(block_id, face)  applied to  biome_colors[biome_id]
```

Both tables are built per background task, indexed by id, exactly the way
`atlas::build_block_uv_table` already works — same pattern, same reason
(ids are interned lazily as chunks stream in, so there is no fixed table to
build once at startup).

### `TintSource` — what a face is tinted *by*

```rust
enum TintSource {
    None,
    Grass,          // sample colormap/grass.png at the biome's (temp, downfall)
    Foliage,        // sample colormap/foliage.png likewise
    Water,          // the biome's flat water colour (not a colormap)
    Fixed(LinearRgba),
}
```

Per-face, mirroring `atlas::BlockFaces`:

```rust
struct BlockTint { top: TintSource, bottom: TintSource, side: TintSource }
```

Per-face matters for exactly one important block: `grass_block` tints its
**top only**. Bottom is dirt, side is dirt (+ 014's overlay). A whole-block
tint would turn every grass block into a green cube.

`build_block_tint_table(&BlockRegistry) -> Vec<BlockTint>`, warn-once on
names that look like they should be tinted but aren't in the table (a
`_leaves` suffix not in the list is the useful heuristic).

### `BiomeColors` — the colours a biome resolves to

```rust
struct BiomeColors { grass: LinearRgba, foliage: LinearRgba, water: LinearRgba }
```

`build_biome_tint_table(&BiomeRegistry, &ColorMaps) -> Vec<BiomeColors>`.

## The colormaps

`assets/minecraft/textures/colormap/grass.png` and `foliage.png`, both
256x256, already in the repo. Load them in `main.rs::setup` next to the
atlas build, into a plain-data `ColorMaps { grass: Vec<[u8;3]>, foliage: … }`
(65536 entries each, ~384 KB total) published as
`SharedColorMaps(Arc<ColorMaps>)`, mirroring `SharedAtlasIndex` — see below
on the `Send` boundary.

Indexing, matching vanilla's `GrassColor.get(temperature, downfall)`:

```rust
let temp = temperature.clamp(0.0, 1.0);
let rain = downfall.clamp(0.0, 1.0) * temp;   // note: scaled by temperature
let x = ((1.0 - temp) * 255.0) as usize;
let y = ((1.0 - rain) * 255.0) as usize;
let texel = pixels[y * 256 + x];
```

Vanilla returns magenta for out-of-range indices; clamp instead — a
hardcoded biome table (below) can't produce an out-of-range value, and
magenta terrain from a typo'd table entry is worse than a clamped colour.

## The biome parameter table — the part with no source in the repo

A biome's temperature and downfall are **not** in the save file and **not**
in the resource pack. They live in the Minecraft server's code / the
vanilla data pack's `worldgen/biome/*.json`, neither of which is vendored
here. So `src/world/biome_data.rs` gets a hardcoded table:

```rust
// (name without "minecraft:", temperature, downfall, water_color)
const BIOMES: &[(&str, f32, f32, u32)] = &[
    ("plains",         0.8,  0.4, 0x3F76E4),
    ("forest",         0.7,  0.8, 0x3F76E4),
    ("taiga",          0.25, 0.8, 0x3F76E4),
    ("desert",         2.0,  0.0, 0x3F76E4),
    ("jungle",         0.95, 0.9, 0x3F76E4),
    ("savanna",        2.0,  0.0, 0x3F76E4),
    ("snowy_plains",   0.0,  0.5, 0x3F76E4),
    ("swamp",          0.8,  0.9, 0x617B64),
    ("warm_ocean",     0.5,  0.5, 0x43D5EE),
    ("lukewarm_ocean", 0.5,  0.5, 0x45ADF2),
    ("cold_ocean",     0.5,  0.5, 0x3D57D6),
    ("frozen_ocean",   0.0,  0.5, 0x3938C9),
    // …~60 overworld biomes, plus nether/end
];
```

Fill it out against the vanilla `worldgen/biome` data for the version the
`assets/` pack came from. Unknown name → plains values, warn once. Getting
an entry slightly wrong shifts one biome's green a little; leaving a biome
out entirely makes it look like plains, which is a visible but survivable
failure — so ship the table incomplete rather than blocking on all of it,
and note which families are missing.

### Hardcoded exceptions (vanilla does not use the colormap for these)

- `swamp`, `mangrove_swamp`: grass and foliage both fixed `0x6A7039`.
- `badlands`, `eroded_badlands`, `wooded_badlands`: grass `0x90814D`,
  foliage `0x9E814D`.
- `dark_forest`: `(colormap_value & 0xFEFEFE) + 0x28340A) >> 1` — a blend
  toward dark green, applied *after* the colormap lookup.
- `cherry_grove`, `meadow` and a few others have their own overrides in
  recent versions. Check against the vendored assets' version; where you're
  unsure, use the colormap value and leave a `// approximate` comment
  rather than inventing a constant.

These make swamps and badlands read correctly, and they're the two biomes
where a plain colormap lookup looks obviously wrong.

## Which blocks get tinted

| blocks | source |
|---|---|
| `grass_block` (**top face only**) | Grass |
| `short_grass`, `tall_grass`, `fern`, `large_fern`, `potted_fern`, `sugar_cane` | Grass |
| `oak_leaves`, `jungle_leaves`, `acacia_leaves`, `dark_oak_leaves`, `mangrove_leaves`, `vine` | Foliage |
| `birch_leaves` | Fixed `0x80A755` |
| `spruce_leaves` | Fixed `0x619961` |
| `lily_pad` | Fixed `0x208030` |
| `cherry_leaves`, `azalea_leaves`, `flowering_azalea_leaves`, `pale_oak_leaves` | None — these ship pre-coloured |
| `water`, `water_cauldron`, `bubble_column` | Water |

## Mesher changes

`mesh_chunk_column` gains two parameters: `block_tint: &[BlockTint]` and
`biome_colors: &[BiomeColors]`. For each emitted face:

```rust
let source = block_tint[id.0 as usize].for_face(face);
let color = match source {
    TintSource::None     => LinearRgba::WHITE,
    TintSource::Grass    => biome_colors[section.biome_at(lx, ly, lz).0 as usize].grass,
    TintSource::Foliage  => …,
    TintSource::Water    => …,
    TintSource::Fixed(c) => c,
};
```

and pass it to `push_quad` in place of 011's white.

`biome_at` takes the block's **section-local** y (`ly`), not `world_y` —
easy to get wrong given the loop already computes both.

## Colour space — read this before writing any conversion

Colormap texels and the hex literals above are **sRGB**. The vertex colour
channel is **linear** (see 011). Convert once, at table-build time:

```rust
Color::srgb_u8(r, g, b).to_linear()
```

Storing sRGB values in `BiomeColors` and converting at emit time would work
too but costs a conversion per face. Convert in `build_biome_tint_table`
and keep `LinearRgba` in the struct so the type itself says which space
it's in.

Symptom of getting this wrong: grass that is green but noticeably too
bright and milky, with no error anywhere.

## Send boundary

Everything above must be reachable from inside a `chunk_pipeline`
background task. `ColorMaps` is plain data (`Vec<[u8;3]>`), so
`SharedColorMaps(Arc<ColorMaps>)` crosses fine, the same way
`SharedAtlasIndex` does. Do **not** reach for `Image` or any Bevy render
type in the tint path — that's precisely what `TextureAtlas::uv_index`
exists to avoid, and the same split applies here.

`mesh_column_with_neighbors` in `chunk_pipeline.rs` is the one place that
builds the per-task tables; both the load and re-mesh paths already funnel
through it, so both pick this up for free.

## Out of scope (name them in the ticket's follow-ups)

- **Biome blending.** Vanilla averages the colormap over a 5x5 block
  neighbourhood, which is why biome borders fade instead of stepping. v1
  has hard edges. The mesher already has `Neighbors` for ±1 chunk, so a
  radius-2 blend is reachable later without new plumbing — but it turns one
  table lookup per face into 25, so it wants measuring.
- **Grass block sides.** 014.
- **Leaves' alpha holes.** Also 014 — tinting them makes the holes *more*
  obvious, since they'll be green-on-opaque instead of grey-on-opaque.

## Tests

- the colormap index formula: known (temp, downfall) pairs → known (x, y);
  in particular that downfall is multiplied by temperature first;
- `plains` and `jungle` resolve to visibly different grass colours;
- the swamp and badlands overrides bypass the colormap;
- `grass_block`'s top is `Grass` and its bottom/side are `None`;
- an unknown biome name falls back to plains and warns once;
- sRGB→linear actually happened (a mid-grey texel does not come back as the
  same numeric value);
- a meshed grass block's top-face vertices carry the biome colour and its
  bottom-face vertices carry white.

## Done when

- Grass, leaves and water are coloured, and the colour differs between
  biomes (jungle vs taiga vs savanna vs swamp is the useful spot-check).
- `cargo test` passes.
- `todo.md` gets a manual check: fly across a biome border in the real save
  and confirm the tint changes; look at a grass block from above (green)
  and from the side (brown dirt — expected until 014); confirm nothing
  looks washed out (colour-space regression).
