//! Packs the vanilla resource pack's `textures/block/*.png` into one atlas
//! [`Image`] at startup, and maps interned [`BlockId`]s to per-face atlas
//! UV rects. See ticket 004, pass 1: convention-based (`<name>`,
//! `<name>_top`/`_bottom`/`_side`) plus a small hardcoded override list —
//! reading `models/block/*.json` for the general answer is pass 2, a
//! separate ticket.

use std::collections::HashSet;
use std::path::Path;

use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use image::{Rgba, RgbaImage};

use super::block::{BlockId, BlockRegistry};

/// Width/height of one texture tile, in pixels — vanilla's block texture
/// convention. Animated textures (`water_still.png`, ...) stack extra
/// frames below the first one in the same file; [`load_tiles`] only ever
/// reads the top `TILE` rows, i.e. frame 0.
const TILE: u32 = 16;
/// 1px border, extruded from each tile's own edge pixels, placed around it
/// in the atlas. A sample that strays past the tile's UV rect (mip
/// generation, filtering, float rounding) then lands on more of the same
/// texture instead of bleeding in a neighbour's — ticket 004's "no visible
/// bleeding at the far edge of render distance". [`blit_padded`] assumes
/// exactly a 1px ring; widening this needs that function extended too.
const PAD: u32 = 1;
const CELL: u32 = TILE + PAD * 2;

/// Normalized (0..1) UV rectangle of one tile inside the packed atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvRect {
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
}

/// Per-face atlas rects for one block. Most blocks show the same texture on
/// every face (`top == bottom == side`); grass, logs, and the like differ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockFaces {
    pub top: UvRect,
    pub bottom: UvRect,
    pub side: UvRect,
}

/// The packed atlas: an [`Image`] ready to hand to [`Assets<Image>`], plus
/// every source tile's rect keyed by its file stem (e.g. `"oak_log_top"`),
/// and the loud checker rect for anything unmapped.
pub struct TextureAtlas {
    pub image: Image,
    tiles: std::collections::HashMap<String, UvRect>,
    pub fallback: UvRect,
}

impl TextureAtlas {
    /// The atlas rect for a source file's stem (e.g. `"stone"`,
    /// `"oak_log_top"`), if that texture existed in the packed directory.
    fn tile(&self, name: &str) -> Option<UvRect> {
        self.tiles.get(name).copied()
    }
}

/// Reads every `<TILE>`-pixel-wide PNG directly under `dir` (i.e. skips the
/// two non-square flowing-liquid textures, `water_flow.png`/`lava_flow.png`
/// — not used as a static block face) and returns its file stem alongside
/// the decoded pixels. Frames past the first `TILE` rows of an animated
/// texture are ignored.
fn load_tiles(dir: &Path) -> std::io::Result<Vec<(String, RgbaImage)>> {
    let mut tiles = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(decoded) = image::open(&path) else {
            println!("block_viewer: skipping unreadable texture {}", path.display());
            continue;
        };
        if decoded.width() != TILE || decoded.height() < TILE {
            continue;
        }
        tiles.push((stem.to_string(), decoded.to_rgba8()));
    }
    // Deterministic atlas layout regardless of directory iteration order —
    // makes a packed atlas reproducible run to run, which is worth having
    // even though nothing here depends on it byte-for-byte yet.
    tiles.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(tiles)
}

/// A 4px-checkered magenta/black tile — loud and unmistakable in the
/// rendered world, so unmapped blocks are obviously wrong rather than
/// silently stone-coloured (ticket 004's "done when").
fn fallback_tile() -> RgbaImage {
    const SQUARE: u32 = 4;
    let magenta = Rgba([255, 0, 255, 255]);
    let black = Rgba([0, 0, 0, 255]);
    RgbaImage::from_fn(TILE, TILE, |x, y| {
        if (x / SQUARE + y / SQUARE) % 2 == 0 {
            magenta
        } else {
            black
        }
    })
}

/// Copies `src`'s top-left `TILE x TILE` pixels into the atlas at cell
/// `(col, row)`, extruding a 1px border from the tile's own edges (see
/// [`PAD`]) so the atlas can be sampled right up to that border safely.
fn blit_padded(atlas: &mut RgbaImage, col: u32, row: u32, src: &RgbaImage) {
    let ox = col * CELL;
    let oy = row * CELL;
    for y in 0..TILE {
        for x in 0..TILE {
            atlas.put_pixel(ox + PAD + x, oy + PAD + y, *src.get_pixel(x, y));
        }
    }
    for x in 0..TILE {
        atlas.put_pixel(ox + PAD + x, oy, *src.get_pixel(x, 0));
        atlas.put_pixel(ox + PAD + x, oy + PAD + TILE, *src.get_pixel(x, TILE - 1));
    }
    for y in 0..TILE {
        atlas.put_pixel(ox, oy + PAD + y, *src.get_pixel(0, y));
        atlas.put_pixel(ox + PAD + TILE, oy + PAD + y, *src.get_pixel(TILE - 1, y));
    }
    atlas.put_pixel(ox, oy, *src.get_pixel(0, 0));
    atlas.put_pixel(ox + PAD + TILE, oy, *src.get_pixel(TILE - 1, 0));
    atlas.put_pixel(ox, oy + PAD + TILE, *src.get_pixel(0, TILE - 1));
    atlas.put_pixel(
        ox + PAD + TILE,
        oy + PAD + TILE,
        *src.get_pixel(TILE - 1, TILE - 1),
    );
}

/// Builds the packed atlas from every block texture under `dir` (pass
/// `assets/minecraft/textures/block` — see ticket 004). Nearest filtering
/// is set on the returned [`Image`] to match `main.rs`'s existing sampler
/// choice.
pub fn build(dir: &Path) -> std::io::Result<TextureAtlas> {
    let mut named_tiles = load_tiles(dir)?;
    // The fallback checker gets packed into the atlas like any other tile,
    // so sampling it costs nothing extra at draw time.
    named_tiles.push((String::new(), fallback_tile()));

    let count = named_tiles.len() as u32;
    let grid = (count as f32).sqrt().ceil() as u32;
    let atlas_px = grid * CELL;

    let mut buf = RgbaImage::new(atlas_px, atlas_px);
    let mut tiles = std::collections::HashMap::with_capacity(named_tiles.len());
    let mut fallback = UvRect {
        u0: 0.0,
        v0: 0.0,
        u1: 1.0,
        v1: 1.0,
    };

    for (i, (name, pixels)) in named_tiles.iter().enumerate() {
        let i = i as u32;
        let col = i % grid;
        let row = i / grid;
        blit_padded(&mut buf, col, row, pixels);

        let u0 = (col * CELL + PAD) as f32 / atlas_px as f32;
        let v0 = (row * CELL + PAD) as f32 / atlas_px as f32;
        let rect = UvRect {
            u0,
            v0,
            u1: u0 + TILE as f32 / atlas_px as f32,
            v1: v0 + TILE as f32 / atlas_px as f32,
        };

        if name.is_empty() {
            fallback = rect;
        } else {
            tiles.insert(name.clone(), rect);
        }
    }

    println!(
        "block_viewer: packed {} block textures into a {atlas_px}x{atlas_px} atlas",
        tiles.len()
    );

    let mut image = Image::new(
        Extent3d {
            width: atlas_px,
            height: atlas_px,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        buf.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::nearest();

    Ok(TextureAtlas {
        image,
        tiles,
        fallback,
    })
}

/// Per-face texture-name overrides for the handful of blocks the naming
/// convention (`<name>`, `<name>_top`/`_bottom`/`_side`) can't resolve on
/// its own — e.g. grass's underside is dirt, not a `grass_block_bottom.png`
/// that doesn't exist in the resource pack. `(block name without
/// "minecraft:", top, bottom, side)`.
const OVERRIDES: &[(&str, &str, &str, &str)] = &[
    ("grass_block", "grass_block_top", "dirt", "grass_block_side"),
    ("podzol", "podzol_top", "dirt", "podzol_side"),
    ("mycelium", "mycelium_top", "dirt", "mycelium_side"),
    ("water", "water_still", "water_still", "water_still"),
    ("lava", "lava_still", "lava_still", "lava_still"),
];

fn resolve_faces(name: &str, atlas: &TextureAtlas, warned: &mut HashSet<String>) -> BlockFaces {
    if let Some(&(_, top, bottom, side)) = OVERRIDES.iter().find(|&&(n, ..)| n == name) {
        return BlockFaces {
            top: atlas.tile(top).unwrap_or(atlas.fallback),
            bottom: atlas.tile(bottom).unwrap_or(atlas.fallback),
            side: atlas.tile(side).unwrap_or(atlas.fallback),
        };
    }

    let base = atlas.tile(name);
    let top = atlas.tile(&format!("{name}_top")).or(base);
    let bottom = atlas.tile(&format!("{name}_bottom")).or(base);
    let side = atlas.tile(&format!("{name}_side")).or(base);

    if (top.is_none() || bottom.is_none() || side.is_none()) && warned.insert(name.to_string()) {
        println!("block_viewer: no texture mapping for 'minecraft:{name}' — using fallback checker");
    }

    BlockFaces {
        top: top.unwrap_or(atlas.fallback),
        bottom: bottom.unwrap_or(atlas.fallback),
        side: side.unwrap_or(atlas.fallback),
    }
}

/// Resolves every name interned in `registry` to a [`BlockFaces`], indexed
/// directly by [`BlockId`] (i.e. `table[id.0 as usize]` — valid because
/// [`BlockRegistry`] hands out ids `0..len()`). Unmapped names are logged
/// once each, not once per block instance, so the gaps stay enumerable.
pub fn build_block_uv_table(registry: &BlockRegistry, atlas: &TextureAtlas) -> Vec<BlockFaces> {
    let mut warned = HashSet::new();
    (0..registry.len())
        .map(|i| {
            let id = BlockId(i as u16);
            let full_name = registry.name(id);
            let name = full_name.strip_prefix("minecraft:").unwrap_or(full_name);
            resolve_faces(name, atlas, &mut warned)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distinct, easy-to-assert-on rect — `n` shows up in every field so
    /// two calls with different `n` are never accidentally equal.
    fn rect(n: f32) -> UvRect {
        UvRect {
            u0: n,
            v0: n,
            u1: n + 1.0,
            v1: n + 1.0,
        }
    }

    /// A [`TextureAtlas`] with the given named tiles, skipping the real
    /// file-packing [`build`] does — these tests are about
    /// [`resolve_faces`]'s naming rules, not pixel packing.
    fn atlas_with(named: &[(&str, UvRect)]) -> TextureAtlas {
        TextureAtlas {
            image: Image::default(),
            tiles: named.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
            fallback: rect(999.0),
        }
    }

    #[test]
    fn default_texture_used_for_every_face_when_no_suffixed_variant_exists() {
        let atlas = atlas_with(&[("stone", rect(1.0))]);
        let faces = resolve_faces("stone", &atlas, &mut HashSet::new());
        assert_eq!(faces.top, rect(1.0));
        assert_eq!(faces.bottom, rect(1.0));
        assert_eq!(faces.side, rect(1.0));
    }

    #[test]
    fn suffixed_variants_win_over_the_default_per_face() {
        // oak_log: _top exists, side falls back to the base texture (no
        // oak_log_side.png in the real resource pack either).
        let atlas = atlas_with(&[("oak_log", rect(1.0)), ("oak_log_top", rect(2.0))]);
        let faces = resolve_faces("oak_log", &atlas, &mut HashSet::new());
        assert_eq!(faces.top, rect(2.0));
        assert_eq!(faces.bottom, rect(1.0));
        assert_eq!(faces.side, rect(1.0));
    }

    #[test]
    fn unmapped_name_falls_back_to_the_checker_and_warns_once() {
        let atlas = atlas_with(&[]);
        let mut warned = HashSet::new();
        let faces = resolve_faces("some_unknown_block", &atlas, &mut warned);
        assert_eq!(faces.top, atlas.fallback);
        assert_eq!(faces.bottom, atlas.fallback);
        assert_eq!(faces.side, atlas.fallback);

        resolve_faces("some_unknown_block", &atlas, &mut warned);
        assert_eq!(
            warned.len(),
            1,
            "the same unmapped name should only be recorded once"
        );
    }

    #[test]
    fn override_table_wins_for_grass_blocks_dirt_underside() {
        let atlas = atlas_with(&[
            ("grass_block_top", rect(1.0)),
            ("grass_block_side", rect(2.0)),
            ("dirt", rect(3.0)),
        ]);
        let faces = resolve_faces("grass_block", &atlas, &mut HashSet::new());
        assert_eq!(faces.top, rect(1.0));
        assert_eq!(faces.side, rect(2.0));
        assert_eq!(faces.bottom, rect(3.0), "grass's underside is dirt, not a (nonexistent) grass_block_bottom.png");
    }

    /// Packs the real vanilla resource pack checked into `assets/` and
    /// spot-checks the textures ticket 004's "done when" calls out by name
    /// (grass, dirt, logs, water) actually made it into the atlas.
    #[test]
    fn builds_atlas_from_the_real_asset_pack() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/minecraft/textures/block");
        let atlas = build(&dir).expect("assets/minecraft/textures/block should be readable");
        for name in [
            "stone",
            "dirt",
            "oak_log",
            "oak_log_top",
            "grass_block_top",
            "grass_block_side",
            "water_still",
        ] {
            assert!(atlas.tile(name).is_some(), "expected a packed tile for {name}");
        }
    }

    #[test]
    fn build_block_uv_table_is_indexed_by_block_id() {
        let mut registry = BlockRegistry::new(); // interns "minecraft:air" as id 0
        let stone = registry.intern("minecraft:stone");
        let dirt = registry.intern("minecraft:dirt");

        let atlas = atlas_with(&[("stone", rect(1.0)), ("dirt", rect(2.0))]);
        let table = build_block_uv_table(&registry, &atlas);

        assert_eq!(table.len(), registry.len());
        assert_eq!(table[stone.0 as usize].side, rect(1.0));
        assert_eq!(table[dirt.0 as usize].side, rect(2.0));
        // "minecraft:air" has no texture at all — falls back like any
        // other unmapped name (never actually sampled, air isn't meshed).
        assert_eq!(table[BlockRegistry::AIR.0 as usize].side, atlas.fallback);
    }
}
