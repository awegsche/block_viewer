//! Biome tint (ticket 013): colours grass, leaves and water at mesh time —
//! see the module docs on [`super::mesh`] for how the resulting colour lands
//! in the vertex colour channel (011). Two independent tables, both built
//! per background chunk task the same way [`super::atlas::build_block_uv_table`]
//! is (ids are interned lazily as chunks stream in, so there's no fixed
//! table to build once at startup):
//!
//! - [`build_block_tint_table`] — what a block's faces are tinted *by*
//!   ([`TintSource`]/[`BlockTint`], indexed by [`super::BlockId`]).
//! - [`build_biome_tint_table`] — what each biome resolves a tint source to
//!   ([`BiomeColors`], indexed by [`super::BiomeId`]).
//!
//! Because both are per-task, the "this has no tint mapping" / "unknown
//! biome" warnings dedupe against the process-wide ledgers in
//! [`super::warn`] rather than a set local to the build (ticket 081) —
//! otherwise every streamed chunk reprints them.
//!
//! `mesh_chunk_column` multiplies the two together per emitted face.
//!
//! ## Colour space
//!
//! Colormap texels and this module's hex literals are **sRGB**; everything
//! this module hands back is **linear** (`LinearRgba`), converted once at
//! table-build time via `Color::srgb_u8(..).to_linear()` rather than per
//! face. Storing sRGB and converting later would also work but costs a
//! conversion per face, and the type wouldn't say which space it's in.

use std::path::Path;

use bevy::prelude::*;

use super::atlas::{AtlasUvIndex, UvRect};
use super::biome::{BiomeId, BiomeRegistry};
use super::biome_data;
use super::block::{BlockId, BlockRegistry};
use super::warn::WarnLedger;

/// What a face is tinted *by* — resolved against a [`BiomeColors`] (for the
/// biome-dependent variants) at the point a face is emitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TintSource {
    /// No tint — the texture is used as-is (opaque white in the vertex
    /// colour channel).
    None,
    /// Sample `colormap/grass.png` at the biome's (temperature, downfall).
    Grass,
    /// Sample `colormap/foliage.png` likewise.
    Foliage,
    /// The biome's flat water colour (not a colormap).
    Water,
    /// A colour fixed regardless of biome (e.g. birch/spruce leaves, which
    /// vanilla hardcodes rather than reading off either colormap).
    Fixed(LinearRgba),
}

/// Per-face tint source for one block, mirroring [`super::atlas::BlockFaces`].
/// Per-face matters for exactly one important block: `grass_block` tints its
/// **top only** (bottom is dirt, side is dirt + 014's overlay) — a
/// whole-block tint would turn every grass block into a green cube.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockTint {
    pub top: TintSource,
    pub bottom: TintSource,
    pub side: TintSource,
    /// Extra tinted quad emitted over each *emitted* side face (014) — the
    /// atlas tile to draw it with, and what tints it. `grass_block` is the
    /// only entry using this today (`grass_block_side_overlay.png`, the
    /// green fringe over its dirt sides); `podzol`/`mycelium` have no
    /// overlay in vanilla (their side textures are pre-coloured).
    pub side_overlay: Option<(UvRect, TintSource)>,
}

impl BlockTint {
    /// No tint on any face — what every block not named in this module's
    /// tables resolves to.
    pub const NONE: BlockTint = BlockTint {
        top: TintSource::None,
        bottom: TintSource::None,
        side: TintSource::None,
        side_overlay: None,
    };

    /// The same source on every face, no side overlay — every tinted block
    /// except `grass_block` (see the struct docs).
    fn uniform(source: TintSource) -> BlockTint {
        BlockTint { top: source, bottom: source, side: source, side_overlay: None }
    }
}

/// The colours one biome resolves [`TintSource::Grass`]/`Foliage`/`Water` to,
/// already converted to linear space — see the module docs on colour space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiomeColors {
    pub grass: LinearRgba,
    pub foliage: LinearRgba,
    pub water: LinearRgba,
}

/// One 256x256 Minecraft colormap, decoded to plain `[u8; 3]` texels indexed
/// `y * 256 + x` — no Bevy render types, so [`ColorMaps`] crosses the `Send`
/// boundary into a background chunk task the same way
/// [`super::AtlasUvIndex`] does (see `chunk_pipeline`'s module docs).
#[derive(Debug, Clone)]
pub struct ColorMaps {
    pub grass: Vec<[u8; 3]>,
    pub foliage: Vec<[u8; 3]>,
}

/// Side length of a vanilla colormap PNG, in pixels — also the modulus
/// [`colormap_xy`]'s indices are computed against.
const COLORMAP_SIZE: u32 = 256;

/// Loads `grass.png` and `foliage.png` from `dir` (pass
/// `assets/minecraft/textures/colormap` — both files are already checked
/// into the repo). Errors if either is missing, unreadable, or not
/// `256x256` — a colormap of the wrong shape would silently mis-sample every
/// biome's tint, which is worse than failing loudly at startup.
pub fn load_color_maps(dir: &Path) -> std::io::Result<ColorMaps> {
    Ok(ColorMaps {
        grass: load_colormap_texels(&dir.join("grass.png"))?,
        foliage: load_colormap_texels(&dir.join("foliage.png"))?,
    })
}

fn load_colormap_texels(path: &Path) -> std::io::Result<Vec<[u8; 3]>> {
    let image = image::open(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?
        .to_rgb8();
    if image.width() != COLORMAP_SIZE || image.height() != COLORMAP_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "expected a {COLORMAP_SIZE}x{COLORMAP_SIZE} colormap at {}, got {}x{}",
                path.display(),
                image.width(),
                image.height()
            ),
        ));
    }
    Ok(image.pixels().map(|p| [p[0], p[1], p[2]]).collect())
}

/// The colormap cell (x, y) for a given (temperature, downfall), matching
/// vanilla's `GrassColor.get(temperature, downfall)`. Note `downfall` is
/// multiplied by the (already-clamped) temperature *before* either is turned
/// into a pixel coordinate — dropping that scaling is the easy way to get
/// this formula subtly wrong.
fn colormap_xy(temperature: f32, downfall: f32) -> (usize, usize) {
    let temp = temperature.clamp(0.0, 1.0);
    let rain = downfall.clamp(0.0, 1.0) * temp;
    let x = ((1.0 - temp) * (COLORMAP_SIZE - 1) as f32) as usize;
    let y = ((1.0 - rain) * (COLORMAP_SIZE - 1) as f32) as usize;
    (x, y)
}

/// `Color::srgb_u8(..).to_linear()` for a `0xRRGGBB` hex literal — every
/// fixed colour in this module (biome overrides, birch/spruce leaves, ...)
/// goes through this rather than hand-rolling the bit-shifts at each site.
fn linear_from_hex(hex: u32) -> LinearRgba {
    let r = ((hex >> 16) & 0xFF) as u8;
    let g = ((hex >> 8) & 0xFF) as u8;
    let b = (hex & 0xFF) as u8;
    Color::srgb_u8(r, g, b).to_linear()
}

fn linear_from_texel([r, g, b]: [u8; 3]) -> LinearRgba {
    Color::srgb_u8(r, g, b).to_linear()
}

/// Vanilla's dark-green blend for `dark_forest`, applied *after* the
/// colormap lookup: `(color & 0xFEFEFE) + 0x28340A) >> 1`, done here on the
/// sRGB `u8` texel (matching how vanilla computes it) before the result goes
/// through [`linear_from_texel`].
fn dark_forest_blend([r, g, b]: [u8; 3]) -> [u8; 3] {
    let packed = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
    let blended = ((packed & 0xFE_FE_FE) + 0x28_34_0A) >> 1;
    [
        ((blended >> 16) & 0xFF) as u8,
        ((blended >> 8) & 0xFF) as u8,
        (blended & 0xFF) as u8,
    ]
}

/// Block names tinted by [`TintSource::Grass`] on every face — `grass_block`
/// is handled separately (top only, see [`resolve_block_tint`]).
const GRASS_TINTED: &[&str] = &[
    "short_grass",
    "tall_grass",
    "fern",
    "large_fern",
    "potted_fern",
    "sugar_cane",
];

/// Block names tinted by [`TintSource::Foliage`] on every face.
const FOLIAGE_TINTED: &[&str] = &[
    "oak_leaves",
    "jungle_leaves",
    "acacia_leaves",
    "dark_oak_leaves",
    "mangrove_leaves",
    "vine",
];

/// Block names tinted a fixed colour regardless of biome, `(name, 0xRRGGBB)`
/// — vanilla hardcodes these rather than reading off either colormap.
const FIXED_TINTED: &[(&str, u32)] = &[
    ("birch_leaves", 0x80A755),
    ("spruce_leaves", 0x619961),
    ("lily_pad", 0x208030),
];

/// Leaves that ship pre-coloured in the resource pack — explicitly *not*
/// tinted, and exempted from [`resolve_block_tint`]'s "`_leaves` should
/// probably be tinted" warning heuristic.
const NO_TINT_LEAVES: &[&str] = &[
    "cherry_leaves",
    "azalea_leaves",
    "flowering_azalea_leaves",
    "pale_oak_leaves",
];

/// Block names tinted by [`TintSource::Water`] on every face.
const WATER_TINTED: &[&str] = &["water", "water_cauldron", "bubble_column"];

/// Source file stem of grass's green side fringe (014) — a separate texture
/// vanilla composites over `grass_block_side`, not part of the top/bottom/
/// side split `super::atlas::resolve_faces` resolves.
const GRASS_SIDE_OVERLAY: &str = "grass_block_side_overlay";

/// Names already reported as missing a tint mapping (ticket 081) — the
/// grass side overlay, and `_leaves` blocks the tables below don't cover.
///
/// Process-wide rather than per [`build_block_tint_table`] call, because
/// that table is rebuilt in every background chunk task — see
/// [`super::warn`].
pub(crate) static MISSING_TINT: WarnLedger = WarnLedger::new();

/// `pub(crate)`: [`super::super::blueprint::mesh`] (ticket 037, roadmap B2)
/// resolves a blueprint palette entry's tint the same way, straight off its
/// `BlockState::name` — a blueprint has no [`BlockRegistry`] to route
/// through [`build_block_tint_table`].
pub(crate) fn resolve_block_tint(name: &str, atlas: &AtlasUvIndex, warned: &WarnLedger) -> BlockTint {
    if name == "grass_block" {
        // Bottom is dirt, side is dirt + 014's green fringe overlay — only
        // the top face is grass texture at all.
        let side_overlay = atlas.tile(GRASS_SIDE_OVERLAY).map(|uv| (uv, TintSource::Grass));
        if side_overlay.is_none() && warned.first_time(GRASS_SIDE_OVERLAY) {
            println!(
                "block_viewer: no '{GRASS_SIDE_OVERLAY}' texture found — grass blocks will have plain dirt sides"
            );
        }
        return BlockTint {
            top: TintSource::Grass,
            bottom: TintSource::None,
            side: TintSource::None,
            side_overlay,
        };
    }
    if GRASS_TINTED.contains(&name) {
        return BlockTint::uniform(TintSource::Grass);
    }
    if FOLIAGE_TINTED.contains(&name) {
        return BlockTint::uniform(TintSource::Foliage);
    }
    if let Some(&(_, hex)) = FIXED_TINTED.iter().find(|&&(n, _)| n == name) {
        return BlockTint::uniform(TintSource::Fixed(linear_from_hex(hex)));
    }
    if WATER_TINTED.contains(&name) {
        return BlockTint::uniform(TintSource::Water);
    }

    // Heuristic for catching an omission in the tables above: a `_leaves`
    // block not accounted for anywhere (tinted, fixed, or explicitly
    // pre-coloured) is suspicious enough to warn about once.
    if !NO_TINT_LEAVES.contains(&name) && name.ends_with("_leaves") && warned.first_time(name) {
        println!(
            "block_viewer: 'minecraft:{name}' looks like it should be tinted (ends with _leaves) but has no tint mapping"
        );
    }

    BlockTint::NONE
}

/// Resolves every name interned in `registry` to a [`BlockTint`], indexed
/// directly by [`BlockId`] (i.e. `table[id.0 as usize]`), mirroring
/// [`super::atlas::build_block_uv_table`]. Takes `atlas` (rather than just
/// the registry, like 013 originally had it) because grass's side-overlay
/// entry (014) needs to resolve `grass_block_side_overlay`'s atlas rect —
/// a texture lookup outside any [`super::atlas::BlockFaces`]'s top/bottom/
/// side split, so it can't come from `uv_table` the way other UVs do.
pub fn build_block_tint_table(registry: &BlockRegistry, atlas: &AtlasUvIndex) -> Vec<BlockTint> {
    (0..registry.len())
        .map(|i| {
            let id = BlockId(i as u16);
            let full_name = registry.name(id);
            let name = full_name.strip_prefix("minecraft:").unwrap_or(full_name);
            resolve_block_tint(name, atlas, &MISSING_TINT)
        })
        .collect()
}

/// Biome names already reported as absent from [`biome_data`]'s table
/// (ticket 081). Process-wide, for the same reason as [`MISSING_TINT`].
static UNKNOWN_BIOME: WarnLedger = WarnLedger::new();

/// Resolves one biome's [`BiomeColors`], applying the hardcoded exceptions
/// (swamp/mangrove_swamp, badlands family, dark_forest) documented in ticket
/// 013 before falling back to a plain colormap lookup.
fn biome_colors_for(name: &str, maps: &ColorMaps, warned: &WarnLedger) -> BiomeColors {
    let params = biome_data::params_for(name).unwrap_or_else(|| {
        if warned.first_time(name) {
            println!(
                "block_viewer: unknown biome 'minecraft:{name}' — falling back to plains tint values"
            );
        }
        biome_data::params_for("plains").expect("plains must be in biome_data's table")
    });
    let water = linear_from_hex(params.water_color);

    // Vanilla does not use the colormap for these — fixed colours regardless
    // of temperature/downfall.
    if name == "swamp" || name == "mangrove_swamp" {
        let fixed = linear_from_hex(0x6A_70_39);
        return BiomeColors { grass: fixed, foliage: fixed, water };
    }
    if matches!(name, "badlands" | "eroded_badlands" | "wooded_badlands") {
        return BiomeColors {
            grass: linear_from_hex(0x90_81_4D),
            foliage: linear_from_hex(0x9E_81_4D),
            water,
        };
    }

    let (x, y) = colormap_xy(params.temperature, params.downfall);
    let grass_texel = maps.grass[y * COLORMAP_SIZE as usize + x];
    let foliage_texel = maps.foliage[y * COLORMAP_SIZE as usize + x];

    // dark_forest blends toward dark green *after* the colormap lookup —
    // approximate: applied to both grass and foliage alike, since vanilla's
    // per-biome override list doesn't distinguish them here.
    let (grass_texel, foliage_texel) = if name == "dark_forest" {
        (dark_forest_blend(grass_texel), dark_forest_blend(foliage_texel))
    } else {
        (grass_texel, foliage_texel)
    };

    BiomeColors {
        grass: linear_from_texel(grass_texel),
        foliage: linear_from_texel(foliage_texel),
        water,
    }
}

/// Resolves every name interned in `registry` to a [`BiomeColors`], indexed
/// directly by [`BiomeId`], mirroring [`build_block_tint_table`].
pub fn build_biome_tint_table(registry: &BiomeRegistry, maps: &ColorMaps) -> Vec<BiomeColors> {
    (0..registry.len())
        .map(|i| {
            let id = BiomeId(i as u16);
            let full_name = registry.name(id);
            let name = full_name.strip_prefix("minecraft:").unwrap_or(full_name);
            biome_colors_for(name, maps, &UNKNOWN_BIOME)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `ColorMaps` where every texel encodes its own (x, y) cell as
    /// `[x, y, 0]` — cheap to build (no real PNG needed) and lets a test
    /// assert exactly which cell got sampled.
    fn coordinate_encoding_maps() -> ColorMaps {
        let mut grass = vec![[0u8, 0, 0]; (COLORMAP_SIZE * COLORMAP_SIZE) as usize];
        for y in 0..COLORMAP_SIZE {
            for x in 0..COLORMAP_SIZE {
                grass[(y * COLORMAP_SIZE + x) as usize] = [x as u8, y as u8, 0];
            }
        }
        let foliage = grass.clone();
        ColorMaps { grass, foliage }
    }

    #[test]
    fn colormap_index_multiplies_downfall_by_temperature_first() {
        // temperature 0.5, downfall 1.0 -> rain = 1.0 * 0.5 = 0.5, not 1.0.
        let (x, y) = colormap_xy(0.5, 1.0);
        assert_eq!(x, ((1.0 - 0.5) * 255.0) as usize);
        assert_eq!(y, ((1.0 - 0.5) * 255.0) as usize);
    }

    #[test]
    fn colormap_index_extremes_land_on_the_texture_corners() {
        assert_eq!(colormap_xy(1.0, 1.0), (0, 0));
        assert_eq!(colormap_xy(0.0, 0.0), (255, 255));
    }

    #[test]
    fn plains_and_jungle_resolve_to_different_grass_colours() {
        let mut registry = BiomeRegistry::new(); // interns plains as id 0
        let jungle = registry.intern("minecraft:jungle");
        let maps = coordinate_encoding_maps();

        let table = build_biome_tint_table(&registry, &maps);
        assert_ne!(
            table[BiomeRegistry::PLAINS.0 as usize].grass,
            table[jungle.0 as usize].grass
        );
    }

    #[test]
    fn swamp_and_badlands_bypass_the_colormap() {
        let mut registry = BiomeRegistry::new();
        let swamp = registry.intern("minecraft:swamp");
        let badlands = registry.intern("minecraft:badlands");
        // A colormap that would produce a very different colour than the
        // fixed override, so a wrongly-colormap-derived result is obvious.
        let maps = ColorMaps {
            grass: vec![[0, 0, 0]; (COLORMAP_SIZE * COLORMAP_SIZE) as usize],
            foliage: vec![[0, 0, 0]; (COLORMAP_SIZE * COLORMAP_SIZE) as usize],
        };

        let table = build_biome_tint_table(&registry, &maps);
        assert_eq!(table[swamp.0 as usize].grass, linear_from_hex(0x6A_70_39));
        assert_eq!(table[badlands.0 as usize].grass, linear_from_hex(0x90_81_4D));
    }

    #[test]
    fn grass_blocks_top_is_tinted_and_its_bottom_and_side_are_not() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        // No `grass_block_side_overlay` tile in this atlas — irrelevant to
        // what this test checks (see `grass_blocks_side_overlay_resolves_to_the_atlas_tile`
        // for that).
        let atlas = AtlasUvIndex::for_test(&[]);

        let table = build_block_tint_table(&registry, &atlas);
        let tint = table[grass_block.0 as usize];
        assert_eq!(tint.top, TintSource::Grass);
        assert_eq!(tint.bottom, TintSource::None);
        assert_eq!(tint.side, TintSource::None);
    }

    /// Ticket 014: `grass_block`'s side overlay resolves to whatever atlas
    /// rect the packed atlas gave `grass_block_side_overlay`, tinted by the
    /// biome's grass colour (the same source as the top face).
    #[test]
    fn grass_blocks_side_overlay_resolves_to_the_atlas_tile() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        let overlay_rect = UvRect { u0: 0.5, v0: 0.5, u1: 0.6, v1: 0.6 };
        let atlas = AtlasUvIndex::for_test(&[("grass_block_side_overlay", overlay_rect)]);

        let table = build_block_tint_table(&registry, &atlas);
        let (uv, source) = table[grass_block.0 as usize]
            .side_overlay
            .expect("grass_block should carry a side overlay when the atlas has the tile");
        assert_eq!(uv, overlay_rect);
        assert_eq!(source, TintSource::Grass);
    }

    /// If the atlas ever lacks the overlay tile (a stripped/incomplete
    /// resource pack), grass falls back to no overlay rather than panicking
    /// or guessing a fallback rect that would be silently wrong.
    #[test]
    fn grass_blocks_side_overlay_is_none_when_the_atlas_lacks_the_tile() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        let atlas = AtlasUvIndex::for_test(&[]);

        let table = build_block_tint_table(&registry, &atlas);
        assert_eq!(table[grass_block.0 as usize].side_overlay, None);
    }

    #[test]
    fn unknown_biome_falls_back_to_plains_and_warns_once() {
        let mut registry = BiomeRegistry::new();
        let mystery = registry.intern("minecraft:totally_made_up_biome");
        let maps = coordinate_encoding_maps();

        let table = build_biome_tint_table(&registry, &maps);
        assert_eq!(
            table[mystery.0 as usize].grass,
            table[BiomeRegistry::PLAINS.0 as usize].grass
        );
    }

    #[test]
    fn srgb_to_linear_conversion_actually_happens() {
        // A mid-grey sRGB texel should not come back with the same numeric
        // value in linear space (sRGB gamma is nonlinear near mid-grey).
        let linear = linear_from_texel([128, 128, 128]);
        assert!((linear.red - 128.0 / 255.0).abs() > 0.05);
    }

    /// Loads the real vendored colormaps (mirrors `atlas`'s
    /// `builds_atlas_from_the_real_asset_pack`) — a spot-check that
    /// `assets/minecraft/textures/colormap/{grass,foliage}.png` are still
    /// 256x256 and readable, since [`load_color_maps`] is what `lib.rs`
    /// calls at startup.
    #[test]
    fn loads_the_real_vendored_colormaps() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/minecraft/textures/colormap");
        let maps = load_color_maps(&dir).expect("the vendored colormaps should load");
        assert_eq!(maps.grass.len(), (COLORMAP_SIZE * COLORMAP_SIZE) as usize);
        assert_eq!(maps.foliage.len(), (COLORMAP_SIZE * COLORMAP_SIZE) as usize);
    }
}
