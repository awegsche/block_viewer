//! Hardcoded biome temperature/downfall/water-colour table (ticket 013).
//!
//! A biome's temperature and downfall are **not** in the save file and
//! **not** in the vendored resource pack — they live in vanilla's
//! `worldgen/biome/*.json`, which isn't vendored in this repo either. This
//! table transcribes that data by hand for the overworld biomes a real save
//! is most likely to contain.
//!
//! Deliberately incomplete (see the ticket's "ship it incomplete rather than
//! block on all of it"): the Nether and the End aren't covered (this viewer
//! targets overworld terrain first), and a handful of rare
//! climate-parameter-only variants (`sunflower_plains` siblings,
//! `ice_spikes`, cave biomes) are approximated from their closest documented
//! relative rather than independently verified. An unlisted name falls back
//! to plains (see [`params_for`]'s caller, [`super::tint::build_biome_tint_table`])
//! rather than failing to render — a visible but survivable gap.
//!
//! Water colour is vanilla's per-biome still-water tint, `0xRRGGBB` sRGB
//! (the same hex literals Minecraft's own biome JSON uses) — most biomes
//! share [`DEFAULT_WATER`]; oceans and swamps are the exceptions worth
//! calling out by name.

/// A biome's climate parameters plus its still-water tint (sRGB `0xRRGGBB`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiomeParams {
    pub temperature: f32,
    pub downfall: f32,
    pub water_color: u32,
}

/// The water colour the overwhelming majority of overworld biomes use —
/// oceans and swamps are the documented exceptions in [`BIOMES`].
const DEFAULT_WATER: u32 = 0x3F76E4;

// (name without "minecraft:", temperature, downfall, water_color)
const BIOMES: &[(&str, f32, f32, u32)] = &[
    ("plains", 0.8, 0.4, DEFAULT_WATER),
    ("sunflower_plains", 0.8, 0.4, DEFAULT_WATER),
    ("forest", 0.7, 0.8, DEFAULT_WATER),
    ("flower_forest", 0.7, 0.8, DEFAULT_WATER),
    ("birch_forest", 0.6, 0.6, DEFAULT_WATER),
    ("old_growth_birch_forest", 0.6, 0.6, DEFAULT_WATER),
    // grass/foliage get vanilla's dark-green blend on top of this, applied
    // in `tint::biome_colors_for` — the climate params here just pick the
    // colormap cell that blend starts from.
    ("dark_forest", 0.7, 0.8, DEFAULT_WATER),
    ("taiga", 0.25, 0.8, DEFAULT_WATER),
    ("snowy_taiga", -0.5, 0.4, DEFAULT_WATER),
    ("old_growth_pine_taiga", 0.3, 0.8, DEFAULT_WATER),
    ("old_growth_spruce_taiga", 0.25, 0.8, DEFAULT_WATER),
    ("desert", 2.0, 0.0, DEFAULT_WATER),
    ("savanna", 1.2, 0.0, DEFAULT_WATER),
    ("savanna_plateau", 1.0, 0.0, DEFAULT_WATER),
    ("windswept_savanna", 1.1, 0.0, DEFAULT_WATER), // approximate
    ("jungle", 0.95, 0.9, DEFAULT_WATER),
    ("sparse_jungle", 0.95, 0.8, DEFAULT_WATER),
    ("bamboo_jungle", 0.95, 0.9, DEFAULT_WATER),
    // grass/foliage are fixed colours in `tint::biome_colors_for`, bypassing
    // the colormap entirely — the climate params here only matter for
    // anything else that might key off them later (e.g. water, if badlands
    // ever needs a non-default one).
    ("badlands", 2.0, 0.0, DEFAULT_WATER),
    ("eroded_badlands", 2.0, 0.0, DEFAULT_WATER),
    ("wooded_badlands", 2.0, 0.0, DEFAULT_WATER),
    ("swamp", 0.8, 0.9, 0x617B64),
    ("mangrove_swamp", 0.8, 0.9, DEFAULT_WATER), // approximate
    ("windswept_hills", 0.2, 0.3, DEFAULT_WATER),
    ("windswept_gravelly_hills", 0.2, 0.3, DEFAULT_WATER),
    ("windswept_forest", 0.2, 0.3, DEFAULT_WATER),
    ("stony_shore", 0.2, 0.3, DEFAULT_WATER),
    ("beach", 0.8, 0.4, DEFAULT_WATER),
    ("snowy_beach", 0.05, 0.3, DEFAULT_WATER),
    ("mushroom_fields", 0.9, 1.0, DEFAULT_WATER),
    ("snowy_plains", 0.0, 0.5, DEFAULT_WATER),
    ("ice_spikes", 0.0, 0.5, DEFAULT_WATER), // approximate — shares snowy_plains' climate
    ("river", 0.5, 0.5, DEFAULT_WATER),
    ("frozen_river", 0.0, 0.5, DEFAULT_WATER),
    ("ocean", 0.5, 0.5, DEFAULT_WATER),
    ("deep_ocean", 0.5, 0.5, DEFAULT_WATER),
    ("warm_ocean", 0.5, 0.5, 0x43D5EE),
    ("lukewarm_ocean", 0.5, 0.5, 0x45ADF2),
    ("deep_lukewarm_ocean", 0.5, 0.5, 0x45ADF2),
    ("cold_ocean", 0.5, 0.5, 0x3D57D6),
    ("deep_cold_ocean", 0.5, 0.5, 0x3D57D6),
    ("frozen_ocean", 0.0, 0.5, 0x3938C9),
    ("deep_frozen_ocean", 0.0, 0.5, 0x3938C9),
    ("meadow", 0.5, 0.8, DEFAULT_WATER),       // approximate
    ("cherry_grove", 0.5, 0.8, DEFAULT_WATER), // approximate, see ticket 013's note on overrides
    ("grove", -0.2, 0.8, DEFAULT_WATER),
    ("snowy_slopes", -0.3, 0.9, DEFAULT_WATER),
    ("frozen_peaks", -0.7, 0.9, DEFAULT_WATER),
    ("jagged_peaks", -0.7, 0.9, DEFAULT_WATER),
    ("stony_peaks", 1.0, 0.3, DEFAULT_WATER),
    ("lush_caves", 0.5, 0.5, DEFAULT_WATER),      // approximate
    ("dripstone_caves", 0.8, 0.4, DEFAULT_WATER), // approximate
    ("deep_dark", 0.8, 0.4, DEFAULT_WATER),       // approximate
];

/// Looks up `name`'s (already `"minecraft:"`-stripped) climate parameters.
/// `None` means `name` isn't in [`BIOMES`] at all — the caller falls back to
/// plains and warns once (see [`super::tint::build_biome_tint_table`]).
pub fn params_for(name: &str) -> Option<BiomeParams> {
    BIOMES.iter().find(|&&(n, ..)| n == name).map(|&(_, temperature, downfall, water_color)| BiomeParams {
        temperature,
        downfall,
        water_color,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plains_is_in_the_table() {
        let params = params_for("plains").expect("plains should be in the table");
        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.downfall, 0.4);
    }

    #[test]
    fn unknown_biome_is_not_in_the_table() {
        assert!(params_for("some_unknown_biome").is_none());
    }

    #[test]
    fn jungle_and_taiga_have_different_climates() {
        let jungle = params_for("jungle").unwrap();
        let taiga = params_for("taiga").unwrap();
        assert_ne!(
            (jungle.temperature, jungle.downfall),
            (taiga.temperature, taiga.downfall)
        );
    }
}
