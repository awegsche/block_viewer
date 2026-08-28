//! `ranvil-cli get` (ticket 091) — the simplest block-level read: resolve a
//! block position's region/chunk through [`RegionCache`] and
//! [`mc_anvil::chunkregion::ChunkRegion::get_block`], then parse the palette
//! entry it hands back into a [`BlockState`] with
//! [`BlockState::from_palette_entry`] — the same parse `chunkregion`'s own
//! internals and the viewer's block inspector (ticket 007) already share, not
//! a second NBT-to-`BlockState` decode grown here.
//!
//! Reserved for `get-area`/`column`/`scan` (tickets 092–093), which reuse
//! [`get`]'s coordinate-resolution shape for a box rather than one point.

use bevy::math::IVec3;
use mc_anvil::BlockState;
use serde_json::{json, Value};

use crate::edit::address_of;
use crate::region_cache::RegionCache;

use super::cli::{Cli, GetArgs};
use super::error::CliError;
use super::format::Render;
use super::save::resolve_save;

pub struct GetResult {
    pub save_name: String,
    pub pos: IVec3,
    pub state: BlockState,
}

/// Runs `get`: resolves `args.pos` to a region + region-local coordinates via
/// [`address_of`] (the same address math [`crate::edit`]/the citybuilder use
/// for every other block read/write, so `get` can't drift onto a different
/// `div_euclid`/`rem_euclid` convention), loads that region through a
/// one-off [`RegionCache`], and parses the palette entry `get_block` returns.
///
/// A position outside any loaded/generated chunk is [`CliError::Data`] naming
/// the position — `get_block`'s own [`mc_anvil::MCLoadError::ChunkNotFound`]/
/// [`mc_anvil::MCLoadError::SectionNotFound`] pass through wrapped with that
/// context, never a silent "air".
pub fn get(cli: &Cli, args: &GetArgs) -> Result<GetResult, CliError> {
    let meta = resolve_save(cli)?;
    let pos = args.pos.0;
    let address = address_of(pos);

    let mut cache = RegionCache::new(meta.clone(), 1);
    let region = cache.get_or_load(address.region).map_err(|e| {
        CliError::Data(format!(
            "could not read the region for block ({}, {}, {}): {e}",
            pos.x, pos.y, pos.z
        ))
    })?;

    let entry = region
        .get_block(address.local_x, address.y, address.local_z)
        .map_err(|e| {
            CliError::Data(format!(
                "block ({}, {}, {}) is not in a generated chunk: {e}",
                pos.x, pos.y, pos.z
            ))
        })?;

    let state = BlockState::from_palette_entry(entry).ok_or_else(|| {
        CliError::Data(format!(
            "block ({}, {}, {}) has a palette entry ranvil-cli cannot parse (no Name, or non-string Properties)",
            pos.x, pos.y, pos.z
        ))
    })?;

    Ok(GetResult {
        save_name: meta.name,
        pos,
        state,
    })
}

impl Render for GetResult {
    /// The same string [`BlockState`]'s `Display` already produces, e.g.
    /// `minecraft:oak_stairs[facing=east,half=top]` — copy-pasteable
    /// straight into a `set` command, per the ticket. `render_compact`'s
    /// default (identical to this) is exactly what's wanted here too: `get`
    /// is already one line in any format.
    fn render_text(&self) -> String {
        self.state.to_string()
    }

    fn render_json(&self) -> Value {
        let mut properties = serde_json::Map::new();
        for (key, value) in self.state.properties() {
            properties.insert(key.clone(), json!(value));
        }
        json!({
            "pos": [self.pos.x, self.pos.y, self.pos.z],
            "name": self.state.name(),
            "properties": properties,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mc_anvil::chunkregion::ChunkRegion;
    use mc_anvil::region::{Region, CHUNKS_PER_REGION};
    use rnbt::{NbtField, NbtList};

    use super::*;

    /// A palette entry with no `Properties` compound — same shape
    /// `blueprint::extract`'s and `chunk.rs`'s own tests build.
    fn plain(name: &str) -> NbtField {
        NbtField::new_compound("", vec![NbtField::new_string("Name", name)])
    }

    fn with_properties(name: &str, properties: &[(&str, &str)]) -> NbtField {
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_string("Name", name),
                NbtField::new_compound(
                    "Properties",
                    properties
                        .iter()
                        .map(|(k, v)| NbtField::new_string(*k, *v))
                        .collect::<Vec<_>>(),
                ),
            ],
        )
    }

    fn byte_field(name: &str, value: i8) -> NbtField {
        NbtField {
            name: name.to_string(),
            value: rnbt::NbtValue::Byte(value as u8),
        }
    }

    fn section(y: i8, palette: Vec<NbtField>, data: Option<Vec<i64>>) -> NbtField {
        let mut block_states = vec![NbtField::new_list("palette", NbtList::Compound(palette))];
        if let Some(data) = data {
            block_states.push(NbtField::new_long_array("data", data));
        }
        NbtField::new_compound(
            "",
            vec![
                byte_field("Y", y),
                NbtField::new_compound("block_states", block_states),
            ],
        )
    }

    fn chunk_nbt(sections: Vec<NbtField>) -> NbtField {
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_string("Status", "minecraft:full"),
                NbtField::new_list("sections", NbtList::Compound(sections)),
            ],
        )
    }

    /// A `ChunkRegion` with a single chunk at (0, 0) set directly on the
    /// public `chunks` field — bypassing `load_chunks`/`set_chunk`, which
    /// would otherwise require a real `.mca` file on disk, per this crate's
    /// existing fixture-region convention (`region_cache.rs`'s own tests
    /// assume a real save; this stays a pure-unit fixture instead, the way
    /// `chunk.rs`'s tests build synthetic chunk NBT rather than reading one).
    fn region_with_chunk_zero(chunk: NbtField) -> ChunkRegion {
        let mut region: ChunkRegion = Region::new(0, 0, PathBuf::from("unused")).into();
        let mut chunks = vec![None; CHUNKS_PER_REGION];
        chunks[0] = Some(chunk);
        region.chunks = Some(chunks);
        region
    }

    #[test]
    fn get_block_reads_a_no_properties_block_via_the_single_entry_fast_path() {
        // A single-entry palette section omits `data` entirely — the fast
        // path `get_block`'s own docs describe.
        let region = region_with_chunk_zero(chunk_nbt(vec![section(
            0,
            vec![plain("minecraft:stone")],
            None,
        )]));

        let entry = region.get_block(0, 0, 0).expect("block should resolve");
        let state = BlockState::from_palette_entry(entry).expect("valid palette entry");

        assert_eq!(state.name(), "minecraft:stone");
        assert!(state.properties().is_empty());
        assert_eq!(state.to_string(), "minecraft:stone");
    }

    #[test]
    fn get_block_reads_a_with_properties_block_through_a_packed_multi_entry_palette() {
        // Two entries, 4 bits/index (minimum packed width), every index 0
        // (air) except position (0,0,0) which is index 1 (the stairs).
        let palette = vec![
            plain("minecraft:air"),
            with_properties("minecraft:oak_stairs", &[("facing", "east"), ("half", "top")]),
        ];
        let mut indices = [0u64; 4096];
        indices[0] = 1;
        let bit_size = 4;
        let indices_per_long = 64 / bit_size;
        let mut data = vec![0i64; indices.len().div_ceil(indices_per_long)];
        for (i, &value) in indices.iter().enumerate() {
            data[i / indices_per_long] |= (value as i64) << ((i % indices_per_long) * bit_size);
        }

        let region = region_with_chunk_zero(chunk_nbt(vec![section(0, palette, Some(data))]));
        let entry = region.get_block(0, 0, 0).expect("block should resolve");
        let state = BlockState::from_palette_entry(entry).expect("valid palette entry");

        assert_eq!(state.name(), "minecraft:oak_stairs");
        assert_eq!(state.property("facing"), Some("east"));
        assert_eq!(state.property("half"), Some("top"));
        assert_eq!(state.to_string(), "minecraft:oak_stairs[facing=east,half=top]");
    }

    #[test]
    fn missing_chunk_is_not_silently_air() {
        let region: ChunkRegion = Region::new(0, 0, PathBuf::from("unused")).into();
        let mut region = region;
        region.chunks = Some(vec![None; CHUNKS_PER_REGION]);

        let err = region.get_block(0, 0, 0).unwrap_err();
        assert!(matches!(err, mc_anvil::MCLoadError::ChunkNotFound(0, 0)));
    }

    #[test]
    fn render_json_matches_the_documented_shape() {
        let result = GetResult {
            save_name: "world".to_string(),
            pos: IVec3::new(200, 60, 150),
            state: BlockState::new("minecraft:oak_stairs")
                .with_property("facing", "east")
                .with_property("half", "top"),
        };

        assert_eq!(
            result.render_json(),
            json!({
                "pos": [200, 60, 150],
                "name": "minecraft:oak_stairs",
                "properties": {"facing": "east", "half": "top"},
            })
        );
    }

    #[test]
    fn render_text_and_compact_are_the_copy_pasteable_block_state_string() {
        let result = GetResult {
            save_name: "world".to_string(),
            pos: IVec3::new(1, 2, 3),
            state: BlockState::new("minecraft:stone"),
        };

        assert_eq!(result.render_text(), "minecraft:stone");
        assert_eq!(result.render_compact(), "minecraft:stone");
    }
}
