//! NBT -> dense block grids.
//!
//! [`decode_chunk`] turns a loaded chunk's [`NbtField`] into a
//! [`ChunkColumn`] of [`ChunkSection`]s, decoding each section's packed
//! `block_states` once instead of re-walking the NBT tree per block (see
//! ticket 002). `ChunkRegion::get_block` (in `ranvil`) stays useful for
//! one-off lookups; this is for meshing a whole chunk.
//!
//! Ticket 012 adds each section's `biomes` grid alongside `block_states` —
//! same palette + packed-data shape, but a string palette, no 4-bit
//! minimum on the bit width, and a 4x4x4 (not 16x16x16) grid. See
//! [`decode_biomes`] for the differences in full.

use std::collections::HashSet;

use rnbt::NbtField;

use super::biome::{BiomeId, BiomeRegistry};
use super::block::{BlockId, BlockRegistry};

/// Width/height/depth of a chunk section, in blocks.
pub const SECTION_SIZE: usize = 16;
/// Number of blocks in a chunk section (16x16x16).
pub const SECTION_VOLUME: usize = SECTION_SIZE * SECTION_SIZE * SECTION_SIZE;
/// Width/height/depth of a section's biome grid, in 4-block cells.
pub const BIOME_GRID_SIZE: usize = 4;
/// Number of biome cells in a chunk section (4x4x4, one per 4x4x4 block
/// cube — ticket 012).
pub const BIOME_GRID_VOLUME: usize = BIOME_GRID_SIZE * BIOME_GRID_SIZE * BIOME_GRID_SIZE;

/// One 16x16x16 layer of a chunk column, decoded to a flat array of
/// [`BlockId`]s. Indexed `x + z*16 + y*256` (locals 0..16 each), matching
/// the order Minecraft packs `block_states.data` in.
#[derive(Debug, Clone)]
pub struct ChunkSection {
    /// Section Y coordinate (`world_y.div_euclid(16)`), signed — e.g. -4 is
    /// the bottommost section of a 1.18+ world.
    pub y: i8,
    pub blocks: Box<[BlockId; SECTION_VOLUME]>,
    /// One [`BiomeId`] per 4x4x4 sub-cube (ticket 012), indexed
    /// `x + z*4 + y*16` at *cell* resolution — go through [`Self::biome_at`]
    /// rather than indexing this directly, since callers work in block
    /// locals.
    pub biomes: Box<[BiomeId; BIOME_GRID_VOLUME]>,
}

impl ChunkSection {
    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + z * SECTION_SIZE + y * SECTION_SIZE * SECTION_SIZE
    }

    /// Block at section-local coordinates (each 0..16).
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockId {
        self.blocks[Self::index(x, y, z)]
    }

    #[inline]
    fn biome_index(x: usize, y: usize, z: usize) -> usize {
        (x / 4) + (z / 4) * BIOME_GRID_SIZE + (y / 4) * BIOME_GRID_SIZE * BIOME_GRID_SIZE
    }

    /// Biome at the 4x4x4 cell containing section-local **block**
    /// coordinates (each 0..16) — the division from block locals down to
    /// biome-cell locals happens here, once, rather than at every call
    /// site.
    pub fn biome_at(&self, x: usize, y: usize, z: usize) -> BiomeId {
        self.biomes[Self::biome_index(x, y, z)]
    }
}

/// A decoded chunk column: every non-uniform-air section of a single (x, z)
/// chunk, block names resolved to [`BlockId`]s via a shared
/// [`BlockRegistry`]. Sections with no blocks at all (uniform air, or
/// lighting-only sentinel sections) are simply absent from `sections`.
#[derive(Debug, Clone)]
pub struct ChunkColumn {
    /// World chunk coordinates (this column covers block x in
    /// `x*16..x*16+16`, and likewise for z). Nothing outside tests reads
    /// these directly since ticket 005-e — every caller already has the
    /// coordinate from whatever key it looked the column up by (a
    /// `DecodedWorld.columns` key, a `ChunkLoadResult.coord`, ...) — but
    /// they stay on the decoded value itself as a sanity-checkable source of
    /// truth (see `chunk_pipeline`'s tests) rather than something only ever
    /// inferred from a map key.
    #[allow(dead_code)]
    pub x: i32,
    #[allow(dead_code)]
    pub z: i32,
    pub sections: Vec<ChunkSection>,
}

impl ChunkColumn {
    /// Topmost non-air block at local column `(local_x, local_z)` (each
    /// 0..16), scanning sections top-down. Returns `(world_y, block_id)`.
    ///
    /// Used to place the camera above the terrain surface at startup by
    /// ticket 006's original eager-decode version of `lib.rs::spawn_point`;
    /// ticket 005-e's streaming startup can no longer do that (nothing is
    /// decoded yet at startup), but this stays for the block-under-cursor
    /// readout planned in ticket 007.
    #[allow(dead_code)]
    pub fn topmost_non_air(&self, local_x: usize, local_z: usize) -> Option<(i32, BlockId)> {
        let mut by_height: Vec<&ChunkSection> = self.sections.iter().collect();
        by_height.sort_by(|a, b| b.y.cmp(&a.y));

        for section in by_height {
            for dy in (0..SECTION_SIZE).rev() {
                let id = section.get(local_x, dy, local_z);
                if id != BlockRegistry::AIR {
                    let world_y = section.y as i32 * SECTION_SIZE as i32 + dy as i32;
                    return Some((world_y, id));
                }
            }
        }
        None
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The chunk's `Status` isn't `"minecraft:full"` (partially generated).
    NotFullyGenerated(String),
    MissingField(&'static str),
    UnexpectedType(&'static str),
    EmptyPalette,
    PaletteIndexOutOfRange(usize),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::NotFullyGenerated(status) => {
                write!(f, "chunk is not fully generated (Status = {status})")
            }
            DecodeError::MissingField(name) => write!(f, "missing NBT field: {name}"),
            DecodeError::UnexpectedType(name) => {
                write!(f, "unexpected NBT type for field: {name}")
            }
            DecodeError::EmptyPalette => write!(f, "section palette is empty"),
            DecodeError::PaletteIndexOutOfRange(i) => write!(f, "palette index out of range: {i}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decodes a single chunk's root NBT (as handed back by
/// `mc_anvil::ChunkRegion::get_chunk`/`get_chunk_or_load`) into a dense
/// [`ChunkColumn`].
///
/// Interns every encountered block name into `registry`, and every
/// encountered biome name into `biomes` (ticket 012); pass the same
/// registries across a whole save so [`BlockId`]s and [`BiomeId`]s stay
/// stable.
///
/// Returns [`DecodeError::NotFullyGenerated`] for chunks whose `Status`
/// isn't `"minecraft:full"` — callers should skip these rather than treat
/// them as a hard failure.
pub fn decode_chunk(
    nbt: &NbtField,
    registry: &mut BlockRegistry,
    biomes: &mut BiomeRegistry,
) -> Result<ChunkColumn, DecodeError> {
    let status = nbt
        .get_string("Status")
        .ok_or(DecodeError::MissingField("Status"))?;
    if status != "minecraft:full" {
        return Err(DecodeError::NotFullyGenerated(status.clone()));
    }

    let x = nbt.get_int("xPos").ok_or(DecodeError::MissingField("xPos"))?;
    let z = nbt.get_int("zPos").ok_or(DecodeError::MissingField("zPos"))?;

    let section_entries = nbt
        .get_list("sections")
        .ok_or(DecodeError::MissingField("sections"))?
        .as_compound_list()
        .ok_or(DecodeError::UnexpectedType("sections"))?;

    // Dedupes the "no biome data" log line to once per chunk (mirrors
    // `atlas::build_block_uv_table`'s per-call warned set) rather than once
    // per missing section — a chunk missing biomes on one section is
    // usually missing it on all of them.
    let mut warned_missing_biomes: HashSet<&'static str> = HashSet::new();

    let mut sections = Vec::new();
    for section in section_entries {
        // Select sections by their own `Y` tag, not list position — the
        // list may carry lighting-only sentinel sections or gaps, so
        // position alone doesn't tell you the section's world Y.
        let Some(y) = section.get_byte("Y") else {
            continue;
        };
        let y = y as i8;

        // Sections without `block_states` (e.g. those lighting-only
        // sentinels) carry no blocks — and since they render nothing, their
        // `biomes` is never sampled either, so skip before touching it.
        let Some(block_states) = section.get("block_states") else {
            continue;
        };

        let palette = block_states
            .get_list("palette")
            .ok_or(DecodeError::MissingField("block_states.palette"))?
            .as_compound_list()
            .ok_or(DecodeError::UnexpectedType("block_states.palette"))?;

        if palette.is_empty() {
            return Err(DecodeError::EmptyPalette);
        }

        let palette_ids = palette
            .iter()
            .map(|entry| {
                let name = entry
                    .get_string("Name")
                    .ok_or(DecodeError::MissingField("Name"))?;
                Ok(registry.intern(name))
            })
            .collect::<Result<Vec<_>, DecodeError>>()?;

        if palette_ids.len() == 1 {
            let id = palette_ids[0];
            // Fast path: a uniform section needs no packed `data` at all
            // (Minecraft omits it). A uniform *air* section costs nothing —
            // it's simply left out of `sections`, and every consumer treats
            // absence as air.
            if id == BlockRegistry::AIR {
                continue;
            }
            let section_biomes =
                decode_biomes(section, biomes, &mut warned_missing_biomes)?;
            sections.push(ChunkSection {
                y,
                blocks: Box::new([id; SECTION_VOLUME]),
                biomes: section_biomes,
            });
            continue;
        }

        let data = block_states
            .get_long_array("data")
            .ok_or(DecodeError::MissingField("block_states.data"))?;

        // Minecraft enforces a minimum of 4 bits per block-state index.
        let bit_size = ((palette_ids.len() as u32 - 1).ilog2() + 1).max(4);
        let indices_per_long = 64 / bit_size as usize;
        let mask = (1u64 << bit_size) - 1;

        let mut blocks = Box::new([BlockRegistry::AIR; SECTION_VOLUME]);
        for (idx, block) in blocks.iter_mut().enumerate() {
            // Indices don't span longs (1.16+ format): any leftover high
            // bits of a long are padding and are simply never read here.
            let long_index = idx / indices_per_long;
            let slot = idx % indices_per_long;
            let long_value = *data
                .get(long_index)
                .ok_or(DecodeError::UnexpectedType("block_states.data"))? as u64;
            let palette_index = ((long_value >> (slot * bit_size as usize)) & mask) as usize;
            *block = *palette_ids
                .get(palette_index)
                .ok_or(DecodeError::PaletteIndexOutOfRange(palette_index))?;
        }

        let section_biomes = decode_biomes(section, biomes, &mut warned_missing_biomes)?;
        sections.push(ChunkSection { y, blocks, biomes: section_biomes });
    }

    Ok(ChunkColumn { x, z, sections })
}

/// Decodes one section's `biomes` compound (a sibling of `block_states`)
/// into [`BIOME_GRID_VOLUME`] [`BiomeId`]s. Same palette + packed-data shape
/// as `block_states`, with three differences (ticket 012):
///
/// 1. The palette is a list of *strings* (`TAG_List<TAG_String>`), not
///    compounds — the biome name is the entry itself, no `Name` field to
///    dig out.
/// 2. No 4-bit minimum on the bit width: `((len - 1).ilog2() + 1).max(1)`,
///    so a 2-entry palette packs at 1 bit.
/// 3. The grid is 4x4x4 (64 entries, one per 4x4x4 block cube), not
///    16x16x16 — same X-fastest, Y-slowest index order as `block_states`,
///    just at a quarter resolution per axis.
///
/// A section with no `biomes` compound, or an empty palette, fills with
/// [`BiomeRegistry::PLAINS`] and logs once per chunk via `warned` (the
/// caller's `HashSet`, not a fresh one per section) rather than once per
/// section.
fn decode_biomes(
    section: &NbtField,
    registry: &mut BiomeRegistry,
    warned: &mut HashSet<&'static str>,
) -> Result<Box<[BiomeId; BIOME_GRID_VOLUME]>, DecodeError> {
    let missing = |warned: &mut HashSet<&'static str>| {
        if warned.insert("biomes") {
            println!(
                "block_viewer: section has no usable biome data — defaulting to minecraft:plains"
            );
        }
        Box::new([BiomeRegistry::PLAINS; BIOME_GRID_VOLUME])
    };

    let Some(biomes) = section.get("biomes") else {
        return Ok(missing(warned));
    };
    let Some(palette) = biomes.get_list("palette").and_then(|list| list.as_string_list()) else {
        return Ok(missing(warned));
    };
    if palette.is_empty() {
        return Ok(missing(warned));
    }

    let palette_ids: Vec<BiomeId> = palette.iter().map(|name| registry.intern(name)).collect();

    if palette_ids.len() == 1 {
        // Same fast path as `block_states`: a uniform palette omits `data`
        // entirely.
        return Ok(Box::new([palette_ids[0]; BIOME_GRID_VOLUME]));
    }

    let data = biomes
        .get_long_array("data")
        .ok_or(DecodeError::MissingField("biomes.data"))?;

    // Unlike `block_states`, biome indices have no 4-bit minimum.
    let bit_size = ((palette_ids.len() as u32 - 1).ilog2() + 1).max(1);
    let indices_per_long = 64 / bit_size as usize;
    let mask = (1u64 << bit_size) - 1;

    let mut ids = Box::new([BiomeRegistry::PLAINS; BIOME_GRID_VOLUME]);
    for (idx, biome) in ids.iter_mut().enumerate() {
        // Same padding rule as `block_states`: indices don't span longs, so
        // leftover high bits of a long are never read.
        let long_index = idx / indices_per_long;
        let slot = idx % indices_per_long;
        let long_value = *data
            .get(long_index)
            .ok_or(DecodeError::UnexpectedType("biomes.data"))? as u64;
        let palette_index = ((long_value >> (slot * bit_size as usize)) & mask) as usize;
        *biome = *palette_ids
            .get(palette_index)
            .ok_or(DecodeError::PaletteIndexOutOfRange(palette_index))?;
    }

    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnbt::{NbtField, NbtList, NbtValue};

    fn byte_field(name: &str, value: i8) -> NbtField {
        NbtField {
            name: name.to_string(),
            value: NbtValue::Byte(value as u8),
        }
    }

    fn palette(names: &[&str]) -> NbtList {
        NbtList::Compound(
            names
                .iter()
                .map(|n| NbtField::new_compound("", vec![NbtField::new_string("Name", *n)]))
                .collect(),
        )
    }

    fn biome_palette(names: &[&str]) -> NbtList {
        NbtList::String(names.iter().map(|n| n.to_string()).collect())
    }

    /// A `biomes` compound with a palette but no `data` — matches how
    /// Minecraft omits `data` for a single-entry palette.
    fn biomes_uniform(names: &[&str]) -> NbtField {
        NbtField::new_compound("biomes", vec![NbtField::new_list("palette", biome_palette(names))])
    }

    /// A `biomes` compound with a packed multi-entry palette.
    fn biomes_packed(names: &[&str], data: Vec<i64>) -> NbtField {
        NbtField::new_compound(
            "biomes",
            vec![
                NbtField::new_list("palette", biome_palette(names)),
                NbtField::new_long_array("data", data),
            ],
        )
    }

    fn section_uniform(y: i8, name: &str) -> NbtField {
        let block_states = NbtField::new_compound(
            "block_states",
            vec![NbtField::new_list("palette", palette(&[name]))],
        );
        NbtField::new_compound("", vec![byte_field("Y", y), block_states])
    }

    fn section_packed(y: i8, names: &[&str], data: Vec<i64>) -> NbtField {
        let block_states = NbtField::new_compound(
            "block_states",
            vec![
                NbtField::new_list("palette", palette(names)),
                NbtField::new_long_array("data", data),
            ],
        );
        NbtField::new_compound("", vec![byte_field("Y", y), block_states])
    }

    /// [`section_uniform`] plus an explicit `biomes` compound, for tests
    /// that care what the biome grid decodes to rather than letting it fall
    /// back to plains.
    fn section_uniform_with_biomes(y: i8, block_name: &str, biomes: NbtField) -> NbtField {
        let block_states = NbtField::new_compound(
            "block_states",
            vec![NbtField::new_list("palette", palette(&[block_name]))],
        );
        NbtField::new_compound("", vec![byte_field("Y", y), block_states, biomes])
    }

    fn chunk_root(x: i32, z: i32, status: &str, sections: Vec<NbtField>) -> NbtField {
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_string("Status", status),
                NbtField::new_i32("xPos", x),
                NbtField::new_i32("zPos", z),
                NbtField::new_list("sections", NbtList::Compound(sections)),
            ],
        )
    }

    #[test]
    fn single_entry_air_section_costs_nothing() {
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_uniform(-4, "minecraft:air")],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();
        assert!(
            column.sections.is_empty(),
            "uniform air section should be omitted, not stored"
        );
    }

    #[test]
    fn single_entry_non_air_section_fills_uniformly() {
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_uniform(-4, "minecraft:stone")],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();
        assert_eq!(column.sections.len(), 1);
        let stone = registry.intern("minecraft:stone");
        assert!(column.sections[0].blocks.iter().all(|&b| b == stone));
    }

    #[test]
    fn small_palette_decodes_at_four_bit_minimum() {
        // 3-entry palette -> naive `ilog2(len-1)+1` would give 2 bits; the
        // real minimum is 4 (ticket 001), which needs ceil(4096/16) = 256
        // longs to cover every index. Single dirt block at flat idx 0
        // (dx=0, dy=0, dz=0) -> long 0, slot 0.
        let mut data = vec![0i64; 256];
        data[0] = 2; // palette index 2 = "minecraft:dirt"
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_packed(
                -4,
                &["minecraft:air", "minecraft:stone", "minecraft:dirt"],
                data,
            )],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();
        let dirt = registry.intern("minecraft:dirt");
        assert_eq!(column.sections[0].get(0, 0, 0), dirt);
        assert_eq!(column.sections[0].get(1, 0, 0), BlockRegistry::AIR);
    }

    #[test]
    fn large_palette_uses_five_bits() {
        // 17-entry palette -> bit_size = ilog2(16)+1 = 5, needing
        // ceil(4096/12) = 342 longs (matches ticket 001's real-save
        // evidence for a 17-entry palette). Block at flat idx 1
        // (dx=1, dy=0, dz=0) -> long 0, slot 1 -> bit offset 5.
        let names: Vec<String> = (0..17).map(|i| format!("minecraft:block_{i}")).collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut data = vec![0i64; 342];
        data[0] = 16i64 << 5; // palette index 16
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_packed(-4, &name_refs, data)],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();
        let expected = registry.intern("minecraft:block_16");
        assert_eq!(column.sections[0].get(1, 0, 0), expected);
    }

    #[test]
    fn sections_are_selected_by_y_field_not_list_position() {
        // A gap and reversed order: list position 0 has Y=10, position 1
        // has Y=-4. Positional indexing would misattribute these.
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![
                section_uniform(10, "minecraft:stone"),
                section_uniform(-4, "minecraft:dirt"),
            ],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();
        assert_eq!(column.sections.len(), 2);

        let stone = registry.intern("minecraft:stone");
        let dirt = registry.intern("minecraft:dirt");
        let at_y10 = column.sections.iter().find(|s| s.y == 10).unwrap();
        let at_ym4 = column.sections.iter().find(|s| s.y == -4).unwrap();
        assert_eq!(at_y10.get(0, 0, 0), stone);
        assert_eq!(at_ym4.get(0, 0, 0), dirt);
    }

    #[test]
    fn skips_chunks_that_are_not_fully_generated() {
        let root = chunk_root(0, 0, "minecraft:carvers", vec![]);
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let err = decode_chunk(&root, &mut registry, &mut biomes).unwrap_err();
        assert!(
            matches!(err, DecodeError::NotFullyGenerated(ref status) if status == "minecraft:carvers")
        );
    }

    #[test]
    fn topmost_non_air_scans_sections_top_down() {
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![
                section_uniform(-4, "minecraft:stone"),
                section_uniform(0, "minecraft:air"),
                section_uniform(1, "minecraft:grass_block"),
            ],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();

        let (world_y, id) = column.topmost_non_air(0, 0).unwrap();
        assert_eq!(world_y, SECTION_SIZE as i32 + 15);
        assert_eq!(registry.name(id), "minecraft:grass_block");
    }

    #[test]
    fn uniform_biome_palette_fills_every_cell() {
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_uniform_with_biomes(
                -4,
                "minecraft:stone",
                biomes_uniform(&["minecraft:forest"]),
            )],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();

        let forest = biomes.intern("minecraft:forest");
        assert!(column.sections[0].biomes.iter().all(|&b| b == forest));
    }

    #[test]
    fn two_entry_biome_palette_packs_at_one_bit() {
        // 2-entry palette -> naive `.max(4)` (the block_states minimum)
        // would misread this; the real minimum here is 1 bit. 64 cells at
        // 1 bit each fit in a single long, so index 1 (cell x=1,y=0,z=0)
        // lands at bit offset 1.
        let data = vec![0b10i64];
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_uniform_with_biomes(
                -4,
                "minecraft:stone",
                biomes_packed(&["minecraft:plains", "minecraft:desert"], data),
            )],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();

        let desert = biomes.intern("minecraft:desert");
        assert_eq!(column.sections[0].biomes[1], desert);
        assert_eq!(column.sections[0].biomes[0], BiomeRegistry::PLAINS);
    }

    #[test]
    fn five_entry_biome_palette_reads_the_right_cell_at_three_bits() {
        // 5-entry palette -> bit_size = ilog2(4)+1 = 3, 21 cells per long.
        // Cell index 1 -> long 0, bit offset 3.
        let mut data = vec![0i64; 4];
        data[0] = 4i64 << 3; // palette index 4
        let names = [
            "minecraft:plains",
            "minecraft:forest",
            "minecraft:desert",
            "minecraft:taiga",
            "minecraft:swamp",
        ];
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_uniform_with_biomes(
                -4,
                "minecraft:stone",
                biomes_packed(&names, data),
            )],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();

        let swamp = biomes.intern("minecraft:swamp");
        assert_eq!(column.sections[0].biomes[1], swamp);
    }

    #[test]
    fn section_with_no_biomes_compound_defaults_to_plains_without_panicking() {
        let root = chunk_root(
            0,
            0,
            "minecraft:full",
            vec![section_uniform(-4, "minecraft:stone")],
        );
        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let column = decode_chunk(&root, &mut registry, &mut biomes).unwrap();

        assert!(column.sections[0]
            .biomes
            .iter()
            .all(|&b| b == BiomeRegistry::PLAINS));
    }

    #[test]
    fn biome_at_maps_block_locals_to_the_right_4x4x4_cell() {
        let mut cells = Box::new([BiomeRegistry::PLAINS; BIOME_GRID_VOLUME]);
        let desert = BiomeId(1);
        cells[1] = desert; // cell (x=1, y=0, z=0)
        let section = ChunkSection {
            y: 0,
            blocks: Box::new([BlockRegistry::AIR; SECTION_VOLUME]),
            biomes: cells,
        };

        // (0,0,0) and (3,3,3) both fall in cell (0,0,0) — still plains.
        assert_eq!(section.biome_at(0, 0, 0), BiomeRegistry::PLAINS);
        assert_eq!(section.biome_at(3, 3, 3), BiomeRegistry::PLAINS);
        // (4,0,0) crosses into the next cell on X — cell index 1.
        assert_eq!(section.biome_at(4, 0, 0), desert);
    }

    /// Mirrors `chunk_pipeline`'s
    /// `load_and_mesh_chunk_decodes_and_meshes_a_real_chunk` convention for
    /// resolving a real region, but decodes a spread of chunks across it
    /// (its diagonal) rather than just the centre one — a single chunk can
    /// easily land entirely inside one biome, which would make "the biome
    /// registry is non-trivial" flaky depending on exactly which chunk the
    /// centre happens to be.
    #[test]
    fn decodes_a_plausible_biome_set_from_a_real_chunk() {
        use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;

        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        let mut cache = crate::region_cache::RegionCache::new(meta, 4);
        let region = cache.get_or_load((rx, rz)).expect("region should load");

        let mut registry = BlockRegistry::new();
        let mut biomes = BiomeRegistry::new();
        let mut decoded_any = false;
        for step in 0..REGION_WIDTH_IN_CHUNKS {
            let Some(nbt) = region.get_chunk(step, step) else { continue };
            let nbt = nbt.clone();
            match decode_chunk(&nbt, &mut registry, &mut biomes) {
                Ok(_) => decoded_any = true,
                Err(DecodeError::NotFullyGenerated(_)) => continue,
                Err(err) => panic!("failed to decode chunk ({step}, {step}): {err}"),
            }
        }

        assert!(decoded_any, "expected at least one fully-generated chunk along the region's diagonal");
        assert!(
            biomes.len() > 1,
            "expected at least one real biome name interned beyond the default minecraft:plains"
        );
    }
}
