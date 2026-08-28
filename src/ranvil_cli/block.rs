//! `ranvil-cli get` (ticket 091) — the simplest block-level read: resolve a
//! block position's region/chunk through [`RegionCache`] and
//! [`mc_anvil::chunkregion::ChunkRegion::get_block`], then parse the palette
//! entry it hands back into a [`BlockState`] with
//! [`BlockState::from_palette_entry`] — the same parse `chunkregion`'s own
//! internals and the viewer's block inspector (ticket 007) already share, not
//! a second NBT-to-`BlockState` decode grown here.
//!
//! `get-area` (ticket 092) is the same read over a box: it builds a
//! [`SelectionBounds`] directly (no `App`, no plugin — see the ticket and
//! `crate::selection`'s module docs on why that's safe) and hands it to
//! [`extract_blueprint`], the same walk the viewer's structure export uses.
//! Reserved for `column`/`scan` (ticket 093), which reuse [`get`]'s
//! coordinate-resolution shape for a column rather than a point or a box.

use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use mc_anvil::BlockState;
use serde_json::{json, Value};

use crate::blueprint::{
    extract_blueprint, BlockState as AreaBlockState, ExtractProgress, MAX_BLOCKS,
};
use crate::edit::address_of;
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;
use crate::world::SECTION_SIZE;

use super::chunk::region_span;
use super::cli::{Cli, GetAreaArgs, GetArgs};
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

/// One box of blocks, extracted: a palette of distinct block states plus a
/// dense index array — the same shape [`crate::blueprint::Blueprint`] already
/// is, per the ticket's "documented rather than reinvented" call.
#[derive(Debug)]
pub struct GetAreaResult {
    pub save_name: String,
    /// The selection's min corner in Minecraft coordinates — `Blueprint`'s
    /// own `origin` field, carried through unrenamed.
    pub origin: IVec3,
    pub size: IVec3,
    /// Distinct block states, in first-seen order; index 0 is always
    /// `minecraft:air` (see [`crate::blueprint::extract`]'s `Accumulator::new`).
    pub palette: Vec<AreaBlockState>,
    /// One palette index per block, in [`SelectionBounds::iter_blocks`]'s
    /// Y-outer / Z-middle / X-inner order.
    pub blocks: Vec<u16>,
    pub failed_columns: usize,
}

/// Runs `get-area`: builds a [`SelectionBounds`] from `args.from`/`args.to`
/// (either corner may be given in either order — `from_corners` normalizes),
/// then walks it through [`extract_blueprint`] the same way the viewer's
/// structure export does.
///
/// The [`MAX_BLOCKS`] check happens before [`resolve_save`] even runs — a
/// selection over the cap is a bad request regardless of which save it names,
/// so there's nothing to gain by resolving one first, and the ticket's "exits
/// 2 before touching the region cache" is satisfied by construction rather
/// than by ordering two side effects carefully.
pub fn get_area(cli: &Cli, args: &GetAreaArgs) -> Result<GetAreaResult, CliError> {
    let bounds = SelectionBounds::from_corners(args.from.0, args.from.0, args.to.0);

    let volume = bounds.volume();
    if volume > MAX_BLOCKS {
        return Err(CliError::Usage(format!(
            "selection ({}) to ({}) is {volume} blocks — over the {MAX_BLOCKS}-block get-area limit",
            args.from.0, args.to.0
        )));
    }

    let meta = resolve_save(cli)?;

    // Sized the same way `chunks` sizes its own cache: exactly the regions
    // this box's chunk columns span, so `extract_blueprint`'s region-major
    // walk never evicts and re-loads a region file mid-selection.
    let size = SECTION_SIZE as i32;
    let (min_cx, min_cz) = (bounds.min.x.div_euclid(size), bounds.min.z.div_euclid(size));
    let (max_cx, max_cz) = (bounds.max.x.div_euclid(size), bounds.max.z.div_euclid(size));
    let capacity = region_span(min_cx, max_cx, min_cz, max_cz);

    let cache = Arc::new(Mutex::new(RegionCache::new(meta.clone(), capacity)));
    let progress = ExtractProgress::default();
    let blueprint = extract_blueprint(bounds, &cache, &progress).map_err(|e| {
        CliError::Data(format!(
            "could not extract ({}) to ({}): {e}",
            args.from.0, args.to.0
        ))
    })?;

    Ok(GetAreaResult {
        save_name: meta.name,
        origin: blueprint.origin,
        size: blueprint.size,
        palette: blueprint.palette,
        blocks: blueprint.blocks,
        failed_columns: blueprint.failed_columns,
    })
}

impl GetAreaResult {
    /// The min corner's opposite: `origin + size - 1`, inclusive — recomputed
    /// rather than stored, since [`SelectionBounds`] already guarantees
    /// `size >= 1` on every axis.
    fn max_corner(&self) -> IVec3 {
        self.origin + self.size - IVec3::ONE
    }

    /// `size X x Y x Z (N blocks), P distinct states, F failed columns` — the
    /// one line every format shares: `text`'s first line, `compact`'s whole
    /// output. `failed_columns` is in this line unconditionally (not just in
    /// `json`) per the ticket: a caller reading `compact` alone still needs to
    /// know its palette might read "mostly air" for the wrong reason.
    fn summary_line(&self) -> String {
        let max = self.max_corner();
        format!(
            "get-area {} to {} in {}: size {}x{}x{} ({} blocks), {} distinct states, {} failed columns",
            self.origin,
            max,
            self.save_name,
            self.size.x,
            self.size.y,
            self.size.z,
            self.blocks.len(),
            self.palette.len(),
            self.failed_columns,
        )
    }
}

impl Render for GetAreaResult {
    /// The summary line, plus the palette listing — the full per-block index
    /// array is `--format json`'s job, per the ticket ("the full index array
    /// is not dumped to a terminal").
    fn render_text(&self) -> String {
        let mut lines = vec![self.summary_line()];
        for (index, state) in self.palette.iter().enumerate() {
            lines.push(format!("  {index}: {state}"));
        }
        lines.join("\n")
    }

    /// `{"origin": [...], "size": [...], "palette": [...], "blocks": [...],
    /// "failed_columns": N}` — the shape the ticket commits to, `blocks`
    /// indexing into `palette` in `Blueprint`'s own dense-array order.
    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "origin": [self.origin.x, self.origin.y, self.origin.z],
            "size": [self.size.x, self.size.y, self.size.z],
            "palette": self.palette.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "blocks": self.blocks,
            "failed_columns": self.failed_columns,
        })
    }

    /// The summary line alone, no palette listing — what an agent asks for
    /// when it only needs "how big / how many distinct blocks" before
    /// deciding whether to pull the full `--format json`.
    fn render_compact(&self) -> String {
        self.summary_line()
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

    // -----------------------------------------------------------------------------------------
    // ---- get-area (ticket 092) --------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    use super::super::cli::{Command, SavesArgs};
    use super::super::coords::BlockPos;
    use super::super::format::OutputFormat;

    fn area_result() -> GetAreaResult {
        GetAreaResult {
            save_name: "world".to_string(),
            origin: IVec3::new(0, 60, 0),
            size: IVec3::new(2, 1, 1),
            palette: vec![
                AreaBlockState::air(),
                AreaBlockState { name: "minecraft:stone".to_string(), properties: vec![] },
            ],
            blocks: vec![0, 1],
            failed_columns: 0,
        }
    }

    #[test]
    fn get_area_render_json_matches_the_documented_shape() {
        assert_eq!(
            area_result().render_json(),
            json!({
                "save": "world",
                "origin": [0, 60, 0],
                "size": [2, 1, 1],
                "palette": ["minecraft:air", "minecraft:stone"],
                "blocks": [0, 1],
                "failed_columns": 0,
            })
        );
    }

    #[test]
    fn get_area_render_text_is_the_summary_line_plus_the_palette_listing() {
        assert_eq!(
            area_result().render_text(),
            "get-area [0, 60, 0] to [1, 60, 0] in world: size 2x1x1 (2 blocks), 2 distinct states, 0 failed columns\n\
             \x20\x200: minecraft:air\n\
             \x20\x201: minecraft:stone"
        );
    }

    #[test]
    fn get_area_render_compact_is_the_summary_line_alone_no_palette() {
        let compact = area_result().render_compact();
        assert_eq!(
            compact,
            "get-area [0, 60, 0] to [1, 60, 0] in world: size 2x1x1 (2 blocks), 2 distinct states, 0 failed columns"
        );
        assert!(!compact.contains("stone"), "compact must not list the palette");
    }

    /// Every format surfaces `failed_columns`, not just `json` — a caller
    /// reading `compact` alone still needs to know its palette might read
    /// "mostly air" for the wrong reason (the ticket's own wording).
    #[test]
    fn failed_columns_shows_up_in_every_format() {
        let mut result = area_result();
        result.failed_columns = 3;

        assert!(result.render_text().contains("3 failed columns"));
        assert!(result.render_compact().contains("3 failed columns"));
        assert_eq!(result.render_json()["failed_columns"], json!(3));
    }

    fn dummy_cli(save: Option<String>) -> Cli {
        Cli {
            save,
            instance: Some(PathBuf::from("does-not-exist")),
            format: OutputFormat::Json,
            command: Command::Saves(SavesArgs {}),
        }
    }

    /// The ticket's own "Done when": over `MAX_BLOCKS` is a `Usage` error
    /// (exit 2), and it fires before `resolve_save` runs at all — a `--save`/
    /// `--instance` pointing nowhere would otherwise surface as a *different*
    /// `Usage` error ("could not read instance directory ..."), so getting
    /// this one specifically proves the volume check ran first.
    #[test]
    fn a_selection_over_max_blocks_is_a_usage_error_before_touching_a_save() {
        let args = GetAreaArgs {
            from: BlockPos(IVec3::new(0, crate::selection::WORLD_MIN_Y, 0)),
            to: BlockPos(IVec3::new(9999, crate::selection::WORLD_MAX_Y, 9999)),
        };
        let volume = SelectionBounds::from_corners(args.from.0, args.from.0, args.to.0).volume();
        assert!(volume > MAX_BLOCKS, "fixture must actually exceed the cap");

        let err = get_area(&dummy_cli(None), &args).unwrap_err();
        match err {
            CliError::Usage(message) => {
                assert!(message.contains(&MAX_BLOCKS.to_string()), "{message}");
                assert!(
                    !message.contains("instance directory"),
                    "should fail on the volume check, not on resolving a save: {message}"
                );
            }
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    /// End to end against the real save, same convention as
    /// `extract.rs`'s own real-save tests: `get-area` over a small box should
    /// report, at each corner, exactly what `get` reports for that corner on
    /// its own — the ticket's headline "Done when".
    #[test]
    fn get_area_matches_get_at_each_corner_on_a_real_save() {
        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        let region_width =
            mc_anvil::region::REGION_WIDTH_IN_CHUNKS as i32 * crate::world::SECTION_SIZE as i32;
        let origin = IVec3::new(
            rx * region_width + region_width / 2,
            60,
            rz * region_width + region_width / 2,
        );
        let far_corner = origin + IVec3::new(3, 3, 3);

        let cli = dummy_cli(Some(meta.path.to_string_lossy().to_string()));
        let area = get_area(
            &cli,
            &GetAreaArgs { from: BlockPos(origin), to: BlockPos(far_corner) },
        )
        .expect("a small selection should extract");

        let bounds = SelectionBounds::from_corners(origin, origin, far_corner);
        for corner in [origin, far_corner] {
            let single = get(&cli, &GetArgs { pos: BlockPos(corner) }).expect("get should resolve");
            let index = bounds.index_of(corner).expect("corner is inside the selection");
            let area_state = &area.palette[area.blocks[index] as usize];
            assert_eq!(
                area_state.to_string(),
                single.state.to_string(),
                "at {corner}"
            );
        }
    }
}
