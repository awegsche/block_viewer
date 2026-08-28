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
//!
//! `column` and `scan` (ticket 093) are both built on [`get_area`] rather
//! than a fresh box walk: `column` is a single-column `get_area` call (`x1 ==
//! x2`, `z1 == z2`) whose per-Y results get collapsed into ranges; `scan`
//! runs `get_area` over the given box and filters its palette for the
//! requested block name.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use mc_anvil::BlockState;
use serde_json::{json, Value};

use crate::blueprint::{
    extract_blueprint, BlockState as AreaBlockState, ExtractProgress, MAX_BLOCKS,
};
use crate::edit::address_of;
use crate::region_cache::RegionCache;
use crate::selection::{SelectionBounds, WORLD_MAX_Y, WORLD_MIN_Y};
use crate::world::SECTION_SIZE;

use super::chunk::region_span;
use super::cli::{Cli, ColumnArgs, GetAreaArgs, GetArgs, ScanArgs};
use super::coords::BlockPos;
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

// -------------------------------------------------------------------------------------------------
// ---- column (ticket 093) -------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One Y in a column's read: the block that occupies it.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnEntry {
    pub y: i32,
    pub state: AreaBlockState,
}

/// A run of consecutive Y values holding the same block — what `text`/
/// `compact` print instead of one line per Y (see [`collapse_ranges`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnRange {
    pub from: i32,
    pub to: i32,
    pub state: AreaBlockState,
}

pub struct ColumnResult {
    pub save_name: String,
    pub x: i32,
    pub z: i32,
    /// Requested Y bounds, normalized `from <= to` — independent of
    /// `top_down`, which only reorders how [`Self::entries`]/[`Self::ranges`]
    /// are displayed.
    pub from: i32,
    pub to: i32,
    pub top_down: bool,
    /// Always ascending-Y order (`from` to `to`) regardless of `top_down` —
    /// the canonical order [`collapse_ranges`] and every renderer read
    /// through [`Self::display_entries`]/[`Self::display_ranges`] instead of
    /// re-deriving it.
    pub entries: Vec<ColumnEntry>,
    pub ranges: Vec<ColumnRange>,
}

/// Runs `column`: a single-column [`get_area`] (`x1 == x2`, `z1 == z2`),
/// defaulting `--from`/`--to` to [`WORLD_MIN_Y`]/[`WORLD_MAX_Y`] — the same
/// bounds [`SelectionBounds`] itself clamps Y to, reused rather than a
/// hardcoded `-64`/`320` (see the ticket).
pub fn column(cli: &Cli, args: &ColumnArgs) -> Result<ColumnResult, CliError> {
    let (x, z) = (args.pos.0.x, args.pos.0.y);
    let requested_from = args.from.unwrap_or(WORLD_MIN_Y);
    let requested_to = args.to.unwrap_or(WORLD_MAX_Y);
    let (y_lo, y_hi) = (requested_from.min(requested_to), requested_from.max(requested_to));

    let area = get_area(
        cli,
        &GetAreaArgs {
            from: BlockPos(IVec3::new(x, y_lo, z)),
            to: BlockPos(IVec3::new(x, y_hi, z)),
        },
    )?;

    // `size.x == size.z == 1`, so `get_area`'s Y-outer/Z-middle/X-inner
    // dense index for row `dy` (0-based from `area.origin.y`) is just `dy` —
    // no need to go through `SelectionBounds::index_of` for a column.
    let entries: Vec<ColumnEntry> = (0..area.size.y as usize)
        .map(|dy| ColumnEntry {
            y: area.origin.y + dy as i32,
            state: area.palette[area.blocks[dy] as usize].clone(),
        })
        .collect();
    let ranges = collapse_ranges(&entries);

    Ok(ColumnResult {
        save_name: area.save_name,
        x,
        z,
        from: y_lo,
        to: y_hi,
        top_down: args.top_down,
        entries,
        ranges,
    })
}

/// Merges consecutive entries with an identical [`AreaBlockState`] into
/// [`ColumnRange`]s — three distinct runs in, three ranges out, regardless
/// of how many Y values each run spans.
fn collapse_ranges(entries: &[ColumnEntry]) -> Vec<ColumnRange> {
    let mut ranges: Vec<ColumnRange> = Vec::new();
    for entry in entries {
        match ranges.last_mut() {
            Some(last) if last.state == entry.state => last.to = entry.y,
            _ => ranges.push(ColumnRange {
                from: entry.y,
                to: entry.y,
                state: entry.state.clone(),
            }),
        }
    }
    ranges
}

impl ColumnResult {
    /// [`Self::entries`] in display order: ascending Y, or descending when
    /// `--top-down` was given.
    fn display_entries(&self) -> Box<dyn Iterator<Item = &ColumnEntry> + '_> {
        if self.top_down {
            Box::new(self.entries.iter().rev())
        } else {
            Box::new(self.entries.iter())
        }
    }

    /// [`Self::ranges`] in the same display order as [`Self::display_entries`].
    /// Each range's own `from..=to` stays ascending either way — `top_down`
    /// only reorders *which range comes first*, not which end of a range is
    /// which.
    fn display_ranges(&self) -> Box<dyn Iterator<Item = &ColumnRange> + '_> {
        if self.top_down {
            Box::new(self.ranges.iter().rev())
        } else {
            Box::new(self.ranges.iter())
        }
    }

    fn summary_line(&self) -> String {
        format!(
            "column ({}, {}) in {}: Y {}..{} ({} blocks), {} ranges",
            self.x,
            self.z,
            self.save_name,
            self.from,
            self.to,
            self.entries.len(),
            self.ranges.len(),
        )
    }
}

fn properties_json(state: &AreaBlockState) -> Value {
    let mut properties = serde_json::Map::new();
    for (key, value) in &state.properties {
        properties.insert(key.clone(), json!(value));
    }
    Value::Object(properties)
}

impl Render for ColumnResult {
    /// A summary line plus one range line per run, in display order — never
    /// the naive one-line-per-Y listing (see the ticket: a 379-block stone
    /// run is one line, not 379).
    fn render_text(&self) -> String {
        let mut lines = vec![self.summary_line()];
        for range in self.display_ranges() {
            lines.push(format!("  {}..{} {}", range.from, range.to, range.state));
        }
        lines.join("\n")
    }

    /// `{"blocks": [...]}` the full per-Y array, plus `"ranges"` alongside it
    /// (not instead of it) for a caller wanting the collapsed form in
    /// machine form. Both follow `top_down`'s display order.
    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "x": self.x,
            "z": self.z,
            "from": self.from,
            "to": self.to,
            "top_down": self.top_down,
            "blocks": self.display_entries().map(|entry| json!({
                "y": entry.y,
                "name": entry.state.name,
                "properties": properties_json(&entry.state),
            })).collect::<Vec<_>>(),
            "ranges": self.display_ranges().map(|range| json!({
                "from": range.from,
                "to": range.to,
                "block": range.state.to_string(),
            })).collect::<Vec<_>>(),
        })
    }

    /// One line: the summary plus every range, semicolon-separated. This is
    /// the one command in the block-read group where `compact` isn't just a
    /// shorter summary of `text` — both carry the full range list, `text` as
    /// multiple lines and `compact` packed into one.
    fn render_compact(&self) -> String {
        let ranges: Vec<String> = self
            .display_ranges()
            .map(|range| format!("{}..{} {}", range.from, range.to, range.state))
            .collect();
        format!("{}: {}", self.summary_line(), ranges.join("; "))
    }
}

// -------------------------------------------------------------------------------------------------
// ---- scan (ticket 093) -----------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `scan`'s `--limit` default when none is given — "a few hundred", per the
/// ticket: enough for most finds, never unbounded (an unbounded scan for
/// `minecraft:air` over a loaded region would print millions of positions).
pub const DEFAULT_SCAN_LIMIT: usize = 500;

pub struct ScanResult {
    pub save_name: String,
    pub block: String,
    pub from: IVec3,
    pub to: IVec3,
    pub limit: usize,
    pub positions: Vec<IVec3>,
    /// Whether more matches existed than `limit` allowed reporting — set at
    /// the boundary (limit exactly met is `false`; exceeded by even one is
    /// `true`), never inferred from `positions.len() == limit` alone (a scan
    /// that happens to find exactly `limit` matches and no more must not
    /// look truncated).
    pub truncated: bool,
}

/// Runs `scan`: [`get_area`]'s extraction over `args.from`/`args.to`, filtered
/// by [`matching_positions`].
pub fn scan(cli: &Cli, args: &ScanArgs) -> Result<ScanResult, CliError> {
    let limit = args.limit.unwrap_or(DEFAULT_SCAN_LIMIT);
    let area = get_area(cli, &GetAreaArgs { from: args.from, to: args.to })?;
    let (positions, truncated) = matching_positions(&area, &args.block, limit);

    Ok(ScanResult {
        save_name: area.save_name,
        block: args.block.clone(),
        from: area.origin,
        to: area.origin + area.size - IVec3::ONE,
        limit,
        positions,
        truncated,
    })
}

/// Filters `area`'s blocks for positions whose palette entry's `name`
/// matches `block` exactly (properties are not part of the match — see the
/// ticket's `oak_door` example: `scan --block minecraft:oak_door` finds every
/// door regardless of open/closed/hinge). Positions are the `Blueprint`'s
/// dense-array order translated back to absolute world coordinates via
/// [`position_of`], capped at `limit` with `truncated` set exactly at the
/// boundary (limit met exactly is `false`; exceeded by even one is `true`).
///
/// Split out from [`scan`] so this — the actual match/cap logic — is
/// testable against a synthetic [`GetAreaResult`] rather than only end to end
/// against a real save. `pub(super)`: `replace` (ticket 096) reuses this
/// exact match rule rather than growing a second "match by name, ignore
/// properties" filter.
pub(super) fn matching_positions(area: &GetAreaResult, block: &str, limit: usize) -> (Vec<IVec3>, bool) {
    let matching: HashSet<u16> = area
        .palette
        .iter()
        .enumerate()
        .filter(|(_, state)| state.name == block)
        .map(|(index, _)| index as u16)
        .collect();

    if matching.is_empty() {
        return (Vec::new(), false);
    }

    let mut positions = Vec::new();
    let mut truncated = false;
    for (index, &palette_index) in area.blocks.iter().enumerate() {
        if !matching.contains(&palette_index) {
            continue;
        }
        if positions.len() >= limit {
            truncated = true;
            break;
        }
        positions.push(position_of(area.origin, area.size, index));
    }

    (positions, truncated)
}

/// The inverse of [`SelectionBounds::index_of`]: the absolute world position
/// at dense-array `index` within a box of `size` blocks starting at `origin`,
/// in the same Y-outer/Z-middle/X-inner order [`SelectionBounds::iter_blocks`]
/// documents.
fn position_of(origin: IVec3, size: IVec3, index: usize) -> IVec3 {
    let (width, depth) = (size.x as usize, size.z as usize);
    let plane = width * depth;
    let y = index / plane;
    let remainder = index % plane;
    let z = remainder / width;
    let x = remainder % width;
    origin + IVec3::new(x as i32, y as i32, z as i32)
}

impl ScanResult {
    fn summary_line(&self) -> String {
        let count = self.positions.len();
        let suffix = if self.truncated {
            format!(" (truncated at limit {})", self.limit)
        } else {
            String::new()
        };
        format!(
            "scan {} in {}: {} to {}, {count} match{}{suffix}",
            self.block,
            self.save_name,
            self.from,
            self.to,
            if count == 1 { "" } else { "es" },
        )
    }
}

impl Render for ScanResult {
    /// The summary line plus one position per line — the full listing,
    /// capped at `limit` the way [`ScanResult::positions`] already is.
    fn render_text(&self) -> String {
        let mut lines = vec![self.summary_line()];
        for pos in &self.positions {
            lines.push(format!("  {pos}"));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "block": self.block,
            "from": [self.from.x, self.from.y, self.from.z],
            "to": [self.to.x, self.to.y, self.to.z],
            "limit": self.limit,
            "truncated": self.truncated,
            "positions": self.positions.iter()
                .map(|p| json!([p.x, p.y, p.z]))
                .collect::<Vec<_>>(),
        })
    }

    /// The summary line alone, no position listing — same "how many /
    /// truncated or not" role `get-area`'s compact plays for its palette.
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

    // -----------------------------------------------------------------------------------------
    // ---- column (ticket 093) ------------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    use super::super::cli::{ColumnArgs, ScanArgs};
    use super::super::coords::ColumnPos;
    use bevy::math::IVec2;

    fn state(name: &str) -> AreaBlockState {
        AreaBlockState { name: name.to_string(), properties: vec![] }
    }

    fn entries(pairs: &[(i32, &str)]) -> Vec<ColumnEntry> {
        pairs.iter().map(|&(y, name)| ColumnEntry { y, state: state(name) }).collect()
    }

    /// The ticket's own "Done when": a synthetic column with three distinct
    /// runs collapses to exactly three ranges, each spanning its whole run.
    #[test]
    fn collapse_ranges_merges_three_runs_into_three_ranges() {
        let column = entries(&[
            (-64, "minecraft:bedrock"),
            (-63, "minecraft:stone"),
            (-62, "minecraft:stone"),
            (-61, "minecraft:stone"),
            (-60, "minecraft:air"),
            (-59, "minecraft:air"),
        ]);

        let ranges = collapse_ranges(&column);

        assert_eq!(
            ranges,
            vec![
                ColumnRange { from: -64, to: -64, state: state("minecraft:bedrock") },
                ColumnRange { from: -63, to: -61, state: state("minecraft:stone") },
                ColumnRange { from: -60, to: -59, state: state("minecraft:air") },
            ]
        );
    }

    /// The same three runs, read through `render_text`: exactly three range
    /// lines, not one line per Y.
    #[test]
    fn render_text_prints_exactly_one_line_per_range() {
        let column = entries(&[
            (0, "minecraft:stone"),
            (1, "minecraft:stone"),
            (2, "minecraft:dirt"),
            (3, "minecraft:air"),
        ]);
        let result = ColumnResult {
            save_name: "world".to_string(),
            x: 5,
            z: 9,
            from: 0,
            to: 3,
            top_down: false,
            ranges: collapse_ranges(&column),
            entries: column,
        };

        let text = result.render_text();
        let range_lines: Vec<&str> = text.lines().skip(1).collect();
        assert_eq!(range_lines.len(), 3, "{text}");
        assert_eq!(range_lines[0], "  0..1 minecraft:stone");
        assert_eq!(range_lines[1], "  2..2 minecraft:dirt");
        assert_eq!(range_lines[2], "  3..3 minecraft:air");
    }

    /// `--top-down` reverses which range is listed first, but never which
    /// end of a range is `from` vs `to` — a range's own extent doesn't
    /// depend on the direction it's read in.
    #[test]
    fn top_down_reverses_range_order_not_range_direction() {
        let column = entries(&[(0, "minecraft:stone"), (1, "minecraft:air")]);
        let ranges = collapse_ranges(&column);
        let result = ColumnResult {
            save_name: "world".to_string(),
            x: 0,
            z: 0,
            from: 0,
            to: 1,
            top_down: true,
            ranges,
            entries: column,
        };

        let displayed: Vec<&ColumnRange> = result.display_ranges().collect();
        assert_eq!(displayed[0].state.name, "minecraft:air");
        assert_eq!(displayed[1].state.name, "minecraft:stone");
        // Each range's own from/to is unaffected by display order.
        assert!(displayed.iter().all(|r| r.from <= r.to));
    }

    /// `render_json`'s documented shape: the full per-Y `blocks` array
    /// alongside (not instead of) the collapsed `ranges` array.
    #[test]
    fn column_render_json_carries_both_blocks_and_ranges() {
        let column = entries(&[(10, "minecraft:stone"), (11, "minecraft:stone")]);
        let result = ColumnResult {
            save_name: "world".to_string(),
            x: 1,
            z: 2,
            from: 10,
            to: 11,
            top_down: false,
            ranges: collapse_ranges(&column),
            entries: column,
        };

        let json = result.render_json();
        assert_eq!(json["blocks"].as_array().unwrap().len(), 2);
        assert_eq!(
            json["ranges"],
            serde_json::json!([{"from": 10, "to": 11, "block": "minecraft:stone"}])
        );
    }

    /// `column`'s CLI args parse into the same `x`/`z` a hand-built
    /// `ColumnArgs` would — cheap smoke test that `ColumnPos`'s field order
    /// (`x`, then `z`) lines up with how `column()` reads it.
    #[test]
    fn column_args_pos_reads_as_x_then_z() {
        let args = ColumnArgs {
            pos: ColumnPos(IVec2::new(7, -3)),
            from: None,
            to: None,
            top_down: false,
        };
        assert_eq!((args.pos.0.x, args.pos.0.y), (7, -3));
    }

    // -----------------------------------------------------------------------------------------
    // ---- scan (ticket 093) --------------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    fn area_with_blocks(palette: Vec<&str>, blocks: Vec<u16>, size: IVec3) -> GetAreaResult {
        GetAreaResult {
            save_name: "world".to_string(),
            origin: IVec3::ZERO,
            size,
            palette: palette.into_iter().map(state).collect(),
            blocks,
            failed_columns: 0,
        }
    }

    #[test]
    fn matching_positions_finds_every_block_by_name_ignoring_properties() {
        let mut chest_with_props = state("minecraft:chest");
        chest_with_props.properties = vec![("facing".to_string(), "north".to_string())];
        let area = GetAreaResult {
            save_name: "world".to_string(),
            origin: IVec3::new(100, 60, 100),
            size: IVec3::new(2, 1, 2),
            palette: vec![state("minecraft:air"), chest_with_props],
            blocks: vec![0, 1, 1, 0],
            failed_columns: 0,
        };

        let (positions, truncated) = matching_positions(&area, "minecraft:chest", 10);

        assert!(!truncated);
        assert_eq!(
            positions,
            vec![IVec3::new(101, 60, 100), IVec3::new(100, 60, 101)]
        );
    }

    #[test]
    fn matching_positions_is_empty_when_the_block_never_appears() {
        let area = area_with_blocks(vec!["minecraft:air", "minecraft:stone"], vec![0, 1, 0], IVec3::new(3, 1, 1));
        let (positions, truncated) = matching_positions(&area, "minecraft:diamond_block", 10);
        assert!(positions.is_empty());
        assert!(!truncated);
    }

    /// The ticket's own "Done when": limit exactly met is not truncated,
    /// exceeded by one is.
    #[test]
    fn matching_positions_sets_truncated_exactly_at_the_boundary() {
        // Four matching blocks in a row.
        let area = area_with_blocks(
            vec!["minecraft:air", "minecraft:stone"],
            vec![1, 1, 1, 1],
            IVec3::new(4, 1, 1),
        );

        let (met, met_truncated) = matching_positions(&area, "minecraft:stone", 4);
        assert_eq!(met.len(), 4);
        assert!(!met_truncated, "limit exactly met must not be truncated");

        let (exceeded, exceeded_truncated) = matching_positions(&area, "minecraft:stone", 3);
        assert_eq!(exceeded.len(), 3);
        assert!(exceeded_truncated, "limit exceeded by one must be truncated");
    }

    #[test]
    fn scan_render_json_matches_the_documented_shape() {
        let result = ScanResult {
            save_name: "world".to_string(),
            block: "minecraft:chest".to_string(),
            from: IVec3::new(0, 60, 0),
            to: IVec3::new(1, 60, 1),
            limit: 500,
            positions: vec![IVec3::new(0, 60, 0)],
            truncated: false,
        };

        assert_eq!(
            result.render_json(),
            serde_json::json!({
                "save": "world",
                "block": "minecraft:chest",
                "from": [0, 60, 0],
                "to": [1, 60, 1],
                "limit": 500,
                "truncated": false,
                "positions": [[0, 60, 0]],
            })
        );
    }

    /// `position_of` is the inverse of `SelectionBounds::index_of` — cross
    /// checked against it directly rather than trusted on its own arithmetic.
    #[test]
    fn position_of_is_the_inverse_of_selection_bounds_index_of() {
        let origin = IVec3::new(-4, 60, 8);
        let bounds = SelectionBounds::from_corners(origin, origin, origin + IVec3::new(2, 3, 4) - IVec3::ONE);
        let size = bounds.size();
        for block in bounds.iter_blocks() {
            let index = bounds.index_of(block).unwrap();
            assert_eq!(position_of(origin, size, index), block, "index {index}");
        }
    }

    /// End to end against a real save, `scan`'s own "Done when": every
    /// reported position is inside the box, and `get` at that position really
    /// is the block scanned for.
    #[test]
    fn scan_matches_agree_with_get_on_a_real_save() {
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
        let far_corner = origin + IVec3::new(7, 7, 7);

        let cli = dummy_cli(Some(meta.path.to_string_lossy().to_string()));
        // Whatever sits at the box's own corner is guaranteed to be found —
        // no fixture with a known planted block is available yet (095 hasn't
        // landed), so scan for that corner's own block name instead.
        let corner_block = get(&cli, &GetArgs { pos: BlockPos(origin) })
            .expect("get should resolve")
            .state
            .name()
            .to_string();

        let result = scan(
            &cli,
            &ScanArgs {
                from: BlockPos(origin),
                to: BlockPos(far_corner),
                block: corner_block.clone(),
                limit: Some(500),
            },
        )
        .expect("scan should resolve");

        assert!(
            result.positions.contains(&origin),
            "the box's own corner should be among the matches for its own block"
        );
        for pos in &result.positions {
            let single = get(&cli, &GetArgs { pos: BlockPos(*pos) }).expect("get should resolve");
            assert_eq!(single.state.name(), corner_block, "at {pos}");
        }
    }
}
