//! `ranvil-cli chunk`/`chunks` (ticket 089) — the first commands that decode
//! a chunk's own NBT rather than just a region's metadata (ticket 088's
//! `regions`) or a whole save's directory listing.
//!
//! Both read through [`mc_anvil::chunkregion::ChunkRegion::get_chunk`]
//! against a one-off [`RegionCache`], built the same way `citybuilder`'s
//! startup builds one, just without the streaming/unload machinery around
//! it: a capacity sized to whatever the command touches, dropped at the end
//! of the process.
//!
//! Neither command decodes `block_states`: every field either lives at the
//! chunk root (`Status`, `DataVersion`, `InhabitedTime`, `block_entities`),
//! comes from a section's `Y` tag or its `biomes.palette` (no index
//! unpacking needed — see [`distinct_biomes`]), or comes straight out of the
//! chunk's `Heightmaps` compound via [`mc_anvil::heightmap::read_heightmap`].
//! That is what keeps `chunks`' bulk survey cheap over a few hundred chunks.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use mc_anvil::heightmap::{read_heightmap, ColumnHeights, HeightmapKind};
use mc_anvil::region::{Region, REGION_WIDTH_IN_CHUNKS};
use mc_anvil::{MCLoadError, SaveMeta};
use rnbt::NbtField;
use serde_json::{json, Value};

use crate::chunk_pipeline::local_chunk_index;
use crate::region_cache::{chunk_to_region_coord, RegionCache};

use super::cli::{ChunkArgs, ChunksArgs, Cli};
use super::error::CliError;
use super::format::Render;
use super::save::resolve_save;

/// Hard ceiling on `chunks`' rectangle, in chunk count — a backstop against
/// an accidental (or malicious) bounds typo asking for a range spanning
/// billions of chunks, the same role [`crate::blueprint::extract::MAX_BLOCKS`]
/// plays for `struct export`/the viewer's selection.
const MAX_CHUNKS_IN_RANGE: u64 = 1_000_000;

/// `text`/`compact`'s cap on how many ungenerated chunk coordinates `chunks`
/// lists inline — `--format json` always includes the full list.
const UNGENERATED_DISPLAY_CAP: usize = 50;

// -------------------------------------------------------------------------------------------------
// ---- chunk ----------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One of `WORLD_SURFACE`'s min/max/average across a chunk's 256 columns —
/// "is this chunk flat" without a full `heightmap` (ticket 090) call.
/// Values are the heightmap's own convention (one *above* the highest
/// qualifying block; see [`mc_anvil::heightmap`]'s module docs), not the
/// block's own Y.
pub struct HeightmapPeek {
    pub min: i32,
    pub max: i32,
    pub avg: f64,
}

pub struct ChunkResult {
    pub save_name: String,
    pub cx: i32,
    pub cz: i32,
    pub status: Option<String>,
    pub data_version: Option<i32>,
    /// Min/max `Y` among sections that carry `block_states` — not assumed
    /// `-4..19`, and not counting the light-only sentinel section vanilla
    /// writes below the world bottom (see
    /// [`mc_anvil::chunkregion::ChunkRegion::mark_for_relight`]'s docs),
    /// which has no `block_states` at all.
    pub section_y_range: Option<(i32, i32)>,
    pub inhabited_time: Option<i64>,
    pub block_entity_count: usize,
    /// From the chunk's own NBT if it carries an entity list (pre-1.17
    /// saves), else from the save's separate `entities/` region tree
    /// (1.17+) if the chunk has one there — see [`EntitiesRegions`].
    pub entity_count: usize,
    /// Distinct biome names across the chunk's `biomes` palettes, sorted.
    pub biomes: Vec<String>,
    /// `None` when the chunk has no `Heightmaps.WORLD_SURFACE` at all — a
    /// normal state for a chunk this crate (or an older save format) never
    /// wrote one for, not a failure.
    pub heightmap: Option<HeightmapPeek>,
}

/// Runs `chunk`: decodes one chunk and reports the fields ticket 089 scopes.
///
/// A chunk the save doesn't have at all — ungenerated terrain, or a
/// coordinate outside every region the save has — is a real answer about
/// the save, not a bad request: [`CliError::Data`], exit 1, per the
/// roadmap's exit-code split.
pub fn chunk(cli: &Cli, args: &ChunkArgs) -> Result<ChunkResult, CliError> {
    let meta = resolve_save(cli)?;
    let (cx, cz) = (args.pos.0.x, args.pos.0.y);

    let mut cache = RegionCache::new(meta.clone(), 1);
    let nbt = load_chunk_nbt(&mut cache, (cx, cz)).map_err(|e| {
        CliError::Data(format!(
            "could not read the region for chunk ({cx}, {cz}): {e}"
        ))
    })?;
    let Some(nbt) = nbt else {
        return Err(CliError::Data(format!(
            "chunk ({cx}, {cz}) is not generated"
        )));
    };

    let mut entities = EntitiesRegions::new(&meta);
    Ok(build_chunk_result(meta.name, cx, cz, &nbt, &mut entities))
}

fn build_chunk_result(
    save_name: String,
    cx: i32,
    cz: i32,
    nbt: &NbtField,
    entities: &mut EntitiesRegions,
) -> ChunkResult {
    ChunkResult {
        save_name,
        cx,
        cz,
        status: nbt.get_string("Status").cloned(),
        data_version: nbt.get_int("DataVersion"),
        section_y_range: section_y_range(nbt),
        inhabited_time: nbt.get_long("InhabitedTime"),
        block_entity_count: list_len(nbt, "block_entities"),
        entity_count: chunk_entity_count(nbt, entities, cx, cz),
        biomes: distinct_biomes(nbt),
        heightmap: read_heightmap(nbt, HeightmapKind::WorldSurface)
            .ok()
            .map(|h| heightmap_peek(&h)),
    }
}

/// Loads the chunk at `(cx, cz)` through `cache`, or `None` if the save has
/// no region there (ungenerated terrain, per [`MCLoadError::PathNotFoundError`])
/// or the region has no NBT in this chunk's slot (an ungenerated chunk
/// inside a region the save does have).
///
/// `pub(crate)`: `heightmap` (ticket 090) reuses this rather than re-deriving
/// its own chunk-load path — the "no new decode logic" rule the roadmap sets
/// for read-only commands applies to *this* module's own code too, not just
/// to `ranvil`'s.
pub(crate) fn load_chunk_nbt(cache: &mut RegionCache, (cx, cz): (i32, i32)) -> Result<Option<NbtField>, MCLoadError> {
    let region_coord = chunk_to_region_coord((cx, cz));
    match cache.get_or_load(region_coord) {
        Ok(region) => {
            let (local_x, local_z) = local_chunk_index((cx, cz), region_coord);
            Ok(region.get_chunk(local_x, local_z).cloned())
        }
        Err(MCLoadError::PathNotFoundError) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Min/max `Y` among `sections` entries that carry `block_states` — see
/// [`ChunkResult::section_y_range`].
fn section_y_range(nbt: &NbtField) -> Option<(i32, i32)> {
    let sections = nbt.get_list("sections")?.as_compound_list()?;
    let ys: Vec<i32> = sections
        .iter()
        .filter(|section| section.get("block_states").is_some())
        .filter_map(|section| section.get_byte("Y").map(|y| y as i8 as i32))
        .collect();

    if ys.is_empty() {
        return None;
    }
    Some((
        *ys.iter().min().expect("checked non-empty above"),
        *ys.iter().max().expect("checked non-empty above"),
    ))
}

/// The length of a chunk-root compound list, treating "absent" and
/// vanilla's untyped-empty-list encoding (`NbtList::End`, see
/// [`mc_anvil::chunkregion`]'s `remove_orphans` docs) the same as zero.
fn list_len(nbt: &NbtField, key: &str) -> usize {
    nbt.get_list(key)
        .and_then(|list| list.as_compound_list())
        .map(Vec::len)
        .unwrap_or(0)
}

/// Distinct biome names across every section's `biomes.palette` — a
/// per-chunk summary, not a per-block breakdown (that stays in `get`/
/// `get-area`'s scope), so this only ever reads a palette's string list and
/// never unpacks the packed per-block indices next to it.
fn distinct_biomes(nbt: &NbtField) -> Vec<String> {
    let mut names = std::collections::BTreeSet::new();
    if let Some(sections) = nbt.get_list("sections").and_then(|list| list.as_compound_list()) {
        for section in sections {
            if let Some(palette) = section
                .get("biomes")
                .and_then(|biomes| biomes.get_list("palette"))
                .and_then(|list| list.as_string_list())
            {
                names.extend(palette.iter().cloned());
            }
        }
    }
    names.into_iter().collect()
}

fn heightmap_peek(heights: &ColumnHeights) -> HeightmapPeek {
    let min = *heights.iter().min().expect("heightmap always has 256 columns");
    let max = *heights.iter().max().expect("heightmap always has 256 columns");
    let avg = heights.iter().map(|&h| h as f64).sum::<f64>() / heights.len() as f64;
    HeightmapPeek { min, max, avg }
}

/// The entity count for a chunk: its own NBT if it carries an entity list
/// (pre-1.17 saves keep them with the blocks), else the save's separate
/// `entities/` region tree (1.17+) if it has one for this chunk.
fn chunk_entity_count(nbt: &NbtField, entities: &mut EntitiesRegions, cx: i32, cz: i32) -> usize {
    // Pre-flattening saves capitalize the tag; nothing else in this crate
    // supports that layout (see `mc_anvil::chunkregion`'s note on `Level`),
    // but checking costs nothing and a stray "entities" is worth the same
    // read either way.
    for key in ["Entities", "entities"] {
        if let Some(list) = nbt.get_list(key).and_then(|list| list.as_compound_list()) {
            return list.len();
        }
    }
    entities.count(cx, cz)
}

/// Lazily-loaded byte cache over a save's separate `entities/` region tree
/// (split out of the terrain regions in 1.17), so `chunks`' bulk survey
/// reads each entities region file's bytes once rather than once per chunk
/// slot inside it.
///
/// **Assumed, not verified against a real save**: the overworld's entity
/// region files sit at `<save>/entities/`, parallel to `<save>/region/` —
/// *not* moved under `dimensions/` the way [`mc_anvil::save::resolve_region_dir`]
/// prefers for terrain regions on some save layouts. A missing directory,
/// missing region file, or unreadable region/chunk all fall back to "no
/// count available" (0) rather than an error — this is a bonus data point,
/// not something either command's success hinges on.
struct EntitiesRegions {
    dir: Option<PathBuf>,
    loaded: HashMap<(i32, i32), Option<Vec<u8>>>,
}

impl EntitiesRegions {
    fn new(meta: &SaveMeta) -> Self {
        let dir = meta.path.join("entities");
        Self {
            dir: dir.is_dir().then_some(dir),
            loaded: HashMap::new(),
        }
    }

    fn count(&mut self, cx: i32, cz: i32) -> usize {
        let Some(dir) = &self.dir else { return 0 };
        let region_coord = chunk_to_region_coord((cx, cz));
        let path = dir.join(format!("r.{}.{}.mca", region_coord.0, region_coord.1));

        let data = self
            .loaded
            .entry(region_coord)
            .or_insert_with(|| std::fs::read(&path).ok());
        let Some(data) = data else { return 0 };

        let region = Region::new(region_coord.0, region_coord.1, &path);
        let (local_x, local_z) = local_chunk_index((cx, cz), region_coord);
        let index = local_x + local_z * REGION_WIDTH_IN_CHUNKS;

        let Ok(Some(bytes)) = region.get_chunk_nbt_data(data, index) else {
            return 0;
        };
        let Ok(nbt) = rnbt::from_bytes(&bytes) else {
            return 0;
        };
        nbt.get_list("Entities")
            .and_then(|list| list.as_compound_list())
            .map(Vec::len)
            .unwrap_or(0)
    }
}

impl Render for ChunkResult {
    fn render_text(&self) -> String {
        vec![
            format!("chunk ({}, {}) in {}", self.cx, self.cz, self.save_name),
            format!("  status: {}", self.status.as_deref().unwrap_or("unknown")),
            format!(
                "  data version: {}",
                self.data_version.map(|v| v.to_string()).unwrap_or_else(|| "unknown".to_string())
            ),
            format!(
                "  sections: {}",
                self.section_y_range
                    .map(|(lo, hi)| format!("Y {lo}..{hi}"))
                    .unwrap_or_else(|| "none".to_string())
            ),
            format!(
                "  inhabited time: {}",
                self.inhabited_time.map(|t| t.to_string()).unwrap_or_else(|| "unknown".to_string())
            ),
            format!("  block entities: {}", self.block_entity_count),
            format!("  entities: {}", self.entity_count),
            format!(
                "  biomes: {}",
                if self.biomes.is_empty() { "none".to_string() } else { self.biomes.join(", ") }
            ),
            format!(
                "  world surface height: {}",
                self.heightmap
                    .as_ref()
                    .map(|h| format!("min {} max {} avg {:.1}", h.min, h.max, h.avg))
                    .unwrap_or_else(|| "unavailable".to_string())
            ),
        ]
        .join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "x": self.cx,
            "z": self.cz,
            "status": self.status,
            "data_version": self.data_version,
            "section_y_min": self.section_y_range.map(|(lo, _)| lo),
            "section_y_max": self.section_y_range.map(|(_, hi)| hi),
            "inhabited_time": self.inhabited_time,
            "block_entities": self.block_entity_count,
            "entities": self.entity_count,
            "biomes": self.biomes,
            "heightmap": self.heightmap.as_ref().map(|h| json!({
                "min": h.min,
                "max": h.max,
                "avg": h.avg,
            })),
        })
    }

    fn render_compact(&self) -> String {
        // The roadmap's own example: `0,0 full sections=-4..19 entities=3
        // blockents=1 inhabited=118402`, plus the fields this ticket added
        // beyond that sketch (data version, biome count, surface peek) —
        // a biome *count* rather than the full name list, since `compact`
        // optimises for token cost over completeness.
        let sections = self
            .section_y_range
            .map(|(lo, hi)| format!("{lo}..{hi}"))
            .unwrap_or_else(|| "none".to_string());
        let surface = self
            .heightmap
            .as_ref()
            .map(|h| format!("{}/{}/{:.0}", h.min, h.max, h.avg))
            .unwrap_or_else(|| "?".to_string());

        format!(
            "{},{} {} dv={} sections={sections} entities={} blockents={} inhabited={} biomes={} surface={surface}",
            self.cx,
            self.cz,
            self.status.as_deref().unwrap_or("unknown"),
            self.data_version.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
            self.entity_count,
            self.block_entity_count,
            self.inhabited_time.map(|t| t.to_string()).unwrap_or_else(|| "?".to_string()),
            self.biomes.len(),
        )
    }
}

// -------------------------------------------------------------------------------------------------
// ---- chunks ---------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

pub struct ChunksResult {
    pub save_name: String,
    pub from: (i32, i32),
    pub to: (i32, i32),
    pub total: usize,
    /// Keyed by `Status` string — a `BTreeMap` so `text`/`compact` list them
    /// in a stable order run to run.
    pub status_counts: BTreeMap<String, usize>,
    /// Chunks with no NBT at all: ungenerated terrain, *and* chunks whose
    /// region exists but wouldn't load (`RegionCache` has already logged
    /// that once) — both read as "no data here" for this tally.
    pub ungenerated: Vec<(i32, i32)>,
    pub block_entity_count: usize,
    pub entity_count: usize,
}

/// Runs `chunks`: walks every chunk in the (inclusive) rectangle
/// `args.from..=args.to` through one shared [`RegionCache`] sized to cover
/// exactly the regions the rectangle spans (so nothing is evicted and
/// re-loaded mid-walk), tallying `Status`, presence, and block-entity/entity
/// counts. Never decodes `block_states` — see the module docs.
pub fn chunks(cli: &Cli, args: &ChunksArgs) -> Result<ChunksResult, CliError> {
    let meta = resolve_save(cli)?;
    let (x1, z1) = (args.from.0.x, args.from.0.y);
    let (x2, z2) = (args.to.0.x, args.to.0.y);
    let (cx_min, cx_max) = (x1.min(x2), x1.max(x2));
    let (cz_min, cz_max) = (z1.min(z2), z1.max(z2));

    let width = (cx_max - cx_min) as u64 + 1;
    let depth = (cz_max - cz_min) as u64 + 1;
    let total_u64 = width * depth;
    if total_u64 > MAX_CHUNKS_IN_RANGE {
        return Err(CliError::Usage(format!(
            "chunk range ({cx_min},{cz_min}) to ({cx_max},{cz_max}) covers {total_u64} chunks — over the {MAX_CHUNKS_IN_RANGE}-chunk limit"
        )));
    }
    let total = total_u64 as usize;

    let mut cache = RegionCache::new(meta.clone(), region_span(cx_min, cx_max, cz_min, cz_max));
    let mut entities = EntitiesRegions::new(&meta);

    let mut status_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut ungenerated = Vec::new();
    let mut block_entity_count = 0usize;
    let mut entity_count = 0usize;

    for cz in cz_min..=cz_max {
        for cx in cx_min..=cx_max {
            match load_chunk_nbt(&mut cache, (cx, cz)) {
                Ok(Some(nbt)) => {
                    let status = nbt.get_string("Status").cloned().unwrap_or_else(|| "unknown".to_string());
                    *status_counts.entry(status).or_insert(0) += 1;
                    block_entity_count += list_len(&nbt, "block_entities");
                    entity_count += chunk_entity_count(&nbt, &mut entities, cx, cz);
                }
                Ok(None) | Err(_) => ungenerated.push((cx, cz)),
            }
        }
    }

    Ok(ChunksResult {
        save_name: meta.name,
        from: (cx_min, cz_min),
        to: (cx_max, cz_max),
        total,
        status_counts,
        ungenerated,
        block_entity_count,
        entity_count,
    })
}

/// The number of distinct regions `(cx_min..=cx_max, cz_min..=cz_max)`
/// spans — sized so [`RegionCache`] never has to evict mid-walk, rather than
/// the render-distance-shaped budget [`crate::region_cache::recommended_capacity`]
/// is for.
fn region_span(cx_min: i32, cx_max: i32, cz_min: i32, cz_max: i32) -> usize {
    let (rx_min, rz_min) = chunk_to_region_coord((cx_min, cz_min));
    let (rx_max, rz_max) = chunk_to_region_coord((cx_max, cz_max));
    (((rx_max - rx_min) as usize + 1) * ((rz_max - rz_min) as usize + 1)).max(1)
}

impl ChunksResult {
    fn render_ungenerated_list(&self) -> String {
        let shown: Vec<String> = self
            .ungenerated
            .iter()
            .take(UNGENERATED_DISPLAY_CAP)
            .map(|(x, z)| format!("({x},{z})"))
            .collect();
        let overflow = self.ungenerated.len().saturating_sub(UNGENERATED_DISPLAY_CAP);
        if overflow > 0 {
            format!("{}, ... +{overflow} more", shown.join(", "))
        } else {
            shown.join(", ")
        }
    }
}

impl Render for ChunksResult {
    fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "chunks ({}, {}) to ({}, {}) in {}: {} total",
            self.from.0, self.from.1, self.to.0, self.to.1, self.save_name, self.total
        )];
        for (status, count) in &self.status_counts {
            lines.push(format!("  {status}: {count}"));
        }
        lines.push(format!("  ungenerated: {}", self.ungenerated.len()));
        if !self.ungenerated.is_empty() {
            lines.push(format!("    {}", self.render_ungenerated_list()));
        }
        lines.push(format!("  block entities: {}", self.block_entity_count));
        lines.push(format!("  entities: {}", self.entity_count));
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "from": { "x": self.from.0, "z": self.from.1 },
            "to": { "x": self.to.0, "z": self.to.1 },
            "total": self.total,
            "status_counts": self.status_counts,
            "ungenerated": self.ungenerated.iter().map(|(x, z)| json!({"x": x, "z": z})).collect::<Vec<_>>(),
            "block_entities": self.block_entity_count,
            "entities": self.entity_count,
        })
    }

    fn render_compact(&self) -> String {
        let statuses: Vec<String> = self
            .status_counts
            .iter()
            .map(|(status, count)| format!("{status}={count}"))
            .collect();
        format!(
            "{},{}..{},{} total={} {} ungenerated={} blockents={} entities={}",
            self.from.0,
            self.from.1,
            self.to.0,
            self.to.1,
            self.total,
            statuses.join(" "),
            self.ungenerated.len(),
            self.block_entity_count,
            self.entity_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnbt::{NbtList, NbtValue};

    fn byte_field(name: &str, value: i8) -> NbtField {
        NbtField {
            name: name.to_string(),
            value: NbtValue::Byte(value as u8),
        }
    }

    /// A palette entry with no `Properties` compound — same shape
    /// `blueprint::extract`'s tests build, reused here rather than
    /// hand-rolled again.
    fn plain(name: &str) -> NbtField {
        NbtField::new_compound("", vec![NbtField::new_string("Name", name)])
    }

    fn section(y: i8, palette: Vec<NbtField>, biomes: Option<Vec<&str>>) -> NbtField {
        let mut fields = vec![
            byte_field("Y", y),
            NbtField::new_compound(
                "block_states",
                vec![NbtField::new_list("palette", NbtList::Compound(palette))],
            ),
        ];
        if let Some(biomes) = biomes {
            fields.push(NbtField::new_compound(
                "biomes",
                vec![NbtField::new_list(
                    "palette",
                    NbtList::String(biomes.into_iter().map(str::to_string).collect()),
                )],
            ));
        }
        NbtField::new_compound("", fields)
    }

    /// A light-only sentinel section (below the world bottom): a `Y` tag and
    /// nothing else — no `block_states`, which is what [`section_y_range`]
    /// has to exclude it by.
    fn sentinel_section(y: i8) -> NbtField {
        NbtField::new_compound("", vec![byte_field("Y", y)])
    }

    fn packed_heightmap(value: i32) -> Vec<i64> {
        mc_anvil::heightmap::pack_heightmap(&[value; 256], mc_anvil::heightmap::DEFAULT_MIN_Y)
    }

    fn chunk_nbt(status: &str, data_version: i32, sections: Vec<NbtField>) -> NbtField {
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_string("Status", status),
                NbtField::new_i32("DataVersion", data_version),
                NbtField::new_long("InhabitedTime", 118402),
                NbtField::new_list("sections", NbtList::Compound(sections)),
                NbtField::new_compound(
                    "Heightmaps",
                    vec![NbtField::new_long_array("WORLD_SURFACE", packed_heightmap(64))],
                ),
                NbtField::new_list(
                    "block_entities",
                    NbtList::Compound(vec![plain("minecraft:chest")]),
                ),
            ],
        )
    }

    fn no_entities_region(_meta: &SaveMeta) -> EntitiesRegions {
        EntitiesRegions { dir: None, loaded: HashMap::new() }
    }

    #[test]
    fn extracts_status_data_version_and_inhabited_time() {
        let nbt = chunk_nbt(
            "minecraft:full",
            4189,
            vec![section(0, vec![plain("minecraft:stone")], None)],
        );
        let mut entities = EntitiesRegions { dir: None, loaded: HashMap::new() };
        let result = build_chunk_result("save".to_string(), 0, 0, &nbt, &mut entities);

        assert_eq!(result.status.as_deref(), Some("minecraft:full"));
        assert_eq!(result.data_version, Some(4189));
        assert_eq!(result.inhabited_time, Some(118402));
    }

    #[test]
    fn section_y_range_ignores_the_light_only_sentinel() {
        let nbt = chunk_nbt(
            "minecraft:full",
            4189,
            vec![
                sentinel_section(-5),
                section(-4, vec![plain("minecraft:stone")], None),
                section(19, vec![plain("minecraft:air")], None),
            ],
        );
        assert_eq!(section_y_range(&nbt), Some((-4, 19)));
    }

    #[test]
    fn section_y_range_is_none_without_any_real_sections() {
        let nbt = chunk_nbt("minecraft:full", 4189, vec![sentinel_section(-5)]);
        assert_eq!(section_y_range(&nbt), None);
    }

    #[test]
    fn block_entity_count_reads_the_chunk_roots_list() {
        let nbt = chunk_nbt("minecraft:full", 4189, vec![]);
        assert_eq!(list_len(&nbt, "block_entities"), 1);
    }

    #[test]
    fn biomes_are_distinct_and_sorted_across_sections() {
        let nbt = chunk_nbt(
            "minecraft:full",
            4189,
            vec![
                section(0, vec![plain("minecraft:stone")], Some(vec!["minecraft:forest", "minecraft:plains"])),
                section(1, vec![plain("minecraft:stone")], Some(vec!["minecraft:plains"])),
            ],
        );
        assert_eq!(
            distinct_biomes(&nbt),
            vec!["minecraft:forest".to_string(), "minecraft:plains".to_string()]
        );
    }

    #[test]
    fn heightmap_peek_reports_min_max_and_average_of_a_flat_surface() {
        let nbt = chunk_nbt("minecraft:full", 4189, vec![]);
        let mut entities = EntitiesRegions { dir: None, loaded: HashMap::new() };
        let result = build_chunk_result("save".to_string(), 0, 0, &nbt, &mut entities);

        let peek = result.heightmap.expect("chunk_nbt always writes WORLD_SURFACE");
        assert_eq!(peek.min, 64);
        assert_eq!(peek.max, 64);
        assert_eq!(peek.avg, 64.0);
    }

    #[test]
    fn entity_count_reads_a_pre_flattening_style_entities_list_at_the_chunk_root() {
        let mut nbt = chunk_nbt("minecraft:full", 4189, vec![]);
        nbt.insert(NbtField::new_list(
            "Entities",
            NbtList::Compound(vec![plain("minecraft:cow"), plain("minecraft:pig")]),
        ));
        let mut entities = EntitiesRegions { dir: None, loaded: HashMap::new() };
        assert_eq!(chunk_entity_count(&nbt, &mut entities, 0, 0), 2);
    }

    #[test]
    fn entity_count_is_zero_with_no_entities_list_and_no_entities_region() {
        let nbt = chunk_nbt("minecraft:full", 4189, vec![]);
        let mut entities = no_entities_region(&SaveMeta {
            name: "s".to_string(),
            path: PathBuf::from("does-not-exist"),
            region_dir: PathBuf::from("does-not-exist/region"),
            regions: vec![],
        });
        assert_eq!(chunk_entity_count(&nbt, &mut entities, 0, 0), 0);
    }

    #[test]
    fn a_status_string_is_missing_when_the_tag_is_absent() {
        let nbt = NbtField::new_compound("", vec![NbtField::new_i32("DataVersion", 4189)]);
        assert_eq!(nbt.get_string("Status"), None);
        assert_eq!(section_y_range(&nbt), None);
        assert_eq!(distinct_biomes(&nbt), Vec::<String>::new());
        assert_eq!(list_len(&nbt, "block_entities"), 0);
    }

    #[test]
    fn region_span_covers_a_rectangle_within_one_region() {
        // 2..10 on both axes sits entirely inside region (0, 0) — chunks
        // 0..31 — unlike -2..2, which straddles the region boundary at 0.
        assert_eq!(region_span(2, 10, 2, 10), 1);
    }

    #[test]
    fn region_span_covers_a_rectangle_crossing_a_region_boundary() {
        // x = -1 is region -1, x = 32 is region 1: three regions wide.
        assert_eq!(region_span(-1, 32, 0, 0), 3);
    }
}
