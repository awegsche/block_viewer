//! The extraction walk itself (ticket 022): a [`SelectionBounds`] in, a
//! [`Blueprint`] out. No Bevy beyond [`IVec3`], and no threading — putting
//! this on a background task is [`super`]'s job.
//!
//! ## Why this doesn't read `DecodedWorld`
//!
//! [`crate::world::decode_chunk`] interns only a palette entry's `Name`
//! (the mesher never needed anything else), so its `Properties` compound is
//! dropped on the floor. A blueprint built from the decoded grid would come
//! back with every stair, slab, log, fence, door and repeater in its
//! *default* orientation: geometry preserved, orientation destroyed, and
//! silently so. So extraction goes through the raw NBT the region cache
//! already holds — the same path the block inspector (ticket 007) reads a
//! single block through, which is what makes the two cross-checkable.
//!
//! A useful side effect: the region cache loads from disk on demand, so a
//! selection is **not** limited to chunks inside the render distance.
//!
//! ## Sections are selected by their `Y` tag
//!
//! `ChunkRegion::get_block` indexes `sections` positionally, which assumes
//! every chunk's section list starts at the world bottom and is contiguous.
//! This walk reads each section's own `Y` byte instead, the way
//! [`crate::world::decode`] does — one less assumption, and the two block
//! readers in this repo then agree with each other. On a save where the
//! assumption doesn't hold, the block inspector is the one that would be
//! wrong (see the ticket's `mc_anvil` note).
//!
//! ## Locking
//!
//! The region cache lock is shared with every streaming chunk-load task, so
//! holding it across a multi-million-block walk would stall terrain
//! streaming outright. [`extract_blueprint`] takes it **per chunk column**
//! and releases between, and visits columns region-major so each region file
//! is loaded once rather than once per row.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;
use mc_anvil::MCLoadError;
use rnbt::NbtField;

use crate::chunk_pipeline::local_chunk_index;
use crate::region_cache::{chunk_to_region_coord, RegionCache};
use crate::selection::SelectionBounds;
use crate::world::{DecodeError, SECTION_SIZE};

/// Hard ceiling on how many blocks one extraction will walk.
///
/// Also what ticket 021's panel disables its export button above, so the two
/// can't drift apart — the panel reads this constant rather than keeping its
/// own copy. 16,000,000 blocks is ~32 MB of `u16` indices before the palette,
/// and takes seconds rather than minutes.
pub const MAX_BLOCKS: u64 = 16_000_000;

/// `DataVersion` used when *no* chunk in the selection carried one — i.e.
/// the selection covers nothing but ungenerated terrain, so the blueprint is
/// entirely air and the version barely matters. 3953 is Minecraft 1.21.
///
/// Ticket 023 writes this into the structure file; guessing it wrong for a
/// blueprint with real blocks in it would make the file load incorrectly in a
/// different game version, which is why the real value is copied off a chunk
/// whenever there is one to copy it from.
pub const FALLBACK_DATA_VERSION: i32 = 3953;

/// One distinct block state: a name plus its properties, **sorted by key**.
///
/// The sort is what makes the palette dedupe correct. NBT compound field
/// order isn't guaranteed stable, so an unsorted key would emit the same
/// stair twice under two orderings — bloating the palette and making the
/// output differ between runs of the same extraction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockState {
    pub name: String,
    /// `(key, value)` pairs, sorted by key. Vanilla block-state properties
    /// are all strings; any non-string property is dropped rather than
    /// guessed at (the block inspector does the same).
    pub properties: Vec<(String, String)>,
}

impl BlockState {
    pub const AIR: &'static str = "minecraft:air";

    /// The state every unreadable, ungenerated or out-of-range block comes
    /// back as — see [`Accumulator::new`], which seeds the palette with it.
    pub fn air() -> Self {
        Self {
            name: Self::AIR.to_string(),
            properties: Vec::new(),
        }
    }

    /// Reads one `block_states.palette` entry: its `Name`, plus its
    /// `Properties` compound if it has one.
    ///
    /// `pub(crate)` for ticket 031's edit model, which records what a write
    /// replaced (the as-built baseline) and must spell those states exactly
    /// the way an extraction would — same sort, same non-string handling — or
    /// a blueprint and a baseline of the same blocks wouldn't compare equal.
    pub(crate) fn from_palette_entry(entry: &NbtField) -> Result<Self, DecodeError> {
        let name = entry
            .get_string("Name")
            .ok_or(DecodeError::MissingField("Name"))?
            .clone();
        let mut properties: Vec<(String, String)> = entry
            .get_compound("Properties")
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|f| f.as_string().map(|v| (f.name.clone(), v.clone())))
                    .collect()
            })
            .unwrap_or_default();
        properties.sort();
        Ok(Self { name, properties })
    }
}

/// The vanilla block-state string: `minecraft:oak_stairs[facing=north,half=bottom]`.
/// Used for the palette log line ticket 022's manual check reads, and the
/// natural thing for ticket 023 to write into a debug dump.
impl std::fmt::Display for BlockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if self.properties.is_empty() {
            return Ok(());
        }
        write!(f, "[")?;
        for (i, (key, value)) in self.properties.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{key}={value}")?;
        }
        write!(f, "]")
    }
}

/// A selected volume, extracted: a palette of distinct block states and one
/// index into it per block.
///
/// Block entities (chest contents, sign text), entities and biomes are all
/// deliberately absent — the vanilla structure format ticket 023 writes has
/// slots for them, and they're each their own follow-up (chest contents in
/// particular are a second NBT path entirely, `block_entities` in the chunk
/// root).
#[derive(Debug, Clone)]
pub struct Blueprint {
    /// Selection size in blocks (X, Y, Z), Minecraft axes.
    pub size: IVec3,
    /// Where in the world this came from — the selection's `min` corner. Not
    /// written to the file (a structure is position-independent), but worth
    /// carrying for the UI and for ticket 024's default filename.
    pub origin: IVec3,
    /// Distinct block states, in first-seen order — except index 0, which is
    /// always `minecraft:air` (see [`Accumulator::new`]).
    pub palette: Vec<BlockState>,
    /// One palette index per block, `size.x * size.y * size.z` of them, in
    /// [`SelectionBounds::iter_blocks`]'s Y-outer / Z-middle / X-inner order.
    pub blocks: Vec<u16>,
    /// The save's `DataVersion`, copied off the first chunk in the selection
    /// that has one, else [`FALLBACK_DATA_VERSION`].
    pub data_version: i32,
    /// Chunk columns whose NBT was present but unreadable (malformed
    /// sections, a palette index past the end of its palette, ...). Those
    /// columns are air in `blocks`; the count is here so the UI can say so
    /// rather than the extraction silently coming back short.
    ///
    /// Columns the save simply doesn't have — ungenerated chunks, a region
    /// file that was never written — are **not** counted: that's normal, not
    /// a failure.
    pub failed_columns: usize,
}

impl Blueprint {
    /// Total blocks, i.e. `blocks.len()`.
    pub fn volume(&self) -> usize {
        self.blocks.len()
    }

    /// The state at a Minecraft block coordinate, or `None` if it's outside
    /// the extracted volume.
    ///
    /// No caller outside this module's tests yet, where it's how every
    /// "the right block landed at the right index" assertion is written —
    /// which is also what makes it the obvious thing for ticket 023/024 to
    /// spot-check a blueprint with, so it stays.
    #[allow(dead_code)]
    pub fn block_at(&self, block: IVec3) -> Option<&BlockState> {
        let bounds = SelectionBounds::from_corners(
            self.origin,
            self.origin,
            self.origin + self.size - IVec3::ONE,
        );
        let index = bounds.index_of(block)?;
        self.palette.get(*self.blocks.get(index)? as usize)
    }
}

/// Why an extraction gave up entirely. Everything survivable — an
/// ungenerated chunk, a Y above the highest section, a region file the save
/// doesn't have, even one corrupt column — is handled as air and counted
/// instead (see [`Blueprint::failed_columns`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    /// The selection is over [`MAX_BLOCKS`]. Ticket 021's panel disables its
    /// export button at the same threshold, so this is a backstop rather than
    /// something a user should be able to hit through the UI.
    TooLarge { volume: u64 },
    /// More than 65,535 *distinct* block states, which `u16` indices can't
    /// address. Doesn't happen in practice; erroring beats truncating.
    PaletteTooLarge,
    /// No save is loaded, so there's no region cache to read from (ticket
    /// 008's empty-save startup path).
    NoSaveLoaded,
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::TooLarge { volume } => write!(
                f,
                "selection is {volume} blocks — over the {MAX_BLOCKS}-block export limit"
            ),
            ExtractError::PaletteTooLarge => write!(
                f,
                "selection contains more than {} distinct block states",
                u16::MAX
            ),
            ExtractError::NoSaveLoaded => write!(f, "no save is loaded"),
        }
    }
}

impl std::error::Error for ExtractError {}

/// Why one chunk column came back unusable.
///
/// Split from [`ExtractError`] because the two get opposite treatment: `Nbt`
/// leaves that column as air and carries on (a corrupt chunk shouldn't lose a
/// million-block extraction), while `PaletteTooLarge` can only get worse the
/// longer the walk continues.
#[derive(Debug)]
enum ColumnError {
    Nbt(DecodeError),
    PaletteTooLarge,
}

impl From<DecodeError> for ColumnError {
    fn from(err: DecodeError) -> Self {
        ColumnError::Nbt(err)
    }
}

/// Coarse progress for the UI, shared with the extraction task.
///
/// Columns rather than blocks: columns are the unit the walk actually loops
/// over, and a per-block counter would be an atomic increment per block for
/// a number nobody can read that fast anyway. `Relaxed` throughout — this is
/// a progress bar, and nothing else is ordered against it.
#[derive(Debug, Default)]
pub struct ExtractProgress {
    done: AtomicUsize,
    total: AtomicUsize,
}

impl ExtractProgress {
    fn begin(&self, total: usize) {
        self.total.store(total, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
    }

    fn advance(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// `(columns done, columns total)`. `total` is 0 until the walk starts.
    pub fn columns(&self) -> (usize, usize) {
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }

    /// Fraction done in `0.0..=1.0`, or 0.0 before the total is known.
    pub fn fraction(&self) -> f32 {
        let (done, total) = self.columns();
        if total == 0 {
            return 0.0;
        }
        (done as f32 / total as f32).clamp(0.0, 1.0)
    }
}

/// Extracts `bounds` through `region_cache`, reporting coarse progress into
/// `progress`.
///
/// Blocking and synchronous: `ChunkRegion`'s NBT is walked by field name, and
/// a big selection is seconds of work, so callers run this on
/// `AsyncComputeTaskPool` (see [`super::start_extraction`]) rather than in a
/// frame.
pub fn extract_blueprint(
    bounds: SelectionBounds,
    region_cache: &Arc<Mutex<RegionCache>>,
    progress: &ExtractProgress,
) -> Result<Blueprint, ExtractError> {
    let mut acc = Accumulator::new(bounds)?;

    let columns = column_order(bounds);
    progress.begin(columns.len());

    for coord in columns {
        let region_coord = chunk_to_region_coord(coord);
        let (local_x, local_z) = local_chunk_index(coord, region_coord);

        {
            let mut cache = region_cache.lock().expect("region cache mutex poisoned");
            match cache.get_or_load(region_coord) {
                Ok(region) => {
                    // A chunk the region doesn't have is ungenerated terrain:
                    // air, and entirely normal at the edge of an explored
                    // area.
                    if let Some(nbt) = region.get_chunk(local_x, local_z) {
                        match acc.sample_column(coord, nbt) {
                            Ok(()) => {}
                            Err(ColumnError::Nbt(err)) => acc.note_failed_column(coord, &err),
                            Err(ColumnError::PaletteTooLarge) => {
                                return Err(ExtractError::PaletteTooLarge)
                            }
                        }
                    }
                }
                // The save has no region file here at all — the whole 512x512
                // block area is unexplored. Air, not a failure.
                Err(MCLoadError::PathNotFoundError) => {}
                // A region that exists but wouldn't load (truncated/corrupt
                // `.mca`). `RegionCache` has already logged it once; count the
                // columns it costs us so the UI can report them.
                //
                // Note this only counts the *first* request for a broken
                // region: the cache remembers the failure and answers later
                // ones with `PathNotFoundError`, so a corrupt region shows up
                // as one failed column plus a lot of quiet air.
                Err(err) => acc.note_failed_region(region_coord, &err),
            }
        }

        progress.advance();
    }

    Ok(acc.finish())
}

/// Every chunk column `bounds` overlaps, ordered **region-major**: all of one
/// region's columns before any of the next region's.
///
/// A naive Z-then-X sweep across a wide selection revisits every region on
/// every row, and the region cache is only sized for a render distance, so it
/// would evict and re-parse the same `.mca` files hundreds of times. This
/// order touches each region file once.
fn column_order(bounds: SelectionBounds) -> Vec<(i32, i32)> {
    let size = SECTION_SIZE as i32;
    let width = REGION_WIDTH_IN_CHUNKS as i32;

    let (min_cx, min_cz) = (bounds.min.x.div_euclid(size), bounds.min.z.div_euclid(size));
    let (max_cx, max_cz) = (bounds.max.x.div_euclid(size), bounds.max.z.div_euclid(size));
    let (min_rx, min_rz) = chunk_to_region_coord((min_cx, min_cz));
    let (max_rx, max_rz) = chunk_to_region_coord((max_cx, max_cz));

    let mut columns = Vec::new();
    for rz in min_rz..=max_rz {
        for rx in min_rx..=max_rx {
            for cz in (rz * width).max(min_cz)..=((rz + 1) * width - 1).min(max_cz) {
                for cx in (rx * width).max(min_cx)..=((rx + 1) * width - 1).min(max_cx) {
                    columns.push((cx, cz));
                }
            }
        }
    }
    columns
}

/// One section's palette resolved to blueprint palette indices, plus what it
/// takes to read a block out of it.
///
/// Borrows `data` out of the chunk NBT rather than copying it — a section's
/// packed array is up to 1024 longs and there are up to 24 of them per
/// column.
enum SectionBlocks<'a> {
    /// A single-entry palette: every block in the section is that one state,
    /// and Minecraft omits the packed `data` array entirely.
    Uniform(u16),
    Packed {
        palette: Vec<u16>,
        data: &'a [i64],
        bits: u32,
    },
}

/// The extraction in progress: the dense output array, the palette being
/// built, and the counters the [`Blueprint`] carries out.
struct Accumulator {
    bounds: SelectionBounds,
    palette: Vec<BlockState>,
    /// `BlockState` -> its index in `palette`. A `HashMap` over the *sorted*
    /// state (see [`BlockState`]) is what makes two spellings of the same
    /// stair collapse into one entry.
    lookup: HashMap<BlockState, u16>,
    blocks: Vec<u16>,
    data_version: Option<i32>,
    failed_columns: usize,
    /// Whether a failure has already been logged for this extraction — one
    /// broken region is thousands of columns, and thousands of identical log
    /// lines are worse than one plus a count.
    logged_failure: bool,
}

impl Accumulator {
    /// Allocates the dense block array, pre-filled with air.
    ///
    /// Air is interned as palette index **0** before anything is read, which
    /// is the one deliberate departure from "first-seen order": the fill
    /// value has to exist in the palette, and every missing chunk,
    /// out-of-range Y and unreadable column resolves to it. A selection with
    /// no air in it still carries an unused air entry, which costs one
    /// palette slot and no blocks.
    fn new(bounds: SelectionBounds) -> Result<Self, ExtractError> {
        let volume = bounds.volume();
        if volume > MAX_BLOCKS {
            return Err(ExtractError::TooLarge { volume });
        }
        let volume = volume as usize;

        let air = BlockState::air();
        Ok(Self {
            bounds,
            palette: vec![air.clone()],
            lookup: HashMap::from([(air, 0u16)]),
            blocks: vec![0u16; volume],
            data_version: None,
            failed_columns: 0,
            logged_failure: false,
        })
    }

    fn intern(&mut self, state: BlockState) -> Result<u16, ColumnError> {
        if let Some(&index) = self.lookup.get(&state) {
            return Ok(index);
        }
        let index = u16::try_from(self.palette.len()).map_err(|_| ColumnError::PaletteTooLarge)?;
        self.palette.push(state.clone());
        self.lookup.insert(state, index);
        Ok(index)
    }

    /// Reads the part of chunk column `coord` that falls inside the
    /// selection, writing it into `blocks` at
    /// [`SelectionBounds::index_of`]'s positions.
    ///
    /// Resolving the column once and sampling its slice is the whole point of
    /// iterating by column: `ChunkRegion::get_block` re-walks the chunk's NBT
    /// by field name on every single call, which is fine for the inspector's
    /// one block per frame and hopeless a million times over.
    ///
    /// A failure part-way through leaves whatever it already wrote in place;
    /// the rest of that column stays air and the column is counted in
    /// [`Blueprint::failed_columns`].
    fn sample_column(&mut self, coord: (i32, i32), nbt: &NbtField) -> Result<(), ColumnError> {
        // Any chunk in the selection will do — they all come from the same
        // save, so the first one that has it settles it.
        if self.data_version.is_none() {
            self.data_version = nbt.get_int("DataVersion");
        }

        // Note there's no `Status == "minecraft:full"` check, unlike
        // `world::decode_chunk`: a partially generated chunk still has real
        // blocks in it, and `ChunkRegion::get_block` — the block inspector's
        // path, the thing an extraction gets cross-checked against — doesn't
        // check it either.
        let size = SECTION_SIZE as i32;
        let (x_lo, x_hi) = axis_overlap(coord.0 * size, self.bounds.min.x, self.bounds.max.x);
        let (z_lo, z_hi) = axis_overlap(coord.1 * size, self.bounds.min.z, self.bounds.max.z);

        let sections = nbt
            .get_list("sections")
            .ok_or(DecodeError::MissingField("sections"))?
            .as_compound_list()
            .ok_or(DecodeError::UnexpectedType("sections"))?;

        for section in sections {
            // By the section's own `Y` tag, not its position in the list —
            // see the module docs.
            let Some(y) = section.get_byte("Y") else {
                continue;
            };
            let base_y = y as i8 as i32 * size;
            let y_lo = self.bounds.min.y.max(base_y);
            let y_hi = self.bounds.max.y.min(base_y + size - 1);
            if y_lo > y_hi {
                continue; // This section is entirely outside the selection.
            }

            // Lighting-only sentinel sections carry no blocks at all.
            let Some(block_states) = section.get("block_states") else {
                continue;
            };
            let decoded = self.decode_section(block_states)?;

            for y in y_lo..=y_hi {
                let dy = (y - base_y) as usize;
                for z in z_lo..=z_hi {
                    let dz = (z - coord.1 * size) as usize;
                    for x in x_lo..=x_hi {
                        let dx = (x - coord.0 * size) as usize;
                        let index = decoded.at(dx, dy, dz)?;
                        // The array is already all air; skipping saves the
                        // index arithmetic on the majority of most selections.
                        if index == 0 {
                            continue;
                        }
                        let out = self
                            .bounds
                            .index_of(IVec3::new(x, y, z))
                            .expect("sampled coordinates are clamped to the selection");
                        self.blocks[out] = index;
                    }
                }
            }
        }

        Ok(())
    }

    /// Resolves one section's palette into blueprint palette indices. Same
    /// packed-array rules as [`crate::world::decode`]: a 4-bit minimum on the
    /// index width (ticket 001), and indices never spanning two longs.
    fn decode_section<'a>(
        &mut self,
        block_states: &'a NbtField,
    ) -> Result<SectionBlocks<'a>, ColumnError> {
        let palette = block_states
            .get_list("palette")
            .ok_or(DecodeError::MissingField("block_states.palette"))?
            .as_compound_list()
            .ok_or(DecodeError::UnexpectedType("block_states.palette"))?;
        if palette.is_empty() {
            return Err(DecodeError::EmptyPalette.into());
        }

        let mut indices = Vec::with_capacity(palette.len());
        for entry in palette {
            indices.push(self.intern(BlockState::from_palette_entry(entry)?)?);
        }

        if indices.len() == 1 {
            return Ok(SectionBlocks::Uniform(indices[0]));
        }

        let data = block_states
            .get_long_array("data")
            .ok_or(DecodeError::MissingField("block_states.data"))?;
        let bits = ((indices.len() as u32 - 1).ilog2() + 1).max(4);
        Ok(SectionBlocks::Packed {
            palette: indices,
            data,
            bits,
        })
    }

    fn note_failed_column(&mut self, coord: (i32, i32), err: &DecodeError) {
        self.failed_columns += 1;
        if !self.logged_failure {
            self.logged_failure = true;
            println!(
                "block_viewer: extraction: chunk {coord:?} is unreadable ({err}) — \
                 treating it as air (later failures counted, not logged)"
            );
        }
    }

    fn note_failed_region(&mut self, region: (i32, i32), err: &MCLoadError) {
        self.failed_columns += 1;
        if !self.logged_failure {
            self.logged_failure = true;
            println!(
                "block_viewer: extraction: region {region:?} would not load ({err}) — \
                 treating it as air (later failures counted, not logged)"
            );
        }
    }

    fn finish(self) -> Blueprint {
        Blueprint {
            size: self.bounds.size(),
            origin: self.bounds.min,
            palette: self.palette,
            blocks: self.blocks,
            data_version: self.data_version.unwrap_or(FALLBACK_DATA_VERSION),
            failed_columns: self.failed_columns,
        }
    }
}

impl SectionBlocks<'_> {
    /// The palette index at section-local `(x, y, z)` (each 0..16).
    fn at(&self, x: usize, y: usize, z: usize) -> Result<u16, ColumnError> {
        match self {
            SectionBlocks::Uniform(index) => Ok(*index),
            SectionBlocks::Packed {
                palette,
                data,
                bits,
            } => {
                let bits = *bits as usize;
                let flat = y * SECTION_SIZE * SECTION_SIZE + z * SECTION_SIZE + x;
                let per_long = 64 / bits;
                let long = *data
                    .get(flat / per_long)
                    .ok_or(DecodeError::UnexpectedType("block_states.data"))?
                    as u64;
                let mask = (1u64 << bits) - 1;
                let index = ((long >> ((flat % per_long) * bits)) & mask) as usize;
                palette
                    .get(index)
                    .copied()
                    .ok_or_else(|| DecodeError::PaletteIndexOutOfRange(index).into())
            }
        }
    }
}

/// The overlap between one 16-block chunk axis starting at `base` and the
/// inclusive selection range `min..=max`, as an inclusive `(lo, hi)` pair.
/// Only called for columns the selection actually overlaps, so the result is
/// never empty.
fn axis_overlap(base: i32, min: i32, max: i32) -> (i32, i32) {
    (
        min.max(base),
        max.min(base + SECTION_SIZE as i32 - 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnbt::{NbtField, NbtList, NbtValue};

    fn bounds(min: IVec3, max: IVec3) -> SelectionBounds {
        SelectionBounds::from_corners(min, min, max)
    }

    fn byte_field(name: &str, value: i8) -> NbtField {
        NbtField {
            name: name.to_string(),
            value: NbtValue::Byte(value as u8),
        }
    }

    /// A palette entry with no `Properties` compound.
    fn plain(name: &str) -> NbtField {
        NbtField::new_compound("", vec![NbtField::new_string("Name", name)])
    }

    /// A palette entry with `Properties`, in exactly the field order given —
    /// which is the thing the dedupe test needs control over.
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

    fn section(y: i8, palette: Vec<NbtField>, data: Option<Vec<i64>>) -> NbtField {
        let mut fields = vec![NbtField::new_list("palette", NbtList::Compound(palette))];
        if let Some(data) = data {
            fields.push(NbtField::new_long_array("data", data));
        }
        NbtField::new_compound(
            "",
            vec![
                byte_field("Y", y),
                NbtField::new_compound("block_states", fields),
            ],
        )
    }

    fn chunk(sections: Vec<NbtField>) -> NbtField {
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_i32("DataVersion", 4189),
                NbtField::new_list("sections", NbtList::Compound(sections)),
            ],
        )
    }

    /// Runs the walk over a handful of synthetic columns without a save on
    /// disk — the sampler is the half that has all the arithmetic in it, and
    /// `extract_blueprint` only adds the region-cache lookup around it.
    fn extract(bounds: SelectionBounds, columns: &[((i32, i32), NbtField)]) -> Blueprint {
        let mut acc = Accumulator::new(bounds).expect("bounds should be extractable");
        for (coord, nbt) in columns {
            acc.sample_column(*coord, nbt).expect("column should sample");
        }
        acc.finish()
    }

    /// A 4-bit packed `data` array (the minimum width — any palette of 2..=16
    /// entries uses it) with `index` at section-local `(x, y, z)`.
    fn packed_4bit(cells: &[(usize, usize, usize, i64)]) -> Vec<i64> {
        let mut data = vec![0i64; 256]; // 4096 cells / 16 per long
        for &(x, y, z, index) in cells {
            let flat = y * 256 + z * 16 + x;
            data[flat / 16] |= index << ((flat % 16) * 4);
        }
        data
    }

    #[test]
    fn air_is_always_palette_index_zero() {
        let blueprint = extract(bounds(IVec3::new(0, 0, 0), IVec3::new(1, 1, 1)), &[]);
        assert_eq!(blueprint.palette[0], BlockState::air());
        assert!(blueprint.blocks.iter().all(|&b| b == 0));
    }

    /// The ticket's headline dedupe case: NBT compound field order isn't
    /// stable, so the palette key has to be order-independent or the same
    /// stair lands in the palette twice.
    #[test]
    fn identical_states_in_a_different_field_order_dedupe_to_one_entry() {
        let stairs_a = with_properties(
            "minecraft:oak_stairs",
            &[("facing", "north"), ("half", "bottom"), ("shape", "straight")],
        );
        let stairs_b = with_properties(
            "minecraft:oak_stairs",
            &[("shape", "straight"), ("facing", "north"), ("half", "bottom")],
        );

        // Two sections, each uniform, each with one of the two spellings.
        let nbt = chunk(vec![
            section(0, vec![stairs_a], None),
            section(1, vec![stairs_b], None),
        ]);
        let blueprint = extract(bounds(IVec3::new(0, 0, 0), IVec3::new(0, 31, 0)), &[((0, 0), nbt)]);

        assert_eq!(
            blueprint.palette.len(),
            2,
            "air plus one stair state, not two: {:?}",
            blueprint.palette
        );
        assert_eq!(
            blueprint.block_at(IVec3::new(0, 0, 0)),
            blueprint.block_at(IVec3::new(0, 16, 0))
        );
    }

    /// The regression test for the whole reason this doesn't read
    /// `DecodedWorld`: two stairs that differ only in their properties are
    /// two distinct states, not one.
    #[test]
    fn the_same_block_with_different_properties_stays_two_entries() {
        let north = with_properties("minecraft:oak_stairs", &[("facing", "north")]);
        let south = with_properties("minecraft:oak_stairs", &[("facing", "south")]);
        let nbt = chunk(vec![
            section(0, vec![north], None),
            section(1, vec![south], None),
        ]);
        let blueprint = extract(bounds(IVec3::new(0, 0, 0), IVec3::new(0, 31, 0)), &[((0, 0), nbt)]);

        assert_eq!(blueprint.palette.len(), 3, "air + two stair states");
        let low = blueprint.block_at(IVec3::new(0, 0, 0)).unwrap();
        let high = blueprint.block_at(IVec3::new(0, 16, 0)).unwrap();
        assert_ne!(low, high);
        assert_eq!(low.to_string(), "minecraft:oak_stairs[facing=north]");
        assert_eq!(high.to_string(), "minecraft:oak_stairs[facing=south]");
    }

    /// `blocks` is exactly the volume, and a known block lands where
    /// [`SelectionBounds::index_of`] says it should.
    #[test]
    fn a_known_block_lands_at_the_documented_index() {
        // Palette index 1 = dirt, at section-local (2, 3, 4).
        let data = packed_4bit(&[(2, 3, 4, 1)]);
        let nbt = chunk(vec![section(
            0,
            vec![plain("minecraft:air"), plain("minecraft:dirt")],
            Some(data),
        )]);

        let selection = bounds(IVec3::new(0, 0, 0), IVec3::new(7, 7, 7));
        let blueprint = extract(selection, &[((0, 0), nbt)]);

        assert_eq!(blueprint.blocks.len() as u64, selection.volume());
        assert_eq!(blueprint.size, IVec3::splat(8));
        assert_eq!(blueprint.origin, IVec3::ZERO);

        let dirt = IVec3::new(2, 3, 4);
        assert_eq!(
            blueprint.block_at(dirt).map(|s| s.name.as_str()),
            Some("minecraft:dirt")
        );
        // The index the order documented on `iter_blocks` predicts: y outer,
        // z middle, x inner.
        let expected = 3 * 8 * 8 + 4 * 8 + 2;
        assert_eq!(selection.index_of(dirt), Some(expected));
        assert_eq!(blueprint.blocks[expected], 1);
        // Everything else is air.
        assert_eq!(blueprint.blocks.iter().filter(|&&b| b != 0).count(), 1);
    }

    /// Ungenerated chunks and Y values above the highest section are normal,
    /// not failures: air, and nothing counted.
    #[test]
    fn missing_columns_and_out_of_range_y_come_back_as_air() {
        // One section at Y=0 (world Y 0..16); the selection reaches up to 40.
        let nbt = chunk(vec![section(0, vec![plain("minecraft:stone")], None)]);
        let selection = bounds(IVec3::new(0, 0, 0), IVec3::new(0, 40, 0));
        let blueprint = extract(selection, &[((0, 0), nbt)]);

        assert_eq!(blueprint.failed_columns, 0);
        for y in 0..=15 {
            assert_eq!(
                blueprint.block_at(IVec3::new(0, y, 0)).map(|s| s.name.as_str()),
                Some("minecraft:stone"),
                "y = {y}"
            );
        }
        for y in 16..=40 {
            assert_eq!(
                blueprint.block_at(IVec3::new(0, y, 0)).map(|s| s.name.as_str()),
                Some(BlockState::AIR),
                "y = {y}"
            );
        }
    }

    /// A column that never gets sampled at all (its chunk isn't in the save)
    /// is air too — same as above, but for the horizontal axes.
    #[test]
    fn a_column_the_save_does_not_have_is_air() {
        let nbt = chunk(vec![section(0, vec![plain("minecraft:stone")], None)]);
        // Two columns wide, only the first one present.
        let selection = bounds(IVec3::new(0, 0, 0), IVec3::new(31, 0, 0));
        let blueprint = extract(selection, &[((0, 0), nbt)]);

        assert_eq!(
            blueprint.block_at(IVec3::new(15, 0, 0)).map(|s| s.name.as_str()),
            Some("minecraft:stone")
        );
        assert_eq!(
            blueprint.block_at(IVec3::new(16, 0, 0)).map(|s| s.name.as_str()),
            Some(BlockState::AIR)
        );
    }

    /// A selection spanning a chunk boundary *and* a region boundary at
    /// negative coordinates — the `div_euclid`/`rem_euclid` path, where a
    /// plain `/` and `%` would put blocks in the wrong column and silently
    /// mirror part of the blueprint.
    #[test]
    fn a_selection_across_a_negative_chunk_and_region_boundary_samples_the_right_blocks() {
        // Region boundary at x = -512, chunk boundary at x = -16.
        // Selection: x in -513..=-511 crosses both region (-2 -> -1) and
        // chunk (-33 -> -32) boundaries.
        let stone_at = |x: usize, z: usize| {
            chunk(vec![section(
                0,
                vec![plain("minecraft:air"), plain("minecraft:stone")],
                Some(packed_4bit(&[(x, 0, z, 1)])),
            )])
        };

        let selection = bounds(IVec3::new(-513, 0, -17), IVec3::new(-511, 0, -15));

        // Chunk (-33, -2) holds x = -513 (local 15) and z = -17 (local 15).
        // Chunk (-32, -1) holds x = -512..-511 (locals 0, 1) and
        // z = -16..-15 (locals 0, 1).
        let columns = [
            ((-33, -2), stone_at(15, 15)),
            ((-32, -1), stone_at(1, 1)),
            ((-33, -1), chunk(vec![])),
            ((-32, -2), chunk(vec![])),
        ];
        let blueprint = extract(selection, &columns);

        assert_eq!(
            blueprint.block_at(IVec3::new(-513, 0, -17)).map(|s| s.name.as_str()),
            Some("minecraft:stone"),
            "the block in the negative-region column"
        );
        assert_eq!(
            blueprint.block_at(IVec3::new(-511, 0, -15)).map(|s| s.name.as_str()),
            Some("minecraft:stone"),
            "the block across both boundaries"
        );
        assert_eq!(
            blueprint.blocks.iter().filter(|&&b| b != 0).count(),
            2,
            "exactly the two stone blocks, nothing mirrored into a neighbour"
        );
    }

    /// The `column_order` those coordinates go through: every overlapped
    /// column exactly once, grouped by region file.
    #[test]
    fn column_order_covers_every_column_once_grouped_by_region() {
        // x from -513 to 16 spans regions -2, -1 and 0 on X.
        let selection = bounds(IVec3::new(-513, 0, 0), IVec3::new(16, 0, 16));
        let columns = column_order(selection);

        let unique: std::collections::HashSet<(i32, i32)> = columns.iter().copied().collect();
        assert_eq!(unique.len(), columns.len(), "no column visited twice");

        // Chunks -33..=1 on X, 0..=1 on Z.
        assert_eq!(columns.len(), 35 * 2);
        assert!(unique.contains(&(-33, 0)));
        assert!(unique.contains(&(1, 1)));

        // Region-major: every column of one region is contiguous in the
        // sequence, which is what stops the cache thrashing.
        let mut seen_regions = Vec::new();
        for &coord in &columns {
            let region = chunk_to_region_coord(coord);
            if seen_regions.last() != Some(&region) {
                assert!(
                    !seen_regions.contains(&region),
                    "region {region:?} revisited after moving away from it"
                );
                seen_regions.push(region);
            }
        }
        assert_eq!(seen_regions.len(), 3);
    }

    /// The 4-bit minimum (ticket 001) applies here too: a 3-entry palette
    /// packs at 4 bits, not the 2 a naive `ilog2` would compute.
    #[test]
    fn a_small_palette_reads_at_the_four_bit_minimum() {
        let mut data = vec![0i64; 256];
        data[0] = 2; // flat index 0 -> palette index 2
        let nbt = chunk(vec![section(
            0,
            vec![
                plain("minecraft:air"),
                plain("minecraft:stone"),
                plain("minecraft:dirt"),
            ],
            Some(data),
        )]);
        let blueprint = extract(bounds(IVec3::ZERO, IVec3::ZERO), &[((0, 0), nbt)]);
        assert_eq!(
            blueprint.block_at(IVec3::ZERO).map(|s| s.name.as_str()),
            Some("minecraft:dirt")
        );
    }

    /// Wide palettes need more than 4 bits per index, and the section list
    /// is read by `Y` tag rather than by position.
    #[test]
    fn sections_are_read_by_their_y_tag_not_their_position() {
        // Reversed order and a gap: position 0 is Y=2, position 1 is Y=0.
        let nbt = chunk(vec![
            section(2, vec![plain("minecraft:glass")], None),
            section(0, vec![plain("minecraft:stone")], None),
        ]);
        let blueprint = extract(bounds(IVec3::new(0, 0, 0), IVec3::new(0, 47, 0)), &[((0, 0), nbt)]);

        assert_eq!(
            blueprint.block_at(IVec3::new(0, 0, 0)).map(|s| s.name.as_str()),
            Some("minecraft:stone")
        );
        assert_eq!(
            blueprint.block_at(IVec3::new(0, 32, 0)).map(|s| s.name.as_str()),
            Some("minecraft:glass")
        );
        // The Y=1 gap is air.
        assert_eq!(
            blueprint.block_at(IVec3::new(0, 16, 0)).map(|s| s.name.as_str()),
            Some(BlockState::AIR)
        );
    }

    /// A column whose NBT is present but malformed is air *and* counted,
    /// rather than losing the whole extraction.
    #[test]
    fn a_malformed_column_is_counted_and_left_as_air() {
        let good = chunk(vec![section(0, vec![plain("minecraft:stone")], None)]);
        // No `sections` list at all.
        let bad = NbtField::new_compound("", vec![NbtField::new_i32("DataVersion", 4189)]);

        let selection = bounds(IVec3::new(0, 0, 0), IVec3::new(31, 0, 0));
        let mut acc = Accumulator::new(selection).unwrap();
        acc.sample_column((0, 0), &good).unwrap();
        let err = acc.sample_column((1, 0), &bad).unwrap_err();
        assert!(matches!(
            err,
            ColumnError::Nbt(DecodeError::MissingField("sections"))
        ));
        acc.note_failed_column((1, 0), &DecodeError::MissingField("sections"));
        let blueprint = acc.finish();

        assert_eq!(blueprint.failed_columns, 1);
        assert_eq!(
            blueprint.block_at(IVec3::new(0, 0, 0)).map(|s| s.name.as_str()),
            Some("minecraft:stone"),
            "the good column still extracted"
        );
        assert_eq!(
            blueprint.block_at(IVec3::new(16, 0, 0)).map(|s| s.name.as_str()),
            Some(BlockState::AIR)
        );
    }

    #[test]
    fn data_version_comes_from_a_chunk_and_falls_back_when_there_is_none() {
        let nbt = chunk(vec![section(0, vec![plain("minecraft:stone")], None)]);
        let blueprint = extract(bounds(IVec3::ZERO, IVec3::ZERO), &[((0, 0), nbt)]);
        assert_eq!(blueprint.data_version, 4189);

        let empty = extract(bounds(IVec3::ZERO, IVec3::ZERO), &[]);
        assert_eq!(empty.data_version, FALLBACK_DATA_VERSION);
    }

    #[test]
    fn a_selection_over_the_block_limit_errors_instead_of_allocating() {
        let selection = bounds(
            IVec3::new(0, crate::selection::WORLD_MIN_Y, 0),
            IVec3::new(9999, crate::selection::WORLD_MAX_Y, 9999),
        );
        assert!(selection.volume() > MAX_BLOCKS);
        // Not `unwrap_err` — the `Ok` half is the accumulator, which has no
        // `Debug` and doesn't want one (it owns the whole block array).
        match Accumulator::new(selection) {
            Err(err) => assert_eq!(
                err,
                ExtractError::TooLarge {
                    volume: selection.volume()
                }
            ),
            Ok(_) => panic!("an over-limit selection must not allocate"),
        }
    }

    /// Properties come back in sorted order regardless of how they were
    /// written, which is what the dedupe key relies on.
    #[test]
    fn palette_entry_properties_are_sorted_by_key() {
        let entry = with_properties(
            "minecraft:repeater",
            &[("locked", "false"), ("delay", "3"), ("facing", "east")],
        );
        let state = BlockState::from_palette_entry(&entry).unwrap();
        assert_eq!(
            state.properties,
            vec![
                ("delay".to_string(), "3".to_string()),
                ("facing".to_string(), "east".to_string()),
                ("locked".to_string(), "false".to_string()),
            ]
        );
        assert_eq!(
            state.to_string(),
            "minecraft:repeater[delay=3,facing=east,locked=false]"
        );
    }

    #[test]
    fn a_palette_entry_without_a_name_is_a_column_failure() {
        let nameless = NbtField::new_compound("", vec![NbtField::new_string("Nome", "typo")]);
        assert_eq!(
            BlockState::from_palette_entry(&nameless).unwrap_err(),
            DecodeError::MissingField("Name")
        );
    }

    /// Not a correctness test — measures a large extraction against the real
    /// save the way `region_cache`'s `measure_single_region_load_time` does,
    /// so ticket 021's `VOLUME_WARN`/`VOLUME_CAP` thresholds (chosen with no
    /// measurement at all) have a real number behind them. Run with
    /// `cargo test measure_large_extraction -- --nocapture` to see it.
    #[test]
    fn measure_large_extraction() {
        use std::time::Instant;

        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        let region_width = REGION_WIDTH_IN_CHUNKS as i32 * SECTION_SIZE as i32;
        // 128x128x128 = 2,097,152 blocks — past `VOLUME_WARN`, an eighth of
        // `MAX_BLOCKS`, and 64 chunk columns spread over one region.
        let origin = IVec3::new(rx * region_width + 128, 0, rz * region_width + 128);
        let selection = bounds(origin, origin + IVec3::splat(127));

        let cache = Arc::new(Mutex::new(RegionCache::new(meta, 4)));
        let progress = ExtractProgress::default();

        let start = Instant::now();
        let blueprint = extract_blueprint(selection, &cache, &progress).expect("should extract");
        let elapsed = start.elapsed();

        println!(
            "extracted {} blocks ({} columns) in {elapsed:?} — {} distinct states, \
             {} unreadable columns",
            blueprint.volume(),
            progress.columns().1,
            blueprint.palette.len(),
            blueprint.failed_columns,
        );
    }

    /// End to end against the real save, the same convention
    /// `region_cache.rs` and `chunk_pipeline.rs` use: a small selection at a
    /// region's centre should extract without erroring, and any block it
    /// reports should agree with `ChunkRegion::get_block` — the block
    /// inspector's own path — at the same coordinates.
    #[test]
    fn extraction_agrees_with_the_block_inspectors_path_on_a_real_save() {
        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        let region_width = REGION_WIDTH_IN_CHUNKS as i32 * SECTION_SIZE as i32;
        // The middle of a region a player has actually visited is the best
        // bet for generated terrain, same as `chunk_pipeline`'s tests.
        let origin = IVec3::new(
            rx * region_width + region_width / 2,
            60,
            rz * region_width + region_width / 2,
        );
        let selection = bounds(origin, origin + IVec3::new(7, 7, 7));

        let cache = Arc::new(Mutex::new(RegionCache::new(meta, 4)));
        let progress = ExtractProgress::default();
        let blueprint =
            extract_blueprint(selection, &cache, &progress).expect("a small selection should extract");

        assert_eq!(blueprint.blocks.len() as u64, selection.volume());
        assert_eq!(blueprint.origin, origin);
        assert_eq!(progress.columns(), (1, 1), "8x8x8 fits in one chunk column");

        // Cross-check every block against `get_block`, the raw path the
        // inspector reads — the two disagreeing means one of them resolves
        // sections differently (see the module docs).
        let mut cache = cache.lock().unwrap();
        let region = cache.get_or_load((rx, rz)).expect("region should load");
        for block in selection.iter_blocks() {
            let local_x = block.x.rem_euclid(region_width) as usize;
            let local_z = block.z.rem_euclid(region_width) as usize;
            let expected = region
                .get_block(local_x, block.y, local_z)
                .ok()
                .and_then(|f| f.get_string("Name"))
                .cloned();
            let Some(expected) = expected else { continue };
            assert_eq!(
                blueprint.block_at(block).map(|s| s.name.clone()),
                Some(expected),
                "at {block}"
            );
        }
    }
}
