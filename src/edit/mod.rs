//! The chunk edit model (ticket 031, roadmap task W4): what a well-formed
//! edit to a world is, and in what order the pieces happen.
//!
//! `mc_anvil` owns the Anvil *format* — packing block states, writing sector
//! tables, cleaning up orphaned block entities. This module owns what we are
//! allowed to do to a world: which chunks may be written at all, what has to
//! be true before a single block moves, and what has to happen afterwards so
//! the chunk isn't left internally inconsistent.
//!
//! # The sequence
//!
//! The roadmap describes five steps; three of them moved upstream while the
//! `mc_anvil` tickets landed. [`ChunkRegion::set_blocks`] already cleans up
//! the `block_entities` / `block_ticks` / `fluid_ticks` entries an overwrite
//! orphans, and already clears the chunk's `isLightOn` flag so Minecraft
//! relights it — both in their own passes, and both only if the batch was
//! accepted. So what's left here is:
//!
//! 1. [`plan`] — validate the whole edit, mutating nothing.
//! 2. `set_blocks` — one call, itself all-or-nothing.
//! 3. The heightmap policy, once per touched chunk.
//!
//! Step 3 is why heightmap policy lives here rather than upstream: it has to
//! run *after the last block write of the whole edit*, not once per
//! `set_blocks` call. A recompute over half-applied blocks would bake a
//! surface that never existed.
//!
//! # Scope: one region, or several
//!
//! [`apply`] takes a single `&mut ChunkRegion` and refuses anything outside
//! it. [`route::apply_routed`] (ticket 032) is the entry point for an edit
//! that spans several region files — a building near a region corner touches
//! up to four — and it is built out of [`plan`] and [`apply`] rather than
//! around them. Neither of them saves: `apply` deliberately does **not** call
//! `ChunkRegion::save`, because a function that both edits and writes to disk
//! can't be tested without a disk, and because 032's rollback is "throw the
//! in-memory region away", which only works while nothing has been written.
//!
//! [`session::WriteSession`] (ticket 033) is the step past that — the only
//! thing here that touches the user's save. It holds the world's
//! `session.lock` so Minecraft can't have it, copies each region file aside
//! before this session's first write to it, and saves. Its
//! [`plan`](session::WriteSession::plan) is the dry run.
//!
//! # Coordinates
//!
//! Minecraft world coordinates, everywhere, with no exceptions. The
//! `bevy.z = -mc.z` flip belongs to the rendering boundary
//! ([`crate::world::mesh`], [`crate::selection::gizmo`]) and must not leak in
//! here — that flip is precisely how mirrored buildings happen. The build
//! limits are [`crate::selection`]'s, the same ones ticket 020's face keys
//! clamp to.

use std::collections::{BTreeMap, BTreeSet};

use bevy::math::IVec3;
use mc_anvil::chunkregion::ChunkRegion;
use mc_anvil::heightmap::HeightmapClass;
use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;
use mc_anvil::BlockState as AnvilBlockState;

use crate::blueprint::BlockState;
use crate::chunk_pipeline::local_chunk_index;
use crate::region_cache::chunk_to_region_coord;
use crate::selection::{SelectionBounds, WORLD_MAX_Y, WORLD_MIN_Y};
use crate::world::SECTION_SIZE;

pub mod route;
pub mod session;
pub use route::{apply_routed, plan_routed, route, RegionSource, RegionUnavailable};
pub use session::{
    RegionWrite, WriteError, WritePlan, WriteSafety, WriteSession, WriteSummary, BACKUP_DIR,
};

/// Blocks across a region file, both horizontal axes: 32 chunks of 16.
const REGION_WIDTH_IN_BLOCKS: i32 = REGION_WIDTH_IN_CHUNKS as i32 * SECTION_SIZE as i32;

/// The chunk-root tag saying how finished a chunk is. Only `minecraft:full`
/// chunks are safe to write into; anything else may still be visited by the
/// generator, which would overwrite what we put there.
const STATUS: &str = "Status";
const STATUS_FULL: &str = "minecraft:full";

/// The chunk-root tag naming the Minecraft data version the chunk was written
/// by. Block names and properties are not stable across versions.
const DATA_VERSION: &str = "DataVersion";

// -------------------------------------------------------------------------------------------------
// ---- data version compatibility -----------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The `DataVersion`s at which block names or properties migrated, sorted
/// ascending. Each entry opens a new *compatibility band*: block states
/// written at or after it may not mean the same thing as block states written
/// before it.
///
/// ## Why a table and not `edit == save`
///
/// Both versions in the comparison are outside the user's control. A
/// blueprint carries whatever version the Minecraft that exported it wrote,
/// and a save is a *patchwork* — a world only rewrites the chunks it actually
/// loads, so one played across a few updates holds chunks at several versions
/// side by side. Ticket 069 measured a real save at 9327 chunks on 4438 and
/// 9215 on 4903, interleaved region by region. Exact equality refused roughly
/// half of it, in patches, and bought nothing: 4438 and 4903 are both 1.21-era
/// and rename no blocks. What the check is actually defending against is a
/// *migration* between the two versions, which is what this table names.
///
/// ## What goes in it
///
/// The first `DataVersion` of each Minecraft release that renamed or
/// restructured blocks — release granularity rather than snapshot, which is
/// coarser (it refuses some pairs that would in fact have been fine) and
/// therefore errs toward refusing rather than corrupting.
///
/// Versions above the last entry are all one band. **When a new Minecraft
/// release renames blocks, add its first `DataVersion` here** — until then a
/// post-1.21 chunk and a 1.21 blueprint are treated as compatible, which is
/// the permissive direction, and the reason this table has to be maintained
/// rather than inferred.
const BLOCK_FORMAT_BOUNDARIES: &[i32] = &[
    1519, // 1.13, The Flattening: numeric ids and metadata become names and properties.
    1952, // 1.14: `sign` -> `oak_sign`, `wall_sign` -> `oak_wall_sign`, and friends.
    2566, // 1.16: the nether rewrite's block set.
    2724, // 1.17: `grass_path` -> `dirt_path`, `cauldron` splits by contents.
    2860, // 1.18: the world height change moves every section index.
    3105, // 1.19: the deep dark's block set.
    3463, // 1.20, and 3698 within it renamed `grass` -> `short_grass`.
    3953, // 1.21.
];

/// Which band of [`BLOCK_FORMAT_BOUNDARIES`] a version sits in — the count of
/// boundaries at or below it. A version exactly *on* a boundary belongs to the
/// newer band: the boundary is the first version that speaks the new spelling.
fn data_version_band(version: i32) -> usize {
    BLOCK_FORMAT_BOUNDARIES
        .iter()
        .filter(|&&boundary| version >= boundary)
        .count()
}

/// Whether block states spelled for one `DataVersion` can be written into a
/// chunk at another: true when no block migration ([`BLOCK_FORMAT_BOUNDARIES`])
/// lies between them.
///
/// Symmetric, and reflexive — equal versions are always compatible, whatever
/// the table says.
pub fn data_versions_compatible(a: i32, b: i32) -> bool {
    data_version_band(a) == data_version_band(b)
}

// -------------------------------------------------------------------------------------------------
// ---- the edit -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One block to write, at Minecraft world coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockEdit {
    pub at: IVec3,
    pub state: BlockState,
}

/// A set of block writes to apply as one transaction.
///
/// **Air is a block.** An edit carrying `minecraft:air` writes air there.
/// Whether a building's declared-empty space should clear the terrain in it or
/// leave it standing is the blueprint layer's question (roadmap B2/E4), and
/// filtering air here would make "place a building with a hollow interior"
/// inexpressible.
///
/// **Last write wins** for a position written twice, in the order the edits
/// were pushed. That's `set_blocks`' existing behaviour within a section
/// rather than a rule added here, but a caller stamping two overlapping
/// blueprints needs it stated.
#[derive(Debug, Clone, Default)]
pub struct WorldEdit {
    edits: Vec<BlockEdit>,
    /// The `DataVersion` these block states came from, if the source knew one
    /// — a [`crate::blueprint::Blueprint`] does. Checked against the target
    /// chunk's; see [`EditRefusal::DataVersionMismatch`].
    data_version: Option<i32>,
}

impl WorldEdit {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares which Minecraft version these block states are spelled for.
    ///
    /// Leave it unset for an edit built from block names in code — those are
    /// version-independent enough to be the caller's problem, and a made-up
    /// version number would refuse edits for no reason.
    pub fn with_data_version(mut self, data_version: i32) -> Self {
        self.data_version = Some(data_version);
        self
    }

    pub fn set(&mut self, at: IVec3, state: BlockState) -> &mut Self {
        self.edits.push(BlockEdit { at, state });
        self
    }

    /// Every block this edit would write, in the order they were added.
    pub fn edits(&self) -> &[BlockEdit] {
        &self.edits
    }

    /// Which Minecraft version these block states are spelled for, if the
    /// source knew. [`route`] carries it onto every sub-edit it produces.
    pub fn data_version(&self) -> Option<i32> {
        self.data_version
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// Fills every block in `bounds` with `state` — the same block at every
    /// position. Shared by the viewer's paint/fill command (roadmap W8,
    /// ticket 035) and, eventually, terraforming (H1): both are the same
    /// write path with a different source of block changes, so the
    /// construction of a "one block everywhere" edit belongs here rather
    /// than in either caller.
    pub fn fill(bounds: SelectionBounds, state: BlockState) -> Self {
        bounds
            .iter_blocks()
            .map(|at| BlockEdit { at, state: state.clone() })
            .collect()
    }
}

impl FromIterator<BlockEdit> for WorldEdit {
    fn from_iter<T: IntoIterator<Item = BlockEdit>>(iter: T) -> Self {
        Self {
            edits: iter.into_iter().collect(),
            data_version: None,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// ---- policy -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A caller-supplied block taxonomy for [`HeightmapPolicy::Recompute`]: what
/// each block state counts as when Minecraft's per-column surface heights are
/// rebuilt.
///
/// A `fn` pointer rather than a closure so [`EditPolicy`] stays `Copy` and
/// cheap to pass around — and because the real implementation is a table built
/// once against [`crate::world::BlockRegistry`], not a capture.
pub type HeightmapClassifier = fn(&AnvilBlockState) -> HeightmapClass;

/// What to do with an edited chunk's `Heightmaps` — the per-column surface
/// heights Minecraft uses for sky light, mob spawning and weather, which go
/// stale the moment blocks move.
// Not `PartialEq`: `Recompute` carries a `fn` pointer, and comparing those
// doesn't produce a meaningful answer (the same function can have different
// addresses across codegen units, and different ones can share an address).
#[derive(Debug, Clone, Copy)]
pub enum HeightmapPolicy {
    /// Delete the compound and let Minecraft rebuild it on load — the same
    /// "hand it back to the game" reasoning as lighting, and the default.
    Delete,
    /// Rebuild them here, using the caller's block taxonomy.
    ///
    /// Costs a top-down walk of the chunk's sections and needs a real
    /// classifier. Worth taking only if the game turns out not to prime
    /// deleted heightmaps correctly — the manual check in `../todo.md`.
    Recompute(HeightmapClassifier),
    /// Leave them exactly as they are. For a caller that knows its edit can't
    /// move a surface, and for tests that want to see staleness.
    Leave,
}

/// How strict an edit is, and what it records.
#[derive(Debug, Clone, Copy)]
pub struct EditPolicy {
    pub heightmaps: HeightmapPolicy,
    /// Refuse chunks whose `Status` isn't `minecraft:full`. On by default:
    /// writing into a partially generated chunk invites the generator to
    /// overwrite it later.
    pub require_full_status: bool,
    /// Refuse when the edit's `DataVersion` and the chunk's are *incompatible*
    /// — a block migration lies between them ([`data_versions_compatible`]).
    /// On by default; the override exists so a UI can offer "do it anyway"
    /// rather than leaving the user stuck.
    pub enforce_data_version: bool,
    /// Record what each written position held *before* the edit.
    ///
    /// This is the as-built baseline (roadmap I1), undo (D3) and demolish's
    /// terrain restore (E5) — one record, three consumers, and knowable only
    /// while the edit is being applied. Off by default because it costs a read
    /// per position; a caller that might ever want to undo should turn it on,
    /// because afterwards the data is gone from the save for good.
    pub capture_replaced: bool,
    /// Allow a transaction whose target regions already hold unsaved changes.
    ///
    /// Read by the routed entry points ([`route::apply_routed`]) only — a
    /// single-region [`apply`] has no rollback to protect. Off by default
    /// because that rollback discards the whole region: an earlier
    /// transaction's unsaved edit would go with it. Turning it on says the
    /// caller is batching deliberately and accepts that a failure rolls back
    /// further than it caused.
    pub allow_dirty_regions: bool,
}

impl Default for EditPolicy {
    fn default() -> Self {
        Self {
            heightmaps: HeightmapPolicy::Delete,
            require_full_status: true,
            enforce_data_version: true,
            capture_replaced: false,
            allow_dirty_regions: false,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// ---- refusals -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Why an edit was refused, in full, before anything was written.
///
/// Every variant names the coordinate or chunk that caused it, because the
/// caller has to be able to tell the user *where* — a placement refused with
/// no position is a placement nobody can fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditRefusal {
    /// The edit has no blocks in it. Refused rather than silently succeeding:
    /// an empty placement is a bug upstream, and reporting "wrote 0 blocks" as
    /// success hides it.
    Empty,
    /// Y is outside the world's build range.
    OutsideBuildLimits { at: IVec3 },
    /// The position belongs to a different region file than the one being
    /// edited — from the single-region [`apply`], which refuses it rather than
    /// wrapping it into the wrong place the way region-local coordinates
    /// silently would. [`route::apply_routed`] is what makes such an edit
    /// legal, so this is not reachable through it.
    OutsideRegion { at: IVec3, region: (i32, i32) },
    /// The save has no region file there at all: ungenerated terrain, one
    /// scale up from [`EditRefusal::ChunkNotGenerated`]. A building placed past
    /// the edge of explored terrain hits this one first.
    RegionNotGenerated { region: (i32, i32) },
    /// The region file exists and won't load — truncated, corrupt, an
    /// unsupported compression. Not the same news as ungenerated terrain, and
    /// reported separately so the user can tell which they have.
    RegionUnreadable { region: (i32, i32), reason: String },
    /// A target region already holds unsaved changes from an earlier
    /// transaction (ticket 032). Refused because this transaction's rollback
    /// discards the whole region and would take those with it; save first, or
    /// set [`EditPolicy::allow_dirty_regions`].
    RegionHasUnsavedChanges { region: (i32, i32) },
    /// The chunk isn't in the region file: ungenerated terrain. Iteration 1
    /// does not generate terrain, so this is a refusal and not a prompt.
    ChunkNotGenerated { chunk: (i32, i32) },
    /// The chunk exists but isn't finished generating.
    StatusNotFull { chunk: (i32, i32), status: String },
    /// The edit's blocks were written for a Minecraft version that spells
    /// blocks differently than the chunk's does — a block migration
    /// ([`BLOCK_FORMAT_BOUNDARIES`]) sits between the two. A refusal by
    /// default rather than a warning: writing a name the target version has
    /// never heard of leaves the save holding blocks Minecraft will not load.
    ///
    /// Merely *different* versions are not this. Ticket 069: a save played
    /// across updates is a patchwork of versions, and only the ones a
    /// migration separates are a problem.
    DataVersionMismatch {
        chunk: (i32, i32),
        save: i32,
        edit: i32,
    },
    /// No section covers this Y in this chunk. Writes never create sections
    /// (`mc_anvil` ticket 012), and a generated 1.18+ chunk carries all 24, so
    /// this is a backstop rather than a common path.
    SectionMissing { chunk: (i32, i32), section_y: i32 },
    /// `set_blocks` refused the batch after preflight accepted it — a
    /// malformed section, an empty palette, something this layer's checks
    /// don't model. Carries the underlying message rather than the error type,
    /// which isn't `Clone`.
    Rejected { reason: String },
}

impl std::fmt::Display for EditRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditRefusal::Empty => write!(f, "the edit is empty"),
            EditRefusal::OutsideBuildLimits { at } => write!(
                f,
                "({}, {}, {}) is outside the build limits ({WORLD_MIN_Y}..={WORLD_MAX_Y})",
                at.x, at.y, at.z
            ),
            EditRefusal::OutsideRegion { at, region } => write!(
                f,
                "({}, {}, {}) is not in region ({}, {})",
                at.x, at.y, at.z, region.0, region.1
            ),
            EditRefusal::RegionNotGenerated { region } => write!(
                f,
                "region ({}, {}) has not been generated yet",
                region.0, region.1
            ),
            EditRefusal::RegionUnreadable { region, reason } => write!(
                f,
                "region ({}, {}) could not be read: {reason}",
                region.0, region.1
            ),
            EditRefusal::RegionHasUnsavedChanges { region } => write!(
                f,
                "region ({}, {}) has unsaved changes; save them before editing it again",
                region.0, region.1
            ),
            EditRefusal::ChunkNotGenerated { chunk } => write!(
                f,
                "chunk ({}, {}) has not been generated yet",
                chunk.0, chunk.1
            ),
            EditRefusal::StatusNotFull { chunk, status } => write!(
                f,
                "chunk ({}, {}) is not fully generated (Status = {status})",
                chunk.0, chunk.1
            ),
            EditRefusal::DataVersionMismatch { chunk, save, edit } => write!(
                f,
                "chunk ({}, {}) is DataVersion {save}, the edit is {edit}, and blocks were renamed in between",
                chunk.0, chunk.1
            ),
            EditRefusal::SectionMissing { chunk, section_y } => write!(
                f,
                "chunk ({}, {}) has no section at Y = {section_y}",
                chunk.0, chunk.1
            ),
            EditRefusal::Rejected { reason } => write!(f, "the write was rejected: {reason}"),
        }
    }
}

impl std::error::Error for EditRefusal {}

// -------------------------------------------------------------------------------------------------
// ---- the report ---------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// What an edit did, or (from [`plan`]) what it would do.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EditReport {
    /// Positions written. Counts a position written twice once — it's one
    /// block in the world either way.
    pub blocks_written: usize,
    /// The chunks the edit lands in, ascending.
    pub chunks: Vec<(i32, i32)>,
    /// The region files the edit lands in, ascending — one from [`plan`], up
    /// to four from [`route::plan_routed`], which is the number a building
    /// placed on a region corner can reach.
    pub regions: Vec<(i32, i32)>,
    /// What each written position held before, if
    /// [`EditPolicy::capture_replaced`] asked for it. Ascending by position,
    /// so two runs of the same edit produce the same record.
    pub replaced: Option<Vec<(IVec3, BlockState)>>,
}

// -------------------------------------------------------------------------------------------------
// ---- coordinates --------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Where a Minecraft world block coordinate lives: its chunk, and its
/// coordinates *relative to the region file*, which is how
/// [`ChunkRegion::set_block`] and `get_block` address blocks.
///
/// `div_euclid`/`rem_euclid` throughout, never `/` and `%`: at x = -1 the
/// chunk is -1 and the region-local x is 511, whereas truncating division
/// gives chunk 0 and -1. Every negative-coordinate bug in a world editor
/// starts here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockAddress {
    pub chunk: (i32, i32),
    pub region: (i32, i32),
    /// 0..512 within the region file.
    pub local_x: usize,
    pub local_z: usize,
    /// World Y, unchanged — `set_block` takes a real world Y.
    pub y: i32,
}

pub fn address_of(at: IVec3) -> BlockAddress {
    let chunk = (
        at.x.div_euclid(SECTION_SIZE as i32),
        at.z.div_euclid(SECTION_SIZE as i32),
    );
    BlockAddress {
        chunk,
        region: chunk_to_region_coord(chunk),
        local_x: at.x.rem_euclid(REGION_WIDTH_IN_BLOCKS) as usize,
        local_z: at.z.rem_euclid(REGION_WIDTH_IN_BLOCKS) as usize,
        y: at.y,
    }
}

/// The `Y` tag of the section a world Y falls in, rounding towards the world
/// bottom so that -1 is section -1 rather than section 0.
fn section_y_of(y: i32) -> i32 {
    y.div_euclid(SECTION_SIZE as i32)
}

// -------------------------------------------------------------------------------------------------
// ---- planning -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Validates `edit` against `region` without touching anything, and reports
/// what it would do. The dry run, and the first half of [`apply`].
///
/// Every reason to refuse is found here, in one pass over the whole edit,
/// rather than fail-fast during the write — a half-applied edit is the thing
/// this layer exists to prevent, and `set_blocks` can only guarantee that for
/// the checks it makes itself.
pub fn plan(
    edit: &WorldEdit,
    region: &ChunkRegion,
    policy: &EditPolicy,
) -> Result<EditReport, EditRefusal> {
    if edit.is_empty() {
        return Err(EditRefusal::Empty);
    }

    let region_coord = (region.region.get_x_coord(), region.region.get_z_coord());

    // Deduped per position (last write wins) so the reported count is blocks
    // in the world rather than calls made.
    let mut positions: BTreeSet<(i32, i32, i32)> = BTreeSet::new();
    // Which section Ys each chunk needs, so the section check runs once per
    // (chunk, section) rather than once per block.
    let mut sections: BTreeMap<(i32, i32), BTreeSet<i32>> = BTreeMap::new();

    for BlockEdit { at, .. } in edit.edits() {
        if at.y < WORLD_MIN_Y || at.y > WORLD_MAX_Y {
            return Err(EditRefusal::OutsideBuildLimits { at: *at });
        }

        let address = address_of(*at);
        if address.region != region_coord {
            return Err(EditRefusal::OutsideRegion {
                at: *at,
                region: region_coord,
            });
        }

        positions.insert((at.y, at.z, at.x));
        sections
            .entry(address.chunk)
            .or_default()
            .insert(section_y_of(at.y));
    }

    for (chunk, section_ys) in &sections {
        check_chunk(*chunk, region, policy, edit.data_version)?;

        let (local_x, local_z) = local_chunk_index(*chunk, region_coord);
        for section_y in section_ys {
            // Any block in the section answers the same question, so probe the
            // chunk's own corner rather than each edited position.
            let probe_y = section_y * SECTION_SIZE as i32;
            let (block_x, block_z) = (local_x * SECTION_SIZE, local_z * SECTION_SIZE);
            if region.check_set_block(block_x, probe_y, block_z).is_err() {
                return Err(EditRefusal::SectionMissing {
                    chunk: *chunk,
                    section_y: *section_y,
                });
            }
        }
    }

    Ok(EditReport {
        blocks_written: positions.len(),
        chunks: sections.keys().copied().collect(),
        regions: vec![region_coord],
        replaced: None,
    })
}

/// The per-chunk rules: it exists, it's finished, and it's the same Minecraft
/// version the edit was written for.
///
/// Separate and taking the chunk NBT via the region rather than a whole edit,
/// so the rules can be tested against a synthetic chunk without a region file
/// anywhere.
fn check_chunk(
    chunk: (i32, i32),
    region: &ChunkRegion,
    policy: &EditPolicy,
    edit_version: Option<i32>,
) -> Result<(), EditRefusal> {
    let region_coord = (region.region.get_x_coord(), region.region.get_z_coord());
    let (local_x, local_z) = local_chunk_index(chunk, region_coord);

    let nbt = region
        .get_chunk(local_x, local_z)
        .ok_or(EditRefusal::ChunkNotGenerated { chunk })?;

    check_chunk_nbt(chunk, nbt, policy, edit_version)
}

/// [`check_chunk`]'s rules, against a chunk root compound.
fn check_chunk_nbt(
    chunk: (i32, i32),
    nbt: &rnbt::NbtField,
    policy: &EditPolicy,
    edit_version: Option<i32>,
) -> Result<(), EditRefusal> {
    if policy.require_full_status {
        // A chunk with no `Status` at all is not a finished chunk this crate
        // recognises; treat the absence as the refusal it is rather than
        // assuming the best about a file we're about to write into.
        let status = nbt.get_string(STATUS).cloned().unwrap_or_default();
        if status != STATUS_FULL {
            return Err(EditRefusal::StatusNotFull { chunk, status });
        }
    }

    if policy.enforce_data_version {
        // An edit that makes no version claim, or a chunk that carries none,
        // is not a mismatch — there's nothing to compare, and refusing on a
        // missing tag would block every edit built from block names in code.
        //
        // Ticket 069: *incompatible*, not merely different. Two versions with
        // no block migration between them spell blocks the same way, and a
        // save played across updates holds several versions at once — see
        // `BLOCK_FORMAT_BOUNDARIES`.
        match (edit_version, nbt.get_int(DATA_VERSION)) {
            (Some(edit), Some(save)) if !data_versions_compatible(edit, save) => {
                return Err(EditRefusal::DataVersionMismatch { chunk, save, edit });
            }
            _ => {}
        }
    }

    Ok(())
}

// -------------------------------------------------------------------------------------------------
// ---- applying -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Applies `edit` to `region`: [`plan`], then the blocks, then the heightmap
/// policy.
///
/// All or nothing. A refusal leaves the region untouched and undirtied — the
/// preflight runs to completion first, and `set_blocks` is itself
/// all-or-nothing, so there is no window where half an edit exists.
///
/// Does **not** save. The region is dirty afterwards
/// ([`ChunkRegion::is_dirty`]) and it is the caller's job to write it out —
/// which is W6's, along with the backup and the `session.lock` check that
/// have to happen around it.
pub fn apply(
    edit: &WorldEdit,
    region: &mut ChunkRegion,
    policy: &EditPolicy,
) -> Result<EditReport, EditRefusal> {
    let mut report = plan(edit, region, policy)?;

    if policy.capture_replaced {
        report.replaced = Some(capture_replaced(edit, region));
    }

    // One `set_blocks` for the whole edit: it groups by section internally and
    // re-packs each one once, so a 4000-block building costs one re-pack per
    // section rather than 4000.
    let states: Vec<(usize, i32, usize, AnvilBlockState)> = edit
        .edits()
        .iter()
        .map(|BlockEdit { at, state }| {
            let address = address_of(*at);
            (address.local_x, address.y, address.local_z, to_anvil(state))
        })
        .collect();

    region
        .set_blocks(states.iter().map(|(x, y, z, state)| (*x, *y, *z, state)))
        .map_err(|err| EditRefusal::Rejected {
            reason: err.to_string(),
        })?;

    // After the blocks, and once per chunk rather than once per write: a
    // recompute run mid-edit would bake a surface that never existed.
    apply_heightmap_policy(&report.chunks, region, policy)?;

    Ok(report)
}

/// Reads what every position the edit writes holds *now* — the as-built
/// baseline (roadmap I1), which stops being knowable the moment the blocks
/// change.
///
/// Ordered by position rather than by edit order so the record is stable
/// across runs, and deduped for the same reason [`plan`] dedupes: a position
/// written twice had one prior state, not two.
fn capture_replaced(edit: &WorldEdit, region: &ChunkRegion) -> Vec<(IVec3, BlockState)> {
    let positions: BTreeSet<(i32, i32, i32)> = edit
        .edits()
        .iter()
        .map(|BlockEdit { at, .. }| (at.y, at.z, at.x))
        .collect();

    positions
        .into_iter()
        .map(|(y, z, x)| {
            let at = IVec3::new(x, y, z);
            let address = address_of(at);
            // Every one of these passed the preflight, so the chunk and its
            // section exist. An unreadable palette entry still resolves to air
            // rather than failing the edit — the same call extraction makes.
            let state = region
                .get_block(address.local_x, address.y, address.local_z)
                .ok()
                .and_then(|entry| BlockState::from_palette_entry(entry).ok())
                .unwrap_or_else(BlockState::air);
            (at, state)
        })
        .collect()
}

/// Deletes, recomputes or leaves the heightmaps of every chunk the edit
/// touched.
fn apply_heightmap_policy(
    chunks: &[(i32, i32)],
    region: &mut ChunkRegion,
    policy: &EditPolicy,
) -> Result<(), EditRefusal> {
    let region_coord = (region.region.get_x_coord(), region.region.get_z_coord());

    for chunk in chunks {
        let (local_x, local_z) = local_chunk_index(*chunk, region_coord);
        let outcome = match policy.heightmaps {
            HeightmapPolicy::Delete => region.remove_heightmaps(local_x, local_z).map(|_| ()),
            HeightmapPolicy::Recompute(classify) => {
                region.recompute_heightmaps(local_x, local_z, classify)
            }
            HeightmapPolicy::Leave => Ok(()),
        };

        outcome.map_err(|err| EditRefusal::Rejected {
            reason: err.to_string(),
        })?;
    }

    Ok(())
}

/// This crate's [`BlockState`] as `mc_anvil`'s. Both keep their properties
/// sorted by key, so the conversion is a rename.
fn to_anvil(state: &BlockState) -> AnvilBlockState {
    let mut anvil = AnvilBlockState::new(state.name.clone());
    for (key, value) in &state.properties {
        anvil.set_property(key.clone(), value.clone());
    }
    anvil
}

#[cfg(test)]
mod tests;
