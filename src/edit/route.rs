//! Boundary routing and region batching (ticket 032, roadmap W5): applying one
//! edit to the up-to-four region files it lands in.
//!
//! [`super::apply`] deals with a single region and refuses everything else
//! ([`EditRefusal::OutsideRegion`]) — deliberately, because region-local
//! coordinates are `rem_euclid(512)` and a position in the region next door
//! would otherwise wrap silently into the region being edited, putting a
//! building 512 blocks from where it was asked for. This module is what makes
//! that edit legal: split it by region file, and apply each piece to the right
//! one.
//!
//! # Why two phases
//!
//! `set_blocks` is all-or-nothing per call, and [`super::apply`] is
//! all-or-nothing per region. Neither of those is all-or-nothing across four
//! region files, and three quarters of a building is worse than no building:
//! so every region is planned before any region is applied, and a failure
//! during the apply phase rolls the already-applied regions back by discarding
//! them ([`RegionSource::discard`]). Nothing here saves — that's W6 — so
//! throwing the in-memory region away is a complete and exact undo.
//!
//! # What routing does *not* touch
//!
//! The already-decoded columns in [`crate::DecodedWorld`] and their meshes go
//! stale the moment an edit lands. That's W7's job, through 005-f's existing
//! re-mesh queue, and [`EditReport::chunks`] is what it needs. The
//! [`RegionCache`] itself needs no invalidating: it *holds* the mutated region,
//! so every later read already sees post-edit blocks.

use std::collections::BTreeMap;

use mc_anvil::chunkregion::ChunkRegion;

use super::{apply, plan, ChunkBorders, EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::edit::address_of;
use crate::region_cache::RegionCache;

// -------------------------------------------------------------------------------------------------
// ---- where regions come from --------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Why a region file couldn't be handed over.
///
/// Two cases, kept apart because they mean different things to the user:
/// terrain that was never generated is "you can't build there", a region that
/// won't load is "something is wrong with this save".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionUnavailable {
    NotGenerated,
    Unreadable(String),
}

/// Where the routed edit gets its region files.
///
/// A trait rather than `&mut RegionCache` for two reasons: the routing rules
/// can then be tested against a map of fixture regions with no save anywhere
/// (the same instinct ticket 031 followed when it built a synthetic region
/// instead of copying the developer's world), and W6 can wrap this with the
/// backup and `session.lock` layer without teaching the cache about either.
pub trait RegionSource {
    /// The region at `coord`, loading it if that's what this source does.
    fn region_mut(&mut self, coord: (i32, i32)) -> Result<&mut ChunkRegion, RegionUnavailable>;

    /// Throws the in-memory region away, unsaved changes and all, so the next
    /// access starts again from what's on disk. The rollback path; see the
    /// module docs.
    fn discard(&mut self, coord: (i32, i32));

    /// Which regions currently hold unsaved changes, in no particular order.
    ///
    /// On the trait rather than only on [`RegionCache`] so that ticket 033's
    /// [`WriteSession::flush`](super::session::WriteSession::flush) — "save
    /// everything that's dirty" — shares one save path with `commit` instead
    /// of growing a second one for the concrete cache.
    fn dirty_regions(&self) -> Vec<(i32, i32)>;
}

impl RegionSource for RegionCache {
    fn region_mut(&mut self, coord: (i32, i32)) -> Result<&mut ChunkRegion, RegionUnavailable> {
        // Asked first so ungenerated terrain reports as itself: `get_or_load`
        // answers `PathNotFoundError` both for a region the save never had and
        // for one it failed to read, and those are not the same news.
        if !self.has_region(coord) {
            return Err(RegionUnavailable::NotGenerated);
        }
        self.get_or_load_mut(coord)
            .map_err(|err| RegionUnavailable::Unreadable(err.to_string()))
    }

    fn discard(&mut self, coord: (i32, i32)) {
        // Fully qualified: the inherent method and this one share a name, and
        // spelling out which is being called keeps that from being a puzzle.
        RegionCache::discard(self, coord);
    }

    fn dirty_regions(&self) -> Vec<(i32, i32)> {
        // Same name-sharing as `discard` above; the inherent one returns an
        // iterator over the cache's own map.
        RegionCache::dirty_regions(self).collect()
    }
}

// -------------------------------------------------------------------------------------------------
// ---- the split ----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Splits an edit into one sub-edit per region file.
///
/// Each sub-edit carries the parent's `DataVersion` claim and keeps the
/// parent's relative order, so **last write wins** survives the split — two
/// edits at the same position are in the same region by construction, so the
/// rule never has to reason across regions.
///
/// Validates nothing: every refusal stays in [`plan`], which now runs once per
/// region. A position outside the build limits is a property of the position
/// and not of the routing, and checking it twice would mean two places to keep
/// in step.
pub fn route(edit: &WorldEdit) -> BTreeMap<(i32, i32), WorldEdit> {
    let mut by_region: BTreeMap<(i32, i32), WorldEdit> = BTreeMap::new();

    for block in edit.edits() {
        let region = address_of(block.at).region;
        by_region
            .entry(region)
            .or_insert_with(|| match edit.data_version() {
                Some(version) => WorldEdit::new().with_data_version(version),
                None => WorldEdit::new(),
            })
            .set(block.at, block.state.clone());
    }

    by_region
}

// -------------------------------------------------------------------------------------------------
// ---- planning and applying across regions -------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Validates a whole multi-region edit without touching anything, and reports
/// what it would do. The dry run, and the first phase of [`apply_routed`].
pub fn plan_routed(
    edit: &WorldEdit,
    source: &mut impl RegionSource,
    policy: &EditPolicy,
) -> Result<EditReport, EditRefusal> {
    let routed = route(edit);
    if routed.is_empty() {
        return Err(EditRefusal::Empty);
    }
    plan_each(&routed, source, policy)
}

/// Applies `edit` to every region file it lands in, or to none of them.
///
/// Every region is planned first ([`plan_routed`]); only if all of them pass
/// does any of them get written. If the write phase fails anyway — `set_blocks`
/// can reject a batch the preflight accepted — every region already applied is
/// discarded from `source`, which is a complete rollback precisely because
/// nothing here saves.
///
/// Does **not** save, for the same reason [`super::apply`] doesn't: backups,
/// `session.lock` and atomicity are W6, and the regions are left dirty for it.
pub fn apply_routed(
    edit: &WorldEdit,
    source: &mut impl RegionSource,
    policy: &EditPolicy,
) -> Result<EditReport, EditRefusal> {
    let routed = route(edit);
    if routed.is_empty() {
        return Err(EditRefusal::Empty);
    }
    plan_each(&routed, source, policy)?;

    let mut applied: Vec<((i32, i32), EditReport)> = Vec::new();
    for (region_coord, sub_edit) in &routed {
        let region = fetch(source, *region_coord)?;
        match apply(sub_edit, region, policy) {
            Ok(report) => applied.push((*region_coord, report)),
            Err(refusal) => {
                // Phase 1 said yes and phase 2 said no. Undo the regions that
                // did apply by throwing them away: they were only ever changed
                // in memory.
                for (rolled_back, _) in &applied {
                    source.discard(*rolled_back);
                }
                return Err(refusal);
            }
        }
    }

    Ok(merge(applied))
}

/// Phase 1: every region loaded, every rule checked, nothing mutated.
fn plan_each(
    routed: &BTreeMap<(i32, i32), WorldEdit>,
    source: &mut impl RegionSource,
    policy: &EditPolicy,
) -> Result<EditReport, EditRefusal> {
    let mut reports: Vec<((i32, i32), EditReport)> = Vec::new();

    for (region_coord, sub_edit) in routed {
        let region = fetch(source, *region_coord)?;

        // Unsaved changes from an *earlier* transaction would be destroyed by
        // this one's rollback, which discards the whole region. Refusing here
        // is what makes that rollback honest rather than approximately correct;
        // the contract — one transaction at a time, saved before the next — is
        // the one W6 implements anyway.
        if !policy.allow_dirty_regions && region.is_dirty() {
            return Err(EditRefusal::RegionHasUnsavedChanges {
                region: *region_coord,
            });
        }

        reports.push((*region_coord, plan(sub_edit, region, policy)?));
    }

    Ok(merge(reports))
}

/// One region out of the source, with the failure translated into the refusal
/// the caller can show a user.
fn fetch(
    source: &mut impl RegionSource,
    coord: (i32, i32),
) -> Result<&mut ChunkRegion, EditRefusal> {
    source
        .region_mut(coord)
        .map_err(|unavailable| refusal_for(coord, unavailable))
}

/// The refusal a [`RegionUnavailable`] means for a given region — shared with
/// ticket 033's write session, which hits the same two cases when it goes back
/// for a region to save.
pub(crate) fn refusal_for(coord: (i32, i32), unavailable: RegionUnavailable) -> EditRefusal {
    match unavailable {
        RegionUnavailable::NotGenerated => EditRefusal::RegionNotGenerated { region: coord },
        RegionUnavailable::Unreadable(reason) => EditRefusal::RegionUnreadable {
            region: coord,
            reason,
        },
    }
}

/// Folds the per-region reports into one.
///
/// `blocks_written` sums without deduping — a position belongs to exactly one
/// region, so there is nothing to collide. `chunks` and `replaced` are sorted
/// after concatenating: each region's own list is ascending, but region-major
/// order isn't globally ascending once a boundary is crossed on Z (chunk
/// `(5, 32)` sorts before `(7, 0)` and they live in different files).
fn merge(reports: Vec<((i32, i32), EditReport)>) -> EditReport {
    let mut merged = EditReport::default();
    // `chunks` and `borders` are parallel lists, so they're sorted as pairs
    // (ticket 123) — a chunk belongs to exactly one region, so no key
    // repeats across the concatenation.
    let mut chunks: Vec<((i32, i32), ChunkBorders)> = Vec::new();

    for (region_coord, report) in reports {
        merged.blocks_written += report.blocks_written;
        merged.regions.push(region_coord);
        chunks.extend(report.chunks.into_iter().zip(report.borders));
        if let Some(replaced) = report.replaced {
            merged.replaced.get_or_insert_with(Vec::new).extend(replaced);
        }
    }

    merged.regions.sort_unstable();
    chunks.sort_unstable_by_key(|(chunk, _)| *chunk);
    (merged.chunks, merged.borders) = chunks.into_iter().unzip();
    if let Some(replaced) = merged.replaced.as_mut() {
        replaced.sort_unstable_by_key(|(at, _)| (at.y, at.z, at.x));
    }

    merged
}
