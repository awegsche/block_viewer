//! The journal (ticket 044, roadmap D3): every placement and demolition as
//! an appended entry, each carrying the as-built baseline (roadmap I1) —
//! what blocks the action wrote, and what was there immediately before.
//!
//! One record, three consumers here (a fourth, the damage mechanic, waits
//! for roadmap group I):
//!
//! - **Undo** ([`Journal::undo_last`]) — pop the entry, write `previous`
//!   back, and reverse whatever it did to [`City`].
//! - **Demolish's terrain restore** (roadmap E5) — a demolition's own
//!   baseline records what the building's blocks were (`previous`) so the
//!   terrain that had stood there (`written`, from the *placement*'s own
//!   baseline) can be put back. E5 is what will call [`Journal::record_demolition`]
//!   once it exists; this ticket only builds the record and its reversal.
//! - **Reconciliation** ([`reconcile`]) — recompute what the world *should*
//!   hold, from every currently-placed building's own placement baseline,
//!   and diff it against what a [`RegionSource`] actually has. [`repair_edit`]
//!   is the write half: an edit that writes every mismatch back, left for
//!   the caller to run through the write path like any other edit.
//!
//! ## Where the baseline itself comes from
//!
//! Ticket 031 already built this: [`crate::edit::EditPolicy::capture_replaced`]
//! records what every written position held *before* an edit, in
//! [`EditReport::replaced`]. [`Baseline::capture`] is the seam between that
//! and this module — it reads `replaced` as `previous` and the edit's own
//! [`crate::edit::WorldEdit::edits`] as `written`, deduped and sorted the
//! same way so the two line up position-for-position. Nothing here computes
//! a baseline itself; it only records and replays one.
//!
//! ## A real caller, as of ticket 048
//!
//! [`Journal::record_placement`] is called by `city::commit::poll_commit`
//! (roadmap E4) on every successful placement — the first gameplay caller
//! this module has had; until then it was only proven by the app lifecycle
//! (loaded and saved on every real save, per ticket 043's own precedent) and
//! by this module's tests. [`record_demolition`](Journal::record_demolition)
//! still waits on E5, and [`Journal::undo_last`]/[`reconcile`] still wait on
//! whatever UI eventually calls them (G2, roadmap I).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::IVec3;
use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::blueprint::{BlockState, Rotation};
use crate::edit::route::refusal_for;
use crate::edit::{BlockEdit, EditReport, RegionSource, WorldEdit};

use super::state::{BuildingId, City, PlacedBuilding, PlacementError};

// -------------------------------------------------------------------------------------------------
// ---- the as-built baseline (roadmap I1) ----------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// What one edit wrote into the world, and what was there immediately before
/// it, at every position it touched.
///
/// `written` and `previous` are both ascending by position (`(y, z, x)`, the
/// same order [`EditReport::replaced`] already sorts to) so the two line up
/// index-for-index — position `written[i].0 == previous[i].0` for every `i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Baseline {
    /// What the edit wrote — replaying this onto the world reproduces the
    /// as-built state, independent of anything the player did afterwards.
    pub written: Vec<(IVec3, BlockState)>,
    /// What every one of those same positions held immediately before the
    /// edit. Undo and reconciliation's rebuild path both read this half.
    pub previous: Vec<(IVec3, BlockState)>,
    /// The `DataVersion` the edit's blocks were spelled for, if the source
    /// made a claim ([`WorldEdit::data_version`]) — carried onto the record
    /// for the same reason roadmap I2's future scan will need it: block
    /// names and properties aren't stable across versions.
    pub data_version: Option<i32>,
}

impl Baseline {
    /// Builds a baseline from an edit and the report [`crate::edit::apply`]
    /// or [`crate::edit::route::apply_routed`] returned for it.
    ///
    /// `None` if `report.replaced` is `None` — the edit was applied with
    /// [`crate::edit::EditPolicy::capture_replaced`] off, so there is no
    /// record of what it overwrote, and a baseline that can't say what was
    /// there before isn't a baseline, only half of one. A caller that might
    /// ever journal, undo or reconcile an edit has to turn that policy on.
    ///
    /// Called by `city::commit::poll_commit` (ticket 048, roadmap E4) on
    /// every successful placement — `EditPolicy::capture_replaced` is on
    /// specifically so this never comes back `None` there.
    pub fn capture(edit: &WorldEdit, report: &EditReport) -> Option<Baseline> {
        let previous = report.replaced.clone()?;

        // Deduped and sorted the same way `capture_replaced` sorts
        // `previous` — last write wins for a position written twice, and the
        // ordering is what makes the two vectors line up positionally
        // without either side searching the other.
        let mut written: BTreeMap<(i32, i32, i32), BlockState> = BTreeMap::new();
        for BlockEdit { at, state } in edit.edits() {
            written.insert((at.y, at.z, at.x), state.clone());
        }
        let written = written
            .into_iter()
            .map(|((y, z, x), state)| (IVec3::new(x, y, z), state))
            .collect();

        Some(Baseline { written, previous, data_version: edit.data_version() })
    }

    /// The edit that restores `previous` — undo's world half, and the same
    /// edit a demolition's own baseline hands E5 for its terrain restore.
    /// Left uncommitted, like every other edit this crate builds: the caller
    /// runs it through the write path.
    fn restore_edit(&self) -> WorldEdit {
        self.previous
            .iter()
            .cloned()
            .map(|(at, state)| BlockEdit { at, state })
            .collect()
    }
}

// -------------------------------------------------------------------------------------------------
// ---- the journal entry ---------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One appended record: a building placed, or demolished, plus the baseline
/// (roadmap I1) the action captured.
///
/// `placement` is a full snapshot, not a lookup key — [`City::insert_loaded`]
/// needs the whole thing to put a demolished building back, and by the time
/// undo runs, [`City`] itself may no longer have it (that's exactly what
/// undoing a placement removes).
#[derive(Debug, Clone)]
pub enum JournalEntry {
    /// `baseline.written` is the building's own blocks; `baseline.previous`
    /// is the terrain that stood there before it was placed.
    Placed {
        building: BuildingId,
        placement: PlacedBuilding,
        baseline: Baseline,
    },
    /// `baseline.written` is the terrain restored in the building's place;
    /// `baseline.previous` is the building's own blocks, read off the world
    /// at demolition time — not re-derived from the blueprint, so a building
    /// damaged before it was demolished demolishes (and undoes) as what it
    /// actually was, not as though it had never been touched.
    Demolished {
        building: BuildingId,
        placement: PlacedBuilding,
        baseline: Baseline,
    },
}

impl JournalEntry {
    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn building(&self) -> BuildingId {
        match self {
            JournalEntry::Placed { building, .. } | JournalEntry::Demolished { building, .. } => *building,
        }
    }

    /// Used internally by [`Journal::undo_last`]; `pub` because a future
    /// caller inspecting `Journal::entries()` (a city panel showing recent
    /// activity, say) needs the same accessor.
    pub fn baseline(&self) -> &Baseline {
        match self {
            JournalEntry::Placed { baseline, .. } | JournalEntry::Demolished { baseline, .. } => baseline,
        }
    }

    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn placement(&self) -> &PlacedBuilding {
        match self {
            JournalEntry::Placed { placement, .. } | JournalEntry::Demolished { placement, .. } => placement,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// ---- the journal itself ---------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The append-only log: every placement and demolition, oldest first.
#[derive(Resource, Default, Debug)]
pub struct Journal {
    entries: Vec<JournalEntry>,
}

impl Journal {
    /// Appends a placement entry. `placement` is the same [`PlacedBuilding`]
    /// [`City::place_building`] just inserted under `building`, built from
    /// the identical fields rather than read back out of [`City`] — a
    /// caller that constructs both from one set of values (as
    /// `city::commit::try_commit_placement` does) can't let them drift
    /// apart either way.
    ///
    /// Called by `city::commit::poll_commit` (ticket 048, roadmap E4) once a
    /// placement's write has actually succeeded.
    pub fn record_placement(&mut self, building: BuildingId, placement: PlacedBuilding, baseline: Baseline) {
        self.entries.push(JournalEntry::Placed { building, placement, baseline });
    }

    /// Appends a demolition entry. `placement` is what [`City::remove_building`]
    /// just returned — the building as it stood in city state the instant
    /// before it was removed.
    #[allow(dead_code)] // no caller yet — E5 (roadmap), see the module docs
    pub fn record_demolition(&mut self, building: BuildingId, placement: PlacedBuilding, baseline: Baseline) {
        self.entries.push(JournalEntry::Demolished { building, placement, baseline });
    }

    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The most recent *placement* baseline recorded for `building` — the
    /// as-built record [`reconcile`] reads, not a re-derivation from the
    /// blueprint (see the module docs on why demolition keeps its own
    /// separate baseline rather than reusing this). `None` if `building` was
    /// never placed through this journal — an older save, or a building
    /// whose record predates this ticket.
    #[allow(dead_code)] // no caller yet — I2 (roadmap), see the module docs
    pub fn placement_baseline(&self, building: BuildingId) -> Option<&Baseline> {
        self.entries.iter().rev().find_map(|entry| match entry {
            JournalEntry::Placed { building: id, baseline, .. } if *id == building => Some(baseline),
            _ => None,
        })
    }

    /// Undoes the most recent entry: pops it, applies its [`City`]-side
    /// reversal, and returns the world half — an edit that restores
    /// `baseline.previous` — for the caller to commit through the write path
    /// exactly like any other edit ([`crate::edit::route::apply_routed`]/
    /// [`crate::edit::session::WriteSession::commit`]).
    ///
    /// All-or-nothing against the journal and [`City`]: on
    /// [`UndoError::Occupied`] neither is touched, so a failed undo can be
    /// retried once the conflict is cleared rather than leaving the journal
    /// one entry short of what actually happened. The world write is
    /// deliberately outside that guarantee, the same way it's outside
    /// [`crate::edit::apply`]'s: this call has already moved `city` and the
    /// journal on by the time it returns, on the assumption the caller
    /// commits `edit` next.
    #[allow(dead_code)] // no caller yet — E4/E5 (roadmap), see the module docs
    pub fn undo_last(&mut self, city: &mut City) -> Result<UndoStep, UndoError> {
        let entry = self.entries.last().ok_or(UndoError::Empty)?;
        let edit = entry.baseline().restore_edit();

        let building = match entry {
            JournalEntry::Placed { building, .. } => {
                let building = *building;
                // A missing id here means city state and the journal have
                // already drifted apart by some other bug; removing an id
                // that isn't there is a no-op rather than a panic, the same
                // contract `City::remove_building` documents.
                city.remove_building(building);
                building
            }
            JournalEntry::Demolished { building, placement, .. } => {
                let building = *building;
                city.insert_loaded(building, placement.clone()).map_err(UndoError::Occupied)?;
                building
            }
        };

        self.entries.pop();
        Ok(UndoStep { building, edit })
    }
}

/// One step of undo: the [`City`] side has already happened by the time this
/// comes back ([`Journal::undo_last`]); `edit` is what's left — the world
/// half, for the caller to run through the write path.
#[allow(dead_code)] // no caller yet — see the module docs
#[derive(Debug)]
pub struct UndoStep {
    pub building: BuildingId,
    pub edit: WorldEdit,
}

/// Why [`Journal::undo_last`] couldn't undo.
#[derive(Debug)]
pub enum UndoError {
    /// The journal is empty; there is nothing to undo.
    Empty,
    /// Undoing a demolition means putting the building back into [`City`],
    /// and the tile it stood on is occupied now — a building or road placed
    /// since. Carries [`City`]'s own refusal.
    Occupied(PlacementError),
}

impl std::fmt::Display for UndoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UndoError::Empty => write!(f, "there is nothing to undo"),
            UndoError::Occupied(err) => write!(f, "can't undo: {err}"),
        }
    }
}

impl std::error::Error for UndoError {}

// -------------------------------------------------------------------------------------------------
// ---- reconciliation --------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One position where the world disagrees with what a building's placement
/// baseline says should be there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    pub at: IVec3,
    pub expected: BlockState,
    pub actual: BlockState,
}

/// What [`reconcile`] found.
#[allow(dead_code)] // no caller yet — see the module docs
#[derive(Debug, Clone, Default)]
pub struct ReconcileReport {
    /// Positions that disagree with the baseline.
    pub mismatched: Vec<Mismatch>,
    /// Positions that couldn't be checked at all — an unreadable or
    /// ungenerated region, or a chunk/section that won't read. **Not** a
    /// mismatch: roadmap I2 draws exactly this line between "destroyed" and
    /// "unknown", and it applies here a mechanic early because the read path
    /// is the same one either way.
    pub unknown: Vec<(IVec3, String)>,
}

/// Every block a currently-placed building's own placement baseline says
/// should exist, deduped by position and ordered for stable region grouping.
///
/// An overlap between two buildings' baselines should never happen —
/// placement itself refuses overlapping footprints — so "later building in
/// iteration order wins" is a formality here, not a policy anyone should be
/// relying on.
#[allow(dead_code)] // no caller yet — see the module docs
fn expected_blocks(journal: &Journal, city: &City) -> BTreeMap<(i32, i32, i32), BlockState> {
    let mut expected = BTreeMap::new();
    for (id, _) in city.buildings() {
        let Some(baseline) = journal.placement_baseline(id) else { continue };
        for (at, state) in &baseline.written {
            expected.insert((at.y, at.z, at.x), state.clone());
        }
    }
    expected
}

/// Recomputes what the world *should* hold from `city`'s currently placed
/// buildings — each one's own placement baseline, read out of `journal` —
/// and diffs it against what `source` actually has. The rebuild path the
/// roadmap names for D3: a building nobody has touched in Minecraft
/// round-trips as zero mismatches, and a wall knocked out of one shows up as
/// exactly one.
///
/// Reads only; [`repair_edit`] is the write half. Positions are grouped by
/// region first, the same shape [`crate::edit::route`] uses for the write
/// side, so a reconciliation touching several buildings across a region
/// boundary loads each file once rather than once per building.
#[allow(dead_code)] // no caller yet — see the module docs
pub fn reconcile(journal: &Journal, city: &City, source: &mut impl RegionSource) -> ReconcileReport {
    let mut by_region: BTreeMap<(i32, i32), Vec<(IVec3, BlockState)>> = BTreeMap::new();
    for ((y, z, x), state) in expected_blocks(journal, city) {
        let at = IVec3::new(x, y, z);
        by_region.entry(crate::edit::address_of(at).region).or_default().push((at, state));
    }

    let mut report = ReconcileReport::default();
    for (region_coord, positions) in by_region {
        let region = match source.region_mut(region_coord) {
            Ok(region) => region,
            Err(unavailable) => {
                // The whole region is unreadable/ungenerated: every position
                // in it is unknown, not mismatched — reuse `edit::route`'s
                // own wording rather than inventing a second one.
                let reason = refusal_for(region_coord, unavailable).to_string();
                report.unknown.extend(positions.into_iter().map(|(at, _)| (at, reason.clone())));
                continue;
            }
        };

        for (at, expected) in positions {
            let address = crate::edit::address_of(at);
            let actual = region
                .get_block(address.local_x, address.y, address.local_z)
                .ok()
                .and_then(|entry| BlockState::from_palette_entry(entry).ok());

            match actual {
                Some(actual) if actual == expected => {}
                Some(actual) => report.mismatched.push(Mismatch { at, expected, actual }),
                None => report
                    .unknown
                    .push((at, "the chunk or section at this position could not be read".to_string())),
            }
        }
    }

    report
}

/// The write half of [`reconcile`]: an edit that writes every mismatch's
/// `expected` state back — the delta the roadmap describes. `None` when
/// there is nothing to repair, so a caller doesn't have to special-case an
/// empty [`WorldEdit`] itself.
#[allow(dead_code)] // no caller yet — I6 (roadmap), see the module docs
pub fn repair_edit(report: &ReconcileReport) -> Option<WorldEdit> {
    if report.mismatched.is_empty() {
        return None;
    }
    Some(
        report
            .mismatched
            .iter()
            .map(|m| BlockEdit { at: m.at, state: m.expected.clone() })
            .collect(),
    )
}

// -------------------------------------------------------------------------------------------------
// ---- persistence -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
//
// Same split ticket 043 used for `City`: the on-disk shape ([`SavedJournal`]
// and friends) is a plain mirror of the in-memory types, not a derive on
// them directly. `BuildingId`'s inner `u64` stays private outside
// `as_u64`/`from_u64` (043's own call), and `BlockState` is the one type that
// *does* derive `Serialize`/`Deserialize` directly (this ticket) — its two
// fields were already plain serde-able data, and a mirror type for every
// block in every building's baseline would be a lot of copying for no
// benefit.

/// The journal file's schema version. Bumped only alongside a migration
/// path — no migration exists yet, so a mismatch is refused rather than
/// guessed at, the same call ticket 043 made for [`super::persistence`].
pub const CURRENT_VERSION: u32 = 1;

const JOURNAL_FILE: &str = "citybuilder/journal.ron";

/// `<save_root>/citybuilder/journal.ron`, exposed for [`super::run`]'s own
/// log line.
pub(crate) fn journal_file_path_for_log(save_root: &Path) -> PathBuf {
    journal_file_path(save_root)
}

fn journal_file_path(save_root: &Path) -> PathBuf {
    save_root.join(JOURNAL_FILE)
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedJournal {
    version: u32,
    entries: Vec<SavedEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
enum SavedEntry {
    Placed { building: u64, placement: SavedPlacement, baseline: SavedBaseline },
    Demolished { building: u64, placement: SavedPlacement, baseline: SavedBaseline },
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedPlacement {
    definition: String,
    origin: (i32, i32, i32),
    rotation: Rotation,
    footprint: (i32, i32),
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedBaseline {
    written: Vec<((i32, i32, i32), BlockState)>,
    previous: Vec<((i32, i32, i32), BlockState)>,
    data_version: Option<i32>,
}

fn saved_placement(placement: &PlacedBuilding) -> SavedPlacement {
    SavedPlacement {
        definition: placement.definition.clone(),
        origin: (placement.origin.x, placement.origin.y, placement.origin.z),
        rotation: placement.rotation,
        footprint: (placement.footprint.x, placement.footprint.y),
    }
}

fn placement_from_saved(saved: SavedPlacement) -> PlacedBuilding {
    let (x, y, z) = saved.origin;
    let (fx, fz) = saved.footprint;
    PlacedBuilding {
        definition: saved.definition,
        origin: IVec3::new(x, y, z),
        rotation: saved.rotation,
        footprint: bevy::math::IVec2::new(fx, fz),
    }
}

fn saved_baseline(baseline: &Baseline) -> SavedBaseline {
    SavedBaseline {
        written: baseline.written.iter().map(|(at, s)| ((at.x, at.y, at.z), s.clone())).collect(),
        previous: baseline.previous.iter().map(|(at, s)| ((at.x, at.y, at.z), s.clone())).collect(),
        data_version: baseline.data_version,
    }
}

fn baseline_from_saved(saved: SavedBaseline) -> Baseline {
    Baseline {
        written: saved.written.into_iter().map(|((x, y, z), s)| (IVec3::new(x, y, z), s)).collect(),
        previous: saved.previous.into_iter().map(|((x, y, z), s)| (IVec3::new(x, y, z), s)).collect(),
        data_version: saved.data_version,
    }
}

/// Why [`save_journal`] or [`load_journal`] failed. No `Corrupt` variant —
/// unlike [`super::persistence`]'s `City`, the journal has no derived state
/// (an occupancy grid) that a hand-edited file could disagree with itself
/// about; it's a flat log, and every entry deserializes independently.
#[derive(Debug)]
pub enum JournalError {
    Io(std::io::Error),
    Parse(String),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::Io(err) => write!(f, "{err}"),
            JournalError::Parse(msg) => write!(f, "{msg}"),
            JournalError::UnsupportedVersion(version) => {
                write!(f, "journal is version {version}, this build reads version {CURRENT_VERSION}")
            }
        }
    }
}

impl std::error::Error for JournalError {}

/// Writes `journal` to `<save_root>/citybuilder/journal.ron`, creating the
/// `citybuilder` directory if it doesn't exist yet (shared with
/// [`super::persistence::save_city`], which creates the same directory for
/// `city.ron`).
pub fn save_journal(journal: &Journal, save_root: &Path) -> Result<(), JournalError> {
    let path = journal_file_path(save_root);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(JournalError::Io)?;
    }

    let entries = journal
        .entries
        .iter()
        .map(|entry| match entry {
            JournalEntry::Placed { building, placement, baseline } => SavedEntry::Placed {
                building: building.as_u64(),
                placement: saved_placement(placement),
                baseline: saved_baseline(baseline),
            },
            JournalEntry::Demolished { building, placement, baseline } => SavedEntry::Demolished {
                building: building.as_u64(),
                placement: saved_placement(placement),
                baseline: saved_baseline(baseline),
            },
        })
        .collect();

    let save = SavedJournal { version: CURRENT_VERSION, entries };
    let text = ron::ser::to_string_pretty(&save, ron::ser::PrettyConfig::default())
        .map_err(|err| JournalError::Parse(err.to_string()))?;
    fs::write(&path, text).map_err(JournalError::Io)
}

/// Reads `<save_root>/citybuilder/journal.ron` into a fresh [`Journal`]. A
/// missing file is `Ok(Journal::default())` — a save with no journal yet is
/// not an error, same contract [`super::persistence::load_city`] uses for a
/// missing `city.ron`.
pub fn load_journal(save_root: &Path) -> Result<Journal, JournalError> {
    let path = journal_file_path(save_root);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Journal::default()),
        Err(err) => return Err(JournalError::Io(err)),
    };

    let save: SavedJournal = ron::de::from_str(&text).map_err(|err| JournalError::Parse(err.to_string()))?;
    if save.version != CURRENT_VERSION {
        return Err(JournalError::UnsupportedVersion(save.version));
    }

    let entries = save
        .entries
        .into_iter()
        .map(|entry| match entry {
            SavedEntry::Placed { building, placement, baseline } => JournalEntry::Placed {
                building: BuildingId::from_u64(building),
                placement: placement_from_saved(placement),
                baseline: baseline_from_saved(baseline),
            },
            SavedEntry::Demolished { building, placement, baseline } => JournalEntry::Demolished {
                building: BuildingId::from_u64(building),
                placement: placement_from_saved(placement),
                baseline: baseline_from_saved(baseline),
            },
        })
        .collect();

    Ok(Journal { entries })
}

#[cfg(test)]
mod tests;
