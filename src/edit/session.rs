//! Write safety (ticket 033, roadmap W6): the sound way to save a modified
//! world.
//!
//! Tickets 031 and 032 both stop one step short of the disk. [`super::apply`]
//! and [`super::apply_routed`] mutate regions in memory and leave them dirty,
//! because a function that both edits and writes can't be tested without a
//! disk — and because 032's rollback *is* "throw the in-memory region away",
//! which only works while nothing has been saved.
//!
//! This module is that last step, and it is the only place in the crate that
//! writes to somebody's world. Four rules:
//!
//! 1. **The world must not be open in Minecraft.** [`WriteSession`] holds the
//!    save's `session.lock` for as long as it lives, rather than probing it
//!    and hoping — the game opening the world between a probe and a write is
//!    exactly the race a probe cannot close.
//! 2. **Back up before the first write of a session**, per region file.
//! 3. **Write atomically.** Inherited, not written here: `mc_anvil`'s
//!    [`Region::write`](mc_anvil::region::Region::write) writes a temp file in
//!    the same directory, flushes it, `sync_all`s it and renames it over the
//!    original. What this module owes that guarantee is not to undermine it,
//!    and to be honest about where it stops — see [`WriteError::WriteFailed`].
//! 4. **A dry run tells you what would happen** without anything happening.
//!
//! It also enforces the contract 032 could only state: one transaction at a
//! time, saved before the next. A caller that skips this layer meets it as
//! [`EditRefusal::RegionHasUnsavedChanges`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mc_anvil::{SaveMeta, SessionLock};

use super::route::{apply_routed, plan_routed, RegionSource};
use super::{EditPolicy, EditRefusal, EditReport, WorldEdit};

/// The directory backups go in, at the save root — deliberately *not* inside
/// the region directory. `mc_anvil`'s save discovery skips anything that isn't
/// `r.<x>.<z>.mca`, but Minecraft's own directory scan is not ours to bet on.
pub const BACKUP_DIR: &str = "block_viewer_backups";

// -------------------------------------------------------------------------------------------------
// ---- the session --------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Which of the safety rules a session applies. Both default to on; the
/// overrides exist for a caller that has its own equivalent, and for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteSafety {
    /// Take and hold the save's `session.lock`, refusing to open the session
    /// at all if Minecraft (or another editor) already holds it.
    pub require_session_lock: bool,
    /// Copy each region file aside before this session's first write to it.
    pub back_up: bool,
}

impl Default for WriteSafety {
    fn default() -> Self {
        Self {
            require_session_lock: true,
            back_up: true,
        }
    }
}

/// A claim on a save for the duration of some editing: the held
/// `session.lock`, and the backups taken so far.
///
/// A *write* session, not the application's lifetime. Opening one keeps
/// Minecraft out of the world, so a viewer that never edits should never have
/// one; drop it when the user stops editing and the world is theirs again.
pub struct WriteSession {
    /// `None` only when [`WriteSafety::require_session_lock`] was off — the
    /// guard releases the lock when it drops, which is why it is *held* rather
    /// than checked and forgotten.
    _lock: Option<SessionLock>,
    /// Where this session's backups go. Created on first use, so a session
    /// that only ever plans leaves nothing on disk.
    backup_dir: PathBuf,
    /// Region files already copied aside this session, and where to. Keyed so
    /// the second edit of a region can't overwrite the copy of its pre-edit
    /// state with our own output.
    backups: BTreeMap<(i32, i32), PathBuf>,
    back_up: bool,
    save_name: String,
}

impl WriteSession {
    /// Opens a write session on `save` with every safety rule on.
    ///
    /// Fails with [`WriteError::WorldIsOpen`] if the world is open in
    /// Minecraft right now. That's a refusal rather than a warning: writing
    /// region files underneath a running game loses the edit when the game
    /// next flushes those chunks, and corrupts the file if the two writers
    /// interleave.
    pub fn open(save: &SaveMeta) -> Result<Self, WriteError> {
        Self::open_with(save, WriteSafety::default())
    }

    /// [`open`](Self::open) with the safety rules spelled out.
    pub fn open_with(save: &SaveMeta, safety: WriteSafety) -> Result<Self, WriteError> {
        let lock = if safety.require_session_lock {
            match SessionLock::acquire(save) {
                Ok(Some(lock)) => Some(lock),
                Ok(None) => {
                    return Err(WriteError::WorldIsOpen {
                        save: save.name.clone(),
                    });
                }
                Err(err) => {
                    return Err(WriteError::Lock {
                        save: save.name.clone(),
                        reason: err.to_string(),
                    });
                }
            }
        } else {
            None
        };

        Ok(Self {
            _lock: lock,
            backup_dir: save.path.join(BACKUP_DIR).join(session_stamp(SystemTime::now())),
            backups: BTreeMap::new(),
            back_up: safety.back_up,
            save_name: save.name.clone(),
        })
    }

    /// Where this session's backups go. The directory may not exist yet — it
    /// is created by the first write, not by opening the session.
    pub fn backup_dir(&self) -> &Path {
        &self.backup_dir
    }

    /// Region files this session has already copied aside, ascending.
    pub fn backed_up(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.backups.keys().copied()
    }

    /// The dry run: what [`commit`](Self::commit) would do, without doing any
    /// of it.
    ///
    /// Everything [`plan_routed`] reports — regions, chunks, block count, and
    /// every refusal — plus what only a session knows: which file each region
    /// lives in, where its backup would go, and whether it already has one.
    ///
    /// Writes nothing and creates nothing, the backup directory included.
    pub fn plan<S: RegionSource>(
        &self,
        edit: &WorldEdit,
        source: &mut S,
        policy: &EditPolicy,
    ) -> Result<WritePlan, EditRefusal> {
        let report = plan_routed(edit, source, policy)?;

        let mut regions = Vec::with_capacity(report.regions.len());
        for coord in &report.regions {
            // Planned successfully a line ago, so this can only fail if the
            // source changed underneath us — in which case the plan is wrong
            // and saying so beats reporting one region fewer than it would
            // write. The fetch is for the region's *path*, which the report
            // doesn't carry.
            let path = source
                .region_mut(*coord)
                .map(|region| region.region.get_path().to_path_buf())
                .map_err(|unavailable| super::route::refusal_for(*coord, unavailable))?;
            regions.push(self.region_write(*coord, path));
        }

        Ok(WritePlan { report, regions })
    }

    /// Applies `edit` and writes every region it touched to disk.
    ///
    /// The order is the whole point:
    ///
    /// 1. [`apply_routed`] — every region planned, then every region mutated
    ///    in memory, or none of them.
    /// 2. Back up every touched region file this session hasn't backed up yet,
    ///    **before the first save**, so there is no window in which half the
    ///    files are written and the next backup fails.
    /// 3. Save each region, atomically per file.
    ///
    /// A failure in 1 or 2 rolls the whole transaction back and leaves the
    /// save byte-identical. A failure in 3 cannot roll back — see
    /// [`WriteError::WriteFailed`], which names what was written and where the
    /// backups are.
    pub fn commit<S: RegionSource>(
        &mut self,
        edit: &WorldEdit,
        source: &mut S,
        policy: &EditPolicy,
    ) -> Result<WriteSummary, WriteError> {
        let report = apply_routed(edit, source, policy)?;
        let regions = report.regions.clone();

        match self.write_regions(&regions, source) {
            Ok(backups) => Ok(WriteSummary {
                report,
                regions_written: regions,
                backups,
            }),
            Err(err) => {
                // These two both fail with the save untouched, so the
                // transaction can still be made all-or-nothing the way 032
                // does it: throw the in-memory mutations away.
                //
                // `WriteFailed` deliberately isn't in the list even when it
                // failed on the first region. Nothing was lost there — the
                // edit is still in memory and still dirty — and discarding it
                // would turn a retryable disk error into lost work.
                if matches!(err, WriteError::Backup { .. } | WriteError::Refused(_)) {
                    self.roll_back(&regions, source);
                }
                Err(err)
            }
        }
    }

    /// Writes every region in `source` that has unsaved changes.
    ///
    /// [`commit`](Self::commit) saves its own transaction's regions; this is
    /// for a caller that batched several transactions with
    /// [`EditPolicy::allow_dirty_regions`], and for "save before quitting".
    ///
    /// A no-op — and a success — when nothing is dirty.
    pub fn flush<S: RegionSource>(&mut self, source: &mut S) -> Result<WriteSummary, WriteError> {
        // Sorted: `RegionCache::dirty_regions` iterates a `HashMap`, and a
        // save order that varies from run to run is how a test flakes once a
        // month (ticket 032 hit exactly that).
        let mut regions = source.dirty_regions();
        regions.sort_unstable();

        let backups = self.write_regions(&regions, source)?;
        Ok(WriteSummary {
            report: EditReport {
                regions: regions.clone(),
                ..EditReport::default()
            },
            regions_written: regions,
            backups,
        })
    }

    /// Backs up every region in `regions`, then saves every one of them.
    ///
    /// Two passes rather than one interleaved loop, deliberately: a backup
    /// failure while some files are already written is a state this can't undo
    /// and the user can't easily read. Doing all the copying first means a
    /// backup failure always happens with the save untouched — which is what
    /// lets [`commit`](Self::commit) roll one back.
    ///
    /// Rolls nothing back itself. `commit` decides that, because `flush` must
    /// not: discarding a region there would throw away edits this call never
    /// made.
    fn write_regions<S: RegionSource>(
        &mut self,
        regions: &[(i32, i32)],
        source: &mut S,
    ) -> Result<Vec<PathBuf>, WriteError> {
        let mut backups = Vec::new();

        for coord in regions {
            let path = match source.region_mut(*coord) {
                Ok(region) => region.region.get_path().to_path_buf(),
                Err(unavailable) => {
                    return Err(WriteError::Refused(super::route::refusal_for(
                        *coord,
                        unavailable,
                    )));
                }
            };

            match self.back_up(*coord, &path) {
                Ok(Some(backup)) => backups.push(backup),
                Ok(None) => {}
                Err(reason) => {
                    return Err(WriteError::Backup {
                        region: *coord,
                        path,
                        reason,
                    });
                }
            }
        }

        let mut written: Vec<(i32, i32)> = Vec::new();
        for coord in regions {
            let region = match source.region_mut(*coord) {
                Ok(region) => region,
                Err(unavailable) => {
                    return Err(WriteError::WriteFailed {
                        region: *coord,
                        reason: match unavailable {
                            super::RegionUnavailable::NotGenerated => {
                                "the region is no longer available".to_string()
                            }
                            super::RegionUnavailable::Unreadable(reason) => reason,
                        },
                        written,
                        backup_dir: self.backup_dir.clone(),
                    })
                }
            };

            if let Err(err) = region.save() {
                // No rollback here, and none is possible: the regions in
                // `written` are on disk. `ChunkRegion::save` clears the dirty
                // set on success only, so this one and everything after it
                // stay dirty and a retry still writes them.
                return Err(WriteError::WriteFailed {
                    region: *coord,
                    reason: err.to_string(),
                    written,
                    backup_dir: self.backup_dir.clone(),
                });
            }
            written.push(*coord);
        }

        Ok(backups)
    }

    /// Copies a region file aside, unless this session already has a copy of
    /// it (or backups are off). Returns where it went.
    ///
    /// Once per session per region: the backup is worth having because it
    /// holds the state *before we touched it*, and copying again after an
    /// earlier edit would replace that with our own output.
    fn back_up(&mut self, coord: (i32, i32), path: &Path) -> Result<Option<PathBuf>, String> {
        if !self.back_up || self.backups.contains_key(&coord) {
            return Ok(None);
        }

        // A region with no file on disk has nothing to lose — `mc_anvil`
        // doesn't create region files today, so this is a backstop rather than
        // a path anything takes. Record it as done so the check isn't repeated.
        if !path.exists() {
            self.backups.insert(coord, PathBuf::new());
            return Ok(None);
        }

        std::fs::create_dir_all(&self.backup_dir).map_err(|err| err.to_string())?;

        let name = path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(format!("r.{}.{}.mca", coord.0, coord.1)));
        let destination = self.backup_dir.join(name);
        std::fs::copy(path, &destination).map_err(|err| err.to_string())?;

        self.backups.insert(coord, destination.clone());
        Ok(Some(destination))
    }

    /// Throws away the in-memory mutations of a transaction that failed before
    /// anything was saved.
    fn roll_back<S: RegionSource>(&self, regions: &[(i32, i32)], source: &mut S) {
        for coord in regions {
            source.discard(*coord);
        }
    }

    fn region_write(&self, coord: (i32, i32), path: PathBuf) -> RegionWrite {
        let backed_up = self.backups.contains_key(&coord);
        let backup = self.back_up.then(|| match self.backups.get(&coord) {
            Some(existing) => existing.clone(),
            None => self.backup_dir.join(
                path.file_name()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(format!("r.{}.{}.mca", coord.0, coord.1))),
            ),
        });

        RegionWrite {
            coord,
            path,
            backup,
            backed_up,
        }
    }
}

impl std::fmt::Debug for WriteSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteSession")
            .field("save", &self.save_name)
            .field("lock_held", &self._lock.is_some())
            .field("backup_dir", &self.backup_dir)
            .field("backups", &self.backups.len())
            .finish()
    }
}

// -------------------------------------------------------------------------------------------------
// ---- what a write did, or would do ---------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One region file a transaction would touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionWrite {
    pub coord: (i32, i32),
    /// The `.mca` file that would be replaced.
    pub path: PathBuf,
    /// Where its backup is, or would go. `None` when backups are off.
    pub backup: Option<PathBuf>,
    /// Whether that backup already exists, i.e. this session has written to
    /// this region before.
    pub backed_up: bool,
}

/// The dry run: [`EditReport`] plus the files behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WritePlan {
    pub report: EditReport,
    pub regions: Vec<RegionWrite>,
}

impl std::fmt::Display for WritePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} block(s) in {} chunk(s) across {} region file(s):",
            self.report.blocks_written,
            self.report.chunks.len(),
            self.regions.len()
        )?;
        for region in &self.regions {
            write!(
                f,
                "  ({}, {})  {}",
                region.coord.0,
                region.coord.1,
                region.path.display()
            )?;
            match (&region.backup, region.backed_up) {
                (Some(_), true) => writeln!(f, "  [already backed up]")?,
                (Some(backup), false) => writeln!(f, "  -> backup {}", backup.display())?,
                (None, _) => writeln!(f, "  [no backup]")?,
            }
        }
        Ok(())
    }
}

/// What a committed write actually did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteSummary {
    /// The edit's own report — blocks, chunks, regions. From [`flush`], only
    /// the regions are filled in: what was already applied isn't re-derivable
    /// at save time.
    ///
    /// [`flush`]: WriteSession::flush
    pub report: EditReport,
    /// The region files written, ascending.
    pub regions_written: Vec<(i32, i32)>,
    /// Backups taken by *this* call — empty when every region was already
    /// backed up earlier in the session, or when backups are off.
    pub backups: Vec<PathBuf>,
}

// -------------------------------------------------------------------------------------------------
// ---- failures -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Why a write didn't happen — or, in one case, why it half did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteError {
    /// The world is open in Minecraft. Refused rather than warned about:
    /// writing region files underneath a running game loses the edit when the
    /// game next flushes those chunks, and corrupts the file if the two
    /// writers interleave.
    WorldIsOpen { save: String },
    /// The `session.lock` couldn't be opened at all — a permissions problem, a
    /// save directory that isn't there. Distinct from [`WriteError::WorldIsOpen`]
    /// because the answer is different: one means "close Minecraft", the other
    /// means "something is wrong with this save".
    Lock { save: String, reason: String },
    /// The edit itself was refused; nothing was written. Every rule in
    /// [`EditRefusal`], reached through this layer.
    Refused(EditRefusal),
    /// A region file couldn't be copied aside, so the transaction was rolled
    /// back rather than written unprotected.
    Backup {
        region: (i32, i32),
        path: PathBuf,
        reason: String,
    },
    /// A save failed *after* at least one region file had been replaced.
    ///
    /// The one failure this layer cannot undo, and it says so rather than
    /// pretending: `written` names the region files that are already on disk,
    /// `region` the one that failed, and `backup_dir` where the originals are.
    /// The regions that didn't get written stay dirty (`ChunkRegion::save`
    /// clears the dirty set on success only), so a retry still writes them.
    ///
    /// Per-file atomicity is guaranteed by `mc_anvil`; a four-file transaction
    /// is not something any filesystem offers, and a journal that replayed one
    /// would be a bigger machine than the thing it protects.
    WriteFailed {
        region: (i32, i32),
        reason: String,
        written: Vec<(i32, i32)>,
        backup_dir: PathBuf,
    },
}

impl From<EditRefusal> for WriteError {
    fn from(refusal: EditRefusal) -> Self {
        WriteError::Refused(refusal)
    }
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::WorldIsOpen { save } => write!(
                f,
                "\"{save}\" is open in Minecraft; close the world before editing it"
            ),
            WriteError::Lock { save, reason } => {
                write!(f, "could not take \"{save}\"'s session.lock: {reason}")
            }
            WriteError::Refused(refusal) => write!(f, "{refusal}"),
            WriteError::Backup {
                region,
                path,
                reason,
            } => write!(
                f,
                "could not back up region ({}, {}) from {}: {reason} — nothing was written",
                region.0,
                region.1,
                path.display()
            ),
            WriteError::WriteFailed {
                region,
                reason,
                written,
                backup_dir,
            } => write!(
                f,
                "region ({}, {}) could not be written: {reason}. {} region file(s) had already \
                 been written; the originals are in {}",
                region.0,
                region.1,
                written.len(),
                backup_dir.display()
            ),
        }
    }
}

impl std::error::Error for WriteError {}

// -------------------------------------------------------------------------------------------------
// ---- the session's name -------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A session's backup directory name: the UTC time it opened, as
/// `YYYY-MM-DDTHH-MM-SSZ`.
///
/// Hyphens where a timestamp would have colons, because Windows won't have
/// them in a path. Readable rather than an epoch count because this string is
/// what the user is told to look in when something has gone wrong, and
/// "1755264000" is not an answer to "which backup is yesterday's".
///
/// `pub(super)` only so the edit module's test file can pin it to a known
/// epoch; nothing outside this module calls it.
pub(super) fn session_stamp(now: SystemTime) -> String {
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);

    let (days, seconds_of_day) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    let (year, month, day) = civil_from_days(days);

    format!(
        "{year:04}-{month:02}-{day:02}T{:02}-{:02}-{:02}Z",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60,
    )
}

/// Days since the Unix epoch to a civil (proleptic Gregorian) date.
///
/// Howard Hinnant's `civil_from_days`, which is the standard answer to this
/// and handles leap years and centuries without a table. Shifting the era to
/// start on 1 March makes the leap day the *last* day of the year, which is
/// what removes the special cases.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;

    (if month <= 2 { year + 1 } else { year }, month, day)
}
