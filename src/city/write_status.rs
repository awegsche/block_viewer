//! The last edit's and the last save's outcome (ticket 050, roadmap G2;
//! split by ticket 051 into the two things it now covers): what
//! `city::ui::city_panel` shows so "the user needs to know whether what they
//! see has actually reached the world" — the roadmap's own justification for
//! G2's write-status line, worth more than it sounds.
//!
//! This module records; it never dispatches or polls anything itself.
//! [`crate::city::commit::poll_commit`], [`crate::city::demolish::poll_demolish`]
//! and [`crate::city::undo::poll_undo`] call [`WriteStatus::record_success`]/
//! [`WriteStatus::record_failure`] for the *edit* they applied to the shared
//! region cache — since ticket 051, that's memory only, not disk, which is
//! why [`WriteRecord`] carries no backup paths any more.
//! [`crate::city::save::poll_save`] calls [`WriteStatus::record_save_success`]/
//! [`WriteStatus::record_save_failure`] for the separate, later moment the
//! player actually writes those edits to disk.
//!
//! [`WriteStatus`] holds only the *last* edit and the *last* save, not a
//! history — the same "one slot, not a log" shape
//! [`super::commit::CommitState`]/[`super::demolish::DemolishState`] already
//! use for the operations themselves. A history belongs to
//! [`super::journal::Journal`], which already has one; this is a status
//! line, not a second journal.

use std::path::PathBuf;

use bevy::prelude::Resource;

use crate::edit::session::WriteSummary;
use crate::edit::EditReport;

/// Which write path produced a [`WriteRecord`]/failure — the city panel
/// labels its status line differently for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WriteKind {
    Placed,
    Demolished,
    /// `city::undo` (ticket 050, roadmap G2) reversing the most recent
    /// journal entry — which of the other two kinds it actually undid is
    /// [`super::undo::UndoneKind`]'s own distinction, not this one's; the
    /// write itself is neither a placement nor a demolition proper (no new
    /// journal entry comes out of it either way), so it gets a third label
    /// rather than borrowing one of the two above.
    Undo,
    /// `city::road_build` (ticket 055, roadmap F2/F3) committing a drag —
    /// not `Placed`, since a road commit's `building` field names no
    /// `BuildingCatalogue`/`BuildingDefinitions` entry at all (see
    /// [`WriteRecord::building`]'s doc comment).
    Road,
    /// `city::terraform` (ticket 057, roadmap H1) committing a dig or level
    /// drag — same reasoning as [`WriteKind::Road`]: `building` names a tile
    /// count ("N tile(s)"), not a catalogue entry.
    Terraform,
}

/// What a successful edit actually applied to the region cache, trimmed to
/// what the panel shows — [`crate::edit::EditReport`]'s own fields. No
/// backup paths: since ticket 051, applying an edit doesn't touch disk at
/// all, so there's nothing to have backed up yet. See
/// [`SaveRecord`] for the disk side.
#[derive(Debug, Clone)]
pub(super) struct WriteRecord {
    pub(super) kind: WriteKind,
    /// The building's definition id — `PlacedBuilding::definition`, not a
    /// display name (the panel doesn't have `BuildingDefinitions` to hand at
    /// every call site that might record one; the id is enough to read).
    /// For [`WriteKind::Road`] (ticket 055): a short description of the
    /// drag (`"N road cell(s)"`) rather than an id — a road commit names no
    /// single catalogue entry.
    pub(super) building: String,
    pub(super) blocks: usize,
    pub(super) chunks: usize,
    pub(super) regions: Vec<(i32, i32)>,
}

/// The last commit, demolition or undo, success or failure. `None` before
/// any has ever settled once in this session.
#[derive(Debug, Clone)]
pub(super) enum LastWrite {
    Success(WriteRecord),
    Failed {
        kind: WriteKind,
        building: String,
        /// The error's own `Display`, captured at record time rather than
        /// kept as the error itself — nothing here needs to match on the
        /// specific variant, only show what it said.
        message: String,
    },
}

/// What a successful [`super::save::SaveCommand`] flush actually wrote —
/// [`WriteSummary`]'s own region and backup lists, the disk-side counterpart
/// to [`WriteRecord`].
#[derive(Debug, Clone)]
pub(super) struct SaveRecord {
    pub(super) regions: Vec<(i32, i32)>,
    /// Backup files taken by *this* save — empty when every touched region
    /// was already backed up earlier in the same session, or when backups
    /// are off.
    pub(super) backups: Vec<PathBuf>,
}

/// The last "Save world" click, success or failure. `None` before one has
/// ever settled this session.
#[derive(Debug, Clone)]
pub(super) enum LastSave {
    Success(SaveRecord),
    Failed { message: String },
}

/// What [`crate::city::ui::city_panel`] reads. Registered by
/// `city::commit::CommitPlugin`, `city::demolish::DemolishPlugin`,
/// `city::undo::UndoPlugin` and `city::save::SavePlugin` via `init_resource`,
/// an idempotent-either-order shape — it doesn't matter which plugin's
/// `build` runs first.
#[derive(Resource, Default, Debug)]
pub struct WriteStatus {
    last: Option<LastWrite>,
    last_save: Option<LastSave>,
}

impl WriteStatus {
    /// Called from `poll_commit`/`poll_demolish`/`poll_undo` right where they
    /// already match `Ok(report)` on the *edit* they applied to the region
    /// cache — not a disk write; see the module docs.
    pub(super) fn record_success(&mut self, kind: WriteKind, building: impl Into<String>, report: &EditReport) {
        self.last = Some(LastWrite::Success(WriteRecord {
            kind,
            building: building.into(),
            blocks: report.blocks_written,
            chunks: report.chunks.len(),
            regions: report.regions.clone(),
        }));
    }

    /// Called from `poll_commit`/`poll_demolish`/`poll_undo` right where they
    /// already match `Err(err)`. Takes a plain message rather than
    /// `&EditRefusal` directly — `city::undo`'s own pre-write failures
    /// ("nothing to undo", `UndoError::Occupied`) aren't an `EditRefusal` at
    /// all, and this module has no reason to know the difference between the
    /// callers' error types when each one's own `Display` already says what
    /// happened.
    pub(super) fn record_failure(&mut self, kind: WriteKind, building: impl Into<String>, message: impl Into<String>) {
        self.last = Some(LastWrite::Failed { kind, building: building.into(), message: message.into() });
    }

    /// `city::ui::city_panel`'s reader for the last placement/demolition/undo
    /// applied to memory.
    pub(super) fn last(&self) -> Option<&LastWrite> {
        self.last.as_ref()
    }

    /// Called from [`super::save::poll_save`] right where it already matches
    /// `Ok(summary)` on an actual disk write.
    pub(super) fn record_save_success(&mut self, summary: &WriteSummary) {
        self.last_save = Some(LastSave::Success(SaveRecord {
            regions: summary.regions_written.clone(),
            backups: summary.backups.clone(),
        }));
    }

    /// Called from [`super::save::poll_save`] right where it already matches
    /// `Err(err)`.
    pub(super) fn record_save_failure(&mut self, message: impl Into<String>) {
        self.last_save = Some(LastSave::Failed { message: message.into() });
    }

    /// `city::ui::city_panel`'s reader for the last "Save world" click.
    pub(super) fn last_save(&self) -> Option<&LastSave> {
        self.last_save.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::session::WriteError;
    use crate::edit::EditRefusal;

    fn a_report() -> EditReport {
        EditReport { blocks_written: 12, chunks: vec![(0, 0), (0, 1)], regions: vec![(0, 0)], replaced: None, ..Default::default() }
    }

    fn a_summary() -> WriteSummary {
        WriteSummary { report: a_report(), regions_written: vec![(0, 0)], backups: vec![PathBuf::from("world/block_viewer_backups/20260101_000000/r.0.0.mca")] }
    }

    #[test]
    fn starts_empty() {
        let status = WriteStatus::default();
        assert!(status.last().is_none());
        assert!(status.last_save().is_none());
    }

    #[test]
    fn a_success_is_recorded_with_the_reports_own_figures() {
        let mut status = WriteStatus::default();
        status.record_success(WriteKind::Placed, "house01", &a_report());

        let Some(LastWrite::Success(record)) = status.last() else { panic!("expected a success") };
        assert_eq!(record.kind, WriteKind::Placed);
        assert_eq!(record.building, "house01");
        assert_eq!(record.blocks, 12);
        assert_eq!(record.chunks, 2);
        assert_eq!(record.regions, vec![(0, 0)]);
    }

    #[test]
    fn a_failure_is_recorded_with_the_errors_message() {
        let mut status = WriteStatus::default();
        let err = EditRefusal::Empty;
        status.record_failure(WriteKind::Demolished, "house01", err.to_string());

        let Some(LastWrite::Failed { kind, building, message }) = status.last() else { panic!("expected a failure") };
        assert_eq!(*kind, WriteKind::Demolished);
        assert_eq!(building, "house01");
        assert_eq!(message, &err.to_string());
    }

    #[test]
    fn a_second_write_replaces_the_first_rather_than_accumulating() {
        let mut status = WriteStatus::default();
        status.record_success(WriteKind::Placed, "house01", &a_report());
        status.record_failure(WriteKind::Demolished, "house02", EditRefusal::Empty.to_string());

        assert!(matches!(status.last(), Some(LastWrite::Failed { building, .. }) if building == "house02"));
    }

    #[test]
    fn a_save_success_is_recorded_separately_from_the_last_edit() {
        let mut status = WriteStatus::default();
        status.record_success(WriteKind::Placed, "house01", &a_report());
        status.record_save_success(&a_summary());

        // The last-edit slot is untouched by a save recording — they're two
        // different things, per the module docs.
        assert!(matches!(status.last(), Some(LastWrite::Success(record)) if record.building == "house01"));
        let Some(LastSave::Success(record)) = status.last_save() else { panic!("expected a successful save") };
        assert_eq!(record.regions, vec![(0, 0)]);
        assert_eq!(record.backups.len(), 1);
    }

    #[test]
    fn a_save_failure_is_recorded_with_its_message() {
        let mut status = WriteStatus::default();
        let err = WriteError::WorldIsOpen { save: "world".to_string() };
        status.record_save_failure(err.to_string());

        let Some(LastSave::Failed { message }) = status.last_save() else { panic!("expected a failed save") };
        assert_eq!(message, &err.to_string());
    }

    #[test]
    fn a_second_save_replaces_the_first_rather_than_accumulating() {
        let mut status = WriteStatus::default();
        status.record_save_success(&a_summary());
        status.record_save_failure("disk full");

        assert!(matches!(status.last_save(), Some(LastSave::Failed { message }) if message == "disk full"));
    }
}
