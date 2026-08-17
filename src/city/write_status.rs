//! The last write's outcome (ticket 050, roadmap G2): what
//! `city::ui::city_panel` shows so "the user needs to know whether what they
//! see has actually reached the world" — the roadmap's own justification for
//! G2's write-status line, worth more than it sounds.
//!
//! This module records; it never dispatches or polls anything itself.
//! [`crate::city::commit::poll_commit`] and
//! [`crate::city::demolish::poll_demolish`] are the only writers, one call
//! each, at the exact point where they already have a
//! `Result<WriteSummary, WriteError>` in hand — recording it here is a line
//! added to logic that already exists, not a second write-tracking mechanism
//! next to it.
//!
//! [`WriteStatus`] holds only the *last* write, not a history — the same
//! "one slot, not a log" shape [`super::commit::CommitState`]/
//! [`super::demolish::DemolishState`] already use for the write itself. A
//! history belongs to [`super::journal::Journal`], which already has one;
//! this is a status line, not a second journal.

use std::path::PathBuf;

use bevy::prelude::Resource;

use crate::edit::session::WriteSummary;

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
}

/// What a successful write actually did, trimmed to what the panel shows —
/// [`crate::edit::EditReport`]'s own fields, plus [`WriteSummary::backups`],
/// rather than holding the whole [`WriteSummary`] and letting the panel
/// reach into it.
#[derive(Debug, Clone)]
pub(super) struct WriteRecord {
    pub(super) kind: WriteKind,
    /// The building's definition id — `PlacedBuilding::definition`, not a
    /// display name (the panel doesn't have `BuildingDefinitions` to hand at
    /// every call site that might record one; the id is enough to read).
    pub(super) building: String,
    pub(super) blocks: usize,
    pub(super) chunks: usize,
    pub(super) regions: Vec<(i32, i32)>,
    /// Backup files taken by *this* write — empty when every touched region
    /// was already backed up earlier in the same session. See
    /// [`WriteSummary::backups`].
    pub(super) backups: Vec<PathBuf>,
}

/// The last commit or demolition, success or failure. `None` before either
/// has ever settled once in this session.
#[derive(Debug, Clone)]
pub(super) enum LastWrite {
    Success(WriteRecord),
    Failed {
        kind: WriteKind,
        building: String,
        /// [`WriteError`]'s own `Display`, captured at record time rather
        /// than kept as the error itself — nothing here needs to match on
        /// the specific variant, only show what it said.
        message: String,
    },
}

/// What [`crate::city::ui::city_panel`] reads. Registered by both
/// `city::commit::CommitPlugin` and `city::demolish::DemolishPlugin` via
/// `init_resource`, the same idempotent-either-order shape
/// [`super::write_gate::WriteGate`] already uses — it doesn't matter which
/// plugin's `build` runs first.
#[derive(Resource, Default, Debug)]
pub struct WriteStatus {
    last: Option<LastWrite>,
}

impl WriteStatus {
    /// Called from `poll_commit`/`poll_demolish` right where they already
    /// match `Ok(summary)`.
    pub(super) fn record_success(&mut self, kind: WriteKind, building: impl Into<String>, summary: &WriteSummary) {
        self.last = Some(LastWrite::Success(WriteRecord {
            kind,
            building: building.into(),
            blocks: summary.report.blocks_written,
            chunks: summary.report.chunks.len(),
            regions: summary.regions_written.clone(),
            backups: summary.backups.clone(),
        }));
    }

    /// Called from `poll_commit`/`poll_demolish`/`poll_undo` right where they
    /// already match `Err(err)`. Takes a plain message rather than
    /// `&WriteError` directly — `city::undo`'s own failures ("nothing to
    /// undo", `UndoError::Occupied`) aren't a [`WriteError`] at all, and this
    /// module has no reason to know the difference between the write paths'
    /// error types when [`WriteError`]'s own `Display` (or any other error's)
    /// already says what happened.
    pub(super) fn record_failure(&mut self, kind: WriteKind, building: impl Into<String>, message: impl Into<String>) {
        self.last = Some(LastWrite::Failed { kind, building: building.into(), message: message.into() });
    }

    /// `city::ui::city_panel`'s only reader.
    pub(super) fn last(&self) -> Option<&LastWrite> {
        self.last.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::session::WriteError;
    use crate::edit::EditReport;

    fn a_summary() -> WriteSummary {
        WriteSummary {
            report: EditReport { blocks_written: 12, chunks: vec![(0, 0), (0, 1)], regions: vec![(0, 0)], replaced: None },
            regions_written: vec![(0, 0)],
            backups: vec![PathBuf::from("world/block_viewer_backups/20260101_000000/r.0.0.mca")],
        }
    }

    #[test]
    fn starts_empty() {
        assert!(WriteStatus::default().last().is_none());
    }

    #[test]
    fn a_success_is_recorded_with_the_summarys_own_figures() {
        let mut status = WriteStatus::default();
        status.record_success(WriteKind::Placed, "house01", &a_summary());

        let Some(LastWrite::Success(record)) = status.last() else { panic!("expected a success") };
        assert_eq!(record.kind, WriteKind::Placed);
        assert_eq!(record.building, "house01");
        assert_eq!(record.blocks, 12);
        assert_eq!(record.chunks, 2);
        assert_eq!(record.regions, vec![(0, 0)]);
        assert_eq!(record.backups.len(), 1);
    }

    #[test]
    fn a_failure_is_recorded_with_the_errors_message() {
        let mut status = WriteStatus::default();
        let err = WriteError::WorldIsOpen { save: "world".to_string() };
        status.record_failure(WriteKind::Demolished, "house01", err.to_string());

        let Some(LastWrite::Failed { kind, building, message }) = status.last() else { panic!("expected a failure") };
        assert_eq!(*kind, WriteKind::Demolished);
        assert_eq!(building, "house01");
        assert_eq!(message, &err.to_string());
    }

    #[test]
    fn a_second_write_replaces_the_first_rather_than_accumulating() {
        let mut status = WriteStatus::default();
        status.record_success(WriteKind::Placed, "house01", &a_summary());
        status.record_failure(WriteKind::Demolished, "house02", WriteError::WorldIsOpen { save: "world".to_string() }.to_string());

        assert!(matches!(status.last(), Some(LastWrite::Failed { building, .. }) if building == "house02"));
    }
}
