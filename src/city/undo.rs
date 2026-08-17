//! Undo (ticket 050, roadmap G2): the button `journal`'s own module docs
//! have been pointing at since ticket 044 — "gives undo for free" was D3's
//! promise, and this is where it actually gets run through the write path.
//!
//! ## Following `city::commit`/`city::demolish`'s shape, with one real difference
//!
//! Same `request`/`busy`/`state` surface [`crate::viewer::paint::PaintCommand`]
//! established and `city::commit::CommitState`/`city::demolish::DemolishState`
//! already follow: a click sets [`UndoCommand::requested`], [`start_undo`]
//! dispatches onto [`AsyncComputeTaskPool`], [`poll_undo`] polls it with the
//! same `block_on(poll_once(..))` pattern every other task in this crate
//! uses.
//!
//! The one place this can't mirror commit/demolish: neither of *those*
//! decides what to write until a click supplies a target (a hovered tile, a
//! selected catalogue entry). Undo's target is just "whatever
//! [`journal::Journal::undo_last`] does" — and that call **is** the City-side
//! reversal, not a preview of it; there is no way to ask it what it would do
//! without it doing it. So [`start_undo`] checks every precondition it can
//! *without* calling it first — an empty journal, no save loaded, the write
//! gate already held, the same three guards `try_commit_placement`/
//! `try_demolish` already check before touching anything — and only calls
//! `undo_last` once all three are clear. From that point on there is no
//! going back: [`journal::Journal::undo_last`]'s own docs already name this —
//! "this call has already moved `city` and the journal on by the time it
//! returns, on the assumption the caller commits `edit` next." A write
//! failure after that point is not rolled back — it can't be, the entry that
//! would describe how is already gone — and [`poll_undo`] says so rather
//! than pretending otherwise. That's an accepted, documented gap in D3's own
//! contract, not one this ticket introduces; checking what can be checked
//! first just keeps it as rare as possible.
//!
//! ## One write of any kind at a time
//!
//! [`UndoCommand`] acquires [`super::write_gate::WriteGate`] before calling
//! `undo_last` at all — a third writer alongside `city::commit`/
//! `city::demolish`, and the gate is exactly what already keeps any two of
//! the three from racing over the same region files.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::session::{WriteError, WriteSummary};
use crate::edit::EditPolicy;
use crate::region_cache::RegionCache;
use crate::LoadedSave;

use super::commit::commit_building;
use super::journal::{Journal, JournalEntry};
use super::state::{BuildingId, City};
use super::write_gate::WriteGate;
use super::write_status::{WriteKind, WriteStatus};

/// Which half of a [`JournalEntry`] an undo reversed — `city::ui::city_panel`
/// labels its status line "undid placing X" or "undid demolishing X"
/// accordingly. Read off the entry *before* [`journal::Journal::undo_last`]
/// pops it — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UndoneKind {
    Placement,
    Demolition,
}

impl std::fmt::Display for UndoneKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UndoneKind::Placement => write!(f, "placing"),
            UndoneKind::Demolition => write!(f, "demolishing"),
        }
    }
}

/// An undo's write, in flight.
struct PendingUndo {
    #[allow(dead_code)] // kept for parity with `PendingCommit`/`PendingDemolition`; not read yet
    building: BuildingId,
    definition: String,
    kind: UndoneKind,
    task: Task<Result<WriteSummary, WriteError>>,
}

/// What an undo command is doing, or last did — sticky terminal states, the
/// same reasoning [`crate::viewer::paint::PaintState`] gives.
#[derive(Default)]
pub(super) enum UndoState {
    #[default]
    Idle,
    Writing,
    Done { definition: String, kind: UndoneKind },
    Failed { message: String },
}

/// One undo request, from the click to the write — see the module docs.
#[derive(Resource, Default)]
pub(super) struct UndoCommand {
    requested: bool,
    pending: Option<PendingUndo>,
    state: UndoState,
}

impl UndoCommand {
    /// Asks for the journal's most recent entry to be undone. Ignored
    /// (returning `false`) while already busy — `city::ui::city_panel`
    /// disables its button then, so this is a backstop rather than the
    /// normal path, the same contract [`crate::viewer::paint::PaintCommand::request`]
    /// documents.
    pub(super) fn request(&mut self) -> bool {
        if self.busy() {
            return false;
        }
        self.requested = true;
        true
    }

    pub(super) fn busy(&self) -> bool {
        self.requested || self.pending.is_some()
    }

    pub(super) fn state(&self) -> &UndoState {
        &self.state
    }
}

pub struct UndoPlugin;

impl Plugin for UndoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UndoCommand>()
            // Idempotent-either-order, the same shape `WriteGate`/`WriteStatus`
            // already use across `city::commit`/`city::demolish`.
            .init_resource::<WriteGate>()
            .init_resource::<WriteStatus>()
            .add_event::<ChunksEdited>()
            .add_systems(Update, (start_undo, poll_undo).chain());
    }
}

/// Dispatches a requested undo: synchronously reverses the journal's most
/// recent entry against [`City`] (see the module docs for why that can't be
/// deferred), then commits the resulting edit onto
/// [`AsyncComputeTaskPool`].
fn start_undo(
    mut undo: ResMut<UndoCommand>,
    mut journal: ResMut<Journal>,
    mut city: ResMut<City>,
    mut write_gate: ResMut<WriteGate>,
    loaded_save: Option<Res<LoadedSave>>,
    region_cache: Option<Res<SharedRegionCache>>,
) {
    if !std::mem::take(&mut undo.requested) {
        return;
    }

    let Some(last) = journal.entries().last() else {
        undo.state = UndoState::Failed { message: "there is nothing to undo".to_string() };
        return;
    };
    let definition = last.placement().definition.clone();
    let kind = match last {
        JournalEntry::Placed { .. } => UndoneKind::Placement,
        JournalEntry::Demolished { .. } => UndoneKind::Demolition,
    };

    let (Some(loaded_save), Some(region_cache)) = (loaded_save, region_cache) else {
        undo.state = UndoState::Failed { message: "no save is loaded".to_string() };
        return;
    };

    if !write_gate.try_acquire() {
        undo.state = UndoState::Failed { message: "can't undo right now, a write is already in progress".to_string() };
        return;
    }

    // The point of no return — see the module docs: `undo_last` has already
    // mutated `City` and popped the journal entry by the time it returns.
    // A failure here (`UndoError::Occupied`) is all-or-nothing against both,
    // per that function's own docs — nothing was touched, so there's nothing
    // to write, and no write-status line to record either.
    let step = match journal.undo_last(&mut city) {
        Ok(step) => step,
        Err(err) => {
            write_gate.release();
            undo.state = UndoState::Failed { message: err.to_string() };
            return;
        }
    };

    let save_meta = loaded_save.0.meta.clone();
    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy::default();
    let edit = step.edit;

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        commit_building(&save_meta, &mut cache, &edit, &policy)
    });

    undo.pending = Some(PendingUndo { building: step.building, definition, kind, task });
    undo.state = UndoState::Writing;
}

/// Single non-blocking poll of the in-flight undo — same
/// `block_on(poll_once(..))` pattern every other task in this crate uses. A
/// failure here is reported, not rolled back — see the module docs.
fn poll_undo(mut undo: ResMut<UndoCommand>, mut write_gate: ResMut<WriteGate>, mut write_status: ResMut<WriteStatus>, mut edited: EventWriter<ChunksEdited>) {
    let result = {
        let Some(pending) = &mut undo.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still writing.
        };
        result
    };
    let PendingUndo { definition, kind, .. } = undo.pending.take().expect("just matched Some above");
    write_gate.release();

    match result {
        Ok(summary) => {
            println!(
                "block_viewer: undid {kind} {definition} ({} block(s) across {} chunk(s))",
                summary.report.blocks_written,
                summary.report.chunks.len()
            );
            write_status.record_success(WriteKind::Undo, definition.clone(), &summary);
            undo.state = UndoState::Done { definition, kind };
            edited.send(ChunksEdited(summary.report.chunks));
        }
        Err(err) => {
            println!(
                "block_viewer: undo of {definition} failed — city state was already advanced, the world was not: {err}"
            );
            write_status.record_failure(WriteKind::Undo, definition.clone(), err.to_string());
            undo.state = UndoState::Failed { message: err.to_string() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::{BlockState, Rotation};
    use bevy::math::{IVec2, IVec3};
    use bevy::tasks::TaskPool;

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(UndoPlugin).insert_resource(City::default()).insert_resource(Journal::default());
        app
    }

    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if !app.world().resource::<UndoCommand>().busy() {
                return;
            }
        }
        panic!("undo never settled");
    }

    #[test]
    fn requesting_with_an_empty_journal_fails_without_touching_city() {
        let mut app = app();
        app.world_mut().resource_mut::<UndoCommand>().request();
        run_until_settled(&mut app);

        let undo = app.world().resource::<UndoCommand>();
        assert!(matches!(undo.state(), UndoState::Failed { message } if message.contains("nothing to undo")));
        assert!(app.world().resource::<City>().is_empty());
    }

    #[test]
    fn a_request_is_refused_while_another_is_pending() {
        let mut undo = UndoCommand::default();
        assert!(!undo.busy());
        assert!(undo.request());
        assert!(undo.busy());
        assert!(!undo.request());
    }

    /// `start_undo` checks for a loaded save *before* calling `undo_last` —
    /// see the module docs' "point of no return" — so a click with nothing
    /// to write to leaves `City`/`Journal` untouched and retryable, rather
    /// than reversing state it then can't commit.
    #[test]
    fn without_a_save_neither_city_nor_the_journal_are_touched() {
        let mut app = app();
        let building = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let placement = crate::city::state::PlacedBuilding {
            definition: "house01".to_string(),
            origin: IVec3::new(0, 64, 0),
            rotation: Rotation::Deg0,
            footprint: IVec2::ONE,
        };
        let baseline = super::super::journal::Baseline {
            written: vec![(IVec3::new(0, 64, 0), BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() })],
            previous: vec![(IVec3::new(0, 64, 0), BlockState::air())],
            data_version: None,
        };
        app.world_mut().resource_mut::<Journal>().record_placement(building, placement, baseline);

        app.world_mut().resource_mut::<UndoCommand>().request();
        run_until_settled(&mut app);

        let undo = app.world().resource::<UndoCommand>();
        assert!(matches!(undo.state(), UndoState::Failed { message } if message.contains("no save is loaded")));
        assert!(!app.world().resource::<City>().is_empty(), "nothing to write to, so the reversal must not have run");
        assert_eq!(app.world().resource::<Journal>().len(), 1, "the entry must still be there to retry once a save is loaded");
    }

    #[test]
    fn a_finished_undo_reports_done_and_fires_chunks_edited() {
        let mut app = app();
        app.world_mut().resource_mut::<UndoCommand>().pending = Some(PendingUndo {
            building: super::super::state::City::default().place_building("house01", IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap(),
            definition: "house01".to_string(),
            kind: UndoneKind::Placement,
            task: pool().spawn(async {
                Ok(WriteSummary {
                    report: crate::edit::EditReport { blocks_written: 1, chunks: vec![(0, 0)], regions: vec![(0, 0)], replaced: None },
                    regions_written: vec![(0, 0)],
                    backups: vec![],
                })
            }),
        });
        app.world_mut().resource_mut::<UndoCommand>().state = UndoState::Writing;

        run_until_settled(&mut app);

        let undo = app.world().resource::<UndoCommand>();
        assert!(matches!(undo.state(), UndoState::Done { definition, kind: UndoneKind::Placement } if definition == "house01"));

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
    }

    /// The documented gap: once `undo_last` has run, a write failure is
    /// reported, not rolled back — `City`/`Journal` stay exactly where
    /// `undo_last` already left them (see the module docs). This test
    /// exercises `poll_undo`'s side of that, not `start_undo`'s — the same
    /// split `city::commit`/`city::demolish`'s own tests use, dispatching a
    /// task whose result is fixed ahead of time rather than a real fixture.
    #[test]
    fn a_write_failure_after_dispatch_is_reported_not_rolled_back() {
        let mut app = app();
        app.world_mut().resource_mut::<UndoCommand>().pending = Some(PendingUndo {
            building: super::super::state::City::default().place_building("house01", IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap(),
            definition: "house01".to_string(),
            kind: UndoneKind::Demolition,
            task: pool().spawn(async { Err(WriteError::WorldIsOpen { save: "world".to_string() }) }),
        });
        app.world_mut().resource_mut::<UndoCommand>().state = UndoState::Writing;

        run_until_settled(&mut app);

        let undo = app.world().resource::<UndoCommand>();
        assert!(matches!(undo.state(), UndoState::Failed { message } if message.contains("open in Minecraft")));

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "the write never landed, so nothing needs re-meshing");
    }
}
