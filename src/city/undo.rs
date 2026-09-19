//! Undo (ticket 050, roadmap G2): the button `journal`'s own module docs
//! have been pointing at since ticket 044 — "gives undo for free" was D3's
//! promise, and this is where it actually gets run.
//!
//! ## Following `city::commit`/`city::demolish`'s shape, with one real difference
//!
//! Same `request`/`busy`/`state` surface [`crate::viewer::paint::PaintCommand`]
//! established and `city::commit::CommitState`/`city::demolish::DemolishState`
//! already follow: a click sets [`UndoCommand::requested`], [`start_undo`]
//! dispatches onto [`AsyncComputeTaskPool`], [`poll_undo`] polls it with the
//! same `block_on(poll_once(..))` pattern every other task in this crate
//! uses. Since ticket 051, that dispatched work applies the reversal to the
//! shared region cache in memory — see `city::commit`'s module docs'
//! "Applied to memory, not written to disk" — a manual Save (`city::save`)
//! is what later writes it out.
//!
//! The one place this can't mirror commit/demolish: neither of *those*
//! decides what to write until a click supplies a target (a hovered tile, a
//! selected catalogue entry). Undo's target is just "whatever
//! [`journal::Journal::undo_last`] does" — and that call **is** the City-side
//! reversal, not a preview of it; there is no way to ask it what it would do
//! without it doing it. So [`start_undo`] checks every precondition it can
//! *without* calling it first — an empty journal, no save loaded, the same
//! two guards `try_commit_placement`/`try_demolish` already check before
//! touching anything — and only calls `undo_last` once both are clear. From
//! that point on there is no going back: [`journal::Journal::undo_last`]'s
//! own docs already name this — "this call has already moved `city` and the
//! journal on by the time it returns, on the assumption the caller commits
//! `edit` next." An apply failure after that point is not rolled back — it
//! can't be, the entry that would describe how is already gone — and
//! [`poll_undo`] says so rather than pretending otherwise. That's an
//! accepted, documented gap in D3's own contract, not one this ticket
//! introduces; checking what can be checked first just keeps it as rare as
//! possible.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport};
use crate::region_cache::RegionCache;

use super::commit::apply_building_edit;
use super::inventory::{Parcel, Stock};
use super::journal::{Journal, JournalEntry, Ledger};
use super::loading::GameplaySet;
use super::road_build::BuildingFootprintChanged;
use super::state::{BuildingId, City, PlacedBuilding};
use super::warehouse::{storage_capacity, StorageCapacity};
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
    /// The footprint this undo took out of, or put back into, [`City`] —
    /// handed to ticket 110's road re-tile once the write lands.
    placement: PlacedBuilding,
    definition: String,
    kind: UndoneKind,
    task: Task<Result<EditReport, EditRefusal>>,
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
            // Ticket 073 — same `init_resource` reasoning `WriteStatus`
            // below carries.
            .init_resource::<Stock>()
            // Idempotent-either-order, the same shape `WriteStatus` already
            // uses across `city::commit`/`city::demolish`.
            .init_resource::<WriteStatus>()
            .add_event::<ChunksEdited>()
            // Ticket 110 — idempotent, same as `ChunksEdited` above.
            .add_event::<BuildingFootprintChanged>()
            .add_systems(Update, (start_undo, poll_undo).chain().in_set(GameplaySet));
    }
}

/// Puts a journal entry's [`Ledger`] back: adds what it took, takes back what
/// it gave. The materials half of "undo reverses the entry", the same shape
/// [`journal::Journal::undo_last`] gives the world and city halves.
///
/// The entry's **recorded** numbers, not a fresh look at the definition's
/// `cost` — a definition can be edited (hot reload) between a placement and
/// its undo, and refunding a price nobody paid is exactly the bug ticket 073
/// added the ledger to avoid.
///
/// Taking back a credit is clamped by [`Stock::remove_parcel`], so undoing a
/// placement whose yield has since been spent leaves the stock at zero
/// rather than in debt. Putting a debit *back* is capped by the city's
/// storage (ticket 079) and returns whatever didn't fit, for the caller to
/// report — an undo that quietly evaporated a building's materials would be
/// the one operation in the game a player could not see going wrong. A plain function rather than two lines inside
/// [`start_undo`] so it's testable without a loaded save — `start_undo`
/// refuses before it ever reaches this without one.
fn settle_reverse(stock: &mut Stock, ledger: &Ledger, capacity: u64) -> Parcel {
    let overflow = stock.add_parcel_capped(&ledger.debited, capacity);
    stock.remove_parcel(&ledger.credited);
    overflow
}

/// Dispatches a requested undo: synchronously reverses the journal's most
/// recent entry against [`City`] (see the module docs for why that can't be
/// deferred), then applies the resulting edit onto
/// [`AsyncComputeTaskPool`].
fn start_undo(
    mut undo: ResMut<UndoCommand>,
    mut journal: ResMut<Journal>,
    mut city: ResMut<City>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut stock: ResMut<Stock>,
    capacity: Option<Res<StorageCapacity>>,
) {
    if !std::mem::take(&mut undo.requested) {
        return;
    }

    let Some(last) = journal.entries().last() else {
        undo.state = UndoState::Failed { message: "there is nothing to undo".to_string() };
        return;
    };
    let definition = last.placement().catalogue_id.clone();
    let kind = match last {
        JournalEntry::Placed { .. } => UndoneKind::Placement,
        JournalEntry::Demolished { .. } => UndoneKind::Demolition,
    };

    let Some(region_cache) = region_cache else {
        undo.state = UndoState::Failed { message: "no save is loaded".to_string() };
        return;
    };

    // The point of no return — see the module docs: `undo_last` has already
    // mutated `City` and popped the journal entry by the time it returns.
    // A failure here (`UndoError::Occupied`) is all-or-nothing against both,
    // per that function's own docs — nothing was touched, so there's nothing
    // to apply, and no write-status line to record either.
    let step = match journal.undo_last(&mut city) {
        Ok(step) => step,
        Err(err) => {
            undo.state = UndoState::Failed { message: err.to_string() };
            return;
        }
    };

    // Ticket 073: settle the entry's own ledger in reverse — put back what
    // it took, take back what it gave. The entry's recorded numbers, not a
    // fresh look at the definition's `cost`, which may have been edited (hot
    // reload) since the placement was paid for.
    //
    // Settled here, at the same point of no return the `City` mutation above
    // happens at, and outside the same guarantee: if the apply below fails,
    // the world is behind city state *and* the ledger. That gap is the one
    // the module docs already name; this ticket puts a third thing on the
    // near side of it rather than opening a new one.
    let overflow = settle_reverse(&mut stock, &step.ledger, storage_capacity(capacity.as_deref()));
    if !overflow.is_empty() {
        println!("block_viewer: storage full — {overflow} could not be put back");
    }

    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
    let policy = EditPolicy { allow_dirty_regions: true, ..EditPolicy::default() };
    let edit = step.edit;

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        apply_building_edit(&mut cache, &edit, &policy)
    });

    undo.pending = Some(PendingUndo { building: step.building, placement: step.placement, definition, kind, task });
    undo.state = UndoState::Writing;
}

/// Single non-blocking poll of the in-flight undo — same
/// `block_on(poll_once(..))` pattern every other task in this crate uses. A
/// failure here is reported, not rolled back — see the module docs.
fn poll_undo(
    mut undo: ResMut<UndoCommand>,
    mut write_status: ResMut<WriteStatus>,
    mut edited: EventWriter<ChunksEdited>,
    mut footprints: EventWriter<BuildingFootprintChanged>,
) {
    let result = {
        let Some(pending) = &mut undo.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still applying.
        };
        result
    };
    let PendingUndo { placement, definition, kind, .. } = undo.pending.take().expect("just matched Some above");

    match result {
        Ok(report) => {
            println!(
                "block_viewer: undid {kind} {definition} ({} block(s) across {} chunk(s), not yet saved to disk)",
                report.blocks_written,
                report.chunks.len()
            );
            write_status.record_success(WriteKind::Undo, definition.clone(), &report);
            undo.state = UndoState::Done { definition, kind };
            edited.send(ChunksEdited::from_report(&report));
            // Ticket 110: undoing a placement removes a footprint, undoing a
            // demolition puts one back — the road re-tile re-reads the cells
            // beside it the same way either direction.
            footprints.send(BuildingFootprintChanged(placement));
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

    fn a_placement() -> PlacedBuilding {
        PlacedBuilding {
            catalogue_id: "house01".to_string(),
            definition_id: None,
            origin: IVec3::new(0, 64, 0),
            rotation: Rotation::Deg0,
            footprint: IVec2::ONE,
            work_area: None,
        }
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

    fn parcel(items: &[(&str, u64)]) -> super::super::inventory::Parcel {
        let mut parcel = super::super::inventory::Parcel::default();
        for &(item, count) in items {
            parcel.add(item, count);
        }
        parcel
    }

    // --- ticket 073: settling the ledger in reverse -------------------------

    #[test]
    fn undoing_a_placement_returns_the_stock_to_exactly_where_it_stood() {
        // What a placement did: paid 40 planks, cleared 12 dirt out of the
        // ground. Undoing it has to be the exact inverse of that, whichever
        // order the two happened in.
        let mut stock = Stock::default();
        stock.add("minecraft:oak_planks", 60);
        let before = stock.clone();

        stock.remove("minecraft:oak_planks", 40);
        stock.add("minecraft:dirt", 12);
        settle_reverse(&mut stock, &Ledger { credited: parcel(&[("minecraft:dirt", 12)]), debited: parcel(&[("minecraft:oak_planks", 40)]) }, u64::MAX);

        assert_eq!(stock, before);
    }

    #[test]
    fn undoing_a_credit_that_has_since_been_spent_clamps_at_zero() {
        // The dirt a placement yielded was spent on something else before
        // the undo. Taking it back can't put the stock into debt — see
        // `settle_reverse`.
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 3);

        settle_reverse(&mut stock, &Ledger { credited: parcel(&[("minecraft:dirt", 12)]), debited: parcel(&[]) }, u64::MAX);

        assert_eq!(stock.count("minecraft:dirt"), 0);
    }

    #[test]
    fn undoing_an_entry_that_moved_nothing_moves_nothing() {
        // A version-1 journal entry, or a building placed free through the
        // keyboard stand-in.
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 3);
        let before = stock.clone();

        settle_reverse(&mut stock, &Ledger::default(), u64::MAX);

        assert_eq!(stock, before);
    }

    /// Ticket 079: an undo puts materials back under the city's storage cap,
    /// and hands the caller whatever didn't fit rather than dropping it.
    #[test]
    fn undoing_into_a_full_city_reports_what_would_not_fit() {
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 8);

        let overflow =
            settle_reverse(&mut stock, &Ledger { credited: parcel(&[]), debited: parcel(&[("minecraft:dirt", 6)]) }, 10);

        assert_eq!(stock.count("minecraft:dirt"), 10, "the cap is what the stock reaches");
        assert_eq!(overflow.get("minecraft:dirt"), 4, "and the rest comes back to the caller");
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
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let placement = crate::city::state::PlacedBuilding {
            catalogue_id: "house01".to_string(),
            definition_id: None,
            origin: IVec3::new(0, 64, 0),
            rotation: Rotation::Deg0,
            footprint: IVec2::ONE,
            work_area: None,
        };
        let baseline = super::super::journal::Baseline {
            written: vec![(IVec3::new(0, 64, 0), BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() })],
            previous: vec![(IVec3::new(0, 64, 0), BlockState::air())],
            data_version: None,
        };
        app.world_mut().resource_mut::<Journal>().record_placement(building, placement, baseline, Ledger::default());

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
            building: super::super::state::City::default().place_building("house01", None, IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap(),
            placement: a_placement(),
            definition: "house01".to_string(),
            kind: UndoneKind::Placement,
            task: pool().spawn(async {
                Ok(EditReport { blocks_written: 1, chunks: vec![(0, 0)], regions: vec![(0, 0)], replaced: None, ..Default::default() })
            }),
        });
        app.world_mut().resource_mut::<UndoCommand>().state = UndoState::Writing;

        run_until_settled(&mut app);

        let undo = app.world().resource::<UndoCommand>();
        assert!(matches!(undo.state(), UndoState::Done { definition, kind: UndoneKind::Placement } if definition == "house01"));

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);

        // Ticket 110: the footprint that just left `City` reaches the road
        // re-tile, carried on the step rather than looked up in a journal
        // entry that no longer exists.
        let footprints: Vec<_> = app.world_mut().resource_mut::<Events<BuildingFootprintChanged>>().drain().collect();
        assert_eq!(footprints.len(), 1);
        assert_eq!(footprints[0].0.origin, a_placement().origin);
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
            building: super::super::state::City::default().place_building("house01", None, IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap(),
            placement: a_placement(),
            definition: "house01".to_string(),
            kind: UndoneKind::Demolition,
            task: pool().spawn(async { Err(EditRefusal::ChunkNotGenerated { chunk: (5, 0) }) }),
        });
        app.world_mut().resource_mut::<UndoCommand>().state = UndoState::Writing;

        run_until_settled(&mut app);

        let undo = app.world().resource::<UndoCommand>();
        assert!(matches!(undo.state(), UndoState::Failed { message } if message.contains("has not been generated")));

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "the write never landed, so nothing needs re-meshing");
        let footprints = app.world_mut().resource_mut::<Events<BuildingFootprintChanged>>().drain().count();
        assert_eq!(footprints, 0, "the world didn't change, so no road beside the footprint should be re-tiled yet");
    }
}
