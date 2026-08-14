//! Blueprint extraction (ticket 022): the selected volume
//! ([`crate::selection::SelectionBounds`]) read out of the save as an
//! in-memory [`Blueprint`] — a palette of distinct block states plus a dense
//! array of indices into it.
//!
//! [`extract`] holds the walk itself and the reasoning behind it (why it
//! reads raw NBT rather than [`crate::DecodedWorld`], how sections are
//! resolved, how the region-cache lock is held). This module is the Bevy
//! half: one extraction at a time, on [`AsyncComputeTaskPool`], started from
//! a request the selection panel makes and polled from a system the same way
//! [`crate::chunk_pipeline`] polls chunk loads.
//!
//! ## Why a task and not a system
//!
//! A million-block extraction is seconds of NBT walking. Run in a frame it
//! would freeze the window; run holding the region-cache lock throughout it
//! would stall terrain streaming as well. So it goes to the task pool, takes
//! the lock a column at a time, and publishes a coarse column count
//! ([`ExtractProgress`]) the panel can show instead of looking hung.
//!
//! ## What happens to the result
//!
//! Still nothing, as of ticket 023: the finished blueprint is logged (its
//! size, its `DataVersion` and its palette — which is what ticket 022's
//! manual check reads to confirm properties survived) and summarised for the
//! panel, then dropped. Ticket 023 landed [`structure`], which can write it
//! out; ticket 024 adds the save dialog that supplies a filename, at which
//! point [`poll_extraction`] hands the blueprint to the writer instead of
//! dropping it.

mod extract;
mod structure;

use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::chunk_pipeline::SharedRegionCache;
use crate::selection::SelectionBounds;

// `BlockState` has no caller outside this module's own logging and tests
// yet — it's the type ticket 023's writer serializes, and re-exporting it
// here is what makes `extract` an implementation detail rather than
// something 023 has to reach into. Same call as `world/mod.rs`'s re-exports.
#[allow(unused_imports)]
pub use extract::{
    extract_blueprint, BlockState, Blueprint, ExtractError, ExtractProgress, MAX_BLOCKS,
};

// The writer has no caller until ticket 024's save dialog picks a filename
// for it. Re-exported for the same reason as the types above: 024 asks
// `blueprint` for it rather than reaching into a submodule.
#[allow(unused_imports)]
pub use structure::{write_structure_file, STRUCTURE_BLOCK_MAX_SIZE};

/// How many palette entries the finished-extraction log prints before
/// summarising the rest. Long enough to see a structure's whole palette,
/// short enough not to bury the console for a selection covering half a
/// biome.
const LOGGED_PALETTE_ENTRIES: usize = 64;

/// The one extraction slot: a request waiting to start, the task running, and
/// the last finished outcome for the panel to report.
///
/// One at a time, deliberately — extraction is I/O and lock bound, so two at
/// once would finish no sooner and would double the pressure on the region
/// cache that terrain streaming shares. [`Self::request`] refuses while one
/// is in flight rather than queueing.
#[derive(Resource, Default)]
pub struct BlueprintExtraction {
    requested: Option<SelectionBounds>,
    in_flight: Option<InFlight>,
    last: Option<ExtractOutcome>,
}

struct InFlight {
    task: Task<Result<Blueprint, ExtractError>>,
    progress: Arc<ExtractProgress>,
    /// For the elapsed time in [`ExtractSummary`]. Measured on the main
    /// thread across the whole request, which is what a user experiences —
    /// the task's own runtime would exclude the frame it was dispatched on.
    started: Instant,
}

/// What the last extraction did, for the panel.
pub enum ExtractOutcome {
    Done(ExtractSummary),
    Failed(ExtractError),
}

/// The finished blueprint's shape, kept after the blueprint itself is
/// dropped. `elapsed` is here because ticket 021 picked its
/// slow/too-large thresholds without any measurement to go on and asked for
/// real timings — this is where they come from.
pub struct ExtractSummary {
    pub origin: IVec3,
    pub size: IVec3,
    pub blocks: usize,
    pub palette: usize,
    pub failed_columns: usize,
    pub elapsed: Duration,
}

impl BlueprintExtraction {
    /// Asks for `bounds` to be extracted on the next
    /// [`start_extraction`] run. Ignored (returning `false`) while an
    /// extraction is already in flight — the panel disables its button then,
    /// so this is a backstop rather than the normal path.
    pub fn request(&mut self, bounds: SelectionBounds) -> bool {
        if self.busy() {
            return false;
        }
        self.requested = Some(bounds);
        true
    }

    /// Whether an extraction is requested or running — what the panel
    /// disables its export button on.
    pub fn busy(&self) -> bool {
        self.requested.is_some() || self.in_flight.is_some()
    }

    /// `(columns done, columns total)` for a running extraction, or `None`
    /// when nothing is running.
    pub fn progress(&self) -> Option<(usize, usize)> {
        self.in_flight.as_ref().map(|f| f.progress.columns())
    }

    /// Fraction done in `0.0..=1.0` for a running extraction.
    pub fn fraction(&self) -> Option<f32> {
        self.in_flight.as_ref().map(|f| f.progress.fraction())
    }

    /// The last finished extraction's outcome, which survives until another
    /// one starts.
    pub fn last(&self) -> Option<&ExtractOutcome> {
        self.last.as_ref()
    }
}

/// Adds [`BlueprintExtraction`] and the two systems that drive it.
///
/// Ordered `poll` before `start` for the same reason
/// [`crate::chunk_pipeline`] chains its four that way: a request made this
/// frame shouldn't have to wait for the next one to be dispatched, and
/// polling first means the slot a just-finished extraction frees is
/// immediately available.
pub struct BlueprintPlugin;

impl Plugin for BlueprintPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlueprintExtraction>()
            .add_systems(Update, (poll_extraction, start_extraction).chain());
    }
}

/// Dispatches a requested extraction onto [`AsyncComputeTaskPool`].
///
/// [`SharedRegionCache`] only exists once `lib.rs::setup_world` has run against a
/// real save; without it there's nothing to read, which is a reportable
/// failure rather than a silently dropped click (ticket 008's empty-save
/// startup can leave the app in exactly that state).
fn start_extraction(
    mut extraction: ResMut<BlueprintExtraction>,
    region_cache: Option<Res<SharedRegionCache>>,
) {
    let Some(bounds) = extraction.requested.take() else {
        return;
    };
    let Some(region_cache) = region_cache else {
        extraction.last = Some(ExtractOutcome::Failed(ExtractError::NoSaveLoaded));
        return;
    };

    let progress = Arc::new(ExtractProgress::default());
    let cache = region_cache.0.clone();
    let task_progress = progress.clone();
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { extract_blueprint(bounds, &cache, &task_progress) });

    // The previous run's summary would otherwise sit under a running
    // extraction's progress line, reading as this one's result.
    extraction.last = None;
    extraction.in_flight = Some(InFlight {
        task,
        progress,
        started: Instant::now(),
    });
}

/// Single non-blocking poll of the in-flight extraction, the same
/// `block_on(poll_once(..))` pattern
/// [`crate::chunk_pipeline::poll_completed_chunk_loads`] uses.
fn poll_extraction(mut extraction: ResMut<BlueprintExtraction>) {
    let Some(mut in_flight) = extraction.in_flight.take() else {
        return;
    };
    let Some(result) = block_on(poll_once(&mut in_flight.task)) else {
        extraction.in_flight = Some(in_flight); // Still running.
        return;
    };

    let elapsed = in_flight.started.elapsed();
    extraction.last = Some(match result {
        Ok(blueprint) => {
            log_blueprint(&blueprint, elapsed);
            ExtractOutcome::Done(ExtractSummary {
                origin: blueprint.origin,
                size: blueprint.size,
                blocks: blueprint.volume(),
                palette: blueprint.palette.len(),
                failed_columns: blueprint.failed_columns,
                elapsed,
            })
            // `blueprint` is dropped here — ticket 023's writer is what will
            // consume it.
        }
        Err(err) => {
            println!("block_viewer: extraction failed: {err}");
            ExtractOutcome::Failed(err)
        }
    });
}

/// Prints the finished blueprint's shape and palette.
///
/// This is what ticket 022's manual check reads: select a staircase or a log
/// wall, extract, and confirm the properties here match what the block
/// inspector reports for the same blocks. Once ticket 024 writes real files
/// this can go, or drop to a `debug!`.
fn log_blueprint(blueprint: &Blueprint, elapsed: Duration) {
    let size = blueprint.size;
    let origin = blueprint.origin;
    println!(
        "block_viewer: extracted {}x{}x{} at ({}, {}, {}) in {:.2?} — {} blocks, \
         {} distinct states, DataVersion {}{}",
        size.x,
        size.y,
        size.z,
        origin.x,
        origin.y,
        origin.z,
        elapsed,
        blueprint.volume(),
        blueprint.palette.len(),
        blueprint.data_version,
        match blueprint.failed_columns {
            0 => String::new(),
            n => format!(" ({n} unreadable columns treated as air)"),
        }
    );
    for (index, state) in blueprint.palette.iter().take(LOGGED_PALETTE_ENTRIES).enumerate() {
        println!("block_viewer:   [{index}] {state}");
    }
    let rest = blueprint.palette.len().saturating_sub(LOGGED_PALETTE_ENTRIES);
    if rest > 0 {
        println!("block_viewer:   ... and {rest} more");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> SelectionBounds {
        SelectionBounds::from_anchor(IVec3::new(0, 64, 0))
    }

    /// One at a time: a second request while one is in flight is refused
    /// rather than queued behind it or replacing it.
    #[test]
    fn a_request_is_refused_while_another_is_pending() {
        let mut extraction = BlueprintExtraction::default();
        assert!(!extraction.busy());
        assert!(extraction.request(bounds()));
        assert!(extraction.busy());
        assert!(!extraction.request(SelectionBounds::from_anchor(IVec3::new(9, 9, 9))));
        assert_eq!(extraction.requested, Some(bounds()));
    }

    /// Nothing running means nothing to report a progress figure for — the
    /// panel shows its last summary instead.
    #[test]
    fn progress_is_absent_until_something_is_running() {
        let extraction = BlueprintExtraction::default();
        assert_eq!(extraction.progress(), None);
        assert_eq!(extraction.fraction(), None);
    }

    /// Without a save loaded there's no region cache to read, and a click
    /// should say so rather than vanish.
    #[test]
    fn a_request_without_a_region_cache_reports_no_save_loaded() {
        let mut app = App::new();
        app.add_plugins(BlueprintPlugin);
        app.world_mut()
            .resource_mut::<BlueprintExtraction>()
            .request(bounds());
        app.update();

        let extraction = app.world().resource::<BlueprintExtraction>();
        assert!(!extraction.busy(), "the request should have been consumed");
        assert!(matches!(
            extraction.last(),
            Some(ExtractOutcome::Failed(ExtractError::NoSaveLoaded))
        ));
    }
}
