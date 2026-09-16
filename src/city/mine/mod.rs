//! Mines (tickets 113–119, `MINES_DESIGN.md`): a surface complex plus
//! everything it digs underneath itself over time.
//!
//! This module is split across the ticket sequence rather than growing in
//! one file, so 114/115/116 don't fight over the same lines:
//!
//! - [`layout`] (ticket 114): pure integer geometry — the primary shaft's
//!   ring walk, flights and lining, a mining level's rows and galleries,
//!   and the [`layout::SliceGeometry`] a slice resolves to. No Bevy, no
//!   region cache, no `DecodedWorld`.
//! - [`progress`] (ticket 114): [`progress::MineProgress`], the small
//!   cursor that walks the design's "order of work" one slice at a time.
//! - [`plan`] (ticket 115): the region-cache read ([`plan::survey`]), block
//!   classification ([`plan::classify`]), and the `WorldEdit` for one slice
//!   or one sink job ([`plan::plan_slice`]).
//! - This file (ticket 116): [`MinePlugin`], `city::gatherer`'s dispatch/poll
//!   shape applied to a budgeted, multi-slice job instead of one batch of
//!   single-block digs. Read `city::gatherer`'s module docs first — one
//!   write in flight per building, no journal, no [`super::write_status::WriteStatus`]
//!   line, drops credited from the write's baseline, a refused write refunds
//!   the carry — every reason it gives carries over unchanged and is cited
//!   rather than re-argued below.
//!
//! ## Why the carry keeps accruing while a job is in flight
//!
//! Unlike a gatherer's single-block batch (built and applied in the same
//! tick), a mine job is a region-cache read *and* write that can span
//! several frames. Freezing [`super::production::Producer::dig_carry`] for
//! that whole span — the way [`super::gatherer::plan_dig`] freezes it while
//! its own (much shorter) write is pending — would silently throttle a mine
//! to however long its jobs happen to take. [`plan_job`] accrues every tick
//! regardless, and only refuses to *spend* the carry (dispatch a new job)
//! while one is already pending.
//!
//! ## Why the carry is never charged at dispatch
//!
//! A gatherer's batch removes exactly the blocks it was asked for, so
//! [`super::gatherer::plan_dig`] can charge the carry up front and refund it
//! if the write is refused. A mine job's cost isn't known until the slice
//! loop has actually read the terrain — a `Sink` flight's cost depends on
//! how much of the shaft was already rock, and idempotent replanning can
//! cost nothing at all. [`plan_job`] hands out a *budget*; [`settle_job`]
//! charges what the job reports it actually removed.
//!
//! ## `MinedOut`, the third terminal state
//!
//! [`super::production::ProducerState::MinedOut`] sits beside `Depleted`
//! (`city::gatherer`'s "nothing left in the drawn area") with the same
//! "terminal but cheaply re-checked every tick" shape: [`plan_job`] just
//! matches on [`progress::Phase::MinedOut`] and returns immediately, no
//! different in cost from any other tick's dispatch check.
//!
//! ## Heightmaps and the render floor
//!
//! [`MINE_EDIT_POLICY`] applies every mine write with
//! [`crate::edit::HeightmapPolicy::Leave`], unconditionally —
//! `MINES_DESIGN.md`'s "Heightmaps and the render floor" is the reasoning,
//! and it is a hard requirement, not a tuning choice: what a mine digs
//! underground must not change what 030's render floor decodes for the
//! citybuilder, while the blocks themselves still land in the region file.
//! This is the one place in the crate that deliberately leaves a written
//! chunk's heightmaps stale.

pub mod layout;
pub mod plan;
pub mod progress;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use serde::{Deserialize, Serialize};

use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, HeightmapPolicy, WorldEdit};
use crate::region_cache::RegionCache;
use crate::DecodedWorld;

use super::clock::GameClock;
use super::definition::{BuildingDefinitions, Mine};
use super::drops::DropTable;
use super::economy::EconomyConfig;
use super::journal::Baseline;
use super::production::{Producer, ProducerState, ProductionState};
use super::state::{BuildingId, City};

use layout::{slice_geometry, Box3, MineFrame, Slice};
use plan::{plan_slice, survey, BlockSampler};
use progress::{MineProgress, Phase, SliceOutcome};

/// How many stacks (`economy.stack_size`) a mine can hold before it stops
/// digging — [`super::gatherer::gatherer_buffer_capacity`]'s twin, read off
/// [`Mine::buffer_stacks`] instead.
pub fn mine_buffer_capacity(mine: &Mine, economy: &EconomyConfig) -> u64 {
    u64::from(mine.buffer_stacks).saturating_mul(economy.stack_size)
}

/// How many items a mine holds before haulage ships a partial stack (ticket
/// 112) — [`super::gatherer::gatherer_haul_threshold`]'s twin.
pub fn mine_haul_threshold(mine: &Mine, economy: &EconomyConfig) -> u64 {
    u64::from(mine.haul_threshold_stacks()).saturating_mul(economy.stack_size)
}

/// A budgeted job's cap on blocks removed — see `MINES_DESIGN.md`'s "The
/// tick".
const MAX_BLOCKS_PER_JOB: u32 = 96;
/// A budgeted job's cap on slices planned, whatever the budget — what bounds
/// a zero-cost fast-forward through a mine's own already-dug tunnels (a lost
/// `mines.ron`, or a session-start resume) to a fixed amount of work per
/// tick rather than the whole level.
const MAX_SLICES_PER_JOB: usize = 16;

/// `MINES_DESIGN.md`'s "Heightmaps and the render floor": `heightmaps:
/// Leave` unconditionally, `capture_replaced` so [`settle_job`] can credit
/// drops the same way [`super::gatherer::settle_dig`] does, and
/// `allow_dirty_regions` for the same batching reason every other
/// building-edit write path in this crate turns it on.
const MINE_EDIT_POLICY: EditPolicy = EditPolicy {
    heightmaps: HeightmapPolicy::Leave,
    require_full_status: true,
    enforce_data_version: true,
    capture_replaced: true,
    allow_dirty_regions: true,
};

/// One mine job, in flight — [`MineState::pending`]'s value. Unlike
/// [`super::gatherer::PendingDig`] the edit isn't known until the task
/// finishes (it's built slice by slice, under the region-cache lock), so
/// there's nothing to hold here but the task itself and the budget it was
/// given (read by [`poll_jobs`]'s log line).
struct PendingJob {
    task: Task<JobResult>,
    budget: u32,
}

/// What one job's task computes, under the region-cache lock — the slice
/// loop's output plus whatever [`super::commit::apply_building_edit`] said
/// about writing it.
struct JobResult {
    edit: WorldEdit,
    result: Result<EditReport, EditRefusal>,
    /// The cursor after every slice the job planned, whether or not the
    /// write that follows succeeds — [`poll_jobs`] only stores this back
    /// into [`MineState::progress`] on `Ok`; see the module docs' "why the
    /// carry is never charged at dispatch" for the matching reasoning about
    /// why a refused job doesn't get to keep it either.
    progress: MineProgress,
    /// Blocks removed — what [`settle_job`] charges against `dig_carry`.
    cost: u32,
    /// For the log line, and for tests.
    slices: Vec<(Slice, SliceOutcome)>,
}

/// Every placed mine's progress cursor and in-flight job — one write in
/// flight per building, the same shape [`super::gatherer::GathererDigState`]
/// gives its own single-block digs, see the module docs.
#[derive(Resource, Default)]
pub struct MineState {
    /// [`MineProgress`] per building, created on first tick via
    /// [`dispatch_jobs`]'s `or_insert_with`. `pub` because ticket 117's
    /// panel reads it directly, the same way [`super::production::ProductionState`]
    /// is a resource of its own rather than hidden behind an accessor.
    pub progress: HashMap<BuildingId, MineProgress>,
    pending: HashMap<BuildingId, PendingJob>,
    /// Set when a job's write was refused — the *whole* job's progress is
    /// discarded (see the "Refusals" section of `MINES_DESIGN.md`'s ticket),
    /// so the next job for that building is capped to one slice, the same
    /// single-slice retry the design calls for, rather than risking losing
    /// several more slices' worth of progress to the same fault. Cleared the
    /// moment a job for that building lands successfully.
    retry_single: HashSet<BuildingId>,
}

impl MineState {
    /// The budget of the job currently in flight for `id`, if any — ticket
    /// 117's "Job: digging (N blocks budget)" line, shown only while a job
    /// is pending. The one reader of [`PendingJob::budget`] outside this
    /// module.
    pub fn pending_budget(&self, id: BuildingId) -> Option<u32> {
        self.pending.get(&id).map(|job| job.budget)
    }
}

pub struct MinePlugin;

impl Plugin for MinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MineState>()
            // Idempotent-either-order shape `city::gatherer`/`city::commit`
            // already document for these three resources.
            .init_resource::<ProductionState>()
            .init_resource::<DropTable>()
            .add_event::<ChunksEdited>()
            .add_systems(Update, (dispatch_jobs, poll_jobs).chain());
    }
}

/// One mine's dispatch decision for this tick — the pure half of
/// [`dispatch_jobs`], testable without an `App`/task pool, the same split
/// [`super::gatherer::plan_dig`] gets. Mutates `producer`'s
/// `dig_carry`/`state` exactly as the system would; returns the block budget
/// to spend on a new job, or `None` when nothing should be dispatched this
/// tick.
fn plan_job(
    producer: &mut Producer,
    mine: &Mine,
    capacity: u64,
    already_pending: bool,
    progress: &MineProgress,
    minutes: f32,
) -> Option<u32> {
    if matches!(progress.phase, Phase::MinedOut) {
        producer.state = ProducerState::MinedOut;
        return None;
    }
    if producer.buffer.total() >= capacity {
        producer.state = ProducerState::BufferFull;
        return None;
    }

    // The carry accrues even while a job is already pending — see the
    // module docs' "why the carry keeps accruing while a job is in flight".
    producer.dig_carry += mine.blocks_per_minute * minutes;

    if already_pending {
        return None;
    }

    let blocks = producer.dig_carry.floor();
    if blocks < 1.0 {
        return None;
    }

    producer.state = ProducerState::Running;
    Some((blocks as u32).min(MAX_BLOCKS_PER_JOB))
}

/// One job's slice loop — the pure half of the async task, factored out so
/// it's testable without Bevy or a real region cache (`MINES_DESIGN.md`'s
/// "Testable without a world": only this module's system touches Bevy at
/// all). `survey` stands in for a locked `RegionCache` read: `None` (a
/// failed read) is treated as [`SliceOutcome::Refused`] for that slice and
/// ends the job, the same as the design's "where a face stops early".
fn run_job<S: BlockSampler>(
    mine: &Mine,
    frame: &MineFrame,
    mut progress: MineProgress,
    budget: u32,
    max_slices: usize,
    mut survey: impl FnMut(&Box3) -> Option<S>,
) -> (WorldEdit, u32, MineProgress, Vec<(Slice, SliceOutcome)>) {
    let mut edit = WorldEdit::new();
    let mut cost = 0u32;
    let mut slices = Vec::new();
    let mut budget_left = budget;

    while budget_left > 0 && slices.len() < max_slices {
        let Some(slice) = progress.next_slice() else { break };
        let level = progress.level(frame);
        let bottom = progress.bottom;
        let geometry = slice_geometry(frame, mine, bottom, level.as_ref(), slice);
        let new_bottom = bottom - frame.level_spacing();

        let outcome = match survey(&geometry.survey) {
            Some(sampler) => {
                let plan = plan_slice(slice, &geometry, frame, new_bottom, mine, &sampler);
                for e in plan.edit.edits() {
                    edit.set(e.at, e.state.clone());
                }
                cost += plan.cost;
                budget_left = budget_left.saturating_sub(plan.cost);
                plan.outcome
            }
            None => SliceOutcome::Refused,
        };

        slices.push((slice, outcome));
        progress.advance(mine, frame, slice, outcome);

        // One flight of the shaft per job, whatever budget/slices are left —
        // and a face that hit bedrock or couldn't be read closes the job the
        // same as it closes the face; see `MINES_DESIGN.md`'s "where a face
        // stops early".
        if matches!(slice, Slice::Sink) || matches!(outcome, SliceOutcome::Bedrock | SliceOutcome::Refused) {
            break;
        }
    }

    (edit, cost, progress, slices)
}

/// What settling one finished job does to its producer — the pure half of
/// [`poll_jobs`]'s success path, [`super::gatherer::settle_dig`]'s twin.
/// Only called on `Ok`: a refusal never charges the carry in the first place
/// (see the module docs), so there is nothing for the failure path to undo.
fn settle_job(producer: &mut Producer, edit: &WorldEdit, report: &EditReport, drops: &DropTable, cost: u32) {
    if let Some(baseline) = Baseline::capture(edit, report) {
        let credited = drops.parcel_for(baseline.previous.iter().map(|(_, state)| state));
        producer.buffer.add_all(&credited);
    }
    // The job may have removed more than its budget on its last slice
    // (a slice is never split mid-way) — clamped at 0 rather than going
    // negative, the excess is simply free.
    producer.dig_carry = (producer.dig_carry - cost as f32).max(0.0);
    if producer.state != ProducerState::BufferFull {
        producer.state = ProducerState::Running;
    }
}

/// Whether any position `edit` wrote in chunk `(cx, cz)` is at or above that
/// chunk's decoded render floor — `MINES_DESIGN.md`'s "Heightmaps and the
/// render floor": a chunk where every edited position is still below the
/// floor never needs a re-decode, since nothing that changed is visible to
/// the citybuilder's mesh. A chunk not in `world.columns` (unloaded) reads
/// as `false` — the streaming pipeline decodes it fresh when it comes into
/// range, with the mine's writes already in the region file.
fn chunk_reaches_above_floor(edit: &WorldEdit, cx: i32, cz: i32, world: &DecodedWorld) -> bool {
    let Some(column) = world.columns.get(&(cx, cz)) else { return false };
    let size = crate::world::SECTION_SIZE as i32;
    edit.edits().iter().any(|e| e.at.x.div_euclid(size) == cx && e.at.z.div_euclid(size) == cz && e.at.y >= column.floor_y)
}

/// Every placed mine's tick: accrue this frame's [`GameClock`] time, dispatch
/// a budgeted job onto [`AsyncComputeTaskPool`] when [`plan_job`] says to —
/// see the module docs for why this doesn't wait on a pending job's carry,
/// and `city::gatherer`'s for why a building with a write already pending is
/// skipped rather than queued.
fn dispatch_jobs(
    clock: Res<GameClock>,
    city: Res<City>,
    definitions: Res<BuildingDefinitions>,
    economy: Res<EconomyConfig>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut production: ResMut<ProductionState>,
    mut state: ResMut<MineState>,
) {
    let minutes = clock.delta_minutes();
    if minutes <= 0.0 {
        return; // paused, or a frame the clock clamped to nothing
    }
    let Some(region_cache) = region_cache else { return };

    for (id, placed) in city.buildings() {
        let Some(definition) = placed.definition_id.as_deref().and_then(|def_id| definitions.get(def_id)) else { continue };
        let Some(mine) = &definition.building.mine else { continue };

        let frame = MineFrame::from_placement(placed, mine, definition.building.ground_level);
        let already_pending = state.pending.contains_key(&id);
        let max_slices = if state.retry_single.contains(&id) { 1 } else { MAX_SLICES_PER_JOB };

        let capacity = mine_buffer_capacity(mine, &economy);
        let producer = production.entry(id);
        let progress_entry = state.progress.entry(id).or_insert_with(|| MineProgress::new(&frame));
        let progress_snapshot = progress_entry.clone();

        let Some(budget) = plan_job(producer, mine, capacity, already_pending, &progress_snapshot, minutes) else {
            continue;
        };

        let mine_owned = mine.clone();
        let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut cache = cache.lock().expect("region cache mutex poisoned");
            let (edit, cost, progress, slices) =
                run_job(&mine_owned, &frame, progress_snapshot, budget, max_slices, |bounds| survey(bounds, &mut cache).ok());
            let result = if edit.is_empty() {
                // Every slice this job planned already matched its target —
                // the resume path, `MINES_DESIGN.md`'s "an empty merged edit
                // is not applied". Nothing to write; the cursor still moves.
                Ok(EditReport::default())
            } else {
                super::commit::apply_building_edit(&mut cache, &edit, &MINE_EDIT_POLICY)
            };
            JobResult { edit, result, progress, cost, slices }
        });
        state.pending.insert(id, PendingJob { task, budget });
    }
}

/// Single non-blocking poll of every in-flight job — [`super::gatherer::poll_digs`]'s
/// pattern, one entry at a time instead of one shared slot.
fn poll_jobs(
    mut state: ResMut<MineState>,
    mut production: ResMut<ProductionState>,
    drops: Res<DropTable>,
    world: Res<DecodedWorld>,
    mut edited: EventWriter<ChunksEdited>,
) {
    let mut settled: Vec<(BuildingId, JobResult)> = Vec::new();
    state.pending.retain(|&id, pending| match block_on(poll_once(&mut pending.task)) {
        Some(result) => {
            settled.push((id, result));
            false
        }
        None => true, // still applying
    });

    for (id, job) in settled {
        match &job.result {
            Ok(report) => {
                settle_job(production.entry(id), &job.edit, report, &drops, job.cost);
                state.progress.insert(id, job.progress);
                state.retry_single.remove(&id);

                let chunks: Vec<(i32, i32)> =
                    report.chunks.iter().copied().filter(|&(cx, cz)| chunk_reaches_above_floor(&job.edit, cx, cz, &world)).collect();
                if !chunks.is_empty() {
                    edited.send(ChunksEdited(chunks));
                }
            }
            Err(err) => {
                // Which slice was at fault isn't knowable from here — the
                // whole job's progress is discarded (never stored back into
                // `state.progress`) and the carry stays untouched (never
                // charged in the first place, see the module docs). The
                // building's next job is capped to one slice.
                let slice = job.slices.last().map(|(slice, _)| slice);
                println!("block_viewer: mine job refused ({id:?} {slice:?}): {err}");
                state.retry_single.insert(id);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// ---- persistence ---------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The `mines.ron` schema version this build writes.
pub const CURRENT_VERSION: u32 = 1;

const MINES_FILE: &str = "citybuilder/mines.ron";

fn mines_file_path(save_root: &Path) -> PathBuf {
    save_root.join(MINES_FILE)
}

/// `<save_root>/citybuilder/mines.ron`, for `super::mod`'s log lines.
pub(crate) fn mines_file_path_for_log(save_root: &Path) -> PathBuf {
    mines_file_path(save_root)
}

#[derive(Debug, Serialize, Deserialize)]
struct MinesSave {
    version: u32,
    mines: Vec<SavedMine>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedMine {
    building: u64,
    progress: MineProgress,
}

#[derive(Debug)]
pub enum MinesError {
    Io(std::io::Error),
    Parse(String),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for MinesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MinesError::Io(err) => write!(f, "{err}"),
            MinesError::Parse(msg) => write!(f, "{msg}"),
            MinesError::UnsupportedVersion(version) => {
                write!(f, "mines file is version {version}, this build reads version {CURRENT_VERSION}")
            }
        }
    }
}

impl std::error::Error for MinesError {}

/// `production::save_logistics`'s shape verbatim, applied to
/// [`MineState::progress`] — [`MineState::pending`] and `retry_single` are
/// in-flight/tick-scoped state, never written.
pub fn save_mines(state: &MineState, save_root: &Path) -> Result<(), MinesError> {
    let path = mines_file_path(save_root);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(MinesError::Io)?;
    }

    let mut mines: Vec<SavedMine> =
        state.progress.iter().map(|(&id, progress)| SavedMine { building: id.as_u64(), progress: progress.clone() }).collect();
    mines.sort_by_key(|entry| entry.building);

    let save = MinesSave { version: CURRENT_VERSION, mines };
    let text = ron::ser::to_string_pretty(&save, ron::ser::PrettyConfig::default()).map_err(|err| MinesError::Parse(err.to_string()))?;
    fs::write(&path, text).map_err(MinesError::Io)
}

/// Reads `mines.ron` and drops every entry whose building `city` no longer
/// holds — a mine demolished in a previous session leaves no cursor behind.
/// A missing file is an empty [`MineState`]: a mine whose progress is gone
/// starts from [`MineProgress::new`] and fast-forwards through its own
/// tunnels at zero cost the next time it's ticked — see the module docs'
/// "why the carry is never charged at dispatch" and `MINES_DESIGN.md`'s
/// "Idempotence" for why that's a nuisance, not a corruption.
pub fn load_mines(save_root: &Path, city: &City) -> Result<MineState, MinesError> {
    let path = mines_file_path(save_root);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(MineState::default()),
        Err(err) => return Err(MinesError::Io(err)),
    };

    let save: MinesSave = ron::de::from_str(&text).map_err(|err| MinesError::Parse(err.to_string()))?;
    if save.version != CURRENT_VERSION {
        return Err(MinesError::UnsupportedVersion(save.version));
    }

    let mut state = MineState::default();
    for entry in save.mines {
        let id = BuildingId::from_u64(entry.building);
        if city.building(id).is_some() {
            state.progress.insert(id, entry.progress);
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use bevy::math::{IVec2, IVec3};
    use bevy::tasks::TaskPool;
    use std::sync::Arc;

    use super::*;
    use crate::blueprint::{BlockState, Rotation};
    use crate::city::definition::ShaftAt;
    use crate::city::state::City;
    use crate::world::{BiomeRegistry, BlockRegistry, ChunkColumn};
    use layout::{Arm, GallerySide};

    // --- capacity/threshold --------------------------------------------------

    fn mine_with(shaft_size: u32, first_level_depth: u32, min_level_y: i32, max_void_run: u32, blocks_per_minute: f32, buffer_stacks: u32) -> Mine {
        Mine {
            shaft: ShaftAt { x: 5, z: 5 },
            shaft_size,
            first_level_depth,
            min_level_y,
            level_reach: 100,
            gallery_length: 200,
            torch_spacing: 8,
            max_void_run,
            blocks_per_minute,
            buffer_stacks,
            haul_at_stacks: None,
            valuables: Vec::new(),
        }
    }

    #[test]
    fn buffer_capacity_is_stacks_times_stack_size() {
        let mine = mine_with(6, 12, 16, 6, 60.0, 4);
        let economy = EconomyConfig { stack_size: 64, ..EconomyConfig::default() };
        assert_eq!(mine_buffer_capacity(&mine, &economy), 256);
    }

    #[test]
    fn haul_threshold_is_below_the_capacity() {
        let economy = EconomyConfig { stack_size: 8, ..EconomyConfig::default() };
        let mine = mine_with(6, 12, 16, 6, 60.0, 12);
        assert_eq!(mine_haul_threshold(&mine, &economy), 48);
        assert!(mine_haul_threshold(&mine, &economy) < mine_buffer_capacity(&mine, &economy));
    }

    // --- plan_job -------------------------------------------------------------

    fn frame(shaft_size: i32, first_level_depth: i32) -> MineFrame {
        MineFrame { shaft_min: IVec2::new(100, 200), shaft_size, floor_y: 64, first_level_depth }
    }

    #[test]
    fn plan_job_reports_mined_out_and_touches_no_carry() {
        let mine = mine_with(6, 12, 16, 6, 60.0, 4);
        let f = frame(6, 12);
        let mut producer = Producer::default();
        let progress = MineProgress { bottom: f.floor_y, phase: Phase::MinedOut };
        let budget = plan_job(&mut producer, &mine, 1_000_000, false, &progress, 5.0);
        assert!(budget.is_none());
        assert_eq!(producer.state, ProducerState::MinedOut);
        assert_eq!(producer.dig_carry, 0.0);
    }

    #[test]
    fn plan_job_stops_and_does_not_accrue_once_the_buffer_is_full() {
        let mine = mine_with(6, 12, 16, 6, 60.0, 4);
        let f = frame(6, 12);
        let mut producer = Producer::default();
        producer.buffer.add("minecraft:cobblestone", 1_000_000);
        let progress = MineProgress::new(&f);
        let budget = plan_job(&mut producer, &mine, 256, false, &progress, 5.0);
        assert!(budget.is_none());
        assert_eq!(producer.state, ProducerState::BufferFull);
        assert_eq!(producer.dig_carry, 0.0);
    }

    #[test]
    fn plan_job_accrues_the_carry_even_while_a_job_is_pending() {
        let mine = mine_with(6, 12, 16, 6, 60.0, 4);
        let f = frame(6, 12);
        let mut producer = Producer::default();
        let progress = MineProgress::new(&f);
        let budget = plan_job(&mut producer, &mine, 1_000_000, true, &progress, 1.0);
        assert!(budget.is_none(), "a job is already in flight");
        assert_eq!(producer.dig_carry, 60.0, "unlike a gatherer's dig, time keeps accruing");
    }

    #[test]
    fn plan_job_caps_the_budget_at_max_blocks_per_job() {
        let mine = mine_with(6, 12, 16, 6, 6000.0, 4);
        let f = frame(6, 12);
        let mut producer = Producer::default();
        let progress = MineProgress::new(&f);
        let budget = plan_job(&mut producer, &mine, 1_000_000, false, &progress, 1.0).expect("well over a block owed");
        assert_eq!(budget, MAX_BLOCKS_PER_JOB);
    }

    // --- run_job ----------------------------------------------------------

    fn stone_state() -> BlockState {
        BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() }
    }

    fn water_state() -> BlockState {
        BlockState { name: "minecraft:water".to_string(), properties: Vec::new() }
    }

    fn bedrock_state() -> BlockState {
        BlockState { name: "minecraft:bedrock".to_string(), properties: Vec::new() }
    }

    /// A fixed block state at every position — cheap to `Clone` per
    /// `survey()` call, unlike a real [`plan::Blueprint`] read.
    #[derive(Clone)]
    struct UniformWorld(BlockState);

    impl BlockSampler for UniformWorld {
        fn block(&self, _at: IVec3) -> Option<&BlockState> {
            Some(&self.0)
        }
    }

    #[derive(Clone)]
    struct MapWorld(HashMap<IVec3, BlockState>);

    impl BlockSampler for MapWorld {
        fn block(&self, at: IVec3) -> Option<&BlockState> {
            self.0.get(&at)
        }
    }

    /// `mine_with`'s frame, with `first_level_depth` equal to one flight
    /// (`shaft_size - 2`) so a single `Sink` slice lands directly in
    /// `Phase::Mining` — every `run_job` test below that needs a mining
    /// level starts from this rather than driving a multi-flight sink.
    fn one_flight_mine() -> (Mine, MineFrame) {
        let f = frame(6, 4); // level_spacing == 4 == first_level_depth
        let mine = mine_with(6, 4, -1000, 6, 60.0, 512);
        (mine, f)
    }

    #[test]
    fn run_job_stops_at_the_block_budget() {
        let (mine, f) = one_flight_mine();
        let mut progress = MineProgress::new(&f);
        progress.advance(&mine, &f, Slice::Sink, SliceOutcome::Dug { cost: 10 }); // -> Phase::Mining, RowStep::Secondary

        let world = UniformWorld(stone_state());
        let (edit, cost, _progress, slices) = run_job(&mine, &f, progress, 13, MAX_SLICES_PER_JOB, |_| Some(world.clone()));

        assert_eq!(slices.len(), 2, "12/slice: the third would need 12 more than the 1 left");
        assert_eq!(cost, 24);
        assert!(!edit.is_empty());
    }

    #[test]
    fn run_job_stops_at_the_slice_cap() {
        let (mine, f) = one_flight_mine();
        let mut mine = mine;
        mine.max_void_run = 1_000_000; // never auto-close a face mid-test
        let mut progress = MineProgress::new(&f);
        progress.advance(&mine, &f, Slice::Sink, SliceOutcome::Dug { cost: 10 });

        let world = UniformWorld(BlockState::air());
        let (_edit, cost, _progress, slices) = run_job(&mine, &f, progress, 1_000_000, 3, |_| Some(world.clone()));

        assert_eq!(slices.len(), 3, "the slice cap, not the (huge) budget, stops this job");
        assert_eq!(cost, 0, "open air costs nothing to walk through — Secondary always floors itself regardless");
    }

    #[test]
    fn run_job_ends_after_one_sink_slice_and_advances_into_mining() {
        let (mine, f) = one_flight_mine();
        let progress = MineProgress::new(&f);
        assert!(matches!(progress.phase, Phase::Sinking { .. }));

        let world = UniformWorld(stone_state());
        let (_edit, _cost, progress, slices) = run_job(&mine, &f, progress, 1_000_000, MAX_SLICES_PER_JOB, |_| Some(world.clone()));

        assert_eq!(slices.len(), 1, "one Sink per job, whatever budget/slices remain");
        assert_eq!(slices[0].0, Slice::Sink);
        assert!(matches!(progress.phase, Phase::Mining(_)), "one flight was the whole first level's depth");
    }

    #[test]
    fn run_job_ends_on_bedrock() {
        let (mine, f) = one_flight_mine();
        let progress = MineProgress::new(&f);

        // Fill the sink's survey box with stone, then plant bedrock at one
        // ring tile so `plan::plan_sink`'s own bedrock check fires.
        let level = progress.level(&f);
        let geometry = slice_geometry(&f, &mine, progress.bottom, level.as_ref(), Slice::Sink);
        let mut map = HashMap::new();
        for pos in geometry.survey.iter() {
            map.insert(pos, stone_state());
        }
        let ring_tile = f.ring().next().unwrap().tile;
        map.insert(IVec3::new(ring_tile.x, progress.bottom, ring_tile.y), bedrock_state());
        let world = MapWorld(map);

        let (edit, cost, progress_after, slices) =
            run_job(&mine, &f, progress.clone(), 1_000_000, MAX_SLICES_PER_JOB, |_| Some(world.clone()));

        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].1, SliceOutcome::Bedrock);
        assert!(edit.is_empty());
        assert_eq!(cost, 0);
        assert_eq!(progress_after.bottom, progress.bottom, "a bedrock Sink changes nothing — 116 retries next tick");
    }

    #[test]
    fn run_job_survey_failure_ends_the_job_as_refused() {
        let (mine, f) = one_flight_mine();
        let mut progress = MineProgress::new(&f);
        progress.advance(&mine, &f, Slice::Sink, SliceOutcome::Dug { cost: 10 }); // -> Phase::Mining

        let (edit, cost, _progress, slices) =
            run_job::<UniformWorld>(&mine, &f, progress, 1_000_000, MAX_SLICES_PER_JOB, |_| None);

        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].1, SliceOutcome::Refused);
        assert!(edit.is_empty());
        assert_eq!(cost, 0);
    }

    /// `MINES_DESIGN.md`'s "Idempotence" through `run_job`: replanning a
    /// slice already dug against a world reflecting the first job's own
    /// writes costs nothing and writes nothing — mirrors `plan`'s own
    /// `replanning_a_dug_slice_is_free_and_never_void` fixture one level up.
    #[test]
    fn run_job_replanning_a_dug_gallery_slice_is_free() {
        let (mine, f) = one_flight_mine();
        let mut progress = MineProgress::new(&f);
        progress.advance(&mine, &f, Slice::Sink, SliceOutcome::Dug { cost: 10 });
        for d in 0..4 {
            progress.advance(&mine, &f, Slice::Secondary { arm: Arm::North, distance: d }, SliceOutcome::Dug { cost: 1 });
        }
        assert_eq!(
            progress.next_slice(),
            Some(Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance: 0 })
        );

        let level = progress.level(&f);
        let slice = progress.next_slice().unwrap();
        let geometry = slice_geometry(&f, &mine, progress.bottom, level.as_ref(), slice);
        let mut initial = HashMap::new();
        for pos in geometry.survey.iter() {
            initial.insert(pos, stone_state());
        }
        // The floor starts non-solid so the first dig marks it GROUND — the
        // tell `plan`'s own module docs describe.
        for f in &geometry.floor {
            initial.insert(IVec3::new(f.x, geometry.floor_y, f.y), water_state());
        }
        let world = MapWorld(initial.clone());

        let (edit1, cost1, progress1, slices1) = run_job(&mine, &f, progress.clone(), 1_000_000, 1, |_| Some(world.clone()));
        assert_eq!(slices1[0].1, SliceOutcome::Dug { cost: cost1 });
        assert!(cost1 > 0);

        let mut dug = initial;
        for e in edit1.edits() {
            dug.insert(e.at, e.state.clone());
        }
        let dug_world = MapWorld(dug);

        let (edit2, cost2, _progress2, slices2) = run_job(&mine, &f, progress1, 1_000_000, 1, |_| Some(dug_world.clone()));
        assert!(edit2.is_empty());
        assert_eq!(cost2, 0);
        assert_eq!(slices2[0].1, SliceOutcome::Dug { cost: 0 }, "the floor's GROUND tell keeps this from reading as Void");
    }

    // --- settle_job -----------------------------------------------------------

    fn report(chunks: usize, written: usize) -> EditReport {
        EditReport { blocks_written: written, chunks: vec![(0, 0)][..chunks].to_vec(), regions: vec![(0, 0)], replaced: None }
    }

    #[test]
    fn settle_job_credits_drops_and_charges_exactly_cost() {
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), BlockState::air());
        let mut rep = report(1, 1);
        rep.replaced = Some(vec![(IVec3::new(0, 64, 0), stone_state())]);

        let mut producer = Producer::default();
        producer.dig_carry = 5.0;
        settle_job(&mut producer, &edit, &rep, &DropTable::default(), 1);

        assert_eq!(producer.buffer.get("minecraft:stone"), 1);
        assert_eq!(producer.dig_carry, 4.0);
        assert_eq!(producer.state, ProducerState::Running);
    }

    #[test]
    fn settle_job_clamps_the_carry_at_zero_when_the_job_overspent_it() {
        let edit = WorldEdit::new();
        let rep = report(0, 0);
        let mut producer = Producer::default();
        producer.dig_carry = 2.0;
        settle_job(&mut producer, &edit, &rep, &DropTable::default(), 50);
        assert_eq!(producer.dig_carry, 0.0, "the excess on the last slice is simply free");
    }

    #[test]
    fn settle_job_does_not_overwrite_buffer_full() {
        let edit = WorldEdit::new();
        let rep = report(0, 0);
        let mut producer = Producer::default();
        producer.state = ProducerState::BufferFull;
        settle_job(&mut producer, &edit, &rep, &DropTable::default(), 0);
        assert_eq!(producer.state, ProducerState::BufferFull);
    }

    // --- chunk_reaches_above_floor ---------------------------------------

    fn world_with_floor(cx: i32, cz: i32, floor_y: i32) -> DecodedWorld {
        let registry = BlockRegistry::new();
        let column = ChunkColumn { x: cx, z: cz, sections: Vec::new(), floor_y };
        let mut columns: HashMap<(i32, i32), ChunkColumn> = HashMap::new();
        columns.insert((cx, cz), column);
        DecodedWorld { registry: Arc::new(Mutex::new(registry)), biomes: Arc::new(Mutex::new(BiomeRegistry::new())), columns }
    }

    #[test]
    fn a_chunk_where_every_edit_is_below_the_floor_is_suppressed() {
        let world = world_with_floor(0, 0, 32);
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(1, 20, 1), BlockState::air());
        assert!(!chunk_reaches_above_floor(&edit, 0, 0, &world));
    }

    #[test]
    fn a_chunk_with_one_edit_above_the_floor_fires() {
        let world = world_with_floor(0, 0, 32);
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(1, 20, 1), BlockState::air());
        edit.set(IVec3::new(2, 63, 2), BlockState::air());
        assert!(chunk_reaches_above_floor(&edit, 0, 0, &world));
    }

    #[test]
    fn a_chunk_missing_from_columns_never_fires() {
        let world = world_with_floor(0, 0, 32);
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(1, 63, 1), BlockState::air());
        assert!(!chunk_reaches_above_floor(&edit, 5, 5, &world));
    }

    // --- persistence --------------------------------------------------------

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("block_viewer_mines_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn city_with_one_building() -> (City, BuildingId) {
        let mut city = City::default();
        let id = city.place_building("mine01", Some("mine01".to_string()), IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap();
        (city, id)
    }

    #[test]
    fn a_missing_mines_file_is_an_empty_state_not_an_error() {
        let dir = temp_dir("missing");
        let (city, _) = city_with_one_building();
        assert_eq!(load_mines(&dir, &city).unwrap().progress.len(), 0);
    }

    #[test]
    fn a_mine_progress_round_trips() {
        let dir = temp_dir("round_trip");
        let (city, id) = city_with_one_building();
        let f = frame(6, 12);

        let mut state = MineState::default();
        state.progress.insert(id, MineProgress::new(&f));
        save_mines(&state, &dir).unwrap();

        let loaded = load_mines(&dir, &city).unwrap();
        assert_eq!(loaded.progress.get(&id), Some(&MineProgress::new(&f)));
    }

    #[test]
    fn a_demolished_mines_progress_is_dropped_on_load() {
        let dir = temp_dir("orphan");
        let (city, id) = city_with_one_building();
        let f = frame(6, 12);

        let mut state = MineState::default();
        state.progress.insert(id, MineProgress::new(&f));
        state.progress.insert(BuildingId::from_u64(999), MineProgress::new(&f));
        save_mines(&state, &dir).unwrap();

        let loaded = load_mines(&dir, &city).unwrap();
        assert_eq!(loaded.progress.len(), 1);
        assert!(loaded.progress.get(&BuildingId::from_u64(999)).is_none());
    }

    #[test]
    fn a_version_mismatch_is_reported_for_the_caller_to_start_empty_on() {
        let dir = temp_dir("version");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(dir.join(MINES_FILE), "(version: 999, mines: [])").unwrap();

        let (city, _) = city_with_one_building();
        assert!(matches!(load_mines(&dir, &city), Err(MinesError::UnsupportedVersion(999))));
    }

    // --- the system: MinePlugin end to end -----------------------------------

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn mine_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinePlugin);
        app.insert_resource(world_with_floor(0, 0, 32));
        app
    }

    fn run_until_settled(app: &mut App, id: BuildingId) {
        for _ in 0..200 {
            app.update();
            if !app.world().resource::<MineState>().pending.contains_key(&id) {
                return;
            }
        }
        panic!("mine job never settled");
    }

    #[test]
    fn poll_jobs_credits_the_buffer_and_stores_progress_on_success() {
        let mut app = mine_test_app();
        let id = BuildingId::from_u64(0);
        let f = frame(6, 12);

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 63, 0), BlockState::air());
        let mut rep = report(1, 1);
        rep.replaced = Some(vec![(IVec3::new(0, 63, 0), stone_state())]);
        let new_progress = MineProgress::new(&f);
        let task = pool().spawn(async move {
            JobResult { edit, result: Ok(rep), progress: new_progress, cost: 1, slices: Vec::new() }
        });

        app.world_mut().resource_mut::<MineState>().pending.insert(id, PendingJob { task, budget: 1 });
        app.world_mut().resource_mut::<ProductionState>().insert(id, Producer::default());

        run_until_settled(&mut app, id);

        let production = app.world().resource::<ProductionState>();
        assert_eq!(production.get(id).unwrap().buffer.get("minecraft:stone"), 1);
        let state = app.world().resource::<MineState>();
        assert!(state.progress.contains_key(&id));
        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1, "the edit at y=63 is above the fixture's floor_y=32");
    }

    #[test]
    fn poll_jobs_discards_progress_and_marks_retry_single_on_refusal() {
        let mut app = mine_test_app();
        let id = BuildingId::from_u64(0);
        let f = frame(6, 12);

        let mut producer = Producer::default();
        producer.dig_carry = 3.0;
        app.world_mut().resource_mut::<ProductionState>().insert(id, producer);

        let edit = WorldEdit::new();
        let progress = MineProgress::new(&f);
        let task = pool().spawn(async move {
            JobResult { edit, result: Err(EditRefusal::Empty), progress, cost: 0, slices: vec![(Slice::Sink, SliceOutcome::Dug { cost: 0 })] }
        });
        app.world_mut().resource_mut::<MineState>().pending.insert(id, PendingJob { task, budget: 1 });

        run_until_settled(&mut app, id);

        let state = app.world().resource::<MineState>();
        assert!(!state.progress.contains_key(&id), "the job's progress is discarded, never stored");
        assert!(state.retry_single.contains(&id));
        let production = app.world().resource::<ProductionState>();
        assert_eq!(production.get(id).unwrap().dig_carry, 3.0, "never charged, so there is nothing to refund");
    }
}
