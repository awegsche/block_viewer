//! The gatherer's dig (ticket 086, roadmap H2): the tick that finally reads
//! [`super::definition::Gatherer`], which loaded and validated and did
//! nothing since it landed.
//!
//! ## What it reuses, and why there's no fourth resource
//!
//! A gatherer's output goes into a [`super::production::Producer`] buffer —
//! the *same* struct and the *same* [`super::production::ProductionState`]
//! map a `production` block's building already uses, keyed by the same
//! [`super::state::BuildingId`]. That one decision is what makes everything
//! downstream free: [`super::warehouse::compute_coverage`] and
//! [`super::production::dispatch_hauls`]/`deliver_arrivals` already iterate
//! every entry in that map without caring how it got filled, so a gatherer
//! hub is served by a warehouse, shows up in the city panel's producer
//! lines, and displays in the inspect panel exactly like a farm does — the
//! only two call sites that had to learn a gatherer exists at all are
//! [`super::warehouse::is_producer`] (so coverage reaches it) and the two
//! UI panels' own buffer-capacity lookups (so "x/y" doesn't read "x/0").
//!
//! ## What it doesn't reuse: `Production::outputs`
//!
//! There is no chosen recipe here — what comes out is whatever block was
//! actually standing there, resolved through [`super::drops::DropTable`]
//! the same way [`super::terraform`]'s dig already resolves one. So this
//! module never touches `partial`; it accrues a *block* count instead, in
//! [`super::production::Producer::dig_carry`] — the same fractional-carry
//! shape [`super::production::Producer::partial`]/`owed` already use, one
//! more direction over, because [`super::definition::Gatherer::blocks_per_minute`]
//! is a rate of blocks, not of one item.
//!
//! ## Where it stops, without a field to say so
//!
//! "Removes every block top to bottom until it reaches its own ground
//! level, and no deeper" needed no new field: [`super::definition::Building::ground_level`]
//! (ticket 085) already names which layer of the blueprint is this
//! building's own foundation surface, and [`super::placement::resolve_placement`]'s
//! own arithmetic (`origin.y = base_y - ground_level + y_offset`) is exactly
//! what makes `placed.origin.y + ground_level` the world Y that layer landed
//! at once placed — the same value `commit` and the ghost preview already
//! read back. [`next_gather_target`] treats that as the floor: strictly
//! *above* it is diggable, at or below it is not.
//!
//! ## Nearest tile first, never the building's own footprint
//!
//! [`next_gather_target`] scans every tile within [`Gatherer::radius_blocks`]
//! (Chebyshev, [`super::farm::rect_distance`] — the same measure and the
//! same reason [`super::definition::Farm::radius_blocks`] uses it: there's
//! no road for a gathering radius to be routed along) and picks whichever
//! still-diggable one is closest to the building's own footprint, which it
//! never enters. A `claimed` map remembers "the next Y down" per tile
//! already touched *within one dispatch*, so digging several blocks in one
//! tick empties the nearest column before moving outward rather than
//! skipping across the radius one layer at a time.
//!
//! ## One write in flight per building, not one shared slot
//!
//! Unlike [`super::commit::CommitState`]/[`super::terraform::TerraformBuildState`]'s
//! single pending slot — there is only ever one placement or one terraform
//! drag happening at a time — several gatherer huts can legitimately be
//! digging on the same tick, and making them queue behind each other would
//! slow every hub in the city down to whichever one's write is running.
//! [`GathererDigState::pending`] is keyed per [`BuildingId`] instead: a hub
//! with a write already in flight simply skips dispatch (and doesn't accrue
//! more carry) until its own settles, everyone else keeps digging.
//!
//! ## No journal, no `WriteStatus` line, same as production
//!
//! A dig writes real blocks, unlike ordinary production — but it still gets
//! no [`super::journal::Journal`] entry, for the same reason production gets
//! none: undo undoes *builds*, not the passage of time, and clawing back
//! however much ground a hub happened to level would be a different
//! mechanic wearing the same button. It also skips
//! [`super::write_status::WriteStatus`] deliberately, unlike every other
//! write path in this crate: that resource shows the city panel's *last*
//! edit, and a background dig landing every few seconds per hub would mean
//! the "Last edit" line never has a chance to show what the player actually
//! just did. [`ChunksEdited`] still fires, so the dug block disappears from
//! the mesh without a restart — the panel's silence, not the world's.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::blueprint::BlockState;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::{EditPolicy, EditRefusal, EditReport, WorldEdit};
use crate::region_cache::RegionCache;
use crate::DecodedWorld;

use super::clock::GameClock;
use super::definition::{BuildingDefinitions, Gatherer};
use super::drops::DropTable;
use super::economy::EconomyConfig;
use super::farm::rect_distance;
use super::journal::Baseline;
use super::production::{Producer, ProducerState, ProductionState};
use super::state::{footprint_extent, BuildingId, City};

/// How many stacks (`economy.stack_size`) `gatherer` can hold before it stops
/// digging — the same shape and reasoning
/// [`super::production::buffer_capacity`] gives a `Production` block's
/// `buffer_stacks`, read off [`Gatherer::buffer_stacks`] instead.
pub fn gatherer_buffer_capacity(gatherer: &Gatherer, economy: &EconomyConfig) -> u64 {
    u64::from(gatherer.buffer_stacks).saturating_mul(economy.stack_size)
}

/// The next block a gatherer should remove, or `None` when nothing within
/// [`radius`] is still above `floor_y` — see the module docs' "Nearest tile
/// first". `claimed` is a per-dispatch scratchpad (not persisted): once a
/// tile is picked, its entry remembers the *next* Y down so a second call in
/// the same batch keeps digging the same column rather than re-reading its
/// old topmost height from `world` (which the in-flight edit hasn't reached
/// yet).
fn next_gather_target(
    footprint_min: IVec2,
    footprint_max: IVec2,
    radius: i32,
    floor_y: i32,
    world: &DecodedWorld,
    claimed: &mut HashMap<IVec2, i32>,
) -> Option<IVec3> {
    let mut best: Option<(IVec2, i32, i32)> = None; // (tile, y, distance)
    for x in (footprint_min.x - radius)..(footprint_max.x + radius) {
        for z in (footprint_min.y - radius)..(footprint_max.y + radius) {
            let tile = IVec2::new(x, z);
            // Never dig the building's own footprint.
            if tile.x >= footprint_min.x && tile.x < footprint_max.x && tile.y >= footprint_min.y && tile.y < footprint_max.y {
                continue;
            }
            let distance = rect_distance(footprint_min, footprint_max, tile, tile + IVec2::ONE);
            if distance > radius {
                continue;
            }
            let y = match claimed.get(&tile) {
                Some(&next_y) => next_y,
                None => match super::terraform::topmost_block_y(tile, world) {
                    Some(top) => top,
                    None => continue, // chunk not decoded — skip, don't refuse the whole batch
                },
            };
            if y <= floor_y {
                continue; // this tile is already down to (or below) its floor
            }
            if best.is_none_or(|(_, _, best_distance)| distance < best_distance) {
                best = Some((tile, y, distance));
            }
        }
    }
    let (tile, y, _) = best?;
    claimed.insert(tile, y - 1);
    Some(IVec3::new(tile.x, y, tile.y))
}

/// Builds the `WorldEdit` for one dig batch: up to `blocks` individual
/// blocks, each the globally nearest still-diggable one, cleared to air.
/// Returns fewer than `blocks` positions (down to empty) once nothing is
/// left within radius above `floor_y` — the caller reads that as "depleted",
/// not as a partial failure.
fn gather_edit(footprint_min: IVec2, footprint_max: IVec2, radius: i32, floor_y: i32, blocks: u32, world: &DecodedWorld) -> WorldEdit {
    let mut edit = WorldEdit::new();
    let mut claimed: HashMap<IVec2, i32> = HashMap::new();
    for _ in 0..blocks {
        let Some(target) = next_gather_target(footprint_min, footprint_max, radius, floor_y, world, &mut claimed) else {
            break;
        };
        edit.set(target, BlockState::air());
    }
    edit
}

/// One gatherer's dispatch decision for this tick, factored out of
/// [`dispatch_digs`] so it's testable without a real `App`/task pool — the
/// same split [`super::production::advance_producer`] and
/// [`super::terraform::dig_edit`]/`level_edit` use for their own systems.
/// Mutates `producer`'s `dig_carry`/`state` exactly as the system would;
/// returns the edit to dispatch, or `None` when nothing should be sent this
/// tick (a full buffer, a write already in flight, still accumulating, or
/// depleted).
#[allow(clippy::too_many_arguments)]
fn plan_dig(
    producer: &mut Producer,
    gatherer: &Gatherer,
    capacity: u64,
    already_pending: bool,
    footprint_min: IVec2,
    footprint_max: IVec2,
    floor_y: i32,
    minutes: f32,
    world: &DecodedWorld,
) -> Option<WorldEdit> {
    // The cap first, same order `advance_producer` checks it in: a full
    // buffer stops everything, including the carry, so a hub sitting next
    // to an idle warehouse doesn't pointlessly keep "digging" into a buffer
    // that's already full.
    if producer.buffer.total() >= capacity {
        producer.state = ProducerState::BufferFull;
        return None;
    }
    if already_pending {
        return None; // previous dig still applying — see the module docs
    }

    producer.dig_carry += gatherer.blocks_per_minute * minutes;
    let blocks = producer.dig_carry.floor();
    if blocks < 1.0 {
        return None;
    }
    let blocks = blocks as u32;

    let edit = gather_edit(footprint_min, footprint_max, gatherer.radius_blocks as i32, floor_y, blocks, world);
    if edit.is_empty() {
        // Nothing left within radius above the floor — the site really is
        // levelled. Reset the carry rather than letting it grow forever
        // while there's nothing to spend it on; `next_gather_target` is
        // cheap enough to keep re-checking every tick after this, in case a
        // neighbouring edit puts material back in range.
        producer.dig_carry = 0.0;
        producer.state = ProducerState::Depleted;
        return None;
    }

    producer.dig_carry -= edit.len() as f32;
    producer.state = ProducerState::Running;
    Some(edit)
}

/// What settling one finished dig does to its producer — factored out of
/// [`poll_digs`] for the same testing reason [`plan_dig`] is. Success
/// credits the drops of whatever [`Baseline::capture`]'s `previous` says was
/// removed into the buffer; a refusal refunds the carry it claimed but never
/// spent, the same "unaffordable is refunded, not lost" shape ticket 073
/// gives a placement's cost, applied here to time instead of materials.
fn settle_dig(producer: &mut Producer, edit: &WorldEdit, result: &Result<EditReport, EditRefusal>, drops: &DropTable) {
    match result {
        Ok(report) => {
            if let Some(baseline) = Baseline::capture(edit, report) {
                let credited = drops.parcel_for(baseline.previous.iter().map(|(_, state)| state));
                producer.buffer.add_all(&credited);
            }
            // A dig that landed while the buffer was already at capacity
            // (the headroom check ran before this dig started, and a
            // concurrent haul hasn't emptied it since) leaves `BufferFull`
            // alone rather than stomping it back to `Running`.
            if producer.state != ProducerState::BufferFull {
                producer.state = ProducerState::Running;
            }
        }
        Err(_) => {
            producer.dig_carry += edit.len() as f32;
        }
    }
}

/// One dig, in flight — [`GathererDigState::pending`]'s value.
struct PendingDig {
    edit: WorldEdit,
    task: Task<Result<EditReport, EditRefusal>>,
}

/// One write in flight per gatherer building — see the module docs' "One
/// write in flight per building, not one shared slot".
#[derive(Resource, Default)]
struct GathererDigState {
    pending: HashMap<BuildingId, PendingDig>,
}

pub struct GathererPlugin;

impl Plugin for GathererPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GathererDigState>()
            // Idempotent-either-order shape `city::commit`/`city::terraform`
            // already document for these three resources.
            .init_resource::<ProductionState>()
            .init_resource::<DropTable>()
            .add_event::<ChunksEdited>()
            .add_systems(Update, (dispatch_digs, poll_digs).chain());
    }
}

/// Every placed gatherer's tick: accrue this frame's [`GameClock`] time,
/// dispatch a dig onto [`AsyncComputeTaskPool`] when [`plan_dig`] says to —
/// see the module docs for why this reads [`DecodedWorld`] rather than
/// waiting for a fresh one, and why a building with a write already pending
/// is skipped rather than queued.
#[allow(clippy::too_many_arguments)]
fn dispatch_digs(
    clock: Res<GameClock>,
    city: Res<City>,
    definitions: Res<BuildingDefinitions>,
    economy: Res<EconomyConfig>,
    world: Res<DecodedWorld>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut production: ResMut<ProductionState>,
    mut state: ResMut<GathererDigState>,
) {
    let minutes = clock.delta_minutes();
    if minutes <= 0.0 {
        return; // paused, or a frame the clock clamped to nothing
    }
    let Some(region_cache) = region_cache else { return };

    for (id, placed) in city.buildings() {
        let Some(definition) = placed.definition_id.as_deref().and_then(|def_id| definitions.get(def_id)) else { continue };
        let Some(gatherer) = &definition.building.gatherer else { continue };

        let capacity = gatherer_buffer_capacity(gatherer, &economy);
        let extent = footprint_extent(placed.footprint, placed.rotation);
        let footprint_min = IVec2::new(placed.origin.x, placed.origin.z);
        let footprint_max = footprint_min + extent;
        let floor_y = placed.origin.y + definition.building.ground_level as i32;
        let already_pending = state.pending.contains_key(&id);

        let producer = production.entry(id);
        let Some(edit) =
            plan_dig(producer, gatherer, capacity, already_pending, footprint_min, footprint_max, floor_y, minutes, &world)
        else {
            continue;
        };

        let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();
        let policy = EditPolicy { capture_replaced: true, allow_dirty_regions: true, ..EditPolicy::default() };
        let task_edit = edit.clone();
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut cache = cache.lock().expect("region cache mutex poisoned");
            super::commit::apply_building_edit(&mut cache, &task_edit, &policy)
        });
        state.pending.insert(id, PendingDig { edit, task });
    }
}

/// Single non-blocking poll of every in-flight dig, the same
/// `block_on(poll_once(..))` pattern [`super::commit::poll_commit`]/
/// [`super::terraform::poll_terraform`] use — one entry at a time here
/// instead of one shared slot, per the module docs.
fn poll_digs(mut state: ResMut<GathererDigState>, mut production: ResMut<ProductionState>, mut edited: EventWriter<ChunksEdited>, drops: Res<DropTable>) {
    let mut settled: Vec<(BuildingId, WorldEdit, Result<EditReport, EditRefusal>)> = Vec::new();
    state.pending.retain(|&id, pending| match block_on(poll_once(&mut pending.task)) {
        Some(result) => {
            settled.push((id, std::mem::take(&mut pending.edit), result));
            false
        }
        None => true, // still applying
    });

    for (id, edit, result) in settled {
        if let Err(err) = &result {
            println!("block_viewer: gatherer dig failed: {err}");
        }
        let chunks = if let Ok(report) = &result { Some(report.chunks.clone()) } else { None };
        settle_dig(production.entry(id), &edit, &result, &drops);
        if let Some(chunks) = chunks {
            edited.send(ChunksEdited(chunks));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap as StdHashMap;

    use bevy::tasks::TaskPool;

    use super::*;
    use crate::city::definition::Gatherer;
    use crate::city::economy::EconomyConfig;
    use crate::world::{BiomeRegistry, BlockId, BlockRegistry, ChunkColumn, ChunkSection};

    // --- gatherer_buffer_capacity --------------------------------------------

    #[test]
    fn buffer_capacity_is_stacks_times_stack_size() {
        let gatherer = Gatherer { radius_blocks: 6, blocks_per_minute: 2.0, buffer_stacks: 4 };
        let economy = EconomyConfig { stack_size: 64, ..EconomyConfig::default() };
        assert_eq!(gatherer_buffer_capacity(&gatherer, &economy), 256);
    }

    // --- a small synthetic world, same shape terraform's own tests use ------

    fn registry_with_stone() -> (BlockRegistry, BlockId) {
        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");
        (registry, stone)
    }

    /// A single chunk `(0, 0)`, flat stone at `base` across every tile
    /// `0..16`/`0..16` except whatever `overrides` says instead — `None`
    /// leaves a tile as air all the way up (undecoded reads the same as
    /// "nothing above the floor" for this module's purposes, but a genuinely
    /// air column is a distinct fixture worth being able to build).
    fn world_with_heights(base: i32, overrides: &[(IVec2, i32)]) -> DecodedWorld {
        let (registry, stone) = registry_with_stone();
        let size = crate::world::SECTION_SIZE as i32;
        let mut column = ChunkColumn { x: 0, z: 0, sections: Vec::new(), floor_y: crate::world::WORLD_MIN_Y };

        let mut set = |x: i32, z: i32, y: i32| {
            let section_y = y.div_euclid(size) as i8;
            let local_y = y.rem_euclid(size) as usize;
            let section = match column.sections.iter().position(|s| s.y == section_y) {
                Some(index) => index,
                None => {
                    column.sections.push(ChunkSection {
                        y: section_y,
                        blocks: Box::new([BlockRegistry::AIR; crate::world::SECTION_VOLUME]),
                        biomes: Box::new([BiomeRegistry::PLAINS; crate::world::BIOME_GRID_VOLUME]),
                    });
                    column.sections.len() - 1
                }
            };
            column.sections[section].blocks[ChunkSection::index(x as usize, local_y, z as usize)] = stone;
        };

        for x in 0..16 {
            for z in 0..16 {
                let height = overrides.iter().find(|(tile, _)| *tile == IVec2::new(x, z)).map(|(_, y)| *y).unwrap_or(base);
                set(x, z, height);
            }
        }

        let mut columns: StdHashMap<(i32, i32), ChunkColumn> = StdHashMap::new();
        columns.insert((0, 0), column);
        DecodedWorld { registry: Arc::new(Mutex::new(registry)), biomes: Arc::new(Mutex::new(BiomeRegistry::new())), columns }
    }

    // --- next_gather_target ---------------------------------------------------

    #[test]
    fn picks_the_nearest_diggable_tile_outside_the_footprint() {
        // Base terrain sits *at* the floor (not diggable) everywhere except
        // two raised tiles: one touching the footprint's east edge (distance
        // 0), one much further out (distance 5). Only the nearer one should
        // ever be picked.
        let world =
            world_with_heights(60, &[(IVec2::new(7, 5), 64), (IVec2::new(12, 5), 64)]);
        let mut claimed = StdHashMap::new();
        let target =
            next_gather_target(IVec2::new(5, 5), IVec2::new(7, 7), 10, 60, &world, &mut claimed).expect("something in range");
        assert_eq!(target, IVec3::new(7, 64, 5), "the east-touching tile, not one further out");
    }

    #[test]
    fn never_targets_the_buildings_own_footprint() {
        let world = world_with_heights(64, &[]);
        let mut claimed = StdHashMap::new();
        // Radius 0 from a 1x1 footprint at (0,0) reaches only the footprint
        // tile itself, which must never be a target.
        let target = next_gather_target(IVec2::new(0, 0), IVec2::new(1, 1), 0, 60, &world, &mut claimed);
        assert_eq!(target, None);
    }

    #[test]
    fn stops_at_the_floor_and_does_not_go_deeper() {
        let world = world_with_heights(64, &[]);
        let mut claimed = StdHashMap::new();
        // floor_y == 64 (the tile's own topmost block) means nothing is
        // strictly above it — already levelled.
        let target = next_gather_target(IVec2::new(5, 5), IVec2::new(6, 6), 6, 64, &world, &mut claimed);
        assert_eq!(target, None);
    }

    #[test]
    fn a_second_call_digs_the_same_column_one_layer_lower() {
        let world = world_with_heights(64, &[]);
        let mut claimed = StdHashMap::new();
        let first = next_gather_target(IVec2::new(0, 0), IVec2::new(1, 1), 6, 60, &world, &mut claimed).unwrap();
        let second = next_gather_target(IVec2::new(0, 0), IVec2::new(1, 1), 6, 60, &world, &mut claimed).unwrap();
        assert_eq!(first.y, 64);
        assert_eq!(second.y, 63, "reads the claimed map, not the stale world height");
        assert_eq!((first.x, first.z), (second.x, second.z), "same nearest tile both times");
    }

    #[test]
    fn a_tile_beyond_the_radius_never_counts() {
        let world = world_with_heights(64, &[]);
        let mut claimed = StdHashMap::new();
        // 1x1 footprint at origin, radius 1 — reaches x/z in -1..=1 only.
        let target = next_gather_target(IVec2::new(0, 0), IVec2::new(1, 1), 1, 60, &world, &mut claimed);
        assert!(target.is_some(), "something within radius 1 should be found");
        let target = target.unwrap();
        assert!((target.x - 0).abs() <= 1 + 1 && (target.z - 0).abs() <= 1 + 1);
    }

    #[test]
    fn an_undecoded_tile_is_skipped_not_a_refusal() {
        // Only chunk (0,0) is decoded; a footprint far outside it has no
        // diggable neighbours at all.
        let world = world_with_heights(64, &[]);
        let mut claimed = StdHashMap::new();
        let target = next_gather_target(IVec2::new(500, 500), IVec2::new(501, 501), 4, 60, &world, &mut claimed);
        assert_eq!(target, None);
    }

    // --- gather_edit ------------------------------------------------------

    #[test]
    fn gather_edit_removes_the_requested_number_of_blocks() {
        let world = world_with_heights(64, &[]);
        let edit = gather_edit(IVec2::new(0, 0), IVec2::new(1, 1), 6, 60, 3, &world);
        assert_eq!(edit.len(), 3);
        for e in edit.edits() {
            assert_eq!(e.state.name, "minecraft:air");
        }
    }

    #[test]
    fn gather_edit_stops_early_once_the_radius_is_exhausted() {
        // Base terrain sits at the floor (63, not diggable) everywhere
        // except one column raised to 65 — two layers above the floor, and
        // the only diggable material anywhere in range.
        let world = world_with_heights(63, &[(IVec2::new(0, 0), 65)]);
        let edit = gather_edit(IVec2::new(5, 5), IVec2::new(6, 6), 20, 63, 10, &world);
        assert_eq!(edit.len(), 2, "only (0,0) has anything above floor 63, two layers of it");
    }

    // --- plan_dig -----------------------------------------------------------

    fn gatherer(radius: u32, per_minute: f32, buffer_stacks: u32) -> Gatherer {
        Gatherer { radius_blocks: radius, blocks_per_minute: per_minute, buffer_stacks }
    }

    #[test]
    fn plan_dig_does_nothing_before_a_whole_block_accrues() {
        let world = world_with_heights(64, &[]);
        let mut producer = Producer::default();
        let gatherer = gatherer(6, 2.0, 4);
        // 0.1 minutes at 2/min is 0.2 of a block.
        let edit = plan_dig(&mut producer, &gatherer, 256, false, IVec2::new(0, 0), IVec2::new(1, 1), 60, 0.1, &world);
        assert!(edit.is_none());
        assert!(producer.dig_carry > 0.0, "the fraction is kept, not dropped");
    }

    #[test]
    fn plan_dig_dispatches_once_a_whole_block_is_owed_and_carries_the_remainder() {
        let world = world_with_heights(64, &[]);
        let mut producer = Producer::default();
        let gatherer = gatherer(6, 2.0, 4);
        // 1 minute at 2/min owes exactly 2 whole blocks.
        let edit = plan_dig(&mut producer, &gatherer, 256, false, IVec2::new(0, 0), IVec2::new(1, 1), 60, 1.0, &world);
        let edit = edit.expect("two whole blocks were owed");
        assert_eq!(edit.len(), 2);
        assert_eq!(producer.dig_carry, 0.0);
        assert_eq!(producer.state, ProducerState::Running);
    }

    #[test]
    fn plan_dig_stops_and_does_not_accrue_once_the_buffer_is_full() {
        let world = world_with_heights(64, &[]);
        let mut producer = Producer::default();
        producer.buffer.add("minecraft:stone", 256);
        let gatherer = gatherer(6, 2.0, 4);
        let edit = plan_dig(&mut producer, &gatherer, 256, false, IVec2::new(0, 0), IVec2::new(1, 1), 60, 5.0, &world);
        assert!(edit.is_none());
        assert_eq!(producer.state, ProducerState::BufferFull);
        assert_eq!(producer.dig_carry, 0.0, "a full buffer must not keep accruing carry either");
    }

    #[test]
    fn plan_dig_skips_while_a_previous_dig_is_still_pending() {
        let world = world_with_heights(64, &[]);
        let mut producer = Producer::default();
        let gatherer = gatherer(6, 2.0, 4);
        let edit = plan_dig(&mut producer, &gatherer, 256, true, IVec2::new(0, 0), IVec2::new(1, 1), 60, 5.0, &world);
        assert!(edit.is_none());
        assert_eq!(producer.dig_carry, 0.0, "no accrual while a write is already in flight");
    }

    #[test]
    fn plan_dig_goes_depleted_once_nothing_is_left_within_radius() {
        // The whole area is already at floor height.
        let world = world_with_heights(60, &[]);
        let mut producer = Producer::default();
        let gatherer = gatherer(6, 2.0, 4);
        let edit = plan_dig(&mut producer, &gatherer, 256, false, IVec2::new(0, 0), IVec2::new(1, 1), 60, 60.0, &world);
        assert!(edit.is_none());
        assert_eq!(producer.state, ProducerState::Depleted);
        assert_eq!(producer.dig_carry, 0.0, "the carry is reset rather than growing forever");
    }

    // --- settle_dig -----------------------------------------------------------

    fn report(chunks: usize, written: usize) -> EditReport {
        EditReport { blocks_written: written, chunks: vec![(0, 0)][..chunks].to_vec(), regions: vec![(0, 0)], replaced: None }
    }

    #[test]
    fn a_settled_dig_credits_the_removed_blocks_drops() {
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), BlockState::air());
        let mut rep = report(1, 1);
        rep.replaced = Some(vec![(IVec3::new(0, 64, 0), BlockState { name: "minecraft:stone".into(), properties: Vec::new() })]);

        let mut producer = Producer::default();
        settle_dig(&mut producer, &edit, &Ok(rep), &DropTable::default());

        assert_eq!(producer.buffer.get("minecraft:stone"), 1);
        assert_eq!(producer.state, ProducerState::Running);
    }

    #[test]
    fn a_settled_dig_uses_the_drop_table() {
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), BlockState::air());
        let mut rep = report(1, 1);
        rep.replaced = Some(vec![(IVec3::new(0, 64, 0), BlockState { name: "minecraft:stone".into(), properties: Vec::new() })]);

        let table_text = r#"(replaced: { "minecraft:stone": (item: "minecraft:cobblestone") })"#;
        let dir = std::env::temp_dir().join(format!("block_viewer_gatherer_drops_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("drops.ron");
        std::fs::write(&path, table_text).unwrap();
        let table = crate::city::drops::load_drop_table(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        let mut producer = Producer::default();
        settle_dig(&mut producer, &edit, &Ok(rep), &table);
        assert_eq!(producer.buffer.get("minecraft:cobblestone"), 1);
    }

    #[test]
    fn a_failed_dig_refunds_the_carry_it_claimed() {
        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), BlockState::air());
        edit.set(IVec3::new(1, 64, 0), BlockState::air());

        let mut producer = Producer::default();
        settle_dig(&mut producer, &edit, &Err(EditRefusal::Empty), &DropTable::default());

        assert_eq!(producer.dig_carry, 2.0, "both claimed blocks are handed back");
        assert!(producer.buffer.is_empty());
    }

    #[test]
    fn a_failed_dig_does_not_overwrite_buffer_full() {
        let edit = WorldEdit::new();
        let mut producer = Producer::default();
        producer.state = ProducerState::BufferFull;
        settle_dig(&mut producer, &edit, &Err(EditRefusal::Empty), &DropTable::default());
        // settle_dig's Err arm never touches `state` at all — this is really
        // documenting that, not testing new behaviour.
        assert_eq!(producer.state, ProducerState::BufferFull);
    }

    // --- the system: GathererPlugin end to end -------------------------------

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn gatherer_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(GathererPlugin);
        app
    }

    fn run_until_settled(app: &mut App, id: BuildingId) {
        for _ in 0..200 {
            app.update();
            if !app.world().resource::<GathererDigState>().pending.contains_key(&id) {
                return;
            }
        }
        panic!("gatherer dig never settled");
    }

    #[test]
    fn poll_digs_credits_the_buffer_and_fires_chunks_edited() {
        let mut app = gatherer_test_app();
        let id = BuildingId::from_u64(0);

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), BlockState::air());
        let mut rep = report(1, 1);
        rep.replaced = Some(vec![(IVec3::new(0, 64, 0), BlockState { name: "minecraft:dirt".into(), properties: Vec::new() })]);
        let task = pool().spawn(async move { Ok(rep) });

        app.world_mut().resource_mut::<GathererDigState>().pending.insert(id, PendingDig { edit, task });
        app.world_mut().resource_mut::<ProductionState>().insert(id, Producer::default());

        run_until_settled(&mut app, id);

        let production = app.world().resource::<ProductionState>();
        assert_eq!(production.get(id).unwrap().buffer.get("minecraft:dirt"), 1);

        let fired: Vec<_> = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
    }

    #[test]
    fn poll_digs_refunds_the_carry_on_failure() {
        let mut app = gatherer_test_app();
        let id = BuildingId::from_u64(0);

        let mut edit = WorldEdit::new();
        edit.set(IVec3::new(0, 64, 0), BlockState::air());
        let task = pool().spawn(async { Err(EditRefusal::Empty) });

        app.world_mut().resource_mut::<GathererDigState>().pending.insert(id, PendingDig { edit, task });
        app.world_mut().resource_mut::<ProductionState>().insert(id, Producer::default());

        run_until_settled(&mut app, id);

        let production = app.world().resource::<ProductionState>();
        assert_eq!(production.get(id).unwrap().dig_carry, 1.0);
        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0, "nothing changed in the world, so nothing needs re-meshing");
    }
}
