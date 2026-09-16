# 116 - Mine: the simulation tick, level transitions, persistence

Design: `MINES_DESIGN.md` ("The tick", "Order of work", "Heightmaps and
the render floor"). Depends on 113–115. This is the ticket after which a
placed Mine actually digs; 117 gives it a panel, 118 a model.

## Shape: `city::gatherer`, with a budget

`city::mine::MinePlugin` (`src/city/mine/mod.rs`) is `GathererPlugin`'s
dispatch/poll pair with the differences below. Read that module's docs
first; every reason it gives — one write in flight *per building*, no
journal, no `WriteStatus`, drops credited from the write's baseline, a
refused write refunds the carry — carries over unchanged and should be
cited rather than re-argued.

### Resources

```rust
#[derive(Resource, Default)]
pub struct MineState {
    pub progress: HashMap<BuildingId, MineProgress>,   // 114; created on first tick via `entry`
    pending: HashMap<BuildingId, PendingJob>,
}
struct PendingJob { task: Task<JobResult>, budget: u32 }
struct JobResult {
    edit: WorldEdit,
    result: Result<EditReport, EditRefusal>,
    progress: MineProgress,          // the cursor *after* every slice the job planned
    cost: u32,
    slices: Vec<(Slice, SliceOutcome)>,   // for the log line and for tests
}
```

The producer side reuses `production::Producer` exactly as the gatherer
does — `ProductionState::entry(id)`, `buffer`, `state`, and **`dig_carry`**
(its doc comment says "gatherer's dig"; widen it to "gatherer's and mine's
dig — both are rates of blocks, not of an item"). No new field.

### `dispatch_jobs`

Per placed building whose definition has `mine`:

1. `capacity = mine_buffer_capacity(mine, &economy)` (mirror of
   `gatherer_buffer_capacity`; `mine_haul_threshold` likewise — and
   `production::dispatch_hauls`'s threshold lookup learns the third block,
   next to where 112 taught it the gatherer).
2. `plan_job(producer, mine, capacity, already_pending, progress, minutes)
   -> Option<u32>` — the pure half, `plan_dig`'s twin: `MinedOut` →
   `ProducerState::MinedOut`, no carry, `None`; buffer at capacity →
   `BufferFull`, no carry, `None`; pending → `None`; accrue carry; `floor <
   1` → `None`; else `Some(budget)` with `budget = min(floor(carry),
   MAX_BLOCKS_PER_JOB)`. **The carry is not charged here** — the job
   reports what it actually removed, and settling charges that.
3. Spawn the task with the budget, the `MineFrame`, the `Mine` (cloned), a
   copy of `progress`, and the region cache `Arc`.

### The task (one region-cache lock for the whole job)

```
lock cache
loop while budget_left > 0 && slices.len() < MAX_SLICES_PER_JOB:
    slice   = progress.next_slice()            // None → break (MinedOut mid-job: fine)
    geom    = slice_geometry(..)
    survey  = survey(&geom.survey, &mut cache)  // Err → treat as Refused for this slice, stop the job
    plan    = plan_slice(&geom, .., &survey)
    edit.extend(plan.edit); cost += plan.cost; budget_left -= plan.cost (saturating)
    progress.advance(slice, plan.outcome)
    if outcome is Bedrock/Sink → break        // one Sink per job; a Bedrock face closes and the job ends
apply_building_edit(&mut cache, &edit, &MINE_POLICY)
unlock
```

`MAX_BLOCKS_PER_JOB = 96`, `MAX_SLICES_PER_JOB = 16`. The slice cap is what
bounds a **zero-cost fast-forward** (a mine whose `mines.ron` was lost
re-walks its own tunnels at cost 0 per slice — 115's idempotence — and
without the cap one job would read the whole level).

An empty merged edit (every slice already matched its target) is **not
applied** — nothing to write, `Ok` with an empty report, progress still
advances. That's the resume path.

**`MINE_POLICY`**: `capture_replaced: true`, `allow_dirty_regions: true`,
and **`heightmaps: HeightmapPolicy::Leave`, unconditionally** — the
design's "Heightmaps and the render floor" is the reasoning and the user's
explicit requirement: what the mine does underground must not change what
the citybuilder renders, while the blocks still land in the region file.
Cite it in the const's doc comment; this is the one place in the crate that
deliberately leaves a heightmap stale.

**Refusals.** `apply_building_edit` is all-or-nothing per job. On `Err`:
which slice was at fault isn't knowable, so the job's progress is
**discarded**, the carry untouched (never charged), and the *first* slice
of the job is re-planned alone next tick with `budget = 1`-slice semantics
(a `retry_single: bool` on the pending entry's successor, or simpler: on a
refusal set `MAX_SLICES` for that building's next job to 1). A single-slice
job that is refused resolves per 114: `advance(slice,
SliceOutcome::Refused)` — closes a gallery face or a secondary arm, is
logged and retried for a `Sink`. Log line: `block_viewer: mine job refused
(<building> <slice>): <err>`.

### `poll_jobs` and settling

On `Ok(report)`:
- `settle_job(producer, &edit, &report, drops, cost)`: credit
  `drops.parcel_for(baseline.previous)` via `Baseline::capture` (verbatim
  from `settle_dig`), `dig_carry -= cost as f32` (clamped at 0 — the job
  may have removed more than the budget on its last slice, deliberately;
  the excess is simply free), state `Running` unless `BufferFull`.
- Store `result.progress` into `MineState::progress[id]`.
- **`ChunksEdited` only for chunks the edit reached above their render
  floor.** For each `(cx, cz)` in `report.chunks`, look up
  `DecodedWorld.columns[(cx, cz)].floor_y`; if every edited position in that
  chunk has `y < floor_y`, don't include it — a re-decode would decode the
  same sections and remesh the same faces for nothing, and a level has
  ~2500 slices. The well's mouth and the top flights are above the floor
  and do fire, so the hole through the headframe appears. A chunk not in
  `columns` (unloaded) isn't included either — the streaming pipeline
  decodes it fresh when it comes into range.

### Level transitions

Entirely inside 114's `advance` — the tick never inspects the phase except
to label the state. `MinedOut` is `ProducerState::MinedOut` (label
`"mined out"`), added next to `Depleted` and given the same "terminal but
cheaply re-checked" treatment (`plan_job` returns early; nothing accrues).

### `is_producer` and the panels

`warehouse::is_producer` gains `|| mine.is_some()`; `inspect_panel::
producer_capacity` and `city_panel::producer_lines`' capacity lookup gain
the `mine` arm after `gatherer`. Without the first, a mine never gets a
warehouse; without the others its buffer reads `x/0`. (086's own list of
"the only call sites that had to learn a gatherer exists" — same three.)

### Persistence: `citybuilder/mines.ron`

Its own file, `production::save_logistics`/`load_logistics`'s shape
verbatim (`MinesSave { version: 1, mines: Vec<SavedMine { building: u64,
progress: MineProgress }> }`, sorted by id, `retain_placed` on load,
missing file → empty, wrong version → `UnsupportedVersion`). Wired in
`city/mod.rs` next to `load_logistics`/`save_logistics_on_exit`. A mine
whose entry is missing starts from `MineProgress::new(frame)` and
fast-forwards (see above) — write that down in the module docs as the
recovery story, since it's the reason a lost file is a nuisance and not a
corruption.

Demolish: `remove_building` → the entry is dropped by `retain_placed` on
the next save/load, and immediately by a `retain` in `poll_jobs` against
`city.buildings()` (a pending job for a demolished building settles into
nothing). The shaft and galleries stay in the world, as a gatherer's pit
does; the well's mouth is inside the placement's baseline volume and is
restored with the rest of the footprint.

### Plugin wiring

`CityPlugin` adds `MinePlugin` after `GathererPlugin`; same idempotent
`init_resource` set (`ProductionState`, `DropTable`, `ChunksEdited`).

## Tests

- `plan_job`: `MinedOut` → state + `None` + carry unchanged; capacity →
  `BufferFull`; pending → `None` with carry accrued; budget capped at
  `MAX_BLOCKS_PER_JOB`.
- The job loop as a pure fn over a `BlockSampler`-backed fake cache:
  stops at the block budget; stops at the slice cap; a `Sink` ends the
  job; a `Bedrock` outcome ends it; an all-matching survey yields an empty
  edit and advanced progress.
- `settle_job` credits drops (stone → cobblestone, ore → raw) and charges
  exactly `cost`; a refusal charges nothing.
- `ChunksEdited` suppression: an edit at `y = 20` in a chunk with
  `floor_y = 32` fires nothing; one at `y = 63` fires; a chunk absent from
  `columns` fires nothing.
- `is_producer` true for a mine-only definition; both capacity lookups
  return `buffer_stacks * stack_size`.
- `mines.ron` round-trip; demolished building dropped on load; missing
  file → empty.
- `cargo check`, `cargo test --lib`, `cargo clippy` clean.

## Manual check (todo.md — this task doesn't run the app)

Place a Mine (placeholder geometry) on flat stone-heavy terrain with a
warehouse and road; let it run. In the citybuilder: the buffer climbs,
hauls leave, state reads `running`, **nothing underground appears and the
terrain around the mine doesn't re-mesh or flicker** as galleries advance;
the well's mouth is visible through the placeholder floor. Then quit
(flush), open the save in Minecraft and walk down: spiral stairs with
landings at every corner, torches, corner pillars, a 4-wide cobblestone
corridor north and south at the bottom with pillars and torches both sides,
2×3 galleries branching east/west with one-sided torches, cobblestone where
a gallery crossed a cave, and no water in any of it.
