# 123 — Chunk streaming crawls in the citybuilder once mines are digging

## Symptom

"After the last couple of updates, chunk loading seems extremely slow."
Ticket 121's addendum had already seen it while running the game:
"~1 chunk/sec once the mines' jobs were active", parked as a separate
question.

## What it is not

- Not ticket 122. The disc it loads is the same size as the old square
  (441 chunks at the viewer's default either way; ~1000 vs 1089 for the
  citybuilder), the region-cache capacity is unchanged for those radii,
  and nothing per-frame grew. Nearest-first dispatch is also not it — see
  the side note below.
- Not ranvil's 120/121 changes. The only read-path addition is
  `normalize_palettes` (one cheap pass per section at parse); `edit_section`
  is byte-for-byte what it was, and the compact re-render only runs in
  `ChunkRegion::save`.
- Not mine jobs holding the region-cache lock. Measured
  (`city::mine::tests::probe_real_mine_job_lock_hold_time`, `#[ignore]`d):
  a real job against the real save is 2–6 ms under the lock, survey +
  plan + `apply_building_edit` included.

## What it is

Two facts multiplied together:

1. **Meshing is ~10 ms per chunk and fully serialised.** Measured
   (`chunk_pipeline::tests::probe_serial_load_throughput_over_a_real_disc`,
   `#[ignore]`d, dev profile = what `cargo run` builds): over the 441-chunk
   disc, fetch 0.1 s, decode 0.2 s, **mesh 4.5 s** — 11 ms/chunk. Every
   load, re-mesh and reload task holds the `BlockRegistry` mutex across
   decode *and* mesh (the module's "Send boundary" docs), so however many
   `AsyncComputeTaskPool` threads there are, mesh work goes through one
   at a time: ~90 chunks/s ceiling, before any neighbour re-meshes.

2. **Every mine job above the render floor buys a burst of that work.**
   `city::mine::poll_jobs` fires `ChunksEdited` for each edited chunk at or
   above the column's floor; the citybuilder's floor is
   `BelowSurface { margin: 16 }`, and a mine's shaft and first level
   (`first_level_depth: 12`) sit above it. `queue_edited_chunk_reloads`
   then queues a full reload (decode + mesh, ~11 ms) for each edited chunk
   *and* a re-mesh (~10 ms) for each of its four loaded neighbours,
   unconditionally. The probe shows real jobs touching 1–8 chunks each —
   so one job is roughly 4 reloads + 8 re-meshes ≈ 120 ms of serialised
   mesh time, and a job that walks a 200-block gallery across chunk
   borders is more.

   Job rate: `blocks_per_minute: 60` at 1x is one job per second per mine
   whenever the carry is ≥ 1; at 4x, four. And a job whose slices cost 0
   (planning through air, or re-finding already-dug tunnels) doesn't
   consume the carry, so the next one dispatches on the very next frame —
   a run of those is a job *per frame*, each still applying its non-zero
   edit list (floor tiles, sealed fluids, torches) and firing
   `ChunksEdited`.

   Three mines at 1x already put ~0.4 s of mesh work into every second;
   at 4x, or during a zero-cost run, the mines saturate the lock and
   streaming gets whatever is left — the observed ~1 chunk/s.

## Fix

In order of payoff (all three done — see "Done" below):

1. **Release the registry lock before meshing.** Decode is the only
   `&mut` user (interning), and it's 0.5 ms; mesh only reads. Lock for
   decode, `clone()` the registry (142 names in the real save — trivial)
   and the biome registry, unlock, mesh against the clones. Ids in a
   snapshot taken after this chunk's decode cover the column and every
   already-decoded neighbour. Re-mesh tasks then never block on the lock
   at all. Mesh throughput scales with the pool's thread count
   (Bevy's default gives async-compute up to 4) — the same structural
   fix the module docs already reserve for "if this ever shows up in a
   profile".
2. **Only re-mesh the neighbours an edit can actually reach.** A reload
   from a mine edit re-meshes all four neighbours; only a neighbour that
   shares a chunk border with an edited block can have changed faces.
   `ChunksEdited` carries chunk coords only; carrying the touched borders
   (or the edit's bounding box) per chunk would cut the neighbour work by
   ~2/3 for the common interior-gallery case.
3. **Coalesce mine-triggered reloads.** Consecutive jobs edit the same few
   chunks; a short per-chunk debounce (a few hundred ms) before a reload
   is dispatched, or "skip if a reload for this coord is already queued or
   in flight" (`start_chunk_reloads` deliberately keeps a re-queued coord,
   per 034's watch-out, but it could keep *one* rather than one per edit),
   turns a per-frame zero-cost run into one reload per chunk.

## Done

1. **Lock released before meshing.** `load_and_mesh_chunk` holds both
   registry locks for decode only, clones them (`BlockRegistry` and
   `BiomeRegistry` now derive `Clone`) and meshes against the snapshots;
   `remesh_chunk_column` never interns, so it only snapshots. Measured by
   the probe's new pooled pass — the same 441-chunk disc through the real
   task body on an 8-thread pool: **0.77 ms/chunk wall vs 4.3 ms serial**,
   5.6x. (The serial number came down from the 11 ms first measured; that
   run was cold. The ratio is the point.)

2. **Border-aware neighbour re-meshes, queued on completion.**
   `edit::plan` now records, per chunk, which of its four borders the edit
   wrote on (`ChunkBorders`, `EditReport::borders`, carried through
   `route::merge`'s sort as pairs). `ChunksEdited` carries
   `(chunk, borders)` pairs — `ChunksEdited::from_report` for every real
   sender, `all_borders` for one that can't know. `queue_edited_chunk_reloads`
   only queues the reloads (merging borders for a chunk edited twice before
   its turn); `poll_completed_chunk_reloads` queues the neighbours *after*
   the fresh column lands — across touched borders only, or all four if
   the reload moved the column's render floor (the floor culls a
   neighbour's faces along the whole shared side). A neighbour whose own
   reload is still in flight is re-queued for a reload with no borders of
   its own instead of re-meshed, since its running task snapshotted this
   chunk before this reload landed; the empty borders are what stop two
   adjacent chunks re-queuing each other forever.

   This also fixes a latent 034 bug: neighbours used to be re-meshed in
   the same frame the reload was *dispatched*, against the edited chunk's
   stale column, and nothing re-meshed them again once the reload landed.

3. **`ChunkReloadThrottle`** (default 250 ms): a chunk reloads at most
   once per interval. Leading-edge — a chunk not reloaded within the
   interval goes immediately, so a placed building is as snappy as before
   — with anything queued inside the interval accumulating in
   `PendingChunkReloads` and going out as one trailing reload. A mine job
   per frame is now at most four reloads per second per chunk.

Not done, deliberately: nothing about the mine's own dispatch cadence
(`plan_job`'s zero-cost jobs still go out every frame). Their cost was
the reload cascade, which 2 and 3 bound; the jobs themselves are 2-6 ms.

`cargo test --lib`: 1135 passed (was 1125 + the two `#[ignore]`d probes;
new: borders in `edit::tests`, `ChunksEdited::from_report`, the throttle,
and three `poll_completed_chunk_reloads` completion cases). Needs a human
at the window for the actual feel — see `todo.md`.

## Side note on 122's nearest-first dispatch

`diff_chunks` sorts `to_load` nearest-first and the docs say dispatch
order is therefore completion order. It's only partly so: `async_executor`
runners each grab *half of the remaining global queue* into their local
queue when they pop, so with N pool threads the sorted list is cut into N
contiguous slices (near, middle, far, farthest) that then interleave under
the registry lock. The near ring gets ~1/N of the throughput rather than
all of it. Not a slowdown, but part of why the chunk in front of the
camera doesn't land noticeably before the fog line. Fix 1 above makes
this moot for meshing; if strict near-first matters, dispatch in bounded
batches rather than all at once.

## Probes

Both `#[ignore]`d, both against the real save, both in-memory only:

```
cargo test --lib chunk_pipeline::tests::probe -- --ignored --nocapture
cargo test --lib city::mine::tests::probe -- --ignored --nocapture
```
