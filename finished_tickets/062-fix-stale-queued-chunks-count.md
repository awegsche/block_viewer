# 062 - fix stale "queued chunks" count / pending.to_load never draining

## Follow-up: the counter fix alone didn't change the actual loading

The counter fix below is real and still correct, but the user reported
"nothing has changed" after it — the world still only showed some chunks.
The counter bug was cosmetic (it just made loading *look* stuck); it was
never the reason chunks failed to load. Two actual causes, both fixed:

1. **`RegionCache::failed` was a permanent blacklist.** A region whose
   `load_chunks()` failed once (`region_cache.rs`) was remembered as failed
   forever, for the rest of that `RegionCache` instance's life (until a save
   switch rebuilds it). On Windows a `.mca` file can fail to open
   transiently — Explorer or an antivirus scanner holding it briefly, a
   sharing violation from another process touching the save — and one bad
   moment during the initial burst of ~850 concurrent-ish chunk loads was
   enough to permanently blacklist every chunk in that region for the rest
   of the session, with no retry and no visible symptom beyond a one-time
   `block_viewer: skipping region (x, z) — failed to load ...:` console
   line. Fixed: `failed` is now a `HashMap<(i32,i32), Instant>` and a
   blacklist entry expires after `FAILED_RETRY_COOLDOWN` (10s), after which
   the next request gets a real retry.

2. **Nothing ever re-triggered the diff once the camera stopped moving.**
   `streaming::update_pending_chunk_work` only recomputed
   `PendingChunkWork` on a chunk-boundary crossing or a render-distance
   change — a coordinate that fell out of the loading pipeline for *any*
   reason (the region blacklist above, or any other silent drop) had
   nothing to notice it was still desired-but-missing and retry it. Fixed:
   the same diff now also force-recomputes every `FORCE_RECOMPUTE_INTERVAL`
   (2s) regardless of camera movement — cheap (same `HashSet` diff size as
   before) and self-healing.

Together: a transient region-load failure now heals itself within
~10-12 seconds instead of leaving a permanent hole in the loaded world.

## Report

User: "I tested the block_viewer app: when loading a new world, not all
chunks seem to get loaded. I see queued chunks: 850 and the world only
displays some chunks."

## Root cause

`chunk_pipeline::start_chunk_loads` iterated `&pending.to_load` (a shared
borrow) and, unlike its siblings `start_chunk_remeshes`/`start_chunk_reloads`,
never removed a coordinate from the list once it dispatched a task for it
(or found it already loaded/in flight). Every coordinate diffed into
`PendingChunkWork::to_load` therefore sat there forever, unchanged, until
the next camera chunk-boundary crossing recomputed the whole list from
scratch.

The status panel's "Queued chunks" counter
(`pending.to_load.len() + in_flight_loads.len()`, `viewer/ui/status.rs`) was
built on the assumption that `to_load` shrinks as work is dispatched, the
same way `PendingChunkRemeshes`/`PendingChunkReloads` do. Since it didn't,
the counter stayed pinned at the render distance's full chunk count (e.g.
850 for a ~14-chunk render distance) for as long as the camera stood still —
even while loading was actually happening correctly in the background
(loads are all dispatched into `AsyncComputeTaskPool` in the same frame;
`start_chunk_loads` already skips a coordinate that's in flight or already
loaded, so no duplicate work was ever spawned).

Net effect: the counter gave no real signal of progress, so a large render
distance's slow-but-working streaming (decode+mesh for every chunk is
serialized behind one shared `Mutex<BlockRegistry>`/`Mutex<BiomeRegistry>` —
see `chunk_pipeline`'s module docs) reads as permanently stuck rather than
"still catching up."

## Fix

`start_chunk_loads` now takes `ResMut<PendingChunkWork>` and drains
`pending.to_load` via `std::mem::take` each call, mirroring
`start_chunk_remeshes`'s `pending.0.drain()`. Every coordinate gets resolved
in the same pass it's read: spawned into `in_flight`, or dropped because
it's already loaded/already in flight. Nothing is left behind to be
re-scanned (and mis-displayed) on later frames.

## Verification

- `cargo check` — clean.
- `cargo test --lib` — all pass except two pre-existing, environment-dependent
  failures (`chunk_pipeline::tests::load_and_mesh_chunk_decodes_and_meshes_a_real_chunk`,
  `world::decode::tests::decodes_a_plausible_biome_set_from_a_real_chunk`),
  both of which depend on specifics of the real Minecraft save on this
  machine and fail the same way on `main` before this change (confirmed via
  `git stash`) — unrelated to this fix.
- Added `region_cache::tests::a_failure_older_than_the_cooldown_gets_retried`,
  which backdates a recorded failure past the cooldown (via the test-only
  `RegionCache::force_failed_stale`) and confirms the next request actually
  re-attempts the load (a fresh failure timestamp) rather than replaying the
  old blacklist entry.
- Manual/visual verification (does the "Queued chunks" counter now visibly
  count down, and do all chunks in view eventually render, for a freshly
  loaded large world) is noted in `todo.md` — needs a human watching the
  window, per this repo's convention.
