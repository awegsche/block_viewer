# 070 - Citybuilder render distance, and lazier unloading

## Status
Done

## Depends on
005-a (`streaming`), 005-d (`unload`), R (`city::run`'s render floor).

## Goal

Two related complaints about how much world the citybuilder keeps on screen
and for how long:

1. **Render distance is too short for the citybuilder.** `RenderDistance`
   defaults to 10 and `city::run()` never overrides it, so the RTS camera —
   which looks *across* a city rather than out of a first-person head — runs
   out of terrain well inside the frustum. The citybuilder is also the app
   that can afford more: roadmap R's `FloorPolicy::BelowSurface { margin: 16 }`
   already cuts ~7 sections per column out of decode/mesh/draw, and
   `region_cache::recommended_capacity` is flat across 10..=32 (both give 25
   resident regions), so the region cache doesn't grow with the bump.
2. **Unloading is too eager.** `streaming::update_pending_chunk_work` unloads
   the instant a column leaves the render-distance square, so panning a
   little way and coming back re-decodes and re-meshes chunks that were in
   memory seconds earlier. The observed symptom is a camera nudged back and
   forth across a chunk boundary paying a full reload each time.

## Scope

**Render distance** — `city::run()` inserts `RenderDistance(16)`. Viewer keeps
the default 10 and its 2..=32 slider.

**Retention (both apps)** — split "what to load" from "what to keep". Loading
still keys off `RenderDistance`; keeping adds two independent forms of
laziness on top, in `streaming`:

- **Hysteresis margin**: a column stays loaded out to `render_distance +
  margin` chunks. Kills boundary thrash outright — a column that leaves the
  load square is still inside the retain square, so it's never even a
  candidate.
- **Grace period**: past the retain square, a column becomes a *lingering*
  candidate with a timestamp rather than an unload. It's only actually
  unloaded once it has been outside the retain square for `grace` — coming
  back inside within the grace re-arms it for free, with mesh and decoded
  column both untouched.
- **Lingering cap**: flying in a straight line accumulates a trail of
  candidates that all sit inside their grace, so the trail needs a bound.
  Over `max_lingering` candidates, the farthest ones (Chebyshev from the
  camera's chunk) unload immediately regardless of grace.

Defaults, one set shared by both binaries: `margin: 2`, `grace: 30s`,
`max_lingering: 256`.

**Observability** — the viewer's status panel reads `Loaded chunks: N (M
lingering)`. Retention makes the loaded count exceed the render-distance
square by design, and without the split that reads like a leak; `M` is also
the only way to watch the grace and the cap actually working.

**Cancellation follows retention** — `unload::cancel_out_of_range_in_flight_work`
recomputes the desired set itself; it must recompute the *retain* set now, or
it would cancel in-flight loads for coordinates the retain ring is about to
keep. A load already in flight for a column inside the retain square is worth
finishing.

## Out of scope

- A render-distance slider for the citybuilder (its UI has no status panel;
  ticket 007's lives in `viewer::ui`).
- Any change to `region_cache` capacity — see above, the bump doesn't move it.
- Circular (rather than square) falloff, still not worth it.
- A per-frame unload budget: the cap above bounds *how many* linger, and the
  farthest-first eviction it triggers is the only place more than a ring's
  edge can unload at once.

## Watch out

- `PendingChunkWork::to_unload` is not drained by its consumer (`unload_chunks`
  reprocesses it idempotently every frame), so a lingering coordinate must
  **stay** in the lingering map after being emitted — dropping it on emit
  loses the unload if a recompute overwrites `to_unload` before
  `unload_chunks` sees it. It leaves the map when it stops being loaded.
- `Time::elapsed` is a since-startup `Duration`; subtract with `saturating_sub`.
- The retain square is `(2*(rd+margin)+1)^2` columns, so the margin costs
  memory quadratically — 2 is deliberate, not a placeholder to raise later
  without measuring.

## Done when

- `cargo check` and `cargo test` pass; new unit tests cover the margin, the
  grace, and the cap's farthest-first choice.
- A note in `todo.md` for the human-eye check (pan away and back inside the
  grace: no reload hitch; terrain reaches noticeably further in the
  citybuilder).
