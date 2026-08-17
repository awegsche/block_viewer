# 046 - Grid and footprint fit

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 385 tests,
up from 376). See the Resolution.

## Part of
Roadmap E2 (`tickets/CITYBUILDER_ROADMAP.md`), the second ticket in group E
(placement). Depends on D1's `City` (ticket 042, `footprint_tiles`/
`footprint_extent`) and E1's `DecodedWorld`-backed picking (ticket 045) —
reuses both rather than inventing a second tile grid or a second block
lookup. Feeds E3 (ghost preview + validity) and E4 (commit) directly, and
the roadmap notes its terrain-fit answer feeds H (terraforming) too.

## Problem

`City::place_building` (042) checks *occupancy* — is a tile already claimed
— but nothing yet checks *terrain*: whether the ground under a footprint is
actually there to build on, and whether it's flat enough. Without this, E3's
ghost preview has no validity signal to tint by beyond "is the tile free",
and E4's commit has no `base_y` to write the blueprint at.

## Goal

`city::grid::fit_footprint(origin, footprint, rotation, &DecodedWorld) ->
FootprintFit` — samples the ground height at every tile
`footprint_tiles` covers and decides:

- **Fits { base_y }** — the sampled ground is level within a small
  tolerance; `base_y` is where the building's floor belongs.
- **Refused(FitError)** — either a tile's chunk isn't decoded yet
  (`NotLoaded`), or the ground varies too much (`TooSteep`).

Read-only: no world writes, no auto-levelling. See "Watch out" for why.

## Scope

- `city::grid` (new, private module like `state`/`picking`).
- `ground_height_at(tile, &DecodedWorld) -> Option<i32>` — one above the
  topmost non-air block at `tile`, via `ChunkColumn::topmost_non_air`
  against `DecodedWorld.columns` directly (no `RegionCache`, no I/O — the
  same already-decoded data E1's picking reads through
  `camera::block_under_cursor`). `None` if the tile's chunk isn't in
  `DecodedWorld.columns`, or has no solid block in what's decoded.
- `fit_footprint(origin, footprint, rotation, &DecodedWorld) ->
  FootprintFit`, walking `super::state::footprint_tiles` (not a second tile
  walk) and failing fast on the first tile that comes back `None`.
- `MAX_FOOTPRINT_STEP: i32` — the height-variance tolerance. `base_y` is the
  footprint's *lowest* sampled point, not the highest: a low corner
  clipping a little into the building's own foundation reads better than a
  gap floating under it, given nothing here fills the gap.
- "Ground" reuses `world::is_solid`'s notion (not air) via
  `topmost_non_air`, the same predicate the mesher already culls faces
  against — not a new fluid-aware classifier. A lake surface counts as
  ground for now; that's exactly the kind of call I3's damage
  classification table will refine later, not this ticket's job to
  pre-empt.

## Watch out

- **No auto-level.** The roadmap explicitly asks this ticket to decide
  auto-level vs refuse. Levelling means writing blocks — W4/W5's job via a
  real `WorldEdit` — and folding that into a read-only fit check would make
  "is this buildable" secretly depend on the write path, paid for by every
  future caller including E3's every-frame ghost preview. Refuse past
  `MAX_FOOTPRINT_STEP`; terraforming (H1) is the deliberate, later, opt-in
  way to fix a steeper site.
- **Don't re-walk the footprint.** `super::state::footprint_tiles` already
  handles rotation's axis swap (D1); a second implementation here is exactly
  the kind of drift the roadmap's B3/D1 write-ups warn about.
- **Don't touch `RegionCache`.** Sampling only what's already decoded is
  deliberate — matches E1's picking, keeps this synchronous and cheap enough
  to call every frame from E3's ghost preview, and means a footprint reaching
  past the streamed radius reads as `NotLoaded`, not a stall.
- **Guard the empty-footprint edge.** A footprint with a zero extent on
  either axis (not producible by the catalogue — 039 refuses a zero-size
  blueprint axis — but not guarded against by `footprint_tiles` either)
  samples no tiles at all; computing `max_y - min_y` off unset sentinels
  must not overflow/panic. Track the accumulated bounds as an `Option`.

## Out of scope

- Ghost mesh, validity tinting, or any rendering (E3).
- Committing a placement, or anything touching `WorldEdit`/W4/W5 (E4).
- A fluid-aware ground classifier (see the scope note above).
- Auto-levelling terrain (H1, later, and opt-in).

## Done when

- `cargo build`/`cargo test` clean.
- Tests: flat ground under a footprint fits at the sampled height; ground
  varying by more than `MAX_FOOTPRINT_STEP` is refused as `TooSteep`; a
  footprint reaching a tile whose chunk isn't in `DecodedWorld.columns` is
  refused as `NotLoaded`, without sampling being required to reach every
  tile first; a 90°/270° rotation samples the *rotated* rectangle's tiles,
  not the unrotated one (an asymmetric-terrain fixture, so a rotation bug
  would actually be caught); a footprint straddling a chunk boundary reads
  ground from both `ChunkColumn`s correctly; a zero-extent footprint doesn't
  panic.

## Resolution

Landed as scoped, in a new `city/grid.rs` (private module, declared next to
`definition`/`journal`/`persistence`/`picking`/`state` in `city/mod.rs`).

`ground_height_at(tile, &DecodedWorld) -> Option<i32>` converts a tile to
its chunk coordinate (`div_euclid`/`rem_euclid` by `world::SECTION_SIZE`,
the same conversion `region_cache::chunk_to_region_coord` and
`camera::raycast_terrain` each already do their own version of), looks the
column up in `DecodedWorld.columns`, and delegates to
`ChunkColumn::topmost_non_air` -- the exact function its own doc comment
already named this ticket as the eventual caller of, so no second block
lookup was written. `fit_footprint` walks `super::state::footprint_tiles`
directly (D1's rotation-aware tile walk, not a second one) and accumulates
`(min, max)` as an `Option<(i32, i32)>` rather than `i32::MAX`/`MIN`
sentinels, per the ticket's own watch-out -- a zero-extent footprint leaves
the accumulator `None` and returns `Fits { base_y: origin.y }` instead of
computing `max_y - min_y` off unset sentinels and overflowing.

`base_y` is the footprint's *lowest* sampled point (not the highest, not an
average) -- the module docs spell out why: nothing here writes blocks, so a
`base_y` above the low corners would leave a visible floating gap under
them with no auto-level to close it, whereas the low corners of the
building's own foundation quietly absorbing a `MAX_FOOTPRINT_STEP`-block
clip into a slightly higher corner reads as the lesser problem. Refusing
past that tolerance rather than auto-levelling was the ticket's central
decision -- see the module docs' "No auto-level" section for the full
reasoning (levelling means writing blocks, which is W4/W5's job through a
real `WorldEdit`, and a read-only fit check secretly depending on the write
path is a cost paid by every future caller, E3's every-frame ghost preview
included).

"Ground" reuses `world::is_solid`'s not-air predicate (via
`topmost_non_air`) rather than a new fluid-aware classifier -- a lake
surface counts as ground for now, deliberately, per the module docs; that
distinction is roadmap I3's job once damage detection needs it, not this
ticket's to pre-empt.

Testing: nine tests in `city::grid::tests`, using a hand-built
`DecodedWorld`/`ChunkColumn` fixture (`world_with_ground`, in the same style
`camera.rs`'s `single_block_world`/`empty_world` and `world::mesh`'s
`section_with`/`column_with` already established -- no real save needed for
pure-function coverage) rather than a real save. Covers the height read
itself (present block, undecoded chunk); flat ground fitting at the sampled
height; a mid-footprint one-block step still fitting at its lowest point; a
six-block step refused as `TooSteep`; a footprint reaching past two decoded
ground tiles refused as `NotLoaded` without needing every tile resolved; a
90 degree rotation against asymmetric terrain (a ridge at `x=4` outside an
unrotated 3-wide footprint but inside its 90 degree-rotated 5-wide extent)
-- deliberately built so a rotation bug that kept sampling the unrotated
rectangle would still (wrongly) report `Fits`, rather than a symmetric
fixture that couldn't tell the two apart; a footprint straddling a chunk
boundary reading ground from both `ChunkColumn`s; and the zero-extent guard
not panicking. `ground_height_at`/`fit_footprint` carry `#[allow(dead_code)]`
with the module's own "no caller yet" note -- E3's ghost preview and E4's
commit are the eventual readers, the same landing state ticket 042's `City`
and ticket 045's `HoveredBlock` were first proven in.

No manual-verification entry needed: this ticket touches no rendering, no
UI, and no write path -- it's a pure function over already-decoded data,
fully exercised by its own unit tests.
