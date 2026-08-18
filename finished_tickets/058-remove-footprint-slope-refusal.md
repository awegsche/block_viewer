# 058 - Footprint fit stops refusing steep ground

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 523
tests, all passing). User-directed change to ticket 046/052's
`fit_footprint` (`city::grid`, roadmap E2). Roadmap group E
(`tickets/CITYBUILDER_ROADMAP.md`) is the parent.

## The change

`fit_footprint` currently refuses a placement outright
(`FitError::TooSteep`) when the sampled ground under a footprint varies by
more than `MAX_FOOTPRINT_STEP` (1 block). A real Minecraft world is
inherently uneven, and a hard 1-block tolerance restricts where a player can
build far more than the game should — refusing a placement is a much bigger
cost than the mild clipping/floating a steep site produces.

**Remove the refusal.** `fit_footprint` still samples ground height under
the footprint and still returns a `base_y` — that half doesn't change, and
neither does how it's used: `base_y` is only the *initial* suggested Y
(`city::placement::resolve_placement`'s auto-fit), freely overridable by
`PlacementSelection::y_offset` (`Page Up`/`Page Down`/`Home`, ticket 048).
What goes away is the `max_y - min_y > MAX_FOOTPRINT_STEP` check and the
`TooSteep` variant — a footprint's only remaining refusal reason is
`NotLoaded` (ground outside the streamed/decoded radius, or nothing but
clutter all the way down — ticket 052).

`base_y` keeps using the *lowest* sampled point, same as before (a high
corner clips a little into the building's own foundation; that reads better
than a gap floating over a low corner, and nothing here fills terrain in).

## Explicitly not in scope here

**No cost mechanic.** The user's own framing: don't restrict building at
all, but a later iteration could charge for an extreme placement instead of
blocking it — e.g. build time (or a resource cost) proportional to the
number of solid blocks the placement needs to clear, so burying half a
building in a hillside is expensive rather than forbidden or free. That's a
genuinely new mechanic (counting blocks cleared, wiring it to *some* notion
of cost — none exists yet per the roadmap's C3/H2 boundary) and belongs as
its own roadmap item, not folded into this ticket. Noted in
`CITYBUILDER_ROADMAP.md` for later; not implemented now.

**No change to terraforming, occupancy, or the write path.** H1's dig/level
tools, `state::City::is_tile_free`, and `commit::blueprint_edit`'s "air is
written, not skipped" policy are all unaffected — a steep placement still
gets `blueprint_edit`'s full footprint written, air included, the same as a
flat one.

## Scope

- `city::grid`: drop `MAX_FOOTPRINT_STEP` and `FitError::TooSteep`; simplify
  `fit_footprint` to only track the minimum sampled height, not a min/max
  pair (nothing reads the max any more).
- Update the module's own doc comments (the "No auto-level" section) and the
  handful of other doc comments elsewhere in `city` that referenced the old
  tolerance/refusal (`city::commit`, `city::mod`, `city::terraform`).
- Tests: a footprint over a real multi-block terrain step now fits (not
  refused) at its lowest point; the 90°-rotation test that used to prove
  rotation samples the *rotated* rectangle via a `TooSteep` refusal needs a
  different proof now that steepness can't refuse anything — reworked to use
  an unloaded tile past the rotated extent instead.
- `CITYBUILDER_ROADMAP.md`: record the revision under E2, and add the
  build-cost-by-blocks-cleared idea as a noted-for-later item (not a ticket
  yet — no consumer, no cost system to hang it off).

## Done when

- `cargo build`/`cargo test` clean.
- No more references to `MAX_FOOTPRINT_STEP`/`TooSteep` anywhere in `src`.
- A footprint over terrain that previously refused as `TooSteep` now
  reports `Fits` at its lowest sampled point.

## Resolution

Landed as scoped. `city::grid::FitError` dropped `TooSteep` outright —
`NotLoaded` is its only remaining variant — and `fit_footprint` now tracks
only a running minimum height instead of a min/max pair, since nothing
reads the max any more. `MAX_FOOTPRINT_STEP` is gone. `base_y` is unchanged
in how it's computed (the footprint's lowest sampled point) and how it's
used (`city::placement::resolve_placement`'s auto-fit, overridable by
`PlacementSelection::y_offset`) — only the refusal on top of it is gone.

Three tests in `city::grid` needed rework because they proved their point
*through* the old refusal:

- `ground_past_tolerance_is_refused_as_too_steep` →
  `a_steep_step_still_fits_at_its_lowest_point_ticket_058`: same 6-block
  step fixture, now asserting `Fits { base_y: 65 }` instead of `Refused`.
- `a_90_degree_rotation_samples_the_rotated_rectangle` kept its name and
  intent (prove a rotation bug that samples the *unrotated* rectangle would
  still show `Fits`) but swapped its proof mechanism: the old fixture used a
  height ridge that only `TooSteep` could react to; the new one uses ground
  that simply doesn't exist past the unrotated extent, so the same rotation
  bug now shows up as a wrongly-`Fits` result instead of a wrongly-not-
  `TooSteep` one.
- `a_real_slope_still_refuses_even_with_clutter_ignored` → renamed to
  `clutter_on_the_lowest_tile_does_not_inflate_base_y`, and rebuilt as an
  actual test of `is_ground`'s clutter exclusion affecting `base_y` (which
  the old fixture technically didn't — a `min()` reduction is blind to one
  tile reading too *high*, only to one reading too *low* incorrectly). The
  new fixture puts a solid fence post directly on top of the footprint's one
  genuinely low tile and checks `base_y` reads the ground under the fence
  (65), not the fence itself (66) — the case where clutter exclusion
  actually changes the answer.

`a_tree_standing_on_otherwise_flat_ground_does_not_read_as_a_cliff` and
`a_fence_post_on_flat_ground_is_skipped_too` needed no logic change (their
assertions were already about `base_y`, not refusal) but got a doc-comment
pass noting their claim is now a much weaker one than it used to sound like.

Doc-comment sweep beyond the tests: `city::grid`'s own module docs (the "No
auto-level" section, retitled), `city::commit`'s `blueprint_edit` note
(the sliver of exposed terrain it clears can now be a good deal more than
one block on a rough site), `city::mod`'s and `city::terraform`'s own
"Terraforming" write-ups (H1 is no longer the *only* way to build on
uneven ground — it's a player's choice now, not a requirement).
`CITYBUILDER_ROADMAP.md` got a "Revised by ticket 058" addendum under E2,
plus a "noted for later, not scoped" paragraph (cross-linked from H2) for
the build-time/resource-cost-by-blocks-cleared idea the user floated as a
future direction — deliberately not scoped into a ticket, since it needs a
cost system that doesn't exist yet.

No manual verification recorded — same as 052, this is a pure logic change
over already-decoded in-memory data (no I/O, no new UI), so `../todo.md`
doesn't gain an entry for it. The existing 047/048 ghost-preview/commit
manual-verification items already exercise the *visual* result (a ghost
that used to turn red and refuse on a hillside now shows green and fits at
the low point) if that's worth re-checking by eye.
