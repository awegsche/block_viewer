# 038 - blueprint rotation

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 305
tests). See the Resolution.

## Part of
Roadmap B3 (`tickets/CITYBUILDER_ROADMAP.md`) — the third ticket in group B
(blueprints as building models).

## Depends on
- 022's `Blueprint`/`BlockState` (`blueprint::extract`)
- 037's mesher (`blueprint::mesh`) — not called by this ticket, but the
  reason the roadmap draws the mesh/blocks line the way it does (see below)

## Problem

A building placed at anything but its as-authored orientation needs to
turn. The roadmap draws the line precisely: **the mesh part is a
transform** (spin the spawned entity's `Transform` in 90° steps — no new
geometry, nothing this ticket needs to touch), but **the blocks part is
not** — a stair's `facing`, a log's `axis`, a sign's 16-way `rotation`, and
a fence/wall/pane's `north`/`south`/`east`/`west` connections are all
strings in the palette that encode a *world* direction. Turn the building
without rewriting them and the geometry rotates while the block data
doesn't: a staircase that visually turns 90° but is still built from stairs
whose `facing` property says they climb the old way, wrong the moment it's
written into the save.

## Goal

`blueprint::rotate::rotate_blueprint(&Blueprint, Rotation) ->
Result<Blueprint, RotationError>`: 90/180/270 about Y, remapping the block
grid *and* rewriting every rotation-sensitive property in the palette.
Anything the rotation table doesn't recognise is a hard error naming the
block, property and value — not a silent copy of the original (wrong)
value.

## Scope

- `Rotation`: `Deg0`/`Deg90`/`Deg180`/`Deg270`. `Deg0` short-circuits to a
  clone with no palette validation — the default/no-op case shouldn't be
  able to fail on a property this table doesn't know.
- Grid remap: size and block-index transform for a quarter turn, composed
  `turns` times rather than three separately hand-derived formulas for
  180/270 — one transform, checked by construction, not three chances to
  get the arithmetic wrong.
- Palette remap, once per distinct `BlockState` (not once per block):
  - `facing`: the four horizontal values cycle; `up`/`down` pass through.
  - `axis`: `x`/`z` swap on an odd number of turns, `y` and even turns pass
    through.
  - `rotation` (signs/banners, 0–15): `+= 4 * turns (mod 16)`.
  - `north`/`south`/`east`/`west` (fences, walls, panes, bars, redstone
    wire): handled as a group, not independently — a quarter turn moves
    *which key* holds a given value, the same cycle as `facing`.
  - `shape`: rail curves/straights get the same cycle as `facing`+diagonal
    pairs; stair shapes (`straight`/`inner_*`/`outer_*`) are relative to
    the block's own `facing` and pass through unchanged — the two value
    sets don't overlap, so the property is disambiguated by its value, not
    by which block it's on.
  - A whitelist of properties that don't encode a horizontal direction
    (`waterlogged`, `half`, `powered`, `type`, `hinge`, ...) pass through
    unchanged. `hinge` and chest's `type` (`left`/`right`/`single`) are
    listed explicitly, not folded into "unknown", because they're defined
    *relative to `facing`* — the roadmap names `hinge` as needing rewriting,
    but rewriting `facing` and leaving a facing-relative property alone is
    what keeps the two consistent, not what breaks them.
  - Anything else: `RotationError::UnrotatableProperty`, naming the block
    name, property key and value.
- Unit tests: geometry (a non-cubic blueprint's size and a marked corner
  block land where a 90/180/270 turn puts them), each property rule above,
  the shape disambiguation, the whitelist, and the unknown-property error.

## Watch out

- The grid transform must handle non-cubic blueprints — `size.x` and
  `size.z` swap on a 90/270 turn, so `blocks.len()` stays the same but its
  layout doesn't.
- `BlockState::properties` is a sorted `Vec<(String, String)>` (`extract.rs`)
  — re-sort after rewriting rather than assuming the rewrite preserves key
  order, or the palette dedupe two rotations produce for "the same" state
  could disagree on ordering. (Sorting by key is stable across rewrites in
  practice here since no rewrite changes which keys are present, but sort
  defensively rather than rely on that.)
- Don't validate palette entries at `Deg0` — a blueprint with a property
  this table doesn't recognise should still load and preview unrotated.

## Out of scope

B4 (the asset catalogue) and E3 (the placement ghost) — both are callers
this ticket doesn't have yet, same as 036/037 before their callers landed.
Mirroring rotation (for a building that should read as a mirror image, not
just a turn) isn't asked for by the roadmap and isn't attempted here.

## Done when

- `rotate_blueprint` exists, covers the property table above, and detects
  (doesn't silently keep) anything outside it.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed: `blueprint::rotate::{rotate_blueprint, Rotation,
RotationError}`, re-exported from `blueprint` the same way 036/037's
no-caller-yet primitives are — B4 and E3 are what will call this.

The grid transform is one function (`rotate90_size`/`rotate90_pos`),
composed `turns` times, rather than three hand-derived formulas for
90/180/270 — `four_quarter_turns_are_the_identity` is the test that would
have caught a sign error in a hand-derived 270° formula and instead just
confirms the composition round-trips.

The one design call worth recording: `shape` is disambiguated by *value*,
not by block name. A stair's `shape` (`straight`/`inner_left`/...) and a
rail's `shape` (`north_south`/`north_east`/`ascending_north`/...) are the
same property name with disjoint value sets, and the stair values are
relative to the block's own `facing` (so they pass through unchanged) while
the rail values encode an absolute direction (so they rotate). Matching on
the value rather than threading the block name into a lookup table avoids
building a second "which block is a stair vs. a rail" table that would only
ever be consulted for this one property.

`hinge` and a chest's `type` (`left`/`right`/`single`) are called out
explicitly in the whitelist rather than left to fall into "unknown" — both
are relative to the block's own `facing`, the same reasoning as `shape`'s
stair values, so a whole-structure rotation (which rotates `facing`) leaves
them correct as-is. The roadmap's text reads as if `hinge` needs rewriting;
`hinge_passes_through_unchanged` is the test recording why it doesn't, once
`facing` is rotated consistently alongside it.

`north`/`south`/`east`/`west` (fences, walls, panes, bars, redstone wire)
are handled as a group ahead of the per-key rewrite loop, because a
rotation moves *which key* holds a value, not just the value at a fixed
key — the same clockwise cycle `facing` uses, applied to key selection
instead of a string.

34 new tests in `blueprint::rotate::tests` (305 total, up from 271):
geometry (non-cubic size swap, a marked corner landing where a 90° turn
puts it, four quarter-turns round-tripping to the identity), every property
rule (`facing`, `axis`, 16-way `rotation`, the connection-key group, the
stair/rail `shape` split, the whitelist), and the unknown-property error
both at the single-property level and failing a whole blueprint rotation —
paired with the `Deg0`-never-validates case so a blueprint with an
unrecognised property still loads unrotated.

No manual/in-game check needed — this is a pure `Blueprint -> Blueprint`
function with no caller and nothing spawned or written; B4/E-group callers
are what will make this visible in the app.
