# 066 - corner / dead-end / T pieces render at the wrong rotation

## Report

User: "the rotation of corner cells and dead-ends is wrong. please fix (with
human interaction if needed)."

## Root cause

`road::canonical_pattern` declares the connection shape each `.nbt` piece is
*assumed* to have been authored at; `select_piece` then searches for the
quarter-turn count that rotates that canonical shape into a cell's actual
connections, and `road_build::road_write_edit` hands that `Rotation` to
`blueprint::rotate_blueprint`.

The rotation *machinery* is correct — `Direction::rotated`,
`RoadConnections::rotated` and `rotate::rotate90_pos` all agree on
"clockwise viewed from above, north -> east". What's wrong is the table: it
does not describe the pieces ticket 063 actually shipped.

Probing `assets/city/roads/dirt/*.nbt` (6x5x6, surface course at local
`y=1`, `minecraft:dirt_path` = the road surface) gives each piece's real
open edges:

| piece        | authored connections | `canonical_pattern` claimed | error  |
|--------------|----------------------|-----------------------------|--------|
| `isolated`   | south (a stub)       | none                        | n/a\*  |
| `dead_end`   | **south**            | north                       | 180°   |
| `straight`   | north + south        | north + south               | ok     |
| `corner`     | **south + west**     | north + east                | 180°   |
| `t`          | **north+south+east** | north + east + west         | 90°    |
| `cross`      | all four             | all four                    | ok     |

\* `Isolated` has no orientation, so `select_piece` always returns `Deg0`
for it — see "Not fixed by this" below.

Straight and cross were right by accident: both are symmetric under the
rotation they were off by (a straight rotated 180° is the same straight),
so the two kinds a player builds most of hid the bug.

## The convention the shipped assets do follow

Every one of the six pieces connects **south**. Stated per kind, that is:

- `dead_end` — south only
- `straight` — north + south
- `corner` — south + west
- `t` — north + south + east
- `cross` — all four

Coherent, and worth writing down rather than re-deriving: it becomes the
authoring contract for every future style, since `select_piece` is
style-blind and every style's pieces get rotated through the same table.

## Scope

- `road::canonical_pattern`: `DeadEnd`, `Corner` and `T` change to the
  shapes above. `Isolated`, `Straight` and `Cross` are unchanged.
- `road`'s module docs gain the convention.
- `assets/city/roads/dirt/README.md` states it per filename, so the next
  style is authored to it rather than to a guess.
- `road`'s own tests: the round-trip property test
  (`select_piece_round_trips_every_connection_pattern`) is unchanged in
  shape — it derives its expectation from `canonical_pattern` itself — but
  the per-kind example assertions that hard-code a rotation need updating,
  plus a new test pinning the canonical table against the documented
  convention so a future edit to one without the other fails.

## Not fixed by this

`isolated.nbt` is byte-identical to `dead_end.nbt` — a south-pointing stub,
not a self-contained island of road. An isolated cell will keep rendering
as a dead end that opens onto grass to the south. That's asset content, not
code: `select_piece` has no orientation to give `Isolated` and nothing here
can invent one. Noted for the user to re-export if they want a distinct
isolated piece.

## Done when

- The canonical table matches the probed geometry of the shipped `dirt`
  pieces.
- `cargo check --all-targets` and `cargo test --lib` clean.
- Whether a corner *visually* bends the right way needs a human at the
  window — noted in `todo.md` per this repo's convention.

## Resolution

Landed as scoped.

- `road::canonical_pattern` now reads `DeadEnd` = south, `Corner` =
  south + west, `T` = north + south + east; `Isolated`, `Straight` and
  `Cross` unchanged. `Direction::rotated`, `RoadConnections::rotated`,
  `matching_rotation`, `select_piece` and `blueprint::rotate` are all
  untouched — the machinery was never the problem.
- `road`'s module docs gained "The authoring convention every style's `.nbt`
  pieces follow", with the table; `assets/city/roads/dirt/README.md` was
  rewritten around it.
- Three new tests:
  - `road::the_canonical_patterns_match_the_documented_authoring_convention`
    pins the table itself (the round-trip test can't: it derives its
    expectation from the same table).
  - `road::every_oriented_canonical_pattern_opens_to_the_south` — the
    one-line form of the rule.
  - `road_catalogue::the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations`
    is the one with real teeth: it loads
    `assets/city/roads/dirt/*.nbt` and compares each piece's actual open
    edges (derived from the block at the piece's own centre, so it isn't
    hard-coded to `dirt_path` and works for a future stone-paved style)
    against `canonical_pattern`. Verified to fail when the table is bent.
    `Isolated` is skipped — see "Not fixed by this".

`cargo test --lib` — 591 passed, 0 failed. Visual confirmation is in
`todo.md`.
