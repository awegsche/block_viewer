# 041 - tiers and tech tree

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 333 tests,
up from 316). See the Resolution.

## Part of
Roadmap C2 (`tickets/CITYBUILDER_ROADMAP.md`) — the second ticket in group C,
right after 040's schema and loader.

## Depends on
- 040's `city::definition::{Building, BuildingDefinitions, load_definitions_dir}`.
  `Building::requires` already exists and is already parsed — 040 explicitly
  left it "carried but not validated," naming this ticket as the one that
  validates it.

## Problem

`Building::requires: Vec<String>` is parsed but never checked. Two ways a
tech tree can be broken and neither is caught today:

- **Dangling reference**: `requires` names a building id that doesn't exist
  (typo, or the referenced file failed to load for its own unrelated reason).
- **Cycle**: A requires B requires A (or a longer loop). A building on a
  cycle can never unlock.

Per the roadmap: "a tech tree with a cycle is unwinnable and the failure
mode is 'button greyed out forever' if it isn't caught." The same is true of
a dangling reference — the building it points at will never exist, so the
referencing building is just as permanently locked, for a less obvious
reason.

## Goal

At load time, after every individually-valid `.ron` file has been parsed
(040's per-file checks: schema, blueprint reference, integrity/cost/
production/footprint ranges), run a second pass over the *whole* loaded set
that checks `requires` edges against each other: dangling references and
cycles both get caught and reported, not silently left as an unreachable
build-menu entry.

## Scope

- A `requires` id that isn't in the final loaded set is
  `DefinitionError::DanglingRequirement`, reported against the file that
  named it.
- A building on a `requires` cycle is `DefinitionError::CyclicRequirement`,
  carrying every id on the cycle so the message names the whole loop, not
  just one building.
- **Cascading removal.** Removing a cyclic or dangling entry can turn some
  other, otherwise-fine entry's `requires` into a fresh dangling reference
  (it required something that just got removed for an unrelated reason).
  Re-check after every removal until nothing more is removed, rather than
  a single pass that only catches direct problems. The end state is the
  same regardless of removal order: the maximal subset of loaded buildings
  whose `requires` graph, restricted to that subset, is fully resolvable and
  acyclic.
- Entries removed this way are moved from `BuildingDefinitions` to the
  `skipped` list `load_definitions_dir` already returns — same "failure is
  per-file, not per-directory" contract 039/040 established, just extended
  to a cross-file check.
- `Building::requires`'s `#[allow(dead_code)]` comes off — this is its first
  real (non-test) reader.

## Out of scope

- Anything that *reads* `requires` for gameplay (an "is this unlocked"
  query, greying out a build-menu entry). No caller exists yet — that's
  G1's build menu, same "prove the data, no consumer yet" state 039/040
  landed in.
- C3 (production simulation), C4 (hot reload).

## Done when

- Dangling and cyclic `requires` are caught at load and reported per-file,
  not left as an entry nobody can ever place.
- Cascading removal is covered by a test (a chain where the *indirect*
  dependency is the broken one).
- `house01.ron`'s real fixture still loads (`requires: []` trivially
  passes).
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed, entirely inside `city::definition` — no new module,
since this is a second pass over the same `Building`/`BuildingDefinitions`
types 040 already defined.

`build_definitions` now runs `resolve_requirements` over the `HashMap` its
per-file loop produces, before wrapping it in `BuildingDefinitions`.
`resolve_requirements` loops two passes to a fixpoint: a dangling pass
(any `requires` id not among the current keys is removed as
`DanglingRequirement`, all offenders removed together per round, not just
the first) and a cycle pass (`find_cycle`, a plain DFS with a recursion
stack over an id -> `requires` adjacency map, returning one cycle as the ids
on the loop in order; every id on it is removed as `CyclicRequirement`,
carrying the whole loop rather than just the one id being reported). Either
pass removing anything re-triggers the other, since removing an entry can
turn some *other* entry's `requires` into a fresh dangling reference — the
case the ticket names as the reason a single pass isn't enough. The loop
terminates because each round removes at least one entry or nothing at all;
worst case is a handful of rounds for a fully-cascading chain, cheap at the
building counts this schema is for.

`find_cycle` iterates candidate ids in sorted order (not `HashMap` iteration
order) so which cycle it reports first, when several are disjoint, is
deterministic and doesn't depend on hash state — matters for test stability,
not correctness (the *final* surviving set is order-independent either way,
per the module docs' argument).

`Building::requires`'s `#[allow(dead_code)]` came off — `resolve_requirements`
is its first real reader. `DefinitionError` gained `DanglingRequirement(String)`
(the missing id) and `CyclicRequirement(Vec<String>)` (every id on the loop),
matching the existing tuple-variant shape of `UnknownBlueprint`.

4 new tests, plus one rewritten (`city::definition` module total 17, up from
13; project total 333, up from 329): a dangling reference, a two-building
cycle (both ends reported, not just whichever the scan reaches first), a
self-referential one-node cycle, and the cascading case the ticket called out
by name — `c` requires `b` requires `a`, and `a` is self-cyclic; a
non-looping single pass would remove only `a` and leave `b`/`c` looking
valid, but neither can ever unlock once `a` is gone, and the test asserts
all three are reported (one `CyclicRequirement` for `a`, two
`DanglingRequirement`s cascading from it). The existing
`requires_is_carried_but_not_validated` no longer described reality now that
C2 exists — replaced with `a_valid_requires_edge_is_carried_and_survives`,
which needed a second fixture building (`carpenter.ron`, reusing the existing
`house01.nbt` blueprint rather than adding a new one — `requires` names other
*building* ids, not blueprints, so nothing new was needed on that side).

No manual/in-game check — same as 040, this only changes what gets logged at
startup (a bad tech tree now shows up as `DanglingRequirement`/
`CyclicRequirement` lines in the existing skipped-entries log), nothing new
is spawned or rendered.
