# 040 - building definition schema and loader

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 329
tests). See the Resolution.

## Part of
Roadmap C1 (`tickets/CITYBUILDER_ROADMAP.md`) — the first ticket in group C
(definitions and scripting), and the first thing to give a building numbers
beyond its shape.

## Depends on
- 039's `blueprint::{BuildingCatalogue, CatalogueEntry}` — a definition names
  a blueprint file, and cross-checking that name against the catalogue is
  what turns "a typo in a RON file" into a load-time error instead of a
  build-menu entry nobody can place.
- 027's `city` module — same wiring point 039 used.

## Problem

The catalogue (039) knows a building's *shape*: id, blueprint, footprint.
Nothing yet knows its *game data* — tier, what unlocks it, what it produces,
what it costs, how much damage it tolerates before it stops working. The
roadmap's recommendation (see C1's header) is data files now, not a
scripting language: RON, one file per building, parsed into a typed schema.

## Goal

`city::definition::load_definitions_dir(&Path, &BuildingCatalogue) ->
(BuildingDefinitions, Vec<(PathBuf, DefinitionError)>)`: reads every `*.ron`
in `assets/city/buildings` (non-recursive, same shape as 039's scan),
deserializes each into a `Building`, validates it, resolves its footprint
against the catalogue, and returns what loaded next to what didn't.
`city::run()` calls it after loading the catalogue and inserts the result as
a resource.

## Scope

- `Building`: `name`, `blueprint` (a filename, e.g. `"house01.nbt"`), `tier`,
  `requires: Vec<String>` (other building ids, parsed but **not**
  cycle/dangling-checked here — that's C2), `footprint: FootprintSpec`
  (`FromBlueprint` or `Explicit { x, z }`), `production: Option<Production>`,
  `cost: Vec<Cost>`, `integrity: Integrity`. Matches the roadmap's sketch RON
  shape, field for field.
- `Production { outputs: Vec<ProductionItem>, inputs: Vec<ProductionItem>,
  radius: Option<u32> }`, `ProductionItem { item: String, per_minute: f32 }`,
  `Cost { block: String, count: u32 }`, `Integrity { pristine_above: f32,
  ruined_below: f32 }` — parsed, not simulated (C3/I4's job).
- **No `id` field in the RON file.** Same call 039 made for blueprints: the
  filename stem is the id, so the file and its id can't drift apart. The
  roadmap's illustrative sketch has an inline `id:` field; this is a
  deliberate deviation, recorded here rather than discovered.
- Validation beyond what serde's deserialize already guarantees:
  - `blueprint`'s stem must name an entry in the passed `BuildingCatalogue`
    — a definition for a blueprint that doesn't exist (or isn't loaded) is
    an error, not a silently unplaceable building.
  - `integrity.pristine_above` and `.ruined_below` both in `0.0..=1.0`, and
    `pristine_above > ruined_below` — anything else makes I4's later linear
    ramp degenerate or inverted.
  - `cost` entries: `count > 0`. `production` entries: `per_minute >= 0.0`.
  - `footprint: Explicit { x, z }` needs both `> 0`; `FromBlueprint` always
    resolves (it reads the catalogue entry's own already-validated
    footprint).
- `DefinitionError`: `Read(io::Error)`, `Parse(String)` (RON's own message —
  wrapping `ron::error::SpannedError` directly isn't worth it for one call
  site), `UnknownBlueprint(String)`, `InvalidIntegrity { pristine_above,
  ruined_below }`, `InvalidCost`/`InvalidProduction`, `InvalidFootprint`,
  `NoFilenameStem`, `DuplicateId { id, other }` — same shape as 039's
  `CatalogueError`.
- `LoadedBuilding { id, path, building: Building, footprint: IVec2 }` and a
  `BuildingDefinitions` resource keyed by id, same `get`/`len`/`is_empty`/
  `iter` surface as `BuildingCatalogue`.
- `city::run()` wiring: load `assets/city/buildings` after the catalogue,
  log a summary the same shape as 039's, insert `BuildingDefinitions`.
- New deps: `serde` (derive) and `ron` — both already in `Cargo.lock`
  transitively (bevy pulls `serde`, something pulls `ron`), so this is
  naming existing versions as direct deps, not a new download.
- Fixture: `assets/city/buildings/house01.ron`, referencing the existing
  `house01.nbt`, non-functional (`production: None`, matching "iteration 1
  places non-functional buildings").

## Out of scope

- C2: tier/requires cycle detection and dangling-reference checks. `requires`
  is parsed and carried, not validated against other ids.
- C3: nothing simulates `production`/`cost` — they're inert data, shown
  later by G1.
- C4: no hot reload. Loads once at startup, same as 039.
- I4's integrity curve itself — `Integrity` is parsed and range-checked, not
  evaluated against any health number yet.

## Done when

- `load_definitions_dir` exists, validates as scoped above, and never panics
  on a missing directory or a malformed file.
- `city::run()` loads `assets/city/buildings`, logs the result, inserts
  `BuildingDefinitions`.
- `house01.ron` loads through the real path.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed, in a new `city::definition` module — game data belongs
with the game (`city`), not with the blueprint mechanics (`blueprint`) that
039's catalogue lives alongside.

`Building` is field-for-field the roadmap's C1 sketch minus the inline `id`
(derived from the filename stem instead, matching 039's blueprint id — see
the module docs for why). `FootprintSpec` is `FromBlueprint` (default) or
`Explicit { x, z }`, resolved against the matched `CatalogueEntry` in
`resolve_footprint`. `load_definitions_dir(&Path, &BuildingCatalogue) ->
(BuildingDefinitions, Vec<(PathBuf, DefinitionError)>)` mirrors 039's
`load_catalogue_dir` exactly: non-recursive `*.ron` scan matched
case-insensitively, paths sorted before loading for a deterministic
duplicate-id winner, a separate `build_definitions(Vec<PathBuf>, &BuildingCatalogue)`
so the load loop can be tested directly. Validation beyond serde's own
deserialize: `integrity`'s two thresholds must both be in `0.0..=1.0` with
`pristine_above > ruined_below`; `cost` counts must be `> 0`; `production`
rates must be `>= 0.0`; an `Explicit` footprint's axes must both be `> 0`;
and `blueprint`'s filename stem must resolve to a real `BuildingCatalogue`
entry (`DefinitionError::UnknownBlueprint` otherwise) — this last one is the
cross-module check that doesn't exist in 039, since 039 has no definitions
to check against yet.

`city::run()` loads the catalogue, then the definitions (passing `&catalogue`
by reference — the catalogue itself still moves into the `insert_resource`
call afterward), logs a summary in 039's shape (`loaded N building
definitions from assets/city/buildings`, one line per entry: id, name, tier,
resolved footprint), and inserts `BuildingDefinitions`. No consumer reads it
yet — G1's build menu is what will, the same "proven, not yet used" state
039 landed B4 in.

Added `serde` (with `derive`) and `ron` as direct dependencies, pinned to the
versions already resolved transitively in `Cargo.lock` (`serde` 1.0.217,
`ron` 0.8.1) — no new downloads, `Cargo.lock` is otherwise unchanged.

`assets/city/buildings/house01.ron` is the fixture, referencing the existing
`house01.nbt`, non-functional (`production` omitted) per iteration 1's "place
a handful of non-functional buildings."

13 new tests in `city::definition::tests` (329 total, up from 316): missing
directory, a valid load with resolved `FromBlueprint` footprint, an explicit
footprint override, an unknown-blueprint reference, both integrity failure
shapes (inverted, out-of-range), a zero cost count, a negative production
rate, a non-positive explicit footprint, garbage RON (a parse error, not a
panic), `requires` carried but unvalidated, the duplicate-id case, and a read
through the real `house01.ron`/`house01.nbt` pair.

One bug caught by running the *whole* suite rather than just this module's
tests: the shared-fixture helper `catalogue_with_house01()` built its
temporary directory from `name + process::id()` alone (the convention 039's
own tests use), but every test in this module calls it with the *same*
`name`, and `cargo test` runs them concurrently in one process — same pid,
same directory, five tests racing `fs::remove_dir_all`/`write` against each
other and failing nondeterministically only under the full-suite run, never
under `cargo test definition` alone (too few concurrent threads to collide).
Fixed by adding a per-call atomic counter into `temp_dir`'s generated name,
which is what actually makes each call's directory unique rather than merely
each *named* call. Worth remembering for the next module that copies this
fixture-directory convention: a shared helper needs the counter, a
per-test-unique literal `name` alone isn't enough once two tests can call
the same helper.

No manual/in-game check needed beyond what 039 already asked for — nothing
new is spawned or rendered, only logged at startup. `todo.md` has an entry
to `cargo run --bin citybuilder` and confirm the console now also prints the
definitions summary.
