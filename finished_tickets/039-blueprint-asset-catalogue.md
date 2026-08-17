# 039 - blueprint asset catalogue

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 316
tests). See the Resolution.

## Part of
Roadmap B4 (`tickets/CITYBUILDER_ROADMAP.md`) — the fourth and last ticket
in group B (blueprints as building models), and the thing that finally
calls 036/037/038's no-caller-yet primitives.

## Depends on
- 036's reader (`blueprint::read_structure_file`)
- 022's `Blueprint`/`BlockState` (`blueprint::extract`)
- 027's `city` module (`src/city/mod.rs`), the natural place to wire this
  into a running app — this catalogue is citybuilder-specific, unlike
  `blueprint` itself, which `block_viewer` also depends on

## Problem

`assets/city/blueprints/*.nbt` doesn't exist yet, and nothing reads a
directory of them into memory. B1-B3 are all "no caller yet" — a reader, a
mesher, a rotator, each proven only by unit tests against hand-built or
extracted `Blueprint`s. This is the ticket that turns "a `.nbt` file" into
"a building the game knows about": scan a directory, read each file through
036's reader, reject anything that couldn't plausibly be a building, and
hand back something G1's build menu (later) can iterate.

## Goal

`blueprint::catalogue::load_catalogue_dir(&Path) -> (BuildingCatalogue,
Vec<(PathBuf, CatalogueError)>)`: reads every `*.nbt` in the directory
(non-recursive — the roadmap's glob is flat), validates each one, and
separates what loaded from what didn't rather than failing the whole
catalogue over one bad file. `city::run()` calls it against
`assets/city/blueprints`, logs the outcome the way `world_app`'s startup
already logs the atlas/save, and inserts the result as a resource.

## Scope

- `CatalogueEntry`: `id` (the filename stem, e.g. `house01`), `path`, the
  loaded `Blueprint`, and `footprint: IVec2` — `(size.x, size.z)`, per C1's
  sketch (`footprint: FromBlueprint`). Y doesn't factor into a footprint;
  the grid E2 places buildings on is horizontal.
- Validation, beyond what 036's reader already guarantees (dense blocks,
  in-range palette indices):
  - **Size limits.** No axis may be zero (a degenerate, unplaceable
    building), and none may exceed `STRUCTURE_BLOCK_MAX_SIZE` (reused from
    `blueprint::structure`, not a new number invented here) — every file in
    this directory is a structure file, so bounding catalogue entries by
    what a structure block could actually have produced is the right
    ceiling, not an arbitrary one.
  - **Palette sanity.** A palette that's nothing but `minecraft:air` (index
    0, always present per `Accumulator::new`) means the file has no actual
    blocks in it — not a building, whatever else is true about it.
- `CatalogueError`: `Read(StructureReadError)`, `InvalidSize(IVec3)`,
  `Empty`, `DuplicateId { id: String, other: PathBuf }` (two files whose
  stem collides — same detect-don't-silently-pick-one instinct as 038's
  `UnrotatableProperty`).
- `BuildingCatalogue`: holds the entries, keyed for `get(id)` lookup (G1's
  build menu and, later, placement will want "the building named X", not
  just "the nth one").
- `load_catalogue_dir`: never panics and never fails outright — a missing
  directory is an empty catalogue (logged, not fatal; the same "don't take
  the process down" call ticket 008 made for a missing saves directory),
  and a per-file error is skipped and reported alongside the entries that
  did load rather than losing the whole directory to one bad export.
- `city::run()` wiring: load `assets/city/blueprints` before `.run()` (the
  same synchronous-before-`App::new()` shape `world_app()` uses for
  `LoadedSave`), print a one-line-per-entry summary plus any skipped files,
  insert `BuildingCatalogue` as a resource. No consumer reads the resource
  yet — G1 is what displays it — so this is proving the load path end to
  end, the same way 024 proved the write path before anything used it.
- `assets/city/blueprints/house01.nbt`: the first real fixture, moved in
  from the untracked `buildings/house01.nbt` already sitting at the repo
  root (a real structure-block export, not a synthetic test fixture).
- Unit tests: a directory with a valid file, a missing directory, a
  too-large blueprint, a zero-size axis, an all-air blueprint, a duplicate
  id, and a real read through `house01.nbt`.

## Watch out

- The directory scan has to be non-recursive and filter by extension
  case-insensitively (`.nbt`/`.NBT`) rather than assume a platform's
  `read_dir` order is stable — sort by filename before validating so two
  runs against the same directory produce the same catalogue order and the
  same duplicate-id winner.
- `STRUCTURE_BLOCK_MAX_SIZE` is `blueprint::structure`'s constant already —
  reuse it, don't redefine a second "how big is too big" number that could
  drift from it.
- A directory that exists but contains zero `.nbt` files is not an error —
  distinguish "the directory itself is missing" (logged, still not fatal)
  from "empty of buildings" (silently fine); an early build has no assets
  checked in yet at all outside `house01.nbt`.

## Out of scope

- C1 (building definitions — tier, cost, production) doesn't exist yet;
  `CatalogueEntry` carries only what B4 promises (id, blueprint, footprint),
  not a `Building` schema.
- G1 (the build menu) is the eventual reader of `BuildingCatalogue`; this
  ticket only proves the resource is populated and logged at startup.
- Hot reload (C4) — the catalogue loads once, at startup.

## Done when

- `load_catalogue_dir` exists, validates as scoped above, and never panics
  on a missing directory or a malformed file.
- `city::run()` loads `assets/city/blueprints`, logs the result, and
  inserts `BuildingCatalogue` as a resource.
- `house01.nbt` lives at `assets/city/blueprints/house01.nbt` and loads
  through the real path (not just a synthetic fixture).
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed. `blueprint::catalogue::load_catalogue_dir` does the
directory scan (non-recursive, `*.nbt` matched case-insensitively via
`is_nbt_file`, candidates sorted by path before loading for a deterministic
order and a deterministic duplicate-id winner); the load-and-validate loop
itself lives in a separate `build_catalogue(Vec<PathBuf>)` so it can be
exercised directly, not only through a real directory scan (see the test
note below). `CatalogueEntry::footprint` is `(size.x, size.z)` via
`footprint_of`; validation is two checks in `validate()` — `InvalidSize`
when any axis is `<= 0` or exceeds `STRUCTURE_BLOCK_MAX_SIZE` (reused from
`blueprint::structure`, not reinvented), `Empty` when the palette has
nothing past index 0 (`minecraft:air`).

`city::run()` loads `assets/city/blueprints` synchronously before
`world_app()` is built, the same shape `world_app()` itself uses for
`LoadedSave`, logs one line per loaded entry (id, size, palette count) and
one per skipped file (path + reason), and inserts `BuildingCatalogue` as a
resource. No consumer reads it back out yet — same "proven, not yet used"
state 036/037/038 were in before this ticket, just one level up: this is
now the thing without a caller.

One design call worth recording for later callers: `DuplicateId` detection
runs on a `HashMap<String, CatalogueEntry>` keyed by id, checked before each
insert. It's real dead code today — `load_catalogue_dir`'s own scan is
non-recursive and matches extensions case-insensitively, and on a
case-insensitive filesystem (Windows' default) two files whose names differ
only in case can't coexist in one directory, so the loader can never
actually hand `build_catalogue` two paths with the same stem. Kept anyway
(rather than dropped as unreachable) because a future recursive scan or a
build on a case-sensitive filesystem would make it reachable, and the
alternative — one file silently overwriting the other — is exactly the
failure mode 038's `RotationError::UnrotatableProperty` and structure.rs's
duplicate-position check both refuse to allow.
`a_duplicate_id_is_skipped_rather_than_silently_overwriting` exercises
`build_catalogue` directly against two real files in different
subdirectories (so no filesystem collision) for exactly this reason — it
was originally written against two case-varying filenames in the same
directory and failed on Windows because `write_structure_at`'s second write
silently overwrote the first at the OS level before the catalogue ever saw
two candidates.

`assets/city/blueprints/house01.nbt` is the fixture — moved in from the
untracked `buildings/house01.nbt`, a real structure-block export (verified
gzip + NBT compound header, not just a `.nbt`-named file). 11 new tests in
`blueprint::catalogue::tests` (316 total, up from 305): missing directory,
empty directory, a valid load with its footprint, case-insensitive
extension matching, zero-size and oversized axes, an all-air palette,
garbage bytes (a read error, not a panic), the duplicate-id case above,
multiple independent files in one directory, and a read through the real
`house01.nbt` fixture.

No manual/in-game check needed — nothing is spawned or rendered, only
logged to the console at startup. `todo.md` has an entry for a human to
`cargo run --bin citybuilder` and confirm the console prints the catalogue
summary and nothing panics, the same "watch the console" shape several
other entries there already take.
