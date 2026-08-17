# 043 - city save/load

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 351
tests, up from 343). See the Resolution.

## Part of
Roadmap D2 (`tickets/CITYBUILDER_ROADMAP.md`) — the second ticket in group D
(city state), continuing milestone M4 ("a city exists"). D1 (ticket 042) gave
the game an in-memory `City`; this is what makes it survive closing the
window.

## Depends on
- 042's `city::state::City` — `buildings()`, `roads()`, `place_building`, and
  the `BuildingId`/`PlacedBuilding`/`Occupant` types this reads and writes.
- 038's `blueprint::Rotation` — a placed building's stored orientation.
- 027's `city::run()` — the wiring point, same as every prior city ticket.

## Problem

`City` (042) lives in memory only. Closing the citybuilder and reopening it
starts from an empty city every time, which makes every future placement
ticket (E1-E4) untestable end to end and defeats the roadmap's stated point
of D2: "so a save and its city travel together."

## Goal

`City` written to and read from `<save>/citybuilder/city.ron`, versioned from
the first write, wired into `city::run()`'s startup (load) and shutdown
(save) so the round trip is exercised by the real app lifecycle — not just by
tests — even though nothing places a building yet.

## Scope

- `city::persistence::CURRENT_VERSION: u32` and a `CitySave` struct
  (`#[derive(Serialize, Deserialize)]`) that's the RON file's actual shape:
  `version`, `next_id`, a `Vec` of saved buildings (id, definition, origin,
  rotation, footprint), and a `Vec` of road tiles. Not `City` itself
  (`occupancy` is derived, not stored — see Watch out).
- Add `Serialize, Deserialize` to `blueprint::rotate::Rotation`'s existing
  derive list rather than inventing a mirror enum — it's already the exact
  shape a placement's orientation needs on disk.
- `BuildingId` gets `pub(crate) fn as_u64`/`fn from_u64` accessors (state.rs)
  — the only thing standing between it and being serializable directly is
  that its inner `u64` is private, and it should stay private to the type
  outside this one round-trip use.
- `City` gets two additions in `state.rs`:
  - `pub(crate) fn next_id_raw(&self) -> u64` — for the saver.
  - `pub(crate) fn insert_loaded(&mut self, id: BuildingId, building:
    PlacedBuilding) -> Result<(), PlacementError>` — the loader's counterpart
    to `place_building` that takes an *existing* id instead of minting one,
    and advances `next_id` past it. Goes through the same "compute every
    tile, check them all, only then mutate" shape `place_building` already
    uses — a corrupt or hand-edited save file with two overlapping buildings
    must fail closed, not silently occupy whichever tile it processes last.
- `city::persistence::{save_city, load_city}`:
  - `save_city(city: &City, save_root: &Path) -> Result<(), PersistenceError>`
    — creates `<save_root>/citybuilder/` if missing, serializes a `CitySave`
    built from `city.buildings()`/`city.roads()`/`city.next_id_raw()`, writes
    it with `ron::ser::to_string_pretty` (a person may want to read or hand-
    edit this file, same call 040 made for building definitions).
  - `load_city(save_root: &Path) -> Result<City, PersistenceError>` — a
    missing file is `Ok(City::default())` (a save with no citybuilder data
    yet, not an error — same contract 039/040's directory scan uses for a
    missing directory). Otherwise: parse the RON, check
    `version == CURRENT_VERSION` (else `PersistenceError::UnsupportedVersion`
    — no migration exists yet, so a mismatch is refused rather than guessed
    at), then rebuild a `City` by calling `insert_loaded` for every saved
    building (ascending id order, for a deterministic failure point if one
    conflicts) and `add_road` for every saved road tile, and finally raise
    `next_id` to `max(what insertion already produced, the file's own
    next_id)` — see Watch out for why the file's value can't be dropped.
- `PersistenceError`: `Io(std::io::Error)`, `Parse(String)` (ron's message,
  same convention as `DefinitionError::Parse`), `UnsupportedVersion(u32)`,
  `Corrupt(PlacementError)` (a saved building or road collided on load).
  `Display` + `std::error::Error`, matching `DefinitionError`'s shape.
- `city::run()` wiring:
  - Read `LoadedSave`'s `meta.path` off the `App` returned by `world_app()`
    (`app.world().resource::<crate::LoadedSave>()`) before inserting
    anything else. If it's non-empty (a real save loaded, not ticket 008's
    `empty_save`), call `load_city` and log what happened the same way
    `load_building_catalogue`/`load_building_definitions` do; on `Err`, log
    and fall back to `City::default()` rather than panicking. If it's empty
    (no real save found), skip persistence entirely and insert
    `City::default()` — there's no save root to read from or write to.
  - Insert the loaded (or fresh) `City`, plus a small `CitySavePath(PathBuf)`
    resource carrying the same root for the save side.
  - A system, added to `Last`, reading `EventReader<bevy::app::AppExit>`:
    on exit, if `CitySavePath` is non-empty, call `save_city` and log the
    outcome. Runs once, on the same shutdown path for every exit route
    (window close, Alt+F4, process signal Bevy already turns into
    `AppExit`).

## Watch out

- **`occupancy` is never serialized.** It's fully determined by `buildings`
  + `roads` (that's what 042's module docs mean by "derived"); storing it
  too would let a hand-edited file disagree with itself. `insert_loaded`/
  `add_road` is what rebuilds it on load, which also means load re-validates
  the invariants place_building already enforces — a load that would produce
  an inconsistent grid fails instead of silently trusting stale data.
- **`next_id` must be persisted, not recomputed from the surviving
  buildings.** Removing a building doesn't roll `next_id` back (042's
  `BuildingId`s are "never reused, even after `remove_building`"). If
  building 1 of {0, 1} is removed before saving, only building 0 survives to
  disk — recomputing `next_id` as `max(surviving ids) + 1` would come back
  as 1, and the next placement would reissue id 1, which 042 explicitly
  promises never happens. The file's own `next_id` field is what protects
  against this; `insert_loaded`'s per-building bump is only a floor under
  it, not a substitute.
- **A version bump has no migration path yet.** `UnsupportedVersion` is the
  whole story for now — refuse and log, don't guess. A migration function is
  a later ticket's problem, once there's a `version` value in the wild that
  needs one.
- **Don't touch persistence when `LoadedSave` is the ticket-008 placeholder.**
  `empty_save()`'s `meta.path` is `PathBuf::new()`; joining `citybuilder/
  city.ron` onto that resolves relative to the process's CWD, which is not
  "next to the world" and would silently create a stray folder there on
  every run with no save found.
- **`origin`/rotation round-trip through RON's own enum syntax** — `Rotation`
  needs no custom serializer, just the derive; don't hand-write a
  `(x: i32, y: i32, z: i32)` tuple encoding for `IVec3`/`IVec2` when serde's
  own tuple-struct handling already does the job through a plain `(i32, i32,
  i32)`/`(i32, i32)` field.

## Out of scope

- D3 (journal, undo, reconciliation) — this ticket persists current state,
  not history. No record of *why* the file changed between two saves.
- Autosave on an interval, or after every placement — E4 (commit) is where a
  save-on-every-change policy would be decided; today nothing calls
  `place_building` outside tests, so the only meaningful save point is
  shutdown.
- A schema migration path for a future `version` bump.
- Anything in the build menu or a city panel reading `CitySavePath`/showing
  "last saved" — G2's job.

## Done when

- `city::persistence::{save_city, load_city}` exist with the API scoped
  above, `Rotation` derives `Serialize`/`Deserialize`, and `city::run()`
  loads on startup and saves on `AppExit` when a real save is loaded.
- Round-trip tests: an empty city saves and loads back empty; a city with
  several buildings (including one with a non-`Deg0` rotation) and roads
  round-trips exactly; a removed-building gap in the id sequence still
  round-trips `next_id` correctly (the scenario in Watch out); a missing
  file loads as an empty `City` rather than erroring; a version mismatch is
  `PersistenceError::UnsupportedVersion`; a hand-built file with two
  overlapping buildings is `PersistenceError::Corrupt` rather than a bad
  `City`.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed, plus one addition the scope draft hadn't named: a
`raise_next_id` method on `City` (state.rs), split out from `insert_loaded`
because the "raise `next_id` to at least the file's own recorded value" step
happens once at the end of a whole load, not once per building inserted.

`blueprint::rotate::Rotation` now derives `Serialize`/`Deserialize` directly
— no mirror enum in the persistence module, so a placement's orientation on
disk is exactly the type `blueprint::rotate_blueprint` already uses, with
nothing to keep in sync between the two. `BuildingId` gained `as_u64`/
`from_u64` (state.rs), deliberately not a `From<u64>` impl — `from_u64` is
awkward to reach for on purpose, since it's only correct in
`persistence::load_city`, the one place handing back an id
`City::place_building` never minted itself.

`city::persistence::{CitySave, SavedBuilding}` are the RON file's actual
shape — `version`, `next_id`, and flat `Vec`s of buildings/roads — not a
serialized `City`. `occupancy` has no field anywhere on disk: `insert_loaded`
(state.rs, `City`'s new method) and the existing `add_road` are what rebuild
it on load, going through the same "compute every tile, check them all, only
then mutate" shape `place_building` already uses. That turned out to matter
for more than tidiness — two of the eight new tests
(`two_overlapping_saved_buildings_is_corrupt_not_a_bad_city`,
`a_saved_road_colliding_with_a_building_is_corrupt`) hand-write a RON file
with a real conflict in it, and both come back `PersistenceError::Corrupt`
rather than a `City` whose occupancy grid silently disagrees with its own
buildings list.

`next_id` is the one field that has to be persisted rather than derived:
`insert_loaded`'s own bump (raise `next_id` past whatever id it just
inserted) only accounts for buildings that survived to be saved. The
`next_id_survives_removal_of_the_highest_id_building` test is the scenario
that requires the extra field — place two buildings, remove the
higher-numbered one, save, reload, and confirm a fresh placement doesn't
reissue the removed id. Recomputing `next_id` from the surviving buildings
alone would have gotten this wrong, which is why `raise_next_id` runs as
`load_city`'s last step regardless of what `insert_loaded` already did.

`city::run()` reads `LoadedSave`'s `meta.path` straight off the `App`
`world_app()` returns (`app.world().resource::<LoadedSave>()`), before
inserting anything else, and skips persistence entirely when that path is
empty — ticket 008's `empty_save` placeholder, which has nowhere on disk to
read from or write to. `CitySavePath(Option<PathBuf>)` carries the same root
forward to a `Last`-schedule system, `save_city_on_exit`, that fires on
`AppExit`. Load happens synchronously before `App::run()` (mirroring how
`load_building_catalogue`/`load_building_definitions` already work), logging
either "no save loaded, starting with an empty city" or a loaded-building
count with the file path; a load `Err` (parse failure, version mismatch,
corrupt file) logs and falls back to an empty `City` rather than panicking,
matching every other startup path in `city::run()`.

Unlike 039-042, this ticket gives `City` a real caller on both the read and
write side — `buildings()`, `roads()`, `add_road()`, and the two new methods
lost their `#[allow(dead_code)]` markers accordingly (state.rs's module docs
and the struct-level comment were updated to say so). `place_building`,
`remove_building`, `building()`, `is_tile_free`, `occupant_at`, `len`,
`is_empty` are still exercised only by tests — E1-E4 remain what will call
those.

8 new tests in `city::persistence::tests` (351 total, up from 343): a missing
file loads as an empty city; an empty city round-trips; several buildings
(one with a non-`Deg0` rotation, checked against its rotated occupancy
rectangle after reload) plus roads round-trip exactly under their original
`BuildingId`s; the removed-highest-id `next_id` scenario above; a version
mismatch is `UnsupportedVersion`; garbage RON is `Parse` not a panic; and the
two corrupt-file cases described above.

`todo.md` gets a new manual-check entry (043) — the automated suite proves
the persistence functions themselves, but not that Bevy's real `AppExit`
fires from an actual window close on this machine, or that the file lands
where a real save expects it. Recorded rather than checked here, per
CLAUDE.md.
