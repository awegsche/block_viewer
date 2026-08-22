# 076 - A placed building knows its own definition

## Status
Done — `cargo test` green (694 passed).

Landed as described. Two notes on what the work turned up:

- **`journal.ron` moved to version 3 as well**, with `definition_id` read via
  `#[serde(default)]` inside its existing 1..=N band. Deliberately the
  *opposite* call from `city.ron`'s: a journalled placement written before
  this ticket genuinely has no definition, so `None` is the correct value
  rather than a guess — whereas a version-6 `city.ron` has a *catalogue* id
  sitting in the field the new one would be read from, which is a wrong
  answer, not an absent one. Both files now say so in their own docs.
- **`build_menu`'s requirement check gained two tests** that would have
  failed before the change: a placement whose blueprint stem matches but
  whose definition stem doesn't no longer satisfies a `requires`, and a
  placement with no definition unlocks nothing.

## Why

Roadmap H2's own "known gap, deliberately not smuggled in":

> `PlacedBuilding::definition` is a *catalogue* id, so a **placed** building
> still can't find its own definition (this also means `build_menu`'s
> requirement check only works while the two stems coincide). Production is
> the first mechanic that needs it per-instance, and it's a `city.ron`
> version bump — it belongs to that ticket.

This is that ticket. Everything in 077-080 (production rates, warehouse
tiers, radii, haulage) is a *definition* property read per placed instance;
none of it can start until a `PlacedBuilding` can name its `.ron`.

## The mis-naming that caused it

`PlacedBuilding::definition`'s doc comment says "the building's *type* — a key
into `BuildingDefinitions`/`BuildingCatalogue`", as though those two were one
keyspace. They are not: `BuildingCatalogue` is keyed by blueprint stem
(`house01.nbt` -> `house01`) and `BuildingDefinitions` by `.ron` stem, and
`commit::try_commit_placement` stores `selection.catalogue_id`. The two happen
to coincide for the single shipped pair, which is exactly why it went
unnoticed. Ticket 073 already had to add `PlacementSelection::definition_id`
alongside `catalogue_id` for the *selection*; this does the same for the
*placement*.

## Work

1. **Rename the field to what it holds.** `PlacedBuilding::definition` ->
   `PlacedBuilding::catalogue_id`. 19 call sites across `commit`, `demolish`,
   `journal`, `persistence`, `state`, `ui`, `undo` — mechanical, and it stops
   the next reader making the same assumption. `SavedBuilding::definition`
   on disk becomes `catalogue_id` with it.

2. **Add `PlacedBuilding::definition_id: Option<String>`.** `Some` when the
   placement came from a build-menu row (`PlacementSelection::definition_id`),
   `None` for `city::placement`'s keyboard stand-in, which has no definition
   behind it — the same hole that already leaves such a placement with no
   requirements, no cost and no production. `City::place_building` and
   `City::insert_loaded` carry it; `commit::try_commit_placement` passes
   `selection.definition_id.clone()`.

3. **`city.ron` -> version 7.** A version-6 file's buildings have no
   `definition_id` and a *catalogue* id in the renamed field. Refused, not
   defaulted, for the reason `persistence`'s module docs already give five
   times over: the moment one field is quietly defaulted the next one that
   isn't safe to default has to argue against a precedent. Add the bump note
   alongside the others.

4. **`City::definition_of(&self, id) -> Option<&str>`** — the lookup the
   whole of 077-080 goes through, so nothing else reaches into the field.

5. **`build_menu`'s requirement check** (`ui/build_menu.rs:95`) currently
   compares a `requires` entry (a definition id) against
   `placed.definition` (a catalogue id). Switch it to `definition_id`, which
   is the comparison it always meant to make.

## Not in this ticket

No migration function. A version-6 city is refused and logged the way the six
bumps before this one are; a migration is worth writing when one is, and not
per-field here.
