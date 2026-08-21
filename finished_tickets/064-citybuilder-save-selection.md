# 064 — choosing which save the citybuilder plays on

## Problem

There is no way to pick a save. `city::run()` calls `world_app()`, which calls
`load_real_save()` -> `try_load_save_from()`, and that does
`saves.into_iter().next()` — the *first* entry `get_saves_from_instance`
returns for the directory. The only existing lever is `argv[1]`, which selects
the saves *directory* (ticket 008, for CurseForge/MultiMC instances), not a
save within it.

The viewer has a runtime picker (ticket 007, `viewer::ui::save_picker`); the
citybuilder never got one, and it can't trivially reuse it — `City`, `Journal`
and `CitySavePath` are loaded once at startup from the chosen save's root, and
world writes are deferred to a manual Save (ticket 051), so switching saves
live means flushing and swapping all of that too. That is a separate feature;
this ticket is launch-time selection only.

## Scope

A second CLI argument naming the save, shared by both binaries (it lands in
`lib.rs`, below `world_app`, so the viewer honours it as its startup default
too — its runtime picker still overrides it as before).

```
cargo run --bin citybuilder                                  # first save found (unchanged)
cargo run --bin citybuilder -- "" MyCityWorld                # default saves dir, named save
cargo run --bin citybuilder -- D:/instance/saves MyCityWorld # both
cargo run --bin citybuilder -- "" D:/worlds/MyCityWorld      # a path straight to a save
```

- `argv[1]` keeps its ticket 008 meaning (saves directory). An **empty or
  whitespace-only** value now means "unset", so the default
  `.minecraft/saves` can be kept while still naming a save in `argv[2]`.
- `argv[2]` selects within that directory: exact name match first, then
  case-insensitive. A selector that names an **existing directory** is taken
  as the save itself and read with `SaveMeta::from_path`, bypassing the
  listing.
- A name that matches nothing is a ticket-008-style `Err` — the message lists
  the names that *are* available, and startup falls back to `empty_save()`
  with `StartupIssue` set, rather than panicking.

## Discoverability

Nothing on screen currently says which world the citybuilder opened, so a
typo'd name would silently look like an empty world. The city panel gets a
"World" section at the top: the active save's name and region count, plus
`StartupIssue` in red when startup couldn't load what it was asked for (the
viewer already shows this in its own save picker; the citybuilder had no
equivalent).

## Out of scope

- Runtime save switching in the citybuilder (the in-game picker window) —
  needs the `City`/`Journal`/`CitySavePath` swap and a forced world flush.
- A flag parser / `clap`. Two positional arguments, matching what `argv[1]`
  already is.

## Done when

- `cargo check` and `cargo test` pass.
- `saves_directory_from` / the new selection parsing and name matching are
  unit-tested, including the empty-string directory arg and the unknown-name
  error.
