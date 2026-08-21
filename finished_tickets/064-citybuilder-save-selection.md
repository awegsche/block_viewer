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

## Follow-up: the empty first argument didn't survive the shell

Shipped requiring `-- "" <save name>` to keep `argv[1]`'s ticket 008 meaning
unambiguous. That doesn't work: **Windows PowerShell 5.1 does not pass an
empty argument to a native executable intact** — it arrives either dropped
(so the save name lands in `argv[1]` and is read as a saves directory, which
fails with an I/O error naming a path that doesn't exist) or as two literal
quote characters. Either way the documented form failed on the shell this
repo is developed on. pwsh 7 passes it fine, which is why it looked correct
when tested.

Fixed by classifying a **single** argument against the filesystem instead of
by position (`resolve_selection_in`):

- a directory that lists at least one save is ticket 008's saves directory,
  unchanged — `citybuilder D:/curseforge/instance/saves`;
- anything else is the save — `citybuilder nbt_test`, or a path straight to
  one, `citybuilder D:/worlds/nbt_test`. A save path is readable as a
  directory but lists no saves of its own (`data`, `datapacks`, `dimensions`
  aren't saves), which is what separates the two cases.

Two arguments still mean directory then save, and the empty-first form still
resolves for shells that do pass it through — `save_args_from` now treats an
argument that is empty, whitespace, or nothing but quote characters as not
given, which covers PowerShell 5.1's literal-quotes delivery too.
