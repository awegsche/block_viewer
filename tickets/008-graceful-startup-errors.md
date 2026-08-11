# 008 - Don't panic when there's no save to load

## Status
Open

## Depends on
Nothing. Small; pull it forward whenever it gets in the way.

## Problem

`load_real_save()` in `src/main.rs` panics on every unhappy path, before the
window ever opens — the user just sees the process die with a backtrace:

- `get_saves().expect("could not read the Minecraft saves directory")` —
  fires when `.minecraft/saves` doesn't exist (no Minecraft installed, or a
  CurseForge/MultiMC instance elsewhere).
- `.next().expect("no Minecraft saves found in the saves directory")` —
  fires on an empty saves folder.
- `load_chunks().expect("failed to load chunks for the first region")` —
  fires on a truncated or unusual region file, killing the app over one bad
  file among hundreds.

`SaveMeta::from_path` also propagates an error if a save directory has no
`region/` subdirectory, so one odd folder under `saves/` currently takes down
the whole listing (`get_saves_from_instance` `collect()`s into a `Result`).

## Goal

The app always starts. Problems are surfaced, not fatal.

## Scope

- `LoadedSave` becomes a state that can represent "nothing loaded yet" /
  "load failed" rather than an unconditional `Save`.
- Errors reach the user through the UI (007) — or, until that lands, a clear
  single-line log plus an empty world, never a panic.
- A per-region/per-chunk failure skips that region/chunk and logs it once;
  it must not abort the whole load.
- Support saves outside the default directory: `mc_anvil` already exposes
  `get_saves_from_instance(path)` for exactly this (CurseForge etc.). Accept
  a saves directory via CLI arg or config, defaulting to
  `dirs::config_dir()/.minecraft/saves`.
- Consider whether a malformed entry under `saves/` should be skipped rather
  than failing the listing — if so that's an upstream tweak in `../ranvil`'s
  `get_saves_from_instance`.

## Done when

- Running with no `.minecraft` directory at all opens a window and says so.
- Running with an empty `saves/` folder opens a window and says so.
- A deliberately corrupted `.mca` file is skipped with a log line; the rest
  of the world still loads.
- No `expect`/`unwrap` remains on a path reachable from user data.
