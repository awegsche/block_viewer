# 094 - ranvil-cli: the write session (lock, backup, dry-run, force)

The write substrate every command in group D (095–097) and half of 099
(`struct import`) shares. Get this right once rather than five times —
`edit::session::WriteSession` already did exactly this for the game
(ticket 033); this ticket is that same shape, driven headlessly from a CLI
instead of `AppExit`.

## Problem

`edit::session::WriteSession` exists and does the right things
(`SessionLock::acquire` held for the session, every touched region backed
up before any is saved, `WriteError::WriteFailed` naming what was and
wasn't written), but it's built around the game's flow: one session opened
around a manual Save action, edits accumulated in a shared `RegionCache`
over a play session, flushed once. A CLI invocation is a single edit (or
batch) start to finish in one process — the session's lifetime is the whole
command, not something a user triggers separately.

## Scope

`src/ranvil_cli/edit.rs`'s shared entry point, called by every write command
below:

```rust
fn run_write(save: &SaveMeta, dry_run: bool, force: bool,
             build: impl FnOnce(&mut RegionCache) -> Result<edit::WorldEdit, CliError>)
    -> Result<edit::EditReport, CliError>
```

- Acquires `SessionLock` up front (`WriteSession`'s own acquire path);
  `is_locked()` failing without `--force` is `CliError::Data` ("save is open
  in Minecraft — pass --force to write anyway"), matching the roadmap's
  write-command contract. `--force` still acquires the lock (best-effort);
  it overrides the pre-check, not the acquisition itself.
- Builds a `RegionCache` scoped to just this command (capacity sized to the
  edit's own footprint — a `set` needs one region, `set-batch` as many as
  its positions span; no need for the streaming-sized caches the viewer
  uses).
- Runs `build` to get a `WorldEdit`, then `edit::plan` + `edit::apply` (the
  same two-phase "plan every region before applying any" path `edit::route`
  already documents) through an `EditPolicy` — this ticket picks the
  policy's specifics: `Status` gate stays `minecraft:full`-only (unchanged
  from the game's policy — a `ranvil-cli set` into an ungenerated chunk
  should refuse exactly like the game does, not carve a new exception), and
  `HeightmapPolicy` defaults to recompute-at-end-of-transaction, same as the
  game.
- `--dry-run`: runs `edit::plan` and reports what *would* change (regions
  touched, blocks affected, any refusal `plan` would raise) without calling
  `apply` or backing up anything. This is the one behaviour genuinely new
  here versus `WriteSession` — the game has no dry-run mode because a
  player's placement preview already serves that role; a CLI caller has no
  such preview and needs to ask in words.
- On success (not dry-run): backs up every touched region (`WriteSession`'s
  existing backup-before-any-save behavior), saves, and returns an
  `EditReport` for the caller to summarise (blocks written, regions
  touched, backup directory path).
- A mid-write failure reports exactly what `WriteError::WriteFailed` already
  carries — which regions saved, which didn't, where the backups are — and
  exits 1, not a panic.

Whether this calls `WriteSession` directly or duplicates its (small) body
because `WriteSession` is wired to Bevy resources `ranvil-cli` doesn't have
is an implementation decision for whoever picks this up — read
`edit::session` first; if its actual logic is Bevy-free underneath (likely,
given `edit`'s own module only imports `bevy::math::IVec3`), prefer calling
it directly over copying it.

## Out of scope

- The actual write commands (095–097) — this ticket ships with no command
  that uses `run_write` yet beyond a test harness, the same "prove the
  substrate before building on it" shape 033 and 094→095 mirror from the
  citybuilder roadmap's own edit-model/write-safety split.

## Done when

- A unit/integration test drives `run_write` against a fixture save: a
  successful single-block edit backs up and writes correctly; `--dry-run`
  reports the plan and leaves the fixture's mtimes/bytes untouched; a
  locked save without `--force` refuses; a DataVersion-incompatible edit
  refuses with the same error `edit::plan` already raises for the game.
- `cargo check` and `cargo test` pass.
