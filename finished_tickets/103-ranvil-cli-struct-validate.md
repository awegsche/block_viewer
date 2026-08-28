# 103 - ranvil-cli: `struct validate`

Last ticket in the plan. Lets an agent authoring a building model
self-check a file before it's dropped into `assets/city/blueprints` — the
same checks `blueprint::catalogue::load_catalogue_dir` already applies
silently (a bad file there is just skipped and logged), made explicit and
runnable on demand.

## Scope

`ranvil-cli struct validate <file.nbt> [--max-size <x,y,z>]` — runs the
checks `catalogue.rs`'s loader applies to every `*.nbt` it scans (read the
module first rather than re-deriving the list from memory; expected to
include at least: file parses as a structure at all, size is within the
catalogue's configured limit — `--max-size` overrides the default so this
command can be used against a non-building structure with different limits
— and the palette is not air-only, since an all-air structure is what a
failed/empty extraction looks like, not a real building). Reports pass/fail
per check, not just an overall verdict — `json`'s `"checks": [{"name":
"size_limit", "pass": true, "detail": "..."}, ...]` lets an agent see
*which* check failed rather than re-deriving it by hand.

Exit code: `0` if every check passes, `1` if the file parses but fails one
or more checks (a `CliError::Data`-shaped outcome — this is a real answer
about the file, not a bad request), `2` only if the file doesn't parse as a
structure at all.

Whether `catalogue.rs`'s check list itself changes to call through a shared
function this command also calls (rather than this ticket duplicating the
list) is preferred where it's a small refactor; where the loader's checks
are entangled with its directory-scan flow in a way that isn't worth
untangling for this ticket, duplicating the *specific numeric limits* here
(with a comment pointing at `catalogue.rs` so the two don't silently drift)
is an acceptable fallback — call out which one was done in the ticket's
commit.

## Done when

- `struct validate` against a real shipped blueprint (`assets/city/
  blueprints/house01.nbt`) passes every check.
- `struct validate` against an all-air structure (`struct new` with default
  fill) fails the non-air check specifically, other checks unaffected.
- `struct validate` against a structure built oversized (`struct new
  --size` past the catalogue's real limit) fails the size check with the
  actual vs. allowed size in the message.
- If `catalogue.rs`'s limits ever change, this command's `--max-size`
  default changes with them (shared constant/function, not a copied
  number) — or, if duplicated per the fallback above, a comment marks the
  spot to keep in sync.
- `cargo check` and `cargo test` pass.
