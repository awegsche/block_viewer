# ranvil-cli — a third executable for agent-driven save/model inspection

Not a work item: the design document for `ranvil-cli`, a headless command-line
entrypoint for observing, scanning and modifying Anvil saves and exported
structure (building) files. Companion to `ROADMAP.md` (the viewer, 001–029)
and `CITYBUILDER_ROADMAP.md` (the game, 027–). **Next free ticket number:
104** (this document claims 087–103).

## The goal

An agent — Claude Code, in practice — needs to look inside real Minecraft
saves and building models without a human at a keyboard. Today that means
either reading raw NBT by hand or driving the GUI, neither of which an agent
can do. `ranvil-cli` is the missing entrypoint: single commands in, one
formatted answer out, scriptable and composable.

Three concrete jobs motivate it, in the citybuilder's own words:

1. **Survey real saves** to plan citybuilder work: how big is this world,
   what's actually generated, what's at a given spot, before designing a
   feature around it.
2. **Inspect and edit building models** — the `.nbt` structure files under
   `assets/city/blueprints` — extracting information (size, palette, block
   count) or modifying them (add a chest, resize the footprint, add a
   basement or headroom) without opening Minecraft or the viewer.
3. **Author building models from scratch** by prompting an agent: create a
   blank structure, fill it in block by block or region by region, validate
   it, drop it in the catalogue.

Reading a live save and reading a structure file are almost the same
operation (a box of `BlockState`s), so the command surface treats them
uniformly wherever it can: `get`/`set`/`fill`-shaped commands exist for both,
under `ranvil-cli <cmd>` for a live save and `ranvil-cli struct <cmd>` for a
file.

## The architectural rule

**`ranvil-cli` is a thin CLI over the existing `block_viewer` lib — it adds
no new save-format logic.** Region I/O, chunk decode, block-state
read/write, blueprint extraction, structure file read/write, and rotation
all already exist (`mc_anvil`/`ranvil`, and this crate's `region_cache`,
`edit`, `blueprint` modules) because the viewer and citybuilder both needed
them first. `ranvil-cli` wires argument parsing and output formatting around
calls into that code — it does not reimplement NBT decoding, section
packing, or the structure format.

This matters for what's *cheap* here versus what looks cheap and isn't:

- `saves`, `info`, `regions`, `chunk*`, `get*`, `column`, `scan` — read-only,
  wrap `ranvil::save`/`chunkregion`/`heightmap` directly. Cheap.
- `set*`, `replace`, `copy` — wrap `edit::{plan, apply}` and need the same
  write-safety `edit::session::WriteSession` already gives the game (session
  lock, per-region backup, all-or-nothing apply). Not reimplemented, just
  driven headlessly.
- `struct export`/`import` — wrap `blueprint::extract_blueprint` and
  `blueprint::structure::{read,write}_structure_file`, the exact functions
  behind the viewer's selection Export button and the citybuilder's
  catalogue loader. `ranvil-cli` is the first caller that reaches them
  without a running Bevy `App` at all.
- `struct get`/`set`/`fill`/`resize`/`rotate` — mostly new: nothing today
  edits a `Blueprint` in memory outside of extraction and rotation. This is
  the one place real logic gets added, and it's scoped tightly (group E).

None of `extract_blueprint`, `RegionCache`, `edit::plan`/`apply`, or the
structure reader/writer touch Bevy ECS — they're plain Rust (or
`bevy::math::IVec3`, which is glam and carries no runtime). `ranvil-cli`'s
`main` never constructs an `App`, opens a window, or touches wgpu; it calls
these functions directly. `mesh_blueprint` (which *does* return a Bevy
`Mesh`) has no command in this plan — nothing here renders.

## Crate layout

Same shape `CITYBUILDER_ROADMAP.md` set for the game: one more `pub mod` in
the lib, one more three-line `[[bin]]` shim.

```
Cargo.toml            [[bin]] ranvil-cli -> src/bin/ranvil_cli.rs
                       + clap (derive) — the one new dependency this adds
src/
  lib.rs               + pub mod ranvil_cli;
  bin/ranvil_cli.rs    fn main() -> ExitCode { block_viewer::ranvil_cli::run() }
  ranvil_cli/
    mod.rs             pub fn run() -> ExitCode — parses argv, dispatches, maps
                        errors to exit codes
    cli.rs             clap derive: top-level Cli, the Command enum, and every
                        subcommand's args (coordinates, block states, paths)
    format.rs          OutputFormat (Text | Json | Compact) + a `Render` trait
                        implemented per result type — see "Output formats"
    error.rs           CliError, the JSON error envelope, exit code mapping
    coords.rs           "x,y,z" / "x,z" parsing shared by every subcommand
    save.rs            saves / info / regions / lock
    chunk.rs           chunk / chunks / heightmap
    block.rs           get / get-area / column / scan               (read)
    edit.rs            set / set-area / set-batch / replace / copy  (write)
    structure.rs        struct info / new / export / import / get / set /
                        fill / resize / rotate / diff / validate
```

`ranvil_cli` is a `pub mod` in the lib next to `city`/`viewer`, not a
separate crate — the same "one package, one lib" call `CITYBUILDER_ROADMAP.md`
already made, for the same reason (shared modules stay `pub(crate)`-reachable
from all three bins).

## Output formats

Every command accepts `--format text|json|compact` (default `text`).
Human-facing prose lives only in `text`; `json` and `compact` are the
agent-facing contract and are covered by the same tests as their data.

- **`text`** — multi-line, labeled, meant for a human terminal. What
  `--print` on `chunk` already implies in the prompt that started this.
- **`json`** — one JSON value on stdout (pretty-printed unless
  `--compact-json` is also given), stable field names, safe to `jq`/parse.
  This is the primary agent format for anything structured (a block's
  properties, a palette, a diff).
- **`compact`** — one line, terse, whitespace- or comma-delimited,
  optimised for token cost in an agent's own context rather than for parsing
  — e.g. `chunk 0,0 --format compact` might read
  `0,0 full sections=-4..19 entities=3 blockents=1 inhabited=118402`. Not
  every command needs a `compact` form (a single `get` is already one line
  in any format); ticket 087 sets the pattern, later tickets say per-command
  whether `compact` differs from `text`.

Errors follow the format flag too: `text` prints a one-line message to
stderr; `json` prints `{"error": {"kind": "...", "message": "..."}}` to
stdout so a caller parsing JSON never has to branch on exit status to find
out what shape the output is. Exit codes: `0` success, `1` a well-formed
request that failed for a data reason (chunk not found, region locked,
DataVersion mismatch), `2` a usage error (bad arguments, unparseable
coordinates or block state) caught before anything is touched.

## Command reference

Coordinates are `x,y,z` (blocks) or `x,z` (chunks/regions), no spaces.
Block states are `minecraft:oak_stairs[facing=east,half=top]` — the same
text form `blueprint::BlockState`'s `Display`/`FromStr` already round-trips
(ticket 035 added `FromStr` as `Display`'s inverse for the viewer's paint
tool; this reuses it rather than growing a second parser).

### Discovery

| Command | Reads | Purpose |
|---|---|---|
| `ranvil-cli saves [--instance <dir>]` | `mc_anvil::get_saves_from_instance` | List saves under an instance dir: name, path, on-disk size, locked. |
| `ranvil-cli info` | `SaveMeta`, level data | Save-level summary: seed, DataVersion, dimensions present, region count, locked. |
| `ranvil-cli regions [--format grid\|list\|json]` | `SaveMeta`/`GridView` | Region files present: coords, file size, chunk count. `grid` is `GridView`'s ASCII map. |
| `ranvil-cli lock` | `SaveMeta::is_locked` | Is Minecraft holding `session.lock` right now. |

### Chunks

| Command | Purpose |
|---|---|
| `ranvil-cli chunk <cx>,<cz> [--print]` | One chunk: `Status`, DataVersion, section Y-range present, inhabited time, block-entity/entity counts, biome list, a heightmap peek. |
| `ranvil-cli chunks <cx1>,<cz1> <cx2>,<cz2>` | Bulk survey over a rectangle of chunk coordinates: counts by status, aggregate section/entity counts, which cells are ungenerated. The "how much of this world exists" query. |
| `ranvil-cli heightmap <cx>,<cz> [--kind world-surface\|motion-blocking\|motion-blocking-no-leaves\|ocean-floor]` | 16×16 height grid (`ranvil::heightmap`), ascii grid or json. |

### Blocks — read

| Command | Purpose |
|---|---|
| `ranvil-cli get <x>,<y>,<z>` | One block's name + properties. |
| `ranvil-cli get-area <x1>,<y1>,<z1> <x2>,<y2>,<z2>` | A box, as a palette + dense index array (the same shape `Blueprint` already is) rather than one line per block — the cheap form for an agent reading thousands of blocks. |
| `ranvil-cli column <x>,<z> [--from <y>] [--to <y>]` | Vertical block list for one column — terrain/profile reads. |
| `ranvil-cli scan <x1,y1,z1> <x2,y2,z2> --block <name> [--limit N]` | Every position matching a block name inside a box — "find the chests", "find the water". |

### Blocks — write

Every command below shares one write-safety substrate (ticket 094): holds
`SessionLock`, backs up every touched region before saving any, refuses when
DataVersion bands disagree, supports `--dry-run` (report the plan, touch
nothing) and requires `--force` to proceed when the save is open in
Minecraft.

| Command | Purpose |
|---|---|
| `ranvil-cli set <x>,<y>,<z> <blockstate>` | Write one block. |
| `ranvil-cli set-area <x1,y1,z1> <x2,y2,z2> <blockstate>` | Fill a box with one state. |
| `ranvil-cli set-batch <file\|->` | Many discrete `x,y,z blockstate` edits, one per line, from a file or stdin — the shape an agent's own diff comes in. |
| `ranvil-cli replace <x1,y1,z1> <x2,y2,z2> --from <name> --to <blockstate>` | Find-and-replace within a box in one pass. |
| `ranvil-cli copy <x1,y1,z1> <x2,y2,z2> --to <x,y,z>` | Extract a box and re-apply it translated elsewhere in the same save. |

### Structure files (`.nbt` building models)

Everything under `struct` reads/writes a `Blueprint` on disk and — except
`export`/`import`, the two that touch a live save — never opens a save at
all.

| Command | Purpose |
|---|---|
| `ranvil-cli struct info <file.nbt>` | Size, origin, block count, DataVersion, palette (size + listing). |
| `ranvil-cli struct new --size <x,y,z> --out <file.nbt> [--fill <blockstate>]` | A blank structure file from scratch — air, or one state throughout. |
| `ranvil-cli struct export <x1,y1,z1> <x2,y2,z2> --out <file.nbt>` | World volume → structure file. The headless form of the viewer's select-and-Export. |
| `ranvil-cli struct import <file.nbt> --at <x,y,z> [--rotate 90\|180\|270]` | Structure file → world at a position. |
| `ranvil-cli struct get <file.nbt> <x,y,z>` | Read one block inside a structure file. |
| `ranvil-cli struct set <file.nbt> <x,y,z> <blockstate> [--out <file2.nbt>]` | Edit one block in place (or write to a new file, leaving the original untouched). |
| `ranvil-cli struct fill <file.nbt> <x1,y1,z1> <x2,y2,z2> <blockstate> [--out <file2.nbt>]` | Fill a sub-box — carve a chest well, clear a floor. |
| `ranvil-cli struct resize <file.nbt> --out <file2.nbt> [--pad-y-top N] [--pad-y-bottom N] [--pad-x-neg/pos N] [--pad-z-neg/pos N] [--fill <blockstate>]` | Add or trim margin on any face — a basement, headroom, a footprint trim. |
| `ranvil-cli struct rotate <file.nbt> --by 90\|180\|270 --out <file2.nbt>` | Wraps `blueprint::rotate_blueprint` directly. |
| `ranvil-cli struct diff <a.nbt> <b.nbt>` | Per-position block differences between two same-size structures — confirm a modification did what was intended. |
| `ranvil-cli struct validate <file.nbt> [--max-size <x,y,z>]` | The same non-air/size checks `blueprint::catalogue::load_catalogue_dir` applies to every `.nbt` under `assets/city/blueprints`, runnable standalone before a new or edited file is dropped there. |

### Deliberately not in this plan

- **Catalogue/definition-aware commands** (listing `assets/city/buildings`
  `.ron` defs, validating a definition's `ground_level` against its matched
  blueprint's real height, checking `requires` for cycles). All real logic
  this would wrap (`city::definition::load_entry`,
  `resolve_requirements`) already exists and is citybuilder-specific;
  `ranvil-cli` stays a save/structure-file tool. Worth a later ticket if
  agent-driven building authoring turns out to need it, but nothing in the
  three motivating jobs above requires it yet.
- **Rendering a structure to an image** (`struct render`). Would need
  `mesh_blueprint` plus a headless wgpu surface — real work, no motivating
  job asked for it. `struct info`'s palette listing plus `struct get`/`scan`
  cover "what's in here" without a picture.
- **A flag/env var for a second Minecraft instance directory beyond
  `--instance`.** `saves`/the save resolver already take that from
  `ranvil::get_saves_from_instance`; nothing here needs more.
- **Cross-save `copy --dest-save`.** The single-save `copy` (097) already
  covers "move a build"; a second save adds a second `RegionCache`, a second
  lock, and a second DataVersion check for a use case that hasn't come up.
  Revisit if it does.

## Ticket breakdown

```
087  CLI skeleton: bin shim, arg parsing, --format contract,
     error envelope + exit codes, `saves`               <- foundation
      |
      088  info / regions / lock
      089  chunk / chunks
      090  heightmap
      |
      091  get
      092  get-area                          <- shared box-scan primitive
      |     |
      |     093  column / scan
      |
      094  write session: lock + backup +
           dry-run + force                    <- write substrate
      |     |
      |     095  set / set-area
      |     096  set-batch / replace
      |     097  copy                    (uses 092's scan + 094's write)
      |
      098  struct info / struct new
      099  struct export / struct import  (export needs only 092's kind of
      |                                    read; import needs 094's write)
      100  struct get / struct set / struct fill
      101  struct resize
      102  struct rotate / struct diff
      103  struct validate
```

### Ordering advice

- **087 first, always.** Nothing else has anywhere to attach without the
  arg-parsing skeleton and the format/error contract — deciding those
  per-command later is how a CLI ends up with three incompatible ideas of
  what `--format json` means.
- **Group E (098–103) needs only 087.** `struct info/new/get/set/fill/
  resize/rotate/diff/validate` operate on a `Blueprint` read from or written
  to a file and never open a save — they don't depend on groups B/C/D at
  all. If the building-model-authoring job (motivating job 2/3 above) is
  more urgent than the save-survey job, build E right after 087 and do B–D
  later. The one bridge is **099's `export`/`import` pair**, which is why
  it's placed after D in the diagram — `export` only needs a `RegionCache`
  read (group C's tier), `import` needs 094's write session, so 099 can't
  land before both, but its two subcommands could ship as separate tickets
  if that split turns out to matter.
- **092 (`get-area`) is worth getting right early.** It's `extract_blueprint`
  under a CLI wrapper, and `097` (`copy`) and `099` (`struct export`) both
  build on the same box-scan shape. Do not let `093`/`096`/`097` grow their
  own box-walking loop instead of reusing it.
- **094 (write session) is the one ticket in group D worth over-designing
  slightly.** Every write command after it inherits whatever safety
  guarantees it has — get the lock/backup/dry-run/force story right once,
  the way `edit::session::WriteSession` did for the game (ticket 033), and
  095–097 are thin.

## Suggested milestones

1. **M1 — "it runs"**: 087. `ranvil-cli saves` lists real saves; the
   format/error contract is proven end to end.
2. **M2 — "I can look around"**: 088–093. Every read-only survey and
   block-inspection command. This alone answers motivating job 1 (plan
   citybuilder work against a real save) and half of job 2 (inspect a
   structure file's palette via `struct info`, once E starts).
3. **M3 — "I can change the world"**: 094–097. Headless world edits with
   the same safety story the game already has.
4. **M4 — "I can make and fix building models"**: 098–103. Create, inspect,
   edit, resize, rotate, diff and validate structure files without
   Minecraft or the viewer open — the rest of jobs 2 and 3.

M2 and M4 do not depend on each other (see "092 needs only 087" above) and
can run in either order or in parallel.
