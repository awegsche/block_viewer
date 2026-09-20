# model-exporter — one world that houses every building model

Not a work item: the design document for `model-exporter`, a fourth
executable that bridges the two ways building models get made today.
Companion to `ROADMAP.md` (the viewer), `CITYBUILDER_ROADMAP.md` (the game)
and `RANVIL_CLI_ROADMAP.md` (the headless save/structure tool, 087–103).
**This document claims 131–137; next free ticket number: 138.**

## The problem

A building model (`assets/city/blueprints/<name>.nbt`) currently comes from
one of two disconnected pipelines:

1. **Human, in game.** Launch Minecraft, build somewhere in some save,
   launch `block_viewer`, find the build, drag a selection around it, hit
   Export, pick a filename. Nothing records *where* the build was, so
   re-exporting after a tweak means finding and re-selecting it by hand,
   and the same corner-off-by-one mistake is available every time.
2. **Agent, headless.** `ranvil-cli struct new/fill/set` writes the `.nbt`
   directly (tickets 104, 118). Fast to iterate on geometry, but nobody can
   walk through the result, and a human who wants to fix a detail has no
   way in except more `struct set` calls.

Neither path can hand a model to the other. The fix is to give every model
a **fixed, recorded home in one shared world** — the *models world*, an
empty (superflat) save the user provides — so that "the model" is a box of
world coordinates in a file, and both a human in Minecraft and an agent
with `ranvil-cli` are looking at, and editing, the same blocks.

## The shape

```
assets/models/
  world.ron            which save is the models world, its ground level,
                       the area models may be allocated in, spacing, marker block
  house02.ron          one file per model: name, origin (world coords of the
  barn.ron             blueprint's (0,0,0)), size (x,y,z), output .nbt path
  ...
```

`model-exporter` reads that directory and does four things:

| Command | Job |
|---|---|
| `model-exporter list` / `show <name>` | What's registered, where it is, whether its `.nbt` exists, and the `/tp` to get there. |
| `model-exporter new <name> <width> <height> <depth>` | Find a free spot big enough, write `<name>.ron`, place marker blocks *around* (never inside) the box in the world, print the origin and a `/tp` line. |
| `model-exporter export [<name>...]` | Every registered model (or the named ones) → its `.nbt`, via the exact `extract_blueprint`/`write_structure_file` pair `struct export` and the viewer's Export button use. Reports new/updated/unchanged/empty per model. |
| `model-exporter import <name> [--from <file.nbt>]` | The reverse bridge: put a `.nbt` (agent-built, or an older export) *into* the model's box so a human can walk through and refine it, then `export` it back. |

Plus `mark <name>` (re-place a slot's markers) and, optionally, `remove`.

Width/height/depth are the structure's **x/y/z** extents, the same order
`struct new --size x,y,z` takes.

## The architectural rule

Same rule `RANVIL_CLI_ROADMAP.md` set, restated for the new leaf:
**`model-exporter` adds no new save-format or structure-file logic.** It is
a registry (`assets/models/*.ron`) plus a slot allocator, wired around
calls that already exist:

- `ranvil_cli::save::resolve_save` — which save is the models world.
- `ranvil_cli::edit::run_write` — every world write (markers, import,
  clear) goes through 094's substrate: session lock, per-region backup,
  `--dry-run`, `--force`, `minecraft:full`-only chunks. No new write path.
- `blueprint::extract_blueprint` + `blueprint::write_structure_file` — the
  export. `model-exporter export` is `struct export` with the corners
  looked up from a `.ron` instead of typed.
- `blueprint::read_structure_file` + a `WorldEdit` of every position — the
  import, exactly `struct import --at <origin>`.
- `blueprint::catalogue::run_checks` — the "is this actually a building"
  check (`struct validate`), used to refuse exporting an empty slot.
- `ranvil_cli::format::{OutputFormat, Render, print}` and
  `ranvil_cli::error::{CliError, report}` — the `--format text|json|compact`
  and exit-code contract, reused verbatim so an agent driving both tools
  parses one shape.

Nothing here touches Bevy ECS or opens a window. The only genuinely new
logic is the registry loader/validator (131) and the first-fit slot
allocator (133), both pure functions with unit tests.

## Crate layout

One more `pub mod` in the lib, one more three-line `[[bin]]` shim — the
shape 027 set for the game and 087 repeated for `ranvil-cli`.

```
Cargo.toml                 [[bin]] model-exporter -> src/bin/model_exporter.rs
                           (no new dependencies: clap, ron, serde already present)
src/
  lib.rs                   + pub mod model_exporter;
  bin/model_exporter.rs    fn main() -> ExitCode { block_viewer::model_exporter::run() }
  model_exporter/
    mod.rs                 pub fn run() -> ExitCode — parse, dispatch, map errors
    cli.rs                 clap derive: Cli, Command enum, per-command args
    registry.rs            ModelWorld / ModelSlot RON schema, load_registry(),
                           validation (131)
    allocate.rs            first-fit slot placement (133)
    markers.rs             marker-block geometry as a WorldEdit (134)
    list.rs                list / show (132)
    new.rs                 new (133 + 134)
    export.rs              export (135)
    import.rs              import (136)
    remove.rs              remove (137)
```

`ranvil_cli::save::resolve_save` currently takes `&ranvil_cli::cli::Cli`;
ticket 132 splits its body into `resolve_save_from(save: Option<&str>,
instance: Option<&Path>)` and keeps the `Cli` wrapper, so both CLIs share
one implementation of "which save did they mean".

## The registry

### `assets/models/world.ron`

```ron
ModelWorld(
    // The models world: a save name under the instance directory (the
    // same resolution `ranvil-cli --save` does), or a path to one save.
    save: "models",
    // Y of the topmost ground block of the flat world (e.g. the grass
    // layer). The default 1.18+ superflat preset puts it at -61; check
    // with `ranvil-cli --save models column 0,0`.
    ground_y: -61,
    // Models are only ever allocated inside this XZ rectangle (inclusive).
    area: (min: (x: 0, z: 0), max: (x: 511, z: 511)),
    // Minimum clear blocks between two models' boxes. The marker ring
    // sits 1 outside each box, so 3 leaves one empty block between rings.
    gap: 3,
    // Candidate origins are scanned on this XZ step — keeps the layout
    // legible (models line up) without a real packer.
    grid: 8,
    // The delimiter block placed around a new slot.
    marker: "minecraft:orange_terracotta",
    // Where `export` writes by default, joined with `<name>.nbt`.
    blueprints_dir: "assets/city/blueprints",
)
```

### `assets/models/<name>.ron`

```ron
Model(
    name: "house02",
    // World coordinates of the blueprint's own (0,0,0). `origin.y` is
    // normally `ground_y - below` (see `new --below`), so the blueprint's
    // y=0 is its foundation layer, the convention `ground_level: 1`
    // buildings already follow.
    origin: (x: 16, y: -62, z: 0),
    // Extents on x, y, z — the exported structure's `size`.
    size: (x: 12, y: 10, z: 12),
    // Optional. Defaults to `<blueprints_dir>/<name>.nbt`.
    out: None,
)
```

The file's stem must equal `name`. Loading the directory validates: unique
names, every size within `STRUCTURE_BLOCK_MAX_SIZE` per axis, every box
inside `area`, no two boxes closer than `gap`. A registry that fails
validation refuses every command (exit 2) with the offending file named —
the same "a bad definition is a usage error" stance
`city::definition::load_definitions_dir` takes, except that here nothing
is skipped-and-logged, because a wrong `origin` silently exports the wrong
blocks.

## Coordinates and markers

For a model with `origin` and `size`, inclusive box
`min = origin`, `max = origin + size - 1`. The export reads exactly that
box and nothing else — blocks a human leaves outside it are ignored, and
the markers are never inside it.

Markers (ticket 134), all in `world.marker`:

- **Ring** at `y = ground_y`, one block outside the footprint on every side:
  `x ∈ [min.x-1, max.x+1]`, `z ∈ [min.z-1, max.z+1]`, minus the interior.
  Replaces the ground block there, so it's a flush coloured outline you
  build *inside* of.
- **Corner pillars** on the four ring corners, `y ∈ [ground_y+1, max.y]`.
  The pillar top is the highest layer the model may use: *build no higher
  than the pillars.*

Nothing is placed above `max.y` or below `ground_y` on the ring, so a
`--below` foundation layer stays free for digging. `new` prints:

```
house02: origin (16,-62,0) size 12x10x12  box (16,-62,0)..(27,-53,11)
  build inside the orange ring, no higher than the pillar tops
  /tp @s 22 -60 -3
```

(The teleport lands you on the ground just outside the ring's north edge,
facing the slot.)

### Only generated chunks

An empty superflat save only has chunks near spawn. `run_write` refuses
chunks whose `Status` isn't `minecraft:full` — correct, and it means a slot
allocated in ungenerated terrain can't be marked and would export as
`failed_columns`. The allocator (133) therefore only considers candidates
whose chunk columns are all generated, and `new` says so when that (rather
than the area being full) is why nothing fit: *"no free spot with generated
chunks — fly around the area in Minecraft (or pre-generate it, see
`docs/world-pregeneration.md`) and retry."*

## Bridging both pipelines

With the registry in place the two workflows become one loop:

- **Human first.** `new barn 14 9 11` → teleport → build → save & quit →
  `export barn` → `barn.nbt`. Tweak later: teleport back, edit, `export`.
- **Agent first.** `ranvil-cli struct new/fill/set` → `barn.nbt` →
  `import barn --from barn.nbt` (allocates a slot sized from the file if
  `barn` isn't registered yet, marks it, places the blocks) → human walks
  through, fixes the roof → `export barn`.
- **Agent edits a human build.** Every registered model's world box is
  known, so `ranvil-cli --save models get/set/scan <coords>` against the
  models world edits the source of truth directly, and the next `export`
  picks it up. `show <name>` prints the box so an agent doesn't have to
  compute it.

`export` refuses (per model, without failing the others) when the save is
open in Minecraft unless `--force` — a half-flushed region is the one way to
export a build that isn't what the human sees. It also refuses a slot whose
box is still marker-free ground/air (`run_checks`' air-only check) as
`empty`, rather than overwriting a good `.nbt` with nothing.

## Deliberately not in this plan

- **Writing the building's `.ron` definition** (`assets/city/buildings`).
  `export` produces the `.nbt`; `cost`, `tier`, `ground_level` and the rest
  are citybuilder design decisions, not something a coordinate box knows.
  Worth a later `--scaffold-definition` if it turns out to be boilerplate
  every time.
- **A name sign at each slot.** Signs carry text in a block entity, and the
  edit path writes block states only. The ring plus `list` covers "which
  slot is this".
- **Rendering / thumbnails.** Same reason `RANVIL_CLI_ROADMAP.md` gave.
- **Multiple model worlds.** One `world.ron` per `assets/models` directory;
  `--models-dir` already points at a different directory if a second world
  is ever wanted.
- **Auto-detecting the box from what was built** (shrink-wrapping). The
  human builds inside a declared box; if it needs to grow, edit `size` in
  the `.ron` (validation catches a resulting overlap) and `mark` again.

## Ticket breakdown

```
131  registry schema + loader + validation           <- pure, no I/O beyond RON
      |
      132  bin shim, CLI skeleton, list / show,
           resolve_save_from() split                 <- foundation, read-only
      |
      133  slot allocator + `new` (writes the .ron,   <- pure allocator + one file
      |    prints origin and /tp)
      |
      134  marker geometry + `new` places markers,
      |    `mark <name>` re-places them              <- first world write (run_write)
      |
      135  export [names] — the point of the tool
      |
      136  import <name> [--from file.nbt]           <- reverse bridge (uses 133+134)
      |
      137  remove <name> [--clear]                   <- optional
```

### Ordering advice

- **131 → 132 → 133 → 134 → 135 in order.** Each is a few hours and the
  next one has nowhere to attach without it. 135 is the ticket that
  delivers the user-facing goal; 131–134 exist so that 135 is a lookup and
  a call into `extract_blueprint`, nothing more.
- **133 and 134 are split on purpose.** 133 is the allocator (pure, tested
  against synthetic registries) plus writing one `.ron`; 134 is the first
  ticket that writes to a real save. Keeping the world write in its own
  ticket means a bug in marker geometry can't be confused with a bug in
  allocation, and `new --no-markers` (133's form) stays useful for
  registering a slot in a world one doesn't want to touch yet.
- **136 needs 133 and 134** (to allocate and mark a slot for an
  unregistered `--from` file) and reuses `struct import`'s exact write.
- **137 is optional.** Deleting the `.ron` by hand and leaving the markers
  is a fine workaround; the ticket exists so the markers and box can be
  cleaned up properly when it matters.

### Manual verification

Every ticket from 134 on has a check only a human at the Minecraft window
can do (are the markers where the printout says, does the imported model
appear, does the export match what was built). Per `CLAUDE.md`, those go
into `todo.md` with the exact command to run, not into the ticket's "done
when".

### Before 132 can run against the real world

The user provides the empty world. Once it's in the instance directory,
`assets/models/world.ron` needs its `save` name and `ground_y` filled in
(`ranvil-cli --save <name> column 0,0` shows the ground layer) — 131 ships
the file with the superflat defaults above and a comment saying so.
