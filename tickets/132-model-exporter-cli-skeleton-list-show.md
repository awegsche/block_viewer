# 132 - model-exporter: bin shim, CLI skeleton, `list` / `show`

Design: `MODEL_EXPORTER_ROADMAP.md` ("Crate layout"). Depends on 131. The
087 ticket for this tool: the executable exists, the format/error contract
is proven end to end on two read-only commands, and every later command
has somewhere to attach.

## Scope

- **`Cargo.toml`**: `[[bin]] name = "model-exporter", path =
  "src/bin/model_exporter.rs"`. No new dependencies — `clap`, `ron`,
  `serde` are all already here.
- **`src/bin/model_exporter.rs`**: the three-line shim,
  `block_viewer::model_exporter::run()`.
- **`src/model_exporter/cli.rs`**: `Cli` with globals `--models-dir
  <dir>` (default `assets/models`), `--save`/`--instance` (override
  `world.ron`'s `save`; same semantics as `ranvil-cli`'s), `--format
  text|json|compact` (reuses `ranvil_cli::format::OutputFormat`
  directly). `Command` enum with `List` and `Show { name }` for now; later
  tickets add their variants.
- **`src/model_exporter/mod.rs`**: `run() -> ExitCode` and `dispatch`,
  the same shape as `ranvil_cli::run`/`dispatch`. Errors are
  `ranvil_cli::error::CliError`, reported through
  `ranvil_cli::error::report` — one envelope, one exit-code mapping,
  shared. A `RegistryError` maps to `CliError::Usage` (a bad registry is a
  bad argument, caught before anything is touched).
- **`ranvil_cli::save::resolve_save` split**: extract its body into
  `pub fn resolve_save_from(save: Option<&str>, instance: Option<&Path>)
  -> Result<SaveMeta, CliError>`; `resolve_save(cli)` becomes a one-line
  wrapper. `model_exporter` calls `resolve_save_from(cli.save.or(world.save),
  cli.instance)`. `ranvil-cli`'s own tests keep passing unchanged.
- **`src/model_exporter/list.rs`**:
  - `list`: one row per slot — name, origin, size, inclusive box,
    `.nbt` status (`missing` / `present`, and its size from
    `read_structure_file` if present: `present (12x10x12)` / `present
    (SIZE MISMATCH 12x8x12)` when the file's size disagrees with the
    slot's), and the `/tp` line. `text` is a table; `compact` one line per
    slot; `json` an array of objects with `name`, `origin`, `size`,
    `min`, `max`, `out`, `nbt: {present, size}`, `tp: [x, y, z]`.
    Does **not** open the save — `list` must work with Minecraft running.
  - `show <name>`: the same fields for one slot, in full, plus the marker
    ring's `y` and corner coordinates (134's geometry, so a human or agent
    can find the slot without the markers) and the equivalent `ranvil-cli
    --save <world> get-area <min> <max>` line for an agent that wants the
    blocks. Unknown name → `CliError::Usage`.
  - The teleport position, defined once (`tp_position(slot, world)`) and
    reused by 133's `new`: `x = center of the footprint`, `y = ground_y +
    1`, `z = min.z - 3` — on the ground just outside the ring's north
    edge. Printed as `/tp @s x y z`.

## Done when

- `cargo run --bin model-exporter -- list` against the shipped
  `assets/models` (no slots) prints "0 models registered under
  assets/models" in `text` and `[]` in `json`.
- With a hand-written `assets/models/test.ron` in a temp `--models-dir`,
  `list` and `show test` print its box and `/tp` line correctly (unit
  tests over the `Render` output for a fixed slot, the way
  `ranvil_cli::save`'s tests check `SavesResult`).
- A malformed registry (e.g. a slot outside `area`) exits 2 with the file
  named, in both `text` and `json`.
- `ranvil-cli`'s existing tests pass after the `resolve_save` split.
- `cargo check` and `cargo test --lib` pass.
