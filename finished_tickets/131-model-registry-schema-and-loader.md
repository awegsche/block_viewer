# 131 - model-exporter: registry schema and loader

Design: `MODEL_EXPORTER_ROADMAP.md` ("The registry"). First ticket of the
`model-exporter` tool; pure data — no CLI, no save I/O. Everything later
(132–137) reads the registry this ticket defines, so the schema and its
validation are the thing to get right here.

## Scope

- **`src/model_exporter/mod.rs`** + `pub mod model_exporter;` in `lib.rs`
  — just the module root for now (`run()` comes in 132).
- **`src/model_exporter/registry.rs`** — the RON schema and loader:
  - `ModelWorld` (from `assets/models/world.ron`): `save: String`,
    `ground_y: i32`, `area: Area { min: (x, z), max: (x, z) }`,
    `gap: u32`, `grid: u32`, `marker: BlockState`, `blueprints_dir: PathBuf`.
    `marker` deserializes through `BlockState`'s existing `FromStr`
    (ticket 035's parser) via `serde`'s `deserialize_with` or a
    `String`-then-parse step — no second block-state grammar.
  - `ModelSlot` (from `assets/models/<name>.ron`): `name: String`,
    `origin: IVec3`, `size: IVec3`, `out: Option<PathBuf>`. Helpers:
    `min()`/`max()` (inclusive box, `max = origin + size - 1`),
    `bounds() -> SelectionBounds` (via `SelectionBounds::from_corners`, so
    export/import/markers all share 019's inclusive convention),
    `out_path(&ModelWorld) -> PathBuf` (`out` or
    `<blueprints_dir>/<name>.nbt`), `footprint()` (the XZ rectangle).
  - `Registry { dir: PathBuf, world: ModelWorld, slots: Vec<ModelSlot> }`
    and `load_registry(dir: &Path) -> Result<Registry, RegistryError>`.
    Reads `world.ron` (missing → `RegistryError::NoWorld`), then every
    other `*.ron` in the directory, sorted by filename so output order is
    stable.
  - `RegistryError` (with `Display`): `NoWorld`, `Parse { file, error }`,
    `NameMismatch { file, name }` (stem ≠ `name`), `DuplicateName`,
    `SizeOutOfRange { name, size }` (any axis `< 1` or
    `> STRUCTURE_BLOCK_MAX_SIZE`), `OutsideArea { name }`,
    `TooClose { a, b }` (two footprints closer than `world.gap`), plus a
    `BadWorld` for `gap`/`grid` of 0 or an `area` with `min > max`.
  - `save_slot(dir, &ModelSlot) -> io::Result<()>` — writes
    `<name>.ron` with `ron::ser::to_string_pretty`, a header comment
    naming the tool that wrote it. 133's `new` and 136's `import` call
    this; keeping the writer next to the reader is what guarantees a
    written file round-trips.
- **`assets/models/world.ron`** — shipped with the roadmap's superflat
  defaults (`save: "models"`, `ground_y: -61`, area `0..511` on both
  axes, `gap: 3`, `grid: 8`, orange terracotta, `blueprints_dir:
  "assets/city/blueprints"`) and a comment telling the user to set `save`
  and `ground_y` once the empty world exists. No slot files yet.

The overlap rule, spelled out once here and reused by 133's allocator:
two slots `a`, `b` conflict iff `a.footprint()` expanded by `gap` on every
side intersects `b.footprint()`. (Symmetric, and the ring — 1 outside each
footprint — of one slot can't touch the other's ring when `gap ≥ 3`.)

## Done when

- Unit tests (in `registry.rs`, against a temp directory the way
  `city::definition`'s loader tests do): a valid two-slot registry loads
  with slots in filename order; each `RegistryError` variant is produced
  by a fixture that should produce it; `save_slot` followed by
  `load_registry` round-trips `origin`/`size`/`out` exactly, including
  `out: None`.
- `load_registry` on the shipped `assets/models` (world file only, no
  slots) succeeds with zero slots.
- `cargo check` and `cargo test --lib` pass.
