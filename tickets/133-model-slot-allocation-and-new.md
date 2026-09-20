# 133 - model-exporter: slot allocator and `new`

Design: `MODEL_EXPORTER_ROADMAP.md` ("Coordinates and markers", "Only
generated chunks"). Depends on 131, 132. The allocator is the one piece of
genuinely new logic in this tool, so it's a pure function with its own
tests; `new` is the thin command around it. Markers are **not** placed by
this ticket — that's 134, the first world write — so `new` here registers
a slot and tells you where it is.

## Scope

- **`src/model_exporter/allocate.rs`**:
  - `allocate(world: &ModelWorld, existing: &[ModelSlot], size: IVec3,
    below: u32, chunk_generated: impl Fn(IVec2) -> bool) ->
    Result<IVec3, AllocateError>` — returns the `origin` of the first
    candidate that fits, or why none did.
  - Candidates: `x` from `area.min.x` stepping by `world.grid`, `z`
    likewise, **z outer / x inner** so slots fill a row along +X before
    starting the next row (a human flying along the row sees them in
    creation order). A candidate `origin = (x, ground_y - below, z)` fits
    when (a) its footprint lies inside `area`, (b) it doesn't conflict
    with any `existing` slot under 131's overlap rule (expand by `gap`,
    intersect footprint), and (c) every chunk column its *marker
    footprint* (footprint + 1 on each side, since 134's ring sits there)
    touches satisfies `chunk_generated`.
  - `AllocateError::AreaFull` when no candidate passes (a) and (b);
    `AllocateError::NoGeneratedSpot { candidates_skipped }` when some
    passed (a)/(b) but all failed (c) — the message tells the user to fly
    around the area in Minecraft or pre-generate it
    (`docs/world-pregeneration.md`). Also `SizeTooLarge` for an axis over
    `STRUCTURE_BLOCK_MAX_SIZE` or `< 1`, checked before anything else.
  - The predicate is a parameter so the allocator's tests need no save:
    they pass `|_| true`, or a closure over a set, and assert on the
    chosen origin.
- **`new <name> <width> <height> <depth> [--below N] [--no-markers]
  [--dry-run]`** in `src/model_exporter/new.rs`:
  - `width`/`height`/`depth` are the `x`/`y`/`z` extents (documented in
    the arg help with that exact mapping). `--below` defaults to `1`: one
    foundation layer under the ground surface, so the blueprint's `y=0` is
    that layer and the surface is `y=1` — the `ground_level: 1` convention
    `gatherer_hut.ron`/`mine01.ron` already use. `--below 0` gives
    `house01.ron`'s "y=0 is the surface" shape instead.
  - Name must be a valid file stem (`[a-z0-9_]+`, so it also works as the
    `.nbt` stem `blueprint::catalogue` matches definitions against) and not
    already registered → otherwise `CliError::Usage`.
  - Resolves the save (132's `resolve_save_from`) only to answer
    `chunk_generated`: reads each region's chunk `Status` via the same
    per-chunk decode `ranvil_cli::chunk::chunks` uses (a chunk is
    "generated" iff present and `Status == minecraft:full`). Read-only;
    works with Minecraft open.
  - Calls `allocate`, then `registry::save_slot` (unless `--dry-run`),
    then — when 134 has landed and `--no-markers` isn't given — places the
    markers. Until 134 exists, `new` behaves as if `--no-markers` were
    always set and says so in its output.
  - Output (`text`): the roadmap's block — `name`, origin, size, inclusive
    box, the build-inside-the-ring line, and the `/tp` from 132's
    `tp_position`. `json`: `name`, `origin`, `size`, `min`, `max`, `tp`,
    `file` (the written `.ron`), `markers: "placed" | "skipped" |
    "dry-run"`.

## Done when

- Allocator tests: an empty registry places the first slot at
  `(area.min.x, ground_y - below, area.min.z)`; a second slot of the same
  size lands at the next `grid` multiple along +X with at least `gap`
  clear blocks between the two footprints; a slot wider than the row's
  remaining width wraps to the next row; a slot removed from the middle of
  a row is refilled by the next allocation that fits (first-fit reuses
  holes); a footprint that would straddle `area.max` is rejected;
  `chunk_generated` returning `false` for the first row pushes the
  allocation to the second row and `false` everywhere yields
  `NoGeneratedSpot`; an over-size axis yields `SizeTooLarge`.
- `new` against a temp `--models-dir` with `--no-markers` writes a `.ron`
  that `load_registry` accepts and `list` then shows; running the same
  `new` again exits 2 (already registered); `--dry-run` writes nothing.
- `cargo check` and `cargo test --lib` pass.
