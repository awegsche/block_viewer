# 134 - model-exporter: marker blocks around a slot, `mark <name>`

Design: `MODEL_EXPORTER_ROADMAP.md` ("Coordinates and markers"). Depends
on 133. The first ticket that writes to the models world, and the one that
answers the user's "place delimiting blocks just outside its boundaries to
signal to the human where to build". Every write goes through
`ranvil_cli::edit::run_write` — this ticket adds geometry, not a write
path.

## Scope

- **`src/model_exporter/markers.rs`**:
  - `marker_edit(slot: &ModelSlot, world: &ModelWorld) -> WorldEdit`,
    the roadmap's geometry exactly:
    - **Ring** at `y = world.ground_y`: every `(x, z)` with
      `x ∈ [min.x-1, max.x+1]`, `z ∈ [min.z-1, max.z+1]`, excluding
      `x ∈ [min.x, max.x] ∧ z ∈ [min.z, max.z]`. Set to `world.marker`.
    - **Corner pillars** at the four ring corners
      (`(min.x-1|max.x+1, min.z-1|max.z+1)`), `y ∈ [ground_y+1, max.y]`.
      Skipped entirely when `max.y ≤ ground_y` (a below-ground-only slot).
    - Nothing at any position inside the slot's own box — asserted in a
      test, since a marker inside the box would end up in every export.
  - `marker_positions(slot, world) -> Vec<IVec3>` behind it (so 137's
    `remove --clear` can put ground/air back at exactly these positions),
    with `marker_edit` built from it.
  - `ring_y`/`corner` helpers that 132's `show` now uses instead of its
    own arithmetic, so `show` and the world agree by construction.
- **`new` places markers** (133's `--no-markers` flag now does something):
  after `save_slot`, `run_write(save, dry_run, force, |_| Ok(marker_edit(..)))`.
  Gains `--force` (write even though the save looks open — `run_write`'s
  existing gate; the message tells the human to save & quit first). If the
  write fails after the `.ron` is written, the `.ron` stays and the error
  says to run `mark <name>` once the write can succeed — better than a
  slot that silently exists in the world but not in the registry.
- **`mark <name> [--dry-run] [--force]`** in `markers.rs`: re-places the
  markers for an already-registered slot. For a slot whose `.ron` was
  edited by hand (a bigger `size`), after 137's `--clear`, or a world reset.
  Output: the write's `outcome_summary` (regions/blocks written, backup
  dir) — reuse `ranvil_cli::edit::{outcome_summary, outcome_json_fields}`,
  making them `pub` if `pub(super)` is too narrow.

## Done when

- Tests: for a fixed slot, `marker_positions` has exactly
  `2·(sx+2) + 2·sz` ring positions at `ground_y` plus
  `4·(max.y − ground_y)` pillar positions, none inside the box, and the
  four pillar tops are at `max.y`; a slot with `below = 0` and
  `size.y = 1` has a ring and no pillars.
- `mark` against the single-chunk fixture save `ranvil_cli::edit`'s tests
  build (`Fixture`) writes the expected block count and `--dry-run`
  leaves the region file byte-identical.
- `todo.md` gains the manual check: run `model-exporter new ring_test 5 4
  5` against the real models world, `/tp` there in Minecraft, confirm an
  orange ring flush with the grass one block outside a 5×5, four pillars
  three high on its corners, and nothing inside; then `export` (135) it
  later to confirm the ring isn't in the file.
- `cargo check` and `cargo test --lib` pass.
