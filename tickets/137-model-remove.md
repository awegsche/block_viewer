# 137 - model-exporter: `remove <name> [--clear]` (optional)

Design: `MODEL_EXPORTER_ROADMAP.md` ("Ticket breakdown", marked optional).
Depends on 134. Deleting a `.ron` by hand already unregisters a slot and
frees its spot for the allocator; this ticket exists so the world can be
tidied too, once enough slots have come and gone that stray rings matter.

## Scope

- **`remove <name> [--clear] [--dry-run] [--force]`** in
  `src/model_exporter/remove.rs`:
  - Without `--clear`: delete `<name>.ron` only. Markers and blocks stay
    (the `.nbt` in `blueprints_dir` is never touched by this command —
    that file belongs to the citybuilder catalogue).
  - With `--clear`: one `run_write` that puts the ring back to the flat
    world's ground (`world.ground_block`, a new optional `ModelWorld`
    field defaulting to `minecraft:grass_block`; 131's loader gets the
    field with `#[serde(default)]`), the pillars back to air, and every
    position inside the box to ground below `ground_y + 1` (the layers
    `--below` covered, back to `world.below_block`, default
    `minecraft:dirt`) and air from `ground_y + 1` up. Uses 134's
    `marker_positions` for the marker part so the two can't drift.
  - The `.ron` is deleted only after the write succeeds (or immediately
    without `--clear`), so a refused write leaves a registry that still
    matches the world.
- Registry order and 133's first-fit mean the freed spot is reused by the
  next `new` that fits — already tested there; this ticket adds the
  end-to-end: `new a`, `new b`, `remove a`, `new c` (same size as `a`)
  lands where `a` was.

## Done when

- Against the fixture save: `new` (markers) then `remove --clear` leaves
  the region's blocks in the ring/pillar/box positions equal to
  ground/air as specified (compared through `get-area`), and the `.ron`
  gone; `--dry-run` leaves both.
- `remove` of an unknown name exits 2.
- `cargo check` and `cargo test --lib` pass.
