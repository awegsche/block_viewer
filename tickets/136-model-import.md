# 136 - model-exporter: `import <name> [--from <file.nbt>]`

Design: `MODEL_EXPORTER_ROADMAP.md` ("Bridging both pipelines"). Depends
on 133, 134, 135. The reverse bridge: an agent-built `.nbt` (tickets 104,
118 style) or an older export goes *into* the models world so a human can
walk through it and fix it in game, then `export` brings it back. Without
this, the "agent first" pipeline still dead-ends in a file nobody can
enter.

## Scope

- **`import <name> [--from <file.nbt>] [--below N] [--dry-run] [--force]`**
  in `src/model_exporter/import.rs`.
  - `--from` omitted → the slot's own `out_path()` (re-import the last
    export, e.g. after a bad in-game edit). Missing file → `CliError::Usage`.
  - **Registered name**: the file's `size` must equal the slot's `size`
    (otherwise `Usage`, with the hint to edit the `.ron` and `mark` — the
    box is the contract, and silently importing a bigger file would spill
    over the ring). Then the write.
  - **Unregistered name** (needs `--from`): allocate a slot sized from the
    file (133's `allocate`, with `--below` for the origin, default `1`),
    `save_slot`, place markers (134's `marker_edit`) and the blocks in
    **one** `run_write` transaction, so a refused write leaves neither a
    marked slot without blocks nor a `.ron` without markers. On a refused
    write the `.ron` is removed again (it was written this call).
  - The block write is `struct import`'s exactly: a `WorldEdit` of every
    position including air (clears whatever was in the box — a previous
    import, or a human's abandoned attempt), offset by `slot.origin`.
    Factor `ranvil_cli::structure::import`'s blueprint-to-`WorldEdit`
    builder into a `pub fn blueprint_edit(&Blueprint, at: IVec3) ->
    WorldEdit` and call it from both.
  - Output: 133's `new` block (origin, box, `/tp`) when a slot was
    allocated, then the write's `outcome_summary`. `json`: the slot fields
    plus `outcome_json_fields`.

## Done when

- Against the fixture save: `import` of a `struct new --fill dirt` file
  into a registered slot writes `size.x·size.y·size.z` blocks at the
  slot's box and nothing outside it (`get-area` of the box + 1 margin,
  compared); `export` of that slot afterwards is `unchanged` against the
  same file (import → export round-trips).
- `import newname --from file.nbt` with no `.ron` allocates one (visible
  to `list`), places markers and blocks; a subsequent identical call is
  refused only by the size check if the file changed size, else
  overwrites in place.
- `--dry-run` writes neither the `.ron` nor the region.
- `todo.md` gains the manual check: `import` `gatherer_hut.nbt` (an
  agent-built model) into the models world, `/tp` there, confirm the hut
  stands inside its ring with its dirt layer flush with the grass.
- `cargo check` and `cargo test --lib` pass.
