# 085 - Building ground level: which blueprint layer is "the ground"

## Status
Done — `cargo test --lib` green (809 passed). Landed as designed below.

## Why

`grid::fit_footprint`'s `base_y` is one above the topmost sampled terrain
block, and every placement so far has put the blueprint's own `y=0` layer
right there. That's correct for a blueprint whose bottom layer *is* its
visible ground surface (`house01.nbt`: grass/dirt at `y=0`) but wrong for one
that buries a foundation below it — `lumber.nbt` has two solid dirt layers
(`y=0`, `y=1`) before its grass/dirt-path surface at `y=2`. Placed today, two
layers of that hut's own foundation sit *above* the surrounding terrain
instead of being sunk into it.

## Schema

`definition::Building` gains:

```ron
ground_level: 2,  // 0-indexed; which Y layer of the blueprint is the
                   // surface fit_footprint should align to terrain. Default
                   // 0 — a blueprint with no below-grade layers is
                   // unaffected.
```

Validated at load (needs the matched catalogue entry's blueprint height, so
this lives in `load_entry` alongside the footprint resolution, not in the
file-only `validate`): `ground_level` must be a real index into the
blueprint's Y extent (`< size.y`).

## Wiring

`ground_level` shifts where a blueprint's `y=0` lands, not where the player's
cursor is — `placement::resolve_placement` subtracts it from whatever height
the terrain fit (or the refused-fit fallback) produced, before `y_offset`'s
manual nudge is added on top. Both the ghost preview and `commit` compute
through the same function, so what's drawn is what gets written, same as
every other placement invariant here.

The value comes from the selected **definition**, not the catalogue entry —
`BuildingDefinitions::ground_level(definition_id)` — so it inherits the same
gap `cost`/`requires`/`production` already have: a placement made through
`city::placement`'s keyboard stand-in (no `definition_id`) reads as
`ground_level: 0`, the old behaviour.

## Shipped values

Read off each blueprint's own layers (`minecraft:dirt` before the first
grass/path surface):

- `lumber.ron` -> `2` (two dirt layers under the hut's own floor).
- `lumber_farm_01.ron` -> `1` (one dirt layer under the field's grass).
- `house01.ron`, `farm01.ron`, `warehouse01.ron`, `warehouse02.ron` -> `0`
  (grass/dirt is the blueprint's own `y=0`; the latter three are placeholder
  geometry pointing at `house01.nbt` anyway).

## Not in scope

Terraforming or auto-leveling around the sunk layers — ticket 058 already
decided uneven ground isn't refused or auto-filled, and `ground_level` just
moves *which* layer that same lowest-point fit lines up with. Whatever the
fit still leaves poking into the foundation is cleared the same way
`blueprint_edit`'s "air is written, not skipped" already handles a shallow
footprint on rough ground.
