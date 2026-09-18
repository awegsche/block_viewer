# 118 - Mine: a real model (`mine.nbt`)

Design: `MINES_DESIGN.md` ("What a mine is"). Depends on 113 (the three
`.ron` files point at placeholder geometry) and, to be worth looking at,
116. Built headlessly with `ranvil-cli struct new/fill/set`, the way 104
built `gatherer_hut.nbt` — no Minecraft session, no viewer.

## What the shaft needs from the blueprint

The generator (114/115) owns everything from `floor_y` down inside the
lining square: it opens the interior as the well, turns the ring at
`floor_y` into the top landing and first flight, and leaves the lining at
`floor_y` alone. So the blueprint's job over the shaft is only to **put a
headframe around it** — walls and a roof over the `(S+2)²` lining square
plus a margin — and to lay an ordinary floor across it that the generator
will cut. Anything the blueprint puts *inside* the ring at `floor_y`
(rails, a winch) is removed by the first sink job; put set-dressing on the
lining ring or outside it.

## Layout

Footprint **16×16** (`Explicit`, the same the placeholders declare, so the
three `.ron` files change only `blueprint`/`ground_level` and lose their
placeholder comment). Size `16 × 12 × 16`. `shaft: (x: 5, z: 5)`,
`shaft_size: 6` → ring `5..=10`, lining `4..=11`.

- **y=0** — dirt foundation, full footprint.
- **y=1** — ground surface (`ground_level: 1`): gravel yard, cobblestone
  floor under the headframe (`x, z ∈ 3..=12`), a dirt-path trail from the
  yard edge to the headframe door and to the storage shed.
- **Headframe** (`3..=12` on both axes, y=2..=7): spruce-log corner posts
  and a stripped-spruce frame, spruce-plank infill with gaps (it's a
  timber headframe, not a house), a wide double door on the north side, a
  pitched spruce-stair roof with a raised lantern turret over the shaft
  (a 4×4 of open framing at y=7..=9 with lanterns) so the well reads from
  above.
- **Storage shed** (`x 0..=3`, `z 9..=15`, y=2..=5): cobblestone base,
  spruce planks, a slab roof; inside, barrels and a double chest — the
  set-dressing for the 4096-block buffer, the role 104's chest plays.
- **Decoration**: two rail stubs with a minecart (`rail` on the yard,
  a `chest_minecart` entity is out of scope — a static `minecart` isn't a
  block; use a `hopper`+`rail` stub instead), an ore-pile corner
  (`raw_iron_block`, `raw_copper_block`, `coal_block` in a fenced bay), a
  log pile, a water trough (`cauldron[level=3]`), lanterns on fence posts
  along the trail.
- **Keep the lining ring at `floor_y` solid** (the cobblestone floor) so a
  lining corner's pillar meets stone and a wall torch has a wall.

Tiers share the model (the `warehouse01/02` way). If a visible tier
difference is wanted later, a `mine_deep.nbt` with a taller frame is a
one-line `.ron` change.

## Plan

1. Build `assets/city/blueprints/mine.nbt` with `ranvil-cli struct new` /
   `fill` / `set`.
2. `ranvil-cli struct validate` + `struct info`; then the four-rotation
   `struct rotate` check (spruce-stair roofs and doors carry `facing`).
3. `mine01/02/03.ron`: `blueprint: "mine.nbt"`, `ground_level: 1`, drop the
   placeholder comment; confirm 113's `ShaftOutsideFootprint` still passes
   (lining `4..=11` inside `0..=15`).
4. `todo.md`: replace the placeholder-geometry line with an eyeballing
   to-do (in the citybuilder, and in Minecraft: the well opens through the
   cobblestone floor with the first flight visible from the door).
5. `cargo test --lib`.
