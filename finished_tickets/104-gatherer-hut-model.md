# 104 - Gatherer's Hut: a real model

## Status
Done — `cargo test --lib` green (974 passed; the lone `blueprint::export`
failure on the first run was a pre-existing parallel-test flake, confirmed
by re-running it alone and the full suite again, both clean).

## Why

`gatherer_hut.ron` (ticket 086) still borrows `lumber.nbt` as placeholder
geometry — noted as a to-do in both `finished_tickets/086-gatherer-hut.md`
and `todo.md`. `ranvil-cli`'s `struct` group (087-103) is now complete, so
this is the first building modelled entirely headlessly, without opening
Minecraft or the viewer: `struct new` + `struct fill`/`struct set` build
`gatherer_hut.nbt` block by block.

## Design

A small, single-room hut inside a fenced yard, footprint 9x9 (within the
requested 10x10 parcel limit), size 9x8x9 (`x,y,z`):

- **y=0** — dirt foundation, full footprint.
- **y=1** — ground surface (`ground_level: 1`, one dirt layer below):
  grass yard, oak-plank floor under the hut itself, a dirt-path trail from
  the yard gate to the hut door.
- **y=2..4** — hut walls: oak-log corner posts, oak-plank infill, a door on
  the north wall, glass windows on the east/west walls. A fenced yard ring
  (oak fence + one oak fence gate on the north side, torches flanking the
  gate) surrounds it with room to spare.
- Interior: a double chest (the storage the hut's `Production`-shaped buffer
  represents, same set-dressing role `lumber.nbt`'s chests already play —
  see `lumber.ron`'s own comment), a crafting table, a barrel, two wall
  torches.
- **y=5..7** — a simple gable roof (oak-stair tiers descending from an
  oak-log ridge, plank gable-end triangles on the west/east ends).
- Yard decoration: two small log-pile posts and two hay bales in the
  corners.

## Plan

1. Build `assets/city/blueprints/gatherer_hut.nbt` via `ranvil-cli struct
   new`/`fill`/`set` (no Minecraft/viewer involved).
2. `ranvil-cli struct validate` + `struct info` to confirm it's sane
   (non-air, in size bounds) before pointing the definition at it.
3. Update `gatherer_hut.ron`: drop the placeholder-geometry comment,
   `ground_level: 2` -> `1` (one dirt layer now, not two), footprint stays
   `FromBlueprint` (resolves to 9x9).
4. Update `todo.md`'s ticket-086 manual-verification item (it currently
   says the hut "meshes as lumber.nbt's borrowed geometry, same box the
   Lumberjack's Hut uses — not a bug", which is now stale) and add a fresh
   to-do for eyeballing the new model in-game (this task doesn't drive
   `cargo run` itself, per this repo's manual-verification rule).
5. `cargo test --lib` to confirm nothing regresses.
