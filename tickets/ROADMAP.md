# Roadmap — from "loads a save" to "explore a save"

Not a work item; an index of the numbered tickets in this directory and how
they fit together. Numbered tickets are the actual unit of work and move to
`../finished_tickets/` when done (see `CLAUDE.md`).

## Where we are

`src/main.rs` loads the first save it finds and eagerly parses the chunks of
its *first region*, stores it in a `LoadedSave` resource — and then ignores
it. `setup()` renders a hardcoded 8×8×8 placeholder built by
`create_block_mesh()`. Nothing in `src/` calls `get_block`. `bevy_egui` is a
declared dependency with zero uses. `src/pan_orbit_camera_bundle.rs` is never
declared as a module, so it isn't even compiled.

`ranvil`'s `ChunkRegion::get_block` does now return real palette entries
(tickets 001/008 upstream), so the data is finally reachable — with one
correctness bug still in the way (ticket 001 below).

## The shape of the work

Four layers, bottom-up. Each ticket is scoped so the repo builds and runs
after it lands.

```
001  bit-width bug (upstream, in ../ranvil)      <- blocks correct output
002  decode layer: NBT -> BlockState grids       <- blocks everything below
003  mesher: BlockState grids -> Bevy meshes
004  textures: block name -> atlas UV
      |
005  streaming: load/mesh chunks around camera
006  camera: navigation fit for exploring
      |
007  egui: save picker, coord jump, block readout
008  no-panic startup + error surfacing
      |
009  performance pass (only once 003/005 are real)
010  visual fidelity: biome tint, block models  (optional / later)
```

## Dependency order

- **001** is independent and small — do it first, it invalidates 002's tests
  otherwise.
- **002 → 003 → 004** is the critical path to "I can see my world".
  A minimal vertical slice (002+003 with one flat texture) is already a
  visible, testable milestone; don't gold-plate 004 before that lands.
- **005** and **006** are what turn a static render into *exploring*.
- **007** and **008** are the user-facing shell; 008 is small and can be
  pulled forward any time it gets annoying.
- **009** is deliberately last: measure the real mesher, don't pre-optimise.
- **010** is optional polish, listed so it isn't confused with core work.

## Suggested milestones

1. **M1 — "it's my world"**: 001, 002, 003. One region, one texture, no UI.
2. **M2 — "it looks right"**: 004.
3. **M3 — "I can explore it"**: 005, 006, 008.
4. **M4 — "it's a tool"**: 007, 009.
