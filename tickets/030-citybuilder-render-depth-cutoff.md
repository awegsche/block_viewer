# 030 - Citybuilder: stop meshing below the terrain surface

## Status
Open

## Depends on
Nothing that isn't landed. `ranvil` 013 (`Heightmaps` pack/unpack) is **done**
and provides the read side this keys off:
`ChunkRegion::heightmap(x, z, HeightmapKind::OceanFloor)`, or
`ranvil::heightmap::read_heightmap(&chunk_nbt, kind)` straight off a chunk root.

Related but not blocking: ticket 009 (viewer performance pass) lists vertical
splitting and greedy meshing for the *viewer*; this is a different lever aimed
at a different binary, and the two don't conflict. Roadmap task W7 (edits mark
chunks dirty) is what the "dig below the floor" half plugs into when it lands.

## Scope: the citybuilder only

`block_viewer` renders the whole world and keeps doing so. It is the
explore-a-save app: looking at a cave system, following a ravine down, flying
under an overhang and inspecting a block at Y=-59 are all things it exists for,
and a floor that quietly deletes the underground would break its purpose rather
than speed it up.

The citybuilder is the opposite case. Its camera looks at the surface from
above (roadmap E1's RTS rig), it never goes underground, and the terrain under
the city is scenery it can't see. So the floor is **off by default and switched
on by `city::run()`**, which is also what keeps this ticket honest: if the
shared code has to grow a `if citybuilder {}` anywhere, the design is wrong.

## Motivation

A 1.18+ chunk column is 24 sections. On the real save (`nbt_test`, probed for
`ranvil` 013) a surface chunk's sections run Y -4..=19 with the surface around
Y 63 — so roughly seven sections per column sit entirely below the terrain and
every one of them is decoded, meshed, uploaded and drawn today.

Where the saving actually comes from, stated honestly because it isn't where it
first looks:

- **Not from solid rock.** The mesher already culls a face whose neighbour is
  solid, so stone surrounded by stone emits nothing. Deep sections are not
  currently producing many *faces*.
- **From the decode.** Below-surface sections are the ones carrying real
  multi-entry palettes and full packed `block_states.data` arrays. Sections
  *above* the surface are mostly single-entry air, which `decode_chunk` already
  skips almost for free. Skipping the deep half is skipping the expensive half
  of the unpack, per chunk, forever.
- **From caves.** Caves, ravines and mineshafts are surfaces, they are
  underground, and they are invisible from an RTS camera. That is where the
  invisible face count actually lives, and it's the part that scales badly with
  render distance.

## Design

### 1. One floor per chunk, from `min(OCEAN_FLOOR)`

Not per column. A per-column cutoff makes neighbouring columns of different
depth expose vertical walls between them, which *adds* faces — the opposite of
the point. The floor is one value per chunk column, which is also the mesh unit
and the streaming unit.

`OCEAN_FLOOR` specifically, of the four maps: it's the only one that ignores
fluids, so under water it reads the sea bed rather than the surface. It is the
deepest of the four, which is what a floor wants.

Taking the **minimum over the chunk's 256 columns** is what makes this safe
against the case that would otherwise be a bug: a ravine, a cave mouth or a
cliff face inside the chunk drags the whole chunk's floor down with it, so the
cutoff never slices into a hole that's visible from above. A chunk with a
1-column ravine in it saves nothing, and that is the correct outcome.

```rust
// world::decode
fn render_floor(chunk_nbt: &NbtField, margin: i32) -> i32 {
    let Ok(heights) = ranvil::heightmap::read_heightmap(chunk_nbt, HeightmapKind::OceanFloor)
    else {
        return WORLD_MIN_Y; // no heightmaps: mesh everything (see 4.)
    };
    let deepest = heights.iter().copied().min().unwrap_or(WORLD_MIN_Y);
    snap_down_to_section(deepest - margin).max(WORLD_MIN_Y)
}
```

### 2. Snap the floor down to a section boundary

The decode unit is the 16-block section, so a floor at Y=37 buys exactly
nothing over Y=32. Round down to a multiple of `SECTION_SIZE` (careful with
negative Y — `div_euclid`, as `ranvil`'s `split_y` does).

This also makes the margin a coarse knob rather than a fine one. Start at **one
section (16)** and expect to tune it; its real job is not correctness (item 1
covers that) but keeping small terrain edits from crossing the floor and
triggering a re-decode (item 5).

### 3. The floor lives on `ChunkColumn`, and the mesher already has the concept

`world::mesh` already carries exactly the right idea for the world bottom:

```rust
// src/world/mesh.rs:533
// Down does need one: below WORLD_MIN_Y isn't air, it's
// "no world there" — never emit the underside.
face(Face::Down, world_y > WORLD_MIN_Y && ...);
```

So the change is: **the world bottom becomes a per-column value** rather than a
constant. Add `floor_y: i32` to `ChunkColumn`, defaulting to `WORLD_MIN_Y`
(which is what `block_viewer` gets, making its behaviour bit-identical), and:

- **Down faces**: `world_y > column.floor_y` instead of `> WORLD_MIN_Y`.
- **Horizontal faces**: in `block_at`, a lookup into a *neighbour* column below
  that neighbour's `floor_y` must read as **solid**, not as air. Otherwise two
  adjacent chunks with different floors grow a wall of side faces between them
  at exactly the boundary the whole scheme was trying not to draw.

Recording the floor on the column rather than passing it alongside is what makes
that second rule possible at all — the mesher gets neighbour columns, not
neighbour floors.

The general rule, worth writing into the module docs: **below the floor is
opaque, not air.** Treating it as air is the trap; it emits a downward quad for
all 256 columns of every chunk plus the seams above, which is a net loss.

### 4. Off by default, and self-healing when the data isn't there

- The policy is a resource — sketch: `RenderFloor { margin: i32 }` as an
  `Option`-shaped resource, or a `FloorPolicy { WholeWorld, BelowSurface { margin } }`
  passed into `decode_chunk`. `world_app()` inserts `WholeWorld`; `city::run()`
  overrides it. Whichever shape, **`block_viewer`'s path must not change**.
- A chunk with **no `Heightmaps`** falls back to `WholeWorld` for that chunk.
  This is not hypothetical: roadmap task W4's default is to *delete* the
  compound on an edited chunk and let Minecraft rebuild it, so every chunk the
  citybuilder writes to is temporarily in exactly that state. Falling back means
  such a chunk renders fully — slower, never wrong — and quietly starts being
  cheap again once the game has rebuilt its heightmaps.
- An empty/void column (`OCEAN_FLOOR` reads the world bottom everywhere) gives
  a floor of `WORLD_MIN_Y`, i.e. no cutoff. Nothing to decode there anyway.

### 5. Lowering the floor when the city digs

The dynamic half. When an edit puts a block below a chunk's current `floor_y`
(terraforming, a cellar, a foundation cut), that chunk needs its floor lowered.

The thing to get right: **this is a re-decode, not a re-mesh.** 005-f's re-mesh
path (`PendingChunkRemeshes` → `remesh_chunk_column`) rebuilds a mesh from an
already-decoded `ChunkColumn`, and the blocks below the floor were never decoded
— they aren't in the column to mesh. So lowering the floor means dropping the
column from `DecodedWorld` (and its entity from `SpawnedChunkEntities`) so the
streaming pipeline reloads it, or adding an explicit reload path next to the
re-mesh one.

Cheap and correct: `new_floor = min(current_floor, snap_down(edited_y - margin))`,
and only re-decode when it actually moved. With a 16-block margin, most edits
never move it.

This half can land after the first: the citybuilder can't dig until roadmap
group W lands anyway. Sequence the ticket so the static floor ships first and
is useful on its own.

## Explicitly not in this ticket

- **`block_viewer` behaviour of any kind.** If the viewer's meshes change by one
  vertex, this ticket did something wrong; there's a test for exactly that.
- **A camera-follows-the-floor rule.** Below the floor you can see through the
  terrain, because that's what "below is opaque, don't draw it" means. The
  citybuilder's RTS camera doesn't go there. If a later debug/free camera does,
  the fix is to lower that chunk's floor (item 5's machinery), not to draw the
  underside.
- **Structures below the floor.** A player's underground base, a deep mineshaft
  hall, the underside of a floating island: gone from the render, present in the
  save, still readable through `RegionCache`/`get_block`. Accepted for a
  citybuilder. Item 1 already covers everything that breaks the *surface*.
- **`DecodedWorld` as a source of truth.** With a floor on, it holds what's
  *rendered*, not what's in the world. Anything asking "what block is at (x,y,z)"
  authoritatively must go through the region cache — which is already what
  `blueprint::extract` does (022), and the reason it does.
- **Vertical mesh splitting, greedy meshing, LOD.** Ticket 009's list, viewer
  side, separate lever.

## Tests

Automated, no window needed:

- `render_floor` from a synthetic heightmap: flat terrain gives surface minus
  margin, snapped down; **one deep column drags the whole chunk's floor to it**;
  an all-empty map gives `WORLD_MIN_Y`; a missing `Heightmaps` gives
  `WORLD_MIN_Y`. The ravine case is the one that matters — write it first.
- Snapping is correct **below Y=0** (the off-by-one-section trap:
  `-70 → -80`, not `-64`).
- `decode_chunk` with a floor drops exactly the sections entirely below it and
  keeps the one straddling it.
- The mesher emits **no Down faces** at the floor, and a column meshed against a
  neighbour with a *higher* floor emits no side faces below that neighbour's
  floor.
- **The viewer is untouched**: the same fixture column decoded and meshed under
  `WholeWorld` produces the identical vertex/index counts it does today. This is
  the regression test that lets the shared modules take the change at all.
- A measurement, against the real save (there's precedent —
  `load_and_mesh_chunk_decodes_and_meshes_a_real_chunk` and 022's
  `extraction_agrees_with_the_block_inspectors_path_on_a_real_save` both do
  this): decode+mesh one region's worth of chunks both ways and print sections
  decoded, vertices emitted and elapsed time. **Record the numbers in the
  Resolution.** The whole ticket is a performance claim and nobody has measured
  it yet; if the saving is small, that's a finding worth having written down.

Manual (needs a human at the window — see `CLAUDE.md`): goes in `../todo.md`.

## Done when

- `cargo run --bin citybuilder` renders the surface with the underground gone,
  and no visible hole, seam or missing face anywhere the camera can see.
- `cargo run --bin block_viewer` is unchanged, verified by the identical-mesh
  test as well as by eye.
- The measured before/after is in the Resolution.

## Resolution
