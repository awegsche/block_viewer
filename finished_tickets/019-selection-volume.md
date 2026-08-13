# 019 - Selection volume: state, coordinate rules, and a gizmo box

## Status
Done (see Resolution)

## Depends on
Nothing. This is the substrate for 020–024 — do it first.

## Goal

One resource describing "the 3D region the user has selected", in Minecraft
block coordinates, plus a wireframe box drawn around it. No input handling
(020), no UI (021), no extraction (022). Just the data model, the coordinate
rules everything downstream depends on, and something visible on screen.

This ticket exists separately because *four* later tickets read these
bounds, and they must all agree on inclusive-vs-exclusive and on where the
box's faces sit relative to a block's cube — the two things that silently
produce off-by-one blueprints if each ticket decides for itself.

## New module

`src/selection/mod.rs` (`SelectionPlugin`, registered in `main.rs` alongside
the other plugins). 020 and 021 add submodules under it.

## The data model

```rust
/// The currently selected 3D region, in **Minecraft** block coordinates,
/// with **inclusive** bounds — `min == max` is a legal 1x1x1 selection of a
/// single block, not an empty one.
#[derive(Resource, Default)]
pub struct Selection(pub Option<SelectionBounds>);

pub struct SelectionBounds {
    /// The block first clicked (020). Kept separately from `min`/`max`
    /// because it survives the box growing away from it, and 021 shows it.
    pub anchor: IVec3,
    pub min: IVec3,
    pub max: IVec3,
}
```

`SelectionBounds` must expose, with tests:

- `from_anchor(block: IVec3) -> Self` — the 1x1x1 selection.
- `size(&self) -> IVec3` — `max - min + 1`, always ≥ 1 on each axis.
- `volume(&self) -> u64` — as `u64`, not `usize`/`i32`: a careless drag can
  reach billions of blocks and this number is what 021 and 022 guard on.
- `contains(&self, block: IVec3) -> bool`.
- `normalized(min, max)` — sorts each axis so `min <= max` componentwise.
  Every constructor goes through it; nothing downstream should ever have to
  wonder whether `min` is really the minimum.
- `iter_blocks()` (or an explicit `for` order documented in one place) —
  **Y outer, Z middle, X inner**, matching the section index order in
  `world::decode::ChunkSection::index` and the vanilla structure format 023
  writes. Getting this consistent here means 022 can stream blocks out in
  the order the file wants without buffering the whole box twice.

### World height clamp

Y is clamped to the world's build limits (1.18+: `-64..=319`). Put the two
constants in this module with a comment that they're the modern-world
values and that a pre-1.18 save would need them read from `level.dat`
instead — `mc_anvil`'s `ZERO_OFFSET` already hardcodes the same assumption
on the read side, so this is consistent with, not worse than, what the app
already does. X and Z are not clamped (the world is effectively unbounded
there); 021/022 guard on volume instead.

## Coordinate rules — read `src/world/mesh.rs`'s module docs first

Two facts from there that this ticket has to encode exactly once, in a
single conversion function every later ticket calls:

1. **Axis mapping.** `bevy.x = mc.x`, `bevy.y = mc.y`, `bevy.z = -mc.z`.
2. **A block occupies a unit cube from its coordinate**, i.e. MC block
   `(x, y, z)` covers `x..x+1`, `y..y+1`, `z..z+1` — it is *not* centred on
   the integer coordinate.

So the Bevy-space AABB of an inclusive MC-space selection is:

```text
bevy_min = ( min.x,        min.y,       -(max.z + 1) )
bevy_max = ( max.x + 1,    max.y + 1,   -(min.z)     )
```

Note the Z negation swaps which end is the minimum. Write the test for this
before the drawing code — a box that's one block short on the north face or
flipped in Z looks almost right on screen and produces a wrong blueprint in
022.

## Drawing it

`bevy::gizmos` — `Gizmos::cuboid(Transform { translation: centre, scale:
size, .. }, colour)` in an `Update` system. Reasons over the alternatives:

- **Why not a spawned wireframe mesh entity?** It would need a material, a
  despawn path on every bounds change, and it would land in the same
  `Assets<Mesh>` churn the streaming pipeline already manages. Gizmos are
  immediate-mode: no entity lifecycle to keep in sync with the resource.
- **Why not `bevy::pbr::wireframe`?** That renders *existing* meshes as
  wireframes; there's no mesh here to wireframe.

Details to get right:

- **Visible through terrain.** A selection box you can't see because it's
  inside a hill is useless. `GizmoConfig::depth_bias` (negative values bias
  toward the camera; `-1.0` draws on top of everything) is the knob. Start
  with the box drawn on top and note it in `todo.md` — whether it should be
  fully on top or only faintly visible through terrain is a look call a
  human has to make at the window.
- **Render layers.** The sky camera (ticket 016) is a second `Camera3d` on
  `RenderLayers::layer(1)`. Gizmos default to layer 0, so the default should
  be right — but confirm the box isn't drawn by the sky pass (it would be
  occluded by, or fight with, the dome).
- **Highlight the anchor** — a second, small gizmo cube on the anchor block
  so 020's "which block did I click" is legible. Cheap and it makes the
  keyboard extrusion much easier to reason about.
- Draw nothing at all when `Selection.0` is `None`.

## Out of scope

- Any input (020), any panel (021), reading blocks (022).
- Multiple/named selections. One selection, one resource. If that's ever
  wanted, `Selection` becomes a `Vec` and everything downstream keeps
  working against `SelectionBounds`.
- Face highlighting for "which face will the next keystroke move". Nice to
  have; 020 can add it if the extrusion turns out to be hard to follow.

## Tests

- `normalized` sorts each axis independently (pass a min/max pair that is
  inverted on Z only).
- `from_anchor` gives `size == (1,1,1)` and `volume == 1`.
- `size`/`volume` on a known box, including one crossing `y == 0` and one
  crossing `z == 0` (the negated axis).
- The MC→Bevy AABB conversion for a known box, asserting the Z ends swap and
  the `+1` lands on the max face of each axis.
- `iter_blocks` yields exactly `volume` blocks, all `contains`-true, in the
  documented Y/Z/X order.
- Y clamping at both build limits.

## Done when

- A `Selection` set from code (a hardcoded box in the plugin's startup, or a
  test) draws a correctly-placed wireframe box around exactly those blocks.
- `cargo test` passes.
- `todo.md` gets a manual check: set a selection covering a landmark you can
  identify (e.g. a 3x3 patch of a flat roof), confirm the box's faces line
  up with the block edges rather than sitting half a block off or one block
  short, and confirm it reads correctly on the Z axis specifically (looking
  north vs. south) — Z is the axis a sign error hides in. Also decide the
  depth-bias look.

## Resolution

`src/selection/mod.rs` (data model, coordinate rules) and
`src/selection/gizmo.rs` (drawing), wired in as `selection::SelectionPlugin`
from `main.rs`. 14 tests, all of the ones listed above plus `index_of`
against `iter_blocks` and a saturation check on `volume`.

Implemented as specified, with these decisions worth recording:

- **`block_bevy_aabb` is the only place the coordinate rules live.**
  `SelectionBounds::bevy_aabb` unions the two corner blocks' unit cubes
  rather than open-coding `-(max.z + 1)`, so the `+1` on each max face and
  the Z flip are each written exactly once. The test asserts the resulting
  extent equals `size().as_vec3()`, which is the same rule stated
  independently of the arithmetic that produced it.
- **`index_of` was added** beyond the ticket's list. `iter_blocks`'s Y/Z/X
  order is a contract 022 and 023 both depend on, and a documented order
  with no inverse is a contract only prose enforces; the two are tested
  against each other.
- **Own gizmo config group** (`SelectionGizmos`) rather than configuring
  `DefaultGizmoConfigGroup`, so `depth_bias: -1.0` is scoped to this feature
  instead of applying to any gizmo added later.
- **Render layers left at the default.** Confirmed against
  `bevy_gizmos-0.15.0`: `GizmoConfig::default()` is `RenderLayers::layer(0)`
  and gizmos draw to every camera whose layers intersect, so the sky camera
  (016, layer 1) can't pick the box up.
- **Dead-code allow at module scope.** Most of the API's callers are 020,
  021 and 022; trimming to what `gizmo` happens to use today would defeat
  the point of settling the coordinate rules first. Commented, and flagged
  to come off when 022 lands.

### Deviation: no hardcoded startup selection

The "done when" offered a hardcoded box in the plugin's startup as a way to
see the box before 020 exists. Not done, deliberately: the alignment check
this ticket actually cares about — do the box's faces sit on the block
edges, and is the Z sign right — needs the box placed on terrain the human
can recognise, and nothing here can do that. A box at fixed coordinates in
an arbitrary save lands in mid-air or inside a hill, where "one block off"
is invisible. So the visual check rides along with 020's, which is where a
click puts the box on a block you chose; `todo.md` says so. The coordinate
maths itself is covered by tests in the meantime.
