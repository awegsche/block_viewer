# 026 - Selection box legibility: which part of it is underground

## Status
Done — pending the manual look call in `todo.md`

## Depends on
019 (the gizmo and the coordinate rules). Nothing depends on this.

## The problem, observed at the window

`gizmo.rs`'s `DEPTH_BIAS` is `-1.0` — the whole selection box always draws in
front of terrain. Ticket 019 flagged that as a look call needing a human;
the human looked, and the verdict is that **a box extending into the ground
is unreadable**. Every edge is drawn at full strength over the hillside, so
the wireframe reads as a flat outline pasted on the screen with no relation
to the surface: you can't tell which edges are buried, where the box enters
the ground, or how far down it goes.

Flipping the bias to depth-tested is not the fix. Then the buried part
disappears completely and the box you're building is invisible exactly when
you're extending it downward.

## The fix: draw it twice

What Blender, Blockbench and the WorldEdit CUI all do — one pass for the
part in open air, a visually distinct one for the part behind geometry:

- **Solid pass**, depth tested with a *small* negative bias (`-0.02`), full
  colour, current line width. Terrain occludes it. The bias only exists to
  stop z-fighting where the box's faces sit exactly on block faces, which
  they do constantly — every selection boundary is a block boundary. It is
  perspective-correct in Bevy's gizmo shader (`clip.z * (clip.w/clip.z)^-bias`),
  so it scales with distance rather than needing per-range tuning.
- **Buried pass**, `depth_bias: -1.0` as now, but **dotted**
  (`GizmoLineStyle::Dotted`) and dimmed. Dotted rather than only dimmed
  because the two passes overlap where the box is visible: both draw the same
  lines at the same pixels, and alpha alone would just make the visible parts
  slightly muddier without ever reading as a different state. Dotted-behind /
  solid-in-front is unambiguous at a glance.

The transition between the two is the thing the ticket exists for: it is
exactly the line where the box enters the ground.

## Depth cues inside the box

A 12-edge wireframe has no interior perspective, so a deep box gives the eye
nothing to judge depth against even once the buried part is distinguishable.
Add **horizontal slice outlines at chunk boundaries** — a rectangle across
the box at every multiple of `CHUNK_STEP` (16) strictly inside it. A tall
selection then reads as a stack of plates rather than an outline, and the
slices double as a scale readout.

- Chunk-aligned rather than box-relative: it lines up with the terrain's own
  16-block grid, and it matches what `Ctrl` + a direction already steps by.
- **Y only.** X/Z slices were considered and rejected: three axes of slices
  is a cage, and it's the *vertical* extent that's unreadable underground.
- **Budget the count.** Y is clamped to the build limits (384 blocks), so a
  full-height selection is 24 slices at a 16 step. Double the step until the
  count is within a cap rather than either drawing 24 rectangles or hiding
  the slices entirely on the boxes that most need them.
- Slices go through both passes like everything else, so the buried ones are
  dotted too. That's most of the depth cue: the plates that are dotted are
  the plates underground.

## Considered and not done

- **A translucent fill.** Depth-tested, terrain would occlude it and its
  silhouette would show where the volume enters the ground — the strongest
  cue of the lot, and the one most likely to be visually intolerable, since
  it tints everything you're trying to look at. It also can't be a gizmo
  (gizmos are lines only); it needs a mesh entity, an `AlphaMode::Blend`
  material and `Cull::None` so it still reads from inside. Worth trying only
  if the two-pass wireframe and the slices turn out not to be enough — judge
  that at the window, and if it's wanted it's its own ticket.
- **Per-block tinting of the selected blocks.** The mesher bakes vertex
  colours per chunk at build time (011/013), so tinting selected blocks means
  re-meshing every affected chunk column on every arrow keypress. The
  alternative is a custom material carrying the selection AABB as a uniform,
  which puts a custom material in front of 014's cutouts and 017's shadows.
  Not worth it for feedback the passes above give for a fraction of the cost.
- **Face highlighting** for "which face moves next" (019 deferred it to 020,
  020 deferred it to the window). Still open, still worth doing, but it's a
  different question — that one is about the input, this one is about the
  geometry.

## Tests

Rendering can't be asserted headlessly; the arithmetic can.

- The slice step coarsens past the cap: a full-height selection stays within
  the budget, a 100-block one still steps by 16.
- Slice heights are chunk-aligned and *strictly inside* the box — never on
  the top or bottom face, which the box's own edges already draw.
- A selection inside one 16-block layer has no slices at all.
- Slice heights at negative Y (the `div_euclid` path, since a selection can
  sit below Y 0).

## Done when

- `cargo test` passes.
- `todo.md` gets the manual check, since the whole ticket is a look call:
  build a box that runs from open air down into a hillside and confirm the
  buried part is dotted and the visible part solid, that the boundary between
  them tracks the terrain surface as the camera moves, and that the slice
  plates read as depth rather than as clutter. Judge `SOLID_DEPTH_BIAS`
  against a box coplanar with a flat surface (z-fighting means it's too
  small) and the dotted pass at distance (invisible means the alpha is too
  low). This supersedes item (3) of 019's to-do.

---

## Resolution

All of it in `src/selection/gizmo.rs`; nothing else in the app changed.

### What was built

- Two gizmo config groups instead of one. `SelectionGizmos` keeps the name
  and becomes the depth-tested pass (`SOLID_DEPTH_BIAS = -0.02`, solid,
  2.5px); `BuriedSelectionGizmos` is the always-in-front one (`-1.0`,
  `GizmoLineStyle::Dotted`, 2.0px, alpha 0.55).
- `draw_selection` is generic over the config group, so a pass is a
  `GizmoConfig` plus an alpha and the two can't drift apart in what they
  draw. The box, the anchor cube and the slice plates all go through both.
- `slice_heights` / `slice_step`: horizontal plates on world chunk
  boundaries strictly inside the box, the step doubling from `CHUNK_STEP`
  until at most `MAX_SLICES` (12) of them.

### Decisions worth knowing about

- **Both passes draw the same geometry**, rather than working out which
  edges are buried and drawing each once. That's what the depth buffer is
  for, and against streamed terrain there's no cheap answer anyway. Where
  the box is in open air the two overlap and the solid pass is what reads.
- **Dotted, not just dimmer.** Because of the overlap above, alpha alone
  would only make the *visible* parts slightly muddier — the two states
  have to differ in something that survives being drawn on top of each
  other. This is also why `BURIED_ALPHA` is 0.55 rather than the 0.3 the
  first sketch had: dotting has already halved the ink on those lines.
- **Slice plates are chunk-aligned in world space**, not spaced from the
  box's own bottom, so they stay put while a face is extended past them
  rather than sliding with every keypress.
- **Plates are strictly inside the box.** One coincident with the top or
  bottom face would only z-fight against the cuboid's own edge.
- **`slice_step` doubles rather than giving up past a threshold.** The tall
  boxes are exactly the ones whose depth is hardest to read, so they keep
  their plates and lose resolution: a full-height selection (384 blocks,
  Y being clamped to the build limits) steps by 32 and draws 11.

### Not done, deliberately

The translucent fill and per-block tinting are both written up under
"considered and not done" above. The fill is the next thing to try if the
manual check says the wireframe still isn't enough; per-block tinting would
mean re-meshing chunk columns on every keypress or putting a custom material
in front of 014's cutouts and 017's shadows, which is out of proportion to
the problem.

Every constant here is a guess by someone who cannot look at the window.
`todo.md` has the check, and it names which knob to turn for each way it can
be wrong.

7 tests in the module (5 new); `cargo test` is 181 passing.
