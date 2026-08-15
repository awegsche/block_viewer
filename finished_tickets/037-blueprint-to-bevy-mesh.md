# 037 - `Blueprint` -> Bevy `Mesh`

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 271
tests). See the Resolution.

## Part of
Roadmap B2 (`tickets/CITYBUILDER_ROADMAP.md`) — the second ticket in group B
(blueprints as building models).

## Depends on
- 022's `Blueprint`/`BlockState` (`blueprint::extract`)
- 036's structure reader (`blueprint::structure`) — an alternate source of a
  `Blueprint` to mesh, though this ticket doesn't need a caller to prove
  itself against either source
- `world::mesh`'s quad emission (003/010/013/014), `world::atlas`'s face
  resolution (004), `world::tint`'s block tint resolution (013/014)

## Problem

`world::mesh::mesh_chunk_column` is the only thing in this codebase that
turns blocks into a `Mesh`, and it's built entirely around
`world::block::BlockId`/`BlockRegistry` — a save-wide interning table. A
`Blueprint`'s palette is `BlockState`s (name + properties) with no registry
behind it, so there's no `BlockId` to index a UV/tint table with. Nothing
today can preview or place a loaded blueprint; B3 (rotation) and B4 (the
asset catalogue) both need this first.

## Goal

`blueprint::mesh_blueprint(&Blueprint, &AtlasUvIndex, BiomeColors) ->
Option<Mesh>`: the same per-face quad emission `mesh_chunk_column` uses,
fed by a palette-entry lookup instead of a `BlockId` table.

## Scope

- Widened three things in `world` from private to `pub(crate)` rather than
  duplicating them: `world::mesh`'s `Face` enum and quad-emission functions
  (`face_geometry`, `push_quad`, `push_quad_offset`, `OVERLAY_EPSILON`,
  `WHITE`, `resolve_tint_color`), `world::atlas::resolve_faces`, and
  `world::tint::resolve_block_tint` — the two `resolve_*` functions were
  already name-keyed (not `BlockId`-keyed), so no change to their logic was
  needed, only their visibility. Also added `world::mesh::is_solid_name`,
  the name half of `is_solid` with no `BlockRegistry` in the way.
- `blueprint::mesh::resolve_palette`: resolves every `blueprint.palette`
  entry once, by name, into `{ solid, faces, tint }` — the blueprint-sized
  equivalent of `mesh_chunk_column`'s save-sized tables.
- `blueprint::mesh::mesh_blueprint`: walks the blueprint's `0..size` volume,
  emitting the same six faces per solid block as `mesh_chunk_column`,
  including 014's grass-side overlay.
- Unit tests mirroring `world::mesh`'s own (isolated block, adjacent-block
  culling, empty/all-air blueprints, biome tint, an unmapped block name
  falling back to the checker texture) plus the two roadmap-specific
  behaviours below.

## Watch out

- **Faces at the blueprint's outer boundary are always emitted.** Unlike a
  chunk column, a blueprint has no loaded neighbour to close a seam
  against — it's a free-standing object. A lookup past `0..size` reads as
  "exposed", not "air" and not "solid".
- **Air in the palette stays air, and not just at index 0.** A blueprint's
  palette can have an air entry anywhere (whatever an extraction or
  structure-file read happened to intern first) — `is_solid_name` is
  checked per palette entry, not assumed from position.
- **No per-block biome.** `Blueprint` doesn't carry one (`blueprint::extract`
  says so explicitly), so every biome-dependent `TintSource` on the whole
  mesh resolves against one `BiomeColors` the caller passes in. This isn't
  wrong so much as it's a decision B4's catalogue preview (or whatever else
  calls this) has to make consciously — "biome tint doesn't mean anything for
  a mesh that isn't standing in a specific spot yet."
- **Mesh coordinates are blueprint-local on all three axes**, `0..size.x/y/z`
  — including Y, unlike `mesh_chunk_column`'s chunk-local X/Z + absolute-world
  Y. A blueprint has no absolute world height until something places it, so
  there's nothing for Y to be relative to except its own origin. Callers
  apply their own `Transform`, same convention as the chunk mesher.

## Out of scope

B3 (rotation — a stair's `facing`, a log's `axis` etc. need rewriting on a
90/180/270 turn, which this ticket doesn't touch), B4 (the asset catalogue
that will actually load a `.nbt` and call this). No caller yet, same as
036's reader had none — this is the meshing primitive B3/B4 will use.

## Done when

- `mesh_blueprint` exists, sharing `world::mesh`'s quad emission rather than
  duplicating it.
- Boundary-always-exposed and non-zero-air-stays-air are both covered by a
  test.
- `cargo build` and `cargo test` are clean.

## Resolution

Landed as designed. `blueprint::mesh::mesh_blueprint`, re-exported from
`blueprint` the same way 036's reader is (no caller yet, same reasoning).

The visibility widening turned out to be the whole trick: `resolve_faces`
and `resolve_block_tint` were already written against a bare `&str` name,
not a `BlockId` — ticket 004/013's authors apparently kept the name-based
core separate from the `BlockId`-table wrapper around it without needing to
for this to pay off later. Only `pub(crate)` had to change, not the
functions themselves. The `Face` enum and the quad-push functions needed
the same treatment for the same reason — sharing geometry emission rather
than re-deriving the "per-face UV winding" logic `world::mesh`'s module docs
warn is easy to get subtly wrong.

8 new tests in `blueprint::mesh::tests` (271 total, up from 263). All pass,
including against a real unmapped block name (falls back to the checker
texture, same as the chunk mesher) and a grass block's biome-tinted top
face.

No manual/in-game check needed — nothing here spawns an entity or touches a
world; it's a pure `Blueprint`/`AtlasUvIndex` -> `Mesh` function, exercised
entirely by unit tests. B4's catalogue is what will spawn the result and be
worth an eyeball check.
