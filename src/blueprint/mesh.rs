//! `Blueprint` -> Bevy `Mesh` (ticket 037, roadmap B2): a quad per exposed
//! face of a blueprint's palette-and-indices grid, the same way
//! [`crate::world::mesh::mesh_chunk_column`] meshes a decoded chunk column.
//!
//! ## Why this isn't `mesh_chunk_column`
//!
//! `mesh_chunk_column` resolves UVs and tint through tables indexed by
//! [`crate::world::block::BlockId`], built once per save off a
//! [`crate::world::block::BlockRegistry`]. A [`Blueprint`] has neither: its
//! palette is [`BlockState`]s (name + properties, see `blueprint::extract`),
//! with no registry behind it. So this module resolves each *palette entry*
//! — typically a handful, not a whole save's worth of block names — straight
//! off its name, reusing [`crate::world::atlas::resolve_faces`] and
//! [`crate::world::tint::resolve_block_tint`] rather than building a second
//! copy of either lookup. Geometry emission (the `Face` enum, `face_geometry`,
//! `push_quad`/`push_quad_offset`) is shared outright with `world::mesh` —
//! only *what* gets fed into it differs.
//!
//! ## Two decisions from the roadmap
//!
//! - **Faces at the blueprint's outer boundary are always emitted.** A
//!   blueprint is a free-standing object being previewed or stamped into the
//!   world, not a chunk with streamed neighbours to close a seam against —
//!   there's nothing there to look up, so the boundary reads as "exposed"
//!   rather than "air" or "solid".
//! - **Air in the palette stays air.** A blueprint routinely has air
//!   entries away from index 0 (whatever an extraction or a structure file
//!   happened to intern first) — a blueprint is not a solid box, and every
//!   air-named palette entry is skipped the same way index 0 is.
//!
//! ## Biome tint
//!
//! A blueprint carries no per-block biome (`blueprint::extract`'s module
//! docs list it among the deliberately-dropped fields), so there is no
//! per-face biome lookup the way `mesh_chunk_column` has one. Callers pass a
//! single [`BiomeColors`] and every biome-dependent [`TintSource`] on the
//! blueprint resolves against it — a reasonable default (e.g. plains) for a
//! catalogue preview or a placement ghost, not a claim about where the
//! building will actually stand.
//!
//! Mesh coordinates are blueprint-local: X/Z/Y all run `0..size`, with no
//! world offset baked in — the same "spawn with a `Transform`" convention
//! `mesh_chunk_column` uses for chunk-local X/Z, extended here to Y as well
//! since a blueprint has no absolute world height of its own.

use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::Indices};

use crate::world::atlas::{resolve_faces, AtlasUvIndex, BlockFaces, MISSING_TEXTURE};
use crate::world::mesh::{
    is_solid_name, push_quad, push_quad_offset, resolve_tint_color, Face, OVERLAY_EPSILON,
};
use crate::world::tint::{resolve_block_tint, BiomeColors, BlockTint, MISSING_TINT};

use super::Blueprint;

/// One palette entry's resolved rendering data, computed once per distinct
/// [`BlockState`](super::BlockState) rather than once per block instance —
/// mirrors `world::mesh`'s per-`BlockId` tables, just sized to the
/// blueprint's own (typically much smaller) palette instead of a whole
/// save's registry.
struct PaletteEntry {
    solid: bool,
    faces: BlockFaces,
    tint: BlockTint,
}

/// Resolves every entry in `blueprint.palette` once, in palette order — the
/// per-block loop below then does one `Vec` index per block instead of a
/// name lookup.
fn resolve_palette(blueprint: &Blueprint, atlas: &AtlasUvIndex) -> Vec<PaletteEntry> {
    // The same process-wide ledgers the world mesher warns against (ticket
    // 081), not sets local to this call: a blueprint is re-meshed on every
    // ghost-preview rebuild, and a palette entry the resource pack has no
    // texture for is the same missing texture each time.
    blueprint
        .palette
        .iter()
        .map(|state| {
            let name = state.name.strip_prefix("minecraft:").unwrap_or(&state.name);
            PaletteEntry {
                solid: is_solid_name(&state.name),
                faces: resolve_faces(name, atlas, &MISSING_TEXTURE),
                tint: resolve_block_tint(name, atlas, &MISSING_TINT),
            }
        })
        .collect()
}

/// The palette index at blueprint-local `(dx, dy, dz)`, or `None` outside
/// the blueprint's volume — the "always emit boundary faces" rule (see the
/// module docs) reads a `None` neighbour as exposed.
fn block_at(blueprint: &Blueprint, dx: i32, dy: i32, dz: i32) -> Option<u16> {
    if dx < 0
        || dy < 0
        || dz < 0
        || dx >= blueprint.size.x
        || dy >= blueprint.size.y
        || dz >= blueprint.size.z
    {
        return None;
    }
    let (sx, sz) = (blueprint.size.x as usize, blueprint.size.z as usize);
    let index = (dy as usize) * sz * sx + (dz as usize) * sx + (dx as usize);
    blueprint.blocks.get(index).copied()
}

/// Meshes a whole [`Blueprint`] into a single Bevy [`Mesh`], with a quad per
/// block face whose neighbour is non-solid or outside the blueprint (see the
/// module docs).
///
/// `biome` resolves every biome-dependent [`crate::world::tint::TintSource`]
/// on the blueprint — there is no per-block biome to look one up against
/// (see the module docs).
///
/// Returns `None` for an empty blueprint or one with no exposed faces at
/// all (e.g. entirely air), mirroring `mesh_chunk_column`.
pub fn mesh_blueprint(blueprint: &Blueprint, atlas: &AtlasUvIndex, biome: BiomeColors) -> Option<Mesh> {
    if blueprint.blocks.is_empty() {
        return None;
    }

    let palette = resolve_palette(blueprint, atlas);
    let is_solid_index = |index: u16| palette.get(index as usize).is_some_and(|e| e.solid);

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();

    let (sx, sy, sz) = (blueprint.size.x, blueprint.size.y, blueprint.size.z);
    for dy in 0..sy {
        for dz in 0..sz {
            for dx in 0..sx {
                // `block_at` never returns `None` inside `0..size` — this is
                // just the array lookup, not a boundary check.
                let Some(index) = block_at(blueprint, dx, dy, dz) else {
                    continue;
                };
                let Some(entry) = palette.get(index as usize) else {
                    continue;
                };
                if !entry.solid {
                    continue;
                }

                // Exposed if the neighbour is outside the blueprint (always
                // emit at the boundary) or is a non-solid palette entry.
                let exposed = |ndx: i32, ndy: i32, ndz: i32| match block_at(blueprint, ndx, ndy, ndz) {
                    None => true,
                    Some(neighbor) => !is_solid_index(neighbor),
                };

                let mut emit = |f: Face, is_exposed: bool| {
                    if !is_exposed {
                        return;
                    }
                    push_quad(
                        &mut vertices,
                        &mut normals,
                        &mut uvs,
                        &mut colors,
                        &mut indices,
                        f,
                        dx,
                        dy,
                        dz,
                        f.uv_rect(&entry.faces),
                        resolve_tint_color(f.tint_source(&entry.tint), biome),
                    );
                    // Same grass-side-overlay treatment as
                    // `mesh_chunk_column` (014) — inherits face culling for
                    // free since it only fires when the base quad above did.
                    if f.is_side() {
                        if let Some((overlay_uv, overlay_source)) = entry.tint.side_overlay {
                            push_quad_offset(
                                &mut vertices,
                                &mut normals,
                                &mut uvs,
                                &mut colors,
                                &mut indices,
                                f,
                                dx,
                                dy,
                                dz,
                                overlay_uv,
                                resolve_tint_color(overlay_source, biome),
                                OVERLAY_EPSILON,
                            );
                        }
                    }
                };

                emit(Face::East, exposed(dx + 1, dy, dz));
                emit(Face::West, exposed(dx - 1, dy, dz));
                emit(Face::South, exposed(dx, dy, dz + 1));
                emit(Face::North, exposed(dx, dy, dz - 1));
                emit(Face::Up, exposed(dx, dy + 1, dz));
                emit(Face::Down, exposed(dx, dy - 1, dz));
            }
        }
    }

    if indices.is_empty() {
        return None;
    }

    Some(
        Mesh::new(
            bevy::render::mesh::PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::BlockState;

    fn white_biome() -> BiomeColors {
        BiomeColors {
            grass: LinearRgba::WHITE,
            foliage: LinearRgba::WHITE,
            water: LinearRgba::WHITE,
        }
    }

    fn state(name: &str) -> BlockState {
        name.parse().unwrap()
    }

    /// A blueprint of `size` filled with `blocks[index]` at each position —
    /// `blocks` must already be in the module docs' Y-outer/Z-middle/X-inner
    /// order (same as `blueprint::extract`'s `Accumulator`).
    fn blueprint_with(size: IVec3, palette: Vec<BlockState>, blocks: Vec<u16>) -> Blueprint {
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette,
            blocks,
            data_version: 0,
            failed_columns: 0,
        }
    }

    #[test]
    fn a_single_block_blueprint_emits_all_six_faces() {
        let blueprint = blueprint_with(
            IVec3::ONE,
            vec![BlockState::air(), state("minecraft:stone")],
            vec![1],
        );
        let mesh = mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 6 * 6, "one isolated block, all faces exposed");
    }

    #[test]
    fn adjacent_blocks_cull_the_shared_face() {
        // 2x1x1: two solid blocks side by side on X.
        let blueprint = blueprint_with(
            IVec3::new(2, 1, 1),
            vec![BlockState::air(), state("minecraft:stone")],
            vec![1, 1],
        );
        let mesh = mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // Two isolated blocks would be 72 indices; the shared face pair (2
        // quads) is culled from both, leaving 60 — same arithmetic as
        // `world::mesh`'s equivalent test.
        assert_eq!(indices.len(), (6 * 6 * 2) - (2 * 6));
    }

    /// The roadmap's first B2 decision: unlike a chunk, a blueprint has no
    /// loaded neighbour to ask about — every boundary face is emitted.
    #[test]
    fn boundary_faces_are_always_emitted_even_with_no_neighbour_loaded() {
        let blueprint = blueprint_with(
            IVec3::ONE,
            vec![BlockState::air(), state("minecraft:stone")],
            vec![1],
        );
        let mesh = mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 6 * 6);
    }

    /// The roadmap's second B2 decision: an air palette entry away from
    /// index 0 is still air, not a solid block that happens to be
    /// unmapped.
    #[test]
    fn a_non_zero_air_palette_entry_stays_air() {
        // Palette: [stone, air] — air is index 1 here, not 0.
        let blueprint = blueprint_with(
            IVec3::new(2, 1, 1),
            vec![state("minecraft:stone"), BlockState::air()],
            vec![0, 1],
        );
        let mesh = mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // Only the one stone block meshes, and its face toward the air
        // neighbour is exposed rather than culled: all 6 faces.
        assert_eq!(indices.len(), 6 * 6);
    }

    #[test]
    fn an_entirely_air_blueprint_yields_no_mesh() {
        let blueprint = blueprint_with(IVec3::ONE, vec![BlockState::air()], vec![0]);
        assert!(mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).is_none());
    }

    #[test]
    fn an_empty_blueprint_yields_no_mesh() {
        let blueprint = blueprint_with(IVec3::ZERO, vec![BlockState::air()], vec![]);
        assert!(mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).is_none());
    }

    /// Ticket 013/037: a grass block's top face carries the biome colour
    /// passed in, exercised through the palette-entry path this module adds
    /// rather than `mesh_chunk_column`'s `BlockId` table.
    #[test]
    fn grass_blocks_top_face_carries_the_passed_in_biome_colour() {
        let blueprint = blueprint_with(
            IVec3::ONE,
            vec![BlockState::air(), state("minecraft:grass_block")],
            vec![1],
        );
        let green = LinearRgba { red: 0.2, green: 0.8, blue: 0.1, alpha: 1.0 };
        let biome = BiomeColors { grass: green, foliage: LinearRgba::WHITE, water: LinearRgba::WHITE };

        let mesh = mesh_blueprint(&blueprint, &AtlasUvIndex::default(), biome).unwrap();
        let bevy::render::mesh::VertexAttributeValues::Float32x4(colors) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap()
        else {
            panic!("expected vertex colour attribute to be Float32x4");
        };
        let green_arr = [green.red, green.green, green.blue, green.alpha];
        assert!(
            colors.iter().any(|&c| c == green_arr),
            "expected at least one vertex (the top face) tinted with the biome's grass colour"
        );
        assert!(
            colors.iter().any(|&c| c == crate::world::mesh::WHITE),
            "expected at least one vertex (bottom/side faces) left untinted"
        );
    }

    /// A block state the atlas has no texture for still meshes — it falls
    /// back to the checker rect rather than being dropped, same as
    /// `mesh_chunk_column`'s registry-indexed path.
    #[test]
    fn an_unmapped_block_name_still_meshes_with_the_fallback_texture() {
        let blueprint = blueprint_with(
            IVec3::ONE,
            vec![BlockState::air(), state("minecraft:totally_made_up_block")],
            vec![1],
        );
        let mesh = mesh_blueprint(&blueprint, &AtlasUvIndex::default(), white_biome()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 6 * 6);
    }
}
