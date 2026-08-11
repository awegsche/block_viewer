//! Turns a decoded [`ChunkColumn`] into a single Bevy [`Mesh`] — a quad per
//! block face whose neighbour is non-solid. See ticket 003.
//!
//! ## Axis mapping
//!
//! Minecraft is Y-up, X-east, Z-south, and (taken with the right-hand rule
//! implied by "east/up/south") left-handed relative to Bevy's right-handed
//! Y-up/X-right/Z-towards-viewer convention. We keep Minecraft X and Y as
//! Bevy X and Y unchanged and **negate Z** to flip the handedness:
//!
//! ```text
//! bevy.x = mc.x
//! bevy.y = mc.y
//! bevy.z = -mc.z
//! ```
//!
//! One consequence: the Bevy `+Z` face of a cube is the block's Minecraft
//! **north** face (`-Z` in Minecraft), and Bevy `-Z` is Minecraft **south**.
//! `+X`/`-X` and `+Y`/`-Y` are unaffected. [`face_geometry`] is the single
//! place that bakes this mapping in; every other function in this module
//! talks about neighbours in Minecraft directions (east/west/north/south).
//!
//! A block at Minecraft coordinates `(x, y, z)` occupies `x..x+1`, `y..y+1`,
//! `z..z+1` — not centred on the integer coordinate — so world-space
//! positions and the block-under-cursor readout (ticket 007) agree.
//!
//! Each face's four corners are listed in the order that is
//! counter-clockwise as seen from outside the cube (i.e. from the direction
//! the face's normal points), so every face triangulates as `(0,1,2,0,2,3)`
//! with no special-casing per direction — the placeholder mesher this
//! replaced mixed two different winding conventions across faces.

use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::Indices};

use super::atlas::{BlockFaces, UvRect};
use super::block::{BlockId, BlockRegistry};
use super::decode::{ChunkColumn, SECTION_SIZE};

/// World Y of the bottom of the world (1.18+ worlds start at Y=-64). Below
/// this, there is no column to compare against — it's the underside of the
/// world, never visible, so [`mesh_chunk_column`] never emits a face there
/// rather than treating the missing data as "air below".
const WORLD_MIN_Y: i32 = -64;

/// Block names that don't occlude neighbouring faces. Water and leaves are
/// ticket 010's problem — treated as solid (opaque, face-culling) for now.
const NON_SOLID: [&str; 3] = [
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air",
];

/// Whole-atlas UV rect used only as a last-resort fallback if `uv_table`
/// passed to [`mesh_chunk_column`] is shorter than the registry it was
/// built from — should never trigger in practice (see the call site).
const FALLBACK_UV: UvRect = UvRect {
    u0: 0.0,
    v0: 0.0,
    u1: 1.0,
    v1: 1.0,
};
const FALLBACK_FACES: BlockFaces = BlockFaces {
    top: FALLBACK_UV,
    bottom: FALLBACK_UV,
    side: FALLBACK_UV,
};

/// Whether `id` should count as "there" for face-culling purposes.
pub fn is_solid(id: BlockId, registry: &BlockRegistry) -> bool {
    id != BlockRegistry::AIR && !NON_SOLID.contains(&registry.name(id))
}

/// The four chunk columns horizontally adjacent to the one being meshed,
/// used to close seams at chunk boundaries. `None` means "neighbour not
/// loaded (yet)" — that boundary is treated as air, producing a visible
/// seam rather than a hard error, per ticket 003's staged rollout.
#[derive(Debug, Clone, Copy, Default)]
pub struct Neighbors<'a> {
    /// Chunk at `(x, z - 1)` — Minecraft north.
    pub north: Option<&'a ChunkColumn>,
    /// Chunk at `(x, z + 1)` — Minecraft south.
    pub south: Option<&'a ChunkColumn>,
    /// Chunk at `(x + 1, z)` — Minecraft east.
    pub east: Option<&'a ChunkColumn>,
    /// Chunk at `(x - 1, z)` — Minecraft west.
    pub west: Option<&'a ChunkColumn>,
}

/// Block at chunk-local `(dx, world_y, dz)` within a single column. Missing
/// sections (uniform-air, see ticket 002) and out-of-range `world_y` both
/// read back as air.
fn block_in_column(column: &ChunkColumn, dx: usize, world_y: i32, dz: usize) -> BlockId {
    let section_y = world_y.div_euclid(SECTION_SIZE as i32) as i8;
    let local_y = world_y.rem_euclid(SECTION_SIZE as i32) as usize;
    column
        .sections
        .iter()
        .find(|s| s.y == section_y)
        .map(|s| s.get(dx, local_y, dz))
        .unwrap_or(BlockRegistry::AIR)
}

/// Block at chunk-local `(dx, world_y, dz)`, where `dx`/`dz` may step one
/// past the 0..16 range — in which case the lookup crosses into the
/// matching neighbour column (or reads as air if that neighbour isn't
/// loaded).
fn block_at(column: &ChunkColumn, neighbors: &Neighbors, dx: i32, world_y: i32, dz: i32) -> BlockId {
    let size = SECTION_SIZE as i32;
    if dx < 0 {
        return neighbors
            .west
            .map_or(BlockRegistry::AIR, |c| {
                block_in_column(c, (dx + size) as usize, world_y, dz as usize)
            });
    }
    if dx >= size {
        return neighbors
            .east
            .map_or(BlockRegistry::AIR, |c| {
                block_in_column(c, (dx - size) as usize, world_y, dz as usize)
            });
    }
    if dz < 0 {
        return neighbors
            .north
            .map_or(BlockRegistry::AIR, |c| {
                block_in_column(c, dx as usize, world_y, (dz + size) as usize)
            });
    }
    if dz >= size {
        return neighbors
            .south
            .map_or(BlockRegistry::AIR, |c| {
                block_in_column(c, dx as usize, world_y, (dz - size) as usize)
            });
    }
    block_in_column(column, dx as usize, world_y, dz as usize)
}

/// The six directions a face can be exposed in, named in Minecraft terms
/// (see the module docs for how these map onto Bevy's axes).
#[derive(Debug, Clone, Copy)]
enum Face {
    East,
    West,
    Up,
    Down,
    South,
    North,
}

impl Face {
    /// Which of a block's three distinct textures (ticket 004) this face
    /// samples: `Up`/`Down` get the top/bottom texture, every horizontal
    /// face shares the side texture.
    fn uv_rect(self, faces: &BlockFaces) -> UvRect {
        match self {
            Face::Up => faces.top,
            Face::Down => faces.bottom,
            Face::East | Face::West | Face::South | Face::North => faces.side,
        }
    }
}

/// The four corners (CCW from outside) and outward normal, in Bevy space,
/// of `face` for the unit block at chunk-local `(dx, world_y, dz)`.
fn face_geometry(face: Face, dx: i32, world_y: i32, dz: i32) -> ([Vec3; 4], Vec3) {
    let x0 = dx as f32;
    let x1 = x0 + 1.0;
    let y0 = world_y as f32;
    let y1 = y0 + 1.0;
    // Minecraft z spans dz..dz+1; bevy.z = -mc.z flips that to -(dz+1)..-dz.
    let z0 = -(dz as f32 + 1.0);
    let z1 = -(dz as f32);

    match face {
        Face::East => (
            [
                Vec3::new(x1, y0, z0),
                Vec3::new(x1, y1, z0),
                Vec3::new(x1, y1, z1),
                Vec3::new(x1, y0, z1),
            ],
            Vec3::X,
        ),
        Face::West => (
            [
                Vec3::new(x0, y0, z0),
                Vec3::new(x0, y0, z1),
                Vec3::new(x0, y1, z1),
                Vec3::new(x0, y1, z0),
            ],
            Vec3::NEG_X,
        ),
        Face::Up => (
            [
                Vec3::new(x0, y1, z0),
                Vec3::new(x0, y1, z1),
                Vec3::new(x1, y1, z1),
                Vec3::new(x1, y1, z0),
            ],
            Vec3::Y,
        ),
        Face::Down => (
            [
                Vec3::new(x0, y0, z0),
                Vec3::new(x1, y0, z0),
                Vec3::new(x1, y0, z1),
                Vec3::new(x0, y0, z1),
            ],
            Vec3::NEG_Y,
        ),
        // Minecraft south (+Z) is Bevy -Z.
        Face::South => (
            [
                Vec3::new(x0, y0, z0),
                Vec3::new(x0, y1, z0),
                Vec3::new(x1, y1, z0),
                Vec3::new(x1, y0, z0),
            ],
            Vec3::NEG_Z,
        ),
        // Minecraft north (-Z) is Bevy +Z.
        Face::North => (
            [
                Vec3::new(x0, y0, z1),
                Vec3::new(x1, y0, z1),
                Vec3::new(x1, y1, z1),
                Vec3::new(x0, y1, z1),
            ],
            Vec3::Z,
        ),
    }
}

fn push_quad(
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    face: Face,
    dx: i32,
    world_y: i32,
    dz: i32,
    uv_rect: UvRect,
) {
    let (corners, normal) = face_geometry(face, dx, world_y, dz);
    let base = vertices.len() as u32;
    for corner in corners {
        vertices.push(corner.into());
        normals.push(normal.into());
    }
    // Same corner-to-corner pattern the placeholder single-texture UVs used
    // (full 0..1 per face) — just scaled/offset into this face's atlas tile
    // (ticket 004) instead of the whole image.
    uvs.extend_from_slice(&[
        [uv_rect.u0, uv_rect.v1],
        [uv_rect.u0, uv_rect.v0],
        [uv_rect.u1, uv_rect.v0],
        [uv_rect.u1, uv_rect.v1],
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Meshes one chunk column into a single Bevy [`Mesh`], with a quad per
/// block face whose neighbour is non-solid (see [`is_solid`]).
///
/// Vertex positions are chunk-local (Bevy X/Z in `0..16`/`-16..0`, Y at the
/// block's absolute world height) — spawn the resulting mesh with a
/// `Transform` at the chunk's world origin (`(column.x*16, 0, -column.z*16)`
/// under this module's axis mapping) rather than baking that offset in here,
/// so the same mesh data could be reused for multiple transforms if needed.
///
/// Returns `None` if the column has no exposed faces at all (e.g. an
/// entirely air column), so callers can skip spawning a pointless entity.
///
/// `uv_table` gives each block's per-face atlas rect (ticket 004),
/// indexed directly by [`BlockId`] — build it once per save with
/// [`super::atlas::build_block_uv_table`] and reuse it across every column.
pub fn mesh_chunk_column(
    column: &ChunkColumn,
    registry: &BlockRegistry,
    neighbors: &Neighbors,
    uv_table: &[BlockFaces],
) -> Option<Mesh> {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for section in &column.sections {
        let base_y = section.y as i32 * SECTION_SIZE as i32;
        for ly in 0..SECTION_SIZE {
            let world_y = base_y + ly as i32;
            for lz in 0..SECTION_SIZE {
                for lx in 0..SECTION_SIZE {
                    let id = section.get(lx, ly, lz);
                    if !is_solid(id, registry) {
                        continue;
                    }
                    let dx = lx as i32;
                    let dz = lz as i32;
                    // `uv_table` is built from the same registry these ids
                    // came from, so this is always in range in practice;
                    // fall back to the whole-atlas rect rather than panic
                    // if a caller ever passes a mismatched table.
                    let faces = uv_table.get(id.0 as usize).copied().unwrap_or(FALLBACK_FACES);

                    let mut face = |f: Face, exposed: bool| {
                        if exposed {
                            push_quad(
                                &mut vertices,
                                &mut normals,
                                &mut uvs,
                                &mut indices,
                                f,
                                dx,
                                world_y,
                                dz,
                                f.uv_rect(&faces),
                            );
                        }
                    };

                    face(
                        Face::East,
                        !is_solid(block_at(column, neighbors, dx + 1, world_y, dz), registry),
                    );
                    face(
                        Face::West,
                        !is_solid(block_at(column, neighbors, dx - 1, world_y, dz), registry),
                    );
                    face(
                        Face::South,
                        !is_solid(block_at(column, neighbors, dx, world_y, dz + 1), registry),
                    );
                    face(
                        Face::North,
                        !is_solid(block_at(column, neighbors, dx, world_y, dz - 1), registry),
                    );
                    // Up never needs a special case: "nothing above" is
                    // genuinely air (the top of the world), so the default
                    // missing-section-is-air behaviour is exactly right.
                    face(
                        Face::Up,
                        !is_solid(block_in_column(column, lx, world_y + 1, lz), registry),
                    );
                    // Down does need one: below WORLD_MIN_Y isn't air, it's
                    // "no world there" — never emit the underside.
                    face(
                        Face::Down,
                        world_y > WORLD_MIN_Y
                            && !is_solid(block_in_column(column, lx, world_y - 1, lz), registry),
                    );
                }
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
        .with_inserted_indices(Indices::U32(indices)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::decode::ChunkSection;
    use crate::world::decode::SECTION_VOLUME;

    fn section_with(y: i8, set: &[((usize, usize, usize), BlockId)]) -> ChunkSection {
        let mut blocks = Box::new([BlockRegistry::AIR; SECTION_VOLUME]);
        for &((x, ly, z), id) in set {
            blocks[ChunkSection::index(x, ly, z)] = id;
        }
        ChunkSection { y, blocks }
    }

    fn column_with(x: i32, z: i32, sections: Vec<ChunkSection>) -> ChunkColumn {
        ChunkColumn { x, z, sections }
    }

    fn stone_registry() -> (BlockRegistry, BlockId) {
        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");
        (registry, stone)
    }

    /// Face-culling geometry is what these tests check, not UVs — a
    /// same-for-every-block table (whole-atlas rect) sidesteps building a
    /// real [`super::super::atlas::TextureAtlas`] in unit tests.
    fn uv_table_for(registry: &BlockRegistry) -> Vec<BlockFaces> {
        vec![FALLBACK_FACES; registry.len()]
    }

    #[test]
    fn isolated_block_emits_all_six_faces() {
        let (registry, stone) = stone_registry();
        // Placed well above WORLD_MIN_Y and away from section edges so
        // every neighbour lookup reads air.
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), stone)])]);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry)).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 6 * 6, "6 faces * 2 tris * 3 indices");
    }

    #[test]
    fn adjacent_blocks_cull_the_shared_face() {
        let (registry, stone) = stone_registry();
        let column = column_with(
            0,
            0,
            vec![section_with(
                0,
                &[((5, 5, 5), stone), ((6, 5, 5), stone)],
            )],
        );
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry)).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // Two isolated blocks would be 72 indices; the shared east/west
        // face pair (2 quads) is culled from both, leaving 60.
        assert_eq!(indices.len(), (6 * 6 * 2) - (2 * 6));
    }

    #[test]
    fn world_floor_has_no_underside() {
        let (registry, stone) = stone_registry();
        // WORLD_MIN_Y = -64 -> section -4, local y 0.
        let column = column_with(0, 0, vec![section_with(-4, &[((5, 0, 5), stone)])]);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry)).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // 5 faces instead of 6: the bottom face at the world floor is skipped.
        assert_eq!(indices.len(), 5 * 6);
    }

    #[test]
    fn missing_neighbor_leaves_a_seam() {
        let (registry, stone) = stone_registry();
        // Block at the chunk's east edge (x=15); no east neighbour loaded.
        let column = column_with(0, 0, vec![section_with(0, &[((15, 5, 0), stone)])]);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry)).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // All 6 faces render, including the east face facing the unloaded
        // neighbour (treated as air) -> visible seam, as documented.
        assert_eq!(indices.len(), 6 * 6);
    }

    #[test]
    fn loaded_neighbor_closes_the_seam() {
        let (registry, stone) = stone_registry();
        let column = column_with(0, 0, vec![section_with(0, &[((15, 5, 0), stone)])]);
        let east_neighbor = column_with(1, 0, vec![section_with(0, &[((0, 5, 0), stone)])]);
        let neighbors = Neighbors {
            east: Some(&east_neighbor),
            ..Default::default()
        };

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry)).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // The east face is now culled by the loaded neighbour.
        assert_eq!(indices.len(), 5 * 6);
    }

    #[test]
    fn fully_air_column_yields_no_mesh() {
        let (registry, _) = stone_registry();
        let column = column_with(0, 0, vec![]);
        let neighbors = Neighbors::default();
        assert!(mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry)).is_none());
    }
}
