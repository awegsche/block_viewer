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
//!
//! ## Per-face UV winding
//!
//! Because each face lists its four corners starting from a different
//! physical corner (whichever keeps the CCW-from-outside/`(0,1,2,0,2,3)`
//! rule above true without special-casing), a single fixed
//! `(u0,v1)-(u0,v0)-(u1,v0)-(u1,v1)` UV order does **not** land on the same
//! four corners for every face. Worked out by hand (viewer standing on the
//! normal's side, looking at the face, `up` = Bevy `+Y`): East/South's
//! corner list comes out as bottom-right→top-right→top-left→bottom-left,
//! which that fixed order maps correctly (it's a pure horizontal mirror of
//! the naive expectation); West/North's comes out as
//! bottom-left→bottom-right→top-right→top-left, which the *same* fixed
//! order instead maps as a diagonal transpose — right on two corners, wrong
//! on the other two. A mirror on one pair of faces and a transpose on the
//! other differ by a 90° rotation, which is exactly the "side textures look
//! rotated" symptom this produces on any side texture with directional
//! detail (grain, an asymmetric bevel, ...). [`Face::corner_uvs`] gives each
//! face the UV order that actually matches its own corner list instead of
//! sharing one. `Up`/`Down` are left on the original order — nothing
//! reported a problem there, and top/bottom UV orientation has no single
//! "correct" answer the way the four side faces (which must at least agree
//! with each other) do without also matching Minecraft's real north-aligned
//! convention, which is out of scope here.
//!
//! ## Vertex colour channel (ticket 011)
//!
//! Every mesh carries `Mesh::ATTRIBUTE_COLOR`, a **multiplicative
//! modulation** of the sampled atlas texel, in **linear** (not sRGB) space.
//! White (`[1.0, 1.0, 1.0, 1.0]`) means unmodified. Contributors multiply
//! into it independently:
//!
//! ```text
//! vertex_color = biome_tint (013) * baked_light (010) * ao (010)
//! ```
//!
//! Anything writing this channel converts from sRGB itself — colours read
//! out of a PNG or written as a hex literal are sRGB and must go through
//! `Color::srgb_u8(..).to_linear()` before they land here. Feeding sRGB
//! values in directly makes tinted surfaces visibly too bright and washed
//! out, since `StandardMaterial` multiplies vertex colour into an
//! already-linearised base colour — there's no error, just a wrong-looking
//! world.
//!
//! The attribute is emitted unconditionally, even for chunks with nothing
//! to tint: Bevy specialises the render pipeline on the mesh's vertex
//! layout, so a mix of with-colour and without-colour chunk meshes would
//! mean two pipelines and two draw-call batches for the same material.

use bevy::{asset::RenderAssetUsages, prelude::*, render::mesh::Indices};

use super::atlas::{BlockFaces, UvRect};
use super::block::{BlockId, BlockRegistry};
use super::decode::{ChunkColumn, SECTION_SIZE};
use super::tint::{BiomeColors, BlockTint, TintSource};

/// Block names that don't occlude neighbouring faces. Water and leaves are
/// ticket 010's problem — treated as solid (opaque, face-culling) for now.
const NON_SOLID: [&str; 3] = [
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air",
];

/// The name half of [`is_solid`], with no [`BlockRegistry`] in the way —
/// what [`super::super::blueprint::mesh`] (ticket 037, roadmap B2) checks a
/// palette entry's `BlockState::name` against directly, since a blueprint's
/// palette has no registry to resolve a [`BlockId`] through.
pub(crate) fn is_solid_name(name: &str) -> bool {
    !NON_SOLID.contains(&name)
}

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
    id != BlockRegistry::AIR && is_solid_name(registry.name(id))
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
/// read back as air — including sections below `column.floor_y` (ticket
/// 030) that were never decoded at all, which is exactly why callers
/// checking occlusion go through [`occludes`] rather than this function
/// directly: "missing == air" is right for a genuinely uniform-air section,
/// but wrong for one that simply wasn't decoded.
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

/// Whether chunk-local `(dx, world_y, dz)` within `column` occludes a face
/// against it: solid per `registry` ([`is_solid`]), or below `column`'s own
/// render floor (ticket 030). The module docs' "below the floor is opaque,
/// not air" rule lives here — `column`'s sections below `floor_y` were
/// never decoded, so [`block_in_column`] would otherwise read them back as
/// air and draw a face into the removed volume.
fn occludes(column: &ChunkColumn, registry: &BlockRegistry, dx: usize, world_y: i32, dz: usize) -> bool {
    world_y < column.floor_y || is_solid(block_in_column(column, dx, world_y, dz), registry)
}

/// The cross-column counterpart of [`occludes`], for `dx`/`dz` that may step
/// one past the 0..16 range into a neighbour column — mirrors the old
/// `block_at`'s boundary dispatch, but returns occlusion directly rather
/// than a [`BlockId`] a caller has to resolve through [`is_solid`]
/// separately, which is what let the below-the-floor check disappear into a
/// registry lookup that has no floor to consult. A neighbour that isn't
/// loaded yet never occludes — same "seam until it loads" rule ticket 003
/// already accepted for `block_at`.
fn occludes_at(
    column: &ChunkColumn,
    neighbors: &Neighbors,
    registry: &BlockRegistry,
    dx: i32,
    world_y: i32,
    dz: i32,
) -> bool {
    let size = SECTION_SIZE as i32;
    if dx < 0 {
        return neighbors
            .west
            .is_some_and(|c| occludes(c, registry, (dx + size) as usize, world_y, dz as usize));
    }
    if dx >= size {
        return neighbors
            .east
            .is_some_and(|c| occludes(c, registry, (dx - size) as usize, world_y, dz as usize));
    }
    if dz < 0 {
        return neighbors
            .north
            .is_some_and(|c| occludes(c, registry, dx as usize, world_y, (dz + size) as usize));
    }
    if dz >= size {
        return neighbors
            .south
            .is_some_and(|c| occludes(c, registry, dx as usize, world_y, (dz - size) as usize));
    }
    occludes(column, registry, dx as usize, world_y, dz as usize)
}

/// The six directions a face can be exposed in, named in Minecraft terms
/// (see the module docs for how these map onto Bevy's axes).
///
/// `pub(crate)`, along with the quad-emission functions below it
/// ([`face_geometry`], [`push_quad`], [`push_quad_offset`]) — shared with
/// [`super::super::blueprint::mesh`] (ticket 037, roadmap B2), which meshes
/// a `Blueprint`'s palette-and-indices grid through the same geometry
/// instead of duplicating it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Face {
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
    pub(crate) fn uv_rect(self, faces: &BlockFaces) -> UvRect {
        match self {
            Face::Up => faces.top,
            Face::Down => faces.bottom,
            Face::East | Face::West | Face::South | Face::North => faces.side,
        }
    }

    /// Which of a block's three per-face tint sources (ticket 013) this
    /// face uses — same top/bottom/side split as [`Face::uv_rect`].
    pub(crate) fn tint_source(self, tint: &BlockTint) -> TintSource {
        match self {
            Face::Up => tint.top,
            Face::Down => tint.bottom,
            Face::East | Face::West | Face::South | Face::North => tint.side,
        }
    }

    /// Whether this is one of the four horizontal faces — the only ones a
    /// [`BlockTint::side_overlay`] (014) can apply to.
    pub(crate) fn is_side(self) -> bool {
        matches!(self, Face::East | Face::West | Face::South | Face::North)
    }

    /// The four UV corners to zip against [`face_geometry`]'s corners, in
    /// the same order — see the module docs' "Per-face UV winding" section
    /// for how East/South and West/North ended up needing different
    /// mappings out of the same `UvRect`.
    pub(crate) fn corner_uvs(self, rect: UvRect) -> [[f32; 2]; 4] {
        let UvRect { u0, v0, u1, v1 } = rect;
        match self {
            Face::East | Face::South => [[u1, v1], [u1, v0], [u0, v0], [u0, v1]],
            Face::West | Face::North => [[u0, v1], [u1, v1], [u1, v0], [u0, v0]],
            Face::Up | Face::Down => [[u0, v1], [u0, v0], [u1, v0], [u1, v1]],
        }
    }
}

/// The four corners (CCW from outside) and outward normal, in Bevy space,
/// of `face` for the unit block at chunk-local `(dx, world_y, dz)`.
pub(crate) fn face_geometry(face: Face, dx: i32, world_y: i32, dz: i32) -> ([Vec3; 4], Vec3) {
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

pub(crate) fn push_quad(
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    face: Face,
    dx: i32,
    world_y: i32,
    dz: i32,
    uv_rect: UvRect,
    color: [f32; 4],
) {
    push_quad_offset(
        vertices, normals, uvs, colors, indices, face, dx, world_y, dz, uv_rect, color, 0.0,
    );
}

/// Same as [`push_quad`] but displaces every corner outward along the face
/// normal by `offset` blocks before pushing it — used for 014's grass-side
/// overlay quad, which is coplanar with the base side quad it sits on and
/// needs a small nudge to avoid z-fighting (see [`OVERLAY_EPSILON`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_quad_offset(
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    face: Face,
    dx: i32,
    world_y: i32,
    dz: i32,
    uv_rect: UvRect,
    color: [f32; 4],
    offset: f32,
) {
    let (corners, normal) = face_geometry(face, dx, world_y, dz);
    let base = vertices.len() as u32;
    for corner in corners {
        vertices.push((corner + normal * offset).into());
        normals.push(normal.into());
        colors.push(color);
    }
    // Per-face corner order — see the module docs' "Per-face UV winding"
    // section for why this can't be one fixed pattern shared by every face.
    uvs.extend_from_slice(&face.corner_uvs(uv_rect));
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// World-space nudge applied to 014's grass-side overlay quad, pushed
/// outward from its coplanar base quad along their shared face normal to
/// avoid z-fighting. Bevy's reversed-Z depth buffer concentrates precision
/// near the camera, so a fixed world-space epsilon stays well-behaved out to
/// the far plane at this project's render distances — see ticket 014's
/// "epsilon offset" section for the number to revisit if shimmering shows up
/// at the render-distance edge (the manual check in `todo.md`).
pub(crate) const OVERLAY_EPSILON: f32 = 0.001;

/// Opaque white — the identity value for the multiplicative vertex colour
/// channel (see the module docs), and what [`TintSource::None`] resolves to.
/// 010 (baked light / AO) is the one remaining contributor still owed a real
/// factor.
pub(crate) const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// The [`BlockTint`] every id resolves to if `block_tint` (ticket 013) is
/// shorter than the registry it was built from — should never trigger in
/// practice, mirrors [`FALLBACK_FACES`].
const FALLBACK_TINT: BlockTint = BlockTint::NONE;

/// The [`BiomeColors`] every biome id resolves to if `biome_colors` (ticket
/// 013) is shorter than the registry it was built from — opaque white on
/// every source, same reasoning as [`FALLBACK_TINT`].
const FALLBACK_BIOME_COLORS: BiomeColors = BiomeColors {
    grass: bevy::color::LinearRgba::WHITE,
    foliage: bevy::color::LinearRgba::WHITE,
    water: bevy::color::LinearRgba::WHITE,
};

/// Resolves `source` against `biome` to the vertex colour a face tinted by
/// it should carry.
pub(crate) fn resolve_tint_color(source: TintSource, biome: BiomeColors) -> [f32; 4] {
    let c = match source {
        TintSource::None => return WHITE,
        TintSource::Grass => biome.grass,
        TintSource::Foliage => biome.foliage,
        TintSource::Water => biome.water,
        TintSource::Fixed(c) => c,
    };
    [c.red, c.green, c.blue, c.alpha]
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
///
/// `block_tint`/`biome_colors` (ticket 013) resolve each emitted face's
/// vertex colour: `block_tint[id]` says what a face is tinted *by*
/// ([`TintSource`]), and for the biome-dependent sources,
/// `biome_colors[biome_id]` says what that resolves to. Build both with
/// [`super::tint::build_block_tint_table`]/[`super::tint::build_biome_tint_table`]
/// and reuse across every column, same as `uv_table`.
pub fn mesh_chunk_column(
    column: &ChunkColumn,
    registry: &BlockRegistry,
    neighbors: &Neighbors,
    uv_table: &[BlockFaces],
    block_tint: &[BlockTint],
    biome_colors: &[BiomeColors],
) -> Option<Mesh> {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut colors = Vec::new();
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
                    // `uv_table`/`block_tint` are built from the same
                    // registry these ids came from, so both lookups are
                    // always in range in practice; fall back rather than
                    // panic if a caller ever passes a mismatched table.
                    let faces = uv_table.get(id.0 as usize).copied().unwrap_or(FALLBACK_FACES);
                    let tint = block_tint.get(id.0 as usize).copied().unwrap_or(FALLBACK_TINT);
                    // Section-local `ly`, not `world_y` — `biome_at` expects
                    // coordinates within this section's own 16x16x16 grid.
                    let biome_id = section.biome_at(lx, ly, lz);
                    let biome = biome_colors
                        .get(biome_id.0 as usize)
                        .copied()
                        .unwrap_or(FALLBACK_BIOME_COLORS);

                    let mut face = |f: Face, exposed: bool| {
                        if !exposed {
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
                            world_y,
                            dz,
                            f.uv_rect(&faces),
                            resolve_tint_color(f.tint_source(&tint), biome),
                        );
                        // 014: an extra tinted quad over the side faces —
                        // only for blocks whose tint table says so
                        // (grass_block today), and only where the base side
                        // quad above was actually emitted, so it inherits
                        // face culling for free.
                        if f.is_side() {
                            if let Some((overlay_uv, overlay_source)) = tint.side_overlay {
                                push_quad_offset(
                                    &mut vertices,
                                    &mut normals,
                                    &mut uvs,
                                    &mut colors,
                                    &mut indices,
                                    f,
                                    dx,
                                    world_y,
                                    dz,
                                    overlay_uv,
                                    resolve_tint_color(overlay_source, biome),
                                    OVERLAY_EPSILON,
                                );
                            }
                        }
                    };

                    face(
                        Face::East,
                        !occludes_at(column, neighbors, registry, dx + 1, world_y, dz),
                    );
                    face(
                        Face::West,
                        !occludes_at(column, neighbors, registry, dx - 1, world_y, dz),
                    );
                    face(
                        Face::South,
                        !occludes_at(column, neighbors, registry, dx, world_y, dz + 1),
                    );
                    face(
                        Face::North,
                        !occludes_at(column, neighbors, registry, dx, world_y, dz - 1),
                    );
                    // Up never needs a special case: "nothing above" is
                    // genuinely air (the top of the world), so the default
                    // missing-section-is-air behaviour is exactly right —
                    // `occludes` reduces to plain `is_solid` here since
                    // `world_y + 1` is always above every column's floor.
                    face(
                        Face::Up,
                        !occludes(column, registry, lx, world_y + 1, lz),
                    );
                    // Down does need one: below `column.floor_y` isn't air,
                    // it's either the true world bottom (ticket 003, under
                    // `FloorPolicy::WholeWorld`) or the removed volume a
                    // render floor cut off (ticket 030) — either way, never
                    // emit the underside.
                    face(
                        Face::Down,
                        world_y > column.floor_y
                            && !occludes(column, registry, lx, world_y - 1, lz),
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
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

/// A distinguishable, non-square `UvRect` for the "Per-face UV winding"
/// regression tests below — asymmetric on purpose (`u0 != v0`, `u1 != v1`,
/// and neither pair is a multiple of the other) so a swapped U/V or a
/// mixed-up corner shows up as a wrong *value*, not just a coincidentally
/// equal one.
#[cfg(test)]
const TEST_UV_RECT: UvRect = UvRect {
    u0: 0.1,
    v0: 0.2,
    u1: 0.7,
    v1: 0.9,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::biome::BiomeRegistry;
    use crate::world::decode::ChunkSection;
    use crate::world::decode::{BIOME_GRID_VOLUME, SECTION_VOLUME, WORLD_MIN_Y};

    /// Regression test for the "Per-face UV winding" bug (see the module
    /// docs): every one of the four side faces must sample the texture's
    /// top (`v0`) at its geometrically-highest corners and the texture's
    /// bottom (`v1`) at its lowest, full stop — the bug this catches (found
    /// on West/North specifically) was a diagonal transpose that swapped
    /// `v0`/`v1` on two of a face's four corners, which this would have
    /// failed on before the fix.
    #[test]
    fn every_side_faces_top_corners_sample_v0_and_bottom_corners_sample_v1() {
        for face in [Face::East, Face::West, Face::South, Face::North] {
            let (corners, _) = face_geometry(face, 3, 10, 5);
            let uvs = face.corner_uvs(TEST_UV_RECT);
            let max_y = corners.iter().map(|c| c.y).fold(f32::MIN, f32::max);
            let min_y = corners.iter().map(|c| c.y).fold(f32::MAX, f32::min);
            for (corner, uv) in corners.iter().zip(uvs.iter()) {
                if corner.y == max_y {
                    assert_eq!(
                        uv[1], TEST_UV_RECT.v0,
                        "{face:?}: top corner {corner:?} should sample v0, got uv {uv:?}"
                    );
                } else {
                    assert_eq!(
                        corner.y, min_y,
                        "{face:?}: corner {corner:?} is neither the top nor bottom of the face"
                    );
                    assert_eq!(
                        uv[1], TEST_UV_RECT.v1,
                        "{face:?}: bottom corner {corner:?} should sample v1, got uv {uv:?}"
                    );
                }
            }
        }
    }

    /// Regression test for the same bug from the other side: East/South
    /// share one UV corner order (a horizontal mirror of the naive
    /// expectation) and West/North share a different one (see the module
    /// docs) — pin both down explicitly rather than only checking the V
    /// invariant above, which alone wouldn't catch a horizontal-only bug.
    #[test]
    fn east_and_south_share_a_uv_order_distinct_from_west_and_norths() {
        let east_south = Face::East.corner_uvs(TEST_UV_RECT);
        assert_eq!(Face::South.corner_uvs(TEST_UV_RECT), east_south);

        let west_north = Face::West.corner_uvs(TEST_UV_RECT);
        assert_eq!(Face::North.corner_uvs(TEST_UV_RECT), west_north);

        assert_ne!(
            east_south, west_north,
            "East/South's corner order and West/North's should differ — see the module docs"
        );
    }

    fn section_with(y: i8, set: &[((usize, usize, usize), BlockId)]) -> ChunkSection {
        let mut blocks = Box::new([BlockRegistry::AIR; SECTION_VOLUME]);
        for &((x, ly, z), id) in set {
            blocks[ChunkSection::index(x, ly, z)] = id;
        }
        // Face culling is what these tests check; the biome grid is
        // irrelevant here (ticket 013 is what reads it), so every section
        // just gets the plains default.
        ChunkSection { y, blocks, biomes: Box::new([BiomeRegistry::PLAINS; BIOME_GRID_VOLUME]) }
    }

    /// [`WORLD_MIN_Y`] floor — every pre-030 test in this module wants the
    /// old, uncut behaviour, matching `FloorPolicy::WholeWorld`.
    fn column_with(x: i32, z: i32, sections: Vec<ChunkSection>) -> ChunkColumn {
        column_with_floor(x, z, sections, WORLD_MIN_Y)
    }

    /// [`column_with`], with an explicit `floor_y` for ticket 030's own
    /// tests.
    fn column_with_floor(x: i32, z: i32, sections: Vec<ChunkSection>, floor_y: i32) -> ChunkColumn {
        ChunkColumn { x, z, sections, floor_y }
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

    /// No tint on any block — most of these tests check face-culling
    /// geometry, not colour, so an all-`None` table keeps the vertex colour
    /// channel at opaque white without pulling `tint::build_block_tint_table`
    /// (and a real block name) into every one of them.
    fn no_tint_table_for(registry: &BlockRegistry) -> Vec<BlockTint> {
        vec![BlockTint::NONE; registry.len()]
    }

    /// A single biome (`BiomeRegistry::PLAINS`, id 0) resolving every source
    /// to opaque white — pairs with [`no_tint_table_for`] so a plains lookup
    /// never changes what these tests assert on.
    fn white_biome_colors() -> Vec<BiomeColors> {
        vec![BiomeColors {
            grass: bevy::color::LinearRgba::WHITE,
            foliage: bevy::color::LinearRgba::WHITE,
            water: bevy::color::LinearRgba::WHITE,
        }]
    }

    #[test]
    fn isolated_block_emits_all_six_faces() {
        let (registry, stone) = stone_registry();
        // Placed well above WORLD_MIN_Y and away from section edges so
        // every neighbour lookup reads air.
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), stone)])]);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
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

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
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

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // 5 faces instead of 6: the bottom face at the world floor is skipped.
        assert_eq!(indices.len(), 5 * 6);
    }

    /// Ticket 030: the same "no underside" rule as
    /// [`world_floor_has_no_underside`], but for a per-column render floor
    /// above [`WORLD_MIN_Y`] — a block sitting right at `floor_y` gets no
    /// `Down` face either, since below it is the removed volume, not air.
    #[test]
    fn render_floor_has_no_underside_either() {
        let (registry, stone) = stone_registry();
        let column = column_with_floor(0, 0, vec![section_with(2, &[((5, 0, 5), stone)])], 32);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        assert_eq!(indices.len(), 5 * 6);
    }

    /// Ticket 030's other correctness case: chunk A kept its terrain all the
    /// way down (`floor_y = WORLD_MIN_Y`); its east neighbour B was cut off
    /// at `floor_y = 32` and never decoded anything below that. A block in A
    /// right at the shared boundary, well below B's floor, must not draw an
    /// east face into B's removed volume — otherwise two adjacent chunks
    /// with different floors would grow a wall of faces at exactly the
    /// boundary this scheme exists to avoid.
    #[test]
    fn a_neighbor_with_a_higher_floor_occludes_below_its_own_floor() {
        let (registry, stone) = stone_registry();
        // world_y = 0*16 + 5 = 5, well below the neighbour's floor of 32.
        let column = column_with(0, 0, vec![section_with(0, &[((15, 5, 0), stone)])]);
        let east_neighbor = column_with_floor(1, 0, vec![], 32);
        let neighbors = Neighbors { east: Some(&east_neighbor), ..Default::default() };

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // 5 faces instead of 6: the east face is occluded by the
        // neighbour's render floor even though the neighbour has no actual
        // block data there to look up.
        assert_eq!(indices.len(), 5 * 6);
    }

    #[test]
    fn missing_neighbor_leaves_a_seam() {
        let (registry, stone) = stone_registry();
        // Block at the chunk's east edge (x=15); no east neighbour loaded.
        let column = column_with(0, 0, vec![section_with(0, &[((15, 5, 0), stone)])]);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
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

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();
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
        assert!(mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).is_none());
    }

    #[test]
    fn vertex_colors_are_opaque_white() {
        let (registry, stone) = stone_registry();
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), stone)])]);
        let neighbors = Neighbors::default();

        let mesh = mesh_chunk_column(&column, &registry, &neighbors, &uv_table_for(&registry), &no_tint_table_for(&registry), &white_biome_colors()).unwrap();

        let position_count = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len();
        let color_attr = mesh
            .attribute(Mesh::ATTRIBUTE_COLOR)
            .expect("mesh should carry a vertex colour attribute");
        let bevy::render::mesh::VertexAttributeValues::Float32x4(colors) = color_attr else {
            panic!("expected vertex colour attribute to be Float32x4");
        };

        assert_eq!(colors.len(), position_count);
        assert!(colors.iter().all(|&c| c == [1.0, 1.0, 1.0, 1.0]));
    }

    /// Ticket 013: a grass block's top-face vertices carry the biome's
    /// grass colour, and its bottom/side vertices stay white — the same
    /// per-face split [`super::tint::resolve_block_tint`] establishes,
    /// exercised all the way through mesh emission this time.
    #[test]
    fn grass_blocks_top_face_carries_the_biome_colour_and_its_other_faces_stay_white() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        // Isolated in open air so every one of its six faces is emitted.
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), grass_block)])]);
        let neighbors = Neighbors::default();

        let uv_table = uv_table_for(&registry);
        let mut block_tint = no_tint_table_for(&registry);
        block_tint[grass_block.0 as usize] = BlockTint {
            top: TintSource::Grass,
            bottom: TintSource::None,
            side: TintSource::None,
            side_overlay: None,
        };
        let green = bevy::color::LinearRgba {
            red: 0.2,
            green: 0.8,
            blue: 0.1,
            alpha: 1.0,
        };
        let biome_colors = vec![BiomeColors {
            grass: green,
            foliage: bevy::color::LinearRgba::WHITE,
            water: bevy::color::LinearRgba::WHITE,
        }];

        let mesh =
            mesh_chunk_column(&column, &registry, &neighbors, &uv_table, &block_tint, &biome_colors)
                .unwrap();
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
            colors.iter().any(|&c| c == WHITE),
            "expected at least one vertex (bottom/side faces) left untinted"
        );
    }

    /// A fixed, easy-to-assert-on overlay colour/rect pair, distinct from
    /// [`WHITE`] and [`FALLBACK_UV`] so mixing it up with the base side
    /// quad's colour/UV is obvious in a failing assertion.
    fn overlay_green() -> bevy::color::LinearRgba {
        bevy::color::LinearRgba { red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0 }
    }

    fn overlay_uv() -> UvRect {
        UvRect { u0: 0.25, v0: 0.25, u1: 0.5, v1: 0.5 }
    }

    /// Ticket 014: a block whose tint has a `side_overlay` emits one extra
    /// quad per emitted side face (4 of its 6 faces are sides), on top of
    /// the usual 6 base quads.
    #[test]
    fn side_overlay_emits_one_extra_quad_per_side_face() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        // Isolated in open air so all 6 base faces (and so all 4 side
        // overlays) are emitted.
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), grass_block)])]);
        let neighbors = Neighbors::default();

        let uv_table = uv_table_for(&registry);
        let mut block_tint = no_tint_table_for(&registry);
        block_tint[grass_block.0 as usize] = BlockTint {
            top: TintSource::None,
            bottom: TintSource::None,
            side: TintSource::None,
            side_overlay: Some((overlay_uv(), TintSource::Fixed(overlay_green()))),
        };

        let mesh = mesh_chunk_column(
            &column,
            &registry,
            &neighbors,
            &uv_table,
            &block_tint,
            &white_biome_colors(),
        )
        .unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // 6 base quads + 4 side-overlay quads = 10 quads.
        assert_eq!(indices.len(), 10 * 6);
    }

    /// The overlay quad carries its own tint (the biome's grass colour, via
    /// `TintSource::Fixed` here), while the base side quad underneath it
    /// stays at whatever the block's own `side` tint source resolves to
    /// (`None` -> white) — the two colours are independent.
    #[test]
    fn side_overlay_quad_carries_its_own_colour_independent_of_the_base_quad() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), grass_block)])]);
        let neighbors = Neighbors::default();

        let uv_table = uv_table_for(&registry);
        let mut block_tint = no_tint_table_for(&registry);
        block_tint[grass_block.0 as usize] = BlockTint {
            top: TintSource::None,
            bottom: TintSource::None,
            side: TintSource::None,
            side_overlay: Some((overlay_uv(), TintSource::Fixed(overlay_green()))),
        };

        let mesh = mesh_chunk_column(
            &column,
            &registry,
            &neighbors,
            &uv_table,
            &block_tint,
            &white_biome_colors(),
        )
        .unwrap();
        let bevy::render::mesh::VertexAttributeValues::Float32x4(colors) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap()
        else {
            panic!("expected vertex colour attribute to be Float32x4");
        };

        let green = overlay_green();
        let green_arr = [green.red, green.green, green.blue, green.alpha];
        assert!(
            colors.iter().any(|&c| c == green_arr),
            "expected at least one vertex (an overlay quad) tinted green"
        );
        assert!(
            colors.iter().any(|&c| c == WHITE),
            "expected at least one vertex (a base side quad) to stay white"
        );
    }

    /// Ticket 014: the overlay quad is coplanar with the base side quad it
    /// sits on, so it must be pushed outward along their shared face normal
    /// by exactly [`OVERLAY_EPSILON`] to avoid z-fighting.
    #[test]
    fn side_overlay_quad_is_offset_outward_along_the_face_normal() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        let column = column_with(0, 0, vec![section_with(0, &[((5, 5, 5), grass_block)])]);
        let neighbors = Neighbors::default();

        let uv_table = uv_table_for(&registry);
        let mut block_tint = no_tint_table_for(&registry);
        block_tint[grass_block.0 as usize] = BlockTint {
            top: TintSource::None,
            bottom: TintSource::None,
            side: TintSource::None,
            side_overlay: Some((overlay_uv(), TintSource::Fixed(overlay_green()))),
        };

        let mesh = mesh_chunk_column(
            &column,
            &registry,
            &neighbors,
            &uv_table,
            &block_tint,
            &white_biome_colors(),
        )
        .unwrap();
        let bevy::render::mesh::VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("expected position attribute to be Float32x3");
        };
        let bevy::render::mesh::VertexAttributeValues::Float32x3(normals) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap()
        else {
            panic!("expected normal attribute to be Float32x3");
        };

        // Emission order (see the `face` closure): East(base), East(overlay),
        // West(base), West(overlay), South(base), South(overlay),
        // North(base), North(overlay), Up(base), Down(base) — 4 vertices
        // each. Each side's overlay quad (vertices 4..8, 12..16, ...)
        // matches its base quad (0..4, 8..12, ...) corner-for-corner, offset
        // by the normal.
        for side in 0..4 {
            let base_start = side * 8;
            let overlay_start = base_start + 4;
            for corner in 0..4 {
                let base_pos = Vec3::from(positions[base_start + corner]);
                let overlay_pos = Vec3::from(positions[overlay_start + corner]);
                let normal = Vec3::from(normals[base_start + corner]);
                let expected = base_pos + normal * OVERLAY_EPSILON;
                assert!(
                    (overlay_pos - expected).length() < 1e-6,
                    "side {side} corner {corner}: expected {expected:?}, got {overlay_pos:?}"
                );
            }
        }
    }

    /// Ticket 014: a side face culled by a loaded neighbour never gets its
    /// overlay either — the overlay inherits face culling from the base
    /// quad it sits on, rather than being emitted independently.
    #[test]
    fn side_overlay_is_not_emitted_for_a_face_culled_by_a_neighbour() {
        let mut registry = BlockRegistry::new();
        let grass_block = registry.intern("minecraft:grass_block");
        // Two adjacent grass blocks: the shared east/west face (and its
        // overlay) is culled on both sides.
        let column = column_with(
            0,
            0,
            vec![section_with(0, &[((5, 5, 5), grass_block), ((6, 5, 5), grass_block)])],
        );
        let neighbors = Neighbors::default();

        let uv_table = uv_table_for(&registry);
        let mut block_tint = no_tint_table_for(&registry);
        block_tint[grass_block.0 as usize] = BlockTint {
            top: TintSource::None,
            bottom: TintSource::None,
            side: TintSource::None,
            side_overlay: Some((overlay_uv(), TintSource::Fixed(overlay_green()))),
        };

        let mesh = mesh_chunk_column(
            &column,
            &registry,
            &neighbors,
            &uv_table,
            &block_tint,
            &white_biome_colors(),
        )
        .unwrap();
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!("expected U32 indices");
        };
        // Two isolated grass blocks would be 10 quads each (20 total). The
        // shared east/west face pair is culled from both blocks: 2 base
        // quads gone, and their 2 matching overlay quads gone with them.
        assert_eq!(indices.len(), (10 * 6 * 2) - (4 * 6));
    }

    /// Ticket 030 is a performance claim, and nobody had measured it before
    /// this test existed: decode + mesh every fully-generated chunk in one
    /// real region twice, once under each [`FloorPolicy`], and print
    /// sections decoded, vertices emitted and elapsed time for each. Not a
    /// pass/fail assertion beyond "both runs produced at least one mesh" —
    /// the numbers themselves are what matter, and they're recorded in the
    /// ticket's Resolution.
    #[test]
    fn measures_the_render_floors_effect_on_a_real_region() {
        use crate::world::decode::{decode_chunk, FloorPolicy};
        use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;
        use std::time::Instant;

        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        let mut cache = crate::region_cache::RegionCache::new(meta, 4);
        let region = cache.get_or_load((rx, rz)).expect("region should load");

        for (label, policy) in [
            ("WholeWorld", FloorPolicy::WholeWorld),
            ("BelowSurface(margin=16)", FloorPolicy::BelowSurface { margin: 16 }),
        ] {
            let mut registry = BlockRegistry::new();
            let mut biomes = BiomeRegistry::new();
            let mut sections_decoded = 0usize;
            let mut vertices = 0usize;
            let mut meshed_any = false;
            let start = Instant::now();

            for cx in 0..REGION_WIDTH_IN_CHUNKS {
                for cz in 0..REGION_WIDTH_IN_CHUNKS {
                    let Some(nbt) = region.get_chunk(cx, cz) else { continue };
                    let nbt = nbt.clone();
                    let Ok(column) = decode_chunk(&nbt, &mut registry, &mut biomes, policy) else {
                        continue;
                    };
                    sections_decoded += column.sections.len();

                    let uv_table = uv_table_for(&registry);
                    let block_tint = no_tint_table_for(&registry);
                    if let Some(mesh) = mesh_chunk_column(
                        &column,
                        &registry,
                        &Neighbors::default(),
                        &uv_table,
                        &block_tint,
                        &white_biome_colors(),
                    ) {
                        vertices += mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len();
                        meshed_any = true;
                    }
                }
            }

            println!(
                "ticket 030 measurement [{label}]: {sections_decoded} sections decoded, {vertices} vertices, {:?}",
                start.elapsed()
            );
            assert!(meshed_any, "expected at least one mesh out of a real region under {label}");
        }
    }
}
