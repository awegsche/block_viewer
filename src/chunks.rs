use bevy::{
    asset::{LoadState, RenderAssetUsages},
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
    render::mesh::Indices,
    sprite::TextureAtlasBuilder,
    state::commands,
};
use ranvil::{get_saves, Save};
use rnbt::{from_bytes, NbtField, NbtList, NbtValue};

use crate::{AppState, StateText};

const REGION_WIDTH: u32 = 32;
const REGION_HEIGHT: u32 = 32;
const CHUNK_WIDTH: u32 = 16;
const CHUNK_DEPTH: u32 = 16;
const CHUNK_HEIGHT: u32 = 384;

const NBT_NAME_SECTIONS: &'static str = "sections";
const NBT_NAME_XPOS: &'static str = "xPos";
const NBT_NAME_ZPOS: &'static str = "zPos";
const NBT_NAME_BLOCK_STATES: &'static str = "block_states";
const NBT_NAME_PALETTE: &'static str = "palette";
const NBT_NAME_DATA: &'static str = "data";
const NBT_NAME_BLOCKNAME: &'static str = "Name";

// ---- structs ------------------------------------------------------------------------------------
//
/// A struct holding the expanded chunks of a region (save game?)
///
#[derive(Component)]
struct BlockMesh;

#[derive(Resource, Debug)]
pub(crate) struct Chunks {
    chunks: Vec<Option<Chunk>>,
    width: u32,
    depth: u32,
    coord_offset_x: i32,
    coord_offset_z: i32,
}

impl Chunks {
    pub fn load_mesh_of_chunk(
        &self,
        x: i32,
        z: i32,
        textures_atlas: &TextureAtlasRes,
    ) -> Option<Mesh> {
        let adjacent_chunks = AdjacentChunks {
            x,
            z,
            middle: self.get_chunk(x, z),
            left: self.get_chunk(x - 1, z),
            right: self.get_chunk(x + 1, z),
            top: self.get_chunk(x, z + 1),
            bottom: self.get_chunk(x, z - 1),
            //top_left: self.get_chunk(x - 1, z + 1),
            //top_right: self.get_chunk(x + 1, z + 1),
            //bottom_left: self.get_chunk(x - 1, z - 1),
            //bottom_right: self.get_chunk(x + 1, z - 1),
        };

        adjacent_chunks.load_mesh(textures_atlas)
    }

    /// Returns the bounding rectangle of all loaded chunks.
    /// For any chunk coords inside the rectangle, there might still be non-exisiting chunks.
    /// Use `get_chunk` to get an `Option<&Chunk>`, which is `None` if the chunk is not loaded.
    pub(crate) fn get_bounding_rect(&self) -> IRect {
        IRect::from_corners(
            IVec2::new(self.coord_offset_x, self.coord_offset_z),
            IVec2::new(self.width as i32, self.depth as i32),
        )
    }
    pub(crate) fn get_bounding_rect_block(&self) -> IRect {
        IRect::from_corners(
            IVec2::new(
                self.coord_offset_x * CHUNK_WIDTH as i32,
                self.coord_offset_z * CHUNK_DEPTH as i32,
            ),
            IVec2::new(
                (self.width * CHUNK_WIDTH) as i32,
                (self.depth * CHUNK_DEPTH) as i32,
            ),
        )
    }
    pub(crate) fn get_chunk(&self, x: i32, z: i32) -> Option<&Chunk> {
        if x < self.coord_offset_x || z < self.coord_offset_z {
            return None;
        }
        let idx = self.get_index(x, z);
        if idx >= self.chunks.len() {
            return None;
        }
        self.chunks[idx].as_ref()
    }

    pub(crate) fn get_chunk_mut(&mut self, x: i32, z: i32) -> Option<&mut Chunk> {
        let idx = self.get_index(x, z);
        self.chunks[idx].as_mut()
    }

    pub(crate) fn get_chunk_absolute(&self, x: usize, z: usize) -> Option<&Chunk> {
        if x >= self.width as usize || z >= self.depth as usize {
            return None;
        }
        self.chunks[z * self.width as usize + x].as_ref()
    }

    /// Sets the chunk at the given coordinates.
    /// Note: coordinates are in world chunk coordinates, i.e. chunks in region (-1,-1) span from
    /// (-32, -32) to (-1, -1).
    pub(crate) fn set_chunk(&mut self, x: i32, z: i32, chunk: Chunk) {
        let idx = self.get_index(x, z);
        self.chunks[idx] = Some(chunk);
    }

    /// Gets the render block information at the given world block coordinates.
    /// Attention: world block coordinates are global ones, so the blocks of region (-1,-1) span from
    /// (-512, -512) to (-1, -1).///
    pub(crate) fn get_render_block(&self, x: i32, y: u32, z: i32) -> Option<&RenderBlock> {
        let x_block = self.x_coord_block(x);
        let z_block = self.z_coord_block(z);
        let chunk_x = x_block / CHUNK_WIDTH as usize;
        let chunk_z = z_block / CHUNK_WIDTH as usize;
        if let Some(chunk) = self.get_chunk_absolute(chunk_x, chunk_z) {
            let x_coord = x_block - (chunk_x * CHUNK_WIDTH as usize);
            let z_coord = z_block - (chunk_z * CHUNK_WIDTH as usize);

            return Some(chunk.get_render_block(x_coord as u32, y, z_coord as u32));
        }

        None
    }

    pub(crate) fn get_index(&self, x: i32, z: i32) -> usize {
        (self.z_coord(z) * self.depth as usize + self.x_coord(x)) as usize
    }

    pub(crate) fn x_coord(&self, x: i32) -> usize {
        (x - self.coord_offset_x) as usize
    }
    pub(crate) fn z_coord(&self, z: i32) -> usize {
        (z - self.coord_offset_z) as usize
    }

    pub(crate) fn x_coord_block(&self, x: i32) -> usize {
        (x - self.coord_offset_x * CHUNK_WIDTH as i32) as usize
    }
    pub(crate) fn z_coord_block(&self, z: i32) -> usize {
        (z - self.coord_offset_z * CHUNK_WIDTH as i32) as usize
    }
}

#[derive(Debug, Component)]
pub(crate) struct Chunk {
    pub sections: Vec<Section>,
    pub x: i32,
    pub z: i32,
    pub mesh: Option<Handle<Mesh>>,
}

#[derive(Debug)]
pub struct Section {
    pub palette: Palette,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Palette {
    pub blocks: Vec<NbtValue>,
    pub render_blocks: Vec<RenderBlock>,
    pub bit_length: u32,
    pub bit_mask: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBlock {
    Air,
    Stone,
    Dirt,
}

// ---- impls --------------------------------------------------------------------------------------
//
impl Palette {
    pub fn from_nbt(nbt: &NbtValue) -> Result<Self, LoadChunksError> {
        if let NbtValue::List(NbtList::Compound(blocks)) = nbt {
            let bit_length = if blocks.len() <= 1 {
                1
            } else if blocks.len() <= 16 {
                4
            } else {
                (blocks.len() as u32 - 1).ilog2() + 1
            };

            let blocks: Vec<NbtValue> = blocks.iter().map(|b| b.value.clone()).collect();
            let render_blocks = blocks.iter().map(RenderBlock::new).collect();
            //dbg!(&render_blocks);
            Ok(Palette {
                blocks,
                render_blocks,
                bit_length,
                bit_mask: 2_u64.pow(bit_length) - 1,
            })
        } else {
            Err(LoadChunksError::NotListTag)
        }
    }

    pub fn expand_blocks(&self, blocks: &[i64]) -> Result<Vec<u8>, LoadChunksError> {
        if self.bit_length > 8 {
            return Err(LoadChunksError::PaletteToBig);
        }

        //println!("len blocks: {}", blocks.len());
        let blocks_per_u64 = 64 / self.bit_length;
        let mut expanded = Vec::with_capacity(blocks.len() * blocks_per_u64 as usize);
        for block in blocks {
            let idx: u64 = unsafe { std::mem::transmute(*block) };
            for i in 0..blocks_per_u64 {
                expanded.push(((idx >> (i * self.bit_length)) & self.bit_mask) as u8);
            }
        }
        Ok(expanded)
    }
}

impl RenderBlock {
    pub fn new(block: &NbtValue) -> Self {
        match block.get("Name") {
            Some(NbtField {
                value: NbtValue::String(name),
                ..
            }) => match name.as_str() {
                "minecraft:air" => RenderBlock::Air,
                "minecraft:bedrock" => RenderBlock::Stone,
                "minecraft:stone" => RenderBlock::Stone,
                "minecraft:dirt" => RenderBlock::Dirt,
                _ => RenderBlock::Air,
            },
            _ => RenderBlock::Air,
        }
    }
}

impl Section {
    pub fn from_nbt(nbt: &NbtField) -> Result<Self, LoadChunksError> {
        let block_states = nbt
            .get(NBT_NAME_BLOCK_STATES)
            .ok_or(LoadChunksError::UnableToFindField(NBT_NAME_BLOCK_STATES))?;

        let palette = &block_states
            .get(NBT_NAME_PALETTE)
            .ok_or(LoadChunksError::UnableToFindField(NBT_NAME_PALETTE))?
            .value;
        let data = match block_states.get(NBT_NAME_DATA) {
            Some(NbtField {
                value: NbtValue::LongArray(data),
                ..
            }) => data,
            _ => &vec![],
        };
        let palette = Palette::from_nbt(palette)?;
        let data = palette.expand_blocks(data)?;
        return Ok(Section { palette, data });
    }

    /// Returns the index into the palette of the block at the given position.
    /// This function takes into account the case where only one block is present.
    pub fn get_block_idx(&self, x: u32, y: u32, z: u32) -> u8 {
        if self.palette.blocks.len() == 1 {
            0
        } else {
            let idx = y * 16 * 16 + z * 16 + x;
            self.data[idx as usize]
        }
    }

    pub fn is_block_transparent(&self, x: u32, y: u32, z: u32) -> bool {
        match self.get_render_block(x, y, z) {
            RenderBlock::Air => true,
            _ => false,
        }
    }

    /// returns the block's pallette entry
    pub fn get_block(&self, x: u32, y: u32, z: u32) -> &NbtValue {
        &self.palette.blocks[self.get_block_idx(x, y, z) as usize]
    }

    /// returns the block's pallette entry
    pub fn get_render_block(&self, x: u32, y: u32, z: u32) -> &RenderBlock {
        &self.palette.render_blocks[self.get_block_idx(x, y, z) as usize]
    }
}

impl std::error::Error for LoadChunksError {}

#[derive(Debug, Component)]
pub struct ChunkMesh {
    pub x: i32,
    pub z: i32,
    pub mesh: Option<Handle<Mesh>>,
}

impl Chunks {
    pub fn from_savegame(mut save: Save, commands: &mut Commands) -> Result<(), LoadChunksError> {
        let width = 32;
        let depth = 32;
        let coord_offset_x = -32;
        let coord_offset_z = -32;

        let region_x = -1;
        let region_z = -1;

        save.load_region(region_x, region_z);

        let mut chunks = Self {
            chunks: (0..(width * depth)).map(|_| None).collect(),
            width,
            depth,
            coord_offset_x,
            coord_offset_z,
        };

        if let Some(region) = save.get_region(region_x, region_z) {
            for z in 0..REGION_HEIGHT {
                for x in 0..REGION_HEIGHT {
                    let idx = z * width + x;
                    if let Ok(Some(chunk_data)) = region.get_chunk_nbt_data(idx as usize) {
                        if let Ok(chunk) = from_bytes(&chunk_data) {
                            let cx = x as i32 + region_x * REGION_WIDTH as i32;
                            let cz = z as i32 + region_z * REGION_WIDTH as i32;
                            chunks.set_chunk(cx, cz, Chunk::from_nbt(chunk)?);
                            commands.spawn(ChunkMesh {
                                x: cx,
                                z: cz,
                                mesh: None,
                            });
                        }
                    }
                }
            }
        }

        commands.insert_resource(chunks);
        Ok(())
    }
}

pub struct AdjacentChunks<'a> {
    pub x: i32,
    pub z: i32,

    pub middle: Option<&'a Chunk>,

    pub left: Option<&'a Chunk>,
    pub right: Option<&'a Chunk>,
    pub top: Option<&'a Chunk>,
    pub bottom: Option<&'a Chunk>,
    //pub top_left: Option<&'a Chunk>,
    //pub top_right: Option<&'a Chunk>,
    //pub bottom_left: Option<&'a Chunk>,
    //pub bottom_right: Option<&'a Chunk>,
}

impl<'a> AdjacentChunks<'a> {
    fn is_block_transparent(
        &self,
        mut section_y: usize,
        mut x: i32,
        mut y: i32,
        mut z: i32,
    ) -> bool {
        if y == -1 {
            if section_y == 0 {
                return true;
            }
            section_y -= 1;
            y = 15;
        }
        if y == 16 {
            if section_y == 15 {
                return true;
            }
            section_y += 1;
            y = 0;
        }

        let chunk = if x < 0 {
            if let Some(chunk) = self.left {
                x = 15;
                chunk
            } else {
                return true;
            }
        } else if x >= CHUNK_WIDTH as i32 {
            if let Some(chunk) = self.right {
                x = 0;
                chunk
            } else {
                return true;
            }
        } else if y < 0 {
            if let Some(chunk) = self.bottom {
                y = 15;
                chunk
            } else {
                return true;
            }
        } else if y >= CHUNK_DEPTH as i32 {
            if let Some(chunk) = self.top {
                y = 0;
                chunk
            } else {
                return true;
            }
        } else if z < 0 {
            if let Some(chunk) = self.bottom {
                z = 15;
                chunk
            } else {
                return true;
            }
        } else if z >= CHUNK_WIDTH as i32 {
            if let Some(chunk) = self.top {
                z = 0;
                chunk
            } else {
                return true;
            }
        } else {
            if let Some(chunk) = self.middle {
                chunk
            }
            else {
                return true;
            }
        };

        chunk.sections[section_y as usize].is_block_transparent(x as u32, y as u32, z as u32)
    }

    pub fn load_mesh(&self, textures_atlas: &TextureAtlasRes) -> Option<Mesh> {
        if let Some(middle) = self.middle {
            let chunk_x = middle.x as f32 * CHUNK_WIDTH as f32;
            let chunk_z = middle.z as f32 * CHUNK_WIDTH as f32;

            //todo!("write the edge detection algorithm for the middle chunk");
            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut uvs = Vec::new();
            let mut indices = Vec::new();

            let atlas_size = textures_atlas.layout.size;

            let texture_coords = middle
                .sections
                .iter()
                .map(|section| {
                    section
                        .palette
                        .blocks
                        .iter()
                        .filter_map(|block| {
                            if let Some(NbtField {
                                value: NbtValue::String(texture),
                                ..
                            }) = block.get(NBT_NAME_BLOCKNAME)
                            {
                                println!("texture: {}", texture);
                                let idx =
                                    textures_atlas.keys.iter().position(|key| key == texture)?;

                                //todo!("convert rect corners to f32_vecs, rescale by atlas size");
                                let t = &textures_atlas.layout.textures[idx];
                                let min = t.min.as_vec2() / atlas_size.as_vec2();
                                let max = t.max.as_vec2() / atlas_size.as_vec2();

                                return Some((min, max));
                            }
                            None
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            println!("texture_rects: {:?}", texture_coords);

            for section_y in 0..24 {
                let y_offset = section_y as f32 * CHUNK_WIDTH as f32;
                let texture_rects = &texture_coords[section_y];

                for y in 0..CHUNK_WIDTH as i32 {
                    // interior, only need to check the middle chunk
                    for z in 0..CHUNK_WIDTH as i32 {
                        for x in 0..CHUNK_WIDTH as i32 {
                            let x_pos = chunk_x + x as f32;
                            let z_pos = chunk_z + z as f32;
                            let y_pos = y as f32 + y_offset;
                            if self.is_block_transparent(section_y, x, y, z) {
                                continue;
                            }

                            let the_section = &middle.sections[section_y];
                            let idx = the_section.get_block_idx(x as u32, y as u32, z as u32) as usize;
                            let (p0, p1) = texture_rects[idx];

                            if self.is_block_transparent(section_y, x - 1, y, z) {
                                let n = vertices.len() as u32;
                                indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                vertices.extend_from_slice(&get_left_tris(x_pos, y_pos, z_pos));
                                uvs.extend_from_slice(&[
                                    [p0.x, p1.y],
                                    [p0.x, p0.y],
                                    [p1.x, p0.y],
                                    [p1.x, p1.y],
                                ]);
                                normals.extend_from_slice(&[
                                    [-1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                ]);
                            }
                            if self.is_block_transparent(section_y, x + 1, y, z) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_right_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                                //indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                uvs.extend_from_slice(&[
                                    [p0.x, p1.y],
                                    [p0.x, p0.y],
                                    [p1.x, p0.y],
                                    [p1.x, p1.y],
                                ]);
                                normals.extend_from_slice(&[
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                ]);
                            }
                            if self.is_block_transparent(section_y, x, y, z - 1) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_front_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                uvs.extend_from_slice(&[
                                    [p0.x, p1.y],
                                    [p0.x, p0.y],
                                    [p1.x, p0.y],
                                    [p1.x, p1.y],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, 0.0, -1.0],
                                    [0.0, 0.0, -1.0],
                                    [0.0, 0.0, -1.0],
                                    [0.0, 0.0, -1.0],
                                ]);
                            }
                            if self.is_block_transparent(section_y, x, y, z + 1) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_back_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                                uvs.extend_from_slice(&[
                                    [p0.x, p1.y],
                                    [p0.x, p0.y],
                                    [p1.x, p0.y],
                                    [p1.x, p1.y],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                ]);
                            }
                            if self.is_block_transparent(section_y, x, y - 1, z) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_bottom_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                uvs.extend_from_slice(&[
                                    [p0.x, p0.y],
                                    [p0.x, p1.y],
                                    [p1.x, p0.y],
                                    [p1.x, p1.y],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, -1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                ]);
                            }
                            if self.is_block_transparent(section_y, x, y + 1, z) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_top_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                                uvs.extend_from_slice(&[
                                    [p0.x, p0.y],
                                    [p0.x, p1.y],
                                    [p1.x, p0.y],
                                    [p1.x, p1.y],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                ]);
                            }
                        }
                    }
                }
            }

            println!("{} vertices", vertices.len());
            println!("{} uvs", uvs.len());
            println!("{} normals", normals.len());
            println!("{} indices", indices.len());

            let mesh = Mesh::new(
                bevy::render::mesh::PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_indices(Indices::U32(indices));

            return Some(mesh);
        }

        return None;
    }
}

impl Chunk {
    pub fn from_nbt(chunk: NbtField) -> Result<Self, LoadChunksError> {
        let x = match chunk
            .get(NBT_NAME_XPOS)
            .ok_or(LoadChunksError::UnableToFindField(NBT_NAME_XPOS))?
            .value
        {
            NbtValue::Int(x) => x,
            _ => return Err(LoadChunksError::NotIntTag),
        };
        let z = match chunk
            .get(NBT_NAME_ZPOS)
            .ok_or(LoadChunksError::UnableToFindField(NBT_NAME_ZPOS))?
            .value
        {
            NbtValue::Int(z) => z,
            _ => return Err(LoadChunksError::NotIntTag),
        };

        match chunk {
            NbtField {
                value: NbtValue::Compound(mut chunk_fields),
                ..
            } => {
                let sections_idx = chunk_fields
                    .iter()
                    .position(|f| f.name == "sections")
                    .ok_or(LoadChunksError::UnableToFindField("sections"))?;

                let sections = chunk_fields.swap_remove(sections_idx);

                if let NbtValue::List(NbtList::Compound(sections)) = sections.value {
                    return Ok(Chunk {
                        sections: sections
                            .iter()
                            .map(Section::from_nbt)
                            .map(|s| s.unwrap())
                            //.filter_map(Result::ok)
                            .collect(),
                        x,
                        z,
                        mesh: None,
                    });
                }
            }
            _ => return Err(LoadChunksError::NotCompoundTag),
        }

        Err(LoadChunksError::UnableToDecompressChunkNbt)
    }

    /*
        pub fn load_mesh(adjacent_chunks: &[Option<&mut Chunk>; 9]) {
            let (chunk_x, chunk_z) = match adjacent_chunks[4] {
                Some(chunk) => (
                    chunk.x as f32 * CHUNK_WIDTH as f32,
                    chunk.z as f32 * CHUNK_WIDTH as f32,
                ),
                None => return,
            };
            //todo!("write the edge detection algorithm for the middle chunk");
            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut uvs = Vec::new();
            let mut indices = Vec::new();

            let is_opaque = |block: Option<&RenderBlock>| {
                if let Some(b) = block {
                    return match b {
                        RenderBlock::Air => false,
                        _ => true,
                    };
                }
                false
            };

            for y in 0..CHUNK_HEIGHT {
                for z in 0..CHUNK_WIDTH as i32 {
                    for x in 0..CHUNK_WIDTH as i32 {
                        let x_pos = x as f32;
                        let z_pos = z as f32;
                        let y_pos = y as f32;
                        if let Some(this_block) = chunks.get_render_block(x, y, z) {
                            if this_block == &RenderBlock::Air {
                                continue;
                            }

                            if !is_opaque(chunks.get_render_block(x - 1, y, z)) {
                                let n = vertices.len() as u32;
                                indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                vertices.extend_from_slice(&get_left_tris(x_pos, y_pos, z_pos));
                                uvs.extend_from_slice(&[
                                    [0.0, 1.0],
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                ]);
                                normals.extend_from_slice(&[
                                    [-1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                ]);
                            }
                            if !is_opaque(chunks.get_render_block(x + 1, y, z)) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_right_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                                //indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                uvs.extend_from_slice(&[
                                    [0.0, 1.0],
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                ]);
                                normals.extend_from_slice(&[
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                ]);
                            }
                            if !is_opaque(chunks.get_render_block(x, y, z - 1)) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_front_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                uvs.extend_from_slice(&[
                                    [0.0, 1.0],
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, 0.0, -1.0],
                                    [0.0, 0.0, -1.0],
                                    [0.0, 0.0, -1.0],
                                    [0.0, 0.0, -1.0],
                                ]);
                            }
                            if !is_opaque(chunks.get_render_block(x, y, z + 1)) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_back_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                                uvs.extend_from_slice(&[
                                    [0.0, 1.0],
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                ]);
                            }
                            if !is_opaque(chunks.get_render_block(x, y - 1, z)) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_bottom_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                                uvs.extend_from_slice(&[
                                    [0.0, 1.0],
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, -1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                    [0.0, -1.0, 0.0],
                                ]);
                            }
                            if !is_opaque(chunks.get_render_block(x, y + 1, z)) {
                                let n = vertices.len() as u32;
                                vertices.extend_from_slice(&get_top_tris(x_pos, y_pos, z_pos));
                                indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                                uvs.extend_from_slice(&[
                                    [0.0, 1.0],
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                ]);
                                normals.extend_from_slice(&[
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                ]);
                            }
                        }
                    }
                }
            }
        }
    */

    pub fn unload_mesh(&mut self) {}

    pub fn get_render_block(&self, x: u32, y: u32, z: u32) -> &RenderBlock {
        let section = y / 16;
        &self.sections[section as usize].get_render_block(x, y - (section * 16), z)
    }

    pub fn get_block(&self, x: u32, y: u32, z: u32) -> &NbtValue {
        let section = y / 16;
        &self.sections[section as usize].get_block(x, y - (section * 16), z)
    }
}

// ---- rendering functions ------------------------------------------------------------------------
//

#[derive(Resource, Default)]
pub(crate) struct BlockTextures(Vec<(String, Handle<Image>)>);

pub fn load_block_textures(mut commands: Commands, asset_server: Res<AssetServer>) {
    //asset_server.load_folder(path)

    commands.insert_resource(BlockTextures(
        ["stone", "dirt", "grass_block_top"]
            .map(|block| {
                (
                    format!("minecraft:{}", block),
                    asset_server.load_with_settings(
                        format!("minecraft/textures/block/{}.png", block),
                        |settings: &mut ImageLoaderSettings| {
                            settings.sampler = ImageSampler::nearest();
                        },
                    ),
                )
            })
            .to_vec(),
    ));
}

pub(crate) fn check_block_texture_loading(
    mut next_state: ResMut<NextState<AppState>>,
    block_textures: Res<BlockTextures>,
    server: Res<AssetServer>,
    mut query: Query<&mut Text, With<StateText>>,
    mut textures: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    if block_textures
        .0
        .iter()
        .map(|(key, h)| server.is_loaded(h.id()))
        .all(|b| b)
    {
        next_state.set(AppState::TexturesLoaded);

        create_texture_atlas(
            &block_textures,
            &mut textures,
            &mut commands,
            &mut materials,
        );

        println!("textures loaded");
        let mut spans = 0;

        for mut span in &mut query {
            **span = "Textures loaded, atlas created".to_owned();
            spans += 1;
        }
        println!("spans: {}", spans);
    } else {
        for mut span in &mut query {
            **span = "loading textures".to_owned();
        }
    }
}

#[derive(Resource)]
pub struct TextureAtlasRes {
    layout: TextureAtlasLayout,
    //sources: Vec<Handle<Image>>,
    material_handle: Handle<StandardMaterial>,
    keys: Vec<String>,
}

pub struct BlockInfo {}

pub fn create_texture_atlas(
    block_textures: &BlockTextures,
    textures: &mut Assets<Image>,
    //mut texture_atlasses: ResMut<Assets<TextureAtlasLayout>>,
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
) {
    let mut texture_atlas_builder = TextureAtlasBuilder::default();

    let mut keys = Vec::new();

    for (key, texture) in block_textures.0.iter() {
        if let Some(the_texture) = textures.get(texture.id()) {
            texture_atlas_builder.add_texture(Some(texture.id()), the_texture);
            keys.push(key.clone());
        }
    }

    if let Ok((layout, sources, texture)) = texture_atlas_builder.build() {
        let texture_handle = textures.add(texture);
        let material_handle = materials.add(StandardMaterial {
            base_color_texture: Some(texture_handle),
            ..default()
        });
        commands.insert_resource(TextureAtlasRes {
            layout,
            material_handle,
            keys,
        });
    }
}

pub fn check_for_texture_atlas(
    mut next_state: ResMut<NextState<AppState>>,
    texture_atlas_res: Option<Res<TextureAtlasRes>>,
    mut query: Query<&mut Text, With<StateText>>,
) {
    if let Some(res) = texture_atlas_res {
        next_state.set(AppState::AtlasInserted);

        for mut span in &mut query {
            **span = "Atlas inserted".to_owned();
        }
    }
}

/*
pub fn load_chunks(
    texture_atlas_res: Res<TextureAtlasRes>,
    mut commands: Commands,
    //mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<&mut Chunks>,
) {
    println!("loading chunks");
    for mut chunks in &mut query {
        for z in chunks.get_bounding_rect().min.y..chunks.get_bounding_rect().max.y {
            for x in chunks.get_bounding_rect().min.x..chunks.get_bounding_rect().max.x {
                chunks.load_mesh_of_chunk(x, z, &texture_atlas_res, &mut commands, &mut meshes);
            }
        }
    }
    println!("chunks loaded");
}
*/

pub fn update_chunk_system(
    mut chunkmeshes: Query<&mut ChunkMesh>,
    texture_atlas_res: Res<TextureAtlasRes>,
    chunks: Res<Chunks>,
    mut commands: Commands,
    mut mesh_assets: ResMut<Assets<Mesh>>,
) {
    for mut chunkmesh in &mut chunkmeshes {
        if chunkmesh.mesh.is_none() {
            println!("loading mesh: {:?}", chunkmesh);
            if let Some(new_mesh) =
                chunks.load_mesh_of_chunk(chunkmesh.x, chunkmesh.z, &texture_atlas_res)
            {
                let handle = mesh_assets.add(new_mesh);
                commands.spawn((
                    Mesh3d(handle.clone()),
                    MeshMaterial3d(texture_atlas_res.material_handle.clone()),
                ));
                chunkmesh.mesh = Some(handle);
            }
        }
    }
}

pub fn spawn_meshes() {}

/*
pub fn create_block_mesh(/*texture_atlas_res: Res<TextureAtlasRes>*/) -> Mesh {

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    let is_opaque = |block: Option<&RenderBlock>| {
        if let Some(b) = block {
            return match b {
                RenderBlock::Air => false,
                _ => true,
            };
        }
        false
    };

    let rect = chunks.get_bounding_rect_block();

    for x in rect.min.x..rect.max.x {
        for z in rect.min.y..rect.max.y {
            for y in 1..CHUNK_HEIGHT {
                let x_pos = x as f32;
                let z_pos = z as f32;
                let y_pos = y as f32;
                if let Some(this_block) = chunks.get_render_block(x, y, z) {
                    if this_block == &RenderBlock::Air {
                        continue;
                    }

                    if !is_opaque(chunks.get_render_block(x - 1, y, z)) {
                        let n = vertices.len() as u32;
                        indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                        vertices.extend_from_slice(&get_left_tris(x_pos, y_pos, z_pos));
                        uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                        normals.extend_from_slice(&[
                            [-1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0],
                        ]);
                    }
                    if !is_opaque(chunks.get_render_block(x + 1, y, z)) {
                        let n = vertices.len() as u32;
                        vertices.extend_from_slice(&get_right_tris(x_pos, y_pos, z_pos));
                        indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                        //indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                        uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                        normals.extend_from_slice(&[
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                        ]);
                    }
                    if !is_opaque(chunks.get_render_block(x, y, z - 1)) {
                        let n = vertices.len() as u32;
                        vertices.extend_from_slice(&get_front_tris(x_pos, y_pos, z_pos));
                        indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                        uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                        normals.extend_from_slice(&[
                            [0.0, 0.0, -1.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 0.0, -1.0],
                        ]);
                    }
                    if !is_opaque(chunks.get_render_block(x, y, z + 1)) {
                        let n = vertices.len() as u32;
                        vertices.extend_from_slice(&get_back_tris(x_pos, y_pos, z_pos));
                        indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                        uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                        normals.extend_from_slice(&[
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0],
                        ]);
                    }
                    if !is_opaque(chunks.get_render_block(x, y - 1, z)) {
                        let n = vertices.len() as u32;
                        vertices.extend_from_slice(&get_bottom_tris(x_pos, y_pos, z_pos));
                        indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                        uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                        normals.extend_from_slice(&[
                            [0.0, -1.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, -1.0, 0.0],
                        ]);
                    }
                    if !is_opaque(chunks.get_render_block(x, y + 1, z)) {
                        let n = vertices.len() as u32;
                        vertices.extend_from_slice(&get_top_tris(x_pos, y_pos, z_pos));
                        indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                        uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                        normals.extend_from_slice(&[
                            [0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ]);
                    }
                }
            }
        }
    }

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}
    */

fn get_bottom_tris(x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [x - 0.5, y - 0.5, z - 0.5], // vertex with index 0
        [x + 0.5, y - 0.5, z - 0.5], // vertex with index 1
        [x + 0.5, y - 0.5, z + 0.5], // etc. until 23
        [x - 0.5, y - 0.5, z + 0.5],
    ]
}

fn get_top_tris(x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [x - 0.5, y + 0.5, z - 0.5], // vertex with index 0
        [x + 0.5, y + 0.5, z - 0.5], // vertex with index 1
        [x + 0.5, y + 0.5, z + 0.5], // etc. until 23
        [x - 0.5, y + 0.5, z + 0.5],
    ]
}

fn get_left_tris(x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [x - 0.5, y - 0.5, z - 0.5],
        [x - 0.5, y - 0.5, z + 0.5],
        [x - 0.5, y + 0.5, z + 0.5],
        [x - 0.5, y + 0.5, z - 0.5],
    ]
}

fn get_right_tris(x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [x + 0.5, y - 0.5, z - 0.5],
        [x + 0.5, y - 0.5, z + 0.5],
        [x + 0.5, y + 0.5, z + 0.5], // This vertex is at the same position as vertex with index 2, but they'll have different UV and normal
        [x + 0.5, y + 0.5, z - 0.5],
    ]
}

fn get_front_tris(x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [x - 0.5, y - 0.5, z - 0.5],
        [x - 0.5, y + 0.5, z - 0.5],
        [x + 0.5, y + 0.5, z - 0.5],
        [x + 0.5, y - 0.5, z - 0.5],
    ]
}

fn get_back_tris(x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    [
        [x - 0.5, y - 0.5, z + 0.5],
        [x - 0.5, y + 0.5, z + 0.5],
        [x + 0.5, y + 0.5, z + 0.5],
        [x + 0.5, y - 0.5, z + 0.5],
    ]
}

// ---- Error --------------------------------------------------------------------------------------

#[derive(Debug)]
pub enum LoadChunksError {
    UnableToFindField(&'static str),
    PaletteToBig,
    NotCompoundTag,
    NotListTag,
    NotIntTag,
    UnableToDecompressChunkNbt,
    UnableToFindRegion((usize, usize)),
    UnableToFindSaveGame,
}

impl std::fmt::Display for LoadChunksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to load chunks")?;

        match self {
            LoadChunksError::UnableToFindField(s) => write!(f, ": Unable to find field {}", s),
            LoadChunksError::PaletteToBig => write!(f, ": Palette too big"),
            LoadChunksError::NotCompoundTag => write!(f, ": Chunk is not a compound tag"),
            LoadChunksError::NotListTag => write!(f, ": Expected a list tag"),
            LoadChunksError::NotIntTag => write!(f, ": Expected an i32 tag"),
            LoadChunksError::UnableToDecompressChunkNbt => {
                write!(f, ": Unable to decompress chunk nbt")
            }
            LoadChunksError::UnableToFindRegion((x, y)) => {
                write!(f, ": Unable to find region ({}, {})", x, y)
            }
            LoadChunksError::UnableToFindSaveGame => write!(f, ": Unable to find save game"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_bit_expand_even() {
        // four palette entries => bit_length = 2
        let palette_entries = vec![
            NbtField::new_compound("air", vec![]),
            NbtField::new_compound("stone", vec![]),
            NbtField::new_compound("dirt", vec![]),
            NbtField::new_compound("grass", vec![]),
        ];

        let palette =
            Palette::from_nbt(&NbtValue::List(NbtList::Compound(palette_entries))).unwrap();

        assert_eq!(palette.bit_length, 4);
        assert_eq!(palette.bit_mask, 0b1111);

        let (expanded, compressed) = {
            // create 64 blocks, those will be 2bit wide => fill two u64 in the compressed vector
            let mut expanded_blocks: Vec<u8> = Vec::with_capacity(64);
            let mut compressed_blocks: Vec<i64> = Vec::with_capacity(2);

            for _ in 0..(64 / 4) {
                for i in 0..4 {
                    expanded_blocks.push(i as u8);
                }
            }

            for _ in 0..4 {
                let mut block: u64 = 0;
                for i in 0..16 {
                    block |= (i % 4 as u64) << (i * 4);
                }
                compressed_blocks.push(unsafe { std::mem::transmute(block) });
            }

            (expanded_blocks, compressed_blocks)
        };

        let expended_by_palette = palette.expand_blocks(&compressed).unwrap();

        assert_eq!(expanded, expended_by_palette);
    }

    #[test]
    fn test_section_from_nbt_odd() {
        // 5 palette entries => bit_length = 3
        let palette_entries = vec![
            NbtField::new_compound("air", vec![]),
            NbtField::new_compound("stone", vec![]),
            NbtField::new_compound("dirt", vec![]),
            NbtField::new_compound("grass", vec![]),
            NbtField::new_compound("bedrock", vec![]),
        ];

        let palette = NbtValue::List(NbtList::Compound(palette_entries));

        let (expanded, compressed) = {
            // create 42 blocks, those will be 3bit wide => fill two u64 in the compressed vector
            // with one empty bit per u64
            let mut expanded_blocks: Vec<u8> = Vec::with_capacity(32);
            let mut compressed_blocks: Vec<i64> = Vec::with_capacity(2);

            for _ in 0..2 {
                for i in 0..16 {
                    expanded_blocks.push(i % 5 as u8);
                }
            }
            expanded_blocks.resize(32, 0);

            for _ in 0..2 {
                let mut block: u64 = 0;
                for i in 0..16 {
                    block |= (i % 5 as u64) << (i * 4);
                }
                compressed_blocks.push(unsafe { std::mem::transmute(block) });
            }

            (expanded_blocks, compressed_blocks)
        };

        let section = Section::from_nbt(&NbtField::new_compound(
            "section",
            vec![NbtField::new_compound(
                "block_states",
                vec![
                    NbtField {
                        name: "palette".to_string(),
                        value: palette,
                    },
                    NbtField {
                        name: "data".to_string(),
                        value: NbtValue::LongArray(compressed),
                    },
                ],
            )],
        ))
        .unwrap();

        assert_eq!(section.palette.bit_length, 4);
        assert_eq!(section.palette.bit_mask, 0b1111);

        assert_eq!(expanded, section.data);

        let block = section.get_block(0, 0, 0);
    }

    #[test]
    fn test_chunk_coords_minus1_minus1() {
        let chunks = Chunks {
            chunks: vec![],
            width: 32,
            depth: 32,
            coord_offset_x: -32,
            coord_offset_z: -32,
        };

        assert_eq!(chunks.x_coord(-1), 31);
        assert_eq!(chunks.z_coord(-1), 31);
        assert_eq!(chunks.x_coord(-32), 0);
        assert_eq!(chunks.z_coord(-32), 0);
    }

    #[test]
    fn test_chunk_coords_0_0() {
        let chunks = Chunks {
            chunks: vec![],
            width: 32,
            depth: 32,
            coord_offset_x: 0,
            coord_offset_z: 0,
        };

        assert_eq!(chunks.x_coord(31), 31);
        assert_eq!(chunks.z_coord(31), 31);
        assert_eq!(chunks.x_coord(0), 0);
        assert_eq!(chunks.z_coord(0), 0);
    }
}
