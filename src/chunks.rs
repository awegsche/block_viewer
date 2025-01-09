use bevy::{
    asset::{LoadState, RenderAssetUsages},
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
    render::mesh::Indices,
    sprite::TextureAtlasBuilder,
};
use ranvil::{get_saves, Save};
use rnbt::{from_bytes, NbtField, NbtList, NbtValue};

use crate::{AppState, StateText};

// ---- structs ------------------------------------------------------------------------------------
//
/// A struct holding the expanded chunks of a region (save game?)
#[derive(Debug)]
pub(crate) struct Chunk {
    pub sections: Vec<Section>,
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
    pub fn from_nbt(nbt: NbtValue) -> Result<Self, LoadChunksError> {
        if let NbtValue::List(NbtList::Compound(blocks)) = nbt {
            let bit_length = if blocks.len() <= 1 {
                1
            } else if blocks.len() <= 16 {
                4
            } else {
                (blocks.len() as u32 - 1).ilog2() + 1
            };

            let blocks: Vec<NbtValue> = blocks.into_iter().map(|b| b.value).collect();
            let render_blocks = blocks.iter().map(RenderBlock::new).collect();
            dbg!(&render_blocks);
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

        println!("len blocks: {}", blocks.len());
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
    pub fn from_nbt(mut nbt: NbtField) -> Result<Self, LoadChunksError> {
        let mut block_states =
            nbt.swap_remove("block_states")
                .ok_or(LoadChunksError::UnableToFindField(
                    "block_states".to_string(),
                ))?;

        let palette = block_states
            .swap_remove("palette")
            .ok_or(LoadChunksError::UnableToFindField("palette".to_string()))?
            .value;
        let data = match block_states.swap_remove("data") {
            Some(NbtField {
                value: NbtValue::LongArray(data),
                ..
            }) => data,
            _ => vec![],
        };
        let palette = Palette::from_nbt(palette)?;
        let data = palette.expand_blocks(&data)?;
        return Ok(Section { palette, data });
    }

    /// returns the block's pallette entry
    pub fn get_block(&self, x: u32, y: u32, z: u32) -> &NbtValue {
        if self.palette.blocks.len() == 1 {
            return &self.palette.blocks[0];
        }
        let idx = y * 16 * 16 + z * 16 + x;
        &self.palette.blocks[self.data[idx as usize] as usize]
    }

    /// returns the block's pallette entry
    pub fn get_render_block(&self, x: u32, y: u32, z: u32) -> &RenderBlock {
        if self.palette.blocks.len() == 1 {
            return &self.palette.render_blocks[0];
        }
        let idx = y * 16 * 16 + z * 16 + x + 4096;
        let palette_idx = self.data[idx as usize] as usize;
        &self.palette.render_blocks[palette_idx as usize]
    }
}

impl std::error::Error for LoadChunksError {}

impl Chunk {
    pub fn new(mut save: Save) -> Result<Self, LoadChunksError> {
        save.load_region(0, 0);

        if let Some(region) = save.get_region(0, 0) {
            let chunk_data = region.get_chunk_nbt_data(0).unwrap().unwrap();
            if let Ok(chunk) = from_bytes(&chunk_data) {
                match chunk {
                    NbtField {
                        value: NbtValue::Compound(mut chunk_fields),
                        ..
                    } => {
                        let sections_idx = chunk_fields
                            .iter()
                            .position(|f| f.name == "sections")
                            .ok_or(LoadChunksError::UnableToFindField(
                            "sections".to_string(),
                        ))?;

                        let sections = chunk_fields.swap_remove(sections_idx);

                        if let NbtValue::List(NbtList::Compound(sections)) = sections.value {
                            return Ok(Chunk {
                                sections: sections
                                    .into_iter()
                                    .map(Section::from_nbt)
                                    .map(|s| s.unwrap())
                                    //.filter_map(Result::ok)
                                    .collect(),
                            });
                        }
                    }
                    _ => return Err(LoadChunksError::NotCompoundTag),
                }
            }
        }

        Err(LoadChunksError::UnableToDecompressChunkNbt)
    }

    pub fn get_render_block(&self, x: u32, y: u32, z: u32) -> &RenderBlock {
        let section = y / 16;
        &self.sections[section as usize].get_render_block(x, y - (section * 16), z)
    }
}

// ---- rendering functions ------------------------------------------------------------------------
//

#[derive(Resource, Default)]
pub(crate) struct BlockTextures(Vec<Handle<Image>>);

pub fn load_block_textures(mut commands: Commands, asset_server: Res<AssetServer>) {
    //asset_server.load_folder(path)

    commands.insert_resource(BlockTextures(
        ["stone", "dirt", "grass_block_top"]
            .map(|block| {
                asset_server.load_with_settings(
                    format!("minecraft/textures/block/{}.png", block),
                    |settings: &mut ImageLoaderSettings| {
                        settings.sampler = ImageSampler::nearest();
                    },
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
) {
    if block_textures
        .0
        .iter()
        .map(|h| server.is_loaded(h.id()))
        .all(|b| b)
    {
        next_state.set(AppState::Running);

        println!("textures loaded");
        let mut spans = 0;

        for mut span in &mut query {
            **span = "TExtures loaded".to_owned();
            spans += 1;
        }
        println!("spans: {}", spans);
    } else {
        for mut span in &mut query {
            **span = "loading textures".to_owned();
        }
    }
}

pub fn create_texture_atlas(asset_server: Res<AssetServer>) {
    let mut texture_atlas_builder = TextureAtlasBuilder::default();

    let block_texture_handle: Handle<Image> = asset_server.load_with_settings(
        "minecraft/textures/block/stone.png",
        |settings: &mut ImageLoaderSettings| {
            // Need to use nearest filtering to avoid bleeding between the slices with tiling
            settings.sampler = ImageSampler::nearest();
        },
    );

    asset_server.get
    texture_atlas_builder.add_texture(Some(block_texture_handle.id()), block_texture_handle.);
}

pub fn create_block_mesh() -> Mesh {
    // we load the region here, this should ideally be done seperately
    //
    let new_world = ranvil::get_save("New World").unwrap();

    let chunks = Chunk::new(new_world).unwrap();

    println!("Loaded {} chunks.", chunks.sections.len());

    println!("{:?}", chunks.sections[0]);
    for chunk in chunks.sections.iter() {
        println!("data.len = {}", chunk.data.len());
    }

    // let's test our edge finding algorithm.
    //
    const WIDTH: u32 = 16;
    const DEPTH: u32 = 16;
    const HEIGHT: u32 = 384;

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for x in 0..WIDTH {
        for z in 0..DEPTH {
            for y in 100..HEIGHT {
                let this_block = chunks.get_render_block(x, y, z);
                if this_block == &RenderBlock::Air {
                    continue;
                }

                if x == 0 || chunks.get_render_block(x - 1, y, z) == &RenderBlock::Air {
                    let n = vertices.len() as u32;
                    indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    vertices.extend_from_slice(&get_left_tris(x as f32, y as f32, z as f32));
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [-1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                    ]);
                }
                if x == WIDTH - 1 || chunks.get_render_block(x + 1, y, z) == &RenderBlock::Air {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_right_tris(x as f32, y as f32, z as f32));
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
                if z == 0 || chunks.get_render_block(x, y, z - 1) == &RenderBlock::Air {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_front_tris(x as f32, y as f32, z as f32));
                    indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ]);
                }
                if z == DEPTH - 1 || chunks.get_render_block(x, y, z + 1) == &RenderBlock::Air {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_back_tris(x as f32, y as f32, z as f32));
                    indices.extend_from_slice(&[n, n + 3, n + 1, n + 1, n + 3, n + 2]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ]);
                }
                if y == 0 || chunks.get_render_block(x, y - 1, z) == &RenderBlock::Air {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_bottom_tris(x as f32, y as f32, z as f32));
                    indices.extend_from_slice(&[n, n + 1, n + 3, n + 1, n + 2, n + 3]);
                    uvs.extend_from_slice(&[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
                    normals.extend_from_slice(&[
                        [0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, -1.0, 0.0],
                    ]);
                }
                if y == HEIGHT - 1 || chunks.get_render_block(x, y + 1, z) == &RenderBlock::Air {
                    let n = vertices.len() as u32;
                    vertices.extend_from_slice(&get_top_tris(x as f32, y as f32, z as f32));
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

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

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
    UnableToFindField(String),
    PaletteToBig,
    NotCompoundTag,
    NotListTag,
    UnableToDecompressChunkNbt,
}

impl std::fmt::Display for LoadChunksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to load chunks")?;

        match self {
            LoadChunksError::UnableToFindField(s) => write!(f, ": Unable to find field {}", s),
            LoadChunksError::PaletteToBig => write!(f, ": Palette too big"),
            LoadChunksError::NotCompoundTag => write!(f, ": Chunk is not a compound tag"),
            LoadChunksError::NotListTag => write!(f, ": Expected a list tag"),
            LoadChunksError::UnableToDecompressChunkNbt => {
                write!(f, ": Unable to decompress chunk nbt")
            }
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
            Palette::from_nbt(NbtValue::List(NbtList::Compound(palette_entries))).unwrap();

        assert_eq!(palette.bit_length, 2);
        assert_eq!(palette.bit_mask, 0b11);

        let (expanded, compressed) = {
            // create 64 blocks, those will be 2bit wide => fill two u64 in the compressed vector
            let mut expanded_blocks: Vec<u8> = Vec::with_capacity(64);
            let mut compressed_blocks: Vec<i64> = Vec::with_capacity(2);

            for _ in 0..(64 / 4) {
                for i in 0..4 {
                    expanded_blocks.push(i as u8);
                }
            }

            for _ in 0..2 {
                let mut block: u64 = 0;
                for i in 0..32 {
                    block |= (i % 4 as u64) << (i * 2);
                }
                compressed_blocks.push(unsafe { std::mem::transmute(block) });
            }

            (expanded_blocks, compressed_blocks)
        };

        let expended_by_palette = palette.expand_blocks(&compressed).unwrap();

        assert_eq!(expanded, expended_by_palette);
    }

    #[test]
    fn test_palette_bit_expand_odd() {
        // 5 palette entries => bit_length = 3
        let palette_entries = vec![
            NbtField::new_compound("air", vec![]),
            NbtField::new_compound("stone", vec![]),
            NbtField::new_compound("dirt", vec![]),
            NbtField::new_compound("grass", vec![]),
            NbtField::new_compound("bedrock", vec![]),
        ];

        let palette =
            Palette::from_nbt(NbtValue::List(NbtList::Compound(palette_entries))).unwrap();

        assert_eq!(palette.bit_length, 3);
        assert_eq!(palette.bit_mask, 0b111);

        let (expanded, compressed) = {
            // create 42 blocks, those will be 3bit wide => fill two u64 in the compressed vector
            // with one empty bit per u64
            let mut expanded_blocks: Vec<u8> = Vec::with_capacity(42);
            let mut compressed_blocks: Vec<i64> = Vec::with_capacity(2);

            for _ in 0..2 {
                for i in 0..21 {
                    expanded_blocks.push(i % 5 as u8);
                }
            }
            expanded_blocks.resize(42, 0);

            for _ in 0..2 {
                let mut block: u64 = 0;
                for i in 0..21 {
                    block |= (i % 5 as u64) << (i * 3);
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
            let mut expanded_blocks: Vec<u8> = Vec::with_capacity(42);
            let mut compressed_blocks: Vec<i64> = Vec::with_capacity(2);

            for _ in 0..2 {
                for i in 0..21 {
                    expanded_blocks.push(i % 5 as u8);
                }
            }
            expanded_blocks.resize(42, 0);

            for _ in 0..2 {
                let mut block: u64 = 0;
                for i in 0..21 {
                    block |= (i % 5 as u64) << (i * 3);
                }
                compressed_blocks.push(unsafe { std::mem::transmute(block) });
            }

            (expanded_blocks, compressed_blocks)
        };

        let section = Section::from_nbt(NbtField::new_compound(
            "section",
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
        ))
        .unwrap();

        assert_eq!(section.palette.bit_length, 3);
        assert_eq!(section.palette.bit_mask, 0b111);

        assert_eq!(expanded, section.data);

        let block = section.get_block(0, 0, 0);
    }
}
