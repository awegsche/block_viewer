//! Decodes Minecraft chunk NBT into dense, render-friendly block grids.
//!
//! [`block`] and [`decode`] are deliberately decoupled from Bevy — they only
//! know about [`rnbt::NbtField`] in and [`BlockId`]-indexed grids out (see
//! ticket 002). [`mesh`] and [`atlas`] are the exception: [`mesh`] turns
//! those grids into Bevy [`bevy::render::mesh::Mesh`]es (ticket 003), and
//! [`atlas`] packs the vanilla texture set into one atlas [`bevy::image::Image`]
//! and maps block names to per-face UV rects for it (ticket 004).

pub mod atlas;
pub mod block;
pub mod decode;
pub mod mesh;

// Not every re-export has a caller in this repo yet (`BlockId`, the raw
// section/error types, and `is_solid` are part of the public API for
// callers like ticket 007, not dead weight) — allow the unused ones rather
// than trim the API down to today's only caller (`main.rs`).
#[allow(unused_imports)]
pub use atlas::{build_block_uv_table, AtlasUvIndex, BlockFaces, TextureAtlas, UvRect};
#[allow(unused_imports)]
pub use block::{BlockId, BlockRegistry};
#[allow(unused_imports)]
pub use decode::{
    decode_chunk, ChunkColumn, ChunkSection, DecodeError, SECTION_SIZE, SECTION_VOLUME,
};
#[allow(unused_imports)]
pub use mesh::{is_solid, mesh_chunk_column, Neighbors};
