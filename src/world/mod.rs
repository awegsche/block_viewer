//! Decodes Minecraft chunk NBT into dense, render-friendly block grids.
//!
//! This module is deliberately decoupled from Bevy — it only knows about
//! [`rnbt::NbtField`] in and [`BlockId`]-indexed grids out. See ticket 002.
//! [`mesh`] is the exception: it turns those grids into Bevy [`bevy::render::mesh::Mesh`]es
//! (ticket 003), so it's the only submodule with a Bevy dependency.

pub mod block;
pub mod decode;
pub mod mesh;

// Not every re-export has a caller in this repo yet (`BlockId`, the raw
// section/error types, and `is_solid` are part of the public API for
// callers like ticket 004/007, not dead weight) — allow the unused ones
// rather than trim the API down to today's only caller (`main.rs`).
#[allow(unused_imports)]
pub use block::{BlockId, BlockRegistry};
#[allow(unused_imports)]
pub use decode::{
    decode_chunk, ChunkColumn, ChunkSection, DecodeError, SECTION_SIZE, SECTION_VOLUME,
};
#[allow(unused_imports)]
pub use mesh::{is_solid, mesh_chunk_column, Neighbors};
