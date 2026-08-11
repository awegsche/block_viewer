//! Decodes Minecraft chunk NBT into dense, render-friendly block grids.
//!
//! This module is deliberately decoupled from Bevy — it only knows about
//! [`rnbt::NbtField`] in and [`BlockId`]-indexed grids out. See ticket 002.

pub mod block;
pub mod decode;

// Not every re-export has a caller yet (the mesher in ticket 003 will use
// `ChunkColumn`/`ChunkSection`/`BlockId` directly) — allow the unused ones
// rather than trim the public API down to today's only caller.
#[allow(unused_imports)]
pub use block::{BlockId, BlockRegistry};
#[allow(unused_imports)]
pub use decode::{
    decode_chunk, ChunkColumn, ChunkSection, DecodeError, SECTION_SIZE, SECTION_VOLUME,
};
