//! Interns Minecraft biome names into [`BiomeId`]s — the biome equivalent
//! of [`super::block::BlockRegistry`] (ticket 001/002), used by ticket 012's
//! per-section biome grids.

use std::collections::HashMap;

/// Interned handle for a biome name (e.g. `"minecraft:plains"`).
/// Comparisons and copies are cheap integer operations; resolve the
/// human-readable name back out via [`BiomeRegistry::name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BiomeId(pub u16);

/// Interns biome names into [`BiomeId`]s, shared across every chunk decoded
/// from the same save so IDs stay stable.
///
/// `BiomeId(0)` is reserved for `"minecraft:plains"` —
/// [`BiomeRegistry::new`] interns it first, so a section with no `biomes`
/// data at all ([`decode_chunk`](crate::world::decode::decode_chunk)) can
/// fall back to it and read back as plains rather than forcing an `Option`
/// through the mesher.
///
/// `Clone` for the same reason [`super::block::BlockRegistry`] is (ticket
/// 123): a chunk task meshes against a snapshot rather than under the lock.
#[derive(Debug, Clone)]
pub struct BiomeRegistry {
    names: Vec<String>,
    ids: HashMap<String, BiomeId>,
}

impl BiomeRegistry {
    pub const PLAINS: BiomeId = BiomeId(0);

    pub fn new() -> Self {
        let mut registry = Self {
            names: Vec::new(),
            ids: HashMap::new(),
        };
        let plains = registry.intern("minecraft:plains");
        debug_assert_eq!(plains, Self::PLAINS);
        registry
    }

    /// Returns the [`BiomeId`] for `name`, interning it if this is the first
    /// time it's been seen.
    pub fn intern(&mut self, name: &str) -> BiomeId {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = BiomeId(self.names.len() as u16);
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Resolves a previously interned [`BiomeId`] back to its name.
    ///
    /// Panics if `id` was never returned by [`BiomeRegistry::intern`] on
    /// this registry.
    pub fn name(&self, id: BiomeId) -> &str {
        &self.names[id.0 as usize]
    }

    /// Number of distinct names interned so far — always at least 1, since
    /// [`BiomeRegistry::new`] interns `"minecraft:plains"` up front.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Always false, kept alongside `len` per the standard `len`/`is_empty`
    /// pairing (clippy's `len_without_is_empty`) — no caller yet.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl Default for BiomeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
