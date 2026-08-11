use std::collections::HashMap;

/// Interned handle for a block-state name (e.g. `"minecraft:stone"`).
/// Comparisons and copies are cheap integer operations; resolve the
/// human-readable name back out via [`BlockRegistry::name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u16);

/// Interns block-state names into [`BlockId`]s, shared across every chunk
/// decoded from the same save so IDs stay stable.
///
/// `BlockId(0)` is reserved for `"minecraft:air"` — [`BlockRegistry::new`]
/// interns it first, and [`decode_chunk`](crate::world::decode::decode_chunk)
/// relies on that to fast-path uniform-air sections.
#[derive(Debug)]
pub struct BlockRegistry {
    names: Vec<String>,
    ids: HashMap<String, BlockId>,
}

impl BlockRegistry {
    pub const AIR: BlockId = BlockId(0);

    pub fn new() -> Self {
        let mut registry = Self {
            names: Vec::new(),
            ids: HashMap::new(),
        };
        let air = registry.intern("minecraft:air");
        debug_assert_eq!(air, Self::AIR);
        registry
    }

    /// Returns the [`BlockId`] for `name`, interning it if this is the first
    /// time it's been seen.
    pub fn intern(&mut self, name: &str) -> BlockId {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = BlockId(self.names.len() as u16);
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Resolves a previously interned [`BlockId`] back to its name.
    ///
    /// Panics if `id` was never returned by [`BlockRegistry::intern`] on
    /// this registry.
    pub fn name(&self, id: BlockId) -> &str {
        &self.names[id.0 as usize]
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}
