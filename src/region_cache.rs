//! LRU cache of decoded Anvil regions (ticket 005-b).
//!
//! Region files are the I/O unit (32x32 chunks per `r.<x>.<z>.mca`):
//! opening and inflating one is per-region work, not per-chunk. Streaming
//! individual chunks naively would re-open/re-parse the same region file
//! repeatedly unless decoded regions are cached and shared across every
//! chunk that needs them, and evicted (actually freeing the decoded NBT,
//! not just unlinking a pointer) once nothing does.
//!
//! This cache is internally synchronous (`get_or_load` blocks on file I/O +
//! decode) — [`crate::chunk_pipeline`] (005-c) is what puts calls to it on
//! a background task, sharing one instance across every task via
//! `Arc<Mutex<RegionCache>>` rather than giving each task its own.
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use mc_anvil::chunkregion::ChunkRegion;
use mc_anvil::region::{Region, REGION_WIDTH_IN_CHUNKS};
use mc_anvil::{MCLoadError, SaveMeta};

/// Converts a chunk coordinate into the coordinate of the region that
/// contains it. Anvil regions are 32x32 chunks; `div_euclid` floors toward
/// negative infinity so chunk `(-1, 0)` lands in region `(-1, 0)`, not
/// region `(0, 0)` (`ranvil` doesn't expose this conversion itself — only
/// filename <-> region-coord parsing — so it lives here).
pub fn chunk_to_region_coord((cx, cz): (i32, i32)) -> (i32, i32) {
    let w = REGION_WIDTH_IN_CHUNKS as i32;
    (cx.div_euclid(w), cz.div_euclid(w))
}

/// A capacity budget (number of resident regions) that comfortably covers a
/// [`RenderDistance`](crate::streaming::RenderDistance) radius, per 005-b's
/// scope note. Regions needed from the center chunk to one edge of the
/// radius is `ceil(render_distance / 32) + 1` (the `+1` covers the center
/// chunk not landing on a region boundary); doubling that (both sides of
/// the center) plus the center region itself gives a square span that's
/// generous rather than tight — 005-e wires this into `RegionCache::new`.
pub fn recommended_capacity(render_distance: u32) -> usize {
    let per_side = (render_distance as f64 / REGION_WIDTH_IN_CHUNKS as f64).ceil() as usize + 1;
    let span = 2 * per_side + 1;
    span * span
}

/// An LRU cache of decoded [`ChunkRegion`]s, keyed by region coordinate.
/// `get_or_load` loads + parses a region on first access
/// (`ChunkRegion::load_chunks`, which decodes all 1024 chunks in one call —
/// see the ticket's "Watch out" note for measured timing) and reuses the
/// decoded result on every later access, until the region is evicted.
pub struct RegionCache {
    /// The save's metadata (region file layout + root path); used to build
    /// each region's file path and to skip regions the save doesn't have.
    save_meta: SaveMeta,
    /// Maximum number of decoded regions to keep resident before evicting
    /// the least-recently-used one. Passed in rather than hardcoded —
    /// [`recommended_capacity`] sizes it to a render distance.
    capacity: usize,
    entries: HashMap<(i32, i32), ChunkRegion>,
    /// Access order, least-recently-used first. `capacity` is small (a
    /// handful of regions covering one render distance), so an O(n)
    /// remove+push on every access is cheap — not worth a real intrusive
    /// LRU list at this size.
    order: Vec<(i32, i32)>,
    /// Regions whose `load_chunks` has already failed once (truncated or
    /// otherwise corrupt `.mca`, ticket 008) — remembered so a bad region is
    /// only attempted, and logged, once; every later request for the same
    /// coordinate fails fast without re-touching disk or logging again.
    /// Deliberately separate from `entries`, which only ever holds
    /// successfully loaded regions.
    failed: HashSet<(i32, i32)>,
}

impl RegionCache {
    /// `capacity` must be at least 1 (clamped up if given 0) or every
    /// access would immediately evict itself.
    pub fn new(save_meta: SaveMeta, capacity: usize) -> Self {
        Self {
            save_meta,
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: Vec::new(),
            failed: HashSet::new(),
        }
    }

    /// Number of regions currently resident. No caller outside this
    /// module's own tests yet.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Kept alongside `len` per the standard `len`/`is_empty` pairing
    /// (clippy's `len_without_is_empty`) — no caller yet, same as
    /// `BlockRegistry::is_empty`.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the decoded region at `region_coord`, loading and parsing it
    /// on first access and reusing that decode on every later access.
    /// Evicts the least-recently-used region first if this would exceed
    /// `capacity`.
    ///
    /// Errors if the save has no region at `region_coord`, or if loading it
    /// fails (missing/corrupt file, unsupported compression, ...) — the
    /// latter is remembered in `failed` (ticket 008) so a permanently broken
    /// region only gets logged, and its file re-read, once rather than on
    /// every chunk that lands inside it.
    pub fn get_or_load(&mut self, region_coord: (i32, i32)) -> Result<&ChunkRegion, MCLoadError> {
        if self.failed.contains(&region_coord) {
            return Err(MCLoadError::PathNotFoundError);
        }

        if !self.entries.contains_key(&region_coord) {
            let (rx, rz) = region_coord;
            if !self.save_meta.has_region(rx, rz) {
                return Err(MCLoadError::PathNotFoundError);
            }

            if self.entries.len() >= self.capacity {
                self.evict_lru();
            }

            let path = self.region_path(region_coord);
            let path_display = path.display().to_string();
            let mut region: ChunkRegion = Region::new(rx, rz, path).into();
            if let Err(err) = region.load_chunks() {
                self.failed.insert(region_coord);
                println!(
                    "block_viewer: skipping region ({rx}, {rz}) — failed to load {path_display}: {err}"
                );
                return Err(err);
            }
            self.entries.insert(region_coord, region);
        }

        self.touch(region_coord);
        Ok(self
            .entries
            .get(&region_coord)
            .expect("just inserted or touched above"))
    }

    fn region_path(&self, (rx, rz): (i32, i32)) -> PathBuf {
        self.save_meta.get_region_path(rx, rz)
    }

    /// Moves `region_coord` to the most-recently-used end of `order`.
    fn touch(&mut self, region_coord: (i32, i32)) {
        self.order.retain(|&k| k != region_coord);
        self.order.push(region_coord);
    }

    /// Drops the least-recently-used region, freeing its decoded chunk
    /// data (`ChunkRegion::chunks`) along with the rest of the struct.
    fn evict_lru(&mut self) {
        if self.order.is_empty() {
            return;
        }
        let lru = self.order.remove(0);
        self.entries.remove(&lru);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// This repo assumes a real Minecraft saves directory is reachable on
    /// the dev machine (see `main.rs::load_real_save`); these tests follow
    /// the same convention rather than faking one.
    fn real_save_meta() -> SaveMeta {
        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        saves
            .into_iter()
            .find(|s| s.regions.len() >= 3)
            .expect("need a save with at least 3 regions for these tests")
    }

    #[test]
    fn chunk_to_region_coord_floors_toward_negative_infinity() {
        assert_eq!(chunk_to_region_coord((0, 0)), (0, 0));
        assert_eq!(chunk_to_region_coord((31, 31)), (0, 0));
        assert_eq!(chunk_to_region_coord((32, 0)), (1, 0));
        assert_eq!(chunk_to_region_coord((-1, 0)), (-1, 0));
        assert_eq!(chunk_to_region_coord((-32, -33)), (-1, -2));
    }

    #[test]
    fn recommended_capacity_grows_with_render_distance() {
        // radius 10 fits inside one region's worth of margin either side.
        assert!(recommended_capacity(10) >= 9);
        assert!(recommended_capacity(64) > recommended_capacity(10));
    }

    #[test]
    fn get_or_load_returns_correct_chunk_data_for_a_real_region() {
        let meta = real_save_meta();
        let (rx, rz) = meta.regions[0];
        let mut cache = RegionCache::new(meta.clone(), 4);

        let region = cache.get_or_load((rx, rz)).expect("region should load");
        assert_eq!(region.region.get_x_coord(), rx);
        assert_eq!(region.region.get_z_coord(), rz);
        assert!(region.chunks.is_some(), "load_chunks should have populated chunks");

        // Loading again returns the same decoded data (from cache, not a
        // fresh parse) — spot-check chunk (0, 0) is stable across accesses.
        let first_chunk = region.get_chunk(0, 0).is_some();
        let region_again = cache.get_or_load((rx, rz)).expect("cached region should load");
        assert_eq!(region_again.get_chunk(0, 0).is_some(), first_chunk);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn get_or_load_errors_for_a_region_the_save_does_not_have() {
        let meta = real_save_meta();
        // Far outside any real save's region grid.
        let missing = (100_000, 100_000);
        assert!(!meta.has_region(missing.0, missing.1));

        let mut cache = RegionCache::new(meta, 4);
        assert!(cache.get_or_load(missing).is_err());
    }

    /// Ticket 008: unlike a region the save's metadata never listed (above),
    /// this simulates a region the save *claims* to have but whose `.mca`
    /// file is missing/corrupt on disk — `get_or_load` should fail cleanly
    /// (not panic), and a repeat request for the same coordinate should keep
    /// failing cleanly too rather than somehow succeeding once the failure
    /// is remembered in `failed`.
    #[test]
    fn get_or_load_remembers_a_load_failure_instead_of_retrying_forever() {
        let meta = SaveMeta {
            name: "broken".to_string(),
            path: std::path::PathBuf::from("does-not-exist-on-disk"),
            regions: vec![(0, 0)],
        };
        let mut cache = RegionCache::new(meta, 4);

        assert!(cache.get_or_load((0, 0)).is_err());
        assert!(cache.get_or_load((0, 0)).is_err());
        assert_eq!(cache.len(), 0, "a failed region must never end up cached as loaded");
    }

    #[test]
    fn capacity_exceeding_accesses_evict_the_least_recently_used_region() {
        let meta = real_save_meta();
        assert!(
            meta.regions.len() >= 3,
            "test save needs at least 3 regions"
        );
        let [a, b, c] = [meta.regions[0], meta.regions[1], meta.regions[2]];

        let mut cache = RegionCache::new(meta, 2);
        cache.get_or_load(a).unwrap();
        cache.get_or_load(b).unwrap();
        // Touch `a` again so `b` becomes the least-recently-used one.
        cache.get_or_load(a).unwrap();
        // Loading a third region should evict `b`, not `a`.
        cache.get_or_load(c).unwrap();

        assert_eq!(cache.len(), 2);
        assert!(cache.entries.contains_key(&a), "a was touched most recently, should survive");
        assert!(!cache.entries.contains_key(&b), "b was least-recently-used, should be evicted");
        assert!(cache.entries.contains_key(&c));
    }

    #[test]
    fn eviction_frees_the_decoded_chunk_data() {
        let meta = real_save_meta();
        let (a, b, c) = (meta.regions[0], meta.regions[1], meta.regions[2]);

        let mut cache = RegionCache::new(meta, 1);
        cache.get_or_load(a).unwrap();
        assert!(cache.entries.get(&a).unwrap().chunks.is_some());

        // Capacity 1: loading b evicts a's entry (and its chunk data)
        // outright, not just the LRU bookkeeping.
        cache.get_or_load(b).unwrap();
        assert!(!cache.entries.contains_key(&a));
        assert_eq!(cache.len(), 1);

        cache.get_or_load(c).unwrap();
        assert!(!cache.entries.contains_key(&b));
        assert_eq!(cache.len(), 1);
    }

    /// Not a correctness test — measures single-region load+decode time
    /// against the real save, per the ticket's "Watch out" note, so 005-c
    /// knows whether a whole region is a reasonable unit of background
    /// work. Run with `cargo test region_cache -- --nocapture` to see the
    /// timing; the ticket asks for the number to be written down once
    /// measured (see the ticket file / commit message).
    #[test]
    fn measure_single_region_load_time() {
        let meta = real_save_meta();
        let (rx, rz) = meta.regions[0];
        let mut region: ChunkRegion = Region::new(rx, rz, meta.get_region_path(rx, rz)).into();

        let start = Instant::now();
        region.load_chunks().expect("region should load");
        let elapsed = start.elapsed();

        let chunk_count = region
            .chunks
            .as_ref()
            .map(|chunks| chunks.iter().filter(|c| c.is_some()).count())
            .unwrap_or(0);
        println!(
            "region ({rx}, {rz}): loaded {chunk_count} chunks in {elapsed:?}"
        );
    }
}
