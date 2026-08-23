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
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mc_anvil::chunkregion::ChunkRegion;
use mc_anvil::region::{Region, REGION_WIDTH_IN_CHUNKS};
use mc_anvil::{MCLoadError, SaveMeta};

use crate::world::warn::WarnLedger;

/// How long a region that failed to load stays remembered as failed before
/// [`RegionCache::load_if_absent`] will try it again (ticket 062 follow-up).
/// A region's own file can fail transiently — Windows Explorer or an
/// antivirus scanner briefly holding a `.mca` file open, a sharing
/// violation while another process touches the save — and without a
/// cooldown that one bad moment permanently blacklisted every chunk in the
/// region for the rest of the session, no matter how long the user waited
/// or how many times the streaming diff re-ran. A save that genuinely
/// doesn't have the region at all (`SaveMeta::has_region` false) is a
/// different, permanent case and never touches `failed` at all — see
/// [`RegionCache::load_if_absent`].
const FAILED_RETRY_COOLDOWN: Duration = Duration::from_secs(10);

/// Region coordinates already reported as unreadable (ticket 081) — the
/// cooldown above governs *retries*, this governs the console line.
static UNREADABLE_REGION: WarnLedger = WarnLedger::new();

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
    /// Regions whose `load_chunks` has failed, keyed to when the failure
    /// happened (truncated or otherwise corrupt `.mca`, ticket 008; or a
    /// transient I/O failure, ticket 062 follow-up) — remembered so a bad
    /// region fails fast, without re-touching disk or re-logging, for
    /// [`FAILED_RETRY_COOLDOWN`] after the failure, then gets one more
    /// attempt. Deliberately separate from `entries`, which only ever holds
    /// successfully loaded regions.
    failed: HashMap<(i32, i32), Instant>,
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
            failed: HashMap::new(),
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
    /// fails (missing/corrupt file, unsupported compression, a transient I/O
    /// error, ...) — the latter is remembered in `failed` (ticket 008) for
    /// [`FAILED_RETRY_COOLDOWN`] so a bad region only gets logged, and its
    /// file re-read, once per cooldown window rather than on every chunk
    /// that lands inside it, but a region that failed only transiently
    /// (ticket 062 follow-up) isn't blacklisted for the rest of the session.
    pub fn get_or_load(&mut self, region_coord: (i32, i32)) -> Result<&ChunkRegion, MCLoadError> {
        self.load_if_absent(region_coord)?;
        Ok(self
            .entries
            .get(&region_coord)
            .expect("just inserted or touched above"))
    }

    /// [`get_or_load`](Self::get_or_load), mutably: the write path's way in
    /// (ticket 032), since [`crate::edit`] needs `&mut ChunkRegion` to apply
    /// an edit and the cache is the only thing that knows where a region file
    /// lives.
    ///
    /// The returned region stays resident until it is saved: `evict_lru`
    /// refuses to drop a dirty one, so an edit can't be lost to ordinary
    /// streaming traffic between being applied and being written.
    pub fn get_or_load_mut(
        &mut self,
        region_coord: (i32, i32),
    ) -> Result<&mut ChunkRegion, MCLoadError> {
        self.load_if_absent(region_coord)?;
        Ok(self
            .entries
            .get_mut(&region_coord)
            .expect("just inserted or touched above"))
    }

    /// Loads `region_coord` if it isn't resident and marks it most-recently-
    /// used. Shared by [`get_or_load`](Self::get_or_load) and
    /// [`get_or_load_mut`](Self::get_or_load_mut) so there's one copy of the
    /// load path rather than two that can drift.
    fn load_if_absent(&mut self, region_coord: (i32, i32)) -> Result<(), MCLoadError> {
        if let Some(&failed_at) = self.failed.get(&region_coord) {
            if failed_at.elapsed() < FAILED_RETRY_COOLDOWN {
                return Err(MCLoadError::PathNotFoundError);
            }
            // Cooldown elapsed — worth one more attempt below rather than
            // staying blacklisted forever over what may have been transient.
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
                self.failed.insert(region_coord, Instant::now());
                // Logged once per region for the whole run (ticket 081).
                // The retry below is unchanged — but a region that stays
                // unreadable would otherwise print this line afresh every
                // time `FAILED_RETRY_COOLDOWN` expires, forever.
                if UNREADABLE_REGION.first_time(&format!("{rx},{rz}")) {
                    println!(
                        "block_viewer: skipping region ({rx}, {rz}) — failed to load {path_display}: {err}"
                    );
                }
                return Err(err);
            }
            // A cooldown-expired retry that succeeds clears the old failure
            // record — otherwise a region that failed once and then loaded
            // fine would still short-circuit-fail again the moment the next
            // cooldown window happened to be checked mid-eviction.
            self.failed.remove(&region_coord);
            self.entries.insert(region_coord, region);
        }

        self.touch(region_coord);
        Ok(())
    }

    /// Whether the save this cache reads has a region file at `region_coord`
    /// at all — ungenerated terrain rather than a load failure. The write path
    /// (ticket 032) reports the two differently: one is "you can't build
    /// there", the other is "something is wrong with this save".
    pub fn has_region(&self, (rx, rz): (i32, i32)) -> bool {
        self.save_meta.has_region(rx, rz)
    }

    /// Whether a region is currently loaded, without touching its LRU
    /// position — the read `discard` and the eviction guard are observed
    /// through, and what W6 asks before deciding whether saving a region needs
    /// it loaded first.
    pub fn is_resident(&self, region_coord: (i32, i32)) -> bool {
        self.entries.contains_key(&region_coord)
    }

    /// Every resident region with unsaved changes, in no particular order.
    ///
    /// W6 saves these; ticket 032's routing refuses to start a new transaction
    /// while one of its target regions is in here, and the city panel (roadmap
    /// G2) shows them so the user can tell whether what they see has reached
    /// the world.
    pub fn dirty_regions(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.entries
            .iter()
            .filter(|(_, region)| region.is_dirty())
            .map(|(coord, _)| *coord)
    }

    /// Drops a resident region **including any unsaved changes in it**, so the
    /// next access re-reads it from disk.
    ///
    /// This is the write path's rollback (ticket 032): `edit::apply` never
    /// saves, so throwing the in-memory region away is a complete and exact
    /// undo of everything applied to it since it was loaded — which is also
    /// why it must only be called on a region whose *only* unsaved changes are
    /// the ones being rolled back.
    ///
    /// Returns whether a region was actually resident.
    pub fn discard(&mut self, region_coord: (i32, i32)) -> bool {
        self.order.retain(|&k| k != region_coord);
        self.entries.remove(&region_coord).is_some()
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
    ///
    /// **Skips regions with unsaved changes** (ticket 032): `ChunkRegion`'s
    /// own `is_dirty` doc comment names this cache as the reason it exists,
    /// and evicting an edited region drops the edit silently — a building near
    /// a region corner spans four files, and ordinary streaming traffic in the
    /// same frame is enough to push one of them out. If every resident region
    /// is dirty nothing is evicted and `capacity` is exceeded; the overshoot
    /// is bounded by how much was edited before the next save, which is a far
    /// better failure than losing the edit.
    fn evict_lru(&mut self) {
        let lru = self.order.iter().position(|coord| {
            self.entries
                .get(coord)
                .is_none_or(|region| !region.is_dirty())
        });
        let Some(index) = lru else {
            return;
        };
        let coord = self.order.remove(index);
        self.entries.remove(&coord);
    }

    /// Test-only: back-dates `region_coord`'s failure record to just past
    /// [`FAILED_RETRY_COOLDOWN`], so a cooldown-expiry test doesn't need to
    /// actually sleep for it. No-op if `region_coord` isn't currently
    /// recorded as failed.
    #[cfg(test)]
    fn force_failed_stale(&mut self, region_coord: (i32, i32)) {
        if let Some(at) = self.failed.get_mut(&region_coord) {
            *at = Instant::now() - FAILED_RETRY_COOLDOWN - Duration::from_secs(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This repo assumes a real Minecraft saves directory is reachable on
    /// the dev machine (see `lib.rs::load_real_save`); these tests follow
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
    /// (not panic), and a repeat request for the same coordinate within the
    /// cooldown window should keep failing cleanly too rather than somehow
    /// succeeding once the failure is remembered in `failed`.
    #[test]
    fn get_or_load_remembers_a_load_failure_within_the_cooldown_window() {
        let meta = SaveMeta {
            name: "broken".to_string(),
            path: std::path::PathBuf::from("does-not-exist-on-disk"),
            region_dir: std::path::PathBuf::from("does-not-exist-on-disk/region"),
            regions: vec![(0, 0)],
        };
        let mut cache = RegionCache::new(meta, 4);

        assert!(cache.get_or_load((0, 0)).is_err());
        assert!(cache.get_or_load((0, 0)).is_err());
        assert_eq!(cache.len(), 0, "a failed region must never end up cached as loaded");
    }

    /// Ticket 062 follow-up: a region that failed only transiently (a
    /// sharing violation, an antivirus scan mid-read, ...) must not stay
    /// blacklisted for the rest of the session — once [`FAILED_RETRY_COOLDOWN`]
    /// has passed, the next request gets a real retry, not just the
    /// remembered failure replayed. Backdates the failure timestamp with the
    /// test-only [`RegionCache::force_failed_stale`] rather than actually
    /// sleeping for the cooldown.
    #[test]
    fn a_failure_older_than_the_cooldown_gets_retried() {
        let meta = SaveMeta {
            name: "broken".to_string(),
            path: std::path::PathBuf::from("does-not-exist-on-disk"),
            region_dir: std::path::PathBuf::from("does-not-exist-on-disk/region"),
            regions: vec![(0, 0)],
        };
        let mut cache = RegionCache::new(meta, 4);

        assert!(cache.get_or_load((0, 0)).is_err());
        let first_failure_at = *cache.failed.get(&(0, 0)).expect("recorded as failed");

        cache.force_failed_stale((0, 0));
        let staled_at = *cache.failed.get(&(0, 0)).expect("still recorded as failed");
        assert!(staled_at < first_failure_at, "test setup should have backdated the failure");

        // The file still doesn't exist, so this still errors — the point is
        // *whether* a fresh attempt happened, not whether it succeeds: a
        // fresh attempt refreshes the failure timestamp, a short-circuit on
        // the old blacklist entry wouldn't touch it at all.
        assert!(cache.get_or_load((0, 0)).is_err());
        let second_failure_at = *cache.failed.get(&(0, 0)).expect("recorded as failed again");
        assert!(
            second_failure_at > staled_at,
            "a cooldown-expired failure should be retried, not just replayed from the old record"
        );
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
