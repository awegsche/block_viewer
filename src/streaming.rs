//! Render-distance chunk streaming (ticket 005-a): computes which chunks
//! *should* be loaded around the camera and diffs that against what's
//! currently loaded, publishing the delta as [`PendingChunkWork`] for later
//! tickets (005-b onward) to actually load/spawn/unload.
//!
//! This slice is pure logic — no I/O, no async, nothing spawned yet. The
//! "currently loaded" set is read from [`DecodedWorld`], the only notion of
//! "loaded chunks" that exists before 005-b/005-c introduce real streaming
//! state.
//!
//! Three radii, nested (ticket 122 on top of 070):
//!
//! - [`RenderDistance`] is what the player *sees*: `camera`'s fog goes
//!   opaque at exactly this many chunks out.
//! - [`ChunkPreload`] extends what gets *loaded* a little past that, so the
//!   streaming frontier — chunks popping in, seams closing — always sits
//!   inside opaque fog rather than in plain view. See [`load_radius`].
//! - [`ChunkRetention`] says how reluctantly to give a loaded column back up
//!   again — a margin of chunks kept past the load edge, plus a grace period
//!   a column has to spend outside even that before it's dropped. See
//!   [`ChunkRetention`] for why both.
//!
//! The loaded set is a **disc**, not a square ([`desired_chunks`]): the fog
//! is radial, so a square's corners were always fully fogged — decoded,
//! meshed and drawn for nothing.
//!
//! The diff recomputes only when the camera crosses a chunk boundary, not
//! every frame — a render distance of 12 is at most ~625 chunks, cheap to
//! diff, but there's no reason to redo the same `HashSet` diff dozens of
//! times a second while the camera sits still. 005-c/005-d can assume
//! [`PendingChunkWork`] only changes on frames where the camera's chunk
//! coordinate actually changed.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::{camera, world, DecodedWorld};

/// How far (in chunks, not blocks) the player can *see*: `camera`'s fog is
/// fully opaque at `render_distance * 16` blocks, in every direction. What
/// gets loaded is a little more than this — see [`ChunkPreload`] — so the
/// terrain never visibly ends short of the fog. Default sits in the 8-12
/// range the parent ticket (005) documents.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderDistance(pub u32);

impl Default for RenderDistance {
    fn default() -> Self {
        Self(10)
    }
}

/// Extra chunk radius *loaded* past [`RenderDistance`] (ticket 122), so the
/// streaming frontier — chunks popping in, 005-f's seams closing, columns
/// unloading — happens inside opaque fog instead of in plain view.
///
/// 2 is the minimum that actually hides it: a point `d` blocks from the
/// camera lies in a chunk at most `d / 16 + sqrt(2)` chunks (Euclidean, in
/// chunk offsets) from the camera's own chunk, so a disc of radius
/// `render_distance + 2` contains every chunk that has *any* block inside
/// the fog end. 1 leaves diagonal gaps; more than 2 buys nothing visible,
/// only earlier loading for a fast-moving camera, at a quadratic memory
/// cost.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ChunkPreload(pub u32);

impl Default for ChunkPreload {
    fn default() -> Self {
        Self(2)
    }
}

/// The radius (in chunks) columns are *loaded* out to: what the player can
/// see plus the preload ring that hides the frontier. Everything that sizes
/// itself to "what's loaded" — the retain radius, the in-flight cancel
/// check, the region cache — takes this, not [`RenderDistance`] directly.
pub fn load_radius(render_distance: &RenderDistance, preload: &ChunkPreload) -> u32 {
    render_distance.0 + preload.0
}

/// How reluctant [`update_pending_chunk_work`] is to give a loaded column
/// back up again (ticket 070). [`load_radius`] decides what gets *loaded*;
/// everything here only decides how long what's already loaded survives
/// leaving that disc. Nothing below ever causes a load.
///
/// Two knobs rather than one because they answer two different movements:
///
/// - `margin` (hysteresis) handles the camera **jittering across a chunk
///   boundary**. Columns stay loaded out to `load_radius + margin`, so
///   drifting one chunk out and back never even produces a candidate — no
///   timer involved, no reload possible.
/// - `grace` handles the camera **going somewhere and coming back**. Past
///   the retain disc a column becomes a *lingering* candidate carrying
///   the time it left rather than an unload; re-entering the retain disc
///   within `grace` costs nothing at all, since its mesh and its
///   [`crate::world::ChunkColumn`] were never touched.
///
/// `max_lingering` is the bound the grace needs: flying in a straight line
/// leaves a trail of candidates that are all still inside their grace, and
/// that trail is otherwise limited only by how long the player flies. Over
/// the cap, the farthest candidates unload immediately. Each chunk crossing
/// sheds roughly one retain-disc diameter of columns (~40 at the
/// citybuilder's radius 20), so the default 1024 is about 25 crossings —
/// 400 blocks of straight flight — before the trail starts being cut; the
/// old 256 tripped after ~7, which is what "fly away a bit and come back"
/// reloading everything looked like (ticket 122).
///
/// The retain disc is `~pi * (load_radius + margin)^2` columns — each of
/// which holds a decoded column *and* a GPU mesh — so `margin` costs
/// memory quadratically. The default 2 is a measured-cheap value, not a
/// placeholder to raise casually.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ChunkRetention {
    /// Extra chunk radius kept loaded past [`load_radius`].
    pub margin: u32,
    /// How long a column must stay outside the retain disc before it's
    /// actually unloaded.
    pub grace: Duration,
    /// Cap on columns lingering outside the retain disc; over it, the
    /// farthest ones go regardless of `grace`.
    pub max_lingering: usize,
}

impl Default for ChunkRetention {
    fn default() -> Self {
        Self {
            margin: 2,
            grace: Duration::from_secs(90),
            max_lingering: 1024,
        }
    }
}

impl ChunkRetention {
    /// The radius columns are *kept* out to, as opposed to the
    /// [`load_radius`] they're loaded out to.
    pub fn retain_radius(&self, load_radius: u32) -> u32 {
        load_radius + self.margin
    }
}

/// Loaded columns currently outside the retain square, each with the
/// [`Time::elapsed`] value at which it left (ticket 070). A column in here
/// is *not* unloaded yet — see [`ChunkRetention`] — and leaves the map
/// either by coming back inside the retain square or by actually being
/// unloaded, both of which [`update_pending_chunk_work`] notices by
/// rebuilding against `DecodedWorld` rather than being told.
///
/// Entries deliberately survive being emitted into
/// [`PendingChunkWork::to_unload`]: that list isn't drained by its consumer
/// (`unload::unload_chunks` reprocesses it idempotently every frame), so a
/// recompute landing before the unload system next runs would otherwise
/// overwrite the list and lose the unload with nothing left holding the
/// coordinate. Re-emitting an already-unloaded coordinate is a cheap no-op;
/// silently keeping a column loaded forever is not.
#[derive(Resource, Debug, Default)]
pub(crate) struct LingeringChunks(HashMap<(i32, i32), Duration>);

impl LingeringChunks {
    /// How many loaded columns are only still loaded because of the grace
    /// period — the part of the viewer's "Loaded chunks" count that is
    /// retention rather than render distance (ticket 070).
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

/// The chunk coordinate the camera occupied as of the last recompute, so
/// [`update_pending_chunk_work`] can skip re-diffing every frame and only
/// act when the camera actually crosses into a new chunk.
///
/// `pub(crate)` (rather than private) so the UI layer (ticket 007) can
/// force a fresh recompute by resetting this to its `Default` — e.g. after
/// switching saves, where the camera may land in the same chunk coordinate
/// it started in (a fresh save's streaming state still needs rebuilding
/// even though the camera didn't "enter" a new chunk). The field is
/// `pub(crate)` too: `Some` is how `city::loading` (ticket 124) tells that
/// the diff has been published at least once.
#[derive(Resource, Debug, Default)]
pub(crate) struct LastCameraChunk(pub(crate) Option<(i32, i32)>);

/// The chunk load/unload delta computed by [`update_pending_chunk_work`].
/// Later tickets (005-b onward) drain this to actually load/spawn/unload
/// chunks; this ticket only ever writes it, never acts on it.
#[derive(Resource, Debug, Default)]
pub struct PendingChunkWork {
    pub to_load: Vec<(i32, i32)>,
    pub to_unload: Vec<(i32, i32)>,
}

/// Adds [`RenderDistance`], [`ChunkPreload`], [`ChunkRetention`] and
/// [`PendingChunkWork`], and the system that keeps the latter up to date
/// with the camera's position.
pub struct ChunkStreamingPlugin;

impl Plugin for ChunkStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderDistance>()
            .init_resource::<ChunkPreload>()
            .init_resource::<ChunkRetention>()
            .init_resource::<LastCameraChunk>()
            .init_resource::<LingeringChunks>()
            .init_resource::<PendingChunkWork>()
            .add_systems(Update, update_pending_chunk_work);
    }
}

/// The disc of chunk coordinates within `radius` chunks (Euclidean, on
/// chunk offsets) of `center`, inclusive — `~pi * radius^2` chunks, the
/// `(2*radius+1)^2` square minus its corners. The fog is radial (ticket
/// 122), so the corners were the ~quarter of a square that was always fully
/// fogged: decoded, meshed and drawn for nothing. Same square iteration as
/// before, with one comparison per cell to drop what's outside.
pub fn desired_chunks(center: (i32, i32), radius: u32) -> HashSet<(i32, i32)> {
    let r = radius as i32;
    let r_sq = r * r;
    // A disc fills ~pi/4 of its bounding square; rounding up is fine.
    let side = (2 * r + 1) as usize;
    let mut set = HashSet::with_capacity(side * side * 4 / 5);
    for dx in -r..=r {
        for dz in -r..=r {
            if dx * dx + dz * dz <= r_sq {
                set.insert((center.0 + dx, center.1 + dz));
            }
        }
    }
    set
}

/// Splits `loaded` against the two discs (ticket 070): chunks that need
/// **loading** (in `desired`, not yet `loaded`) and chunks that have left
/// **retention** (`loaded`, but outside `retain`).
///
/// The second half is deliberately not "to unload" — a column outside
/// `retain` is a *candidate*, which [`ChunkRetention::grace`] then holds on
/// to for a while longer. `retain` is expected to be a superset of
/// `desired` (see [`ChunkRetention::retain_radius`]), so a column that is
/// loaded and still wanted lands in neither list.
///
/// `to_load` comes back **nearest-first** (ticket 122). `chunk_pipeline`
/// dispatches it in order, and its tasks serialise on the block-registry
/// lock (see that module's "Send boundary" docs), so dispatch order is
/// effectively completion order — sorting here is what makes the chunk in
/// front of the camera land before a corner of the disc behind it.
/// `HashSet` iteration order gave no such guarantee.
pub fn diff_chunks(
    center: (i32, i32),
    desired: &HashSet<(i32, i32)>,
    retain: &HashSet<(i32, i32)>,
    loaded: &HashSet<(i32, i32)>,
) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
    let mut to_load: Vec<(i32, i32)> = desired.difference(loaded).copied().collect();
    to_load.sort_unstable_by_key(|&coord| (chunk_distance_sq(center, coord), coord));
    let left_retention = loaded.difference(retain).copied().collect();
    (to_load, left_retention)
}

/// Squared Euclidean distance in chunks, the metric that matches the disc
/// [`desired_chunks`] builds: it's exactly the radius at which `other` first
/// enters `center`'s disc, which is what makes it the right ordering both
/// for loading the nearest columns first and for evicting the farthest
/// lingering ones first. Squared so it stays integral — only ever compared,
/// never measured.
fn chunk_distance_sq(center: (i32, i32), other: (i32, i32)) -> i32 {
    let dx = other.0 - center.0;
    let dz = other.1 - center.1;
    dx * dx + dz * dz
}

/// Which lingering columns actually unload this tick: the ones whose grace
/// has run out, plus — when `lingering` is over
/// [`ChunkRetention::max_lingering`] — enough of the farthest ones from
/// `center` to bring it back under the cap (ticket 070).
///
/// Split out as a pure function so the grace and the cap can be tested
/// without a camera, a clock or a world.
fn expired_lingering(
    lingering: &HashMap<(i32, i32), Duration>,
    center: (i32, i32),
    now: Duration,
    retention: &ChunkRetention,
) -> Vec<(i32, i32)> {
    let mut expired: Vec<(i32, i32)> = lingering
        .iter()
        .filter(|&(_, &left_at)| now.saturating_sub(left_at) >= retention.grace)
        .map(|(&coord, _)| coord)
        .collect();

    let over_cap = lingering.len().saturating_sub(retention.max_lingering);
    if over_cap > expired.len() {
        // Farthest first, so the cap sheds the columns the camera is least
        // likely to come back to. `sort_unstable_by_key` on the negated
        // distance rather than a reversed comparator keeps the key integral.
        let mut by_distance: Vec<(i32, i32)> = lingering.keys().copied().collect();
        by_distance.sort_unstable_by_key(|&coord| {
            (-chunk_distance_sq(center, coord), coord.0, coord.1)
        });
        let already: HashSet<(i32, i32)> = expired.iter().copied().collect();
        expired.extend(
            by_distance
                .into_iter()
                .filter(|coord| !already.contains(coord))
                .take(over_cap - expired.len()),
        );
    }

    expired
}

/// Converts a Bevy-space translation into the Minecraft chunk coordinate it
/// falls in. `bevy.x = mc.x`, `bevy.z = -mc.z` (see `world::mesh` docs) is
/// the only axis that flips.
///
/// `pub(crate)` (rather than private) so [`crate::unload`] (005-d) can
/// recompute the same desired set this module diffs against, to catch
/// in-flight loads (005-c) for coordinates the camera has since left — see
/// that module's docs for why `to_unload` alone can't see those.
pub(crate) fn camera_chunk_coord(translation: Vec3) -> (i32, i32) {
    let size = world::SECTION_SIZE as f32;
    let mc_x = translation.x;
    let mc_z = -translation.z;
    ((mc_x / size).floor() as i32, (mc_z / size).floor() as i32)
}

/// How often [`update_pending_chunk_work`] force-recomputes the load/unload
/// diff even when the camera hasn't crossed a chunk boundary (ticket 062
/// follow-up). The chunk-crossing/render-distance-change triggers below
/// assume every coordinate that ever lands in [`PendingChunkWork::to_load`]
/// eventually makes it into [`DecodedWorld`] on its own — true for ordinary
/// streaming, but not for a coordinate whose only region failed to load (see
/// [`crate::region_cache::RegionCache`]'s retry cooldown) or any other way a
/// load could silently drop a coordinate without ever loading it, with
/// nothing left to notice and retry it. Re-running the same diff on a timer
/// regardless of camera movement re-queues anything still desired but
/// missing from [`DecodedWorld`] — cheap (the same O(desired) `HashSet` diff
/// `unload`'s own cancel system already redoes every frame), and
/// self-healing rather than depending on the camera happening to cross
/// another chunk boundary to notice.
const FORCE_RECOMPUTE_INTERVAL: Duration = Duration::from_secs(2);

/// Recomputes [`PendingChunkWork`] whenever the camera's chunk coordinate
/// changes, whenever [`RenderDistance`] itself changes (ticket 007's
/// status-panel slider) — the camera can sit in the same chunk while the
/// desired radius around it grows or shrinks, and that needs the same
/// re-diff a chunk crossing does — or every [`FORCE_RECOMPUTE_INTERVAL`]
/// regardless of either, as a self-healing retry (see that constant's
/// docs). Otherwise does not act on the delta — loading/spawning/unloading
/// is 005-b onward.
///
/// Unloads run through [`ChunkRetention`] rather than straight off the diff
/// (ticket 070): what leaves the retain square becomes a
/// [`LingeringChunks`] entry, and only a run-out grace (or the cap on how
/// many may linger) turns that into an actual unload. Both the periodic
/// retry and ordinary chunk crossings drive this, so the grace is checked
/// at least every [`FORCE_RECOMPUTE_INTERVAL`] — coarse enough to be free,
/// fine enough for a grace measured in seconds.
fn update_pending_chunk_work(
    camera: Query<&Transform, With<camera::CameraRig>>,
    render_distance: Res<RenderDistance>,
    preload: Res<ChunkPreload>,
    retention: Res<ChunkRetention>,
    decoded_world: Res<DecodedWorld>,
    mut last_chunk: ResMut<LastCameraChunk>,
    mut lingering: ResMut<LingeringChunks>,
    mut pending: ResMut<PendingChunkWork>,
    time: Res<Time>,
    mut since_last_recompute: Local<Duration>,
) {
    let Ok(transform) = camera.get_single() else {
        return;
    };

    let center = camera_chunk_coord(transform.translation);

    *since_last_recompute += time.delta();
    let force_recompute = *since_last_recompute >= FORCE_RECOMPUTE_INTERVAL;
    if force_recompute {
        *since_last_recompute = Duration::ZERO;
    }

    if last_chunk.0 == Some(center)
        && !render_distance.is_changed()
        && !preload.is_changed()
        && !retention.is_changed()
        && !force_recompute
    {
        return; // Same chunk, same settings, not due for a retry yet.
    }
    last_chunk.0 = Some(center);

    let load_radius = load_radius(&render_distance, &preload);
    let desired = desired_chunks(center, load_radius);
    let retain = desired_chunks(center, retention.retain_radius(load_radius));
    let loaded: HashSet<(i32, i32)> = decoded_world.columns.keys().copied().collect();
    let (to_load, left_retention) = diff_chunks(center, &desired, &retain, &loaded);

    // Rebuild the lingering set against reality first: an entry that came
    // back inside the retain square, or that has actually been unloaded
    // since, is simply gone — no unload event to handle, and a column that
    // comes back a third time starts its grace over from now.
    let now = time.elapsed();
    lingering
        .0
        .retain(|coord, _| loaded.contains(coord) && !retain.contains(coord));
    for coord in left_retention {
        // `or_insert`, not `insert`: a column that left the retain square
        // several recomputes ago keeps the time it *first* left, which is
        // what the grace is measured from.
        lingering.0.entry(coord).or_insert(now);
    }
    let to_unload = expired_lingering(&lingering.0, center, now, &retention);

    // Worth a console line only when there's an actual delta — the periodic
    // retry runs whether or not anything changed, and logging an empty diff
    // every couple of seconds would just be noise once streaming has caught
    // up (which is the common, steady-state case this runs in).
    if !to_load.is_empty() || !to_unload.is_empty() {
        println!(
            "Chunk streaming: camera entered chunk {:?} ({} to load, {} to unload, {} lingering)",
            center,
            to_load.len(),
            to_unload.len(),
            lingering.0.len(),
        );
    }

    pending.to_load = to_load;
    pending.to_unload = to_unload;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desired_chunks_is_a_disc() {
        let set = desired_chunks((5, -3), 2);
        // A radius-2 disc: the centre, 4 at distance 1, 4 diagonals at
        // sqrt(2), 4 at distance 2 — 13 chunks. The square would be 25.
        assert_eq!(set.len(), 13);
        // The cardinal extremes are in.
        assert!(set.contains(&(7, -3)));
        assert!(set.contains(&(5, -5)));
        // The square's corners (offset (2, 2), distance 2.83) are out.
        assert!(!set.contains(&(3, -5)));
        assert!(!set.contains(&(7, -1)));
        // One chunk past the radius is out.
        assert!(!set.contains(&(8, -3)));
        assert!(!set.contains(&(5, -6)));
    }

    /// Ticket 122's reason for the disc: at a real radius it's about a
    /// quarter fewer chunks than the square, all of them from the corners
    /// the fog fully hid anyway. (441 at radius 12 — exactly the old
    /// radius-10 square, so the viewer's default costs what it did.)
    #[test]
    fn desired_chunks_disc_drops_about_a_quarter_of_the_square() {
        let radius = 12;
        let disc = desired_chunks((0, 0), radius).len();
        let square = (2 * radius as usize + 1).pow(2);
        assert_eq!(disc, 441);
        let ratio = disc as f64 / square as f64;
        assert!(
            (0.68..0.76).contains(&ratio),
            "disc {disc} / square {square} = {ratio:.3}"
        );
    }

    #[test]
    fn desired_chunks_radius_zero_is_just_the_center() {
        let set = desired_chunks((0, 0), 0);
        assert_eq!(set, HashSet::from([(0, 0)]));
    }

    /// The guarantee [`ChunkPreload`]'s docs make: with a preload of 2,
    /// every chunk that has any block within `render_distance * 16` blocks
    /// of the camera is inside the load disc. Checked by brute force over
    /// a fine grid of camera positions inside its chunk and of block
    /// positions around it, rather than trusting the algebra.
    #[test]
    fn preload_of_two_covers_every_block_inside_the_fog_end() {
        let render_distance = RenderDistance(5);
        let radius = load_radius(&render_distance, &ChunkPreload(2));
        let fog_end = render_distance.0 as f32 * world::SECTION_SIZE as f32;
        let disc = desired_chunks((0, 0), radius);

        // Camera anywhere inside chunk (0, 0), block anywhere in a box
        // that comfortably contains the fog circle.
        for cam_x in [0.0_f32, 3.7, 8.0, 15.99] {
            for cam_z in [0.0_f32, 5.2, 15.99] {
                let reach = fog_end + 16.0;
                let mut bx = -reach;
                while bx <= reach + 16.0 {
                    let mut bz = -reach;
                    while bz <= reach + 16.0 {
                        let dx = bx - cam_x;
                        let dz = bz - cam_z;
                        if dx * dx + dz * dz <= fog_end * fog_end {
                            let chunk = camera_chunk_coord(Vec3::new(bx, 0.0, -bz));
                            assert!(
                                disc.contains(&chunk),
                                "block ({bx}, {bz}) seen from ({cam_x}, {cam_z}) is in chunk {chunk:?}, outside the load disc"
                            );
                        }
                        bz += 2.0;
                    }
                    bx += 2.0;
                }
            }
        }
    }

    #[test]
    fn diff_chunks_buckets_entering_leaving_and_unchanged_chunks() {
        let desired = HashSet::from([(0, 0), (1, 0), (2, 0)]);
        // The retain disc is the desired one plus a ring holding (3, 0).
        let retain = HashSet::from([(0, 0), (1, 0), (2, 0), (3, 0)]);
        let loaded = HashSet::from([(0, 0), (1, 0), (3, 0), (5, 5)]);

        let (to_load, mut left_retention) = diff_chunks((0, 0), &desired, &retain, &loaded);
        left_retention.sort();

        // (2, 0) is newly desired: entering.
        assert_eq!(to_load, vec![(2, 0)]);
        // (5, 5) is loaded and outside retention: a candidate. (3, 0) is
        // outside the *load* disc but inside retention, so it is not.
        assert_eq!(left_retention, vec![(5, 5)]);
        // (0, 0) and (1, 0) are loaded and wanted: neither list.
        assert!(!to_load.contains(&(0, 0)));
        assert!(!left_retention.contains(&(0, 0)));
    }

    /// Ticket 122: what `chunk_pipeline` dispatches first is what finishes
    /// first, so the list it drains has to be nearest-first.
    #[test]
    fn diff_chunks_orders_loads_nearest_first() {
        let desired = HashSet::from([(4, 0), (0, 1), (3, 3), (-1, 0), (0, 0)]);
        let loaded = HashSet::new();

        let (to_load, _) = diff_chunks((0, 0), &desired, &desired, &loaded);
        // (0,0) at 0, (-1,0) and (0,1) at 1 (tie broken by coordinate),
        // (4,0) at 16, (3,3) at 18.
        assert_eq!(to_load, vec![(0, 0), (-1, 0), (0, 1), (4, 0), (3, 3)]);
    }

    #[test]
    fn diff_chunks_empty_when_sets_match() {
        let set = HashSet::from([(0, 0), (1, 1)]);
        let (to_load, left_retention) = diff_chunks((0, 0), &set, &set, &set);
        assert!(to_load.is_empty());
        assert!(left_retention.is_empty());
    }

    /// Ticket 070: the hysteresis margin is what keeps a camera nudging
    /// across a chunk boundary from paying a reload — a column one chunk
    /// outside the load disc is still inside the retain disc, so it never
    /// becomes a candidate in the first place. Ticket 122: both sit on top
    /// of the *load* radius (render distance plus preload), not the render
    /// distance itself.
    #[test]
    fn retain_radius_is_load_radius_plus_margin() {
        let retention = ChunkRetention {
            margin: 2,
            ..ChunkRetention::default()
        };
        let load_radius = load_radius(&RenderDistance(10), &ChunkPreload(2));
        assert_eq!(load_radius, 12);
        assert_eq!(retention.retain_radius(load_radius), 14);

        let desired = desired_chunks((0, 0), load_radius);
        let retain = desired_chunks((0, 0), retention.retain_radius(load_radius));
        let loaded = HashSet::from([(13, 0), (15, 0)]);

        let (_, left_retention) = diff_chunks((0, 0), &desired, &retain, &loaded);
        // (13, 0) is past the load radius but inside the margin: kept.
        assert_eq!(left_retention, vec![(15, 0)]);
    }

    #[test]
    fn lingering_chunks_only_unload_once_the_grace_has_run_out() {
        let retention = ChunkRetention {
            grace: Duration::from_secs(30),
            max_lingering: 256,
            margin: 2,
        };
        let lingering = HashMap::from([
            ((5, 0), Duration::from_secs(10)),  // left 25s ago
            ((6, 0), Duration::from_secs(4)),   // left 31s ago
        ]);

        let expired = expired_lingering(&lingering, (0, 0), Duration::from_secs(35), &retention);
        assert_eq!(expired, vec![(6, 0)]);

        // Ten seconds later the other one is over its grace too.
        let mut expired =
            expired_lingering(&lingering, (0, 0), Duration::from_secs(45), &retention);
        expired.sort();
        assert_eq!(expired, vec![(5, 0), (6, 0)]);
    }

    /// A camera moving in a straight line leaves a trail of columns that are
    /// all still inside their grace; the cap is what bounds it.
    #[test]
    fn over_the_cap_the_farthest_lingering_chunks_go_regardless_of_grace() {
        let retention = ChunkRetention {
            grace: Duration::from_secs(30),
            max_lingering: 2,
            margin: 2,
        };
        // Four columns, all one second old — nothing is near its grace.
        let now = Duration::from_secs(1);
        let lingering = HashMap::from([
            ((3, 0), now),
            ((9, 0), now),
            ((5, 0), now),
            ((7, 0), now),
        ]);

        let mut expired = expired_lingering(&lingering, (0, 0), now, &retention);
        expired.sort();
        // Two over the cap, so the two farthest from the camera go.
        assert_eq!(expired, vec![(7, 0), (9, 0)]);
    }

    /// The cap counts what the grace already took: if enough columns have
    /// timed out to get back under it, nothing extra is evicted early.
    #[test]
    fn the_cap_does_not_evict_beyond_what_the_grace_already_covers() {
        let retention = ChunkRetention {
            grace: Duration::from_secs(30),
            max_lingering: 1,
            margin: 2,
        };
        let lingering = HashMap::from([
            ((1, 0), Duration::from_secs(0)),   // over its grace
            ((2, 0), Duration::from_secs(0)),   // over its grace
            ((3, 0), Duration::from_secs(40)),  // fresh, and the farthest
        ]);

        let mut expired =
            expired_lingering(&lingering, (0, 0), Duration::from_secs(41), &retention);
        expired.sort();
        // Two expired brings 3 lingering down to 1, which is the cap — the
        // farthest column (3, 0) keeps its grace.
        assert_eq!(expired, vec![(1, 0), (2, 0)]);
    }

    #[test]
    fn chunk_distance_is_squared_euclidean() {
        assert_eq!(chunk_distance_sq((0, 0), (3, 1)), 10);
        assert_eq!(chunk_distance_sq((0, 0), (-4, 2)), 20);
        assert_eq!(chunk_distance_sq((2, 2), (2, 2)), 0);
    }

    #[test]
    fn camera_chunk_coord_matches_mesh_axis_convention() {
        // bevy.x = mc.x, bevy.z = -mc.z; chunk 16 blocks wide.
        assert_eq!(camera_chunk_coord(Vec3::new(0.0, 0.0, 0.0)), (0, 0));
        assert_eq!(camera_chunk_coord(Vec3::new(20.0, 0.0, 0.0)), (1, 0));
        // bevy.z = -20 => mc.z = 20 => chunk 1.
        assert_eq!(camera_chunk_coord(Vec3::new(0.0, 0.0, -20.0)), (0, 1));
        // Negative mc coordinates floor toward negative infinity.
        assert_eq!(camera_chunk_coord(Vec3::new(-1.0, 0.0, 0.0)), (-1, 0));
    }
}
