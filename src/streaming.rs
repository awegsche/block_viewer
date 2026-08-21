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
//! Loading and unloading are deliberately *not* the same square (ticket
//! 070): [`RenderDistance`] says what to load, [`ChunkRetention`] says how
//! reluctantly to give it back up again — a margin of chunks kept past the
//! load edge, plus a grace period a column has to spend outside even that
//! before it's dropped. See [`ChunkRetention`] for why both.
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

/// Chunk radius (in chunks, not blocks) to keep loaded around the camera.
/// Default sits in the 8-12 range `camera.rs`'s far-plane placeholder and
/// the parent ticket (005) both document.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderDistance(pub u32);

impl Default for RenderDistance {
    fn default() -> Self {
        Self(10)
    }
}

/// How reluctant [`update_pending_chunk_work`] is to give a loaded column
/// back up again (ticket 070). [`RenderDistance`] decides what gets
/// *loaded*; everything here only decides how long what's already loaded
/// survives leaving that square. Nothing below ever causes a load.
///
/// Two knobs rather than one because they answer two different movements:
///
/// - `margin` (hysteresis) handles the camera **jittering across a chunk
///   boundary**. Columns stay loaded out to `render_distance + margin`, so
///   drifting one chunk out and back never even produces a candidate — no
///   timer involved, no reload possible.
/// - `grace` handles the camera **going somewhere and coming back**. Past
///   the retain square a column becomes a *lingering* candidate carrying
///   the time it left rather than an unload; re-entering the retain square
///   within `grace` costs nothing at all, since its mesh and its
///   [`crate::world::ChunkColumn`] were never touched.
///
/// `max_lingering` is the bound the grace needs: flying in a straight line
/// leaves a trail of candidates that are all still inside their grace, and
/// that trail is otherwise limited only by how long the player flies. Over
/// the cap, the farthest candidates unload immediately.
///
/// The retain square is `(2 * (render_distance + margin) + 1)^2` columns —
/// each of which holds a decoded column *and* a GPU mesh — so `margin`
/// costs memory quadratically. The default 2 is a measured-cheap value,
/// not a placeholder to raise casually.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ChunkRetention {
    /// Extra chunk radius kept loaded past [`RenderDistance`].
    pub margin: u32,
    /// How long a column must stay outside the retain square before it's
    /// actually unloaded.
    pub grace: Duration,
    /// Cap on columns lingering outside the retain square; over it, the
    /// farthest ones go regardless of `grace`.
    pub max_lingering: usize,
}

impl Default for ChunkRetention {
    fn default() -> Self {
        Self {
            margin: 2,
            grace: Duration::from_secs(30),
            max_lingering: 256,
        }
    }
}

impl ChunkRetention {
    /// The radius columns are *kept* out to, as opposed to the
    /// `render_distance` radius they're loaded out to.
    pub fn retain_radius(&self, render_distance: u32) -> u32 {
        render_distance + self.margin
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
/// even though the camera didn't "enter" a new chunk).
#[derive(Resource, Debug, Default)]
pub(crate) struct LastCameraChunk(Option<(i32, i32)>);

/// The chunk load/unload delta computed by [`update_pending_chunk_work`].
/// Later tickets (005-b onward) drain this to actually load/spawn/unload
/// chunks; this ticket only ever writes it, never acts on it.
#[derive(Resource, Debug, Default)]
pub struct PendingChunkWork {
    pub to_load: Vec<(i32, i32)>,
    pub to_unload: Vec<(i32, i32)>,
}

/// Adds [`RenderDistance`], [`ChunkRetention`] and [`PendingChunkWork`],
/// and the system that keeps the latter up to date with the camera's
/// position.
pub struct ChunkStreamingPlugin;

impl Plugin for ChunkStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderDistance>()
            .init_resource::<ChunkRetention>()
            .init_resource::<LastCameraChunk>()
            .init_resource::<LingeringChunks>()
            .init_resource::<PendingChunkWork>()
            .add_systems(Update, update_pending_chunk_work);
    }
}

/// The square of chunk coordinates within `radius` chunks of `center`
/// (inclusive), for a `(2*radius+1)^2`-chunk square — circular falloff
/// isn't worth the complexity yet, per the ticket.
pub fn desired_chunks(center: (i32, i32), radius: u32) -> HashSet<(i32, i32)> {
    let r = radius as i32;
    let side = (2 * r + 1) as usize;
    let mut set = HashSet::with_capacity(side * side);
    for dx in -r..=r {
        for dz in -r..=r {
            set.insert((center.0 + dx, center.1 + dz));
        }
    }
    set
}

/// Splits `loaded` against the two squares (ticket 070): chunks that need
/// **loading** (in `desired`, not yet `loaded`) and chunks that have left
/// **retention** (`loaded`, but outside `retain`).
///
/// The second half is deliberately not "to unload" — a column outside
/// `retain` is a *candidate*, which [`ChunkRetention::grace`] then holds on
/// to for a while longer. `retain` is expected to be a superset of
/// `desired` (see [`ChunkRetention::retain_radius`]), so a column that is
/// loaded and still wanted lands in neither list.
pub fn diff_chunks(
    desired: &HashSet<(i32, i32)>,
    retain: &HashSet<(i32, i32)>,
    loaded: &HashSet<(i32, i32)>,
) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
    let to_load = desired.difference(loaded).copied().collect();
    let left_retention = loaded.difference(retain).copied().collect();
    (to_load, left_retention)
}

/// Chebyshev ("chessboard") distance in chunks, the metric that matches the
/// square [`desired_chunks`] builds: it's exactly the radius at which
/// `other` first enters `center`'s square, which is what makes it the right
/// ordering for evicting the farthest lingering columns first.
fn chunk_distance(center: (i32, i32), other: (i32, i32)) -> i32 {
    (other.0 - center.0).abs().max((other.1 - center.1).abs())
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
            (-chunk_distance(center, coord), coord.0, coord.1)
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
        && !retention.is_changed()
        && !force_recompute
    {
        return; // Same chunk, same settings, not due for a retry yet.
    }
    last_chunk.0 = Some(center);

    let desired = desired_chunks(center, render_distance.0);
    let retain = desired_chunks(center, retention.retain_radius(render_distance.0));
    let loaded: HashSet<(i32, i32)> = decoded_world.columns.keys().copied().collect();
    let (to_load, left_retention) = diff_chunks(&desired, &retain, &loaded);

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
    fn desired_chunks_has_correct_radius_and_count() {
        let set = desired_chunks((5, -3), 2);
        // A radius-2 square is 5x5 = 25 chunks.
        assert_eq!(set.len(), 25);
        // Corners of the square are in.
        assert!(set.contains(&(3, -5)));
        assert!(set.contains(&(7, -1)));
        // One chunk past the radius is out.
        assert!(!set.contains(&(8, -3)));
        assert!(!set.contains(&(5, -6)));
    }

    #[test]
    fn desired_chunks_radius_zero_is_just_the_center() {
        let set = desired_chunks((0, 0), 0);
        assert_eq!(set, HashSet::from([(0, 0)]));
    }

    #[test]
    fn diff_chunks_buckets_entering_leaving_and_unchanged_chunks() {
        let desired = HashSet::from([(0, 0), (1, 0), (2, 0)]);
        // The retain square is the desired one plus a ring holding (3, 0).
        let retain = HashSet::from([(0, 0), (1, 0), (2, 0), (3, 0)]);
        let loaded = HashSet::from([(0, 0), (1, 0), (3, 0), (5, 5)]);

        let (mut to_load, mut left_retention) = diff_chunks(&desired, &retain, &loaded);
        to_load.sort();
        left_retention.sort();

        // (2, 0) is newly desired: entering.
        assert_eq!(to_load, vec![(2, 0)]);
        // (5, 5) is loaded and outside retention: a candidate. (3, 0) is
        // outside the *load* square but inside retention, so it is not.
        assert_eq!(left_retention, vec![(5, 5)]);
        // (0, 0) and (1, 0) are loaded and wanted: neither list.
        assert!(!to_load.contains(&(0, 0)));
        assert!(!left_retention.contains(&(0, 0)));
    }

    #[test]
    fn diff_chunks_empty_when_sets_match() {
        let set = HashSet::from([(0, 0), (1, 1)]);
        let (to_load, left_retention) = diff_chunks(&set, &set, &set);
        assert!(to_load.is_empty());
        assert!(left_retention.is_empty());
    }

    /// Ticket 070: the hysteresis margin is what keeps a camera nudging
    /// across a chunk boundary from paying a reload — a column one chunk
    /// outside the load square is still inside the retain square, so it
    /// never becomes a candidate in the first place.
    #[test]
    fn retain_radius_is_render_distance_plus_margin() {
        let retention = ChunkRetention {
            margin: 2,
            ..ChunkRetention::default()
        };
        assert_eq!(retention.retain_radius(10), 12);

        let desired = desired_chunks((0, 0), 10);
        let retain = desired_chunks((0, 0), retention.retain_radius(10));
        let loaded = HashSet::from([(11, 0), (13, 0)]);

        let (_, left_retention) = diff_chunks(&desired, &retain, &loaded);
        // (11, 0) is past render distance but inside the margin: kept.
        assert_eq!(left_retention, vec![(13, 0)]);
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
    fn chunk_distance_is_chebyshev() {
        assert_eq!(chunk_distance((0, 0), (3, 1)), 3);
        assert_eq!(chunk_distance((0, 0), (-4, 2)), 4);
        assert_eq!(chunk_distance((2, 2), (2, 2)), 0);
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
