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
//! The diff recomputes only when the camera crosses a chunk boundary, not
//! every frame — a render distance of 12 is at most ~625 chunks, cheap to
//! diff, but there's no reason to redo the same `HashSet` diff dozens of
//! times a second while the camera sits still. 005-c/005-d can assume
//! [`PendingChunkWork`] only changes on frames where the camera's chunk
//! coordinate actually changed.

use bevy::prelude::*;
use std::collections::HashSet;

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

/// Adds [`RenderDistance`] and [`PendingChunkWork`], and the system that
/// keeps the latter up to date with the camera's position.
pub struct ChunkStreamingPlugin;

impl Plugin for ChunkStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderDistance>()
            .init_resource::<LastCameraChunk>()
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

/// Splits `desired` vs. `loaded` into chunks that need loading (in
/// `desired` but not `loaded`) and chunks that need unloading (in `loaded`
/// but not `desired`). Chunks present in both land in neither list.
pub fn diff_chunks(
    desired: &HashSet<(i32, i32)>,
    loaded: &HashSet<(i32, i32)>,
) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
    let to_load = desired.difference(loaded).copied().collect();
    let to_unload = loaded.difference(desired).copied().collect();
    (to_load, to_unload)
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

/// Recomputes [`PendingChunkWork`] whenever the camera's chunk coordinate
/// changes, or whenever [`RenderDistance`] itself changes (ticket 007's
/// status-panel slider) — the camera can sit in the same chunk while the
/// desired radius around it grows or shrinks, and that needs the same
/// re-diff a chunk crossing does. Otherwise does not act on the delta —
/// loading/spawning/unloading is 005-b onward.
fn update_pending_chunk_work(
    camera: Query<&Transform, With<camera::CameraRig>>,
    render_distance: Res<RenderDistance>,
    decoded_world: Res<DecodedWorld>,
    mut last_chunk: ResMut<LastCameraChunk>,
    mut pending: ResMut<PendingChunkWork>,
) {
    let Ok(transform) = camera.get_single() else {
        return;
    };

    let center = camera_chunk_coord(transform.translation);
    if last_chunk.0 == Some(center) && !render_distance.is_changed() {
        return; // Same chunk, same render distance; the delta hasn't changed.
    }
    last_chunk.0 = Some(center);

    let desired = desired_chunks(center, render_distance.0);
    let loaded: HashSet<(i32, i32)> = decoded_world.columns.keys().copied().collect();
    let (to_load, to_unload) = diff_chunks(&desired, &loaded);

    println!(
        "Chunk streaming: camera entered chunk {:?} ({} to load, {} to unload)",
        center,
        to_load.len(),
        to_unload.len()
    );

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
        let loaded = HashSet::from([(0, 0), (1, 0), (5, 5)]);

        let (mut to_load, mut to_unload) = diff_chunks(&desired, &loaded);
        to_load.sort();
        to_unload.sort();

        // (2, 0) is newly desired: entering.
        assert_eq!(to_load, vec![(2, 0)]);
        // (5, 5) is loaded but no longer desired: leaving.
        assert_eq!(to_unload, vec![(5, 5)]);
        // (0, 0) and (1, 0) are in both: neither list.
        assert!(!to_load.contains(&(0, 0)));
        assert!(!to_unload.contains(&(0, 0)));
    }

    #[test]
    fn diff_chunks_empty_when_sets_match() {
        let set = HashSet::from([(0, 0), (1, 1)]);
        let (to_load, to_unload) = diff_chunks(&set, &set);
        assert!(to_load.is_empty());
        assert!(to_unload.is_empty());
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
