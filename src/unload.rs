//! Unload path + memory budget (ticket 005-d): the unload half of 005-a's
//! load/unload diff, plus the two things that diff can't express by itself.
//!
//! - **Despawn + free**: for every coordinate 005-a's diff put in
//!   [`PendingChunkWork::to_unload`], despawn its chunk entity, free its
//!   mesh, and drop its [`crate::world::ChunkColumn`] from [`DecodedWorld`].
//! - **Cancel stale in-flight loads**: `to_unload` is computed against
//!   `DecodedWorld.columns` (005-a's notion of "loaded"), so a coordinate
//!   that's mid-load (005-c) but not yet inserted into `columns` never
//!   shows up in `to_unload` at all — the camera can reverse near the
//!   loading edge, leave a coordinate's render-distance square, and its
//!   in-flight task keeps running toward a spawn that should no longer
//!   happen. This module recomputes the desired set itself every frame
//!   (the same cheap `HashSet` build 005-a's own diff does — see
//!   `streaming`'s module docs) and cancels any in-flight task that's
//!   fallen outside it.
//!
//! Region cache eviction (005-b) needs no code here: `RegionCache`'s
//! capacity is already sized to one render distance's worth of regions
//! (`region_cache::recommended_capacity`, wired in `lib.rs::setup_world`), so
//! plain LRU eviction drops regions no loaded chunk still needs — the
//! simpler of the two options the ticket allows, chosen over a per-region
//! refcount because the sizing already holds up.
//!
//! No per-frame unload budget: a flying (not teleporting) camera only ever
//! crosses one chunk boundary at a time, so nothing so far unloads more
//! than a render-distance ring's edge per diff tick — add a budget only if
//! a large jump is measured to spike a frame.

use std::collections::HashSet;

use bevy::prelude::*;

use crate::camera;
use crate::chunk_pipeline::{
    poll_completed_chunk_loads, poll_completed_chunk_remeshes, start_chunk_loads,
    start_chunk_remeshes, InFlightChunkLoads, InFlightChunkRemeshes, PendingChunkRemeshes,
    SpawnedChunkEntities,
};
use crate::streaming::{self, PendingChunkWork, RenderDistance};
use crate::DecodedWorld;

/// Adds the two unload systems. Ordered `.before()` `chunk_pipeline`'s own
/// systems (both `pub(crate)` for exactly this) so a coordinate canceled or
/// unloaded this frame can't also get polled-complete and spawned, or
/// re-requested, in the same frame. The cancel system's `.before()`s cover
/// both the load and re-mesh (005-f) halves of the pipeline, since a
/// coordinate leaving render distance needs to drop in-flight work of
/// either kind the same way.
pub struct ChunkUnloadPlugin;

impl Plugin for ChunkUnloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                cancel_out_of_range_in_flight_work
                    .before(poll_completed_chunk_loads)
                    .before(poll_completed_chunk_remeshes),
                unload_chunks
                    .before(start_chunk_loads)
                    .before(start_chunk_remeshes),
            ),
        );
    }
}

/// Despawns the entity and frees the decoded column + mesh for every
/// coordinate in [`PendingChunkWork::to_unload`]. Reprocesses the same list
/// every frame, the same way `chunk_pipeline::start_chunk_loads` reprocesses
/// `to_load` — idempotent, since a coordinate already unloaded is no longer
/// in [`SpawnedChunkEntities`] or [`DecodedWorld::columns`], so repeat
/// frames are a cheap no-op lookup rather than a double despawn.
fn unload_chunks(
    pending: Res<PendingChunkWork>,
    mut commands: Commands,
    mut decoded_world: ResMut<DecodedWorld>,
    mut spawned: ResMut<SpawnedChunkEntities>,
    mut meshes: ResMut<Assets<Mesh>>,
    mesh_of: Query<&Mesh3d>,
) {
    for &coord in &pending.to_unload {
        if let Some(entity) = spawned.0.remove(&coord) {
            // Every chunk mesh is unique to its entity (never shared), so
            // freeing it here is always safe — no refcount to check.
            if let Ok(mesh3d) = mesh_of.get(entity) {
                meshes.remove(&mesh3d.0);
            }
            commands.entity(entity).despawn();
        }
        // Fully-air columns (no entity, no mesh) still need their decoded
        // data dropped.
        decoded_world.columns.remove(&coord);
    }
}

/// Cancels (drops) any in-flight chunk-load or chunk-re-mesh (005-f) task
/// whose coordinate has left render distance since it was kicked off, and
/// drops any coordinate still waiting in [`PendingChunkRemeshes`] the same
/// way. Simply dropping a `Task` cancels it (`bevy_tasks::Task::cancel`'s
/// doc comment: "it's possible to simply drop the `Task` to cancel it"), so
/// no `.await` is needed. Without the re-mesh half of this, a task that
/// outlives the chunk's unload would complete later and, finding no entity
/// left in [`SpawnedChunkEntities`] to update, respawn one — see
/// [`InFlightChunkRemeshes::cancel_out_of_range`]'s docs.
fn cancel_out_of_range_in_flight_work(
    camera: Query<&Transform, With<camera::CameraRig>>,
    render_distance: Res<RenderDistance>,
    mut in_flight: ResMut<InFlightChunkLoads>,
    mut in_flight_remeshes: ResMut<InFlightChunkRemeshes>,
    mut pending_remeshes: ResMut<PendingChunkRemeshes>,
) {
    let Ok(transform) = camera.get_single() else {
        return;
    };

    let center = streaming::camera_chunk_coord(transform.translation);
    let desired: HashSet<(i32, i32)> = streaming::desired_chunks(center, render_distance.0);
    in_flight.cancel_out_of_range(&desired);
    in_flight_remeshes.cancel_out_of_range(&desired);
    pending_remeshes.cancel_out_of_range(&desired);
}
