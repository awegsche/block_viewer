//! Screen ray → block coordinate for the citybuilder (ticket 045, roadmap
//! E1). [`HoveredBlock`] is the single per-frame answer to "what block is
//! the RTS camera's cursor over right now" — E2's grid/footprint fit and
//! E3's ghost preview are the eventual readers. Nothing consumes it yet,
//! the same "proven, not yet used" state ticket 042's `City` landed in.
//!
//! Reuses [`camera::block_under_cursor`] rather than a second raycast — the
//! same DDA ray-march the viewer's block inspector (007) and orbit re-aim
//! (006) already use, so every caller in the crate agrees on what's under
//! the cursor.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::{camera, DecodedWorld};

/// The Minecraft block coordinate the camera's cursor is currently over,
/// updated once a frame by [`update_hovered_block`]. `None` covers both "the
/// cursor is outside the window" and "the ray hit nothing loaded/solid" —
/// [`camera::block_under_cursor`] already collapses those into one `None`,
/// and nothing downstream needs them told apart yet.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HoveredBlock(pub Option<IVec3>);

pub struct PickingPlugin;

impl Plugin for PickingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HoveredBlock>()
            // After `camera::CameraSet` so this reads the camera's transform
            // once this frame's pan/rotate/zoom has already landed, not the
            // previous frame's — the same reason `viewer::run` orders
            // `CameraSet` relative to the UI panels that read it.
            .add_systems(Update, update_hovered_block.after(camera::CameraSet));
    }
}

fn update_hovered_block(
    mut hovered: ResMut<HoveredBlock>,
    camera_query: Query<(&Camera, &GlobalTransform), With<camera::CameraRig>>,
    settings: Res<camera::CameraSettings>,
    decoded_world: Res<DecodedWorld>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    hovered.0 = (|| {
        let (camera, camera_transform) = camera_query.get_single().ok()?;
        let cursor = windows.get_single().ok()?.cursor_position()?;
        camera::block_under_cursor(
            camera,
            camera_transform,
            cursor,
            settings.max_ray_distance,
            &decoded_world,
        )
    })();
}
