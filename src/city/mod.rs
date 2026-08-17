//! The `citybuilder` game (ticket 027) — milestone M1: a window on a real
//! save with the existing streaming and camera, and nothing else yet.
//!
//! The plan it grows into is `tickets/CITYBUILDER_ROADMAP.md`. The rule the
//! rest of it follows, worth repeating where the code will live: **the city
//! state is authoritative, and the blocks in the world are a projection of
//! it**. Nothing here owns any city state yet — that's roadmap task D1, and
//! it waits on the write path (group W) proving that blocks can land in a
//! world Minecraft opens without complaint.
//!
//! What this deliberately does *not* add, and why it looks so empty:
//!
//! - **No UI.** The viewer's egui panels ([`crate::viewer::ui`]) are its
//!   own; the citybuilder's build menu and city panel are roadmap group G.
//!   Without `EguiPlugin` the camera's [`crate::camera::EguiInputCapture`]
//!   simply stays at its "nothing captured" default, which is correct here.
//! - **No selection.** [`crate::selection`] is shared, but the box and its
//!   drag handles are the explorer's affordance. The citybuilder's own
//!   picking is roadmap task E1.
//!
//! The camera is the viewer's free-flight rig for now. An RTS camera (E1)
//! replaces it, and is free to drop the flight controls the viewer has to
//! keep.
//!
//! ## Render depth (ticket 030, roadmap R1)
//!
//! The one way this binary's world differs from `block_viewer`'s: [`run`]
//! overrides [`chunk_pipeline::RenderFloor`] with
//! [`world::decode::FloorPolicy::BelowSurface`]. The RTS camera this game
//! is built around looks at the surface from above and never goes
//! underground, so the ~7 sections per column below the terrain — caves
//! included — are never decoded or meshed. `block_viewer` keeps
//! [`world_app`]'s default of [`world::decode::FloorPolicy::WholeWorld`],
//! since it's the explore-a-save app and the underground is exactly what it
//! exists to show.

use crate::{chunk_pipeline::RenderFloor, world::decode::FloorPolicy, world_app};

/// Runs the citybuilder. Called by `src/bin/citybuilder.rs`, which is three
/// lines and nothing else.
pub fn run() {
    world_app()
        .insert_resource(RenderFloor(FloorPolicy::BelowSurface { margin: 16 }))
        .run();
}
