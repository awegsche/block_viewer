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

use crate::world_app;

/// Runs the citybuilder. Called by `src/bin/citybuilder.rs`, which is three
/// lines and nothing else.
pub fn run() {
    world_app().run();
}
