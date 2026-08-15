//! The `block_viewer` game (ticket 027): an explorer for a real Minecraft
//! save. Everything below the entry point is shared with [`crate::city`] —
//! see [`crate::world_app`] — and what this module adds is the explorer's
//! own layer: the selection volume's interaction and panel, blueprint
//! extraction started from it, the paint/fill command (ticket 035, roadmap
//! W8), and the egui UI.
//!
//! This was `main.rs` before ticket 027 split the package into a lib plus
//! two binary shims. `run()` is that `main()`'s body, minus the half that
//! moved to [`crate::world_app`].

mod paint;
pub mod ui;

use bevy::prelude::*;

use crate::{blueprint, camera, selection, world_app};

/// Runs the viewer. Called by `src/bin/block_viewer.rs`, which is three
/// lines and nothing else.
pub fn run() {
    world_app()
        .add_plugins(selection::SelectionPlugin)
        .add_plugins(blueprint::BlueprintPlugin)
        .add_plugins(paint::PaintPlugin)
        .add_plugins(ui::UiPlugin)
        // The UI plugin's panels (ticket 007) need to have drawn this
        // frame before `drive_camera` — and, ticket 020, the selection's
        // click/key handling — read whether egui claimed pointer/keyboard
        // input. See `camera::CameraSet`'s docs. This ordering lives here
        // rather than in `world_app` because it only means anything where
        // there *is* a UI; the citybuilder has no egui yet.
        .configure_sets(
            Update,
            (camera::CameraSet, selection::SelectionInputSet).after(ui::UiPanelSet),
        )
        .run();
}
