//! egui-based explorer UI (ticket 007): a save picker with a clickable
//! region grid, a coordinate jump + readout, a block inspector, and a
//! status panel. Every panel is read-only — this is a viewer, not an editor
//! (see the ticket's "out of scope").
//!
//! Each panel is its own system in its own submodule; this module only
//! wires them into the app and owns the two small pieces of state they
//! share ([`AvailableSaves`], [`UiState`]).

mod block_inspector;
mod navigate;
mod save_picker;
mod status;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin};
use mc_anvil::SaveMeta;

use crate::camera;

/// Saves discovered under the Minecraft saves directory, refreshed once at
/// UI startup. Kept as a `Result` rather than unwrapped/propagated — the
/// directory can simply not exist (no Minecraft installed), and this is a
/// display concern for the save picker, not a reason to crash a running
/// viewer (ticket 008 will make the equivalent *startup* path this
/// forgiving too).
#[derive(Resource)]
pub(crate) struct AvailableSaves(pub(crate) Result<Vec<SaveMeta>, String>);

impl Default for AvailableSaves {
    fn default() -> Self {
        // `Result` has no blanket `Default` — start "empty but not an
        // error" so the panel shows "no saves found" rather than a scary
        // red error message for the one frame before `scan_saves` runs.
        Self(Ok(Vec::new()))
    }
}

/// Text-input scratch state for the coordinate-jump panel, kept as a
/// resource (rather than local to the system) only because
/// [`navigate::navigate_panel`] is the sole owner — a plain `Local` would
/// do the same job; this stays a named resource so it's easy to find
/// alongside [`AvailableSaves`].
#[derive(Resource, Default)]
pub(crate) struct UiState {
    coord_x: String,
    coord_y: String,
    coord_z: String,
}

/// [`SystemSet`] every panel-drawing system runs in. `main.rs` orders
/// [`camera::CameraSet`] `.after()` this, and [`sync_egui_input_capture`]
/// also runs `.after()` this (see its own docs) — both need every panel to
/// have already drawn this frame before they read what egui claimed.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UiPanelSet;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            .init_resource::<AvailableSaves>()
            .init_resource::<UiState>()
            .add_systems(Startup, scan_saves)
            .add_systems(
                Update,
                (
                    save_picker::save_picker_panel,
                    navigate::navigate_panel,
                    block_inspector::block_inspector_panel,
                    status::status_panel,
                )
                    .in_set(UiPanelSet),
            )
            .add_systems(Update, sync_egui_input_capture.after(UiPanelSet));
    }
}

/// Populates [`AvailableSaves`] once at startup. The save picker panel has
/// its own "Refresh" affordance for re-running this later, since a saves
/// directory the process started without (or with different contents in)
/// can change while the viewer is running.
fn scan_saves(mut available: ResMut<AvailableSaves>) {
    available.0 = mc_anvil::get_saves().map_err(|e| e.to_string());
}

/// Reads whether egui claimed this frame's pointer/keyboard input, once
/// after every panel above has drawn, and publishes it as
/// [`camera::EguiInputCapture`] for [`camera::drive_camera`] to gate its
/// own input reads on (see that resource's docs for why). Running this
/// after every panel — rather than interleaved with them — means it always
/// reflects the union of every widget drawn this frame, not just whichever
/// panel happened to run first.
fn sync_egui_input_capture(
    mut contexts: EguiContexts,
    mut capture: ResMut<camera::EguiInputCapture>,
) {
    let ctx = contexts.ctx_mut();
    capture.pointer = ctx.wants_pointer_input();
    capture.keyboard = ctx.wants_keyboard_input();
}
