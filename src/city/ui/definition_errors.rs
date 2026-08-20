//! The "Definition Errors" panel (ticket 061, roadmap C4): the egui half of
//! "errors go to an egui panel rather than a panic or a console line nobody
//! sees" — [`super::super::hot_reload`] is the loading half, this is where
//! its [`DefinitionErrors`] resource actually gets read by a player.
//!
//! Always registered, same as [`super::city_panel::city_panel`]'s
//! always-visible window with an empty-state message when there's nothing to
//! show — a panel that only appears once something's already broken is a
//! panel a player doesn't know to go looking for. `default_open` follows
//! whether there are any errors *the first time the window is shown* (egui
//! only consults it once per window id, then remembers the player's own
//! collapse state) — a clean start collapses out of the way, a save that
//! opens with a bad file already on disk opens expanded so it's not missed.
//! The title bar names the count either way, so a later edit that introduces
//! a problem is visible even collapsed.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::super::hot_reload::DefinitionErrors;

/// One `(path, message)` list as a scrollable, monospace block — the same
/// path-display convention [`super::city_panel::save_section`] uses for
/// backup locations.
fn error_list(ui: &mut egui::Ui, heading: &str, errors: &[(std::path::PathBuf, String)]) {
    if errors.is_empty() {
        return;
    }
    ui.label(egui::RichText::new(heading).strong());
    for (path, message) in errors {
        ui.colored_label(egui::Color32::RED, format!("{}: {message}", path.display()));
    }
}

/// Egui window: every currently-skipped `.ron` file, building definitions
/// and road types each under their own heading. See the module docs.
///
/// The window's title stays the fixed string `"Definition Errors"` rather
/// than growing a live count — egui keys a window's identity (position,
/// open/collapsed state) off its title by default, so a title that changes
/// text every time the error count changes would make every reload look
/// like a brand new window to egui, resetting wherever the player had moved
/// or collapsed it. The count goes in the body instead, where it can change
/// freely.
pub(super) fn definition_errors_panel(mut contexts: EguiContexts, errors: Res<DefinitionErrors>) {
    egui::Window::new("Definition Errors").collapsible(true).default_open(!errors.is_empty()).show(contexts.ctx_mut(), |ui| {
        if errors.is_empty() {
            ui.label("(no problems)");
            return;
        }
        ui.label(format!("{} problem(s)", errors.buildings.len() + errors.road_types.len()));
        error_list(ui, "Buildings (assets/city/buildings)", &errors.buildings);
        if !errors.buildings.is_empty() && !errors.road_types.is_empty() {
            ui.separator();
        }
        error_list(ui, "Road types (assets/city/road_types)", &errors.road_types);
    });
}
