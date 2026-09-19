//! The citybuilder's own egui UI (ticket 050, roadmap G): a build menu
//! ([`build_menu`]) and a city panel ([`city_panel`]) — the group `city::mod`'s
//! own docs have named as missing since the game's first commit. Ticket 083
//! (roadmap G3) added a third window, [`inspect_panel`] — what a left-click
//! means once [`super::tool::ActiveTool::Inspect`] is the resting state.
//!
//! ## Why a second `UiPlugin`, not [`crate::viewer::ui::UiPlugin`]
//!
//! The viewer's UI is a save picker, a coordinate jump, a block inspector, a
//! selection panel — every one of them either explorer-only or built around
//! [`crate::selection::Selection`], which the citybuilder deliberately
//! doesn't use (see `city::mod`'s own "No selection" docs). Reusing that
//! plugin would mean carrying five panels nobody asked for, or picking them
//! apart from a module `city` has no reason to depend on more than it
//! already avoids. So this is its own small [`EguiPlugin`] registration and
//! its own [`UiPanelSet`] — the same shape, applied to a different pair of
//! panels.
//!
//! ## Wiring, mirroring `viewer::run`
//!
//! [`crate::city::run`] adds [`UiPlugin`] and orders `camera::CameraSet`
//! after [`UiPanelSet`], exactly the two lines
//! [`crate::viewer::run`](crate::viewer::run) already uses for the same
//! reason: the panels have to have claimed this frame's pointer/keyboard
//! input (`camera::EguiInputCapture`, synced by [`sync_egui_input_capture`]
//! below) before `drive_camera` and the picking/placement/commit/demolish
//! systems that already gate on it read it — see
//! [`crate::viewer::ui`]'s own module docs for the fuller version of this
//! argument.

mod build_menu;
mod city_panel;
mod definition_errors;
mod inspect_panel;
mod loading_screen;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin};

use crate::camera;

use super::loading::{CityPhase, GameplaySet};

/// [`SystemSet`] both panels run in — [`crate::city::run`] orders
/// `camera::CameraSet` `.after()` this, the same relationship
/// [`crate::viewer::ui::UiPanelSet`] has with the viewer's own camera set.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UiPanelSet;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            // The game's panels are game logic as far as ticket 124 is
            // concerned — none of them has anything to say (or any button
            // that should work) under the loading screen, which is the one
            // panel drawn instead until `CityPhase::Playing`.
            .add_systems(
                Update,
                (
                    build_menu::build_menu_panel,
                    city_panel::city_panel,
                    definition_errors::definition_errors_panel,
                    inspect_panel::inspect_panel,
                )
                    .in_set(UiPanelSet)
                    .in_set(GameplaySet),
            )
            .add_systems(Update, loading_screen::loading_screen.in_set(UiPanelSet).run_if(in_state(CityPhase::Loading)))
            .add_systems(Update, sync_egui_input_capture.after(UiPanelSet));
    }
}

/// [`camera::EguiInputCapture`]'s only writer in the citybuilder — identical
/// to [`crate::viewer::ui::sync_egui_input_capture`], duplicated rather than
/// shared because the two are one line of logic each and sharing it would
/// mean threading a public function between two otherwise-independent UI
/// modules for no real reuse.
fn sync_egui_input_capture(mut contexts: EguiContexts, mut capture: ResMut<camera::EguiInputCapture>) {
    let ctx = contexts.ctx_mut();
    capture.pointer = ctx.wants_pointer_input();
    capture.keyboard = ctx.wants_keyboard_input();
}
