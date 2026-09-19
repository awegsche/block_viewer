//! The loading screen (ticket 124): an opaque full-window panel with the
//! save's name and a progress bar, drawn for as long as
//! [`CityPhase::Loading`] lasts — see [`super::super::loading`] for what
//! it's waiting on. The 3D cameras keep rendering underneath (terrain is
//! landing the whole time); this simply covers them.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::LoadedSave;

use super::super::loading::InitialLoad;

/// Opaque, so nothing of the half-streamed world shows through.
const BACKDROP: egui::Color32 = egui::Color32::from_rgb(18, 20, 24);

/// The bar's width — wide enough to read progress off, narrow enough to
/// stay a bar rather than a stripe across a wide window.
const BAR_WIDTH: f32 = 360.0;

pub(super) fn loading_screen(mut contexts: EguiContexts, loaded_save: Res<LoadedSave>, progress: Res<InitialLoad>) {
    egui::CentralPanel::default().frame(egui::Frame::none().fill(BACKDROP)).show(contexts.ctx_mut(), |ui| {
        ui.vertical_centered(|ui| {
            // A little above centre reads as "centred"; dead centre reads
            // as low, with the bar hanging under it.
            ui.add_space((ui.available_height() * 0.4).max(0.0));
            ui.heading(egui::RichText::new("Loading world").size(28.0));
            ui.add_space(8.0);
            ui.label(&loaded_save.0.meta.name);
            ui.add_space(16.0);
            ui.add(egui::ProgressBar::new(progress.fraction()).desired_width(BAR_WIDTH).text(status_line(&progress)));
        });
    });
}

/// The bar's caption. Before the disc is published there's no count to
/// show; a bare "0 / 0 chunks" would read as a broken start.
fn status_line(progress: &InitialLoad) -> String {
    if !progress.started {
        return "Preparing…".to_string();
    }
    format!("{} / {} chunks", progress.resolved(), progress.total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_line_counts_resolved_chunks_once_started() {
        assert_eq!(status_line(&InitialLoad { started: false, total: 0, outstanding: 0 }), "Preparing…");
        assert_eq!(status_line(&InitialLoad { started: true, total: 1009, outstanding: 600 }), "409 / 1009 chunks");
    }
}
