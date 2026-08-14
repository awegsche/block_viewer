//! Coordinate jump + camera position readout (ticket 007) — "the single
//! most useful feature for 'explore a save'" per the ticket.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::region_cache;
use crate::streaming;
use crate::camera;

use super::UiState;

/// Egui window: block/chunk/region readout for the camera's current
/// position, plus X/Y/Z text fields + a "Go" button that teleports it.
pub(crate) fn navigate_panel(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<UiState>,
    mut camera_query: Query<(&mut Transform, &mut camera::CameraRig)>,
) {
    egui::Window::new("Navigate").show(contexts.ctx_mut(), |ui| {
        let Ok((mut transform, mut rig)) = camera_query.get_single_mut() else {
            ui.label("No camera.");
            return;
        };

        // bevy.x = mc.x, bevy.y = mc.y, bevy.z = -mc.z — see `world::mesh`
        // docs for the convention every coordinate in this app follows.
        let translation = transform.translation;
        let (mc_x, mc_y, mc_z) = (translation.x, translation.y, -translation.z);
        ui.label(format!(
            "Block: {}, {}, {}",
            mc_x.floor() as i32,
            mc_y.floor() as i32,
            mc_z.floor() as i32
        ));

        let chunk = streaming::camera_chunk_coord(translation);
        ui.label(format!("Chunk: {}, {}", chunk.0, chunk.1));

        let region = region_cache::chunk_to_region_coord(chunk);
        ui.label(format!("Region: {}, {}", region.0, region.1));

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("X");
            ui.add(egui::TextEdit::singleline(&mut ui_state.coord_x).desired_width(60.0));
            ui.label("Y");
            ui.add(egui::TextEdit::singleline(&mut ui_state.coord_y).desired_width(60.0));
            ui.label("Z");
            ui.add(egui::TextEdit::singleline(&mut ui_state.coord_z).desired_width(60.0));

            if ui.button("Go").clicked() {
                let parsed = (
                    ui_state.coord_x.trim().parse::<f32>(),
                    ui_state.coord_y.trim().parse::<f32>(),
                    ui_state.coord_z.trim().parse::<f32>(),
                );
                if let (Ok(x), Ok(y), Ok(z)) = parsed {
                    // Force fly mode: in orbit mode `drive_camera` derives
                    // translation from `orbit_target` every frame, which
                    // would otherwise silently undo this teleport next
                    // frame.
                    rig.mode = camera::CameraMode::Fly;
                    transform.translation = Vec3::new(x, y, -z);
                }
            }
        });
        if ui_state.coord_x.parse::<f32>().is_err() && !ui_state.coord_x.is_empty()
            || ui_state.coord_y.parse::<f32>().is_err() && !ui_state.coord_y.is_empty()
            || ui_state.coord_z.parse::<f32>().is_err() && !ui_state.coord_z.is_empty()
        {
            ui.colored_label(egui::Color32::RED, "X/Y/Z must be numbers.");
        }
    });
}
