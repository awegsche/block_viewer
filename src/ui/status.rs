//! Status panel (ticket 007): FPS, loaded/queued chunk counts, a render
//! distance slider, and a cheap memory estimate.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::chunk_pipeline::{InFlightChunkLoads, SharedRegionCache};
use crate::streaming::{PendingChunkWork, RenderDistance};
use crate::{world, DecodedWorld};

/// Egui window: read-only counters plus the one control the ticket calls
/// for here — a render distance slider (mutating [`RenderDistance`] drives
/// `streaming`/`chunk_pipeline`/`camera` the same way any other change to
/// it does, via `RenderDistance::is_changed()`).
pub(crate) fn status_panel(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    decoded_world: Res<DecodedWorld>,
    pending: Res<PendingChunkWork>,
    in_flight_loads: Res<InFlightChunkLoads>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut render_distance: ResMut<RenderDistance>,
) {
    egui::Window::new("Status").show(contexts.ctx_mut(), |ui| {
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.smoothed())
            .unwrap_or(0.0);
        ui.label(format!("FPS: {fps:.0}"));

        ui.label(format!("Loaded chunks: {}", decoded_world.columns.len()));
        // "Queued" covers both ends of the pipeline: coordinates diffed in
        // but not yet dispatched (`to_load`), and tasks already running.
        ui.label(format!(
            "Queued chunks: {}",
            pending.to_load.len() + in_flight_loads.len()
        ));

        if let Some(region_cache) = &region_cache {
            let resident = region_cache
                .0
                .lock()
                .expect("region cache mutex poisoned")
                .len();
            ui.label(format!("Cached regions: {resident}"));
        }

        // Cheap on purpose: total section count times a section's fixed
        // in-memory size, not a real allocator walk — good enough for a
        // ballpark, recomputed every frame the panel is open.
        let bytes: usize = decoded_world
            .columns
            .values()
            .map(|column| {
                column.sections.len() * world::SECTION_VOLUME * std::mem::size_of::<world::BlockId>()
            })
            .sum();
        ui.label(format!(
            "Block data: {:.1} MB",
            bytes as f64 / (1024.0 * 1024.0)
        ));

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Render distance");
            ui.add(egui::Slider::new(&mut render_distance.0, 2..=32));
        });
    });
}
