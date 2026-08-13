//! Status panel (ticket 007): FPS, loaded/queued chunk counts, a render
//! distance slider, and a cheap memory estimate.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::chunk_pipeline::{InFlightChunkLoads, SharedRegionCache};
use crate::sky::{self, ShadowSettings};
use crate::streaming::{PendingChunkWork, RenderDistance};
use crate::{world, DecodedWorld};

/// Egui window: read-only counters plus the one control the ticket calls
/// for here — a render distance slider (mutating [`RenderDistance`] drives
/// `streaming`/`chunk_pipeline`/`camera` the same way any other change to
/// it does, via `RenderDistance::is_changed()`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn status_panel(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    decoded_world: Res<DecodedWorld>,
    pending: Res<PendingChunkWork>,
    in_flight_loads: Res<InFlightChunkLoads>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut render_distance: ResMut<RenderDistance>,
    mut shadow_settings: ResMut<ShadowSettings>,
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
        // ballpark, recomputed every frame the panel is open. Includes each
        // section's biome grid (ticket 012) alongside its block grid — 128 B
        // against 8 KB, about 1.5% on top, but small enough to just fold in
        // rather than leave the estimate quietly wrong by more than it
        // already was.
        let bytes: usize = decoded_world
            .columns
            .values()
            .map(|column| {
                column.sections.len()
                    * (world::SECTION_VOLUME * std::mem::size_of::<world::BlockId>()
                        + world::BIOME_GRID_VOLUME * std::mem::size_of::<world::BiomeId>())
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

        // Ticket 017: this is also the fastest way for a human to measure
        // shadows' cost — flip it and watch the FPS label above. The
        // distance shown is derived (`sky::shadow_cascade_distance`), not a
        // separate control — it tracks the render-distance slider the same
        // way the far plane and fog already do (capped well below it; see
        // that function's docs for why).
        ui.horizontal(|ui| {
            ui.checkbox(&mut shadow_settings.enabled, "Shadows");
            ui.label(format!(
                "(cascades out to {:.0} blocks)",
                sky::shadow_cascade_distance(render_distance.0)
            ));
        });
    });
}
