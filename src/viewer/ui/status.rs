//! Status panel (ticket 007): FPS, loaded/queued chunk counts, a render
//! distance slider, and a cheap memory estimate. Ticket 018 adds the day/
//! night clock controls at the bottom.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::chunk_pipeline::{InFlightChunkLoads, SharedRegionCache};
use crate::sky::{self, ShadowSettings, TimeOfDay};
use crate::streaming::{PendingChunkWork, RenderDistance};
use crate::{world, DecodedWorld};

/// Quick-jump buttons (ticket 018's UI section: "what someone actually
/// wants nine times out of ten") — label paired with the `TimeOfDay::ticks`
/// it jumps to, in [`sky`]'s own dawn/noon/dusk/midnight convention.
const QUICK_JUMPS: [(&str, f32); 4] = [
    ("Dawn", 0.0),
    ("Noon", 6000.0),
    ("Dusk", 12000.0),
    ("Midnight", 18000.0),
];

/// Speed multiples of [`sky::REAL_TIME_TICKS_PER_SECOND`] offered while
/// playing — the ticket calls out 1x (a real 20-minute Minecraft day) plus
/// a few multiples explicitly.
const SPEED_MULTIPLIERS: [f32; 4] = [1.0, 2.0, 5.0, 10.0];

/// Egui window: read-only counters plus the controls the ticket calls
/// for here — a render distance slider (mutating [`RenderDistance`] drives
/// `streaming`/`chunk_pipeline`/`camera` the same way any other change to
/// it does, via `RenderDistance::is_changed()`), the shadow toggle (017),
/// and the day/night clock (018, mutating [`TimeOfDay`] the same way —
/// `sky::time_of_day::sync_palette_from_time_of_day` picks up any change to
/// it next frame).
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
    mut time_of_day: ResMut<TimeOfDay>,
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

        // Ticket 018: the day/night clock. Dragging the slider, clicking
        // play/pause, picking a speed, or hitting a quick-jump button all
        // just mutate `TimeOfDay` like any other egui control here (see
        // `RenderDistance` above) — `sky`'s own systems pick the change up
        // from there. All egui widgets, all inside this same panel/window,
        // so the slider drag doesn't fly the camera for the same reason
        // the render-distance one above never has: `ui::sync_egui_input_capture`
        // and `camera::CameraSet`'s ordering (see `viewer::run`) already cover
        // every panel this module draws, this one included.
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Time of day");
            ui.label(sky::ticks_to_clock_string(time_of_day.ticks));
        });
        ui.add(egui::Slider::new(&mut time_of_day.ticks, 0.0..=sky::TICKS_PER_DAY).show_value(false));

        ui.horizontal(|ui| {
            let playing = time_of_day.rate != 0.0;
            if ui.button(if playing { "⏸" } else { "▶" }).clicked() {
                // Resuming always resumes at 1x — whatever multiplier was
                // last picked before pausing isn't remembered, which keeps
                // `TimeOfDay` (just `ticks`/`rate`) the only state this
                // needs, at the cost of a speed reset on every pause.
                time_of_day.rate = if playing { 0.0 } else { sky::REAL_TIME_TICKS_PER_SECOND };
            }
            if playing {
                for multiplier in SPEED_MULTIPLIERS {
                    let rate = sky::REAL_TIME_TICKS_PER_SECOND * multiplier;
                    let selected = (time_of_day.rate - rate).abs() < f32::EPSILON;
                    if ui
                        .selectable_label(selected, format!("{multiplier:.0}x"))
                        .clicked()
                    {
                        time_of_day.rate = rate;
                    }
                }
            }
        });

        ui.horizontal(|ui| {
            for (label, ticks) in QUICK_JUMPS {
                // Jumping snaps to a fixed look at that lighting rather
                // than continuing to run from there — the common case per
                // the ticket ("what someone actually wants nine times out
                // of ten" is a static look, not a mid-jump animation).
                if ui.button(label).clicked() {
                    time_of_day.ticks = ticks;
                    time_of_day.rate = 0.0;
                }
            }
        });
    });
}
