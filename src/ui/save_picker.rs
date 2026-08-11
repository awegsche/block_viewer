//! Save picker + clickable region grid (ticket 007).

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use mc_anvil::SaveMeta;

use crate::chunk_pipeline::{
    InFlightChunkLoads, InFlightChunkRemeshes, PendingChunkRemeshes, SharedRegionCache,
    SpawnedChunkEntities,
};
use crate::streaming::{self, PendingChunkWork, RenderDistance};
use crate::{camera, region_cache, region_center_point, BlockMesh, DecodedWorld, LoadedSave};

use super::AvailableSaves;

/// Bundles every resource a save switch (or a camera-only region-grid
/// teleport) needs to touch. As a flat parameter list this would sit right
/// at Bevy's per-system parameter ceiling; grouping it here also keeps
/// [`save_picker_panel`] itself readable.
#[derive(SystemParam)]
pub(crate) struct WorldReset<'w, 's> {
    commands: Commands<'w, 's>,
    loaded_save: ResMut<'w, LoadedSave>,
    decoded_world: ResMut<'w, DecodedWorld>,
    spawned_meshes: Query<'w, 's, (Entity, &'static Mesh3d), With<BlockMesh>>,
    meshes: ResMut<'w, Assets<Mesh>>,
    spawned: ResMut<'w, SpawnedChunkEntities>,
    shared_region_cache: Option<ResMut<'w, SharedRegionCache>>,
    pending: ResMut<'w, PendingChunkWork>,
    last_chunk: ResMut<'w, streaming::LastCameraChunk>,
    in_flight_loads: ResMut<'w, InFlightChunkLoads>,
    in_flight_remeshes: ResMut<'w, InFlightChunkRemeshes>,
    pending_remeshes: ResMut<'w, PendingChunkRemeshes>,
    camera: Query<'w, 's, (&'static mut Transform, &'static mut camera::CameraRig)>,
}

impl WorldReset<'_, '_> {
    /// Moves the camera to `target` (Bevy space) without touching any other
    /// streaming state — used by the region-grid click (same save, just a
    /// new place to look) and as the last step of [`Self::switch_save`).
    fn teleport_camera(&mut self, target: Vec3) {
        let eye = target + Vec3::new(-24.0, 20.0, 24.0);
        let Ok((mut transform, mut rig)) = self.camera.get_single_mut() else {
            return;
        };
        *transform = Transform::from_translation(eye).looking_at(target, Vec3::Y);
        *rig = camera::CameraRig::looking_at(eye, target);
    }

    /// Tears down every spawned chunk and in-flight load/re-mesh task,
    /// points the region cache and [`LoadedSave`] at `meta`, and moves the
    /// camera to its region centroid — everything the ticket's "switching
    /// tears down spawned chunks and re-streams" needs, so the next
    /// `Update` starts streaming `meta` in from scratch exactly the way
    /// startup does (`main.rs::setup`). Every pipeline resource here
    /// derives `Default`, so resetting is just replacing each with a fresh
    /// one rather than draining it field by field.
    fn switch_save(&mut self, meta: SaveMeta, render_distance: u32) {
        for (entity, mesh3d) in &self.spawned_meshes {
            self.meshes.remove(&mesh3d.0);
            self.commands.entity(entity).despawn();
        }
        self.decoded_world.columns.clear();
        *self.spawned = SpawnedChunkEntities::default();
        *self.in_flight_loads = InFlightChunkLoads::default();
        *self.in_flight_remeshes = InFlightChunkRemeshes::default();
        *self.pending_remeshes = PendingChunkRemeshes::default();
        *self.pending = PendingChunkWork::default();
        *self.last_chunk = streaming::LastCameraChunk::default();

        if let Some(shared) = &mut self.shared_region_cache {
            let capacity = region_cache::recommended_capacity(render_distance);
            *shared.0.lock().expect("region cache mutex poisoned") =
                region_cache::RegionCache::new(meta.clone(), capacity);
        }

        self.teleport_camera(crate::spawn_point(&meta));
        self.loaded_save.0 = meta.into();
    }
}

/// Egui window: pick a save from everything discovered under the Minecraft
/// saves directory, and a clickable grid of the *current* save's regions to
/// jump the camera straight to one.
pub(crate) fn save_picker_panel(
    mut contexts: EguiContexts,
    available: Res<AvailableSaves>,
    render_distance: Res<RenderDistance>,
    mut reset: WorldReset,
) {
    // Cloned once up front so the egui closure below doesn't need to hold
    // a borrow of `reset` across the same window it may end up calling
    // `reset.switch_save`/`teleport_camera` from.
    let current_meta = reset.loaded_save.0.meta.clone();

    let mut to_load: Option<SaveMeta> = None;
    let mut to_teleport: Option<Vec3> = None;

    egui::Window::new("Save").show(contexts.ctx_mut(), |ui| {
        ui.label(format!(
            "Active: {} ({} regions)",
            current_meta.name,
            current_meta.regions.len()
        ));
        ui.separator();

        match &available.0 {
            Ok(saves) if saves.is_empty() => {
                ui.label("No saves found under the Minecraft saves directory.");
            }
            Ok(saves) => {
                egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                    for meta in saves {
                        ui.horizontal(|ui| {
                            let is_current = meta.name == current_meta.name;
                            ui.label(format!(
                                "{}{} ({})",
                                meta.name,
                                if is_current { " — current" } else { "" },
                                meta.regions.len()
                            ));
                            if !is_current && ui.button("Load").clicked() {
                                to_load = Some(meta.clone());
                            }
                        });
                    }
                });
            }
            Err(err) => {
                ui.colored_label(egui::Color32::RED, format!("Could not list saves: {err}"));
            }
        }

        ui.separator();
        ui.label("Regions (click to jump):");
        draw_region_grid(ui, &current_meta, &mut to_teleport);
    });

    // Applied after the window closure returns, so nothing above is still
    // borrowing `available`/`current_meta` while `reset` mutates.
    if let Some(meta) = to_load {
        reset.switch_save(meta, render_distance.0);
    } else if let Some(target) = to_teleport {
        reset.teleport_camera(target);
    }
}

/// Renders `meta`'s regions as a small clickable grid (filled = region
/// exists) when the save's footprint is compact enough for that to be
/// readable; falls back to [`SaveMeta::get_grid_view`]'s ASCII map for
/// saves spanning an impractically large area rather than emitting
/// hundreds of tiny egui buttons.
fn draw_region_grid(ui: &mut egui::Ui, meta: &SaveMeta, clicked: &mut Option<Vec3>) {
    if meta.regions.is_empty() {
        ui.label("(no regions)");
        return;
    }

    let min_x = meta.regions.iter().map(|(x, _)| *x).min().unwrap();
    let max_x = meta.regions.iter().map(|(x, _)| *x).max().unwrap();
    let min_z = meta.regions.iter().map(|(_, z)| *z).min().unwrap();
    let max_z = meta.regions.iter().map(|(_, z)| *z).max().unwrap();

    let cells = (max_x - min_x + 1) as i64 * (max_z - min_z + 1) as i64;
    const MAX_GRID_CELLS: i64 = 400;
    if cells > MAX_GRID_CELLS {
        egui::ScrollArea::both().max_height(200.0).show(ui, |ui| {
            ui.monospace(format!("{}", meta.get_grid_view()));
        });
        return;
    }

    egui::Grid::new("region_grid").spacing([2.0, 2.0]).show(ui, |ui| {
        for z in min_z..=max_z {
            for x in min_x..=max_x {
                if meta.has_region(x, z) {
                    let clicked_button = ui
                        .add_sized([18.0, 18.0], egui::Button::new(""))
                        .on_hover_text(format!("region ({x}, {z})"));
                    if clicked_button.clicked() {
                        *clicked = Some(region_center_point(x, z));
                    }
                } else {
                    ui.add_sized([18.0, 18.0], egui::Label::new(""));
                }
            }
            ui.end_row();
        }
    });
}
