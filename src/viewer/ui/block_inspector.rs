//! Block inspector (ticket 007): the block under the cursor, read via
//! `ChunkRegion::get_block` — the raw single-block NBT lookup, not
//! [`crate::DecodedWorld`]'s decoded grid the mesher reads. Ticket 007
//! calls this out explicitly: going through the separate raw path means a
//! bug in either the decode layer (002) or here would show up as the two
//! disagreeing, rather than one silently trusting the other's mistake.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts};
use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;

use crate::chunk_pipeline::SharedRegionCache;
use crate::{camera, world, DecodedWorld};

/// A block's name plus its decoded `"key = value"` properties, or `None`
/// if [`lookup_block`] couldn't read it.
type BlockLookup = Option<(String, Vec<String>)>;

/// The last voxel looked up and what was found there, so repeated frames
/// with the cursor sitting still over the same block don't re-lock the
/// region cache (and potentially re-load a region that's since been
/// evicted) every single frame — only when the targeted voxel changes.
#[derive(Default)]
pub(crate) struct LastLookup(Option<(IVec3, BlockLookup)>);

/// Egui window: name + decoded properties of the block the camera is
/// currently pointing at.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_inspector_panel(
    mut contexts: EguiContexts,
    egui_input: Res<camera::EguiInputCapture>,
    camera_query: Query<(&Camera, &GlobalTransform), With<camera::CameraRig>>,
    settings: Res<camera::CameraSettings>,
    decoded_world: Res<DecodedWorld>,
    windows: Query<&Window, With<PrimaryWindow>>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut last_lookup: Local<LastLookup>,
) {
    egui::Window::new("Block Inspector").show(contexts.ctx_mut(), |ui| {
        // The cursor is over a panel (maybe this one) rather than the
        // world — showing whatever's behind it would be misleading.
        if egui_input.pointer {
            ui.label("(hover the world to inspect a block)");
            return;
        }
        let Ok((camera, camera_transform)) = camera_query.get_single() else {
            ui.label("No camera.");
            return;
        };
        let Ok(window) = windows.get_single() else {
            ui.label("No window.");
            return;
        };
        let Some(cursor) = window.cursor_position() else {
            ui.label("(cursor outside the window)");
            return;
        };

        let Some(voxel) = camera::block_under_cursor(
            camera,
            camera_transform,
            cursor,
            settings.max_ray_distance,
            &decoded_world,
        ) else {
            ui.label("(not pointing at any loaded terrain)");
            return;
        };

        ui.label(format!("Block: {}, {}, {}", voxel.x, voxel.y, voxel.z));

        // Straight out of `DecodedWorld`'s decoded biome grid (ticket 012),
        // unlike the block name/properties below — the fastest way to
        // sanity-check that the biome decode is correct against a save you
        // know is checking it live against the block under the cursor.
        if let Some(biome) = biome_under(voxel, &decoded_world) {
            ui.label(format!("Biome: {biome}"));
        }

        if last_lookup.0.as_ref().map(|(v, _)| *v) != Some(voxel) {
            let found = region_cache
                .as_deref()
                .and_then(|cache| lookup_block(voxel, cache));
            last_lookup.0 = Some((voxel, found));
        }

        match last_lookup.0.as_ref().and_then(|(_, found)| found.as_ref()) {
            Some((name, properties)) => {
                ui.label(format!("Name: {name}"));
                if properties.is_empty() {
                    ui.label("Properties: (none)");
                } else {
                    ui.label("Properties:");
                    for property in properties {
                        ui.label(format!("  {property}"));
                    }
                }
            }
            None => {
                ui.label("(could not read this block's raw NBT)");
            }
        }
    });
}

/// The biome name at `voxel` (Minecraft block coordinates), read straight
/// out of [`DecodedWorld`]'s decoded biome grid (ticket 012) — unlike
/// [`lookup_block`], there's no raw NBT path to cross-check against here,
/// so this is the decoder's own answer. `None` covers the voxel's column or
/// section not being loaded, same as `camera`'s own world-coordinate block
/// lookups.
fn biome_under(voxel: IVec3, decoded_world: &DecodedWorld) -> Option<String> {
    let size = world::SECTION_SIZE as i32;
    let column = decoded_world
        .columns
        .get(&(voxel.x.div_euclid(size), voxel.z.div_euclid(size)))?;
    let section_y = voxel.y.div_euclid(size) as i8;
    let section = column.sections.iter().find(|s| s.y == section_y)?;

    let local_x = voxel.x.rem_euclid(size) as usize;
    let local_y = voxel.y.rem_euclid(size) as usize;
    let local_z = voxel.z.rem_euclid(size) as usize;
    let id = section.biome_at(local_x, local_y, local_z);

    let registry = decoded_world.biomes.lock().expect("biome registry mutex poisoned");
    Some(registry.name(id).to_string())
}

/// Resolves `voxel` (Minecraft block coordinates) through the shared
/// [`region_cache::RegionCache`](crate::region_cache::RegionCache) and
/// `ChunkRegion::get_block`, on the main thread — see the module's docs for
/// why this goes through the raw NBT path rather than [`DecodedWorld`].
/// `None` covers both "the region isn't in this save" and any of
/// `get_block`'s own error cases (missing chunk, malformed section, ...).
fn lookup_block(voxel: IVec3, region_cache: &SharedRegionCache) -> BlockLookup {
    let region_width = REGION_WIDTH_IN_CHUNKS as i32 * world::SECTION_SIZE as i32;
    let region_coord = (
        voxel.x.div_euclid(region_width),
        voxel.z.div_euclid(region_width),
    );
    let local_x = voxel.x.rem_euclid(region_width) as usize;
    let local_z = voxel.z.rem_euclid(region_width) as usize;

    let mut cache = region_cache.0.lock().expect("region cache mutex poisoned");
    let region = cache.get_or_load(region_coord).ok()?;
    let field = region.get_block(local_x, voxel.y, local_z).ok()?;

    let name = field
        .get_string("Name")
        .cloned()
        .unwrap_or_else(|| "?".to_string());
    let properties = field
        .get_compound("Properties")
        .map(|fields| {
            fields
                .iter()
                .filter_map(|f| f.as_string().map(|v| format!("{} = {}", f.name, v)))
                .collect()
        })
        .unwrap_or_default();
    Some((name, properties))
}
