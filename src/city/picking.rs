//! Screen ray → block coordinate for the citybuilder (ticket 045, roadmap
//! E1). [`HoveredBlock`] is the single per-frame answer to "what block is
//! the RTS camera's cursor over right now" — `city::placement` (ticket 047,
//! roadmap E3) is the first real reader, feeding it into both E2's
//! `fit_footprint` and its own occupancy check every frame to place and
//! validity-tint the ghost preview.
//!
//! Reuses [`camera::block_under_cursor`] rather than a second raycast — the
//! same DDA ray-march the viewer's block inspector (007) and orbit re-aim
//! (006) already use, so every caller in the crate agrees on what's under
//! the cursor.
//!
//! ## Selection (ticket 083, roadmap G3)
//!
//! [`SelectedBuilding`] is the other thing a click means once
//! [`super::tool::ActiveTool::Inspect`] is the resting state:
//! [`update_selected_building`] resolves a left-click against
//! [`super::state::City::occupant_at`] — the same lookup
//! `super::demolish::resolve_demolition_target` uses to find `Delete`'s
//! target — and [`super::ui::inspect_panel`] is the sole reader. A building
//! tile selects it, a road tile or empty ground clears it, and a *missed*
//! click (nothing hovered at all) leaves whatever was selected alone — the
//! same "a missed click does nothing" precedent the viewer's own selection
//! box (ticket 020) set.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::{camera, DecodedWorld};

use super::loading::GameplaySet;
use super::state::{self, BuildingId};
use super::tool::ActiveTool;

/// The Minecraft block coordinate the camera's cursor is currently over,
/// updated once a frame by [`update_hovered_block`]. `None` covers both "the
/// cursor is outside the window" and "the ray hit nothing loaded/solid" —
/// [`camera::block_under_cursor`] already collapses those into one `None`,
/// and nothing downstream needs them told apart yet.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HoveredBlock(pub Option<IVec3>);

pub struct PickingPlugin;

/// Where [`update_hovered_block`] runs, so `city::placement` (ticket 047,
/// roadmap E3) can order its ghost preview after this frame's
/// [`HoveredBlock`] rather than reading last frame's — the same reason
/// [`update_hovered_block`] itself orders after `camera::CameraSet`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PickingSet;

/// Which placed building [`ActiveTool::Inspect`] currently has selected —
/// `None` when nothing is. See the module docs' "Selection".
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SelectedBuilding(pub Option<BuildingId>);

impl Plugin for PickingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HoveredBlock>()
            .init_resource::<SelectedBuilding>()
            // After `camera::CameraSet` so this reads the camera's transform
            // once this frame's pan/rotate/zoom has already landed, not the
            // previous frame's — the same reason `viewer::run` orders
            // `CameraSet` relative to the UI panels that read it.
            .add_systems(Update, update_hovered_block.in_set(PickingSet).after(camera::CameraSet).in_set(GameplaySet))
            // After `PickingSet` for the same reason `city::commit`/
            // `city::demolish` order there — this needs *this* frame's
            // `HoveredBlock`, not last frame's.
            .add_systems(Update, update_selected_building.after(PickingSet).in_set(GameplaySet));
    }
}

fn update_hovered_block(
    mut hovered: ResMut<HoveredBlock>,
    camera_query: Query<(&Camera, &GlobalTransform), With<camera::CameraRig>>,
    settings: Res<camera::CameraSettings>,
    decoded_world: Res<DecodedWorld>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    hovered.0 = (|| {
        let (camera, camera_transform) = camera_query.get_single().ok()?;
        let cursor = windows.get_single().ok()?.cursor_position()?;
        camera::block_under_cursor(
            camera,
            camera_transform,
            cursor,
            settings.max_ray_distance,
            &decoded_world,
        )
    })();
}

/// What [`SelectedBuilding`] should become after a click at `hovered`'s tile
/// — `None` (leave the current selection alone) for a missed click, `Some`
/// otherwise: `Some(Some(id))` on a building tile, `Some(None)` on a road
/// tile or empty ground. A plain function, not a system, so it's directly
/// testable against a bare [`state::City`] — the same split
/// `city::demolish::resolve_demolition_target` uses for the same reason.
fn resolve_selection(hovered: Option<IVec3>, city: &state::City) -> Option<Option<BuildingId>> {
    let hovered = hovered?;
    let tile = IVec2::new(hovered.x, hovered.z);
    Some(match city.occupant_at(tile) {
        Some(state::Occupant::Building(id)) => Some(id),
        _ => None,
    })
}

/// Left-click on [`ActiveTool::Inspect`]: resolves this frame's
/// [`HoveredBlock`] into a new [`SelectedBuilding`] — see [`resolve_selection`]
/// and the module docs' "Selection".
fn update_selected_building(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_input: Res<camera::EguiInputCapture>,
    hovered: Res<HoveredBlock>,
    city: Res<state::City>,
    tool: Option<Res<ActiveTool>>,
    mut selected: ResMut<SelectedBuilding>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::Inspect)) {
        return;
    }
    if egui_input.pointer || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if let Some(new) = resolve_selection(hovered.0, &city) {
        selected.0 = new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;

    #[test]
    fn a_missed_click_leaves_the_selection_alone() {
        let city = state::City::default();
        assert_eq!(resolve_selection(None, &city), None);
    }

    #[test]
    fn a_building_tile_selects_it() {
        let mut city = state::City::default();
        let id = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        assert_eq!(resolve_selection(Some(IVec3::new(0, 64, 0)), &city), Some(Some(id)));
    }

    #[test]
    fn a_road_tile_clears_the_selection() {
        use super::super::road::RoadPieceVariant;
        let mut city = state::City::default();
        city.add_road_cell(IVec2::new(0, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        assert_eq!(resolve_selection(Some(IVec3::new(2, 64, 2)), &city), Some(None));
    }

    #[test]
    fn empty_ground_clears_the_selection() {
        let city = state::City::default();
        assert_eq!(resolve_selection(Some(IVec3::new(5, 64, 5)), &city), Some(None));
    }
}
