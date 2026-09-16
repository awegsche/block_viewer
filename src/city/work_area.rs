//! Drawing a gatherer's working area (ticket 111): the drag that sets
//! [`PlacedBuilding::work_area`], and the gizmo outline that shows it.
//!
//! ## Why the player draws it
//!
//! Ticket 086's hut dug everything within `Gatherer::radius_blocks` of
//! itself from the moment it was placed — including, as it turned out, the
//! road next to it. `city::gatherer` now refuses any tile the city has
//! claimed (that's the bug fix, and it lives there), but the deeper problem
//! was that the hut decided *where* to dig on its own. Now it doesn't: a hut
//! does nothing until the player has drawn a rectangle for it, and digs only
//! inside that. `radius_blocks` survives as the cap on how far from the hut
//! that rectangle may reach ([`WorkArea::clamp_to_reach`]).
//!
//! ## A fifth tool, entered from a button, leaving by itself
//!
//! [`ActiveTool::DrawWorkArea`] is entered only from the inspect panel's
//! "Draw working area" button, with the hut already selected
//! ([`SelectedBuilding`]) — the area belongs to *that* building, so the
//! panel is the one place that knows which one. It leaves by itself: a
//! released drag commits and returns to [`ActiveTool::Inspect`] with the
//! selection intact (so the panel is still open, now showing the area), and
//! `Escape` returns without changing anything. `T` abandons it into the
//! build cycle like it does from `Inspect`; see `tool`'s own docs.
//!
//! The drag itself is `terraform::update_drag_state`'s click/hold/release
//! shape, corner to corner ([`WorkArea::new`]) — a single click is a one-tile
//! area, which is legal if pointless. The panel button's own click frame has
//! `EguiInputCapture::pointer` set, so it can never double as the drag's
//! first press.
//!
//! ## What gets drawn, and why with gizmos
//!
//! Three outlines, all as tile rings hugging the terrain (each perimeter
//! tile's top face, at that tile's own `topmost_block_y + 1`) rather than a
//! flat rectangle at one height — a flat outline on a hillside is unreadable
//! for the same reason ticket 026 gave the selection box its two passes:
//!
//! - the hut's **committed area**, whenever a gatherer is selected under
//!   any tool, so the panel's numbers have a picture;
//! - while drawing, the **reach box** (footprint expanded by
//!   `radius_blocks`), dimmed, so the clamp reads as a limit the player can
//!   see rather than a surprise on release;
//! - while dragging, the **candidate** — already clamped, so what's shown
//!   is exactly what a release would commit.
//!
//! [`Gizmos`] rather than a mesh for the reason `selection::gizmo` gives:
//! the candidate changes every frame of a drag, and a mesh entity would need
//! a material, a despawn path and space in the same `Assets<Mesh>` the
//! streaming pipeline is churning through. Its own [`GizmoConfigGroup`] so
//! the depth bias stays scoped here.

use bevy::prelude::*;

use crate::camera;
use crate::DecodedWorld;

use super::definition::BuildingDefinitions;
use super::picking::{HoveredBlock, PickingSet, SelectedBuilding};
use super::state::{footprint_extent, City, PlacedBuilding, WorkArea};
use super::tool::ActiveTool;

/// Which tile a left click started dragging from, if any — `None` between
/// drags. See the module docs.
#[derive(Resource, Debug, Default)]
struct WorkAreaDragState {
    anchor: Option<IVec2>,
}

/// The one gizmo pass here — depth tested, so terrain occludes the part of
/// a ring behind a hill, which is what makes it read as lying *on* the
/// ground. Same tiny bias `selection::gizmo`'s solid pass uses, for the same
/// reason: every ring edge is coplanar with a block face.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct WorkAreaGizmos;

const DEPTH_BIAS: f32 = -0.0002;
const LINE_WIDTH: f32 = 2.5;

/// A committed area: the hut's own colour, solid.
const AREA_COLOR: Srgba = Srgba::rgb(0.3, 0.9, 0.4);
/// The candidate under the cursor mid-drag — brighter than the committed
/// area it's about to replace.
const CANDIDATE_COLOR: Srgba = Srgba::rgb(1.0, 0.95, 0.3);
/// The reach box — there to be noticed, not read.
const REACH_COLOR: Srgba = Srgba::new(1.0, 1.0, 1.0, 0.25);

pub struct WorkAreaPlugin;

impl Plugin for WorkAreaPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorkAreaDragState>()
            .insert_gizmo_config(WorkAreaGizmos, GizmoConfig { depth_bias: DEPTH_BIAS, line_width: LINE_WIDTH, ..default() })
            .add_systems(Update, (update_drag_state, try_commit_drag, cancel_on_escape, draw_outlines).chain().after(PickingSet));
    }
}

// -----------------------------------------------------------------------------------------------
// ---- pure geometry: the hut's reach, and the rectangle a drag commits -------------------------
// -----------------------------------------------------------------------------------------------

/// The footprint rectangle (`max` exclusive) and reach radius of `placed`,
/// or `None` if it isn't a gatherer at all — the two things both the clamp
/// and the reach-box outline need.
fn reach_of(placed: &PlacedBuilding, definitions: &BuildingDefinitions) -> Option<(IVec2, IVec2, i32)> {
    let definition = definitions.get(placed.definition_id.as_deref()?)?;
    let gatherer = definition.building.gatherer.as_ref()?;
    let min = IVec2::new(placed.origin.x, placed.origin.z);
    let max = min + footprint_extent(placed.footprint, placed.rotation);
    Some((min, max, gatherer.radius_blocks as i32))
}

/// The reach box itself, as an inclusive [`WorkArea`] — what a drawn area
/// is clamped into, and what's outlined while drawing.
fn reach_box(footprint_min: IVec2, footprint_max: IVec2, radius: i32) -> WorkArea {
    WorkArea::new(footprint_min - IVec2::splat(radius), footprint_max + IVec2::splat(radius) - IVec2::ONE)
}

/// What a drag from `anchor` to `end` would commit for `placed`: the
/// rectangle they span, clamped to the hut's reach. `None` if `placed`
/// isn't a gatherer or the rectangle lies entirely out of reach — either
/// way, a release changes nothing.
fn candidate(anchor: IVec2, end: IVec2, placed: &PlacedBuilding, definitions: &BuildingDefinitions) -> Option<WorkArea> {
    let (min, max, radius) = reach_of(placed, definitions)?;
    WorkArea::new(anchor, end).clamp_to_reach(min, max, radius)
}

// -----------------------------------------------------------------------------------------------
// ---- input: the drag, committing, cancelling ----------------------------------------------------
// -----------------------------------------------------------------------------------------------

/// Starts, tracks and clears [`WorkAreaDragState::anchor`] — the same
/// click/hold/release shape `terraform::update_drag_state` uses.
fn update_drag_state(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_input: Res<camera::EguiInputCapture>,
    tool: Option<Res<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    mut drag: ResMut<WorkAreaDragState>,
) {
    if !matches!(tool.as_deref(), Some(ActiveTool::DrawWorkArea)) || egui_input.pointer {
        drag.anchor = None;
        return;
    }
    if mouse.just_pressed(MouseButton::Left) {
        if let Some(hovered) = hovered.0 {
            drag.anchor = Some(IVec2::new(hovered.x, hovered.z));
        }
    }
}

/// Left-click release with the tool active: commits the clamped rectangle
/// to the selected building and returns to [`ActiveTool::Inspect`] — see
/// the module docs. A release that would commit nothing (nothing hovered,
/// out of reach, selection gone) still returns to `Inspect`: the player let
/// go, the drawing is over either way.
fn try_commit_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    tool: Option<ResMut<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    selected: Res<SelectedBuilding>,
    definitions: Res<BuildingDefinitions>,
    mut city: ResMut<City>,
    mut drag: ResMut<WorkAreaDragState>,
) {
    let Some(mut tool) = tool else { return };
    if *tool != ActiveTool::DrawWorkArea || !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(anchor) = drag.anchor.take() else { return };
    *tool = ActiveTool::Inspect;

    let Some(id) = selected.0 else { return };
    let Some(hovered) = hovered.0 else { return };
    let Some(placed) = city.building(id) else { return };
    let Some(area) = candidate(anchor, IVec2::new(hovered.x, hovered.z), placed, &definitions) else { return };
    city.set_work_area(id, Some(area));
}

/// `Escape` while drawing: back to [`ActiveTool::Inspect`], nothing
/// changed. `placement::cycle_selection` also sends `Escape` to `Inspect`
/// (it clears any placement on the way) — harmless twice over; this one
/// exists so the drag's anchor is dropped with it.
fn cancel_on_escape(
    keys: Res<ButtonInput<KeyCode>>,
    egui_input: Res<camera::EguiInputCapture>,
    tool: Option<ResMut<ActiveTool>>,
    mut drag: ResMut<WorkAreaDragState>,
) {
    let Some(mut tool) = tool else { return };
    if *tool != ActiveTool::DrawWorkArea || egui_input.keyboard || !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    drag.anchor = None;
    *tool = ActiveTool::Inspect;
}

// -----------------------------------------------------------------------------------------------
// ---- the outlines ---------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------

/// Draws the selected gatherer's committed area, and — while the tool is
/// active — its reach box and the drag's candidate. See the module docs.
#[allow(clippy::too_many_arguments)]
fn draw_outlines(
    mut gizmos: Gizmos<WorkAreaGizmos>,
    selected: Res<SelectedBuilding>,
    city: Res<City>,
    definitions: Res<BuildingDefinitions>,
    tool: Option<Res<ActiveTool>>,
    hovered: Res<HoveredBlock>,
    drag: Res<WorkAreaDragState>,
    world: Res<DecodedWorld>,
) {
    let Some(id) = selected.0 else { return };
    let Some(placed) = city.building(id) else { return };
    let Some((min, max, radius)) = reach_of(placed, &definitions) else { return };

    if let Some(area) = placed.work_area {
        draw_ring(&mut gizmos, area, &world, AREA_COLOR);
    }
    if !matches!(tool.as_deref(), Some(ActiveTool::DrawWorkArea)) {
        return;
    }
    draw_ring(&mut gizmos, reach_box(min, max, radius), &world, REACH_COLOR);
    if let (Some(anchor), Some(hovered)) = (drag.anchor, hovered.0) {
        if let Some(area) = candidate(anchor, IVec2::new(hovered.x, hovered.z), placed, &definitions) {
            draw_ring(&mut gizmos, area, &world, CANDIDATE_COLOR);
        }
    }
}

/// The outline of every tile on `area`'s perimeter, each on its own top
/// face — a ring that follows the ground. A tile whose chunk isn't decoded
/// is left out, the same silent skip the gatherer's own dig gives it.
fn draw_ring(gizmos: &mut Gizmos<'_, '_, WorkAreaGizmos>, area: WorkArea, world: &DecodedWorld, color: Srgba) {
    for tile in perimeter(area) {
        let Some(top) = super::terraform::topmost_block_y(tile, world) else { continue };
        let (min, max) = crate::selection::block_bevy_aabb(IVec3::new(tile.x, top, tile.y));
        let y = max.y;
        gizmos.linestrip(
            [
                Vec3::new(min.x, y, min.z),
                Vec3::new(max.x, y, min.z),
                Vec3::new(max.x, y, max.z),
                Vec3::new(min.x, y, max.z),
                Vec3::new(min.x, y, min.z),
            ],
            color,
        );
    }
}

/// Every tile on the edge of `area` — all of them for an area one tile wide
/// or deep, each exactly once.
fn perimeter(area: WorkArea) -> impl Iterator<Item = IVec2> {
    area.tiles().filter(move |tile| {
        tile.x == area.min.x || tile.x == area.max.x || tile.y == area.min.y || tile.y == area.max.y
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::{Building, Category, FootprintSpec, Gatherer, Integrity, LoadedBuilding};
    use crate::city::state::BuildingId;
    use std::path::PathBuf;

    fn definitions_with_hut(radius: u32) -> BuildingDefinitions {
        let building = Building {
            name: "Gatherer's Hut".to_string(),
            blueprint: "gatherer_hut.nbt".to_string(),
            tier: 1,
            requires: Vec::new(),
            footprint: FootprintSpec::FromBlueprint,
            production: None,
            cost: Vec::new(),
            warehouse: None,
            farm: None,
            gatherer: Some(Gatherer { radius_blocks: radius, blocks_per_minute: 1.0, buffer_stacks: 1, haul_at_stacks: None }),
            mine: None,
            category: Category::Production,
            ground_level: 0,
            integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
        };
        BuildingDefinitions::from_entries(vec![LoadedBuilding {
            id: "gatherer_hut".to_string(),
            path: PathBuf::new(),
            building,
            footprint: IVec2::splat(2),
            catalogue_id: "gatherer_hut".to_string(),
        }])
    }

    fn hut(definition_id: Option<&str>) -> PlacedBuilding {
        PlacedBuilding {
            catalogue_id: "gatherer_hut".to_string(),
            definition_id: definition_id.map(str::to_string),
            origin: IVec3::new(10, 64, 10),
            rotation: Rotation::Deg0,
            footprint: IVec2::splat(2),
            work_area: None,
        }
    }

    // --- reach / candidate ------------------------------------------------------------------------

    #[test]
    fn reach_of_reads_the_footprint_and_the_definitions_radius() {
        let definitions = definitions_with_hut(18);
        assert_eq!(reach_of(&hut(Some("gatherer_hut")), &definitions), Some((IVec2::new(10, 10), IVec2::new(12, 12), 18)));
    }

    #[test]
    fn reach_of_is_none_without_a_definition_or_a_gatherer_block() {
        let definitions = definitions_with_hut(18);
        assert_eq!(reach_of(&hut(None), &definitions), None);
        assert_eq!(reach_of(&hut(Some("not_a_definition")), &definitions), None);
    }

    #[test]
    fn reach_box_is_the_footprint_grown_by_the_radius_inclusive() {
        assert_eq!(reach_box(IVec2::new(10, 10), IVec2::new(12, 12), 3), WorkArea::new(IVec2::new(7, 7), IVec2::new(14, 14)));
    }

    #[test]
    fn candidate_is_the_drag_rectangle_clamped_to_reach() {
        let definitions = definitions_with_hut(3);
        let placed = hut(Some("gatherer_hut"));
        // Anchor inside reach, end far beyond it on both axes.
        let area = candidate(IVec2::new(9, 9), IVec2::new(30, 30), &placed, &definitions).unwrap();
        assert_eq!(area, WorkArea::new(IVec2::new(9, 9), IVec2::new(14, 14)));
        // Entirely out of reach: nothing.
        assert_eq!(candidate(IVec2::new(20, 20), IVec2::new(30, 30), &placed, &definitions), None);
    }

    // --- the drag, the commit, the cancel --------------------------------------------------------

    fn app_with_selected_hut() -> (App, BuildingId) {
        let mut app = App::new();
        let mut city = City::default();
        let id = city
            .place_building("gatherer_hut", Some("gatherer_hut".to_string()), IVec3::new(10, 64, 10), Rotation::Deg0, IVec2::splat(2))
            .unwrap();
        app.init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<camera::EguiInputCapture>()
            .init_resource::<HoveredBlock>()
            .init_resource::<WorkAreaDragState>()
            .insert_resource(SelectedBuilding(Some(id)))
            .insert_resource(ActiveTool::DrawWorkArea)
            .insert_resource(city)
            .insert_resource(definitions_with_hut(3))
            .add_systems(Update, (update_drag_state, try_commit_drag, cancel_on_escape).chain());
        (app, id)
    }

    fn hover(app: &mut App, tile: IVec2) {
        app.world_mut().resource_mut::<HoveredBlock>().0 = Some(IVec3::new(tile.x, 64, tile.y));
    }

    fn press_mouse(app: &mut App) {
        app.world_mut().resource_mut::<ButtonInput<MouseButton>>().press(MouseButton::Left);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<MouseButton>>().clear_just_pressed(MouseButton::Left);
    }

    fn release_mouse(app: &mut App) {
        app.world_mut().resource_mut::<ButtonInput<MouseButton>>().release(MouseButton::Left);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<MouseButton>>().clear_just_released(MouseButton::Left);
    }

    fn press_key(app: &mut App, key: KeyCode) {
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(key);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().release(key);
    }

    #[test]
    fn a_drag_commits_the_clamped_area_and_returns_to_inspect() {
        let (mut app, id) = app_with_selected_hut();
        hover(&mut app, IVec2::new(8, 8));
        press_mouse(&mut app);
        assert_eq!(app.world().resource::<WorkAreaDragState>().anchor, Some(IVec2::new(8, 8)));

        hover(&mut app, IVec2::new(40, 13));
        release_mouse(&mut app);

        let city = app.world().resource::<City>();
        assert_eq!(city.building(id).unwrap().work_area, Some(WorkArea::new(IVec2::new(8, 8), IVec2::new(14, 13))));
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect);
        assert_eq!(app.world().resource::<SelectedBuilding>().0, Some(id), "the panel stays open on the hut");
        assert_eq!(app.world().resource::<WorkAreaDragState>().anchor, None);
    }

    #[test]
    fn a_drag_entirely_out_of_reach_commits_nothing_but_still_ends_the_tool() {
        let (mut app, id) = app_with_selected_hut();
        let old = WorkArea::new(IVec2::new(9, 9), IVec2::new(11, 11));
        app.world_mut().resource_mut::<City>().set_work_area(id, Some(old));
        hover(&mut app, IVec2::new(30, 30));
        press_mouse(&mut app);
        hover(&mut app, IVec2::new(35, 35));
        release_mouse(&mut app);

        assert_eq!(app.world().resource::<City>().building(id).unwrap().work_area, Some(old), "unchanged");
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect);
    }

    #[test]
    fn escape_cancels_the_drag_and_returns_to_inspect_without_changing_the_area() {
        let (mut app, id) = app_with_selected_hut();
        hover(&mut app, IVec2::new(8, 8));
        press_mouse(&mut app);
        press_key(&mut app, KeyCode::Escape);

        assert_eq!(app.world().resource::<WorkAreaDragState>().anchor, None);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect);
        assert_eq!(app.world().resource::<City>().building(id).unwrap().work_area, None);
    }

    #[test]
    fn egui_owning_the_pointer_never_starts_a_drag() {
        let (mut app, _) = app_with_selected_hut();
        app.world_mut().resource_mut::<camera::EguiInputCapture>().pointer = true;
        hover(&mut app, IVec2::new(8, 8));
        press_mouse(&mut app);
        assert_eq!(app.world().resource::<WorkAreaDragState>().anchor, None);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::DrawWorkArea, "still drawing — nothing was released");
    }

    #[test]
    fn a_release_under_another_tool_does_nothing() {
        let (mut app, id) = app_with_selected_hut();
        *app.world_mut().resource_mut::<ActiveTool>() = ActiveTool::Inspect;
        hover(&mut app, IVec2::new(8, 8));
        press_mouse(&mut app);
        release_mouse(&mut app);
        assert_eq!(app.world().resource::<City>().building(id).unwrap().work_area, None);
    }

    // --- perimeter ---------------------------------------------------------------------------------

    #[test]
    fn perimeter_is_the_edge_tiles_once_each() {
        let area = WorkArea::new(IVec2::new(0, 0), IVec2::new(3, 2));
        let tiles: Vec<IVec2> = perimeter(area).collect();
        assert_eq!(tiles.len(), 10, "a 4x3 ring: everything but the 2x1 middle");
        assert!(!tiles.contains(&IVec2::new(1, 1)));
        assert!(!tiles.contains(&IVec2::new(2, 1)));
        assert_eq!(perimeter(WorkArea::new(IVec2::ONE, IVec2::ONE)).count(), 1);
        assert_eq!(perimeter(WorkArea::new(IVec2::new(0, 0), IVec2::new(5, 0))).count(), 6, "a line is all edge");
    }
}
