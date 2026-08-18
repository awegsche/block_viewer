//! Which tool the player has active — building placement, road building, or
//! terraforming (ticket 055, roadmap F2; extended to a third tool by ticket
//! 057, roadmap H1). All three share the same left-click/hover machinery
//! (`picking::HoveredBlock`, `camera::EguiInputCapture`) but drive completely
//! different previews and commits (`city::placement`/`city::commit` for
//! buildings, `city::road_build` for roads, `city::terraform` for dig/level);
//! [`ActiveTool`] is the one bit that decides which of the three a click and
//! a hover mean this frame, so only one ever reacts to the same input at
//! once.
//!
//! ## `T` cycles; nothing switches it automatically
//!
//! [`toggle_tool`] is the whole mechanic — the same keyboard-stand-in role
//! ticket 047's number keys/`R`/`Escape` play for "which building" until a
//! real tool palette exists. `Building -> Road -> Terraform -> Building`,
//! wrapping rather than a two-way flip now that there are three. Deliberately
//! *not* wired to also flip back to [`ActiveTool::Building`] on a number-key
//! or build-menu pick (both would need a `ResMut<ActiveTool>` threaded into
//! two more call sites for a rough edge — press `T` again after picking a
//! building while in road mode — that is easy to explain and costs nothing to
//! live with, unlike a second, implicit place this resource gets mutated
//! from).
//!
//! ## Optional almost everywhere it's read
//!
//! `placement::resolve_ghost`/`commit::try_commit_placement`/
//! `road_build`/`terraform`'s own systems all read this through
//! `Option<Res<ActiveTool>>`, defaulting to [`ActiveTool::Building`] when the
//! resource is absent — the same tolerant shape
//! [`crate::chunk_pipeline::SharedAtlasIndex`] and friends already get from
//! `placement`'s own systems, so a minimal test `App` that never adds
//! [`ToolPlugin`] keeps behaving exactly like it did before this ticket
//! (building tools only, no road or terraform tool to switch away from).

use bevy::prelude::*;

use crate::camera;

/// Which tool a click and the hovered tile drive right now.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveTool {
    #[default]
    Building,
    Road,
    /// Ticket 057, roadmap H1: dig and level — see `city::terraform`.
    Terraform,
}

pub struct ToolPlugin;

impl Plugin for ToolPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveTool>().add_systems(Update, toggle_tool);
    }
}

/// `T` cycles [`ActiveTool`] — guarded by [`camera::EguiInputCapture`] the
/// same way every other keyboard stand-in in this game is, so typing into an
/// egui panel never also switches tools underneath it.
fn toggle_tool(keys: Res<ButtonInput<KeyCode>>, egui_input: Res<camera::EguiInputCapture>, mut tool: ResMut<ActiveTool>) {
    if egui_input.keyboard {
        return;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        *tool = match *tool {
            ActiveTool::Building => ActiveTool::Road,
            ActiveTool::Road => ActiveTool::Terraform,
            ActiveTool::Terraform => ActiveTool::Building,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_test_app() -> App {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>().init_resource::<camera::EguiInputCapture>().add_plugins(ToolPlugin);
        app
    }

    fn press(app: &mut App, key: KeyCode) {
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(key);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().release(key);
    }

    #[test]
    fn defaults_to_building() {
        let app = tool_test_app();
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Building);
    }

    #[test]
    fn t_cycles_building_road_terraform_and_back() {
        let mut app = tool_test_app();
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Road);
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Terraform);
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Building);
    }

    #[test]
    fn egui_capturing_the_keyboard_suppresses_the_toggle() {
        let mut app = tool_test_app();
        app.world_mut().resource_mut::<camera::EguiInputCapture>().keyboard = true;
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Building);
    }
}
