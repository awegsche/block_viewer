//! Which tool the player has active — inspecting a placed building, building
//! placement, road building, or terraforming (ticket 055, roadmap F2;
//! extended to a third tool by ticket 057, roadmap H1; a fourth, the resting
//! state, by ticket 083). All four share the same left-click/hover machinery
//! (`picking::HoveredBlock`, `camera::EguiInputCapture`) but drive completely
//! different previews and commits (`city::placement`/`city::commit` for
//! buildings, `city::road_build` for roads, `city::terraform` for dig/level,
//! `city::picking::SelectedBuilding` for inspecting); [`ActiveTool`] is the
//! one bit that decides which of the four a click and a hover mean this
//! frame, so only one ever reacts to the same input at once.
//!
//! ## Inspect is the resting state, not a fourth stop on the cycle
//!
//! Ticket 083: [`ActiveTool::Inspect`] is `#[default]` — left-click means
//! "select a placed building" until the player has actually asked to build
//! or dig something, rather than always committing whatever 047's keyboard
//! stand-in (now 082's build menu) last had selected. It's reached by
//! clearing a placement (`Escape`) or, implicitly, at startup; nothing here
//! ever cycles a player *into* it, which is also why [`toggle_tool`]'s match
//! is one arm short of the four-way symmetry that might otherwise suggest —
//! see below.
//!
//! ## `T` cycles the other three; nothing switches it automatically
//!
//! [`toggle_tool`] is the whole mechanic — the same keyboard-stand-in role
//! ticket 047's number keys/`R`/`Escape` play for "which building" until a
//! real tool palette exists. `Building -> Road -> Terraform -> Building`,
//! wrapping rather than a two-way flip now that there are three of them.
//! [`ActiveTool::Inspect`] isn't a stop on that cycle — pressing `T` while
//! inspecting enters the cycle at `Building` (the same place a fresh build
//! menu click would have put a player anyway) rather than looping back to
//! `Inspect`, which is why the cycle's own wraparound still only touches the
//! three build-ish tools. Deliberately *not* wired to also flip back to
//! [`ActiveTool::Building`] on a number-key or build-menu pick (both would
//! need a `ResMut<ActiveTool>` threaded into two more call sites for a rough
//! edge — press `T` again after picking a building while in road mode — that
//! is easy to explain and costs nothing to live with, unlike a second,
//! implicit place this resource gets mutated from).
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
    /// Ticket 083, roadmap G3: the resting state — a left-click selects a
    /// placed building (`city::picking::SelectedBuilding`) rather than
    /// committing a placement. See the module docs' "Inspect is the resting
    /// state".
    #[default]
    Inspect,
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
/// egui panel never also switches tools underneath it. [`ActiveTool::Inspect`]
/// enters the cycle at `Building`; the cycle itself never returns to it — see
/// the module docs.
fn toggle_tool(keys: Res<ButtonInput<KeyCode>>, egui_input: Res<camera::EguiInputCapture>, mut tool: ResMut<ActiveTool>) {
    if egui_input.keyboard {
        return;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        *tool = match *tool {
            ActiveTool::Inspect => ActiveTool::Building,
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
    fn defaults_to_inspect() {
        let app = tool_test_app();
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect);
    }

    #[test]
    fn t_cycles_building_road_terraform_and_back() {
        let mut app = tool_test_app();
        // Starts from `Building`, not the default (`Inspect`, ticket 083) —
        // the cycle under test here is the three build-ish tools, not how
        // one gets into it.
        *app.world_mut().resource_mut::<ActiveTool>() = ActiveTool::Building;
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Road);
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Terraform);
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Building);
    }

    /// Ticket 083: `Inspect` isn't a stop on the cycle, but `T` still has to
    /// do *something* useful from it — enters at `Building`, the same place a
    /// fresh build-menu click would have put the player anyway.
    #[test]
    fn t_from_inspect_enters_the_cycle_at_building() {
        let mut app = tool_test_app();
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect);
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Building);
    }

    #[test]
    fn egui_capturing_the_keyboard_suppresses_the_toggle() {
        let mut app = tool_test_app();
        app.world_mut().resource_mut::<camera::EguiInputCapture>().keyboard = true;
        press(&mut app, KeyCode::KeyT);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect, "the default, untouched");
    }
}
