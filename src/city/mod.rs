//! The `citybuilder` game (ticket 027) — milestone M1: a window on a real
//! save with the existing streaming and camera, and nothing else yet.
//!
//! The plan it grows into is `tickets/CITYBUILDER_ROADMAP.md`. The rule the
//! rest of it follows, worth repeating where the code will live: **the city
//! state is authoritative, and the blocks in the world are a projection of
//! it**. [`state::City`] (ticket 042, roadmap D1) is that state — see its
//! module docs for the full rule and what still isn't built on top of it.
//!
//! What this deliberately does *not* add, and why it looks so empty:
//!
//! - **No selection.** [`crate::selection`] is shared, but the box and its
//!   drag handles are the explorer's affordance. The citybuilder's own
//!   picking is roadmap task E1.
//!
//! The camera is the viewer's free-flight rig for now. An RTS camera (E1)
//! replaces it, and is free to drop the flight controls the viewer has to
//! keep.
//!
//! ## Render depth (ticket 030, roadmap R1)
//!
//! The one way this binary's world differs from `block_viewer`'s: [`run`]
//! overrides [`chunk_pipeline::RenderFloor`] with
//! [`world::decode::FloorPolicy::BelowSurface`]. The RTS camera this game
//! is built around looks at the surface from above and never goes
//! underground, so the ~7 sections per column below the terrain — caves
//! included — are never decoded or meshed. `block_viewer` keeps
//! [`world_app`]'s default of [`world::decode::FloorPolicy::WholeWorld`],
//! since it's the explore-a-save app and the underground is exactly what it
//! exists to show.
//!
//! ## The building catalogue (ticket 039, roadmap B4)
//!
//! [`run`] loads `assets/city/blueprints` through
//! [`blueprint::load_catalogue_dir`] before the app starts, the same
//! synchronous-before-`App::new()` shape [`world_app`] uses for
//! [`crate::LoadedSave`] — this is citybuilder-specific (unlike `blueprint`
//! itself, which `block_viewer` also depends on), so it's wired in here
//! rather than in the shared [`world_app`]. Nothing reads
//! [`blueprint::BuildingCatalogue`] yet — G1's build menu is the eventual
//! consumer — so this is only proving the load path end to end, the same
//! way ticket 024 proved the write path before anything used it.
//!
//! ## Building definitions (ticket 040, roadmap C1)
//!
//! Right after the catalogue, [`run`] loads `assets/city/buildings` through
//! [`definition::load_definitions_dir`] — the game data (tier, cost,
//! production, integrity thresholds) a catalogue entry's shape doesn't
//! carry. It's handed the catalogue so it can cross-check each definition's
//! `blueprint` field against a real entry rather than trusting the filename.
//! Same "no consumer yet" state the catalogue landed in: G1's build menu is
//! what will eventually read [`definition::BuildingDefinitions`].
//!
//! ## City state (ticket 042, roadmap D1)
//!
//! [`run`] inserts [`state::City`] — placed buildings, roads, and the
//! footprint occupancy grid both are checked against. Nothing calls
//! [`state::City::place_building`] yet, since that needs picking (E1) and
//! terrain fit (E2) first — the "proven, not yet used" state the catalogue
//! and definitions landed in, for that half of the type.
//!
//! ## City persistence (ticket 043, roadmap D2)
//!
//! Unlike the catalogue and definitions, [`state::City`] does have a real
//! caller either side of "proven, not yet used": [`load_city`] reads
//! `<save>/citybuilder/city.ron` through [`persistence::load_city`] before
//! `App::run()`, and [`save_city_on_exit`] writes it back through
//! [`persistence::save_city`] on [`AppExit`]. Both are no-ops (an empty city
//! in, nothing to save out) until E1-E4 give something a reason to call
//! `place_building` — but the round trip itself, and the file path it reads
//! and writes, are exercised by the real app lifecycle now, not only by
//! [`persistence`]'s own unit tests. A save with no real world loaded (ticket
//! 008's `empty_save` placeholder) skips persistence entirely via
//! [`CitySavePath`] — there's no save root to read from or write to.
//!
//! ## The journal (ticket 044, roadmap D3)
//!
//! [`run`] loads and saves [`journal::Journal`] the same way, and right
//! alongside `city.ron` — `<save>/citybuilder/journal.ron`, via
//! [`load_journal`]/[`save_journal_on_exit`]. It's the same "proven by the
//! app lifecycle, not yet fed by gameplay" state ticket 043 landed
//! [`state::City`] in: nothing calls [`journal::Journal::record_placement`]
//! yet, since that needs E4's commit, so every real run loads and saves an
//! empty journal — but the round trip, and the file path it reads and
//! writes, are exercised now rather than only by [`journal`]'s own unit
//! tests.
//!
//! ## Ghost preview (ticket 047, roadmap E3)
//!
//! [`placement::PlacementPlugin`] is the first thing in the game that draws
//! anything building-shaped: at the tile [`picking::HoveredBlock`] is over,
//! it shows the currently selected catalogue entry's mesh, translucent and
//! tinted green or red by whether [`grid::fit_footprint`] (E2) and
//! [`state::City::is_tile_free`] (D1) both agree it could actually go there.
//! G1's build menu doesn't exist yet, so a small keyboard stand-in drives
//! *which* building is selected — see [`placement`]'s module docs.
//! [`placement::PlacementSelection`] also carries a manual height offset
//! (`Page Up`/`Page Down`/`Home`), added by ticket 048 alongside commit.
//!
//! ## Commit (ticket 048, roadmap E4)
//!
//! [`commit::CommitPlugin`] is what turns a valid (green) ghost into a real
//! building: a left click claims the tile in [`state::City`], writes the
//! rotated blueprint through the real write path (W4-W6) on
//! `AsyncComputeTaskPool`, and — once that write actually succeeds — records
//! the as-built baseline in [`journal::Journal`] (roadmap I1, landing here
//! per the roadmap's own instruction to ship it with E4) and fires
//! `ChunksEdited` so the building appears without a restart (W7). A failed
//! write rolls the city entry back rather than leaving a phantom building
//! behind — see [`commit`]'s module docs for the whole transaction. This is
//! the first thing in the game that writes to the save at all; everything
//! before it only read.
//!
//! ## Demolish (ticket 049, roadmap E5)
//!
//! [`demolish::DemolishPlugin`] is commit's inverse: `Delete` on a hovered,
//! placed building removes it from [`state::City`] and writes the terrain
//! its own placement baseline (roadmap I1) says stood there back through the
//! write path — no re-derivation from the blueprint, the baseline already
//! *is* the answer. It also lands [`journal::Journal::record_demolition`]'s
//! first real caller, the `Demolished` half of D3's journal (ticket 044)
//! that had existed, unused, since then. Unlike commit, the `City` removal
//! happens *after* the write succeeds rather than before — see
//! [`demolish`]'s module docs for why that ordering, not commit's, is the
//! one that's safe here. [`write_gate::WriteGate`] is a small resource
//! shared with [`commit::CommitPlugin`] so the two can't both have a
//! `WriteSession` open at once.
//!
//! ## The build menu and city panel (ticket 050, roadmap G)
//!
//! [`ui::UiPlugin`] is the citybuilder's first real UI — its own
//! `EguiPlugin` registration, not [`crate::viewer::ui::UiPlugin`] (see
//! [`ui`]'s own module docs for why a second one). Two windows:
//! [`ui`]'s build menu replaces ticket 047's number-key stand-in for
//! *picking* a catalogue entry (rotation/height/clear stay on the keyboard),
//! grouped by tier with locked entries shown, disabled, and naming what
//! unlocks them — the roadmap's own G1 wording, word for word. Its city
//! panel is G2's: building counts, road length, and the last write's own
//! outcome ([`write_status::WriteStatus`], a new small resource
//! `commit`/`demolish` both record into), plus an "Undo" button —
//! [`journal::Journal::undo_last`]'s promised caller, run through the write
//! path by the new [`undo::UndoPlugin`], the same `request`/`busy`/`state`
//! shape `commit`/`demolish` already use.

use std::path::{Path, PathBuf};

mod commit;
mod definition;
mod demolish;
mod grid;
mod journal;
mod persistence;
mod picking;
mod placement;
mod state;
mod ui;
mod undo;
mod write_gate;
mod write_status;

use bevy::app::AppExit;
use bevy::prelude::*;

use crate::{
    blueprint,
    camera::{self, CameraMode, CameraStartMode},
    chunk_pipeline::RenderFloor,
    world::decode::FloorPolicy,
    world_app, LoadedSave,
};

/// Where [`run`] looks for building blueprints — the flat, non-recursive
/// directory the roadmap's `assets/city/blueprints/*.nbt` glob names.
const CATALOGUE_DIR: &str = "assets/city/blueprints";

/// Where [`run`] looks for building definitions — the flat, non-recursive
/// directory the roadmap's C1 sketch implies alongside `CATALOGUE_DIR`.
const DEFINITIONS_DIR: &str = "assets/city/buildings";

/// Where [`save_city`](persistence::save_city)/[`load_city`](persistence::load_city)
/// look, relative to a save's root — `None` when [`world_app`]'s
/// [`LoadedSave`] is ticket 008's placeholder (`empty_save`, `meta.path`
/// empty), which has nowhere on disk to read from or write to.
#[derive(Resource)]
struct CitySavePath(Option<PathBuf>);

/// Runs the citybuilder. Called by `src/bin/citybuilder.rs`, which is three
/// lines and nothing else.
pub fn run() {
    let catalogue = load_building_catalogue();
    let definitions = load_building_definitions(&catalogue);

    let mut app = world_app();
    let save_root = app.world().resource::<LoadedSave>().0.meta.path.clone();
    let city = load_city(&save_root);
    let journal = load_journal(&save_root);

    app.insert_resource(RenderFloor(FloorPolicy::BelowSurface { margin: 16 }))
        // Ticket 045, roadmap E1: pan/zoom/rotate over the terrain rather
        // than the viewer's free-flight rig — see `camera::CameraMode::Rts`.
        .insert_resource(CameraStartMode(CameraMode::Rts))
        .insert_resource(catalogue)
        .insert_resource(definitions)
        .insert_resource(city)
        .insert_resource(journal)
        .insert_resource(CitySavePath(if save_root.as_os_str().is_empty() { None } else { Some(save_root) }))
        .add_plugins(picking::PickingPlugin)
        .add_plugins(placement::PlacementPlugin)
        .add_plugins(commit::CommitPlugin)
        .add_plugins(demolish::DemolishPlugin)
        .add_plugins(undo::UndoPlugin)
        .add_plugins(ui::UiPlugin)
        // Ticket 050's build menu/city panel need to have drawn this frame
        // before `drive_camera` and the picking/placement/commit/demolish
        // systems that already gate on `camera::EguiInputCapture` read it —
        // the same two lines `viewer::run` uses `ui::UiPanelSet` for, see
        // that module's own docs for the fuller argument.
        .configure_sets(Update, camera::CameraSet.after(ui::UiPanelSet))
        .add_systems(Last, (save_city_on_exit, save_journal_on_exit))
        .run();
}

/// Loads [`state::City`] from `save_root`, logging what happened the same
/// way [`load_building_catalogue`]/[`load_building_definitions`] do. Skips
/// persistence entirely (starts from an empty city) when `save_root` is
/// empty — ticket 008's `empty_save` placeholder, with nowhere to read from.
fn load_city(save_root: &Path) -> state::City {
    if save_root.as_os_str().is_empty() {
        println!("block_viewer: no save loaded, starting with an empty city");
        return state::City::default();
    }

    match persistence::load_city(save_root) {
        Ok(city) => {
            println!(
                "block_viewer: loaded {} building{} from {}",
                city.len(),
                if city.len() == 1 { "" } else { "s" },
                persistence::city_file_path_for_log(save_root).display(),
            );
            city
        }
        Err(err) => {
            println!("block_viewer: could not load city save, starting empty: {err}");
            state::City::default()
        }
    }
}

/// Saves [`state::City`] to [`CitySavePath`] on every [`AppExit`] — window
/// close, Alt+F4, or any other route Bevy turns into that event. A no-op
/// when [`CitySavePath`] is `None` (no real save was loaded, see
/// [`load_city`]).
fn save_city_on_exit(mut exit_events: EventReader<AppExit>, city: Res<state::City>, save_path: Res<CitySavePath>) {
    if exit_events.read().count() == 0 {
        return;
    }
    let Some(save_root) = &save_path.0 else { return };

    match persistence::save_city(&city, save_root) {
        Ok(()) => println!("block_viewer: saved city ({} building{})", city.len(), if city.len() == 1 { "" } else { "s" }),
        Err(err) => println!("block_viewer: could not save city: {err}"),
    }
}

/// Loads [`journal::Journal`] from `save_root`, same shape as [`load_city`] —
/// a missing file or no save at all both start from an empty journal, logged
/// only when there's something to say (a non-empty journal loaded, or a real
/// load error).
fn load_journal(save_root: &Path) -> journal::Journal {
    if save_root.as_os_str().is_empty() {
        return journal::Journal::default();
    }

    match journal::load_journal(save_root) {
        Ok(loaded) => {
            if !loaded.is_empty() {
                println!(
                    "block_viewer: loaded {} journal entr{} from {}",
                    loaded.len(),
                    if loaded.len() == 1 { "y" } else { "ies" },
                    journal::journal_file_path_for_log(save_root).display(),
                );
            }
            loaded
        }
        Err(err) => {
            println!("block_viewer: could not load journal, starting empty: {err}");
            journal::Journal::default()
        }
    }
}

/// Saves [`journal::Journal`] to [`CitySavePath`] on every [`AppExit`], the
/// same trigger and the same no-op-when-`None` contract [`save_city_on_exit`]
/// uses — see that function's docs.
fn save_journal_on_exit(
    mut exit_events: EventReader<AppExit>,
    journal: Res<journal::Journal>,
    save_path: Res<CitySavePath>,
) {
    if exit_events.read().count() == 0 {
        return;
    }
    let Some(save_root) = &save_path.0 else { return };

    if let Err(err) = journal::save_journal(&journal, save_root) {
        println!("block_viewer: could not save journal: {err}");
    }
}

/// Loads and logs the building catalogue. Split out from [`run`] so the
/// logging has somewhere to live that isn't crammed into the `run` body —
/// [`blueprint::load_catalogue_dir`] itself never panics (see its docs), so
/// there's no error path here to propagate, only one to print.
fn load_building_catalogue() -> blueprint::BuildingCatalogue {
    let (catalogue, skipped) = blueprint::load_catalogue_dir(Path::new(CATALOGUE_DIR));

    println!(
        "block_viewer: loaded {} building{} from {CATALOGUE_DIR}",
        catalogue.len(),
        if catalogue.len() == 1 { "" } else { "s" }
    );
    for entry in catalogue.iter() {
        println!(
            "block_viewer:   {} — {}x{}x{} ({} states)",
            entry.id,
            entry.blueprint.size.x,
            entry.blueprint.size.y,
            entry.blueprint.size.z,
            entry.blueprint.palette.len(),
        );
    }
    for (path, err) in &skipped {
        println!("block_viewer:   skipped {}: {err}", path.display());
    }

    catalogue
}

/// Loads and logs the building definitions, same shape as
/// [`load_building_catalogue`] — [`definition::load_definitions_dir`] never
/// panics either, so there's no error path to propagate, only one to print.
fn load_building_definitions(catalogue: &blueprint::BuildingCatalogue) -> definition::BuildingDefinitions {
    let (definitions, skipped) = definition::load_definitions_dir(Path::new(DEFINITIONS_DIR), catalogue);

    println!(
        "block_viewer: loaded {} building definition{} from {DEFINITIONS_DIR}",
        definitions.len(),
        if definitions.len() == 1 { "" } else { "s" }
    );
    for entry in definitions.iter() {
        println!(
            "block_viewer:   {} — {:?} tier {}, footprint {}x{}",
            entry.id, entry.building.name, entry.building.tier, entry.footprint.x, entry.footprint.y,
        );
    }
    for (path, err) in &skipped {
        println!("block_viewer:   skipped {}: {err}", path.display());
    }

    definitions
}
