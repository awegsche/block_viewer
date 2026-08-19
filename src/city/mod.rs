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
//! building: a left click claims the tile in [`state::City`], applies the
//! rotated blueprint to the shared region cache (W4/W5) on
//! `AsyncComputeTaskPool`, and — once that apply actually succeeds — records
//! the as-built baseline in [`journal::Journal`] (roadmap I1, landing here
//! per the roadmap's own instruction to ship it with E4) and fires
//! `ChunksEdited` so the building appears without a restart (W7). A failed
//! apply rolls the city entry back rather than leaving a phantom building
//! behind — see [`commit`]'s module docs for the whole transaction, and its
//! "Applied to memory, not written to disk" note (ticket 051): this is the
//! first thing in the game that changes the save's blocks at all, but
//! nothing reaches disk until [`save::SavePlugin`] does.
//!
//! ## Demolish (ticket 049, roadmap E5)
//!
//! [`demolish::DemolishPlugin`] is commit's inverse: `Delete` on a hovered,
//! placed building removes it from [`state::City`] and applies the terrain
//! its own placement baseline (roadmap I1) says stood there back to the
//! region cache — no re-derivation from the blueprint, the baseline already
//! *is* the answer. It also lands [`journal::Journal::record_demolition`]'s
//! first real caller, the `Demolished` half of D3's journal (ticket 044)
//! that had existed, unused, since then. Unlike commit, the `City` removal
//! happens *after* the apply succeeds rather than before — see
//! [`demolish`]'s module docs for why that ordering, not commit's, is the
//! one that's safe here.
//!
//! ## Save world (ticket 051)
//!
//! Commit, demolish and undo all queue their edits in the shared region
//! cache and stop — see `city::commit`'s module docs' "Applied to memory,
//! not written to disk" for why placing a street of houses no longer
//! rewrites a whole region file per house. [`save::SavePlugin`] is the one
//! place any of that actually reaches disk: a "Save world" click
//! (`city::ui::city_panel`) opens a real
//! [`crate::edit::session::WriteSession`] and flushes every dirty region,
//! backing each one up once per session. [`flush_world_on_exit`] calls the
//! same flush synchronously on [`AppExit`], before `city.ron`/`journal.ron`
//! are written — without it, quitting with something unsaved would leave
//! those two files describing buildings the world never actually got.
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
//!
//! ## The road graph (ticket 053, roadmap F1) — and its cell-space revision
//! (ticket 054)
//!
//! [`road`] is the first piece of group F: adjacency and reachability
//! queries over [`state::City`]'s road cells (`add_road_cell`/
//! `road_cells()`, landed with D1 but otherwise unread until now). No new
//! stored state — a road graph is derived from `City::is_road_cell` on
//! every call, so it can never itself go stale. Not wired into `run` as a
//! plugin; it's a pure query module. [`road_build::RoadBuildPlugin`] (ticket
//! 055, below) is F2/F3's real caller; F4's connectivity queries (ticket
//! 056) are the module's other half — see [`road`]'s own docs — and are
//! still the "proven, not yet used" state several earlier tickets (042, 039,
//! 040) landed their own resources in: no UI or logistics reads them yet.
//!
//! Ticket 054 redefined a road from a single-block tile to a
//! [`state::ROAD_CELL_SIZE`]-block cell (a real cross-section, not a 1x1
//! dot) and moved `road`'s whole coordinate space to match — a breaking
//! change to 053's shipped API, not an addition next to it, since a road
//! now has a *shape* that a block-tile model had no way to represent. It
//! also lands `road::select_piece`, the pure classification F3's
//! auto-tiling needs (which of six piece shapes a cell's connections call
//! for, and the rotation that reproduces them), and [`road_catalogue`], a
//! loader for the six `.nbt` blueprints — one per shape — that
//! `select_piece` picks between.
//!
//! Ticket 059 gave that catalogue a second dimension: **style**.
//! `assets/city/roads/<style>/*.nbt` — one subdirectory per style, holding
//! its own six pieces — instead of one flat, system-wide set. A road cell
//! now records which style it was built as directly on [`state::City`]
//! (`add_road_cell`'s own argument, `City::road_style_at`), and
//! [`road_build::RoadStyleSelection`] (`[`/`]`, gated on the road tool) is
//! how a player picks which style *new* cells get built as — the same
//! keyboard-stand-in role ticket 047's number keys play for buildings.
//! Connectivity and shape selection (`road::connections_at`/`select_piece`)
//! stay entirely style-blind — style only decides which `.nbt` gets
//! meshed/written once the shape is already chosen. No real `.nbt` pieces
//! ship for any style yet (they need an actual Minecraft structure-block
//! export); `road_catalogue` is proven against synthetic fixtures the same
//! way ticket 039's building catalogue tests were, and isn't wired into
//! `run` until there's something on disk for it to load.
//!
//! ## Terraforming: dig and level (ticket 057, roadmap H1)
//!
//! [`terraform::TerraformPlugin`] is a third
//! [`tool::ActiveTool`] (`T` now cycles Building -> Road -> Terraform ->
//! Building), gated the same way [`road_build::RoadBuildPlugin`] gates its
//! own drag on [`tool::ActiveTool::Road`]. A left-click drag over a
//! rectangle of tiles either digs (clears the topmost block from every
//! tile, `Z`'s default mode) or levels (flattens the rectangle to the
//! height the drag started on, digging the high tiles and filling the low
//! ones with dirt) — see [`terraform`]'s own docs for why neither reuses
//! [`crate::edit::WorldEdit::fill`] and why there's no city-state entry or
//! journal record for either. Originally justified as the fix for E2's
//! `fit_footprint` refusing uneven ground outright; ticket 058 removed that
//! refusal (a real Minecraft world is inherently uneven, and the game
//! shouldn't block a placement over it), so terraforming is no longer the
//! only way to build on a rough site — it's now how a player *chooses* to
//! flatten one anyway, by hand, rather than build into the slope.

use std::path::{Path, PathBuf};

mod commit;
mod definition;
mod demolish;
mod grid;
mod journal;
mod persistence;
mod picking;
mod placement;
mod road;
mod road_build;
mod road_catalogue;
mod road_definition;
mod save;
mod state;
mod terraform;
mod tool;
mod ui;
mod undo;
mod write_status;

use bevy::app::AppExit;
use bevy::prelude::*;

use crate::{
    blueprint,
    camera::{self, CameraMode, CameraStartMode},
    chunk_pipeline::{RenderFloor, SharedRegionCache},
    edit::session::WriteSession,
    world::decode::FloorPolicy,
    world_app, LoadedSave,
};

/// Where [`run`] looks for building blueprints — the flat, non-recursive
/// directory the roadmap's `assets/city/blueprints/*.nbt` glob names.
const CATALOGUE_DIR: &str = "assets/city/blueprints";

/// Where [`run`] looks for building definitions — the flat, non-recursive
/// directory the roadmap's C1 sketch implies alongside `CATALOGUE_DIR`.
const DEFINITIONS_DIR: &str = "assets/city/buildings";

/// Where [`run`] looks for road pieces (ticket 054/055, roadmap F1/F3;
/// styled subdirectories added by 059) — the directory
/// [`road_catalogue::load_road_catalogue_dir`] scans for style folders.
const ROAD_CATALOGUE_DIR: &str = "assets/city/roads";

/// Where [`run`] looks for road type definitions (ticket 060, roadmap F1b/F3
/// follow-up) — the flat, non-recursive directory
/// [`road_definition::load_road_types_dir`] expects, alongside
/// `ROAD_CATALOGUE_DIR` the same way `DEFINITIONS_DIR` sits alongside
/// `CATALOGUE_DIR`.
const ROAD_TYPES_DIR: &str = "assets/city/road_types";

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
    let road_catalogue = load_road_catalogue();
    let road_types = load_road_types(&road_catalogue);

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
        .insert_resource(road_catalogue)
        .insert_resource(road_types)
        .insert_resource(city)
        .insert_resource(journal)
        .insert_resource(CitySavePath(if save_root.as_os_str().is_empty() { None } else { Some(save_root) }))
        .add_plugins(tool::ToolPlugin)
        .add_plugins(picking::PickingPlugin)
        .add_plugins(placement::PlacementPlugin)
        .add_plugins(commit::CommitPlugin)
        .add_plugins(demolish::DemolishPlugin)
        .add_plugins(road_build::RoadBuildPlugin)
        .add_plugins(terraform::TerraformPlugin)
        .add_plugins(undo::UndoPlugin)
        .add_plugins(save::SavePlugin)
        .add_plugins(ui::UiPlugin)
        // Ticket 050's build menu/city panel need to have drawn this frame
        // before `drive_camera` and the picking/placement/commit/demolish
        // systems that already gate on `camera::EguiInputCapture` read it —
        // the same two lines `viewer::run` uses `ui::UiPanelSet` for, see
        // that module's own docs for the fuller argument.
        .configure_sets(Update, camera::CameraSet.after(ui::UiPanelSet))
        // `flush_world_on_exit` first: `city.ron`/`journal.ron` describe
        // buildings whose blocks need to have actually reached disk by the
        // time they're written — see the module docs' "Save world".
        .add_systems(Last, (flush_world_on_exit, save_city_on_exit, save_journal_on_exit).chain())
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

/// Flushes every dirty region the shared region cache is holding to disk on
/// every [`AppExit`] — see the module docs' "Save world" for why this can't
/// wait for the player to remember to click "Save world" themselves:
/// `city.ron`/`journal.ron` (written right after this, by
/// [`save_city_on_exit`]/[`save_journal_on_exit`]) describe buildings whose
/// blocks need to have actually reached disk, or the three files disagree
/// the moment the app reopens. Synchronous, not dispatched onto
/// `AsyncComputeTaskPool` like [`save::start_save`] — the app is already
/// exiting, so blocking here costs nothing a player would notice, and there
/// is no later frame for an async task to be polled on. A no-op when either
/// resource is missing (no real save loaded) or nothing is dirty.
fn flush_world_on_exit(
    mut exit_events: EventReader<AppExit>,
    loaded_save: Option<Res<LoadedSave>>,
    region_cache: Option<Res<SharedRegionCache>>,
) {
    if exit_events.read().count() == 0 {
        return;
    }
    let (Some(loaded_save), Some(region_cache)) = (loaded_save, region_cache) else { return };

    let mut cache = region_cache.0.lock().expect("region cache mutex poisoned");
    if cache.dirty_regions().count() == 0 {
        return;
    }

    match WriteSession::open(&loaded_save.0.meta) {
        Ok(mut session) => match session.flush(&mut *cache) {
            Ok(summary) => println!(
                "block_viewer: flushed {} unsaved region file(s) on exit ({} new backup(s))",
                summary.regions_written.len(),
                summary.backups.len()
            ),
            Err(err) => println!("block_viewer: could not flush unsaved world edits on exit: {err}"),
        },
        Err(err) => println!("block_viewer: could not flush unsaved world edits on exit: {err}"),
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

/// Loads and logs the road piece catalogue (ticket 054/055, roadmap F1/F3),
/// same shape as [`load_building_catalogue`] —
/// [`road_catalogue::load_road_catalogue_dir`] never panics either, so
/// there's no error path to propagate, only one to print. Missing pieces are
/// expected today (see that module's own "No real assets yet") — `run`
/// always inserts whatever loaded, even an entirely empty catalogue, so
/// `road_build` can read it unconditionally rather than through a second
/// `Option`.
fn load_road_catalogue() -> road_catalogue::RoadCatalogue {
    let (catalogue, skipped) = road_catalogue::load_road_catalogue_dir(Path::new(ROAD_CATALOGUE_DIR));

    println!(
        "block_viewer: loaded {} road piece{} from {ROAD_CATALOGUE_DIR}",
        catalogue.len(),
        if catalogue.len() == 1 { "" } else { "s" }
    );
    for (style, kind, err) in &skipped {
        println!("block_viewer:   skipped {style}/{kind:?}: {err}");
    }

    catalogue
}

/// Loads and logs road type definitions (ticket 060, roadmap F1b/F3
/// follow-up), same shape as [`load_building_definitions`] —
/// [`road_definition::load_road_types_dir`] never panics either, so there's
/// no error path to propagate, only one to print. Validated against
/// `catalogue` the same way [`load_building_definitions`] validates against
/// the blueprint catalogue.
fn load_road_types(catalogue: &road_catalogue::RoadCatalogue) -> road_definition::RoadTypes {
    let (types, skipped) = road_definition::load_road_types_dir(Path::new(ROAD_TYPES_DIR), catalogue);

    println!(
        "block_viewer: loaded {} road type{} from {ROAD_TYPES_DIR}",
        types.len(),
        if types.len() == 1 { "" } else { "s" }
    );
    for entry in types.iter() {
        println!(
            "block_viewer:   {} — {:?} (speed {}, capacity {})",
            entry.id, entry.road_type.name, entry.road_type.travel_speed, entry.road_type.capacity,
        );
    }
    for (path, err) in &skipped {
        println!("block_viewer:   skipped {}: {err}", path.display());
    }

    types
}
