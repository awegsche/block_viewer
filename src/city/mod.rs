//! The `citybuilder` game (ticket 027) — milestone M1: a window on a real
//! save with the existing streaming and camera, and nothing else yet.
//!
//! The plan it grows into is `tickets/CITYBUILDER_ROADMAP.md`. The rule the
//! rest of it follows, worth repeating where the code will live: **the city
//! state is authoritative, and the blocks in the world are a projection of
//! it**. Nothing here owns any city state yet — that's roadmap task D1, and
//! it waits on the write path (group W) proving that blocks can land in a
//! world Minecraft opens without complaint.
//!
//! What this deliberately does *not* add, and why it looks so empty:
//!
//! - **No UI.** The viewer's egui panels ([`crate::viewer::ui`]) are its
//!   own; the citybuilder's build menu and city panel are roadmap group G.
//!   Without `EguiPlugin` the camera's [`crate::camera::EguiInputCapture`]
//!   simply stays at its "nothing captured" default, which is correct here.
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

use std::path::Path;

use crate::{blueprint, chunk_pipeline::RenderFloor, world::decode::FloorPolicy, world_app};

/// Where [`run`] looks for building blueprints — the flat, non-recursive
/// directory the roadmap's `assets/city/blueprints/*.nbt` glob names.
const CATALOGUE_DIR: &str = "assets/city/blueprints";

/// Runs the citybuilder. Called by `src/bin/citybuilder.rs`, which is three
/// lines and nothing else.
pub fn run() {
    let catalogue = load_building_catalogue();

    world_app()
        .insert_resource(RenderFloor(FloorPolicy::BelowSurface { margin: 16 }))
        .insert_resource(catalogue)
        .run();
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
