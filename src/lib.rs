//! Shared core for the two games in this package (ticket 027): the
//! `block_viewer` explorer and the `citybuilder` built on top of it. Both
//! live *inside* this lib rather than in separate crates, so `pub(crate)`
//! keeps working across every shared module and no visibility churn was
//! needed to grow the second entry point. The workspace split
//! (`mc_core` / `block_viewer` / `citybuilder`) stays available as a later
//! move, once the shared core stops changing shape.
//!
//! ## The module tree
//!
//! Everything above [`viewer`] and [`city`] is shared: [`world`] (decode,
//! mesh, atlas, tint, biome, block), [`blueprint`], [`selection`]'s
//! coordinate rules, [`region_cache`], [`streaming`], [`chunk_pipeline`],
//! [`unload`], [`sky`] and [`camera`]. The two games are the leaves.
//!
//! [`selection`] is shared rather than viewer-only — the roadmap's sketch
//! put it under `viewer`, but `blueprint::extract` already takes a
//! [`selection::SelectionBounds`], and a shared module can't depend on a
//! viewer-only one. It's also where ticket 019 fixed the coordinate rules
//! (`bevy.z = -mc.z`, inclusive bounds) that the write path is meant to
//! inherit rather than reinvent. Its *interaction* half (`gizmo`, `input`)
//! is viewer-flavoured; splitting the module along that line is a later
//! refactor if it earns itself.
//!
//! ## The shared startup
//!
//! [`world_app`] is the half of the old `main()` both games need — save
//! loading, the streaming/meshing plugins, and the `Startup` system that
//! builds the atlas, the region cache, the camera and the sky. Each game's
//! `run()` adds its own layer on top of it: see [`viewer::run`] and
//! [`city::run`].

// This is an application, not a published library: the module tree is `pub`
// only so the two binary shims can reach `viewer::run`/`city::run`, and the
// docs are written for `cargo doc --document-private-items`. Doc comments
// linking to a private system or resource are the norm here rather than a
// leak, so ticket 027 silences the lint that started firing the moment the
// tree stopped being a binary's private modules.
#![allow(rustdoc::private_intra_doc_links)]

use bevy::prelude::*;
use mc_anvil::{get_saves_from_instance, region::REGION_WIDTH_IN_CHUNKS, Save, SaveMeta};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub mod blueprint;
pub mod camera;
pub mod chunk_pipeline;
pub mod city;
pub mod edit;
pub mod region_cache;
pub mod selection;
pub mod sky;
pub mod streaming;
pub mod unload;
pub mod viewer;
pub mod world;

/// The currently loaded Minecraft save, populated at startup from a real
/// save directory (`%AppData%\.minecraft\saves` on Windows). `pub(crate)`
/// (field included) so the UI's save picker (ticket 007) can swap it out at
/// runtime without restarting.
///
/// Always holds a real (if possibly empty-of-regions) [`Save`] — never
/// `Option`/`Result` — so every reader ([`setup_world`], the save picker) can
/// read `.0.meta` unconditionally. When nothing could actually be loaded at
/// startup (ticket 008: no `.minecraft` directory, an empty `saves/`
/// folder, ...) this holds [`empty_save`] instead of panicking, and
/// [`StartupIssue`] carries the reason for the UI to show.
#[derive(Resource)]
pub(crate) struct LoadedSave(pub(crate) Save);

/// The reason [`LoadedSave`] is empty at startup, if any (ticket 008) —
/// `None` once a real save has loaded (startup found one, or the save
/// picker loaded one at runtime). Read by the save picker panel to show the
/// problem instead of the app just silently sitting on an empty world.
#[derive(Resource, Default)]
pub(crate) struct StartupIssue(pub(crate) Option<String>);

/// A [`Save`] with no regions and no on-disk backing — what [`LoadedSave`]
/// holds when startup couldn't find a real one (ticket 008). Safe to treat
/// like any other loaded save: an empty `regions` list means the streaming
/// pipeline simply has nothing to load, the same as a real save the camera
/// hasn't flown into any generated terrain of yet.
fn empty_save() -> Save {
    Save {
        meta: SaveMeta {
            name: "(no save loaded)".to_string(),
            path: PathBuf::new(),
            region_dir: PathBuf::new(),
            regions: Vec::new(),
        },
        regions: Vec::new(),
    }
}

/// Every chunk column decoded so far (ticket 002), keyed by world chunk
/// coordinates, plus the [`world::BlockRegistry`] their block names were
/// interned into. Starts empty at startup (ticket 005-e deleted the old
/// eager pre-`App::run()` load) and fills in as [`chunk_pipeline`]'s
/// background chunk-load tasks (ticket 005-c) stream columns in around the
/// camera; also read by [`camera`]'s under-the-cursor ray-march for orbit
/// targeting (ticket 006).
///
/// `registry` is behind an `Arc<Mutex<_>>` — not just an owned
/// [`world::BlockRegistry`] — because those background tasks intern newly
/// decoded chunks' block names directly into it, off the main thread, and
/// [`world::BlockId`]s need to stay globally stable regardless of whether a
/// chunk was decoded eagerly at startup or streamed in later.
///
/// `biomes` (ticket 012) is the same story, one registry over: background
/// tasks intern each section's biome names into it during decode, and
/// [`world::BiomeId`]s need the same global stability as [`world::BlockId`]s.
#[derive(Resource)]
pub(crate) struct DecodedWorld {
    pub(crate) registry: Arc<Mutex<world::BlockRegistry>>,
    pub(crate) biomes: Arc<Mutex<world::BiomeRegistry>>,
    pub(crate) columns: HashMap<(i32, i32), world::ChunkColumn>,
}

/// Marker on every spawned chunk mesh entity — `pub(crate)` so
/// [`chunk_pipeline`]'s polling system can tag entities it spawns, and the
/// save picker's world reset can find and despawn them.
#[derive(Component)]
pub(crate) struct BlockMesh;

/// The two positional CLI arguments, with blank ones dropped — see
/// [`save_args_from`]. What each one *means* is deliberately not decided
/// here: [`resolve_selection_in`] does that, because it depends on what's
/// actually on disk.
struct SaveArgs {
    first: Option<String>,
    second: Option<String>,
}

/// [`SaveArgs`] from the real process args; [`save_args_from`] is the
/// testable half — real process args can't be overridden per-test.
fn save_args() -> SaveArgs {
    let mut args = std::env::args().skip(1);
    save_args_from(args.next(), args.next())
}

/// The actual logic behind [`save_args`]: an argument that's empty, all
/// whitespace, or nothing but quote characters counts as **not given**.
///
/// The quote case is Windows PowerShell 5.1, which doesn't pass an empty
/// `""` argument through to a native executable intact — it arrives either
/// dropped entirely or as two literal quote characters. Both are why
/// [`resolve_selection_in`] below makes a *single* argument work on its own
/// rather than needing an empty one in front of a save name.
fn save_args_from(first: Option<String>, second: Option<String>) -> SaveArgs {
    fn given(arg: String) -> Option<String> {
        let trimmed = arg.trim().trim_matches(QUOTES).trim().to_string();
        (!trimmed.is_empty()).then_some(trimmed)
    }
    SaveArgs { first: first.and_then(given), second: second.and_then(given) }
}

/// The quote characters [`save_args_from`] strips off an argument. A real
/// path never starts or ends with one, so this can't eat a meaningful part
/// of a directory name.
const QUOTES: [char; 2] = ['"', '\''];

/// What the CLI arguments resolved to (ticket 064): the directory to scan,
/// and which save in it to open (`None` = the first one found, which is all
/// this ever did before).
struct SaveSelection {
    directory: PathBuf,
    wanted: Option<String>,
}

/// `dirs::config_dir()/.minecraft/saves`
/// (`C:\Users\<user>\AppData\Roaming\.minecraft\saves` on Windows) — where
/// saves live when no CLI argument says otherwise.
fn default_saves_directory() -> PathBuf {
    dirs::config_dir().unwrap_or_default().join(".minecraft/saves")
}

/// Decides what each argument meant, against what's on disk.
///
/// Two arguments are unambiguous: directory, then save. **One** is the
/// interesting case, and it resolves by what the argument actually is:
///
/// - a directory that lists at least one save is ticket 008's saves/instance
///   directory (`citybuilder D:/curseforge/instance/saves`), unchanged;
/// - anything else is ticket 064's save — a name under
///   [`default_saves_directory`], or a path straight to one save
///   (`citybuilder nbt_test`, `citybuilder D:/worlds/nbt_test`).
///
/// Naming the save is by far the commoner thing to want, and this is what
/// lets it be typed on its own. It originally required an empty first
/// argument (`-- "" nbt_test`) to keep `argv[1]`'s ticket 008 meaning
/// unambiguous — which turned out not to survive Windows PowerShell 5.1's
/// native-argument handling at all (see [`save_args_from`]), and was awkward
/// to type even where it did. The empty-first-argument form still works.
///
/// Takes the default directory as a parameter (rather than calling
/// [`default_saves_directory`] itself) purely so tests can point it at a
/// fixture directory.
fn resolve_selection_in(args: &SaveArgs, default_dir: &Path) -> SaveSelection {
    match (args.first.as_deref(), args.second.as_deref()) {
        (None, None) => SaveSelection { directory: default_dir.to_path_buf(), wanted: None },
        (Some(dir), Some(name)) => {
            SaveSelection { directory: PathBuf::from(dir), wanted: Some(name.to_string()) }
        }
        (None, Some(name)) => {
            SaveSelection { directory: default_dir.to_path_buf(), wanted: Some(name.to_string()) }
        }
        (Some(only), None) if lists_saves(Path::new(only)) => {
            SaveSelection { directory: PathBuf::from(only), wanted: None }
        }
        (Some(only), None) => {
            SaveSelection { directory: default_dir.to_path_buf(), wanted: Some(only.to_string()) }
        }
    }
}

/// Whether `dir` reads as a saves *directory* — i.e. it lists at least one
/// save. Deliberately "at least one" rather than "is readable": a path
/// straight to a single save is readable too, but its `data`/`datapacks`/
/// `dimensions` subdirectories are not themselves saves, so it lists none
/// and resolves as a save instead. See [`resolve_selection_in`].
fn lists_saves(dir: &Path) -> bool {
    get_saves_from_instance(dir).is_ok_and(|saves| !saves.is_empty())
}

/// Directory to scan for Minecraft saves — [`resolve_selection_in`]'s half of
/// the answer. Shared by startup ([`try_load_real_save`]) and the UI's save
/// picker (`viewer::ui::scan_saves`) so both agree on where "the saves
/// directory" is, including when a single CLI argument named a save rather
/// than a directory (the picker then lists the default directory, which is
/// where that save was found).
pub(crate) fn saves_directory() -> PathBuf {
    resolve_selection_in(&save_args(), &default_saves_directory()).directory
}

/// Picks a save under `dir` — the one `wanted` names (ticket 064), or the
/// first one found when it's `None`, which is all this did before. Metadata
/// only (`get_saves_from_instance`/`SaveMeta` -> `Save`) — cheap and
/// synchronous, unlike chunk data, which streams in after `App::run()` via
/// the async pipeline (ticket 005-c) instead of being loaded here (ticket
/// 005-e removed the old eager pre-`App::run()` region load).
///
/// A `wanted` that names an **existing directory** is taken as the save
/// itself and read straight through `SaveMeta::from_path`, so a world living
/// nowhere near any `saves/` folder can be opened.
///
/// `Err` covers every unhappy path ticket 008 calls out — no `.minecraft`
/// directory, an empty `saves/` folder, or any other I/O failure listing
/// it — plus ticket 064's "no save by that name", as a message for
/// [`load_real_save`] to log and show in the UI, rather than a panic that
/// kills the process before the window opens.
/// Takes `dir` as a parameter (rather than calling [`saves_directory`]
/// itself) purely so tests can point it at a fixture directory.
fn try_load_save_from(dir: &Path, wanted: Option<&str>) -> Result<Save, String> {
    if let Some(path) = wanted.map(Path::new).filter(|path| path.is_dir()) {
        let meta = SaveMeta::from_path(path).map_err(|e| {
            format!("{} is not a Minecraft save (no readable region directory): {e}", path.display())
        })?;
        println!("Loading save {}", meta.get_grid_view());
        return Ok(meta.into());
    }

    let saves = get_saves_from_instance(dir)
        .map_err(|e| format!("could not read saves directory {}: {e}", dir.display()))?;
    let meta = match wanted {
        Some(name) => pick_named_save(&saves, name, dir)?,
        None => saves
            .into_iter()
            .next()
            .ok_or_else(|| format!("no Minecraft saves found under {}", dir.display()))?,
    };

    println!("Loading save {}", meta.get_grid_view());

    Ok(meta.into())
}

/// The save called `wanted` out of `saves` — exact name first, then
/// case-insensitively, since a world's folder name is what the player typed
/// into Minecraft and re-typing its capitalisation on a command line is a
/// pointless way to fail (ticket 064).
///
/// The `Err` lists the names that *are* there: a typo'd name would otherwise
/// fall back to [`empty_save`] and look exactly like a world that hasn't
/// generated any terrain yet.
fn pick_named_save(saves: &[SaveMeta], wanted: &str, dir: &Path) -> Result<SaveMeta, String> {
    if let Some(meta) = saves
        .iter()
        .find(|meta| meta.name == wanted)
        .or_else(|| saves.iter().find(|meta| meta.name.eq_ignore_ascii_case(wanted)))
    {
        return Ok(meta.clone());
    }

    let available = if saves.is_empty() {
        "(none)".to_string()
    } else {
        saves.iter().map(|meta| meta.name.as_str()).collect::<Vec<_>>().join(", ")
    };
    Err(format!(
        "no save named \"{wanted}\" under {} — available: {available}",
        dir.display()
    ))
}

/// [`try_load_save_from`] against [`resolve_selection_in`] — the real entry
/// point [`load_real_save`] uses at startup.
fn try_load_real_save() -> Result<Save, String> {
    let selection = resolve_selection_in(&save_args(), &default_saves_directory());
    try_load_save_from(&selection.directory, selection.wanted.as_deref())
}

/// Always returns a usable [`Save`] — [`empty_save`] plus a logged reason
/// when [`try_load_real_save`] couldn't find a real one (ticket 008), rather
/// than the panics that used to kill the process before the window ever
/// opened.
fn load_real_save() -> (Save, Option<String>) {
    match try_load_real_save() {
        Ok(save) => (save, None),
        Err(reason) => {
            println!("block_viewer: {reason}");
            (empty_save(), Some(reason))
        }
    }
}

/// The `App` both games start from (ticket 027): a real save loaded, the
/// streaming/meshing pipeline wired up, and a camera under a sky. Running
/// this as-is gives the citybuilder's M1 window; the viewer adds its
/// selection, blueprint and UI layers on top before calling `run()`.
///
/// Deliberately *not* a `Plugin`: it loads the save synchronously before the
/// `App` exists, because [`LoadedSave`] must be inserted as a resource
/// rather than discovered, and because a save that can't be found is a
/// message for the UI (ticket 008) rather than a startup failure.
pub fn world_app() -> App {
    let (save, startup_issue) = load_real_save();
    let decoded_world = DecodedWorld {
        registry: Arc::new(Mutex::new(world::BlockRegistry::new())),
        biomes: Arc::new(Mutex::new(world::BiomeRegistry::new())),
        columns: HashMap::new(),
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin)
        .add_plugins(camera::CameraControllerPlugin)
        .add_plugins(sky::SkyPlugin)
        .add_plugins(streaming::ChunkStreamingPlugin)
        .add_plugins(chunk_pipeline::ChunkLoadPipelinePlugin)
        .add_plugins(unload::ChunkUnloadPlugin)
        .insert_resource(LoadedSave(save))
        .insert_resource(StartupIssue(startup_issue))
        .insert_resource(decoded_world)
        .add_systems(Startup, setup_world);
    app
}

/// The `Startup` half of [`world_app`]: the texture atlas and biome
/// colormaps, the region cache and terrain material the streaming pipeline
/// runs on, and the camera, sun and sky scene.
fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    loaded_save: Res<LoadedSave>,
    render_distance: Res<streaming::RenderDistance>,
    sky_palette: Res<sky::SkyPalette>,
    camera_start_mode: Res<camera::CameraStartMode>,
) {
    println!(
        "Active save: {} ({} regions)",
        loaded_save.0.meta.name,
        loaded_save.0.meta.regions.len()
    );

    // Pack every block texture into one atlas (ticket 004) — replaces the
    // single hardcoded stone.png the mesher previously stretched over every
    // face. Per-block-id UV resolution (`build_block_uv_table`) happens
    // per-chunk inside `chunk_pipeline`'s background tasks instead of once
    // here, against whatever names are interned into the registry *at the
    // moment that chunk decodes* — block names get interned over the app's
    // whole lifetime as streaming loads new chunks (ticket 005-e), not just
    // once at startup like the old eager decode, so there's no fixed set of
    // names to build a table from up front.
    let atlas = world::atlas::build(Path::new("assets/minecraft/textures/block"))
        .expect("failed to build the block texture atlas");
    let uv_index = atlas.uv_index();
    let atlas_handle = images.add(atlas.image);
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(atlas_handle),
        // Ticket 014: discard transparent fragments instead of the default
        // `AlphaMode::Opaque`, which ignored the alpha channel entirely and
        // rendered leaves as solid bricks. `Mask(0.5)` is a hard cutout
        // (no partial blending, no back-to-front sort needed) — the right
        // trade for leaves/the grass overlay; real translucency (water,
        // glass) is a second material, deferred to 010.
        alpha_mode: AlphaMode::Mask(0.5),
        ..default()
    });

    // Biome-tint colormaps (ticket 013) — same "plain data, resolved per
    // chunk-load task" story as the atlas's `AtlasUvIndex` above: no fixed
    // set of biomes to resolve up front, so `SharedColorMaps` just hands the
    // raw texels to `chunk_pipeline`'s background tasks.
    let color_maps = world::tint::load_color_maps(Path::new("assets/minecraft/textures/colormap"))
        .expect("failed to load the biome colormaps");

    // Nothing is decoded yet — chunks stream in via the async pipeline
    // (ticket 005-c) as the camera moves, driven by these four resources
    // plus `DecodedWorld` (already inserted in `world_app()`). Ticket 005-e
    // deleted the old eager single-region load and per-column spawn loop
    // that used to populate the world here.
    let region_cache = region_cache::RegionCache::new(
        loaded_save.0.meta.clone(),
        region_cache::recommended_capacity(render_distance.0),
    );
    commands.insert_resource(chunk_pipeline::SharedRegionCache(Arc::new(Mutex::new(
        region_cache,
    ))));
    commands.insert_resource(chunk_pipeline::SharedAtlasIndex(Arc::new(uv_index)));
    commands.insert_resource(chunk_pipeline::SharedColorMaps(Arc::new(color_maps)));
    commands.insert_resource(chunk_pipeline::TerrainMaterial(material_handle));

    // Place the camera near the middle of the save's region footprint
    // (ticket 005-e) instead of reading real terrain height like the old
    // eager-decode version did — nothing is decoded yet at startup to read
    // a height from. The camera free-flies at a fixed, generally-safe
    // height while terrain streams in underneath, per the ticket's
    // preference over deferring camera spawn until the first chunk loads.
    let target = spawn_point(&loaded_save.0.meta);
    let eye = target + Vec3::new(-24.0, 20.0, 24.0);

    commands.spawn((
        Name::new("Camera"),
        Camera3d::default(),
        // Ticket 016: the sky camera (spawned below, `order: -1`) draws the
        // dome/sun/moon first; this camera must not clear that away, so it
        // draws terrain on top of the sky pass instead of erasing it.
        Camera {
            clear_color: ClearColorConfig::None,
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            far: camera::far_plane_distance(render_distance.0),
            ..default()
        }),
        camera::atmosphere_fog(render_distance.0, sky_palette.horizon_color),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        camera::CameraRig::looking_at(eye, target).with_mode(camera_start_mode.0),
    ));

    // Ticket 015: a directional light replaces the old `PointLight`, which
    // lit only a small sphere near spawn and left the rest of a streaming
    // voxel world flat. Only its rotation matters (set by `sky::sync_sky_palette`
    // from `SkyPalette::sun_direction`), so it needs no particular position
    // or offset from `target` — unlike the point light, it isn't local to
    // anywhere. Ticket 017: shadows on, sized for the current render
    // distance — see `sky::spawn_sun`'s docs.
    sky::spawn_sun(&mut commands, render_distance.0);

    // Ticket 016: the sky camera, gradient dome, and sun/moon billboards —
    // see `sky::SkyPlugin`'s docs for why this is called from here rather
    // than being a `Startup` system the plugin adds itself.
    sky::spawn_sky_scene(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        &sky_palette,
    );
}

/// Bevy-space height every camera placement in this module uses. Real saves
/// are rarely centred on (0,0), and nothing is decoded yet at startup (or
/// right after the UI switches saves, ticket 007) to read real terrain
/// height from, so every placement lands here instead — generally above
/// ground level — rather than on the surface.
const DEFAULT_CAMERA_HEIGHT: f32 = 100.0;

/// Bevy-space point at the horizontal middle of region `(rx, rz)`, at
/// [`DEFAULT_CAMERA_HEIGHT`]. Shared by [`spawn_point`] (the save's overall
/// region centroid) and the UI's region-grid click-to-teleport (ticket
/// 007), so both land on the same convention for "where a region is".
pub(crate) fn region_center_point(rx: i32, rz: i32) -> Vec3 {
    let region_size = REGION_WIDTH_IN_CHUNKS as i32 * world::SECTION_SIZE as i32;
    let mc_x = rx * region_size + region_size / 2;
    let mc_z = rz * region_size + region_size / 2;

    // bevy.x = mc.x, bevy.z = -mc.z — see `world::mesh` docs.
    Vec3::new(mc_x as f32, DEFAULT_CAMERA_HEIGHT, -(mc_z as f32))
}

/// Bevy-space point used to place the camera at startup, and by the UI's
/// save picker (ticket 007) after switching to a different save: the
/// horizontal middle of the save's region footprint (metadata only —
/// `SaveMeta`'s region-coordinate list, no chunk I/O — so this is safe to
/// call before any streaming has happened). Real saves are rarely centred
/// on (0,0); landing the camera near where regions actually exist means
/// streaming has something to load in view immediately, rather than the
/// camera free-flying over empty space until it happens to reach one.
pub(crate) fn spawn_point(meta: &SaveMeta) -> Vec3 {
    let Some((rx, rz)) = region_centroid(&meta.regions) else {
        return Vec3::new(0.0, DEFAULT_CAMERA_HEIGHT, 0.0);
    };
    region_center_point(rx, rz)
}

/// The save's region closest to the horizontal centroid of every region it
/// has — the centroid itself may land on a coordinate that isn't actually a
/// region (a save's regions needn't form a filled rectangle), so this snaps
/// to whatever region is actually there.
fn region_centroid(regions: &[(i32, i32)]) -> Option<(i32, i32)> {
    if regions.is_empty() {
        return None;
    }
    let n = regions.len() as f64;
    let (sum_x, sum_z) = regions.iter().fold((0i64, 0i64), |(sx, sz), &(rx, rz)| {
        (sx + rx as i64, sz + rz as i64)
    });
    let avg_x = sum_x as f64 / n;
    let avg_z = sum_z as f64 / n;

    regions.iter().copied().min_by(|a, b| {
        let dist2 =
            |&(rx, rz): &(i32, i32)| (rx as f64 - avg_x).powi(2) + (rz as f64 - avg_z).powi(2);
        dist2(a).total_cmp(&dist2(b))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixture saves directory holding `names`, each a save with an empty
    /// (but present, so `SaveMeta::from_path` can read it) region directory.
    fn saves_fixture(tag: &str, names: &[&str]) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("block_viewer_test_saves_{tag}_{}_{seq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for name in names {
            std::fs::create_dir_all(dir.join(name).join("region")).expect("should create fixture");
        }
        std::fs::create_dir_all(&dir).expect("should create fixture");
        dir
    }

    fn args(first: Option<&str>, second: Option<&str>) -> SaveArgs {
        save_args_from(first.map(str::to_string), second.map(str::to_string))
    }

    /// Ticket 008's argument, unchanged: a directory that lists saves is the
    /// saves directory even when it's the only argument given.
    #[test]
    fn a_lone_saves_directory_argument_is_still_the_directory() {
        let fixture = saves_fixture("instance", &["Flat", "MyCityWorld"]);
        let selection =
            resolve_selection_in(&args(Some(fixture.to_str().unwrap()), None), Path::new("default"));
        assert_eq!(selection.directory, fixture);
        assert!(selection.wanted.is_none(), "a directory names no save in particular");
        std::fs::remove_dir_all(&fixture).ok();
    }

    /// Ticket 064's whole point after the `-- "" name` form turned out not to
    /// survive Windows PowerShell 5.1: a lone argument that isn't a saves
    /// directory is a save *name*, looked for under the default directory.
    #[test]
    fn a_lone_non_directory_argument_is_a_save_name() {
        let selection = resolve_selection_in(&args(Some("nbt_test"), None), Path::new("default"));
        assert_eq!(selection.directory, Path::new("default"));
        assert_eq!(selection.wanted.as_deref(), Some("nbt_test"));
    }

    /// A path straight to one save isn't a saves directory either — its own
    /// `region`/`data`/`datapacks` subdirectories aren't saves, so it lists
    /// none and resolves as the save to open.
    #[test]
    fn a_lone_path_to_a_single_save_is_a_save_not_a_directory() {
        let fixture = saves_fixture("single", &["MyCityWorld"]);
        let save_path = fixture.join("MyCityWorld");
        let selection =
            resolve_selection_in(&args(Some(save_path.to_str().unwrap()), None), Path::new("default"));
        assert_eq!(selection.wanted.as_deref(), Some(save_path.to_str().unwrap()));
        std::fs::remove_dir_all(&fixture).ok();
    }

    #[test]
    fn two_arguments_are_the_directory_then_the_save() {
        let selection =
            resolve_selection_in(&args(Some("D:/instance/saves"), Some("MyCityWorld")), Path::new("default"));
        assert_eq!(selection.directory, Path::new("D:/instance/saves"));
        assert_eq!(selection.wanted.as_deref(), Some("MyCityWorld"));
    }

    /// No arguments means "the first save found under the default
    /// directory" — exactly what every run did before ticket 064.
    #[test]
    fn no_arguments_leaves_the_save_unchosen() {
        let selection = resolve_selection_in(&args(None, None), Path::new("default"));
        assert_eq!(selection.directory, Path::new("default"));
        assert!(selection.wanted.is_none());
    }

    /// The original `-- "" MyWorld` form still resolves the same way, for
    /// whoever's shell does pass an empty argument through.
    #[test]
    fn an_empty_first_argument_still_means_the_default_directory() {
        let selection = resolve_selection_in(&args(Some("  "), Some("MyCityWorld")), Path::new("default"));
        assert_eq!(selection.directory, Path::new("default"));
        assert_eq!(selection.wanted.as_deref(), Some("MyCityWorld"));
    }

    /// Windows PowerShell 5.1 can deliver an "empty" argument as two literal
    /// quote characters rather than dropping it — which would otherwise be a
    /// saves directory called `""`.
    #[test]
    fn an_argument_of_nothing_but_quotes_counts_as_not_given() {
        let parsed = args(Some("\"\""), Some("'MyCityWorld'"));
        assert!(parsed.first.is_none());
        assert_eq!(parsed.second.as_deref(), Some("MyCityWorld"));
    }

    fn meta_named(name: &str) -> SaveMeta {
        SaveMeta {
            name: name.to_string(),
            path: PathBuf::from(name),
            region_dir: PathBuf::new(),
            regions: Vec::new(),
        }
    }

    #[test]
    fn pick_named_save_matches_a_saves_name_exactly() {
        let saves = [meta_named("Flat"), meta_named("MyCityWorld")];
        let picked = pick_named_save(&saves, "MyCityWorld", Path::new("saves")).unwrap();
        assert_eq!(picked.name, "MyCityWorld");
    }

    #[test]
    fn pick_named_save_falls_back_to_a_case_insensitive_match() {
        let saves = [meta_named("MyCityWorld")];
        let picked = pick_named_save(&saves, "mycityworld", Path::new("saves")).unwrap();
        assert_eq!(picked.name, "MyCityWorld");
    }

    /// An exact match wins even when a differently-cased name comes first in
    /// the listing.
    #[test]
    fn pick_named_save_prefers_the_exact_match_over_a_cased_one() {
        let saves = [meta_named("MYCITYWORLD"), meta_named("MyCityWorld")];
        let picked = pick_named_save(&saves, "MyCityWorld", Path::new("saves")).unwrap();
        assert_eq!(picked.name, "MyCityWorld");
    }

    /// Ticket 064: a typo'd name must say so *and* list what's there —
    /// silently falling back to the first save (or an empty world) is how a
    /// typo turns into "the citybuilder lost my city".
    #[test]
    fn pick_named_save_errors_with_the_available_names() {
        let saves = [meta_named("Flat"), meta_named("MyCityWorld")];
        let err = pick_named_save(&saves, "Typo", Path::new("saves")).unwrap_err();
        assert!(err.contains("Typo"), "{err}");
        assert!(err.contains("Flat") && err.contains("MyCityWorld"), "{err}");
    }

    #[test]
    fn pick_named_save_errors_when_there_are_no_saves_at_all() {
        let err = pick_named_save(&[], "MyCityWorld", Path::new("saves")).unwrap_err();
        assert!(err.contains("(none)"), "{err}");
    }

    #[test]
    fn the_default_saves_directory_follows_the_minecraft_layout() {
        let dir = default_saves_directory();
        assert!(
            dir.ends_with(Path::new(".minecraft/saves")),
            "expected a `.minecraft/saves` suffix, got {}",
            dir.display()
        );
    }

    /// Ticket 008: no such directory at all (the "no Minecraft installed"
    /// case) should come back as an `Err` with a message, never panic.
    #[test]
    fn try_load_save_from_errors_cleanly_when_the_directory_does_not_exist() {
        let err = try_load_save_from(
            Path::new("definitely-does-not-exist-anywhere/.minecraft/saves"),
            None,
        )
        .unwrap_err();
        assert!(!err.is_empty());
    }

    /// Ticket 008: a saves directory that exists but has nothing under it
    /// (the "empty saves folder" case) should also come back as a clean
    /// `Err`, not a panic — distinct from the directory not existing at all.
    #[test]
    fn try_load_save_from_errors_cleanly_for_an_empty_directory() {
        let empty_dir = std::env::temp_dir().join(format!(
            "block_viewer_test_empty_saves_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&empty_dir).expect("should be able to create a temp dir");

        let err = try_load_save_from(&empty_dir, None).unwrap_err();
        assert!(err.contains("no Minecraft saves found"));

        std::fs::remove_dir_all(&empty_dir).ok();
    }

    #[test]
    fn empty_save_has_no_regions_to_stream() {
        let save = empty_save();
        assert!(save.meta.regions.is_empty());
        assert!(save.regions.is_empty());
    }
}
