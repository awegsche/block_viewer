//! Hot reload for definition files, and the errors a bad edit produces
//! (ticket 061, roadmap C4): "definition files reload on change; errors go
//! to an egui panel rather than a panic or a console line nobody sees."
//!
//! ## What's watched, and what isn't
//!
//! Only the two RON "game data" directories the roadmap's C-group is about —
//! [`super::DEFINITIONS_DIR`] ([`super::definition::BuildingDefinitions`])
//! and [`super::ROAD_TYPES_DIR`] ([`super::road_definition::RoadTypes`]).
//! [`super::CATALOGUE_DIR`]/[`super::ROAD_CATALOGUE_DIR`] hold `.nbt`
//! *geometry*, not the numbers/references C1's schema sketch describes, and
//! a building or road cell already placed carries no live reference back
//! into either catalogue that a reload could dangle — [`super::state::City`]
//! stores ids, resolved against whatever's currently loaded on every read.
//!
//! ## Polling, not a filesystem watcher
//!
//! No new dependency: [`poll_definition_reload`] runs on a plain repeating
//! [`Timer`] and compares each directory's `(path, mtime)` snapshot to the
//! one it saw last tick — [`dir_snapshot`]. A changed, added or removed
//! `.ron` file changes the snapshot; nothing else does. This is a stat() per
//! file per tick, which is nothing next to the terrain streaming already
//! running every frame, so there's no reason to reach for a `notify`-style
//! watcher (a new dependency, and a thread/channel bridge into Bevy's
//! `Resource` world) for a directory that holds a handful of files edited by
//! hand.
//!
//! ## Swapping a `Resource` mid-game is safe here
//!
//! [`BuildingDefinitions`]/[`RoadTypes`] have exactly one kind of reader
//! today: `city::ui::build_menu` (what to show) and this module's own error
//! panel. Nothing holds a `&LoadedBuilding`/`&LoadedRoadType` across frames —
//! every read is a fresh `.get(id)` off the current `Res`, so replacing the
//! whole resource between frames (rather than diffing and patching it in
//! place) is exactly as safe as the startup load already was, just later.
//!
//! ## The first tick doesn't re-log the startup load
//!
//! [`super::run`] seeds [`DefinitionSnapshot`]/[`RoadTypeSnapshot`] from the
//! same directory scan the startup load already did, and
//! [`super::DefinitionErrors`] from that load's own `skipped` list — so
//! [`poll_definition_reload`]'s first `Update` tick sees no change and does
//! nothing, rather than reloading (and re-printing) everything a second time
//! a moment after startup's own log lines.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bevy::prelude::*;

use crate::blueprint::BuildingCatalogue;

use super::definition::{self, BuildingDefinitions};
use super::road_catalogue::RoadCatalogue;
use super::road_definition::{self, RoadTypes};
use super::{DEFINITIONS_DIR, ROAD_TYPES_DIR};

/// How often [`poll_definition_reload`] actually touches the filesystem.
/// Short enough that an edit-then-alt-tab-back feels instant, long enough
/// that stat'ing two small directories is not worth measuring.
const POLL_INTERVAL_SECS: f32 = 1.0;

/// Ticks [`poll_definition_reload`] — a plain repeating [`Timer`], not tied
/// to any particular save or definitions directory.
#[derive(Resource)]
struct ReloadTimer(Timer);

impl Default for ReloadTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(POLL_INTERVAL_SECS, TimerMode::Repeating))
    }
}

/// The last `(path, mtime)` snapshot [`poll_definition_reload`] saw for
/// [`DEFINITIONS_DIR`] — a change here is what triggers a
/// [`BuildingDefinitions`] reload. Seeded by [`super::run`], not
/// `#[derive(Default)]`'s empty map — see the module docs' "first tick"
/// note.
#[derive(Resource, Default, Clone)]
pub(super) struct DefinitionSnapshot(pub(super) HashMap<PathBuf, SystemTime>);

/// [`DefinitionSnapshot`]'s counterpart for [`ROAD_TYPES_DIR`]/[`RoadTypes`].
#[derive(Resource, Default, Clone)]
pub(super) struct RoadTypeSnapshot(pub(super) HashMap<PathBuf, SystemTime>);

/// The definition-loading problems currently in effect — startup's own
/// `skipped` list, replaced wholesale by every reload that finds a change.
/// [`super::ui::definition_errors_panel`] is the reader this exists for: a
/// bad `.ron` file shows up somewhere a player watching the window will
/// actually see, not only in a console scrollback nobody's looking at while
/// they're testing a balance change.
#[derive(Resource, Default)]
pub struct DefinitionErrors {
    /// `(file path, error message)` — the message already includes whatever
    /// context [`definition::DefinitionError`]'s `Display` carries; stored as
    /// a plain `String` because the two source directories' error enums are
    /// otherwise unrelated types, and nothing here needs to match on them.
    pub buildings: Vec<(PathBuf, String)>,
    pub road_types: Vec<(PathBuf, String)>,
    /// `assets/city/drops.ron` (ticket 072) — at most one entry, since the
    /// drop table is a single file with no per-entry recovery. Seeded by
    /// [`super::run`] and never touched again: unlike the two directories
    /// above, the table is **not** hot-reloaded. The snapshot machinery
    /// watches directories, and a one-file watcher is a ticket of its own if
    /// it turns out to be wanted.
    pub drops: Vec<(PathBuf, String)>,
    /// `assets/city/economy.ron` (ticket 074) — same shape and same
    /// not-hot-reloaded caveat as [`Self::drops`].
    pub economy: Vec<(PathBuf, String)>,
}

impl DefinitionErrors {
    pub fn is_empty(&self) -> bool {
        self.buildings.is_empty() && self.road_types.is_empty() && self.drops.is_empty() && self.economy.is_empty()
    }
}

/// Adds [`poll_definition_reload`]. [`super::run`] is expected to have
/// already inserted [`DefinitionSnapshot`], [`RoadTypeSnapshot`] and
/// [`DefinitionErrors`] seeded from the startup load — see the module docs.
pub struct DefinitionHotReloadPlugin;

impl Plugin for DefinitionHotReloadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReloadTimer>()
            .add_systems(Update, poll_definition_reload);
    }
}

/// `dir`'s `*.ron` files as `(path, last-modified)` pairs — the comparable
/// unit [`poll_definition_reload`] diffs tick to tick. A missing directory,
/// or a file whose metadata can't be read, is silently absent rather than an
/// error: the loaders this feeds already treat a missing directory as "zero
/// definitions", and a file that vanished between `read_dir` and `metadata`
/// (mid-save-by-an-editor) is the same "not there right now" case.
pub(super) fn dir_snapshot(dir: &Path) -> HashMap<PathBuf, SystemTime> {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return HashMap::new();
    };
    read_dir
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("ron")))
        .filter_map(|path| {
            let modified = fs::metadata(&path).ok()?.modified().ok()?;
            Some((path, modified))
        })
        .collect()
}

/// Every [`POLL_INTERVAL_SECS`], re-snapshots [`DEFINITIONS_DIR`] and
/// [`ROAD_TYPES_DIR`]; a directory whose snapshot changed since last tick
/// gets fully reloaded and its resource replaced, and
/// [`DefinitionErrors`] is refreshed to match. The two directories are
/// independent — an edit under one reloads only that one, so a syntax error
/// mid-edit in `buildings/` doesn't also discard a perfectly good
/// `road_types/` load.
fn poll_definition_reload(
    time: Res<Time>,
    mut timer: ResMut<ReloadTimer>,
    mut building_snapshot: ResMut<DefinitionSnapshot>,
    mut road_type_snapshot: ResMut<RoadTypeSnapshot>,
    mut definitions: ResMut<BuildingDefinitions>,
    mut road_types: ResMut<RoadTypes>,
    mut errors: ResMut<DefinitionErrors>,
    catalogue: Res<BuildingCatalogue>,
    road_catalogue: Res<RoadCatalogue>,
) {
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }

    let new_building_snapshot = dir_snapshot(Path::new(DEFINITIONS_DIR));
    if new_building_snapshot != building_snapshot.0 {
        let (reloaded, skipped) = definition::load_definitions_dir(Path::new(DEFINITIONS_DIR), &catalogue);
        println!(
            "block_viewer: reloaded {} building definition{} from {DEFINITIONS_DIR}",
            reloaded.len(),
            if reloaded.len() == 1 { "" } else { "s" }
        );
        for (path, err) in &skipped {
            println!("block_viewer:   skipped {}: {err}", path.display());
        }
        errors.buildings = skipped.into_iter().map(|(path, err)| (path, err.to_string())).collect();
        *definitions = reloaded;
        building_snapshot.0 = new_building_snapshot;
    }

    let new_road_type_snapshot = dir_snapshot(Path::new(ROAD_TYPES_DIR));
    if new_road_type_snapshot != road_type_snapshot.0 {
        let (reloaded, skipped) = road_definition::load_road_types_dir(Path::new(ROAD_TYPES_DIR), &road_catalogue);
        println!(
            "block_viewer: reloaded {} road type{} from {ROAD_TYPES_DIR}",
            reloaded.len(),
            if reloaded.len() == 1 { "" } else { "s" }
        );
        for (path, err) in &skipped {
            println!("block_viewer:   skipped {}: {err}", path.display());
        }
        errors.road_types = skipped.into_iter().map(|(path, err)| (path, err.to_string())).collect();
        *road_types = reloaded;
        road_type_snapshot.0 = new_road_type_snapshot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("block_viewer_test_hot_reload_{name}_{}_{unique}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    #[test]
    fn a_missing_directory_snapshots_empty() {
        let dir = std::env::temp_dir().join("block_viewer_test_hot_reload_does_not_exist");
        let _ = fs::remove_dir_all(&dir);
        assert!(dir_snapshot(&dir).is_empty());
    }

    #[test]
    fn only_ron_files_are_included() {
        let dir = temp_dir("filter");
        fs::write(dir.join("a.ron"), "Building()").unwrap();
        fs::write(dir.join("readme.txt"), "not a definition").unwrap();

        let snapshot = dir_snapshot(&dir);
        assert_eq!(snapshot.len(), 1);
        assert!(snapshot.contains_key(&dir.join("a.ron")));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn editing_a_file_changes_its_snapshot_entry() {
        let dir = temp_dir("edit");
        let path = dir.join("a.ron");
        fs::write(&path, "Building()").unwrap();
        let before = dir_snapshot(&dir);

        // Force a distinguishable mtime rather than relying on the clock
        // ticking forward between two writes a few microseconds apart, which
        // is flaky on filesystems with coarse mtime resolution (FAT32's is a
        // full 2 seconds). `set_modified` needs a handle opened for write —
        // `File::open`'s read-only handle is denied permission on Windows.
        let future = SystemTime::now() + std::time::Duration::from_secs(5);
        fs::write(&path, "Building(name: \"changed\")").unwrap();
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(future).unwrap();

        let after = dir_snapshot(&dir);
        assert_ne!(before.get(&path), after.get(&path));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn adding_or_removing_a_file_changes_the_snapshot() {
        let dir = temp_dir("add_remove");
        let before = dir_snapshot(&dir);
        assert!(before.is_empty());

        let path = dir.join("a.ron");
        fs::write(&path, "Building()").unwrap();
        let with_file = dir_snapshot(&dir);
        assert_eq!(with_file.len(), 1);
        assert_ne!(before, with_file);

        fs::remove_file(&path).unwrap();
        let after_remove = dir_snapshot(&dir);
        assert!(after_remove.is_empty());
        assert_ne!(with_file, after_remove);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn definition_errors_is_empty_only_when_both_lists_are() {
        let mut errors = DefinitionErrors::default();
        assert!(errors.is_empty());

        errors.buildings.push((PathBuf::from("a.ron"), "broken".to_string()));
        assert!(!errors.is_empty());

        errors.buildings.clear();
        errors.road_types.push((PathBuf::from("b.ron"), "also broken".to_string()));
        assert!(!errors.is_empty());
    }
}
