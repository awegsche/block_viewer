//! City save/load (ticket 043, roadmap D2): [`state::City`] written to and
//! read from `<save>/citybuilder/city.ron`, so a save and its city travel
//! together — the roadmap's own phrasing for D2.
//!
//! ## Why `occupancy` is never on disk
//!
//! [`CitySave`] is not a serialized [`City`] — it's `buildings`, `road_cells`
//! and `next_id`, the same three things [`State::insert_loaded`](state::City::insert_loaded)
//! and [`state::City::add_road_cell`] already derive `occupancy` from at runtime.
//! Storing the occupancy grid too would let a hand-edited file disagree with
//! itself (two buildings claiming the same tile, say); rebuilding it on load
//! through the same all-or-nothing checks [`state::City::place_building`]
//! uses means a load that would produce an inconsistent grid fails instead
//! of silently trusting stale data.
//!
//! ## `next_id` is persisted, not recomputed
//!
//! [`state::BuildingId`]s are "never reused, even after
//! [`remove_building`](state::City::remove_building)" — 042's own promise.
//! Recomputing `next_id` from only the buildings that survived to be saved
//! would break that the moment the *highest*-numbered building is the one
//! that got removed: the id it held would look free again. The file's
//! `next_id` field is what protects against this; `insert_loaded`'s own bump
//! (raising `next_id` past every id it inserts) is only a floor under that
//! value, not a substitute for it.
//!
//! ## No migration path yet
//!
//! [`CURRENT_VERSION`] is checked for exact equality. A mismatch is
//! [`PersistenceError::UnsupportedVersion`] — refused and logged by the
//! caller, not guessed at. A migration function is later work, once a
//! version actually needs one.
//!
//! Bumped to `2` by ticket 054: version 1's `roads` field held raw block
//! tiles, one per road block; version 2's `road_cells` holds
//! [`super::state::ROAD_CELL_SIZE`]-block cell coordinates instead — the
//! same `(i32, i32)` shape on disk, so a version-1 file would parse cleanly
//! and silently misplace every road cell 6x if this weren't caught by the
//! version check rather than left to guess.

use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::{IVec2, IVec3};
use serde::{Deserialize, Serialize};

use super::state::{BuildingId, City, PlacedBuilding, PlacementError};
use crate::blueprint::Rotation;

/// The `CitySave` schema version this build writes and reads. Bumped only
/// alongside a migration path — see the module docs.
pub const CURRENT_VERSION: u32 = 2;

/// Where [`save_city`]/[`load_city`] look, relative to a save's root
/// (`SaveMeta::path`) — the roadmap's own `<save>/citybuilder/city.ron`.
const CITY_FILE: &str = "citybuilder/city.ron";

/// `<save_root>/citybuilder/city.ron`, exposed for [`super::load_city`]'s
/// own log line — the same path [`save_city`]/[`load_city`] read and write.
pub(crate) fn city_file_path_for_log(save_root: &Path) -> PathBuf {
    city_file_path(save_root)
}

/// The RON file's actual on-disk shape — deliberately not [`City`] itself;
/// see the module docs for why `occupancy` has no field here.
#[derive(Debug, Serialize, Deserialize)]
struct CitySave {
    version: u32,
    next_id: u64,
    buildings: Vec<SavedBuilding>,
    /// Cell coordinates (ticket 054), not block tiles — see the module
    /// docs' version-bump note.
    road_cells: Vec<(i32, i32)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedBuilding {
    id: u64,
    definition: String,
    origin: (i32, i32, i32),
    rotation: Rotation,
    footprint: (i32, i32),
}

/// Why [`save_city`] or [`load_city`] failed.
#[derive(Debug)]
pub enum PersistenceError {
    /// Couldn't read/write/create the file or its directory.
    Io(std::io::Error),
    /// Not well-formed RON, or didn't match [`CitySave`]'s shape. Carries
    /// RON's own message, same convention as
    /// `city::definition::DefinitionError::Parse`.
    Parse(String),
    /// The file's `version` isn't [`CURRENT_VERSION`] — see the module
    /// docs.
    UnsupportedVersion(u32),
    /// A saved building or road tile collided with another on load — a
    /// corrupt or hand-edited file, never a file [`save_city`] itself wrote.
    Corrupt(PlacementError),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::Io(err) => write!(f, "{err}"),
            PersistenceError::Parse(msg) => write!(f, "{msg}"),
            PersistenceError::UnsupportedVersion(version) => {
                write!(f, "city save is version {version}, this build reads version {CURRENT_VERSION}")
            }
            PersistenceError::Corrupt(err) => write!(f, "city save is inconsistent: {err}"),
        }
    }
}

impl std::error::Error for PersistenceError {}

/// `<save_root>/citybuilder/city.ron` — where [`save_city`]/[`load_city`]
/// both read and write.
fn city_file_path(save_root: &Path) -> PathBuf {
    save_root.join(CITY_FILE)
}

/// Writes `city` to `<save_root>/citybuilder/city.ron`, creating the
/// `citybuilder` directory if it doesn't exist yet.
pub fn save_city(city: &City, save_root: &Path) -> Result<(), PersistenceError> {
    let path = city_file_path(save_root);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(PersistenceError::Io)?;
    }

    let mut buildings: Vec<SavedBuilding> = city
        .buildings()
        .map(|(id, building)| SavedBuilding {
            id: id.as_u64(),
            definition: building.definition.clone(),
            origin: (building.origin.x, building.origin.y, building.origin.z),
            rotation: building.rotation,
            footprint: (building.footprint.x, building.footprint.y),
        })
        .collect();
    buildings.sort_by_key(|b| b.id);

    let mut road_cells: Vec<(i32, i32)> = city.road_cells().map(|cell| (cell.x, cell.y)).collect();
    road_cells.sort();

    let save = CitySave { version: CURRENT_VERSION, next_id: city.next_id_raw(), buildings, road_cells };
    // Pretty-printed: a person may want to read or hand-edit this file, the
    // same call city::definition's RON files are written for by hand.
    let text = ron::ser::to_string_pretty(&save, ron::ser::PrettyConfig::default())
        .map_err(|err| PersistenceError::Parse(err.to_string()))?;
    fs::write(&path, text).map_err(PersistenceError::Io)
}

/// Reads `<save_root>/citybuilder/city.ron` into a fresh [`City`]. A missing
/// file is `Ok(City::default())` — a save with no citybuilder data yet is
/// not an error, the same contract `blueprint::load_catalogue_dir`/
/// `city::definition::load_definitions_dir` use for a missing directory.
pub fn load_city(save_root: &Path) -> Result<City, PersistenceError> {
    let path = city_file_path(save_root);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(City::default()),
        Err(err) => return Err(PersistenceError::Io(err)),
    };

    let save: CitySave = ron::de::from_str(&text).map_err(|err| PersistenceError::Parse(err.to_string()))?;
    if save.version != CURRENT_VERSION {
        return Err(PersistenceError::UnsupportedVersion(save.version));
    }

    let mut city = City::default();
    let mut buildings = save.buildings;
    buildings.sort_by_key(|b| b.id);
    for saved in buildings {
        let (x, y, z) = saved.origin;
        let (fx, fz) = saved.footprint;
        let building = PlacedBuilding {
            definition: saved.definition,
            origin: IVec3::new(x, y, z),
            rotation: saved.rotation,
            footprint: IVec2::new(fx, fz),
        };
        city.insert_loaded(BuildingId::from_u64(saved.id), building)
            .map_err(PersistenceError::Corrupt)?;
    }

    let mut road_cells = save.road_cells;
    road_cells.sort();
    for (x, z) in road_cells {
        city.add_road_cell(IVec2::new(x, z)).map_err(PersistenceError::Corrupt)?;
    }

    // `insert_loaded` already raised `next_id` past every id it inserted;
    // the file's own value is the floor beneath that — see the module docs
    // for the removed-highest-building scenario this guards against.
    city.raise_next_id(save.next_id);

    Ok(city)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A fresh empty directory under the OS temp dir, unique per call — same
    /// shape `city::definition`'s tests use.
    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_persistence_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    #[test]
    fn a_missing_file_loads_as_an_empty_city() {
        let dir = temp_dir("missing");
        let city = load_city(&dir).expect("a missing save file is not an error");
        assert!(city.is_empty());
        assert_eq!(city.road_cells().count(), 0);
    }

    #[test]
    fn an_empty_city_round_trips() {
        let dir = temp_dir("empty_round_trip");
        let city = City::default();
        save_city(&city, &dir).unwrap();

        let loaded = load_city(&dir).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.road_cells().count(), 0);
    }

    #[test]
    fn buildings_and_roads_round_trip_exactly() {
        let dir = temp_dir("full_round_trip");
        let mut city = City::default();
        let a = city
            .place_building("house01", IVec3::new(10, 64, 20), Rotation::Deg0, IVec2::new(3, 2))
            .unwrap();
        let b = city
            .place_building("house01", IVec3::new(0, 70, 0), Rotation::Deg90, IVec2::new(3, 5))
            .unwrap();
        city.add_road_cell(IVec2::new(50, 50)).unwrap();
        city.add_road_cell(IVec2::new(50, 51)).unwrap();

        save_city(&city, &dir).unwrap();
        let loaded = load_city(&dir).unwrap();

        assert_eq!(loaded.len(), 2);
        let loaded_a = loaded.building(a).expect("building a should round-trip under the same id");
        assert_eq!(loaded_a.definition, "house01");
        assert_eq!(loaded_a.origin, IVec3::new(10, 64, 20));
        assert_eq!(loaded_a.rotation, Rotation::Deg0);
        assert_eq!(loaded_a.footprint, IVec2::new(3, 2));

        let loaded_b = loaded.building(b).expect("building b should round-trip under the same id");
        assert_eq!(loaded_b.rotation, Rotation::Deg90);
        // The rotated occupancy rectangle should have round-tripped too.
        assert_eq!(loaded.occupant_at(IVec2::new(4, 2)), Some(crate::city::state::Occupant::Building(b)));

        assert_eq!(loaded.road_cells().count(), 2);
        assert!(loaded.is_road_cell(IVec2::new(50, 50)));
    }

    /// The scenario the module docs call out: the *highest*-id building is
    /// the one removed before saving, so recomputing `next_id` from the
    /// survivors alone would reissue its id.
    #[test]
    fn next_id_survives_removal_of_the_highest_id_building() {
        let dir = temp_dir("next_id_gap");
        let mut city = City::default();
        let first = city
            .place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let second = city
            .place_building("house01", IVec3::new(5, 64, 5), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        city.remove_building(second).unwrap();

        save_city(&city, &dir).unwrap();
        let mut loaded = load_city(&dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.building(first).is_some());

        // A fresh placement must not reissue `second`'s old id.
        let third = loaded
            .place_building("house01", IVec3::new(10, 64, 10), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        assert_ne!(third, second, "second's id must never be reissued, even across a save/load round trip");
    }

    #[test]
    fn a_version_mismatch_is_refused() {
        let dir = temp_dir("version_mismatch");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            "(version: 999, next_id: 0, buildings: [], road_cells: [])",
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::UnsupportedVersion(999)));
    }

    /// Ticket 054: a version-1 file's `roads` field held raw block tiles,
    /// not cell coordinates — renamed to `road_cells` on the version-2
    /// struct, so a version-1 file (still carrying the old field name) fails
    /// to deserialize into the new shape at all. Refused as a parse error,
    /// same as any other malformed file — never silently loaded with old
    /// block-tile roads reinterpreted as cell coordinates 6x too large.
    #[test]
    fn an_old_block_tile_road_save_is_refused_not_silently_reinterpreted() {
        let dir = temp_dir("old_road_format");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            "(version: 1, next_id: 0, buildings: [], roads: [(5, 5)])",
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Parse(_)), "{err:?}");
    }

    #[test]
    fn garbage_ron_is_a_parse_error_not_a_panic() {
        let dir = temp_dir("garbage");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(dir.join("citybuilder/city.ron"), b"not valid ron at all {{{").unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Parse(_)));
    }

    #[test]
    fn two_overlapping_saved_buildings_is_corrupt_not_a_bad_city() {
        let dir = temp_dir("overlap");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            r#"(
                version: 2,
                next_id: 2,
                buildings: [
                    (id: 0, definition: "house01", origin: (0, 64, 0), rotation: Deg0, footprint: (4, 4)),
                    (id: 1, definition: "house01", origin: (3, 64, 3), rotation: Deg0, footprint: (4, 4)),
                ],
                road_cells: [],
            )"#,
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Corrupt(PlacementError::TileOccupied { .. })));
    }

    #[test]
    fn a_saved_road_cell_colliding_with_a_building_is_corrupt() {
        let dir = temp_dir("road_overlap");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            r#"(
                version: 2,
                next_id: 1,
                buildings: [
                    (id: 0, definition: "house01", origin: (0, 64, 0), rotation: Deg0, footprint: (2, 2)),
                ],
                road_cells: [(0, 0)],
            )"#,
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Corrupt(PlacementError::TileOccupied { .. })));
    }
}
