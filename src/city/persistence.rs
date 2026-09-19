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
//!
//! Bumped again to `3` by ticket 059: a road cell now remembers which style
//! (`super::road_catalogue::RoadCatalogue`'s key) it was built as, so
//! [`SavedRoadCell`] gained a `style` field a version-2 file's bare
//! `(i32, i32)` tuple doesn't have — same call as the 1 -> 2 bump, refused
//! rather than guessed at.
//!
//! Bumped again to `4` by ticket 065: a road cell also remembers the world
//! `y` it was built at ([`super::state::RoadCell::base_y`]) — before that
//! fix every road piece was written at a hardcoded Y of 0, so a version-3
//! file's cells have no height to recover and defaulting them to *anything*
//! would either re-bury them or drop them on terrain they were never fitted
//! to. Refused rather than guessed at, same as the two bumps before it.
//!
//! Bumped again to `5` by ticket 067: a road cell can now be a *stair*,
//! carrying the direction it climbs ([`super::state::RoadCell::ascent`]).
//! A version-4 file's cells are all implicitly flat, which — unlike the
//! earlier bumps — would actually be a *safe* default to fill in. It's still
//! refused, for the same reason the version check is an equality test at
//! all: the moment one field is quietly defaulted, the next one that isn't
//! safe to default has to argue its case against a precedent. A migration
//! function is the answer when one is worth writing, not a per-field
//! exception here.
//!
//! Bumped again to `7` by ticket 076: a placement now records the
//! *definition* it was built from ([`super::state::PlacedBuilding::definition_id`])
//! alongside the catalogue id it always did — and the field that held the
//! catalogue id, misleadingly called `definition` since 043, is renamed
//! `catalogue_id` to say so. Both halves make a version-6 file
//! unreadable-as-written rather than merely incomplete: its `definition`
//! field is a *catalogue* id, so reading it into the new `definition_id`
//! would hand every loaded building a definition key that is only correct
//! while the two stems coincide — which is the exact bug 076 removes.
//! Refused, same as the five bumps before it.
//!
//! Bumped again to `6` by ticket 071: a road cell also remembers whether it
//! was built as a surface or a *tunnel* piece
//! ([`super::state::RoadCell::variant`]). Defaulting a version-5 file's
//! cells to `Surface` would be wrong in the one case that matters — a road
//! already cut through a hill would re-tile itself back into solid
//! hillside — and this is the field the "no quiet defaults" precedent above
//! was being kept for. Refused, same as the four bumps before it.
//!
//! *Not* bumped by ticket 111, which adds [`SavedBuilding::work_area`] with
//! `#[serde(default)]` instead — the first field to make the case the
//! precedent above asks for. Every earlier default would have been a
//! *guess* (a tunnel read as surface, a catalogue id read as a definition
//! id); a version-7 file's gatherer huts genuinely never had an area drawn,
//! so `None` is the truth about that file, not a stand-in for it. Same
//! argument `super::journal`'s `SavedPlacement::definition_id` already made
//! for itself.

use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::{IVec2, IVec3};
use serde::{Deserialize, Serialize};

use super::road::{Direction, RoadPieceVariant};
use super::state::{BuildingId, City, PlacedBuilding, PlacementError, WorkArea};
use crate::blueprint::Rotation;

/// The `CitySave` schema version this build writes and reads. Bumped only
/// alongside a migration path — see the module docs.
pub const CURRENT_VERSION: u32 = 7;

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
    /// Cell coordinates (ticket 054), not block tiles, each carrying its own
    /// style (ticket 059) — see the module docs' version-bump notes.
    road_cells: Vec<SavedRoadCell>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedBuilding {
    id: u64,
    /// The blueprint stem — `super::blueprint::BuildingCatalogue`'s key.
    /// Called `definition` on disk up to version 6, which is what it had
    /// never been; see the module docs' version-7 note.
    catalogue_id: String,
    /// The `.ron` stem — `super::definition::BuildingDefinitions`'s key, and
    /// `None` for a placement made without a definition behind it. Ticket
    /// 076.
    definition_id: Option<String>,
    origin: (i32, i32, i32),
    rotation: Rotation,
    footprint: (i32, i32),
    /// Ticket 111: a gatherer's drawn working area, `((min_x, min_z),
    /// (max_x, max_z))` inclusive — see [`super::state::WorkArea`]. The
    /// first field here that *defaults* rather than bumping
    /// [`CURRENT_VERSION`]: see the module docs.
    #[serde(default)]
    work_area: Option<((i32, i32), (i32, i32))>,
    /// Ticket 128: `true` while this placement is still a site —
    /// `city::construction` re-enters it as one on load, with carry
    /// restarting at 0. `#[serde(default)]`, same argument `work_area`
    /// already makes: every building in a file written before this ticket
    /// finished its write before it was ever saved, so `false` is the truth
    /// about it, not a guess.
    #[serde(default)]
    under_construction: bool,
}

/// One saved road cell — cell coordinates plus the style
/// (`super::road_catalogue::RoadCatalogue`'s key) it was built as, the world
/// Y its piece sits at (ticket 065), for a stair which way it climbs
/// (ticket 067), and whether it is a tunnel (ticket 071).
#[derive(Debug, Serialize, Deserialize)]
struct SavedRoadCell {
    x: i32,
    z: i32,
    /// World Y, *not* a cell coordinate — unlike `x`/`z`, which are cell
    /// coordinates (ticket 054). See [`super::state::RoadCell::base_y`].
    y: i32,
    style: String,
    /// `None` for a flat cell; the direction a stair cell climbs otherwise.
    /// [`super::road::Direction`] itself, not a mirror enum — see that
    /// type's own docs.
    ascent: Option<Direction>,
    /// Surface or tunnel — [`super::road::RoadPieceVariant`] itself, not a
    /// mirror enum, for the same reason `ascent` carries `Direction`.
    variant: RoadPieceVariant,
    /// Ticket 128, same `#[serde(default)]`/no-bump argument
    /// [`SavedBuilding::under_construction`] carries.
    #[serde(default)]
    under_construction: bool,
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

    // Ticket 128: `placements()`, not `buildings()` — a site is claimed
    // ground with a journal entry, and has to survive a save/load round trip
    // exactly like a completed building does, or reloading mid-clearing
    // would simply lose it.
    let mut buildings: Vec<SavedBuilding> = city
        .placements()
        .map(|(id, building)| SavedBuilding {
            id: id.as_u64(),
            catalogue_id: building.catalogue_id.clone(),
            definition_id: building.definition_id.clone(),
            origin: (building.origin.x, building.origin.y, building.origin.z),
            rotation: building.rotation,
            footprint: (building.footprint.x, building.footprint.y),
            work_area: building.work_area.map(|a| ((a.min.x, a.min.y), (a.max.x, a.max.y))),
            under_construction: building.under_construction,
        })
        .collect();
    buildings.sort_by_key(|b| b.id);

    let mut road_cells: Vec<SavedRoadCell> = city
        .road_cells_with_data()
        .map(|(cell, road)| SavedRoadCell {
            x: cell.x,
            z: cell.y,
            y: road.base_y,
            style: road.style.clone(),
            ascent: road.ascent,
            variant: road.variant,
            under_construction: road.under_construction,
        })
        .collect();
    road_cells.sort_by_key(|cell| (cell.x, cell.z));

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
            catalogue_id: saved.catalogue_id,
            definition_id: saved.definition_id,
            origin: IVec3::new(x, y, z),
            rotation: saved.rotation,
            footprint: IVec2::new(fx, fz),
            work_area: saved.work_area.map(|((ax, az), (bx, bz))| WorkArea { min: IVec2::new(ax, az), max: IVec2::new(bx, bz) }),
            under_construction: saved.under_construction,
        };
        city.insert_loaded(BuildingId::from_u64(saved.id), building)
            .map_err(PersistenceError::Corrupt)?;
    }

    let mut road_cells = save.road_cells;
    road_cells.sort_by_key(|cell| (cell.x, cell.z));
    for cell in road_cells {
        let coord = IVec2::new(cell.x, cell.z);
        city.add_road_cell(coord, cell.style, cell.y, cell.ascent, cell.variant)
            .map_err(PersistenceError::Corrupt)?;
        // Ticket 128: `add_road_cell` always inserts a fresh cell as
        // complete — re-enter it as a site (carry restarts at 0) if that's
        // what it was when it was saved.
        if cell.under_construction {
            city.set_road_cell_under_construction(coord, true);
        }
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
            .place_building("house01", Some("house01".to_string()), IVec3::new(10, 64, 20), Rotation::Deg0, IVec2::new(3, 2))
            .unwrap();
        let b = city
            .place_building("house01", None, IVec3::new(0, 70, 0), Rotation::Deg90, IVec2::new(3, 5))
            .unwrap();
        city.add_road_cell(IVec2::new(50, 50), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.add_road_cell(IVec2::new(50, 51), "paved", 71, None, RoadPieceVariant::Surface).unwrap();
        // Ticket 067: a stair cell, so the ascent field is exercised by the
        // round trip rather than only by its own dedicated test.
        city.add_road_cell(IVec2::new(50, 52), "dirt", 71, Some(Direction::East), RoadPieceVariant::Surface).unwrap();
        // Ticket 071: and a tunnel cell, for the same reason.
        city.add_road_cell(IVec2::new(50, 53), "dirt", 71, None, RoadPieceVariant::Tunnel).unwrap();

        save_city(&city, &dir).unwrap();
        let loaded = load_city(&dir).unwrap();

        assert_eq!(loaded.len(), 2);
        let loaded_a = loaded.building(a).expect("building a should round-trip under the same id");
        assert_eq!(loaded_a.catalogue_id, "house01");
        // Ticket 076: the definition id round-trips as its own field, and a
        // placement made without one comes back without one rather than
        // borrowing the catalogue id.
        assert_eq!(loaded_a.definition_id.as_deref(), Some("house01"));
        assert_eq!(
            loaded.building(b).and_then(|placed| placed.definition_id.clone()),
            None,
            "building b was placed without a definition and must come back without one"
        );
        assert_eq!(loaded_a.origin, IVec3::new(10, 64, 20));
        assert_eq!(loaded_a.rotation, Rotation::Deg0);
        assert_eq!(loaded_a.footprint, IVec2::new(3, 2));

        let loaded_b = loaded.building(b).expect("building b should round-trip under the same id");
        assert_eq!(loaded_b.rotation, Rotation::Deg90);
        // The rotated occupancy rectangle should have round-tripped too.
        assert_eq!(loaded.occupant_at(IVec2::new(4, 2)), Some(crate::city::state::Occupant::Building(b)));

        assert_eq!(loaded.road_cells().count(), 4);
        assert!(loaded.is_road_cell(IVec2::new(50, 50)));
        // Ticket 059: each cell's own style round-trips too, not just its
        // coordinate.
        assert_eq!(loaded.road_style_at(IVec2::new(50, 50)), Some("dirt"));
        // Ticket 065: so does the height it was built at — two cells at
        // different heights come back at their own, not at a shared default.
        assert_eq!(loaded.road_cell_at(IVec2::new(50, 50)).map(|road| road.base_y), Some(64));
        assert_eq!(loaded.road_cell_at(IVec2::new(50, 51)).map(|road| road.base_y), Some(71));
        assert_eq!(loaded.road_style_at(IVec2::new(50, 51)), Some("paved"));
        // Ticket 067: a flat cell comes back flat and a stair comes back
        // climbing the way it was built, not merely "some stair".
        assert_eq!(loaded.road_cell_at(IVec2::new(50, 50)).and_then(|road| road.ascent), None);
        assert_eq!(loaded.road_cell_at(IVec2::new(50, 52)).and_then(|road| road.ascent), Some(Direction::East));
        // Ticket 071: a tunnel comes back a tunnel, and a surface cell isn't
        // quietly promoted into one.
        assert_eq!(
            loaded.road_cell_at(IVec2::new(50, 53)).map(|road| road.variant),
            Some(RoadPieceVariant::Tunnel)
        );
        assert_eq!(
            loaded.road_cell_at(IVec2::new(50, 50)).map(|road| road.variant),
            Some(RoadPieceVariant::Surface)
        );
    }

    /// A placement whose definition stem differs from its blueprint stem
    /// round-trips as two distinct ids — the case that could not be
    /// represented at all before ticket 076, and the reason `city.ron` moved
    /// to version 7 rather than defaulting the new field.
    #[test]
    fn the_two_ids_round_trip_independently() {
        let dir = temp_dir("two_ids");
        let mut city = City::default();
        let id = city
            .place_building("house01", Some("manor".to_string()), IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();

        save_city(&city, &dir).unwrap();
        let loaded = load_city(&dir).unwrap();

        let placed = loaded.building(id).expect("the building should round-trip under the same id");
        assert_eq!(placed.catalogue_id, "house01", "the geometry it was placed from");
        assert_eq!(placed.definition_id.as_deref(), Some("manor"), "the game data it was placed from");
        assert_eq!(loaded.definition_of(id), Some("manor"));
    }

    /// The scenario the module docs call out: the *highest*-id building is
    /// the one removed before saving, so recomputing `next_id` from the
    /// survivors alone would reissue its id.
    #[test]
    fn next_id_survives_removal_of_the_highest_id_building() {
        let dir = temp_dir("next_id_gap");
        let mut city = City::default();
        let first = city
            .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let second = city
            .place_building("house01", None, IVec3::new(5, 64, 5), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        city.remove_building(second).unwrap();

        save_city(&city, &dir).unwrap();
        let mut loaded = load_city(&dir).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.building(first).is_some());

        // A fresh placement must not reissue `second`'s old id.
        let third = loaded
            .place_building("house01", None, IVec3::new(10, 64, 10), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        assert_ne!(third, second, "second's id must never be reissued, even across a save/load round trip");
    }

    /// Ticket 111: a drawn area survives a save/load, and a file written
    /// before areas existed (no `work_area` field at all) loads with every
    /// building at `None` — see the module docs on why this field defaults
    /// rather than bumping the version.
    #[test]
    fn a_work_area_round_trips_and_an_old_file_without_one_loads_as_none() {
        let dir = temp_dir("work_area_round_trip");
        let mut city = City::default();
        let a = city.place_building("gatherer_hut", Some("gatherer_hut".to_string()), IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::splat(2)).unwrap();
        let b = city.place_building("house01", None, IVec3::new(20, 64, 20), Rotation::Deg0, IVec2::ONE).unwrap();
        let area = WorkArea { min: IVec2::new(-5, -3), max: IVec2::new(7, 4) };
        city.set_work_area(a, Some(area));
        save_city(&city, &dir).unwrap();

        let loaded = load_city(&dir).unwrap();
        assert_eq!(loaded.building(a).unwrap().work_area, Some(area));
        assert_eq!(loaded.building(b).unwrap().work_area, None);

        fs::write(
            city_file_path(&dir),
            "(version: 7, next_id: 1, buildings: [(id: 0, catalogue_id: \"gatherer_hut\", definition_id: None, origin: (0, 64, 0), rotation: Deg0, footprint: (2, 2))], road_cells: [])",
        )
        .unwrap();
        let loaded = load_city(&dir).unwrap();
        assert_eq!(loaded.building(BuildingId::from_u64(0)).unwrap().work_area, None);
    }

    // --- construction sites (ticket 128) --------------------------------------

    #[test]
    fn under_construction_round_trips_for_both_buildings_and_road_cells() {
        let dir = temp_dir("under_construction_round_trip");
        let mut city = City::default();
        let site = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        city.mark_under_construction(site);
        let done = city.place_building("house01", None, IVec3::new(1, 64, 1), Rotation::Deg0, IVec2::ONE).unwrap();
        // Cells far from both buildings' single-tile footprints.
        city.add_road_cell(IVec2::new(5, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        city.set_road_cell_under_construction(IVec2::new(5, 0), true);
        city.add_road_cell(IVec2::new(6, 1), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();

        save_city(&city, &dir).unwrap();
        let loaded = load_city(&dir).unwrap();

        assert!(loaded.placements().find(|(id, _)| *id == site).unwrap().1.under_construction);
        assert!(!loaded.placements().find(|(id, _)| *id == done).unwrap().1.under_construction);
        assert!(loaded.road_cell_at(IVec2::new(5, 0)).unwrap().under_construction);
        assert!(!loaded.road_cell_at(IVec2::new(6, 1)).unwrap().under_construction);
        // A site is claimed ground, not a building yet.
        assert!(loaded.buildings().all(|(id, _)| id != site));
    }

    #[test]
    fn a_saved_site_is_still_recorded_even_though_buildings_skips_it() {
        let dir = temp_dir("site_survives_save");
        let mut city = City::default();
        let site = city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        city.mark_under_construction(site);

        save_city(&city, &dir).unwrap();
        let loaded = load_city(&dir).unwrap();
        assert_eq!(loaded.placements().count(), 1, "the site is still there");
        assert_eq!(loaded.buildings().count(), 0, "but it isn't a building yet");
    }

    #[test]
    fn a_file_without_under_construction_fields_loads_as_false() {
        let dir = temp_dir("no_under_construction_field");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            format!(
                r#"(
                version: {CURRENT_VERSION},
                next_id: 1,
                buildings: [
                    (id: 0, catalogue_id: "house01", definition_id: None, origin: (20, 64, 20), rotation: Deg0, footprint: (1, 1)),
                ],
                road_cells: [(x: 0, z: 0, y: 64, style: "dirt", ascent: None, variant: Surface)],
            )"#
            ),
        )
        .unwrap();

        let loaded = load_city(&dir).unwrap();
        assert!(!loaded.building(BuildingId::from_u64(0)).unwrap().under_construction);
        assert!(!loaded.road_cell_at(IVec2::new(0, 0)).unwrap().under_construction);
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

    /// Ticket 059: a version-2 file's `road_cells` was a bare `(i32, i32)`
    /// tuple — no `style` field. Refused the same way, not silently loaded
    /// with every cell guessing at a style it never actually had.
    #[test]
    fn an_old_styleless_road_cell_save_is_refused_not_silently_defaulted() {
        let dir = temp_dir("old_road_cell_shape");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            "(version: 2, next_id: 0, buildings: [], road_cells: [(5, 5)])",
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Parse(_) | PersistenceError::UnsupportedVersion(2)), "{err:?}");
    }

    /// Ticket 067: a version-4 file's `road_cells` have no `ascent`, so
    /// every road in one is flat — which, unlike the earlier bumps, *would*
    /// be a safe thing to default. Refused anyway: see the module docs'
    /// "No migration path yet" for why one safe default is still the wrong
    /// precedent to set here.
    #[test]
    fn an_old_ascentless_road_cell_save_is_refused_not_silently_defaulted() {
        let dir = temp_dir("old_road_cell_ascent");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            r#"(version: 4, next_id: 0, buildings: [], road_cells: [(x: 5, z: 5, y: 64, style: "dirt")])"#,
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Parse(_) | PersistenceError::UnsupportedVersion(4)), "{err:?}");
    }

    /// Ticket 071: a version-5 file's `road_cells` have no `variant`, so
    /// every road in one would default to `Surface` — and for a road already
    /// cut through a hill that default is actively wrong: the first re-tile
    /// would write the open-sky piece back into the hillside and fill the
    /// tunnel in. Refused, like every bump before it.
    #[test]
    fn an_old_variantless_road_cell_save_is_refused_not_silently_defaulted() {
        let dir = temp_dir("old_road_cell_variant");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            r#"(version: 5, next_id: 0, buildings: [], road_cells: [(x: 5, z: 5, y: 64, style: "dirt", ascent: None)])"#,
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Parse(_) | PersistenceError::UnsupportedVersion(5)), "{err:?}");
    }

    /// Ticket 065: a version-3 file's `road_cells` had no `y` — every road
    /// in it was written at the hardcoded Y=0 that ticket fixed, so there is
    /// no height to recover and nothing sensible to default to. Refused, the
    /// same call the 1 -> 2 and 2 -> 3 bumps made.
    #[test]
    fn an_old_heightless_road_cell_save_is_refused_not_silently_defaulted() {
        let dir = temp_dir("old_road_cell_height");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(
            dir.join("citybuilder/city.ron"),
            r#"(version: 3, next_id: 0, buildings: [], road_cells: [(x: 5, z: 5, style: "dirt")])"#,
        )
        .unwrap();

        // Which of the two it trips is incidental — RON deserializes the
        // whole file (and misses `y`) before `load_city` ever gets to look at
        // `version` — so this asserts the same either-way shape ticket 059's
        // own 2 -> 3 test does. What matters is that it's refused.
        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Parse(_) | PersistenceError::UnsupportedVersion(3)), "{err:?}");
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
        // `CURRENT_VERSION` rather than a pinned literal: this fixture is
        // about *occupancy* being rejected, not about versioning, and pinning
        // it made it start failing for the wrong reason on every schema bump.
        fs::write(
            dir.join("citybuilder/city.ron"),
            format!(
                r#"(
                version: {CURRENT_VERSION},
                next_id: 2,
                buildings: [
                    (id: 0, catalogue_id: "house01", origin: (0, 64, 0), rotation: Deg0, footprint: (4, 4)),
                    (id: 1, catalogue_id: "house01", origin: (3, 64, 3), rotation: Deg0, footprint: (4, 4)),
                ],
                road_cells: [],
            )"#
            ),
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Corrupt(PlacementError::TileOccupied { .. })));
    }

    #[test]
    fn a_saved_road_cell_colliding_with_a_building_is_corrupt() {
        let dir = temp_dir("road_overlap");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        // See the note in `two_overlapping_saved_buildings_is_corrupt_not_a_bad_city`.
        fs::write(
            dir.join("citybuilder/city.ron"),
            format!(
                r#"(
                version: {CURRENT_VERSION},
                next_id: 1,
                buildings: [
                    (id: 0, catalogue_id: "house01", origin: (0, 64, 0), rotation: Deg0, footprint: (2, 2)),
                ],
                road_cells: [(x: 0, z: 0, y: 64, style: "dirt", ascent: None, variant: Surface)],
            )"#
            ),
        )
        .unwrap();

        let err = load_city(&dir).unwrap_err();
        assert!(matches!(err, PersistenceError::Corrupt(PlacementError::TileOccupied { .. })));
    }
}
