//! Road type definitions (ticket 060, roadmap F1b/F3 follow-up): the game
//! data a road *style* carries beyond its shape. [`super::road_catalogue`]
//! (ticket 054, extended to multiple styles by 059) knows a style's six
//! piece blueprints; nothing yet knows how fast travel along it is, or how
//! much it can carry. That's what [`RoadType`] and [`load_road_types_dir`]
//! add — one RON file per style under `assets/city/road_types`, mirroring
//! [`super::definition`]'s split between a building's geometry
//! ([`crate::blueprint::BuildingCatalogue`]) and its game data
//! ([`super::definition::BuildingDefinitions`]).
//!
//! ## Schema only, inert
//!
//! Same call the roadmap's C3 makes for a building's production rate:
//! `travel_speed`/`capacity` are parsed, validated, and (eventually)
//! displayed — nothing in this game simulates traffic, throughput, or
//! travel time with them yet. There is no logistics/routing system
//! anywhere in this crate to hang a real simulation off; that's a project
//! of its own, not two fields. See `finished_tickets/060-road-type-schema.md`.
//!
//! ## The filename *is* the style id — no separate field to name it
//!
//! Unlike a building definition (whose `.ron` stem and `blueprint:` field
//! are free to differ — [`super::definition`]'s own docs explain why), a
//! road type's filename stem **is** the style id it describes:
//! `assets/city/road_types/dirt.ron` describes the `"dirt"` style, the same
//! name [`super::road_catalogue`]'s `assets/city/roads/dirt/` directory
//! uses. There's no indirection to unify here the way a building's
//! `.ron`-names-an-`.nbt` reference needs — [`super::road_catalogue::RoadCatalogue`]
//! is already keyed by style name, so the filename-is-the-id convention
//! [`super::definition`] itself prefers (no inline `id:` field) applies with
//! nothing left to disagree with.
//!
//! ## What "validated" means here
//!
//! - **The style is real.** A road type's id must be one of
//!   [`RoadCatalogue::styles`](super::road_catalogue::RoadCatalogue::styles)
//!   — a type file naming a style with no geometry can never be placed, the
//!   same call [`super::definition`] makes for an unknown blueprint
//!   reference. The reverse isn't an error: a style with geometry but no
//!   type file is just undescribed, same as a blueprint with no definition
//!   yet.
//! - **`travel_speed` and `capacity` are positive.** Not a schema violation
//!   serde would catch, but a zero-or-negative speed or a zero-capacity road
//!   isn't a value anything downstream should have to defend against.
//!
//! ## Failure is per-file, not per-directory
//!
//! Same contract as [`load_definitions_dir`](super::definition::load_definitions_dir):
//! never panics. A missing directory is an empty set of road types (logged,
//! not fatal); a malformed or invalid file is skipped and reported alongside
//! whatever else loaded.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::prelude::Resource;
use serde::Deserialize;

use super::road_catalogue::RoadCatalogue;

/// One road style's game data, deserialized directly from a `.ron` file —
/// see the module docs for why there's no `id`/`style` field here.
#[derive(Debug, Clone, Deserialize)]
pub struct RoadType {
    pub name: String,
    /// Inert (ticket 060) — no traffic/routing system reads this yet. A
    /// relative multiplier, not tied to a unit: 1.0 is the implicit
    /// baseline a future simulation would define.
    pub travel_speed: f32,
    /// Inert (ticket 060) — how much a road of this style could carry, once
    /// something exists to carry.
    pub capacity: u32,
}

/// Why a `.ron` file didn't become a [`LoadedRoadType`].
#[derive(Debug)]
pub enum RoadDefinitionError {
    /// Couldn't be opened or read.
    Read(std::io::Error),
    /// Not well-formed RON, or didn't match [`RoadType`]'s shape.
    Parse(String),
    /// The filename stem isn't a style [`RoadCatalogue`] has any geometry
    /// for.
    UnknownStyle(String),
    /// `travel_speed` isn't `> 0.0`.
    InvalidTravelSpeed(f32),
    /// `capacity` is `0`.
    InvalidCapacity,
    /// The filename has nothing usable before its extension.
    NoFilenameStem,
    /// Another file in the same directory already claimed this id — can't
    /// happen from a single non-recursive directory scan on a
    /// case-insensitive filesystem, but [`build_road_types`] is exercised
    /// directly by a test that constructs it anyway, same as
    /// [`super::definition`]'s own `DuplicateId`.
    DuplicateId { id: String, other: PathBuf },
}

impl std::fmt::Display for RoadDefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoadDefinitionError::Read(err) => write!(f, "{err}"),
            RoadDefinitionError::Parse(msg) => write!(f, "{msg}"),
            RoadDefinitionError::UnknownStyle(style) => {
                write!(f, "style {style:?} has no geometry in the road catalogue")
            }
            RoadDefinitionError::InvalidTravelSpeed(speed) => {
                write!(f, "travel_speed must be > 0.0, got {speed}")
            }
            RoadDefinitionError::InvalidCapacity => write!(f, "capacity must be > 0"),
            RoadDefinitionError::NoFilenameStem => write!(f, "filename has no usable stem"),
            RoadDefinitionError::DuplicateId { id, other } => {
                write!(f, "id {id:?} already claimed by {}", other.display())
            }
        }
    }
}

impl std::error::Error for RoadDefinitionError {}

/// One loaded road type: its style id (the filename stem), where it came
/// from, and the parsed [`RoadType`].
pub struct LoadedRoadType {
    /// The filename stem, e.g. `dirt.ron` -> `"dirt"` — and, per the module
    /// docs, exactly the style id [`RoadCatalogue`] files this style's
    /// geometry under.
    pub id: String,
    pub path: PathBuf,
    pub road_type: RoadType,
}

/// Every road type the game currently knows about, keyed by style id.
#[derive(Resource, Default)]
pub struct RoadTypes {
    entries: HashMap<String, LoadedRoadType>,
}

impl RoadTypes {
    /// The same test-only constructor [`super::definition::BuildingDefinitions::from_entries`]
    /// has, and for the same reason.
    #[cfg(test)]
    pub fn from_entries(entries: Vec<LoadedRoadType>) -> Self {
        RoadTypes { entries: entries.into_iter().map(|entry| (entry.id.clone(), entry)).collect() }
    }

    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn get(&self, id: &str) -> Option<&LoadedRoadType> {
        self.entries.get(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)] // no caller yet — see the module docs
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &LoadedRoadType> {
        self.entries.values()
    }
}

/// The range checks described in the module docs, tested directly against a
/// hand-built [`RoadType`] rather than only through a file on disk — the
/// same split [`super::definition::validate`] uses.
fn validate(road_type: &RoadType) -> Result<(), RoadDefinitionError> {
    if road_type.travel_speed <= 0.0 {
        return Err(RoadDefinitionError::InvalidTravelSpeed(road_type.travel_speed));
    }
    if road_type.capacity == 0 {
        return Err(RoadDefinitionError::InvalidCapacity);
    }
    Ok(())
}

/// Reads and validates one `.ron` file into a [`LoadedRoadType`]. `id` comes
/// from the caller (the filename stem) for the same reason
/// `city::definition::load_entry` takes it as a parameter: so the duplicate
/// check in [`build_road_types`] and this function agree on exactly the
/// same string.
fn load_entry(path: &Path, id: String, catalogue: &RoadCatalogue) -> Result<LoadedRoadType, RoadDefinitionError> {
    let text = fs::read_to_string(path).map_err(RoadDefinitionError::Read)?;
    let road_type: RoadType = ron::de::from_str(&text).map_err(|err| RoadDefinitionError::Parse(err.to_string()))?;
    validate(&road_type)?;

    if !catalogue.styles().contains(&id.as_str()) {
        return Err(RoadDefinitionError::UnknownStyle(id));
    }

    Ok(LoadedRoadType { id, path: path.to_path_buf(), road_type })
}

/// Whether `path` has a `.ron` extension, case-insensitively — same
/// reasoning as `blueprint::catalogue::is_nbt_file`.
fn is_ron_file(path: &Path) -> bool {
    path.is_file() && path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("ron"))
}

/// Scans `dir` (non-recursive) for `*.ron` files, reads and validates each
/// one against `catalogue`, and returns what loaded plus what didn't. A
/// missing directory is an empty set of road types, not an error — same
/// contract as [`load_definitions_dir`](super::definition::load_definitions_dir).
pub fn load_road_types_dir(dir: &Path, catalogue: &RoadCatalogue) -> (RoadTypes, Vec<(PathBuf, RoadDefinitionError)>) {
    let paths: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(read_dir) => read_dir.filter_map(|entry| entry.ok()).map(|entry| entry.path()).filter(|path| is_ron_file(path)).collect(),
        Err(_) => Vec::new(),
    };
    build_road_types(paths, catalogue)
}

/// The load loop itself, over an explicit path list — split from
/// [`load_road_types_dir`] for the same testing reason
/// `city::definition::build_definitions` is split from its directory scan.
fn build_road_types(mut paths: Vec<PathBuf>, catalogue: &RoadCatalogue) -> (RoadTypes, Vec<(PathBuf, RoadDefinitionError)>) {
    paths.sort();

    let mut entries: HashMap<String, LoadedRoadType> = HashMap::new();
    let mut skipped = Vec::new();
    for path in paths {
        let Some(id) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            skipped.push((path, RoadDefinitionError::NoFilenameStem));
            continue;
        };
        if let Some(existing) = entries.get(&id) {
            skipped.push((path, RoadDefinitionError::DuplicateId { id, other: existing.path.clone() }));
            continue;
        }
        match load_entry(&path, id.clone(), catalogue) {
            Ok(entry) => {
                entries.insert(id, entry);
            }
            Err(err) => skipped.push((path, err)),
        }
    }

    (RoadTypes { entries }, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::{write_structure_file, BlockState, Blueprint};
    use bevy::math::IVec3;
    use super::super::road::{RoadPieceKind, RoadPieceVariant};
    use super::super::road_catalogue::{load_road_catalogue_dir, piece_path};
    use super::super::state::ROAD_CELL_SIZE;

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_road_definition_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// A one-block-tall stone piece, `ROAD_CELL_SIZE`-square — the same
    /// synthetic fixture shape `road_catalogue::tests::one_stone` builds.
    fn one_stone_piece() -> Blueprint {
        let size = IVec3::new(ROAD_CELL_SIZE, 1, ROAD_CELL_SIZE);
        let volume = (size.x * size.y * size.z) as usize;
        let blocks = vec![1u16; volume];
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() }],
            blocks,
            data_version: 4438,
            failed_columns: 0,
        }
    }

    /// A `RoadCatalogue` with one real style (`"dirt"`, every kind loaded) —
    /// this module's own `catalogue_with_house01`-equivalent fixture.
    fn catalogue_with_dirt() -> RoadCatalogue {
        let dir = temp_dir("catalogue_fixture");
        fs::create_dir_all(dir.join("dirt")).expect("should create style dir");
        for kind in RoadPieceKind::ALL {
            write_structure_file(&piece_path(&dir, "dirt", kind, RoadPieceVariant::Surface), &one_stone_piece())
                .unwrap();
        }
        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        // Only the surface pieces are written here — this fixture exists to
        // give a *style* some geometry to be validated against, not to
        // exercise ticket 071's variants, so every `-tunnel` file is
        // legitimately missing.
        assert!(
            skipped.iter().all(|(_, _, variant, _)| *variant == RoadPieceVariant::Tunnel),
            "{skipped:?}"
        );
        catalogue
    }

    const VALID_RON: &str = r#"
RoadType(
    name: "Dirt Path",
    travel_speed: 1.0,
    capacity: 4,
)
"#;

    #[test]
    fn a_missing_directory_is_empty_road_types_not_an_error() {
        let catalogue = catalogue_with_dirt();
        let dir = std::env::temp_dir().join("block_viewer_test_road_definition_does_not_exist");
        let _ = fs::remove_dir_all(&dir);
        let (types, skipped) = load_road_types_dir(&dir, &catalogue);
        assert!(types.is_empty());
        assert!(skipped.is_empty());
    }

    #[test]
    fn a_valid_file_loads_under_its_filename_stem() {
        let catalogue = catalogue_with_dirt();
        let dir = temp_dir("valid");
        fs::write(dir.join("dirt.ron"), VALID_RON).unwrap();

        let (types, skipped) = load_road_types_dir(&dir, &catalogue);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(types.len(), 1);
        let entry = types.get("dirt").expect("id is the filename stem");
        assert_eq!(entry.road_type.name, "Dirt Path");
        assert_eq!(entry.road_type.travel_speed, 1.0);
        assert_eq!(entry.road_type.capacity, 4);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_style_with_no_geometry_is_skipped() {
        let catalogue = catalogue_with_dirt(); // only "dirt" has geometry
        let dir = temp_dir("unknown_style");
        fs::write(dir.join("paved.ron"), VALID_RON).unwrap();

        let (types, skipped) = load_road_types_dir(&dir, &catalogue);
        assert!(types.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(&skipped[0].1, RoadDefinitionError::UnknownStyle(id) if id == "paved"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_zero_or_negative_travel_speed_is_skipped() {
        let catalogue = catalogue_with_dirt();
        let dir = temp_dir("bad_speed");
        fs::write(
            dir.join("dirt.ron"),
            r#"RoadType(name: "Dirt Path", travel_speed: 0.0, capacity: 4)"#,
        )
        .unwrap();

        let (types, skipped) = load_road_types_dir(&dir, &catalogue);
        assert!(types.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, RoadDefinitionError::InvalidTravelSpeed(_)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_zero_capacity_is_skipped() {
        let catalogue = catalogue_with_dirt();
        let dir = temp_dir("bad_capacity");
        fs::write(
            dir.join("dirt.ron"),
            r#"RoadType(name: "Dirt Path", travel_speed: 1.0, capacity: 0)"#,
        )
        .unwrap();

        let (types, skipped) = load_road_types_dir(&dir, &catalogue);
        assert!(types.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, RoadDefinitionError::InvalidCapacity));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn garbage_ron_is_skipped_as_a_parse_error_not_a_panic() {
        let catalogue = catalogue_with_dirt();
        let dir = temp_dir("garbage");
        fs::write(dir.join("dirt.ron"), b"not valid ron at all {{{").unwrap();

        let (types, skipped) = load_road_types_dir(&dir, &catalogue);
        assert!(types.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, RoadDefinitionError::Parse(_)));

        fs::remove_dir_all(&dir).ok();
    }

    /// Two distinct on-disk files sharing a stem — same reasoning as
    /// `city::definition`'s own duplicate-id test: exercises
    /// [`build_road_types`] directly, since a single non-recursive directory
    /// scan can never hand it two same-stem candidates on a
    /// case-insensitive filesystem.
    #[test]
    fn a_duplicate_id_is_skipped_rather_than_silently_overwriting() {
        let catalogue = catalogue_with_dirt();
        let dir = temp_dir("duplicate");
        fs::create_dir_all(dir.join("a")).unwrap();
        fs::create_dir_all(dir.join("b")).unwrap();
        fs::write(dir.join("a/dirt.ron"), VALID_RON).unwrap();
        fs::write(dir.join("b/dirt.ron"), VALID_RON).unwrap();

        let (types, skipped) = build_road_types(vec![dir.join("a/dirt.ron"), dir.join("b/dirt.ron")], &catalogue);
        assert_eq!(types.len(), 1, "one id, one winner");
        assert_eq!(skipped.len(), 1);
        assert!(matches!(&skipped[0].1, RoadDefinitionError::DuplicateId { id, .. } if id == "dirt"));

        fs::remove_dir_all(&dir).ok();
    }
}
