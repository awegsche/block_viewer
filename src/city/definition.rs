//! Building definitions (ticket 040, roadmap C1): the game data a building
//! carries beyond its shape. [`super`]'s catalogue (ticket 039) knows a
//! blueprint's size, palette and footprint; nothing yet knows a building's
//! tier, what unlocks it, what it costs, what it produces, or how much
//! damage it tolerates. That's what [`Building`] and [`load_definitions_dir`]
//! add — one RON file per building under `assets/city/buildings`.
//!
//! ## Why RON, and why data rather than a scripting language
//!
//! The roadmap's C1 section makes this call explicitly: iteration 1's
//! values are numbers and references — production rate, inputs, outputs,
//! tier, unlock dependencies, blueprint file, cost — which is a schema, not
//! a program. RON (serde, comments, real enums, no whitespace significance)
//! beats TOML for the nested/tagged shape here (`footprint`'s two variants,
//! `production`'s optionality) and beats JSON for a file a person edits by
//! hand. Reach for a scripting language only once a *behaviour* needs to
//! vary per building rather than a number.
//!
//! ## The `id` deviation from the roadmap's sketch
//!
//! The roadmap's illustrative RON has an inline `id: "lumberjack"` field.
//! This module drops it and derives the id from the filename stem instead —
//! the same call [`super::blueprint::load_catalogue_dir`] made for
//! blueprints, and for the same reason: a `name:` field and a filename can
//! say different things, but a filename *is* its own id, so there's nothing
//! for the two to disagree about.
//!
//! ## What "validated" means here
//!
//! Serde's `Deserialize` already guarantees a well-typed [`Building`] — the
//! right fields, the right shapes. What it can't know:
//!
//! - **The blueprint reference is real.** `blueprint: "lumberjack.nbt"`
//!   names a file; [`load_entry`] strips the extension and looks the stem up
//!   in the [`BuildingCatalogue`](super::blueprint::BuildingCatalogue)
//!   passed in, the same catalogue 039 already validated. A definition
//!   pointing at a blueprint that doesn't exist (missing file, failed
//!   validation, typo) is an error here rather than a build-menu entry
//!   nobody can ever place.
//! - **`integrity` describes a real ramp.** Both thresholds must be in
//!   `0.0..=1.0`, and `pristine_above` must exceed `ruined_below` — anything
//!   else makes roadmap I4's later linear health-to-output ramp inverted or
//!   degenerate before it's even built.
//! - **`cost` and `production` numbers are sane.** A zero-or-negative cost
//!   count, or a negative production rate, isn't a schema violation serde
//!   would catch, but it's not a value anything downstream should have to
//!   defend against either.
//! - **`footprint: Explicit { x, z }` is a real footprint.** Both axes
//!   `> 0`. `FromBlueprint` can't fail this check — it resolves to the
//!   catalogue entry's own footprint, already validated by 039.
//!
//! `requires` is parsed and carried on [`Building`] but deliberately **not**
//! checked against other definitions' ids here — dangling-reference and
//! cycle detection across the whole tree is roadmap C2's job, which needs
//! every definition loaded first to check against.
//!
//! ## Failure is per-file, not per-directory
//!
//! Same contract as [`load_catalogue_dir`](super::blueprint::load_catalogue_dir):
//! [`load_definitions_dir`] never panics. A missing directory is an empty
//! set of definitions (logged, not fatal); a malformed or invalid file is
//! skipped and reported alongside whatever else loaded.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::IVec2;
use bevy::prelude::Resource;
use serde::Deserialize;

use crate::blueprint::BuildingCatalogue;

/// One building's game data, deserialized directly from a `.ron` file.
/// Field-for-field the roadmap's C1 sketch, minus the inline `id` — see the
/// module docs.
#[derive(Debug, Clone, Deserialize)]
pub struct Building {
    pub name: String,
    /// A blueprint filename, e.g. `"house01.nbt"` — looked up by its stem in
    /// the [`BuildingCatalogue`] at load time.
    pub blueprint: String,
    pub tier: u32,
    /// Other building ids this one unlocks after. Parsed, not validated —
    /// see the module docs. No non-test reader yet — C2's tech tree is what
    /// will walk this.
    #[serde(default)]
    #[allow(dead_code)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub footprint: FootprintSpec,
    /// `None` for iteration 1's non-functional buildings.
    #[serde(default)]
    pub production: Option<Production>,
    #[serde(default)]
    pub cost: Vec<Cost>,
    pub integrity: Integrity,
}

/// How a building's horizontal footprint is determined. `FromBlueprint` (the
/// default) reads it off the matched [`CatalogueEntry`](super::blueprint::CatalogueEntry);
/// `Explicit` overrides it — for a building whose placeable footprint should
/// be larger than its literal geometry (a yard around a small structure),
/// which the roadmap's sketch leaves room for without naming a use yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub enum FootprintSpec {
    #[default]
    FromBlueprint,
    Explicit {
        x: i32,
        z: i32,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Production {
    #[serde(default)]
    pub outputs: Vec<ProductionItem>,
    #[serde(default)]
    pub inputs: Vec<ProductionItem>,
    /// No non-test reader yet — C3's inert production display is what will
    /// use this.
    #[serde(default)]
    #[allow(dead_code)]
    pub radius: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProductionItem {
    pub item: String,
    pub per_minute: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Cost {
    pub block: String,
    pub count: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Integrity {
    pub pristine_above: f32,
    pub ruined_below: f32,
}

/// Why a `.ron` file didn't become a [`LoadedBuilding`].
#[derive(Debug)]
pub enum DefinitionError {
    /// Couldn't be opened or read.
    Read(std::io::Error),
    /// Not well-formed RON, or didn't match [`Building`]'s shape. Carries
    /// RON's own message rather than the error type itself — this is the
    /// only call site that needs it, so there's nothing a wrapped type would
    /// buy over its `Display` output.
    Parse(String),
    /// `blueprint`'s filename stem isn't in the [`BuildingCatalogue`] passed
    /// to [`load_definitions_dir`].
    UnknownBlueprint(String),
    /// `integrity`'s thresholds aren't both in `0.0..=1.0`, or
    /// `pristine_above` doesn't exceed `ruined_below`.
    InvalidIntegrity { pristine_above: f32, ruined_below: f32 },
    /// A `cost` entry's `count` is zero.
    InvalidCost { block: String, count: u32 },
    /// A `production` entry's `per_minute` is negative.
    InvalidProduction { item: String, per_minute: f32 },
    /// `footprint: Explicit { x, z }` has a non-positive axis.
    InvalidFootprint { x: i32, z: i32 },
    /// The filename has nothing usable before its extension.
    NoFilenameStem,
    /// Another file in the same directory already claimed this id.
    DuplicateId { id: String, other: PathBuf },
}

impl std::fmt::Display for DefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefinitionError::Read(err) => write!(f, "{err}"),
            DefinitionError::Parse(msg) => write!(f, "{msg}"),
            DefinitionError::UnknownBlueprint(blueprint) => {
                write!(f, "blueprint {blueprint:?} is not in the building catalogue")
            }
            DefinitionError::InvalidIntegrity { pristine_above, ruined_below } => write!(
                f,
                "integrity thresholds must be in 0.0..=1.0 with pristine_above > \
                 ruined_below, got pristine_above={pristine_above}, ruined_below={ruined_below}"
            ),
            DefinitionError::InvalidCost { block, count } => {
                write!(f, "cost entry for {block:?} has non-positive count {count}")
            }
            DefinitionError::InvalidProduction { item, per_minute } => write!(
                f,
                "production entry for {item:?} has negative per_minute {per_minute}"
            ),
            DefinitionError::InvalidFootprint { x, z } => {
                write!(f, "explicit footprint {x}x{z} must have both axes > 0")
            }
            DefinitionError::NoFilenameStem => write!(f, "filename has no usable stem"),
            DefinitionError::DuplicateId { id, other } => {
                write!(f, "id {id:?} already claimed by {}", other.display())
            }
        }
    }
}

impl std::error::Error for DefinitionError {}

/// One loaded building: an id, where it came from, the parsed [`Building`],
/// and its footprint resolved against the catalogue.
pub struct LoadedBuilding {
    /// The filename stem, e.g. `lumberjack.ron` -> `"lumberjack"`.
    pub id: String,
    pub path: PathBuf,
    pub building: Building,
    /// `(x, z)` — resolved from [`Building::footprint`] against the
    /// matched catalogue entry when it's [`FootprintSpec::FromBlueprint`].
    pub footprint: IVec2,
}

/// Every building definition the game currently knows about, keyed by id.
#[derive(Resource, Default)]
pub struct BuildingDefinitions {
    entries: HashMap<String, LoadedBuilding>,
}

impl BuildingDefinitions {
    // No non-test caller yet for `get`/`is_empty` — G1's build menu is what
    // will do "the building named X" lookups, the same "no caller yet" state
    // `BuildingCatalogue::get` was in before this ticket. Kept for API
    // symmetry with `BuildingCatalogue` and exercised directly by this
    // module's tests.
    #[allow(dead_code)]
    pub fn get(&self, id: &str) -> Option<&LoadedBuilding> {
        self.entries.get(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &LoadedBuilding> {
        self.entries.values()
    }
}

/// The range/sign checks described in the module docs, tested directly
/// against a hand-built [`Building`] rather than only through a file on
/// disk — the same split [`super::blueprint::catalogue`]'s `validate` uses.
fn validate(building: &Building) -> Result<(), DefinitionError> {
    let integrity = building.integrity;
    let range = 0.0..=1.0;
    if !range.contains(&integrity.pristine_above)
        || !range.contains(&integrity.ruined_below)
        || integrity.pristine_above <= integrity.ruined_below
    {
        return Err(DefinitionError::InvalidIntegrity {
            pristine_above: integrity.pristine_above,
            ruined_below: integrity.ruined_below,
        });
    }
    for cost in &building.cost {
        if cost.count == 0 {
            return Err(DefinitionError::InvalidCost {
                block: cost.block.clone(),
                count: cost.count,
            });
        }
    }
    if let Some(production) = &building.production {
        for item in production.outputs.iter().chain(&production.inputs) {
            if item.per_minute < 0.0 {
                return Err(DefinitionError::InvalidProduction {
                    item: item.item.clone(),
                    per_minute: item.per_minute,
                });
            }
        }
    }
    if let FootprintSpec::Explicit { x, z } = building.footprint {
        if x <= 0 || z <= 0 {
            return Err(DefinitionError::InvalidFootprint { x, z });
        }
    }
    Ok(())
}

/// Resolves [`Building::footprint`] against the catalogue entry it names —
/// split from [`validate`] because it needs the catalogue lookup, which the
/// caller has already done to check [`DefinitionError::UnknownBlueprint`].
fn resolve_footprint(spec: FootprintSpec, catalogue_footprint: IVec2) -> IVec2 {
    match spec {
        FootprintSpec::FromBlueprint => catalogue_footprint,
        FootprintSpec::Explicit { x, z } => IVec2::new(x, z),
    }
}

/// Reads and validates one `.ron` file into a [`LoadedBuilding`]. `id` comes
/// from the caller (the filename stem) for the same reason
/// `blueprint::catalogue::load_entry` takes it as a parameter: so the
/// duplicate check in [`build_definitions`] and this function agree on
/// exactly the same string.
fn load_entry(
    path: &Path,
    id: String,
    catalogue: &BuildingCatalogue,
) -> Result<LoadedBuilding, DefinitionError> {
    let text = fs::read_to_string(path).map_err(DefinitionError::Read)?;
    let building: Building = ron::de::from_str(&text).map_err(|err| DefinitionError::Parse(err.to_string()))?;
    validate(&building)?;

    let blueprint_stem = Path::new(&building.blueprint)
        .file_stem()
        .and_then(|s| s.to_str());
    let catalogue_entry = blueprint_stem
        .and_then(|stem| catalogue.get(stem))
        .ok_or_else(|| DefinitionError::UnknownBlueprint(building.blueprint.clone()))?;

    let footprint = resolve_footprint(building.footprint, catalogue_entry.footprint);
    Ok(LoadedBuilding { id, path: path.to_path_buf(), building, footprint })
}

/// Whether `path` has a `.ron` extension, case-insensitively — same
/// reasoning as `blueprint::catalogue::is_nbt_file`.
fn is_ron_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("ron"))
}

/// Scans `dir` (non-recursive) for `*.ron` files, reads and validates each
/// one against `catalogue`, and returns what loaded plus what didn't. A
/// missing directory is an empty set of definitions, not an error — same
/// contract as [`load_catalogue_dir`](super::blueprint::load_catalogue_dir).
pub fn load_definitions_dir(
    dir: &Path,
    catalogue: &BuildingCatalogue,
) -> (BuildingDefinitions, Vec<(PathBuf, DefinitionError)>) {
    let paths: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(read_dir) => read_dir
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| is_ron_file(path))
            .collect(),
        Err(_) => Vec::new(),
    };
    build_definitions(paths, catalogue)
}

/// The load loop itself, over an explicit path list — split from
/// [`load_definitions_dir`] for the same testing reason
/// `blueprint::catalogue::build_catalogue` is split from its directory scan.
fn build_definitions(
    mut paths: Vec<PathBuf>,
    catalogue: &BuildingCatalogue,
) -> (BuildingDefinitions, Vec<(PathBuf, DefinitionError)>) {
    paths.sort();

    let mut entries: HashMap<String, LoadedBuilding> = HashMap::new();
    let mut skipped = Vec::new();
    for path in paths {
        let Some(id) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            skipped.push((path, DefinitionError::NoFilenameStem));
            continue;
        };
        if let Some(existing) = entries.get(&id) {
            skipped.push((
                path,
                DefinitionError::DuplicateId { id, other: existing.path.clone() },
            ));
            continue;
        }
        match load_entry(&path, id.clone(), catalogue) {
            Ok(entry) => {
                entries.insert(id, entry);
            }
            Err(err) => skipped.push((path, err)),
        }
    }

    (BuildingDefinitions { entries }, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::IVec3;

    use crate::blueprint::BlockState;
    use crate::blueprint::Blueprint;

    /// A fresh empty directory under the OS temp dir, unique per call — a
    /// pid alone (what `blueprint::catalogue`'s tests use) collides here
    /// because [`catalogue_with_house01`] gives every test in this module
    /// the same `name`, and `cargo test` runs them concurrently in one
    /// process; an atomic counter makes each call's directory distinct.
    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_definition_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// A one-entry `BuildingCatalogue` naming `house01` with a 3x5 footprint
    /// (Y=4, unused by a footprint) — built directly rather than through a
    /// real `.nbt` file, since this module never reads blueprints itself.
    fn catalogue_with_house01() -> BuildingCatalogue {
        let dir = temp_dir("catalogue_fixture");
        let blueprint = Blueprint {
            size: IVec3::new(3, 4, 5),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), BlockState { name: "minecraft:stone".into(), properties: Default::default() }],
            blocks: {
                let mut blocks = vec![0u16; 3 * 4 * 5];
                blocks[0] = 1;
                blocks
            },
            data_version: 3953,
            failed_columns: 0,
        };
        let path = dir.join("house01.nbt");
        crate::blueprint::write_structure_file(&path, &blueprint).expect("write fixture file");
        let (catalogue, skipped) = crate::blueprint::load_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        catalogue
    }

    const VALID_RON: &str = r#"
Building(
    name: "House",
    blueprint: "house01.nbt",
    tier: 1,
    integrity: Integrity(pristine_above: 0.95, ruined_below: 0.6),
)
"#;

    #[test]
    fn a_missing_directory_is_empty_definitions_not_an_error() {
        let catalogue = catalogue_with_house01();
        let dir = std::env::temp_dir().join("block_viewer_test_definition_does_not_exist");
        let _ = fs::remove_dir_all(&dir);
        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert!(skipped.is_empty());
    }

    #[test]
    fn a_valid_file_loads_with_its_resolved_footprint() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("valid");
        fs::write(dir.join("house01.ron"), VALID_RON).unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(definitions.len(), 1);
        let entry = definitions.get("house01").expect("id is the filename stem");
        assert_eq!(entry.building.name, "House");
        assert_eq!(entry.footprint, IVec2::new(3, 5), "resolved FromBlueprint");
        assert!(entry.building.production.is_none());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_explicit_footprint_overrides_the_blueprint() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("explicit_footprint");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 1,
                footprint: Explicit(x: 10, z: 12),
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(skipped.is_empty(), "{skipped:?}");
        let entry = definitions.get("house01").unwrap();
        assert_eq!(entry.footprint, IVec2::new(10, 12));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_reference_to_an_unknown_blueprint_is_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("unknown_blueprint");
        fs::write(
            dir.join("ghost.ron"),
            r#"Building(
                name: "Ghost",
                blueprint: "does_not_exist.nbt",
                tier: 1,
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(&skipped[0].1, DefinitionError::UnknownBlueprint(b) if b == "does_not_exist.nbt"));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inverted_integrity_thresholds_are_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("inverted_integrity");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 1,
                integrity: Integrity(pristine_above: 0.4, ruined_below: 0.6),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, DefinitionError::InvalidIntegrity { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_out_of_range_integrity_threshold_is_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("out_of_range_integrity");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 1,
                integrity: Integrity(pristine_above: 1.5, ruined_below: 0.6),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, DefinitionError::InvalidIntegrity { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_zero_cost_count_is_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("zero_cost");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 1,
                cost: [(block: "minecraft:oak_planks", count: 0)],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, DefinitionError::InvalidCost { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_negative_production_rate_is_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("negative_production");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 1,
                production: Some(Production(outputs: [(item: "wood", per_minute: -1.0)])),
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, DefinitionError::InvalidProduction { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_non_positive_explicit_footprint_is_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("bad_footprint");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 1,
                footprint: Explicit(x: 0, z: 4),
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, DefinitionError::InvalidFootprint { .. }));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn garbage_ron_is_skipped_as_a_parse_error_not_a_panic() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("garbage");
        fs::write(dir.join("broken.ron"), b"not valid ron at all {{{").unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, DefinitionError::Parse(_)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn requires_is_carried_but_not_validated() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("requires");
        fs::write(
            dir.join("house01.ron"),
            r#"Building(
                name: "House",
                blueprint: "house01.nbt",
                tier: 2,
                requires: ["a_building_that_does_not_exist"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(skipped.is_empty(), "{skipped:?}");
        let entry = definitions.get("house01").unwrap();
        assert_eq!(entry.building.requires, vec!["a_building_that_does_not_exist".to_string()]);

        fs::remove_dir_all(&dir).ok();
    }

    /// Two distinct on-disk files sharing a stem — same reasoning as
    /// `blueprint::catalogue`'s duplicate-id test: exercises
    /// [`build_definitions`] directly, since a single non-recursive
    /// directory scan can never hand it two same-stem candidates on a
    /// case-insensitive filesystem.
    #[test]
    fn a_duplicate_id_is_skipped_rather_than_silently_overwriting() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("duplicate");
        fs::create_dir_all(dir.join("a")).unwrap();
        fs::create_dir_all(dir.join("b")).unwrap();
        fs::write(dir.join("a/house01.ron"), VALID_RON).unwrap();
        fs::write(dir.join("b/house01.ron"), VALID_RON).unwrap();

        let (definitions, skipped) = build_definitions(
            vec![dir.join("a/house01.ron"), dir.join("b/house01.ron")],
            &catalogue,
        );
        assert_eq!(definitions.len(), 1, "one id, one winner");
        assert_eq!(skipped.len(), 1);
        assert!(matches!(&skipped[0].1, DefinitionError::DuplicateId { id, .. } if id == "house01"));

        fs::remove_dir_all(&dir).ok();
    }

    /// The real fixture this ticket adds.
    #[test]
    fn the_real_house01_fixture_loads() {
        let blueprints = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/city/blueprints");
        let (catalogue, skipped) = crate::blueprint::load_catalogue_dir(&blueprints);
        assert!(skipped.is_empty(), "{skipped:?}");

        let buildings = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/city/buildings");
        let (definitions, skipped) = load_definitions_dir(&buildings, &catalogue);
        assert!(skipped.is_empty(), "{skipped:?}");
        let entry = definitions.get("house01").expect("house01.ron should be in the definitions");
        assert!(!entry.building.name.is_empty());
    }
}
