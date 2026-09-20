//! The registry schema and loader (ticket 131, `MODEL_EXPORTER_ROADMAP.md`
//! "The registry"): `assets/models/world.ron` names the *models world* and
//! the rules every slot must obey; `assets/models/<name>.ron` is one model's
//! recorded box. Everything from ticket 132 on reads a [`Registry`] this
//! module built — no other module parses these files.
//!
//! Pure data: no CLI, no world I/O beyond `.ron` files on disk.
//!
//! ## Failure is whole-directory, not per-file
//!
//! Unlike [`crate::city::definition::load_definitions_dir`] (which skips a
//! bad file and keeps going), [`load_registry`] returns the first
//! [`RegistryError`] it hits and nothing loads. A wrong `origin` here would
//! silently export the wrong blocks rather than just leave a building menu
//! entry missing, so a bad registry refuses every command rather than
//! working around the bad file.
//!
//! ## `name` vs. duplicate names
//!
//! A file's `name` field is checked against its own filename stem
//! ([`RegistryError::NameMismatch`]) *after* checking whether that name was
//! already claimed by an earlier file ([`RegistryError::DuplicateName`]).
//! That order matters: a copy-pasted `.ron` whose `name` still points at the
//! original is a duplicate-name problem even though its stem also disagrees
//! with it, and the duplicate is the more useful diagnosis of the two.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fmt, fs};

use bevy::math::{IVec2, IVec3};
use serde::{Deserialize, Deserializer, Serialize};

use crate::blueprint::{BlockState, STRUCTURE_BLOCK_MAX_SIZE};
use crate::selection::SelectionBounds;

const WORLD_FILE_NAME: &str = "world.ron";

/// One corner of [`Area`] — named `x`/`z` rather than reusing [`IVec2`]'s
/// `x`/`y` so the RON file reads as world coordinates, not a 2D vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Point {
    pub x: i32,
    pub z: i32,
}

/// The inclusive XZ rectangle models may be allocated in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Area {
    pub min: Point,
    pub max: Point,
}

/// `assets/models/world.ron` — which save is the models world, its ground
/// level, the area models may be allocated in, spacing, and the marker
/// block. See `MODEL_EXPORTER_ROADMAP.md`'s "The registry" for the field
/// meanings; the shipped file carries the same explanation as comments.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelWorld {
    pub save: String,
    pub ground_y: i32,
    pub area: Area,
    pub gap: u32,
    pub grid: u32,
    #[serde(deserialize_with = "deserialize_block_state")]
    pub marker: BlockState,
    pub blueprints_dir: PathBuf,
}

/// Deserializes [`BlockState`] through its existing `name[key=value,...]`
/// [`FromStr`] parser (ticket 035) rather than a second block-state grammar
/// — `world.ron`'s `marker` is just a string on the wire.
fn deserialize_block_state<'de, D>(deserializer: D) -> Result<BlockState, D::Error>
where
    D: Deserializer<'de>,
{
    let text = String::deserialize(deserializer)?;
    BlockState::from_str(&text).map_err(serde::de::Error::custom)
}

/// `assets/models/<name>.ron` — one model's recorded box: where the
/// blueprint's own `(0, 0, 0)` sits in the models world, its extents, and
/// where `export` writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSlot {
    pub name: String,
    pub origin: IVec3,
    pub size: IVec3,
    #[serde(default)]
    pub out: Option<PathBuf>,
}

impl ModelSlot {
    /// The box's min corner — `origin`.
    pub fn min(&self) -> IVec3 {
        self.origin
    }

    /// The box's max corner, inclusive — `origin + size - 1`.
    pub fn max(&self) -> IVec3 {
        self.origin + self.size - IVec3::ONE
    }

    /// The inclusive box as a [`SelectionBounds`], so export/import/markers
    /// all share ticket 019's inclusive convention rather than each
    /// re-deriving it from `origin`/`size`.
    pub fn bounds(&self) -> SelectionBounds {
        SelectionBounds::from_corners(self.min(), self.min(), self.max())
    }

    /// The box's XZ rectangle — what the overlap rule and the ring markers
    /// are computed against.
    pub fn footprint(&self) -> Footprint {
        let min = self.min();
        let max = self.max();
        Footprint {
            min: IVec2::new(min.x, min.z),
            max: IVec2::new(max.x, max.z),
        }
    }

    /// Where `export` writes this model: `out` if set, otherwise
    /// `<blueprints_dir>/<name>.nbt`.
    pub fn out_path(&self, world: &ModelWorld) -> PathBuf {
        self.out
            .clone()
            .unwrap_or_else(|| world.blueprints_dir.join(format!("{}.nbt", self.name)))
    }
}

/// A slot's XZ rectangle, inclusive on both corners — `x` is Minecraft `x`,
/// `y` is Minecraft `z` (the same `IVec2`-as-`(x, z)` convention
/// `city::definition::FootprintSpec` already uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    pub min: IVec2,
    pub max: IVec2,
}

impl Footprint {
    /// This rectangle grown by `by` blocks on every side.
    pub fn expanded(&self, by: i32) -> Footprint {
        Footprint {
            min: self.min - IVec2::splat(by),
            max: self.max + IVec2::splat(by),
        }
    }

    /// Whether this rectangle and `other` share any block.
    pub fn intersects(&self, other: &Footprint) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
    }

    /// Whether `other` fits entirely inside this rectangle.
    pub fn contains(&self, other: &Footprint) -> bool {
        self.min.x <= other.min.x && other.max.x <= self.max.x && self.min.y <= other.min.y && other.max.y <= self.max.y
    }
}

/// A loaded, validated `assets/models` directory: the world rules plus every
/// slot, in filename order.
#[derive(Debug, Clone)]
pub struct Registry {
    pub dir: PathBuf,
    pub world: ModelWorld,
    pub slots: Vec<ModelSlot>,
}

/// Why an `assets/models` directory didn't become a [`Registry`].
#[derive(Debug)]
pub enum RegistryError {
    /// No `world.ron` in the directory (or it couldn't be read).
    NoWorld,
    /// A file wasn't well-formed RON, or didn't match its expected shape.
    Parse { file: PathBuf, error: String },
    /// A slot file's `name` field doesn't match its own filename stem.
    NameMismatch { file: PathBuf, name: String },
    /// Another file already claimed this `name` — checked before
    /// [`Self::NameMismatch`], see the module docs.
    DuplicateName { name: String, first: PathBuf, second: PathBuf },
    /// A slot's `size` has an axis `< 1` or `> STRUCTURE_BLOCK_MAX_SIZE`.
    SizeOutOfRange { name: String, size: IVec3 },
    /// A slot's footprint isn't entirely inside `world.area`.
    OutsideArea { name: String },
    /// Two slots' footprints, each expanded by `world.gap`, intersect.
    TooClose { a: String, b: String },
    /// `world.ron`'s `gap`/`grid` is `0`, or its `area.min` is past
    /// `area.max` on some axis.
    BadWorld(&'static str),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::NoWorld => write!(f, "{WORLD_FILE_NAME} is missing or unreadable"),
            RegistryError::Parse { file, error } => write!(f, "{}: {error}", file.display()),
            RegistryError::NameMismatch { file, name } => write!(
                f,
                "{}: name {name:?} does not match its filename",
                file.display()
            ),
            RegistryError::DuplicateName { name, first, second } => write!(
                f,
                "{name:?} is claimed by both {} and {}",
                first.display(),
                second.display()
            ),
            RegistryError::SizeOutOfRange { name, size } => write!(
                f,
                "{name}: size {}x{}x{} must have every axis in 1..={STRUCTURE_BLOCK_MAX_SIZE}",
                size.x, size.y, size.z
            ),
            RegistryError::OutsideArea { name } => write!(f, "{name}: box is outside world.area"),
            RegistryError::TooClose { a, b } => write!(f, "{a} and {b} are closer than world.gap"),
            RegistryError::BadWorld(rule) => write!(f, "{WORLD_FILE_NAME}: {rule}"),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Whether `path` has a `.ron` extension, case-insensitively — same check
/// `city::definition::is_ron_file` makes.
fn is_ron_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("ron"))
}

fn validate_world(world: &ModelWorld) -> Result<(), RegistryError> {
    if world.gap == 0 {
        return Err(RegistryError::BadWorld("gap must be > 0"));
    }
    if world.grid == 0 {
        return Err(RegistryError::BadWorld("grid must be > 0"));
    }
    if world.area.min.x > world.area.max.x || world.area.min.z > world.area.max.z {
        return Err(RegistryError::BadWorld("area.min must be <= area.max on both axes"));
    }
    Ok(())
}

fn check_size(slot: &ModelSlot) -> Result<(), RegistryError> {
    let in_range = |value: i32| (1..=STRUCTURE_BLOCK_MAX_SIZE).contains(&value);
    if !in_range(slot.size.x) || !in_range(slot.size.y) || !in_range(slot.size.z) {
        return Err(RegistryError::SizeOutOfRange {
            name: slot.name.clone(),
            size: slot.size,
        });
    }
    Ok(())
}

fn check_area(slot: &ModelSlot, world: &ModelWorld) -> Result<(), RegistryError> {
    let area = Footprint {
        min: IVec2::new(world.area.min.x, world.area.min.z),
        max: IVec2::new(world.area.max.x, world.area.max.z),
    };
    if !area.contains(&slot.footprint()) {
        return Err(RegistryError::OutsideArea { name: slot.name.clone() });
    }
    Ok(())
}

/// The overlap rule `MODEL_EXPORTER_ROADMAP.md` spells out once for both
/// this ticket's validation and ticket 133's allocator: two slots conflict
/// iff one's footprint, expanded by `gap` on every side, intersects the
/// other's.
fn check_no_overlaps(slots: &[ModelSlot], gap: u32) -> Result<(), RegistryError> {
    for (i, a) in slots.iter().enumerate() {
        for b in &slots[i + 1..] {
            if a.footprint().expanded(gap as i32).intersects(&b.footprint()) {
                return Err(RegistryError::TooClose {
                    a: a.name.clone(),
                    b: b.name.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Reads and validates every `.ron` file in `dir`: `world.ron` first, then
/// every other `*.ron` file sorted by filename so output order is stable.
/// Returns the first [`RegistryError`] hit — see the module docs for why
/// this doesn't skip-and-continue the way [`crate::city::definition`]'s
/// loader does.
pub fn load_registry(dir: &Path) -> Result<Registry, RegistryError> {
    let world_path = dir.join(WORLD_FILE_NAME);
    let world_text = fs::read_to_string(&world_path).map_err(|_| RegistryError::NoWorld)?;
    let world: ModelWorld = ron::de::from_str(&world_text).map_err(|err| RegistryError::Parse {
        file: world_path.clone(),
        error: err.to_string(),
    })?;
    validate_world(&world)?;

    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map(|read_dir| {
            read_dir
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| is_ron_file(path) && path.file_name().and_then(|n| n.to_str()) != Some(WORLD_FILE_NAME))
                .collect()
        })
        .unwrap_or_default();
    paths.sort();

    let mut slots: Vec<ModelSlot> = Vec::new();
    let mut claimed: HashMap<String, PathBuf> = HashMap::new();
    for path in &paths {
        let text = fs::read_to_string(path).map_err(|err| RegistryError::Parse {
            file: path.clone(),
            error: err.to_string(),
        })?;
        let slot: ModelSlot = ron::de::from_str(&text).map_err(|err| RegistryError::Parse {
            file: path.clone(),
            error: err.to_string(),
        })?;

        if let Some(first) = claimed.get(&slot.name) {
            return Err(RegistryError::DuplicateName {
                name: slot.name.clone(),
                first: first.clone(),
                second: path.clone(),
            });
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        if stem != slot.name {
            return Err(RegistryError::NameMismatch {
                file: path.clone(),
                name: slot.name.clone(),
            });
        }
        check_size(&slot)?;
        check_area(&slot, &world)?;

        claimed.insert(slot.name.clone(), path.clone());
        slots.push(slot);
    }

    check_no_overlaps(&slots, world.gap)?;

    Ok(Registry { dir: dir.to_path_buf(), world, slots })
}

/// Writes `slot` to `<dir>/<name>.ron` with a header comment naming the
/// tool that wrote it. Ticket 133's `new` and 136's `import` call this;
/// keeping the writer next to the reader is what guarantees a written file
/// round-trips.
pub fn save_slot(dir: &Path, slot: &ModelSlot) -> io::Result<()> {
    let body = ron::ser::to_string_pretty(slot, ron::ser::PrettyConfig::default())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let contents = format!("// Written by model-exporter — see MODEL_EXPORTER_ROADMAP.md.\n{body}\n");
    fs::write(dir.join(format!("{}.ron", slot.name)), contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh empty directory under the OS temp dir, unique per call — the
    /// same reasoning as `city::definition::tests::temp_dir`: a pid alone
    /// collides when `cargo test` runs several of this module's tests
    /// concurrently.
    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_model_registry_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    const WORLD_RON: &str = r#"
ModelWorld(
    save: "models",
    ground_y: -61,
    area: (min: (x: 0, z: 0), max: (x: 511, z: 511)),
    gap: 3,
    grid: 8,
    marker: "minecraft:orange_terracotta",
    blueprints_dir: "assets/city/blueprints",
)
"#;

    fn write_world(dir: &Path, text: &str) {
        fs::write(dir.join("world.ron"), text).expect("write world.ron");
    }

    fn write_slot(dir: &Path, filename: &str, text: &str) {
        fs::write(dir.join(filename), text).expect("write slot file");
    }

    fn slot_ron(name: &str, origin: (i32, i32, i32), size: (i32, i32, i32)) -> String {
        format!(
            "ModelSlot(name: \"{name}\", origin: ({}, {}, {}), size: ({}, {}, {}), out: None)",
            origin.0, origin.1, origin.2, size.0, size.1, size.2
        )
    }

    #[test]
    fn a_valid_two_slot_registry_loads_in_filename_order() {
        let dir = temp_dir("valid_two_slot");
        write_world(&dir, WORLD_RON);
        // "barn" sorts after "aaa_house" by filename, and the two boxes
        // (each 5x5 at x=0 and x=40) are far apart, well clear of `gap`.
        write_slot(&dir, "aaa_house.ron", &slot_ron("aaa_house", (0, -61, 0), (5, 5, 5)));
        write_slot(&dir, "barn.ron", &slot_ron("barn", (40, -61, 0), (5, 5, 5)));

        let registry = load_registry(&dir).expect("should load");
        assert_eq!(registry.slots.len(), 2);
        assert_eq!(registry.slots[0].name, "aaa_house");
        assert_eq!(registry.slots[1].name, "barn");
    }

    #[test]
    fn a_missing_world_file_is_no_world() {
        let dir = temp_dir("no_world");
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::NoWorld));
    }

    #[test]
    fn a_malformed_world_file_is_a_parse_error() {
        let dir = temp_dir("bad_world_parse");
        write_world(&dir, "not valid ron at all(");
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::Parse { .. }));
    }

    #[test]
    fn a_malformed_slot_file_is_a_parse_error() {
        let dir = temp_dir("bad_slot_parse");
        write_world(&dir, WORLD_RON);
        write_slot(&dir, "broken.ron", "not valid ron at all(");
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::Parse { .. }));
    }

    #[test]
    fn a_slot_name_that_does_not_match_its_filename_is_a_name_mismatch() {
        let dir = temp_dir("name_mismatch");
        write_world(&dir, WORLD_RON);
        write_slot(&dir, "house.ron", &slot_ron("not_house", (0, -61, 0), (5, 5, 5)));
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::NameMismatch { .. }));
    }

    #[test]
    fn two_files_claiming_the_same_name_is_a_duplicate_name() {
        let dir = temp_dir("duplicate_name");
        write_world(&dir, WORLD_RON);
        // "a.ron" loads cleanly under "a". "b.ron" also declares name "a" —
        // its own stem ("b") disagrees too, but the duplicate is checked
        // first, so this reports `DuplicateName` rather than
        // `NameMismatch`. See the module docs.
        write_slot(&dir, "a.ron", &slot_ron("a", (0, -61, 0), (5, 5, 5)));
        write_slot(&dir, "b.ron", &slot_ron("a", (40, -61, 0), (5, 5, 5)));
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::DuplicateName { .. }));
    }

    #[test]
    fn a_size_axis_out_of_range_is_size_out_of_range() {
        let dir = temp_dir("size_out_of_range");
        write_world(&dir, WORLD_RON);
        write_slot(
            &dir,
            "huge.ron",
            &slot_ron("huge", (0, -61, 0), (STRUCTURE_BLOCK_MAX_SIZE + 1, 5, 5)),
        );
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::SizeOutOfRange { .. }));
    }

    #[test]
    fn a_box_outside_area_is_outside_area() {
        let dir = temp_dir("outside_area");
        write_world(&dir, WORLD_RON);
        write_slot(&dir, "far.ron", &slot_ron("far", (600, -61, 0), (5, 5, 5)));
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::OutsideArea { .. }));
    }

    #[test]
    fn two_boxes_closer_than_gap_is_too_close() {
        let dir = temp_dir("too_close");
        write_world(&dir, WORLD_RON);
        // Footprints 0..=4 and 5..=9 on x: touching, so expanded by gap=3
        // they overlap heavily.
        write_slot(&dir, "aaa.ron", &slot_ron("aaa", (0, -61, 0), (5, 5, 5)));
        write_slot(&dir, "bbb.ron", &slot_ron("bbb", (5, -61, 0), (5, 5, 5)));
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::TooClose { .. }));
    }

    #[test]
    fn a_zero_gap_is_a_bad_world() {
        let dir = temp_dir("bad_world_gap");
        write_world(
            &dir,
            r#"ModelWorld(save: "models", ground_y: -61, area: (min: (x: 0, z: 0), max: (x: 511, z: 511)), gap: 0, grid: 8, marker: "minecraft:orange_terracotta", blueprints_dir: "assets/city/blueprints")"#,
        );
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::BadWorld(_)));
    }

    #[test]
    fn an_inverted_area_is_a_bad_world() {
        let dir = temp_dir("bad_world_area");
        write_world(
            &dir,
            r#"ModelWorld(save: "models", ground_y: -61, area: (min: (x: 511, z: 0), max: (x: 0, z: 511)), gap: 3, grid: 8, marker: "minecraft:orange_terracotta", blueprints_dir: "assets/city/blueprints")"#,
        );
        let err = load_registry(&dir).expect_err("should fail");
        assert!(matches!(err, RegistryError::BadWorld(_)));
    }

    #[test]
    fn save_slot_then_load_registry_round_trips_exactly() {
        let dir = temp_dir("round_trip");
        write_world(&dir, WORLD_RON);

        let with_out = ModelSlot {
            name: "house02".to_string(),
            origin: IVec3::new(16, -62, 0),
            size: IVec3::new(12, 10, 12),
            out: Some(PathBuf::from("assets/city/blueprints/house02.nbt")),
        };
        let without_out = ModelSlot {
            name: "barn".to_string(),
            origin: IVec3::new(40, -61, 0),
            size: IVec3::new(14, 9, 11),
            out: None,
        };
        save_slot(&dir, &with_out).expect("save with_out");
        save_slot(&dir, &without_out).expect("save without_out");

        let registry = load_registry(&dir).expect("should load");
        assert_eq!(registry.slots.len(), 2);
        let loaded_with_out = registry.slots.iter().find(|s| s.name == "house02").expect("house02");
        let loaded_without_out = registry.slots.iter().find(|s| s.name == "barn").expect("barn");
        assert_eq!(*loaded_with_out, with_out);
        assert_eq!(*loaded_without_out, without_out);
    }

    #[test]
    fn the_shipped_models_directory_loads_with_no_slots() {
        let registry = load_registry(Path::new("assets/models")).expect("should load");
        assert!(registry.slots.is_empty());
    }

    #[test]
    fn min_max_and_footprint_match_the_inclusive_box_convention() {
        let slot = ModelSlot {
            name: "x".to_string(),
            origin: IVec3::new(16, -62, 0),
            size: IVec3::new(12, 10, 12),
            out: None,
        };
        assert_eq!(slot.min(), IVec3::new(16, -62, 0));
        assert_eq!(slot.max(), IVec3::new(27, -53, 11));
        let footprint = slot.footprint();
        assert_eq!(footprint.min, IVec2::new(16, 0));
        assert_eq!(footprint.max, IVec2::new(27, 11));
    }

    #[test]
    fn out_path_falls_back_to_blueprints_dir() {
        let world: ModelWorld = ron::de::from_str(WORLD_RON).expect("parse world");
        let slot = ModelSlot {
            name: "house02".to_string(),
            origin: IVec3::ZERO,
            size: IVec3::new(1, 1, 1),
            out: None,
        };
        assert_eq!(slot.out_path(&world), PathBuf::from("assets/city/blueprints/house02.nbt"));
    }
}
