//! The building asset catalogue (ticket 039, roadmap B4): every
//! `assets/city/blueprints/*.nbt` file, read through [`super::structure`]'s
//! reader, validated, and indexed by id — the thing that finally calls
//! 036/037/038's no-caller-yet primitives.
//!
//! ## What "validated" means here
//!
//! [`read_structure`](super::read_structure) already guarantees a
//! structurally sound [`Blueprint`]: dense blocks, in-range palette indices.
//! What it can't know is whether the result is a *plausible building* —
//! that's this module's job, and it's deliberately narrow:
//!
//! - **Size.** No axis may be zero (a degenerate, unplaceable footprint),
//!   and none may exceed [`STRUCTURE_BLOCK_MAX_SIZE`] — every file in this
//!   directory is a structure file, so bounding an entry by what a
//!   structure block could actually have produced is the natural ceiling,
//!   not an invented one.
//! - **Palette.** A palette that's nothing but `minecraft:air` (index 0,
//!   always present — see `Accumulator::new`) means the file has no actual
//!   blocks in it.
//!
//! Anything else — production fields, tiers, cost — is C1's job once it
//! exists; a [`CatalogueEntry`] carries only what B4 promises: an id, the
//! [`Blueprint`] itself, and the footprint the city grid (E2) will place it
//! on.
//!
//! ## Failure is per-file, not per-directory
//!
//! [`load_catalogue_dir`] never panics and never fails outright. A missing
//! directory comes back as an empty catalogue (logged, not fatal — the same
//! "don't take the process down" call ticket 008 made for a missing saves
//! directory); a malformed or oversized file is skipped and reported
//! alongside whatever *did* load, rather than losing an entire directory of
//! good buildings to one bad export.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::{IVec2, IVec3};
use bevy::prelude::Resource;

use super::structure::{read_structure_file, StructureReadError, STRUCTURE_BLOCK_MAX_SIZE};
use super::Blueprint;

/// Why a `.nbt` file didn't become a [`CatalogueEntry`].
#[derive(Debug)]
pub enum CatalogueError {
    /// Couldn't be opened or wasn't a well-formed structure file at all —
    /// see [`StructureReadError`].
    Read(StructureReadError),
    /// A size axis is zero (unplaceable) or exceeds
    /// [`STRUCTURE_BLOCK_MAX_SIZE`] (bigger than anything a structure block
    /// could have produced).
    InvalidSize(IVec3),
    /// The palette has nothing in it but `minecraft:air` — structurally
    /// valid, but not a building.
    Empty,
    /// The filename has nothing usable before its extension (e.g. a bare
    /// `.nbt`), so there's no id to key the catalogue by.
    NoFilenameStem,
    /// Another file in the same directory already claimed this id (its
    /// filename stem) — detected rather than one silently overwriting the
    /// other depending on directory listing order.
    DuplicateId { id: String, other: PathBuf },
}

impl std::fmt::Display for CatalogueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogueError::Read(err) => write!(f, "{err}"),
            CatalogueError::InvalidSize(size) => write!(
                f,
                "size {}x{}x{} is out of range: every axis must be 1..={}",
                size.x, size.y, size.z, STRUCTURE_BLOCK_MAX_SIZE
            ),
            CatalogueError::Empty => write!(f, "palette has no blocks besides air"),
            CatalogueError::NoFilenameStem => write!(f, "filename has no usable stem"),
            CatalogueError::DuplicateId { id, other } => {
                write!(f, "id {id:?} already claimed by {}", other.display())
            }
        }
    }
}

impl std::error::Error for CatalogueError {}

/// One loaded building: an id, where it came from, the blueprint itself, and
/// the horizontal footprint the city grid (roadmap E2) places it on.
pub struct CatalogueEntry {
    /// The filename stem, e.g. `house01.nbt` -> `"house01"`. What C1's
    /// `blueprint: "house01.nbt"` field and G1's build menu will key on.
    pub id: String,
    pub path: PathBuf,
    pub blueprint: Blueprint,
    /// `(size.x, size.z)` — Y doesn't factor into a footprint; the grid a
    /// building is placed on is horizontal. Per C1's sketch,
    /// `footprint: FromBlueprint`.
    pub footprint: IVec2,
}

/// Every building the game currently knows about, keyed by id for G1's
/// eventual "the building named X" lookups.
#[derive(Resource)]
pub struct BuildingCatalogue {
    entries: HashMap<String, CatalogueEntry>,
}

impl BuildingCatalogue {
    pub fn get(&self, id: &str) -> Option<&CatalogueEntry> {
        self.entries.get(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &CatalogueEntry> {
        self.entries.values()
    }
}

/// `(size.x, size.z)` — split out because [`load_entry`] and any future
/// caller that already has a [`Blueprint`] in hand (rather than a path)
/// shouldn't have to re-derive it by hand.
fn footprint_of(size: IVec3) -> IVec2 {
    IVec2::new(size.x, size.z)
}

/// The size/palette checks described in the module docs. Split from
/// [`load_entry`] so the geometry and palette rules are each one `if`,
/// tested directly against a hand-built [`Blueprint`] rather than only
/// through a real file on disk.
fn validate(blueprint: &Blueprint) -> Result<(), CatalogueError> {
    let size = blueprint.size;
    if size.min_element() <= 0 || size.max_element() > STRUCTURE_BLOCK_MAX_SIZE {
        return Err(CatalogueError::InvalidSize(size));
    }
    // Index 0 is always `minecraft:air` (`Accumulator::new`); anything past
    // it is a real block. A single-entry palette means there's nothing else
    // in it.
    if blueprint.palette.len() <= 1 {
        return Err(CatalogueError::Empty);
    }
    Ok(())
}

/// Reads and validates one `.nbt` file into a [`CatalogueEntry`]. `id` is
/// taken from the caller rather than re-derived from `path` here, so
/// [`load_catalogue_dir`]'s duplicate check and this function agree on
/// exactly the same string.
fn load_entry(path: &Path, id: String) -> Result<CatalogueEntry, CatalogueError> {
    let blueprint = read_structure_file(path).map_err(CatalogueError::Read)?;
    validate(&blueprint)?;
    let footprint = footprint_of(blueprint.size);
    Ok(CatalogueEntry { id, path: path.to_path_buf(), blueprint, footprint })
}

/// Whether `path` has a `.nbt` extension, case-insensitively — a directory
/// listing can hand back any case on a case-insensitive filesystem (which is
/// most of them on Windows), and a platform default shouldn't decide what
/// counts as a building file.
fn is_nbt_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("nbt"))
}

/// Scans `dir` (non-recursive) for `*.nbt` files, reads and validates each
/// one, and returns what loaded plus what didn't — a directory listing
/// order isn't guaranteed stable across platforms, so candidates are sorted
/// by path first, which is also what makes the duplicate-id winner
/// (whichever sorts first) deterministic.
///
/// A missing directory is not an error — see the module docs — it's an
/// empty catalogue plus nothing to skip either, since there was nothing to
/// even attempt reading.
pub fn load_catalogue_dir(dir: &Path) -> (BuildingCatalogue, Vec<(PathBuf, CatalogueError)>) {
    let paths: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(read_dir) => read_dir
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| is_nbt_file(path))
            .collect(),
        Err(_) => Vec::new(),
    };
    build_catalogue(paths)
}

/// The load loop itself, over an explicit path list rather than a directory
/// — split from [`load_catalogue_dir`] so the duplicate-id rule can be
/// tested against two arbitrary paths directly (on a case-insensitive
/// filesystem, two on-disk files can't actually collide by filename case
/// alone, which is the only way [`load_catalogue_dir`]'s own non-recursive
/// scan could otherwise produce two candidates with the same stem).
fn build_catalogue(mut paths: Vec<PathBuf>) -> (BuildingCatalogue, Vec<(PathBuf, CatalogueError)>) {
    paths.sort();

    let mut entries: HashMap<String, CatalogueEntry> = HashMap::new();
    let mut skipped = Vec::new();
    for path in paths {
        let Some(id) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            skipped.push((path, CatalogueError::NoFilenameStem));
            continue;
        };
        if let Some(existing) = entries.get(&id) {
            skipped.push((
                path,
                CatalogueError::DuplicateId { id, other: existing.path.clone() },
            ));
            continue;
        }
        match load_entry(&path, id.clone()) {
            Ok(entry) => {
                entries.insert(id, entry);
            }
            Err(err) => skipped.push((path, err)),
        }
    }

    (BuildingCatalogue { entries }, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::BlockState;

    /// A fresh empty directory under the OS temp dir, unique per test run
    /// (mirrors `blueprint::structure`'s own tempfile convention, since this
    /// repo has no `tempfile` dependency).
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_catalogue_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    fn state(name: &str, properties: &[(&str, &str)]) -> BlockState {
        BlockState {
            name: name.to_string(),
            properties: properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// Writes a structure file at `path` with the given size/palette/blocks
    /// — the same shape `structure.rs`'s own tests hand-build, reused here
    /// so catalogue tests don't need a real extraction to produce a fixture.
    fn write_structure_at(path: &Path, blueprint: &Blueprint) {
        let mut bytes = Vec::new();
        super::super::structure::write_structure(&mut bytes, blueprint)
            .expect("writing to a Vec should not fail");
        fs::write(path, bytes).expect("should write fixture file");
    }

    fn air_only(size: IVec3) -> Blueprint {
        let volume = (size.x * size.y * size.z) as usize;
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air()],
            blocks: vec![0; volume],
            data_version: 3953,
            failed_columns: 0,
        }
    }

    fn one_stone(size: IVec3) -> Blueprint {
        let volume = (size.x * size.y * size.z) as usize;
        let mut blocks = vec![0u16; volume];
        blocks[0] = 1;
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state("minecraft:stone", &[])],
            blocks,
            data_version: 3953,
            failed_columns: 0,
        }
    }

    #[test]
    fn a_missing_directory_is_an_empty_catalogue_not_an_error() {
        let dir = std::env::temp_dir().join("block_viewer_test_catalogue_does_not_exist");
        let _ = fs::remove_dir_all(&dir);
        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert!(skipped.is_empty());
    }

    #[test]
    fn a_directory_with_no_nbt_files_is_an_empty_catalogue() {
        let dir = temp_dir("empty");
        fs::write(dir.join("readme.txt"), b"not a structure file").unwrap();
        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert!(skipped.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_valid_file_loads_with_its_footprint() {
        let dir = temp_dir("valid");
        let blueprint = one_stone(IVec3::new(3, 4, 5));
        write_structure_at(&dir.join("house01.nbt"), &blueprint);

        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(catalogue.len(), 1);
        let entry = catalogue.get("house01").expect("id is the filename stem");
        assert_eq!(entry.footprint, IVec2::new(3, 5), "footprint is (x, z)");
        assert_eq!(entry.blueprint.size, blueprint.size);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extension_matching_is_case_insensitive() {
        let dir = temp_dir("case");
        write_structure_at(&dir.join("house01.NBT"), &one_stone(IVec3::ONE));
        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(catalogue.len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_zero_size_axis_is_skipped_as_invalid_size() {
        let dir = temp_dir("zero_size");
        // Zero-volume blueprints round-trip fine through the structure
        // format (an empty `blocks` list matches a zero volume) — the
        // catalogue is what rejects them, not the reader.
        write_structure_at(&dir.join("degenerate.nbt"), &air_only(IVec3::new(0, 4, 4)));

        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, CatalogueError::InvalidSize(_)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_too_large_blueprint_is_skipped_as_invalid_size() {
        let dir = temp_dir("too_large");
        let side = STRUCTURE_BLOCK_MAX_SIZE + 1;
        write_structure_at(
            &dir.join("giant.nbt"),
            &air_only(IVec3::splat(side)),
        );

        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, CatalogueError::InvalidSize(_)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_all_air_blueprint_is_skipped_as_empty() {
        let dir = temp_dir("all_air");
        write_structure_at(&dir.join("nothing.nbt"), &air_only(IVec3::splat(4)));

        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, CatalogueError::Empty));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn garbage_bytes_are_skipped_as_a_read_error_not_a_panic() {
        let dir = temp_dir("garbage");
        fs::write(dir.join("broken.nbt"), b"not gzipped, not nbt").unwrap();

        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(skipped[0].1, CatalogueError::Read(_)));

        fs::remove_dir_all(&dir).ok();
    }

    /// Two distinct on-disk files (different subdirectories, so no
    /// filesystem-level collision even on a case-insensitive one) that
    /// happen to share a stem — exercises [`build_catalogue`] directly
    /// rather than through [`load_catalogue_dir`]'s own non-recursive scan,
    /// which can never hand it two same-stem candidates from a single flat
    /// directory in the first place (see the function's docs).
    #[test]
    fn a_duplicate_id_is_skipped_rather_than_silently_overwriting() {
        let dir = temp_dir("duplicate");
        fs::create_dir_all(dir.join("a")).unwrap();
        fs::create_dir_all(dir.join("b")).unwrap();
        write_structure_at(&dir.join("a/house01.nbt"), &one_stone(IVec3::ONE));
        write_structure_at(&dir.join("b/house01.nbt"), &one_stone(IVec3::splat(2)));

        let (catalogue, skipped) =
            build_catalogue(vec![dir.join("a/house01.nbt"), dir.join("b/house01.nbt")]);
        assert_eq!(catalogue.len(), 1, "one id, one winner");
        assert_eq!(skipped.len(), 1);
        assert!(matches!(&skipped[0].1, CatalogueError::DuplicateId { id, .. } if id == "house01"));
        // Paths are sorted before loading, so `a/house01.nbt` (the size-1
        // blueprint) is the deterministic winner.
        let winner = catalogue.get("house01").unwrap();
        assert_eq!(winner.blueprint.size, IVec3::ONE);
        assert!(matches!(&skipped[0].1, CatalogueError::DuplicateId { other, .. } if other == &dir.join("a/house01.nbt")));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn multiple_files_all_load_independently() {
        let dir = temp_dir("multiple");
        write_structure_at(&dir.join("a.nbt"), &one_stone(IVec3::new(2, 2, 2)));
        write_structure_at(&dir.join("b.nbt"), &one_stone(IVec3::new(3, 3, 3)));
        write_structure_at(&dir.join("c.nbt"), &air_only(IVec3::new(2, 2, 2))); // skipped: empty

        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert_eq!(catalogue.len(), 2);
        assert_eq!(skipped.len(), 1);
        assert!(catalogue.get("a").is_some());
        assert!(catalogue.get("b").is_some());
        assert!(catalogue.get("c").is_none());

        fs::remove_dir_all(&dir).ok();
    }

    /// The real fixture this ticket adds: a structure-block export, not a
    /// synthetic one — proves the catalogue reads a genuine Minecraft file,
    /// not just what this module's own writer produces.
    #[test]
    fn the_real_house01_fixture_loads() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/city/blueprints");
        let (catalogue, skipped) = load_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        let entry = catalogue.get("house01").expect("house01.nbt should be in the catalogue");
        assert!(entry.blueprint.palette.len() > 1);
        assert_eq!(entry.footprint, IVec2::new(entry.blueprint.size.x, entry.blueprint.size.z));
    }
}
