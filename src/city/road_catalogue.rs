//! The road piece catalogue (ticket 054, roadmap F1/F3 groundwork): the
//! fixed set of `.nbt` blueprints [`super::road::RoadPieceKind`] selects
//! between, loaded from `assets/city/roads`.
//!
//! ## A fixed set, not a scan
//!
//! [`blueprint::load_catalogue_dir`](crate::blueprint::load_catalogue_dir)
//! (ticket 039) indexes *whatever* `.nbt` files it finds, keyed by filename.
//! A road piece isn't an open-ended catalogue — there are exactly six kinds
//! ([`RoadPieceKind::ALL`](super::road::RoadPieceKind::ALL)), each with one
//! canonical filename, and [`load_road_catalogue_dir`] looks for those six
//! by name rather than scanning the directory. A missing file is reported
//! per kind the same tolerant way 039 reports a bad file — the catalogue
//! that *does* load is still usable, just short a piece — rather than
//! failing the whole directory over one missing corner.
//!
//! ## What "validated" means here
//!
//! Narrower than 039's building validation: a road piece must be exactly
//! [`ROAD_CELL_SIZE`] blocks on `x` and `z` (its footprint is fixed, unlike
//! a building's), with no ceiling on `y` beyond
//! [`STRUCTURE_BLOCK_MAX_SIZE`], and a palette with more than just air —
//! the same "not actually empty" rule 039 already applies.
//!
//! ## No real assets yet
//!
//! Unlike 039's `house01.nbt`, this ticket ships no real `.nbt` road pieces
//! — they need an actual Minecraft structure-block export, not something to
//! fabricate. `load_road_catalogue_dir` against a missing or empty
//! `assets/city/roads` is exercised and comes back an empty catalogue (not
//! an error), and this module's own tests build synthetic fixtures the same
//! way `blueprint::catalogue`'s tests do. Authoring the six real pieces —
//! and wiring this loader into `city::run()` the way ticket 039 wired the
//! building catalogue — is later work, once there's something on disk for
//! it to find.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::math::IVec3;

use crate::blueprint::{read_structure_file, Blueprint, StructureReadError, STRUCTURE_BLOCK_MAX_SIZE};

use super::road::RoadPieceKind;
use super::state::ROAD_CELL_SIZE;

/// Why a road piece file didn't become a catalogue entry.
#[allow(dead_code)] // no non-test caller yet — see the module docs' "No real assets yet"
#[derive(Debug)]
pub enum RoadCatalogueError {
    /// No file exists at the expected path — see [`filename_for`].
    Missing,
    /// Couldn't be opened or wasn't a well-formed structure file — see
    /// [`StructureReadError`].
    Read(StructureReadError),
    /// `x` or `z` isn't exactly [`ROAD_CELL_SIZE`], or `y` is out of range.
    InvalidSize(IVec3),
    /// The palette has nothing in it but `minecraft:air`.
    Empty,
}

impl std::fmt::Display for RoadCatalogueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoadCatalogueError::Missing => write!(f, "no file found"),
            RoadCatalogueError::Read(err) => write!(f, "{err}"),
            RoadCatalogueError::InvalidSize(size) => write!(
                f,
                "size {}x{}x{} is invalid: x and z must be exactly {ROAD_CELL_SIZE}, y must be 1..={STRUCTURE_BLOCK_MAX_SIZE}",
                size.x, size.y, size.z
            ),
            RoadCatalogueError::Empty => write!(f, "palette has no blocks besides air"),
        }
    }
}

impl std::error::Error for RoadCatalogueError {}

/// The filename stem (without `.nbt`) each [`RoadPieceKind`] loads from —
/// the fixed names `assets/city/roads` is expected to contain one of each
/// of.
fn filename_for(kind: RoadPieceKind) -> &'static str {
    match kind {
        RoadPieceKind::Isolated => "isolated",
        RoadPieceKind::DeadEnd => "dead_end",
        RoadPieceKind::Straight => "straight",
        RoadPieceKind::Corner => "corner",
        RoadPieceKind::T => "t",
        RoadPieceKind::Cross => "cross",
    }
}

/// Every loaded road piece, keyed by kind — [`super::road::select_piece`]'s
/// own output is exactly this catalogue's key.
#[allow(dead_code)] // no non-test caller yet — see the module docs' "No real assets yet"
pub struct RoadCatalogue {
    pieces: HashMap<RoadPieceKind, Blueprint>,
}

#[allow(dead_code)] // no non-test caller yet — see the module docs' "No real assets yet"
impl RoadCatalogue {
    pub fn get(&self, kind: RoadPieceKind) -> Option<&Blueprint> {
        self.pieces.get(&kind)
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }
}

/// The size check described in the module docs, split out the same way
/// `blueprint::catalogue::validate` is — one `if` per rule, testable against
/// a hand-built [`Blueprint`] without a real file.
#[allow(dead_code)] // used via load_piece, see the module docs
fn validate(blueprint: &Blueprint) -> Result<(), RoadCatalogueError> {
    let size = blueprint.size;
    if size.x != ROAD_CELL_SIZE || size.z != ROAD_CELL_SIZE || size.y <= 0 || size.y > STRUCTURE_BLOCK_MAX_SIZE {
        return Err(RoadCatalogueError::InvalidSize(size));
    }
    if blueprint.palette.len() <= 1 {
        return Err(RoadCatalogueError::Empty);
    }
    Ok(())
}

/// Reads and validates the `.nbt` file for one `kind` out of `dir`. A
/// missing file is [`RoadCatalogueError::Missing`], not a panic and not
/// treated any differently from a malformed one by [`load_road_catalogue_dir`]
/// — both just mean this kind has no piece to select.
#[allow(dead_code)] // used via load_road_catalogue_dir, see the module docs
fn load_piece(dir: &Path, kind: RoadPieceKind) -> Result<Blueprint, RoadCatalogueError> {
    let path = dir.join(format!("{}.nbt", filename_for(kind)));
    if !path.is_file() {
        return Err(RoadCatalogueError::Missing);
    }
    let blueprint = read_structure_file(&path).map_err(RoadCatalogueError::Read)?;
    validate(&blueprint)?;
    Ok(blueprint)
}

/// Loads every [`RoadPieceKind`] it can find in `dir`, by its fixed
/// filename. Never panics and never fails outright — a missing directory or
/// a missing/bad individual file each just leave that kind (or every kind)
/// absent from the returned catalogue, reported in the second return value —
/// the same per-file tolerance [`crate::blueprint::load_catalogue_dir`]
/// gives buildings.
#[allow(dead_code)] // no non-test caller yet — see the module docs' "No real assets yet"
pub fn load_road_catalogue_dir(dir: &Path) -> (RoadCatalogue, Vec<(RoadPieceKind, RoadCatalogueError)>) {
    let mut pieces = HashMap::new();
    let mut skipped = Vec::new();

    for kind in RoadPieceKind::ALL {
        match load_piece(dir, kind) {
            Ok(blueprint) => {
                pieces.insert(kind, blueprint);
            }
            Err(err) => skipped.push((kind, err)),
        }
    }

    (RoadCatalogue { pieces }, skipped)
}

/// `<dir>/<kind's filename>.nbt`, exposed for callers that want to name the
/// path a piece would load from without loading it (log lines, the like).
#[allow(dead_code)] // no caller yet
pub fn piece_path(dir: &Path, kind: RoadPieceKind) -> PathBuf {
    dir.join(format!("{}.nbt", filename_for(kind)))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::blueprint::{write_structure_file, BlockState};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_road_catalogue_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    fn one_stone(size: IVec3) -> Blueprint {
        let volume = (size.x * size.y * size.z) as usize;
        let mut blocks = vec![0u16; volume];
        blocks[0] = 1;
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() }],
            blocks,
            data_version: 3953,
            failed_columns: 0,
        }
    }

    #[test]
    fn a_missing_directory_is_an_empty_catalogue_with_every_kind_missing() {
        let dir = std::env::temp_dir().join("block_viewer_test_road_catalogue_does_not_exist");
        let _ = fs::remove_dir_all(&dir);

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert_eq!(skipped.len(), 6);
        for (_, err) in &skipped {
            assert!(matches!(err, RoadCatalogueError::Missing));
        }
    }

    #[test]
    fn a_correctly_sized_piece_loads_under_its_kind() {
        let dir = temp_dir("straight");
        write_structure_file(&piece_path(&dir, RoadPieceKind::Straight), &one_stone(IVec3::new(6, 3, 6))).unwrap();

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert_eq!(catalogue.len(), 1);
        assert!(catalogue.get(RoadPieceKind::Straight).is_some());
        assert_eq!(skipped.len(), 5, "the other five kinds still have no file");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wrong_footprint_is_skipped_as_invalid_size() {
        let dir = temp_dir("wrong_size");
        // 5x*x6 instead of 6x*x6 — footprint doesn't match ROAD_CELL_SIZE.
        write_structure_file(&piece_path(&dir, RoadPieceKind::Cross), &one_stone(IVec3::new(5, 3, 6))).unwrap();

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(catalogue.get(RoadPieceKind::Cross).is_none());
        let (_, err) = skipped.iter().find(|(kind, _)| *kind == RoadPieceKind::Cross).unwrap();
        assert!(matches!(err, RoadCatalogueError::InvalidSize(_)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn all_six_kinds_load_independently() {
        let dir = temp_dir("all_six");
        for kind in RoadPieceKind::ALL {
            write_structure_file(&piece_path(&dir, kind), &one_stone(IVec3::new(6, 2, 6))).unwrap();
        }

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(catalogue.len(), 6);
        for kind in RoadPieceKind::ALL {
            assert!(catalogue.get(kind).is_some());
        }

        fs::remove_dir_all(&dir).ok();
    }
}
