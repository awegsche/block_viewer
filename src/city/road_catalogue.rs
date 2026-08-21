//! The road piece catalogue (ticket 054, roadmap F1/F3 groundwork; extended
//! to multiple styles by ticket 059): the fixed set of `.nbt` blueprints
//! [`super::road::RoadPieceKind`] selects between, one full set per road
//! *style*, loaded from `assets/city/roads`.
//!
//! ## A fixed set per style, not an open scan
//!
//! [`blueprint::load_catalogue_dir`](crate::blueprint::load_catalogue_dir)
//! (ticket 039) indexes *whatever* `.nbt` files it finds, keyed by filename.
//! A road piece isn't an open-ended catalogue that way — there are exactly
//! kinds ([`RoadPieceKind::ALL`](super::road::RoadPieceKind::ALL)) per
//! style, each with one canonical filename. What *is* open-ended (ticket
//! 059) is the set of styles: `assets/city/roads/<style>/*.nbt`, one
//! subdirectory per style, the subdirectory's own name serving as the style
//! id — the same "the filename is the id" call
//! [`blueprint::load_catalogue_dir`] makes for a building, just one level up
//! the path. [`load_road_catalogue_dir`] scans `assets/city/roads` for
//! subdirectories, then looks inside each one for those fixed filenames
//! by name rather than scanning it. A missing file is reported per
//! `(style, kind)` the same tolerant way 039 reports a bad file — the
//! catalogue that *does* load is still usable, just short a piece — rather
//! than failing the whole directory over one missing corner. A stray file
//! sitting directly in `assets/city/roads` (not a subdirectory) is ignored,
//! not mistaken for a style.
//!
//! ## What "validated" means here
//!
//! Narrower than 039's building validation: a road piece must be exactly
//! [`ROAD_CELL_SIZE`] blocks on `x` and `z` (its footprint is fixed, unlike
//! a building's), with no ceiling on `y` beyond
//! [`STRUCTURE_BLOCK_MAX_SIZE`], and a palette with more than just air —
//! the same "not actually empty" rule 039 already applies. Unchanged from
//! 054; styles don't loosen or tighten it.
//!
//! ## Style is a lookup key, not a shape rule
//!
//! [`super::road::select_piece`] decides *which* [`RoadPieceKind`] and
//! rotation a cell's connections call for, and that decision never
//! mentions style — a dirt-path cell and a paved-street cell connect to
//! each other exactly like two cells of the same style would; style only
//! picks *which* `.nbt` gets meshed/written for the kind once it's already
//! chosen. See `city::road_build`'s module docs for how a cell's own
//! recorded style (on [`super::state::City`], not passed around separately)
//! feeds [`RoadCatalogue::get`].
//!
//! ## No real assets yet
//!
//! Unlike 039's `house01.nbt`, no style ships with real `.nbt` road pieces
//! — they need an actual Minecraft structure-block export, not something to
//! fabricate. `load_road_catalogue_dir` against a missing or empty
//! `assets/city/roads` is exercised and comes back an empty catalogue (not
//! an error), and this module's own tests build synthetic fixtures the same
//! way `blueprint::catalogue`'s tests do.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::math::IVec3;
use bevy::prelude::Resource;

use crate::blueprint::{read_structure_file, Blueprint, StructureReadError, STRUCTURE_BLOCK_MAX_SIZE};

use super::road::RoadPieceKind;
use super::state::ROAD_CELL_SIZE;

/// Why a road piece file didn't become a catalogue entry.
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

/// The filename stem (without `.nbt`) each [`RoadPieceKind`] loads from,
/// inside a style's own subdirectory — the fixed names
/// `assets/city/roads/<style>` is expected to contain one of each of.
fn filename_for(kind: RoadPieceKind) -> &'static str {
    match kind {
        RoadPieceKind::Isolated => "isolated",
        RoadPieceKind::DeadEnd => "dead_end",
        RoadPieceKind::Straight => "straight",
        RoadPieceKind::Corner => "corner",
        RoadPieceKind::T => "t",
        RoadPieceKind::Cross => "cross",
        // Ticket 068: `stairs`, plural — that's what the shipped export is
        // called, and what the Minecraft block family it's built out of is
        // called. The `RoadPieceKind` variant stays singular.
        RoadPieceKind::Stair => "stairs",
    }
}

/// Every loaded road piece, keyed by style and then kind —
/// [`super::state::City::road_style_at`] plus [`super::road::select_piece`]'s
/// own output is exactly this catalogue's key pair. A nested map, not a flat
/// `HashMap<(String, RoadPieceKind), Blueprint>`, so [`get`](Self::get)
/// doesn't need to allocate a lookup key out of a borrowed `&str`.
///
/// `Resource` (ticket 055): `city::run` inserts this the same way it inserts
/// `BuildingCatalogue`, and `city::road_build` reads it via
/// `Option<Res<RoadCatalogue>>`.
#[derive(Resource, Default)]
pub struct RoadCatalogue {
    styles: HashMap<String, HashMap<RoadPieceKind, Blueprint>>,
}

impl RoadCatalogue {
    pub fn get(&self, style: &str, kind: RoadPieceKind) -> Option<&Blueprint> {
        self.styles.get(style)?.get(&kind)
    }

    /// Every style id with at least one piece loaded, sorted — the order
    /// `city::road_build`'s `[`/`]` style cycle walks.
    pub fn styles(&self) -> Vec<&str> {
        let mut styles: Vec<&str> = self.styles.keys().map(String::as_str).collect();
        styles.sort_unstable();
        styles
    }

    /// Total pieces loaded, across every style — not "how many styles."
    pub fn len(&self) -> usize {
        self.styles.values().map(HashMap::len).sum()
    }

    #[allow(dead_code)] // no non-test caller yet — mirrors `BuildingCatalogue::is_empty`
    pub fn is_empty(&self) -> bool {
        self.styles.is_empty()
    }
}

/// The size check described in the module docs, split out the same way
/// `blueprint::catalogue::validate` is — one `if` per rule, testable against
/// a hand-built [`Blueprint`] without a real file.
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

/// Reads and validates the `.nbt` file for one `kind` out of `style_dir`. A
/// missing file is [`RoadCatalogueError::Missing`], not a panic and not
/// treated any differently from a malformed one by [`load_road_catalogue_dir`]
/// — both just mean this kind has no piece to select for this style.
fn load_piece(style_dir: &Path, kind: RoadPieceKind) -> Result<Blueprint, RoadCatalogueError> {
    let path = style_dir.join(format!("{}.nbt", filename_for(kind)));
    if !path.is_file() {
        return Err(RoadCatalogueError::Missing);
    }
    let blueprint = read_structure_file(&path).map_err(RoadCatalogueError::Read)?;
    validate(&blueprint)?;
    Ok(blueprint)
}

/// Every subdirectory directly inside `dir`, sorted by name — a missing
/// `dir` is an empty list, not an error, the same tolerance a missing style
/// piece gets. A non-directory entry (a stray file) is skipped, not treated
/// as an (inevitably empty) style.
fn style_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(entries) => entries.filter_map(Result::ok).map(|entry| entry.path()).filter(|path| path.is_dir()).collect(),
        Err(_) => Vec::new(),
    };
    dirs.sort();
    dirs
}

/// Loads every style [`load_road_catalogue_dir`] finds under `dir`, and
/// every [`RoadPieceKind`] it can find within each one, by its fixed
/// filename. Never panics and never fails outright — a missing directory, an
/// empty one, or a missing/bad individual file each just leave that style
/// (or every style) absent or incomplete in the returned catalogue, reported
/// in the second return value — the same per-file tolerance
/// [`crate::blueprint::load_catalogue_dir`] gives buildings.
pub fn load_road_catalogue_dir(dir: &Path) -> (RoadCatalogue, Vec<(String, RoadPieceKind, RoadCatalogueError)>) {
    let mut styles = HashMap::new();
    let mut skipped = Vec::new();

    for style_dir in style_dirs(dir) {
        let Some(style) = style_dir.file_name().and_then(|name| name.to_str()) else { continue };
        let style = style.to_string();

        let mut pieces = HashMap::new();
        for kind in RoadPieceKind::ALL {
            match load_piece(&style_dir, kind) {
                Ok(blueprint) => {
                    pieces.insert(kind, blueprint);
                }
                Err(err) => skipped.push((style.clone(), kind, err)),
            }
        }
        if !pieces.is_empty() {
            styles.insert(style, pieces);
        }
    }

    (RoadCatalogue { styles }, skipped)
}

/// `<dir>/<style>/<kind's filename>.nbt`, exposed for callers (and tests)
/// that want to name the path a piece would load from without loading it.
#[allow(dead_code)] // no non-test caller yet
pub fn piece_path(dir: &Path, style: &str, kind: RoadPieceKind) -> PathBuf {
    dir.join(style).join(format!("{}.nbt", filename_for(kind)))
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

    /// Writes `blueprint` at `piece_path(dir, style, kind)`, creating the
    /// style subdirectory first — `write_structure_file` doesn't create
    /// parent directories itself, and `temp_dir` only creates `dir` proper.
    fn write_piece(dir: &Path, style: &str, kind: RoadPieceKind, blueprint: &Blueprint) {
        fs::create_dir_all(dir.join(style)).expect("should create style dir");
        write_structure_file(&piece_path(dir, style, kind), blueprint).unwrap();
    }

    #[test]
    fn a_missing_directory_is_an_empty_catalogue_with_nothing_skipped() {
        let dir = std::env::temp_dir().join("block_viewer_test_road_catalogue_does_not_exist");
        let _ = fs::remove_dir_all(&dir);

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert!(skipped.is_empty(), "no style directories to iterate at all");
    }

    #[test]
    fn a_stray_file_next_to_style_directories_is_ignored() {
        let dir = temp_dir("stray_file");
        fs::write(dir.join("readme.txt"), b"not a style").unwrap();

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(catalogue.is_empty());
        assert!(skipped.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_correctly_sized_piece_loads_under_its_style_and_kind() {
        let dir = temp_dir("straight");
        write_piece(&dir, "dirt", RoadPieceKind::Straight, &one_stone(IVec3::new(6, 3, 6)));

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert_eq!(catalogue.len(), 1);
        assert!(catalogue.get("dirt", RoadPieceKind::Straight).is_some());
        assert_eq!(catalogue.styles(), vec!["dirt"]);
        assert_eq!(skipped.len(), RoadPieceKind::ALL.len() - 1, "every other kind of this one style still has no file");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wrong_footprint_is_skipped_as_invalid_size() {
        let dir = temp_dir("wrong_size");
        // 5x*x6 instead of 6x*x6 — footprint doesn't match ROAD_CELL_SIZE.
        write_piece(&dir, "dirt", RoadPieceKind::Cross, &one_stone(IVec3::new(5, 3, 6)));

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(catalogue.get("dirt", RoadPieceKind::Cross).is_none());
        let (style, _, err) = skipped.iter().find(|(_, kind, _)| *kind == RoadPieceKind::Cross).unwrap();
        assert_eq!(style, "dirt");
        assert!(matches!(err, RoadCatalogueError::InvalidSize(_)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn every_kind_loads_independently_for_one_style() {
        let dir = temp_dir("all_kinds");
        for kind in RoadPieceKind::ALL {
            write_piece(&dir, "dirt", kind, &one_stone(IVec3::new(6, 2, 6)));
        }

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(catalogue.len(), RoadPieceKind::ALL.len());
        for kind in RoadPieceKind::ALL {
            assert!(catalogue.get("dirt", kind).is_some());
        }

        fs::remove_dir_all(&dir).ok();
    }

    /// Ticket 059's own reason to exist: two styles, loaded from two
    /// subdirectories, resolve completely independently of each other.
    #[test]
    fn two_styles_load_independently() {
        let dir = temp_dir("two_styles");
        for style in ["dirt", "paved"] {
            for kind in RoadPieceKind::ALL {
                write_piece(&dir, style, kind, &one_stone(IVec3::new(6, 2, 6)));
            }
        }

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        assert_eq!(catalogue.len(), RoadPieceKind::ALL.len() * 2);
        assert_eq!(catalogue.styles(), vec!["dirt", "paved"]);
        for style in ["dirt", "paved"] {
            for kind in RoadPieceKind::ALL {
                assert!(catalogue.get(style, kind).is_some(), "{style}/{kind:?}");
            }
        }
        // A lookup under a style that was never loaded finds nothing, not a
        // panic, and doesn't fall back to some other style's piece.
        assert!(catalogue.get("gravel", RoadPieceKind::Straight).is_none());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_style_missing_some_pieces_still_loads_the_rest() {
        let dir = temp_dir("partial_style");
        write_piece(&dir, "dirt", RoadPieceKind::Straight, &one_stone(IVec3::new(6, 3, 6)));

        let (catalogue, skipped) = load_road_catalogue_dir(&dir);
        assert!(catalogue.get("dirt", RoadPieceKind::Straight).is_some());
        assert_eq!(skipped.len(), RoadPieceKind::ALL.len() - 1, "every other kind of this style");
        for (style, _, err) in &skipped {
            assert_eq!(style, "dirt");
            assert!(matches!(err, RoadCatalogueError::Missing));
        }

        fs::remove_dir_all(&dir).ok();
    }

    // -- the shipped assets vs the authoring convention (ticket 066) --------

    /// Which of a piece's four edges its road surface actually reaches, read
    /// off the blueprint's surface course.
    ///
    /// "The road surface" is defined as *whatever block sits at the piece's
    /// own centre* — a road cell is always paved through the middle, whatever
    /// the style paves it with — so this doesn't hard-code
    /// `minecraft:dirt_path` and works for a future style paved in stone.
    /// An edge is open when both of its two centre columns carry that same
    /// block; a kerb or a grass shoulder closing the edge off reads as shut.
    fn open_edges(piece: &Blueprint) -> super::super::road::RoadConnections {
        let (sx, sz) = (piece.size.x as usize, piece.size.z as usize);
        let y = super::super::road_build::ROAD_PIECE_SUBGRADE_DEPTH as usize;
        let at = |x: usize, z: usize| piece.palette[piece.blocks[y * sz * sx + z * sx + x] as usize].name.as_str();

        let mid = (ROAD_CELL_SIZE / 2) as usize; // 3 — the far half of the two centre columns
        let surface = at(mid, mid);
        let both = |a: (usize, usize), b: (usize, usize)| at(a.0, a.1) == surface && at(b.0, b.1) == surface;

        super::super::road::RoadConnections {
            north: both((mid - 1, 0), (mid, 0)),
            south: both((mid - 1, sz - 1), (mid, sz - 1)),
            west: both((0, mid - 1), (0, mid)),
            east: both((sx - 1, mid - 1), (sx - 1, mid)),
        }
    }

    /// The stair's own geometry check (ticket 068), pinning the three things
    /// `city::road_build`'s height planning takes on faith about it: that its
    /// **low** end is the south edge (so the piece follows ticket 066's "every
    /// piece opens south" convention, and `stair_rotation`'s "canonical
    /// ascends north" is true of the actual file), that its **high** end is
    /// the north edge, and that the two are exactly `ROAD_STAIR_RISE` apart —
    /// the number every level in a `plan_drag` profile is a multiple of. A
    /// re-export with a five-block rise would leave every road built from it
    /// with a one-block lip at each ramp, and nothing else would notice.
    #[test]
    fn the_shipped_stair_climbs_north_by_exactly_one_stair_rise() {
        let (catalogue, _skipped) = load_road_catalogue_dir(Path::new("assets/city/roads"));
        let Some(piece) = catalogue.get("dirt", RoadPieceKind::Stair) else {
            panic!("assets/city/roads/dirt/{}.nbt should be checked in", filename_for(RoadPieceKind::Stair));
        };

        let (sx, sz) = (piece.size.x as usize, piece.size.z as usize);
        let at = |x: usize, y: usize, z: usize| {
            piece.palette[piece.blocks[y * sz * sx + z * sx + x] as usize].name.as_str()
        };
        let low = super::super::road_build::ROAD_PIECE_SUBGRADE_DEPTH as usize;
        let high = low + super::super::road_build::ROAD_STAIR_RISE as usize;
        assert!(piece.size.y as usize > high, "a stair needs at least its own rise plus clearance");

        // The surface course is whatever paves the low end's centre —
        // `dirt_path` for this style, but read rather than assumed, the same
        // way `open_edges` does it.
        let mid = (ROAD_CELL_SIZE / 2) as usize;
        let surface = at(mid, low, sz - 1);

        for x in [mid - 1, mid] {
            assert_eq!(at(x, low, sz - 1), surface, "the stair's low end should pave the SOUTH edge at y={low}");
            assert_eq!(at(x, high, 0), surface, "the stair's high end should pave the NORTH edge at y={high}");
        }
        // And not level: a flat piece would pave both edges on the same layer,
        // which is exactly what this test exists to tell apart.
        assert_ne!(at(mid, low, 0), surface, "the NORTH edge must be fill at the low layer, not road");
    }

    /// `road_write_edit` reacts to a `RotationError` by printing one line and
    /// **skipping the cell** — an invisible hole in a road. So every shipped
    /// piece is rotated through all four rotations here, where an
    /// unrotatable property is a test failure naming the block instead.
    /// Ticket 068: the stair piece is the most property-dense one yet
    /// (cobblestone stairs with `outer_left`/`outer_right` shapes, oak fences
    /// with four connection booleans).
    #[test]
    fn every_shipped_piece_rotates_through_all_four_rotations() {
        use crate::blueprint::{rotate_blueprint, Rotation};

        let (catalogue, _skipped) = load_road_catalogue_dir(Path::new("assets/city/roads"));
        for style in catalogue.styles() {
            for kind in RoadPieceKind::ALL {
                let Some(piece) = catalogue.get(&style, kind) else { continue };
                for rotation in [Rotation::Deg0, Rotation::Deg90, Rotation::Deg180, Rotation::Deg270] {
                    let rotated = rotate_blueprint(piece, rotation)
                        .unwrap_or_else(|err| panic!("{style}/{}.nbt at {rotation:?}: {err}", filename_for(kind)));
                    // A quarter turn swaps x and z; a road piece is square, so
                    // every rotation of one has to come back the same size.
                    assert_eq!(rotated.size, piece.size, "{style}/{}.nbt at {rotation:?}", filename_for(kind));
                }
            }
        }
    }

    /// The guard ticket 066 exists because nothing had: the `.nbt` files
    /// checked into `assets/city/roads/dirt` are read, and each one's real
    /// open edges are compared against the orientation
    /// [`super::super::road::canonical_pattern`] claims it was authored at.
    /// A re-export at a different orientation, or a table edited without the
    /// assets, fails here rather than silently rotating every corner in the
    /// world by 180°.
    ///
    /// Two kinds are skipped.
    ///
    /// [`RoadPieceKind::Isolated`]: `isolated.nbt` currently ships as a
    /// byte-identical copy of `dead_end.nbt` (a south-pointing stub), which
    /// the canonical "connects to nothing" can't describe and `select_piece`
    /// — which always answers `Deg0` for an unoriented piece — has no
    /// rotation to fix it with. Re-exporting a real island piece is noted in
    /// `todo.md`; it needs Minecraft, not code.
    ///
    /// [`RoadPieceKind::Stair`] (ticket 068): [`open_edges`] reads *one*
    /// layer, which is the whole story for a flat piece and meaningless for
    /// a ramp whose two ends are `ROAD_STAIR_RISE` layers apart — at the
    /// subgrade layer the stair's high edge is solid fill, so this rule would
    /// read it as opening the wrong way entirely. See
    /// [`the_shipped_stair_climbs_north_by_exactly_one_stair_rise`], which
    /// pins more about it than this could.
    #[test]
    fn the_shipped_dirt_pieces_are_authored_at_the_canonical_orientations() {
        let (catalogue, _skipped) = load_road_catalogue_dir(Path::new("assets/city/roads"));
        assert!(catalogue.get("dirt", RoadPieceKind::Straight).is_some(), "the dirt style should be checked in");

        for kind in RoadPieceKind::ALL {
            if matches!(kind, RoadPieceKind::Isolated | RoadPieceKind::Stair) {
                continue; // see the doc comment
            }
            let Some(piece) = catalogue.get("dirt", kind) else {
                continue; // a kind with no shipped asset yet — tolerated, same as the loader does
            };
            assert_eq!(
                open_edges(piece),
                super::super::road::canonical_pattern(kind),
                "assets/city/roads/dirt/{}.nbt is exported at a different orientation than city::road::canonical_pattern                  claims — fix whichever is wrong, and keep road.rs's module docs and the style's README with it",
                filename_for(kind),
            );
        }
    }
}
