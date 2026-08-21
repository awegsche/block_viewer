//! What a block gives you when the game clears it away (ticket 072, roadmap
//! H2) — the one translation step between the world's block names and
//! [`super::inventory::Stock`]'s material ids, which are otherwise the same
//! alphabet on purpose.
//!
//! ## A table, not a catalogue
//!
//! Every other data directory here is one file per id — one `.ron` per
//! building, one per road type, the filename stem *being* the id. Drops
//! aren't a set of entities, they're a mapping over hundreds of block names,
//! so they get a single file: `assets/city/drops.ron`. [`load_drop_table`]
//! is deliberately not shaped like `load_definitions_dir`.
//!
//! ## Four rules, in order
//!
//! 1. **Air drops nothing**, hardcoded (`minecraft:air`, `cave_air`,
//!    `void_air`) rather than table entries. A placement's baseline records
//!    what stood at *every* position it wrote, and a blueprint is mostly
//!    air — a table that forgot to list it would stock air by the thousand
//!    on every house, and "did you remember to exclude air" is not a thing
//!    an asset file should be able to get wrong.
//! 2. A name listed in `nothing` drops nothing — foliage, fluids, fire.
//! 3. A name listed in `replaced` drops what it says — stone gives
//!    cobblestone, the way mining it does.
//! 4. **Anything else drops itself, one for one.** Minecraft's own default,
//!    and the reason a missing `drops.ron` is a crude economy rather than a
//!    dead one.
//!
//! ## What the table deliberately doesn't model
//!
//! The key is [`BlockState::name`] alone: properties are ignored, so an open
//! door and a closed one drop the same thing, and a double slab drops one
//! slab rather than two. Fortune, silk touch, tool requirements and block
//! entities' contents are all out of scope — this is a city's material
//! ledger, not a mining simulator. A name appearing in both lists is a
//! [`DropTableError::Contradiction`], because that is a mistake in the file
//! rather than a precedence question worth answering.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use bevy::prelude::Resource;
use serde::Deserialize;

use crate::blueprint::BlockState;

use super::inventory::Parcel;

// Ticket 073 is the wiring — placement charges, clearing credits, undo
// settles — and it turns every one of these into a real caller. Marked
// rather than left warning for the one commit in between, the same
// "no caller yet" note `city::road`'s F4 queries carry.
/// The three names that are air. `BlockState::AIR` covers the first; the
/// other two only ever appear in generated terrain, which is exactly what a
/// placement clears.
#[allow(dead_code)] // read by `drop_for` — see `impl DropTable`
const AIR_NAMES: [&str; 3] = [BlockState::AIR, "minecraft:cave_air", "minecraft:void_air"];

/// One entry of the file's `replaced` map: what a block gives instead of
/// itself.
#[derive(Debug, Clone, Deserialize)]
pub struct Drop {
    pub item: String,
    /// Defaulted to 1 so the long shipped table doesn't repeat `count: 1`
    /// three dozen times. Zero is refused ([`DropTableError::ZeroCount`]) —
    /// "gives nothing" is what the `nothing` list is for, and two spellings
    /// of one meaning is how a table grows a contradiction later.
    #[serde(default = "one")]
    pub count: u32,
}

fn one() -> u32 {
    1
}

/// The file's on-disk shape. Not [`DropTable`] itself: the loaded form
/// normalizes every name and turns `nothing` into a set, and neither of
/// those should be a thing a hand-written file has to get right.
#[derive(Debug, Deserialize)]
struct DropFile {
    #[serde(default)]
    nothing: Vec<String>,
    #[serde(default)]
    replaced: HashMap<String, Drop>,
}

/// Loaded, normalized drop rules — see the module docs' four rules.
///
/// [`Default`] is the "no `drops.ron`" table: no exceptions at all, so every
/// non-air block drops itself.
#[derive(Resource, Debug, Default)]
pub struct DropTable {
    nothing: HashSet<String>,
    replaced: HashMap<String, Drop>,
}

// Ticket 073 is what wires the table to the write paths (placement credits
// what it cleared, terraform credits what it dug); until that commit lands
// only the tests below call these, the same "no caller yet" mark
// `city::road`'s F4 queries carry.
#[allow(dead_code)]
impl DropTable {
    /// What clearing `block` yields: the item id and how many, or `None` for
    /// air and for anything the table says gives nothing.
    ///
    /// The returned id borrows from either the table or `block` — rule 4
    /// hands back the block's own name rather than allocating a copy of it
    /// for every one of the tens of thousands of positions a placement
    /// touches.
    pub fn drop_for<'a>(&'a self, block: &'a BlockState) -> Option<(&'a str, u64)> {
        let name = block.name.as_str();
        if AIR_NAMES.contains(&name) || self.nothing.contains(name) {
            return None;
        }
        match self.replaced.get(name) {
            Some(drop) => Some((drop.item.as_str(), u64::from(drop.count))),
            None => Some((name, 1)),
        }
    }

    /// Every block in `blocks` run through [`drop_for`](Self::drop_for) and
    /// summed. Takes an iterator of references so a
    /// [`Baseline`](super::journal::Baseline)'s `previous`/`written` can be
    /// fed straight in without collecting anything first.
    pub fn parcel_for<'a>(&self, blocks: impl Iterator<Item = &'a BlockState>) -> Parcel {
        let mut parcel = Parcel::default();
        for block in blocks {
            if let Some((item, count)) = self.drop_for(block) {
                parcel.add(item, count);
            }
        }
        parcel
    }

    /// How many names the table has an opinion about — [`super::run`]'s log
    /// line, the same "say what you loaded" note every other loader here
    /// prints.
    pub fn len(&self) -> usize {
        self.nothing.len() + self.replaced.len()
    }
}

/// Why `drops.ron` didn't load. A missing file is **not** one of these — see
/// [`load_drop_table`].
#[derive(Debug)]
pub enum DropTableError {
    Read(std::io::Error),
    /// Not well-formed RON, or didn't match [`DropFile`]'s shape. Carries
    /// RON's own message, the same call [`super::definition::DefinitionError`]
    /// makes.
    Parse(String),
    /// A name appears in both `nothing` and `replaced`.
    Contradiction(String),
    /// A `replaced` entry with `count: 0` — use `nothing` instead.
    ZeroCount(String),
    /// A `replaced` entry whose `item` is blank.
    EmptyItem(String),
}

impl std::fmt::Display for DropTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DropTableError::Read(err) => write!(f, "{err}"),
            DropTableError::Parse(msg) => write!(f, "{msg}"),
            DropTableError::Contradiction(name) => {
                write!(f, "{name} is listed both as dropping nothing and as dropping something")
            }
            DropTableError::ZeroCount(name) => write!(f, "{name} drops a count of 0 — list it under `nothing` instead"),
            DropTableError::EmptyItem(name) => write!(f, "{name} drops an item with no name"),
        }
    }
}

impl std::error::Error for DropTableError {}

/// `stone` -> `minecraft:stone`, leaving an already-namespaced name alone —
/// the same shorthand [`BlockState`]'s own `FromStr` accepts, applied here so
/// the table's keys and a decoded block's `name` can't miss each other over
/// a prefix nobody wants to type forty times.
fn namespaced(name: &str) -> String {
    let name = name.trim();
    if name.contains(':') {
        name.to_string()
    } else {
        format!("minecraft:{name}")
    }
}

/// Reads `path` into a [`DropTable`]. A **missing file is
/// `Ok(DropTable::default())`** — rule 4 means an absent table still gives a
/// working economy, and the same "a missing directory is an empty
/// catalogue" tolerance `blueprint::load_catalogue_dir` already has. Every
/// other failure is an error the caller shows in the definition-errors
/// panel; unlike a directory of definitions there is no per-entry recovery,
/// because a table half of whose rules didn't apply would be worse than one
/// that isn't there.
pub fn load_drop_table(path: &Path) -> Result<DropTable, DropTableError> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(DropTable::default()),
        Err(err) => return Err(DropTableError::Read(err)),
    };

    let file: DropFile = ron::de::from_str(&text).map_err(|err| DropTableError::Parse(err.to_string()))?;

    let nothing: HashSet<String> = file.nothing.iter().map(|name| namespaced(name)).collect();
    let mut replaced = HashMap::with_capacity(file.replaced.len());
    for (name, drop) in file.replaced {
        let name = namespaced(&name);
        if nothing.contains(&name) {
            return Err(DropTableError::Contradiction(name));
        }
        if drop.count == 0 {
            return Err(DropTableError::ZeroCount(name));
        }
        if drop.item.trim().is_empty() {
            return Err(DropTableError::EmptyItem(name));
        }
        replaced.insert(
            name,
            Drop {
                item: namespaced(&drop.item),
                count: drop.count,
            },
        );
    }

    Ok(DropTable { nothing, replaced })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str) -> BlockState {
        BlockState {
            name: name.to_string(),
            properties: Vec::new(),
        }
    }

    /// One temp directory per call — the tests run in parallel threads and
    /// would otherwise write each other's fixture out from under themselves.
    fn table_from(text: &str) -> Result<DropTable, DropTableError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nth = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("block_viewer_drops_{}_{nth}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("drops.ron");
        fs::write(&path, text).expect("write");
        let result = load_drop_table(&path);
        let _ = fs::remove_dir_all(&dir);
        result
    }

    const TABLE: &str = r#"(
        nothing: ["minecraft:short_grass", "water"],
        replaced: {
            "minecraft:stone": (item: "minecraft:cobblestone"),
            "minecraft:grass_block": (item: "dirt", count: 1),
            "minecraft:coal_ore": (item: "minecraft:coal", count: 2),
        },
    )"#;

    // --- the four rules ----------------------------------------------------

    #[test]
    fn air_drops_nothing_without_being_listed() {
        let table = DropTable::default();
        for name in AIR_NAMES {
            assert_eq!(table.drop_for(&block(name)), None, "{name}");
        }
    }

    #[test]
    fn a_listed_nothing_drops_nothing() {
        let table = table_from(TABLE).expect("loads");
        assert_eq!(table.drop_for(&block("minecraft:short_grass")), None);
    }

    #[test]
    fn a_replaced_block_drops_what_it_says() {
        let table = table_from(TABLE).expect("loads");
        assert_eq!(table.drop_for(&block("minecraft:stone")), Some(("minecraft:cobblestone", 1)));
        assert_eq!(table.drop_for(&block("minecraft:coal_ore")), Some(("minecraft:coal", 2)));
    }

    #[test]
    fn an_unlisted_block_drops_itself() {
        let table = table_from(TABLE).expect("loads");
        assert_eq!(table.drop_for(&block("minecraft:oak_planks")), Some(("minecraft:oak_planks", 1)));
    }

    #[test]
    fn an_empty_table_still_drops_everything_that_is_not_air() {
        let table = DropTable::default();
        assert_eq!(table.drop_for(&block("minecraft:stone")), Some(("minecraft:stone", 1)));
    }

    #[test]
    fn properties_do_not_change_a_drop() {
        let table = table_from(TABLE).expect("loads");
        let open_door = BlockState {
            name: "minecraft:oak_door".to_string(),
            properties: vec![("open".to_string(), "true".to_string())],
        };
        assert_eq!(table.drop_for(&open_door), Some(("minecraft:oak_door", 1)));
    }

    // --- namespacing -------------------------------------------------------

    #[test]
    fn a_bare_name_in_the_file_matches_a_namespaced_block() {
        let table = table_from(TABLE).expect("loads");
        assert_eq!(table.drop_for(&block("minecraft:water")), None, "listed as bare `water`");
        assert_eq!(
            table.drop_for(&block("minecraft:grass_block")),
            Some(("minecraft:dirt", 1)),
            "drops bare `dirt`"
        );
    }

    // --- parcel_for --------------------------------------------------------

    #[test]
    fn a_parcel_sums_a_run_of_blocks_and_skips_the_air() {
        let table = table_from(TABLE).expect("loads");
        let blocks = [
            block("minecraft:stone"),
            block("minecraft:stone"),
            block("minecraft:air"),
            block("minecraft:short_grass"),
            block("minecraft:dirt"),
        ];

        let parcel = table.parcel_for(blocks.iter());

        assert_eq!(parcel.get("minecraft:cobblestone"), 2);
        assert_eq!(parcel.get("minecraft:dirt"), 1);
        assert_eq!(parcel.distinct(), 2, "air and short grass contribute nothing");
    }

    // --- load failures -----------------------------------------------------

    #[test]
    fn a_missing_file_is_an_empty_table_not_an_error() {
        let path = std::env::temp_dir().join("block_viewer_drops_definitely_not_here.ron");
        let _ = fs::remove_file(&path);
        let table = load_drop_table(&path).expect("a missing file is not an error");
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn a_name_in_both_lists_is_refused() {
        let err = table_from(r#"(nothing: ["minecraft:stone"], replaced: {"stone": (item: "cobblestone")})"#)
            .expect_err("a contradiction");
        assert!(matches!(err, DropTableError::Contradiction(_)), "{err}");
    }

    #[test]
    fn a_zero_count_is_refused_rather_than_read_as_nothing() {
        let err = table_from(r#"(replaced: {"stone": (item: "cobblestone", count: 0)})"#).expect_err("zero count");
        assert!(matches!(err, DropTableError::ZeroCount(_)), "{err}");
    }

    #[test]
    fn a_blank_item_is_refused() {
        let err = table_from(r#"(replaced: {"stone": (item: "  ")})"#).expect_err("blank item");
        assert!(matches!(err, DropTableError::EmptyItem(_)), "{err}");
    }

    #[test]
    fn nonsense_is_a_parse_error_not_a_panic() {
        let err = table_from("this is not ron").expect_err("parse");
        assert!(matches!(err, DropTableError::Parse(_)), "{err}");
    }

    #[test]
    fn the_shipped_table_loads() {
        // The real asset, not a fixture — a typo in `assets/city/drops.ron`
        // should fail here rather than at the player's first placement.
        let table = load_drop_table(Path::new("assets/city/drops.ron")).expect("assets/city/drops.ron loads");
        assert!(table.len() > 0, "the shipped table should have rules in it");
        assert_eq!(table.drop_for(&block("minecraft:stone")), Some(("minecraft:cobblestone", 1)));
    }
}
