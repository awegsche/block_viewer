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
//! ## The tech tree (ticket 041, roadmap C2)
//!
//! `requires` names other building ids, and nothing about a single file can
//! tell you whether those ids exist or form a loop — both questions need
//! *every* definition loaded first. So [`build_definitions`] runs a second
//! pass, [`resolve_requirements`], after the per-file loop above: it checks
//! `requires` edges against the whole loaded set and removes anything that
//! can never unlock — a [`DefinitionError::DanglingRequirement`] (points at
//! an id nothing loaded) or a [`DefinitionError::CyclicRequirement`] (on a
//! loop). Per the roadmap: "a tech tree with a cycle is unwinnable and the
//! failure mode is 'button greyed out forever' if it isn't caught" — true of
//! a dangling edge too, for the same reason.
//!
//! Removing an entry can turn some *other* entry's `requires` into a fresh
//! dangling reference (it needed the thing that just got removed for an
//! unrelated problem), so [`resolve_requirements`] loops — dangling pass,
//! then cycle pass — until a pass removes nothing. What survives is the
//! maximal subset of the loaded buildings whose `requires` graph, restricted
//! to that subset, resolves and has no cycle; that end state doesn't depend
//! on which order the passes happen to remove things in.
//!
//! ## Failure is per-file, not per-directory
//!
//! Same contract as [`load_catalogue_dir`](super::blueprint::load_catalogue_dir):
//! [`load_definitions_dir`] never panics. A missing directory is an empty
//! set of definitions (logged, not fatal); a malformed or invalid file is
//! skipped and reported alongside whatever else loaded.

use std::collections::HashMap;
use std::collections::HashSet;
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
    /// Other building ids this one unlocks after. Checked against the whole
    /// loaded set by [`resolve_requirements`] — dangling ids and cycles are
    /// [`DefinitionError`]s, not silently-unreachable build-menu entries.
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub footprint: FootprintSpec,
    /// `None` for iteration 1's non-functional buildings.
    #[serde(default)]
    pub production: Option<Production>,
    #[serde(default)]
    pub cost: Vec<Cost>,
    /// `Some` makes this building a warehouse (ticket 079, roadmap H2) — it
    /// collects from the producers it can reach along the road, and adds its
    /// `storage` to what the city can hold. `None` for everything else.
    ///
    /// A warehouse *tier* is not a concept of its own: it is
    /// [`tier`](Self::tier) and [`requires`](Self::requires), which
    /// `resolve_requirements` (ticket 041) and the build menu already
    /// implement. A tier-2 warehouse is a second `.ron` with bigger numbers.
    #[serde(default)]
    pub warehouse: Option<Warehouse>,
    pub integrity: Integrity,
}

/// What a warehouse does, and the four knobs a tier moves — see
/// [`super::warehouse`] for how coverage is worked out and
/// [`Building::warehouse`] for why there is no `tier` field here.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Warehouse {
    /// How far along the road network this warehouse reaches, in road cells
    /// ([`super::state::ROAD_CELL_SIZE`] blocks each). Hops, not travel time:
    /// a working radius is a *distance*, and a faster road should make a haul
    /// quicker rather than make the warehouse reach further.
    pub radius_cells: u32,
    /// How many stacks this warehouse can have in flight at once (ticket
    /// 080) — the throughput knob. One warehouse serving six farms delivers
    /// them a stack at a time and falls behind; a tier upgrade is how that
    /// gets fixed.
    pub concurrent_hauls: u32,
    /// A fixed load/unload cost added to every haul, on top of its travel
    /// time — so a tier upgrade helps nearby producers too, not only distant
    /// ones.
    #[serde(default)]
    pub handling_minutes: f32,
    /// What this warehouse adds to the city's total storage capacity
    /// (`economy.base_storage` is the floor under it). Not per-warehouse
    /// storage: the pile stays global — see [`super::inventory::Stock`].
    pub storage: u64,
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
    /// How many stacks (`economy.stack_size`) this building can hold before
    /// it stops working and waits for a haul — ticket 078. A *building*
    /// property rather than an economy-wide one: a grain silo holding more
    /// than a fisherman's hut is the kind of difference a definition exists
    /// to state.
    #[serde(default = "default_buffer_stacks")]
    pub buffer_stacks: u32,
}

/// Four stacks: enough that a warehouse a short haul away never starves a
/// producer, few enough that one across the city visibly does.
fn default_buffer_stacks() -> u32 {
    4
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
    /// `production.buffer_stacks: 0` (ticket 078) — see `check_building`.
    ZeroBufferStacks,
    /// A `warehouse` block with a value nothing downstream could use
    /// (ticket 079) — carries the rule it broke, since there are three and
    /// they read the same way in a panel.
    InvalidWarehouse(&'static str),
    /// `footprint: Explicit { x, z }` has a non-positive axis.
    InvalidFootprint { x: i32, z: i32 },
    /// The filename has nothing usable before its extension.
    NoFilenameStem,
    /// Another file in the same directory already claimed this id.
    DuplicateId { id: String, other: PathBuf },
    /// `requires` names an id that isn't in the final loaded set — either
    /// nothing on disk claims it, or it was itself removed by
    /// [`resolve_requirements`] for a problem of its own.
    DanglingRequirement(String),
    /// `requires` puts this building on a dependency cycle. Carries every id
    /// on the cycle, in order, so the message names the whole loop rather
    /// than just this one building.
    CyclicRequirement(Vec<String>),
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
            DefinitionError::ZeroBufferStacks => write!(f, "production.buffer_stacks must be > 0"),
            DefinitionError::InvalidWarehouse(rule) => write!(f, "warehouse.{rule}"),
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
            DefinitionError::DanglingRequirement(missing) => {
                write!(f, "requires {missing:?}, which is not a loaded building")
            }
            DefinitionError::CyclicRequirement(cycle) => {
                write!(f, "requires cycle: {}", cycle.join(" -> "))
            }
        }
    }
}

impl std::error::Error for DefinitionError {}

/// One loaded building: an id, where it came from, the parsed [`Building`],
/// and its footprint resolved against the catalogue.
pub struct LoadedBuilding {
    /// The filename stem, e.g. `lumberjack.ron` -> `"lumberjack"`. A
    /// *definition* id — not necessarily the same string as
    /// [`Self::catalogue_id`], since a `.ron` file and the `.nbt` it names
    /// are free to have different stems (`lumberjack.ron` pointing at
    /// `house01.nbt` is a legal, if confusing, definition).
    pub id: String,
    pub path: PathBuf,
    pub building: Building,
    /// `(x, z)` — resolved from [`Building::footprint`] against the
    /// matched catalogue entry when it's [`FootprintSpec::FromBlueprint`].
    pub footprint: IVec2,
    /// [`Building::blueprint`]'s filename stem — the id this definition's
    /// shape is filed under in [`BuildingCatalogue`], resolved once here
    /// rather than every caller re-deriving it from
    /// [`Path::file_stem`](std::path::Path::file_stem). `city::ui::build_menu`
    /// (ticket 050, roadmap G1) is the reason this needs to be public:
    /// [`super::placement::PlacementSelection::catalogue_id`] and
    /// `city::commit`/`city::state::City` all key a placement by *this* id,
    /// not [`Self::id`] — clicking a build-menu entry has to select the
    /// catalogue id its blueprint actually loaded under, not the `.ron`
    /// filename that happened to describe it.
    pub catalogue_id: String,
}

/// Every building definition the game currently knows about, keyed by id.
#[derive(Resource, Default)]
pub struct BuildingDefinitions {
    entries: HashMap<String, LoadedBuilding>,
}

impl BuildingDefinitions {
    /// A set built from an explicit list rather than a directory scan — for
    /// tests in *other* modules, which need a definition set to look a
    /// placement up in but have no `.ron` files to load.
    /// [`build_definitions`] is the only production constructor.
    #[cfg(test)]
    pub fn from_entries(entries: Vec<LoadedBuilding>) -> Self {
        BuildingDefinitions { entries: entries.into_iter().map(|entry| (entry.id.clone(), entry)).collect() }
    }

    /// `city::ui::build_menu` (ticket 050, roadmap G1) is the real caller
    /// now — a missing-requirement id's display name, and the currently
    /// selected entry's own name.
    pub fn get(&self, id: &str) -> Option<&LoadedBuilding> {
        self.entries.get(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `city::ui::build_menu`'s real caller now: an empty set is a distinct
    /// panel message ("no buildings in assets/city/buildings"), not an empty
    /// window.
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
        // Ticket 078: a producer with nowhere to put its output stalls on its
        // first tick and never recovers, which reads in-game as a building
        // that silently doesn't work.
        if production.buffer_stacks == 0 {
            return Err(DefinitionError::ZeroBufferStacks);
        }
    }
    // Ticket 079: a warehouse that reaches nowhere or can carry nothing is a
    // mistake in the file, not a value the coverage pass should have to
    // defend against — the same call `road_definition` makes for
    // `travel_speed`/`capacity`.
    if let Some(warehouse) = &building.warehouse {
        if warehouse.radius_cells == 0 {
            return Err(DefinitionError::InvalidWarehouse("radius_cells must be > 0"));
        }
        if warehouse.concurrent_hauls == 0 {
            return Err(DefinitionError::InvalidWarehouse("concurrent_hauls must be > 0"));
        }
        if warehouse.handling_minutes < 0.0 {
            return Err(DefinitionError::InvalidWarehouse("handling_minutes must be >= 0"));
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
    // `blueprint_stem` is `Some` by construction: `catalogue_entry` above
    // only matched because it was.
    let catalogue_id = blueprint_stem.expect("catalogue_entry matched on this stem above").to_string();
    Ok(LoadedBuilding { id, path: path.to_path_buf(), building, footprint, catalogue_id })
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

    let (entries, mut requirement_errors) = resolve_requirements(entries);
    skipped.append(&mut requirement_errors);

    (BuildingDefinitions { entries }, skipped)
}

/// Roadmap C2, ticket 041: checks every surviving [`LoadedBuilding`]'s
/// `requires` against the whole set, and removes anything that can never
/// unlock — see the module docs for why this has to run after every file has
/// already loaded, and why it loops instead of making one pass.
///
/// Alternates a dangling-reference pass with a cycle pass until one of them
/// removes nothing. Each pass removes *everything* it finds before the other
/// runs again, rather than stopping at the first hit, so the final surviving
/// set doesn't depend on which order problems happen to be discovered in —
/// it's the maximal subset of `entries` whose `requires` graph, restricted
/// to that subset, resolves and has no cycle.
fn resolve_requirements(
    mut entries: HashMap<String, LoadedBuilding>,
) -> (HashMap<String, LoadedBuilding>, Vec<(PathBuf, DefinitionError)>) {
    let mut skipped = Vec::new();

    loop {
        let ids: HashSet<&str> = entries.keys().map(String::as_str).collect();
        let dangling: Vec<(String, String)> = entries
            .iter()
            .filter_map(|(id, entry)| {
                entry
                    .building
                    .requires
                    .iter()
                    .find(|req| !ids.contains(req.as_str()))
                    .map(|missing| (id.clone(), missing.clone()))
            })
            .collect();

        if !dangling.is_empty() {
            for (id, missing) in dangling {
                let entry = entries.remove(&id).expect("id came from entries.iter() above");
                skipped.push((entry.path, DefinitionError::DanglingRequirement(missing)));
            }
            continue;
        }

        let graph: HashMap<String, Vec<String>> = entries
            .iter()
            .map(|(id, entry)| (id.clone(), entry.building.requires.clone()))
            .collect();
        match find_cycle(&graph) {
            Some(cycle) => {
                for id in &cycle {
                    let entry = entries.remove(id).expect("cycle ids came from entries' own keys");
                    skipped.push((entry.path, DefinitionError::CyclicRequirement(cycle.clone())));
                }
            }
            None => break,
        }
    }

    (entries, skipped)
}

/// Finds one cycle in `graph` (an id -> its `requires` adjacency list), if
/// any exists, as the ids on the loop in order. Plain DFS with a recursion
/// stack: an edge into a node still on the stack (`Visiting`) closes a loop
/// back to that node. Iterates ids in sorted order so which cycle comes back
/// first, when several are disjoint, doesn't depend on `HashMap` iteration
/// order.
fn find_cycle(graph: &HashMap<String, Vec<String>>) -> Option<Vec<String>> {
    enum State {
        Visiting,
        Done,
    }

    fn visit(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        state: &mut HashMap<String, State>,
        stack: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        state.insert(node.to_string(), State::Visiting);
        stack.push(node.to_string());
        if let Some(requires) = graph.get(node) {
            for next in requires {
                match state.get(next.as_str()) {
                    Some(State::Visiting) => {
                        let start = stack.iter().position(|n| n == next).expect(
                            "a node in the Visiting state is still on the stack by construction",
                        );
                        return Some(stack[start..].to_vec());
                    }
                    Some(State::Done) => continue,
                    None => {
                        if let Some(cycle) = visit(next, graph, state, stack) {
                            return Some(cycle);
                        }
                    }
                }
            }
        }
        stack.pop();
        state.insert(node.to_string(), State::Done);
        None
    }

    let mut ids: Vec<&String> = graph.keys().collect();
    ids.sort();

    let mut state: HashMap<String, State> = HashMap::new();
    let mut stack: Vec<String> = Vec::new();
    for id in ids {
        if !matches!(state.get(id.as_str()), Some(State::Done)) {
            if let Some(cycle) = visit(id, graph, &mut state, &mut stack) {
                return Some(cycle);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `.ron` this repo actually ships parses into a [`Building`].
    /// Deliberately only the parse — cross-checking `blueprint` against the
    /// catalogue would need the `.nbt` files loaded, which is
    /// `load_definitions_dir`'s own job and a much heavier test. What this
    /// catches is the thing that actually goes wrong when a definition is
    /// hand-edited: a typo, a renamed field, a missing `Some`.
    #[test]
    fn the_shipped_definitions_all_parse() {
        for entry in fs::read_dir("assets/city/buildings").expect("the shipped definitions directory") {
            let path = entry.unwrap().path();
            if !is_ron_file(&path) {
                continue;
            }
            let text = fs::read_to_string(&path).unwrap();
            let parsed: Result<Building, _> = ron::de::from_str(&text);
            assert!(parsed.is_ok(), "{} does not parse: {}", path.display(), parsed.unwrap_err());
        }
    }
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
    fn a_valid_requires_edge_is_carried_and_survives() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("requires_valid");
        fs::write(dir.join("house01.ron"), VALID_RON).unwrap();
        fs::write(
            dir.join("carpenter.ron"),
            r#"Building(
                name: "Carpenter",
                blueprint: "house01.nbt",
                tier: 2,
                requires: ["house01"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(skipped.is_empty(), "{skipped:?}");
        let entry = definitions.get("carpenter").unwrap();
        assert_eq!(entry.building.requires, vec!["house01".to_string()]);

        fs::remove_dir_all(&dir).ok();
    }

    /// Roadmap C2 (ticket 041): a `requires` id nothing on disk claims can
    /// never unlock, so it's an error rather than data silently carried.
    #[test]
    fn a_dangling_requires_reference_is_skipped() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("dangling_requires");
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
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(
            &skipped[0].1,
            DefinitionError::DanglingRequirement(missing) if missing == "a_building_that_does_not_exist"
        ));

        fs::remove_dir_all(&dir).ok();
    }

    /// A two-building cycle: both are unwinnable, so both are removed and
    /// reported, not just whichever one the scan happens to reach first.
    #[test]
    fn a_requires_cycle_is_skipped_entirely() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("requires_cycle");
        fs::write(
            dir.join("a.ron"),
            r#"Building(
                name: "A",
                blueprint: "house01.nbt",
                tier: 1,
                requires: ["b"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();
        fs::write(
            dir.join("b.ron"),
            r#"Building(
                name: "B",
                blueprint: "house01.nbt",
                tier: 1,
                requires: ["a"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty(), "both ends of the cycle are unwinnable");
        assert_eq!(skipped.len(), 2);
        for (_, err) in &skipped {
            assert!(matches!(err, DefinitionError::CyclicRequirement(cycle) if cycle.len() == 2));
        }

        fs::remove_dir_all(&dir).ok();
    }

    /// A self-referential building is a one-node cycle.
    #[test]
    fn a_building_that_requires_itself_is_a_cycle() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("requires_self");
        fs::write(
            dir.join("a.ron"),
            r#"Building(
                name: "A",
                blueprint: "house01.nbt",
                tier: 1,
                requires: ["a"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty());
        assert_eq!(skipped.len(), 1);
        assert!(matches!(&skipped[0].1, DefinitionError::CyclicRequirement(cycle) if cycle == &vec!["a".to_string()]));

        fs::remove_dir_all(&dir).ok();
    }

    /// Cascading removal: `c` requires `b`, `b` requires `a`, and `a` is
    /// *itself* cyclic (self-referential). `a` gets removed first for its
    /// own problem; `b` only becomes dangling once `a` is gone, and `c` only
    /// becomes dangling once `b` is gone. A single non-looping pass would
    /// stop at `a` and leave `b`/`c` looking fine when neither can ever
    /// unlock — this is the case the module docs call out.
    #[test]
    fn removing_a_cyclic_entry_cascades_to_its_dependents() {
        let catalogue = catalogue_with_house01();
        let dir = temp_dir("cascade");
        fs::write(
            dir.join("a.ron"),
            r#"Building(
                name: "A",
                blueprint: "house01.nbt",
                tier: 1,
                requires: ["a"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();
        fs::write(
            dir.join("b.ron"),
            r#"Building(
                name: "B",
                blueprint: "house01.nbt",
                tier: 2,
                requires: ["a"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();
        fs::write(
            dir.join("c.ron"),
            r#"Building(
                name: "C",
                blueprint: "house01.nbt",
                tier: 3,
                requires: ["b"],
                integrity: Integrity(pristine_above: 0.9, ruined_below: 0.5),
            )"#,
        )
        .unwrap();

        let (definitions, skipped) = load_definitions_dir(&dir, &catalogue);
        assert!(definitions.is_empty(), "a, b and c are all unreachable once a is removed");
        assert_eq!(skipped.len(), 3);
        assert!(matches!(
            skipped.iter().find(|(_, err)| matches!(err, DefinitionError::CyclicRequirement(_))),
            Some(_)
        ));
        let dangling: Vec<&str> = skipped
            .iter()
            .filter_map(|(_, err)| match err {
                DefinitionError::DanglingRequirement(missing) => Some(missing.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(dangling.len(), 2, "b and c both cascade to dangling: {skipped:?}");

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
