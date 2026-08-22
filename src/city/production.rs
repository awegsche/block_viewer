//! Production (ticket 078, roadmap H2): a building with a `production` block
//! turns game time into materials, into a buffer of its own, and stops when
//! that buffer fills.
//!
//! [`super::definition::Production`] has been loaded and inert since ticket
//! 040. This is what finally reads it. Two things had to land first: ticket
//! 076, so a *placed* building can find its own `.ron`, and ticket 077, so
//! there is a game clock to integrate a `per_minute` rate against.
//!
//! ## The buffer, and why filling it stops the building
//!
//! Output does not go to the city's [`Stock`](super::inventory::Stock). It
//! goes into the producing building's own [`Producer::buffer`], and stays
//! there until ticket 080's haulage carries a stack of it to a warehouse.
//! The buffer has a cap — `buffer_stacks * economy.stack_size` — and when it
//! is reached the building goes [`ProducerState::BufferFull`] and *nothing*
//! accrues: not output, not the fractional carry, not the input debt.
//!
//! That stall is the mechanic, not a limitation. A player whose farm has
//! stopped has a visible, fixable problem — build a warehouse, or a road to
//! the one you have. A farm that went on producing into nowhere would lose
//! half its output to a mistake nobody could see.
//!
//! ## One accumulator per item, not one per building
//!
//! A rate of `12.0/min` at 60fps is 0.0033 items a frame, so something has to
//! carry the fraction. [`Producer::partial`] holds it per output item, always
//! in `[0, 1)`, rolling whole units into the buffer as they complete. Inputs
//! use the same shape ([`Producer::owed`]) for the same reason, one direction
//! over: a building accrues a *debt* and pays it out of the global stock a
//! whole unit at a time, because a stock keyed in `u64` cannot be charged
//! 0.0033 of a plank.
//!
//! ## Inputs are all-or-nothing, and a starved producer keeps its debt
//!
//! Every input a tick owes is taken in one go or none of it is: a building
//! that can pay for its planks but not its cobblestone must not eat the
//! planks and produce nothing for them. When it can't pay it goes
//! [`ProducerState::Starved`] and the debt stays owed — so the moment a haul
//! delivers what it needs it resumes from where it stopped, rather than
//! having quietly lost the partial run.
//!
//! ## No journal entry, ever
//!
//! Production moves materials without writing a single block, so it takes no
//! [`Ledger`](super::journal::Ledger) and no
//! [`JournalEntry`](super::journal::JournalEntry). Undo undoes *builds*, not
//! the passage of time; an "Undo" that clawed back a farm's output would be a
//! different mechanic wearing the same button.
//!
//! ## Persistence
//!
//! `<save>/citybuilder/logistics.ron`, its own file for the reason ticket 072
//! gave `stock.ron` one: folding it into `city.ron` would put a fractional
//! carry behind that file's strict version check and discard a whole city to
//! add it.
//!
//! Unlike `city.ron`, a version this build doesn't know is **logged and
//! started empty** rather than refused — and the justification is specific to
//! this file, not a precedent for the others. A producer's buffer regenerates
//! within minutes of play; a placement, or an as-built baseline (roadmap I1),
//! is gone for good. `super::mod`'s loader does the logging, the same shape
//! it already uses for a `city.ron` that fails to load.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::clock::GameClock;
use super::definition::{BuildingDefinitions, Production};
use super::economy::EconomyConfig;
use super::inventory::{Parcel, Stock};
use super::state::{BuildingId, City};

/// What one producing building is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProducerState {
    /// Producing — the ordinary case, and the one a building with no inputs
    /// is always in until its buffer fills.
    #[default]
    Running,
    /// The global stock can't pay this tick's input debt. Nothing accrues
    /// and nothing is taken; see the module docs.
    Starved,
    /// The buffer is at its cap and the building has stopped. Ticket 080's
    /// haulage is what clears it.
    BufferFull,
}

impl ProducerState {
    /// The city panel's own word for it.
    pub fn label(self) -> &'static str {
        match self {
            ProducerState::Running => "running",
            ProducerState::Starved => "starved",
            ProducerState::BufferFull => "buffer full",
        }
    }
}

/// One producing building's own state — everything about it that isn't in its
/// definition or its placement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Producer {
    /// Fractional carry per output item, always in `[0, 1)`. See the module
    /// docs.
    #[serde(default)]
    pub partial: BTreeMap<String, f32>,
    /// Whole items produced and waiting for a haul.
    #[serde(default)]
    pub buffer: Parcel,
    /// Fractional *input* debt, same shape as [`partial`](Self::partial).
    #[serde(default)]
    pub owed: BTreeMap<String, f32>,
    #[serde(default)]
    pub state: ProducerState,
    /// Which input it is short of, when [`state`](Self::state) is
    /// [`ProducerState::Starved`] — so the panel can say *what* to go and
    /// make rather than only that something is missing. Not persisted: it is
    /// re-derived on the first tick after a load.
    #[serde(skip)]
    pub short_of: Option<String>,
}

/// Every producing building's [`Producer`], keyed by placement.
///
/// A resource alongside [`City`] rather than a field inside it, the same
/// relationship [`super::journal::Journal`] already has: `City` is the
/// *placement* record — what is where — and a half-full grain buffer is not
/// a placement.
#[derive(Resource, Debug, Default)]
pub struct ProductionState {
    producers: HashMap<BuildingId, Producer>,
}

impl ProductionState {
    #[allow(dead_code)] // ticket 080's dispatch is the first non-test caller
    pub fn get(&self, id: BuildingId) -> Option<&Producer> {
        self.producers.get(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (BuildingId, &Producer)> {
        self.producers.iter().map(|(&id, producer)| (id, producer))
    }

    pub fn len(&self) -> usize {
        self.producers.len()
    }

    /// Inserts a [`Producer`] under an id the caller supplies — for
    /// [`load_logistics`], which is restoring state a previous session
    /// recorded, and for tests. Ordinary play never calls this: [`tick`]
    /// creates a producer the first time it sees a building that needs one.
    pub fn insert(&mut self, id: BuildingId, producer: Producer) {
        self.producers.insert(id, producer);
    }

    /// Drops every producer whose building `city` no longer holds — a
    /// demolition's cleanup, run on load and on every tick. A demolished
    /// building's buffer goes with it: the goods were sitting *in* the
    /// building, and 073's rule for a demolition is that it salvages
    /// nothing.
    fn retain_placed(&mut self, city: &City) {
        self.producers.retain(|&id, _| city.building(id).is_some());
    }
}

/// How many items `production` can hold before it stalls.
///
/// A `u64` product of two much smaller numbers, saturating anyway: a
/// definition file is hand-written, and a `buffer_stacks` typed with an extra
/// six zeroes should give a very patient farm rather than a panic.
pub fn buffer_capacity(production: &Production, economy: &EconomyConfig) -> u64 {
    u64::from(production.buffer_stacks).saturating_mul(economy.stack_size)
}

pub struct ProductionPlugin;

impl Plugin for ProductionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProductionState>().add_systems(Update, tick);
    }
}

/// One production tick for every placed building that has a `production`
/// block — see the module docs for the order (inputs, then outputs) and why
/// a full buffer stops both.
fn tick(
    clock: Res<GameClock>,
    city: Res<City>,
    definitions: Res<BuildingDefinitions>,
    economy: Res<EconomyConfig>,
    mut stock: ResMut<Stock>,
    mut production: ResMut<ProductionState>,
) {
    production.retain_placed(&city);

    let minutes = clock.delta_minutes();
    if minutes <= 0.0 {
        // Paused, or a frame the clock clamped to nothing. Returning here
        // rather than running the whole tick with a zero delta keeps a
        // paused game from repeatedly re-deriving every producer's state,
        // and is what makes "paused produces nothing" true by construction.
        return;
    }

    for (id, placed) in city.buildings() {
        let Some(definition) = placed.definition_id.as_deref().and_then(|id| definitions.get(id)) else { continue };
        let Some(spec) = &definition.building.production else { continue };

        let mut producer = production.producers.remove(&id).unwrap_or_default();
        advance_producer(&mut producer, spec, economy.as_ref(), &mut stock, minutes);
        production.producers.insert(id, producer);
    }
}

/// The tick for one producer, split out of [`tick`] so it can be tested
/// without an `App` — the same shape `city::grid::fit_footprint` and
/// `city::placement::resolve_placement` already use.
///
/// `minutes` is game time (ticket 077), never `Time::delta`.
pub fn advance_producer(
    producer: &mut Producer,
    spec: &Production,
    economy: &EconomyConfig,
    stock: &mut Stock,
    minutes: f32,
) {
    let capacity = buffer_capacity(spec, economy);

    // The cap first: a full buffer stops *everything*, so a stalled building
    // doesn't quietly go on eating its inputs.
    if producer.buffer.total() >= capacity {
        producer.state = ProducerState::BufferFull;
        producer.short_of = None;
        return;
    }

    if !pay_inputs(producer, spec, stock, minutes) {
        return;
    }

    for item in &spec.outputs {
        let carry = producer.partial.entry(item.item.clone()).or_insert(0.0);
        *carry += item.per_minute * minutes;
        let whole = carry.floor();
        if whole >= 1.0 {
            *carry -= whole;
            producer.buffer.add(&item.item, whole as u64);
        }
    }

    producer.state = ProducerState::Running;
    producer.short_of = None;
}

/// Accrues this tick's input debt and settles whatever whole units of it the
/// stock can cover — all inputs together or none at all. Returns whether the
/// producer may go on to produce; a `false` has already set
/// [`ProducerState::Starved`] and left the debt in place.
///
/// A producer with no inputs always returns `true` without touching the
/// stock.
fn pay_inputs(producer: &mut Producer, spec: &Production, stock: &mut Stock, minutes: f32) -> bool {
    if spec.inputs.is_empty() {
        return true;
    }

    for item in &spec.inputs {
        *producer.owed.entry(item.item.clone()).or_insert(0.0) += item.per_minute * minutes;
    }

    // What's owed in whole units right now. A debt still under 1 is carried,
    // not rounded up — charging for a plank a building has only used a
    // twentieth of would make every producer cost more than its own rate.
    let mut due = Parcel::default();
    for (item, owed) in producer.owed.iter() {
        let whole = owed.floor();
        if whole >= 1.0 {
            due.add(item, whole as u64);
        }
    }

    if due.is_empty() {
        return true;
    }

    if let Some(short) = due.iter().find(|(item, count)| stock.count(item) < *count).map(|(item, _)| item.to_string()) {
        producer.state = ProducerState::Starved;
        producer.short_of = Some(short);
        return false;
    }

    for (item, count) in due.iter() {
        stock.remove(item, count);
        if let Some(owed) = producer.owed.get_mut(item) {
            *owed -= count as f32;
        }
    }
    true
}

// -------------------------------------------------------------------------------------------------
// ---- persistence ---------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The `logistics.ron` schema version this build writes. See the module docs
/// for why a mismatch starts empty here rather than being refused the way
/// `city.ron`'s is.
pub const CURRENT_VERSION: u32 = 1;

const LOGISTICS_FILE: &str = "citybuilder/logistics.ron";

fn logistics_file_path(save_root: &Path) -> PathBuf {
    save_root.join(LOGISTICS_FILE)
}

/// `<save_root>/citybuilder/logistics.ron`, for `super::mod`'s log lines.
pub(crate) fn logistics_file_path_for_log(save_root: &Path) -> PathBuf {
    logistics_file_path(save_root)
}

#[derive(Debug, Serialize, Deserialize)]
struct LogisticsSave {
    version: u32,
    producers: Vec<SavedProducer>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedProducer {
    building: u64,
    producer: Producer,
}

#[derive(Debug)]
pub enum LogisticsError {
    Io(std::io::Error),
    Parse(String),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for LogisticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogisticsError::Io(err) => write!(f, "{err}"),
            LogisticsError::Parse(msg) => write!(f, "{msg}"),
            LogisticsError::UnsupportedVersion(version) => {
                write!(f, "logistics file is version {version}, this build reads version {CURRENT_VERSION}")
            }
        }
    }
}

impl std::error::Error for LogisticsError {}

pub fn save_logistics(production: &ProductionState, save_root: &Path) -> Result<(), LogisticsError> {
    let path = logistics_file_path(save_root);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(LogisticsError::Io)?;
    }

    let mut producers: Vec<SavedProducer> = production
        .producers
        .iter()
        .map(|(&id, producer)| SavedProducer { building: id.as_u64(), producer: producer.clone() })
        .collect();
    producers.sort_by_key(|entry| entry.building);

    let save = LogisticsSave { version: CURRENT_VERSION, producers };
    let text = ron::ser::to_string_pretty(&save, ron::ser::PrettyConfig::default())
        .map_err(|err| LogisticsError::Parse(err.to_string()))?;
    fs::write(&path, text).map_err(LogisticsError::Io)
}

/// Reads `logistics.ron` and drops every producer whose building `city` no
/// longer holds — a building demolished in a previous session leaves no
/// buffer behind. A missing file is an empty [`ProductionState`], the same
/// contract every other loader in this crate gives one.
pub fn load_logistics(save_root: &Path, city: &City) -> Result<ProductionState, LogisticsError> {
    let path = logistics_file_path(save_root);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(ProductionState::default()),
        Err(err) => return Err(LogisticsError::Io(err)),
    };

    let save: LogisticsSave = ron::de::from_str(&text).map_err(|err| LogisticsError::Parse(err.to_string()))?;
    if save.version != CURRENT_VERSION {
        return Err(LogisticsError::UnsupportedVersion(save.version));
    }

    let mut production = ProductionState::default();
    for entry in save.producers {
        production.insert(BuildingId::from_u64(entry.building), entry.producer);
    }
    production.retain_placed(city);
    Ok(production)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use crate::city::definition::ProductionItem;
    use bevy::math::{IVec2, IVec3};

    fn economy() -> EconomyConfig {
        EconomyConfig::default()
    }

    fn spec(outputs: &[(&str, f32)], inputs: &[(&str, f32)], buffer_stacks: u32) -> Production {
        Production {
            outputs: outputs
                .iter()
                .map(|(item, rate)| ProductionItem { item: item.to_string(), per_minute: *rate })
                .collect(),
            inputs: inputs
                .iter()
                .map(|(item, rate)| ProductionItem { item: item.to_string(), per_minute: *rate })
                .collect(),
            radius: None,
            buffer_stacks,
        }
    }

    /// A minute of a 12/min farm is 12 wheat, whether it arrives in one tick
    /// or sixty — the property the fractional carry exists for.
    #[test]
    fn a_rate_accumulates_to_the_same_total_however_it_is_ticked() {
        let spec = spec(&[("minecraft:wheat", 12.0)], &[], 4);
        let economy = economy();

        let mut one_tick = Producer::default();
        advance_producer(&mut one_tick, &spec, &economy, &mut Stock::default(), 1.0);

        let mut many_ticks = Producer::default();
        for _ in 0..60 {
            advance_producer(&mut many_ticks, &spec, &economy, &mut Stock::default(), 1.0 / 60.0);
        }

        assert_eq!(one_tick.buffer.get("minecraft:wheat"), 12);
        assert_eq!(many_ticks.buffer.get("minecraft:wheat"), 12, "the fractional carry must not lose whole items");
    }

    #[test]
    fn a_partial_item_stays_in_the_carry_rather_than_rounding_into_the_buffer() {
        let spec = spec(&[("minecraft:wheat", 12.0)], &[], 4);
        let mut producer = Producer::default();
        advance_producer(&mut producer, &spec, &economy(), &mut Stock::default(), 0.05); // 0.6 of an item

        assert_eq!(producer.buffer.get("minecraft:wheat"), 0);
        assert!(producer.partial["minecraft:wheat"] > 0.0);
    }

    #[test]
    fn the_buffer_fills_and_the_building_stops() {
        // One stack of 64, produced at 64/min: a minute fills it exactly.
        let spec = spec(&[("minecraft:wheat", 64.0)], &[], 1);
        let economy = economy();
        let mut producer = Producer::default();

        advance_producer(&mut producer, &spec, &economy, &mut Stock::default(), 1.0);
        assert_eq!(producer.buffer.get("minecraft:wheat"), 64);

        advance_producer(&mut producer, &spec, &economy, &mut Stock::default(), 1.0);
        assert_eq!(producer.state, ProducerState::BufferFull);
        assert_eq!(producer.buffer.get("minecraft:wheat"), 64, "a full buffer must not keep growing");
    }

    /// The module docs' "a full buffer stops *everything*": a stalled
    /// building must not go on eating inputs it can't turn into anything.
    #[test]
    fn a_full_buffer_stops_the_inputs_too() {
        let spec = spec(&[("minecraft:bread", 64.0)], &[("minecraft:wheat", 64.0)], 1);
        let economy = economy();
        let mut stock = Stock::default();
        stock.add("minecraft:wheat", 1000);

        advance_producer(&mut Producer::default(), &spec, &economy, &mut stock, 1.0);
        let after_first = stock.count("minecraft:wheat");

        let mut full = Producer::default();
        full.buffer.add("minecraft:bread", 64);
        advance_producer(&mut full, &spec, &economy, &mut stock, 1.0);

        assert_eq!(full.state, ProducerState::BufferFull);
        assert_eq!(stock.count("minecraft:wheat"), after_first, "a stalled producer takes nothing from the stock");
    }

    #[test]
    fn inputs_are_taken_from_the_stock_a_whole_unit_at_a_time() {
        let spec = spec(&[("minecraft:bread", 1.0)], &[("minecraft:wheat", 3.0)], 4);
        let economy = economy();
        let mut stock = Stock::default();
        stock.add("minecraft:wheat", 10);

        let mut producer = Producer::default();
        advance_producer(&mut producer, &spec, &economy, &mut stock, 1.0);

        assert_eq!(stock.count("minecraft:wheat"), 7, "one minute at 3/min costs 3 wheat");
        assert_eq!(producer.state, ProducerState::Running);
    }

    /// A tick that owes less than one whole unit charges nothing at all —
    /// rounding it up would make every producer cost more than its own rate.
    #[test]
    fn a_sub_unit_input_debt_is_carried_not_charged() {
        let spec = spec(&[("minecraft:bread", 1.0)], &[("minecraft:wheat", 3.0)], 4);
        let mut stock = Stock::default();
        stock.add("minecraft:wheat", 10);

        let mut producer = Producer::default();
        advance_producer(&mut producer, &spec, &economy(), &mut stock, 0.1); // owes 0.3

        assert_eq!(stock.count("minecraft:wheat"), 10);
        assert!(producer.owed["minecraft:wheat"] > 0.0);
    }

    /// All-or-nothing across inputs: a building short of one of two must not
    /// eat the other.
    #[test]
    fn a_producer_short_of_one_input_takes_neither() {
        let spec = spec(&[("minecraft:bread", 1.0)], &[("minecraft:wheat", 3.0), ("minecraft:coal", 1.0)], 4);
        let mut stock = Stock::default();
        stock.add("minecraft:wheat", 100);

        let mut producer = Producer::default();
        advance_producer(&mut producer, &spec, &economy(), &mut stock, 1.0);

        assert_eq!(producer.state, ProducerState::Starved);
        assert_eq!(producer.short_of.as_deref(), Some("minecraft:coal"));
        assert_eq!(stock.count("minecraft:wheat"), 100, "the payable input must not be taken either");
        assert_eq!(producer.buffer.get("minecraft:bread"), 0);
    }

    /// A starved producer keeps its debt, so it resumes from where it
    /// stopped once the missing material arrives rather than losing the run.
    #[test]
    fn a_starved_producer_resumes_from_its_debt_when_the_stock_arrives() {
        let spec = spec(&[("minecraft:bread", 1.0)], &[("minecraft:wheat", 3.0)], 4);
        let economy = economy();
        let mut stock = Stock::default();
        let mut producer = Producer::default();

        advance_producer(&mut producer, &spec, &economy, &mut stock, 1.0);
        assert_eq!(producer.state, ProducerState::Starved);
        assert!(producer.owed["minecraft:wheat"] >= 3.0);

        stock.add("minecraft:wheat", 3);
        advance_producer(&mut producer, &spec, &economy, &mut stock, 0.0001);

        assert_eq!(producer.state, ProducerState::Running);
        assert_eq!(stock.count("minecraft:wheat"), 0, "the carried debt is what gets paid");
    }

    #[test]
    fn a_producer_with_no_inputs_never_touches_the_stock() {
        let spec = spec(&[("minecraft:wheat", 12.0)], &[], 4);
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 5);

        advance_producer(&mut Producer::default(), &spec, &economy(), &mut stock, 1.0);
        assert_eq!(stock.count("minecraft:dirt"), 5);
    }

    // --- the system ---------------------------------------------------------

    fn tick_app() -> App {
        let mut app = App::new();
        app.init_resource::<City>()
            .init_resource::<BuildingDefinitions>()
            .init_resource::<EconomyConfig>()
            .init_resource::<Stock>()
            .insert_resource(GameClock { elapsed: Default::default(), delta: Default::default() })
            .add_plugins(ProductionPlugin);
        app
    }

    #[test]
    fn a_paused_clock_produces_nothing() {
        let mut app = tick_app();
        let id = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("farm", Some("farm".to_string()), IVec3::ZERO, Rotation::Deg0, IVec2::ONE)
            .unwrap();
        app.update();

        assert!(
            app.world().resource::<ProductionState>().get(id).is_none(),
            "a zero delta must not even create a producer"
        );
    }

    /// A building with no definition behind it (ticket 076's keyboard
    /// stand-in) has no game data, so it never becomes a producer.
    #[test]
    fn a_placement_with_no_definition_never_produces() {
        let mut app = tick_app();
        let id = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("farm", None, IVec3::ZERO, Rotation::Deg0, IVec2::ONE)
            .unwrap();
        app.world_mut().resource_mut::<GameClock>().delta = std::time::Duration::from_secs(1);
        app.update();

        assert!(app.world().resource::<ProductionState>().get(id).is_none());
    }

    /// A demolished building's buffer goes with it — the goods were sitting
    /// in the building, and ticket 073's rule is that a demolition salvages
    /// nothing.
    #[test]
    fn a_demolished_buildings_producer_is_dropped() {
        let mut app = tick_app();
        let id = app
            .world_mut()
            .resource_mut::<City>()
            .place_building("farm", Some("farm".to_string()), IVec3::ZERO, Rotation::Deg0, IVec2::ONE)
            .unwrap();
        app.world_mut().resource_mut::<ProductionState>().insert(id, Producer::default());

        app.world_mut().resource_mut::<City>().remove_building(id);
        app.world_mut().resource_mut::<GameClock>().delta = std::time::Duration::from_secs(1);
        app.update();

        assert!(app.world().resource::<ProductionState>().get(id).is_none());
    }

    // --- persistence --------------------------------------------------------

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("block_viewer_logistics_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn city_with_one_building() -> (City, BuildingId) {
        let mut city = City::default();
        let id = city
            .place_building("farm", Some("farm".to_string()), IVec3::ZERO, Rotation::Deg0, IVec2::ONE)
            .unwrap();
        (city, id)
    }

    #[test]
    fn a_missing_file_is_an_empty_state_not_an_error() {
        let dir = temp_dir("missing");
        let (city, _) = city_with_one_building();
        assert_eq!(load_logistics(&dir, &city).unwrap().len(), 0);
    }

    #[test]
    fn a_buffer_round_trips_with_its_carry_and_its_debt() {
        let dir = temp_dir("round_trip");
        let (city, id) = city_with_one_building();

        let mut producer = Producer::default();
        producer.buffer.add("minecraft:wheat", 37);
        producer.partial.insert("minecraft:wheat".to_string(), 0.5);
        producer.owed.insert("minecraft:water_bucket".to_string(), 0.25);
        producer.state = ProducerState::Starved;

        let mut production = ProductionState::default();
        production.insert(id, producer);
        save_logistics(&production, &dir).unwrap();

        let loaded = load_logistics(&dir, &city).unwrap();
        let restored = loaded.get(id).expect("the producer should round-trip under the same id");
        assert_eq!(restored.buffer.get("minecraft:wheat"), 37);
        assert_eq!(restored.partial["minecraft:wheat"], 0.5);
        assert_eq!(restored.owed["minecraft:water_bucket"], 0.25);
        assert_eq!(restored.state, ProducerState::Starved);
    }

    #[test]
    fn a_producer_whose_building_is_gone_is_dropped_on_load() {
        let dir = temp_dir("orphan");
        let (city, id) = city_with_one_building();

        let mut production = ProductionState::default();
        production.insert(id, Producer::default());
        production.insert(BuildingId::from_u64(999), Producer::default());
        save_logistics(&production, &dir).unwrap();

        let loaded = load_logistics(&dir, &city).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.get(BuildingId::from_u64(999)).is_none());
    }

    #[test]
    fn a_version_mismatch_is_reported_for_the_caller_to_start_empty_on() {
        let dir = temp_dir("version");
        fs::create_dir_all(dir.join("citybuilder")).unwrap();
        fs::write(dir.join(LOGISTICS_FILE), "(version: 999, producers: [])").unwrap();

        let (city, _) = city_with_one_building();
        assert!(matches!(load_logistics(&dir, &city), Err(LogisticsError::UnsupportedVersion(999))));
    }
}
