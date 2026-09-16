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
//! ## Farm-tile scaling (ticket 084)
//!
//! A building with a `farm` link (see [`super::definition::Farm`] and
//! [`super::farm`]) doesn't run at its listed rate outright: [`tick`] looks
//! up [`super::farm::FarmCoverage::tiles_near`] and multiplies **every**
//! rate — outputs *and* inputs alike — by `tiles / tiles_for_full_rate`,
//! clamped to `1.0`, before handing the (possibly scaled) spec to
//! [`advance_producer`]. Both scale together because a hub running at 40%
//! is running at 40% *throughput*, not producing at 40% while still paying
//! full price for the inputs that made it — see [`scale_production`] for the
//! full argument. Zero tiles is zero rate — not [`ProducerState::Starved`],
//! since nothing is owed and nothing is missing, the building just has
//! nothing to scale yet.
//!
//! ## Haulage (ticket 080)
//!
//! A buffer is emptied by a [`Shipment`]: one stack of one item, dispatched
//! to the warehouse ticket 079's [`Coverage`] says serves this producer, and
//! delivered `travel_minutes + handling_minutes` of game time later. What a
//! warehouse can have in flight at once is its `concurrent_hauls`, which is
//! the "transport per time" knob — one warehouse serving six farms delivers
//! them a stack at a time and falls behind.
//!
//! Three rules that are less obvious than they look:
//!
//! - **A stalled producer ships its largest partial stack**, below
//!   `stack_size`. Without this, a building whose outputs are *mixed* can
//!   fill its buffer to the cap without any one item ever reaching a full
//!   stack, and deadlock there forever.
//! - **A delivery the city has no room for blocks at the warehouse** holding
//!   its goods, rather than dropping them. That closes the loop the storage
//!   cap opens: stock full -> hauls block -> hauler slots stay occupied ->
//!   buffers fill -> producers stall, every step of it visible in the city
//!   panel and every step fixed by another warehouse.
//! - **One way only.** The cart coming back empty isn't modelled;
//!   `concurrent_hauls` is what stands in for its occupancy.
//!
//! `RoadType::capacity` is deliberately *not* read here. Congestion — several
//! hauls sharing a cell and slowing each other — is a real mechanic and a
//! different ticket; using a road's number as a per-warehouse limit would put
//! it in the wrong place and make the eventual real thing harder to add.
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
use super::definition::{BuildingDefinitions, Farm, Production};
use super::economy::EconomyConfig;
use super::farm::{FarmCoverage, FarmCoverageSet};
use super::inventory::{Parcel, Stock};
use super::state::{BuildingId, City};
use super::warehouse::{self, Coverage, CoverageSet, StorageCapacity};

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
    /// Ticket 086: a [`super::gatherer`] hub with nothing left inside its
    /// radius above its own ground level — every reachable block down to the
    /// floor is already gone. Distinct from [`Starved`](Self::Starved) (which
    /// names a missing *input*, not a fact about the ground) and from
    /// [`BufferFull`](Self::BufferFull) (which clears itself the moment a
    /// haul empties the buffer; this doesn't clear on its own at all — the
    /// site really is levelled). A building with a `production` block never
    /// reaches this state.
    Depleted,
}

impl ProducerState {
    /// The city panel's own word for it.
    pub fn label(self) -> &'static str {
        match self {
            ProducerState::Running => "running",
            ProducerState::Starved => "starved",
            ProducerState::BufferFull => "buffer full",
            ProducerState::Depleted => "site levelled",
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
    /// Fractional blocks owed to [`super::gatherer`]'s dig, same shape as
    /// [`partial`](Self::partial) and [`owed`](Self::owed) one direction
    /// over again: a `Gatherer::blocks_per_minute` rate is a count of
    /// *blocks*, not of one chosen item, so it can't share `partial`'s
    /// per-item map the way a producer's own output does.
    #[serde(default)]
    pub dig_carry: f32,
    #[serde(default)]
    pub state: ProducerState,
    /// Which input it is short of, when [`state`](Self::state) is
    /// [`ProducerState::Starved`] — so the panel can say *what* to go and
    /// make rather than only that something is missing. Not persisted: it is
    /// re-derived on the first tick after a load.
    #[serde(skip)]
    pub short_of: Option<String>,
}

/// One stack on its way from a producer to the warehouse serving it — ticket
/// 080. See the module docs for why it is one-way and why a full city blocks
/// it rather than losing it.
/// Not `Serialize`/`Deserialize` itself: [`BuildingId`] deliberately isn't
/// either, so every save file in this crate writes the bare `u64` through
/// `as_u64`/`from_u64` — [`SavedShipment`] is this type's mirror, the same
/// shape [`super::persistence::SavedBuilding`] has.
#[derive(Debug, Clone)]
pub struct Shipment {
    pub from: BuildingId,
    pub to: BuildingId,
    /// One stack of one item. A [`Parcel`] rather than a `(String, u64)`
    /// because that is what [`Stock::add_parcel_capped`] takes and what a
    /// blocked shipment hands back.
    pub parcel: Parcel,
    /// Game minutes still to travel. Counted down by [`tick`]; at zero the
    /// stack is delivered, or blocked if the city has no room.
    pub remaining: f32,
    /// Arrived, but the city is at capacity — see the module docs. Retried
    /// every tick, and it delivers the instant room appears.
    pub blocked: bool,
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
    /// In flight right now (ticket 080). Held here rather than in a resource
    /// of their own: dispatch reads a producer's buffer and writes a
    /// shipment in the same tick, and two resources mutating each other
    /// inside one system is a borrow fight for no benefit.
    shipments: Vec<Shipment>,
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

    /// Every stack currently on the road. The panel asks its narrower
    /// questions ([`hauls_to`](Self::hauls_to),
    /// [`blocked_hauls_to`](Self::blocked_hauls_to),
    /// [`is_hauling`](Self::is_hauling)) instead; this is the whole list, for
    /// tests and for whatever wants to draw the carts.
    #[allow(dead_code)]
    pub fn shipments(&self) -> &[Shipment] {
        &self.shipments
    }

    /// How many stacks are on their way to `warehouse` — what
    /// `concurrent_hauls` is checked against.
    pub fn hauls_to(&self, warehouse: BuildingId) -> usize {
        self.shipments.iter().filter(|shipment| shipment.to == warehouse).count()
    }

    /// How many of those have arrived and can't be unloaded, for the panel.
    pub fn blocked_hauls_to(&self, warehouse: BuildingId) -> usize {
        self.shipments.iter().filter(|shipment| shipment.to == warehouse && shipment.blocked).count()
    }

    /// Whether `producer` already has a stack on the road — one at a time
    /// per producer, so a single farm can't monopolise a warehouse's slots.
    pub fn is_hauling(&self, producer: BuildingId) -> bool {
        self.shipments.iter().any(|shipment| shipment.from == producer)
    }

    #[cfg(test)]
    pub fn push_shipment(&mut self, shipment: Shipment) {
        self.shipments.push(shipment);
    }

    /// Inserts a [`Producer`] under an id the caller supplies — for
    /// [`load_logistics`], which is restoring state a previous session
    /// recorded, and for tests. Ordinary play never calls this: [`tick`]
    /// creates a producer the first time it sees a building that needs one.
    pub fn insert(&mut self, id: BuildingId, producer: Producer) {
        self.producers.insert(id, producer);
    }

    /// `id`'s [`Producer`], creating a fresh (idle) one on first touch — the
    /// same "create on first tick" shape [`tick`]'s own loop already gives a
    /// building with a `production` block, exposed here so
    /// [`super::gatherer`]'s tick can share one buffer/state per building
    /// instance rather than keeping a second per-building map beside this
    /// one (which is also what lets [`super::warehouse::compute_coverage`]'s
    /// notion of "a producer" and haulage's dispatch loop reach a gatherer's
    /// buffer without knowing it's a gatherer at all).
    pub fn entry(&mut self, id: BuildingId) -> &mut Producer {
        self.producers.entry(id).or_default()
    }

    /// Drops every producer whose building `city` no longer holds — a
    /// demolition's cleanup, run on load and on every tick. A demolished
    /// building's buffer goes with it: the goods were sitting *in* the
    /// building, and 073's rule for a demolition is that it salvages
    /// nothing.
    fn retain_placed(&mut self, city: &City) {
        self.producers.retain(|&id, _| city.building(id).is_some());
        // A shipment whose producer or warehouse has been demolished goes
        // with it, goods and all — the warehouse it was going to no longer
        // exists, and there is nowhere for it to turn around to.
        self.shipments.retain(|shipment| city.building(shipment.from).is_some() && city.building(shipment.to).is_some());
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
        // After `CoverageSet`, so a haul is never dispatched against last
        // frame's coverage — a road built this frame should be usable this
        // frame, not next. After `FarmCoverageSet` for the same reason: a
        // tile placed this frame should count toward this frame's output.
        app.init_resource::<ProductionState>().add_systems(Update, tick.after(CoverageSet).after(FarmCoverageSet));
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
    coverage: Option<Res<Coverage>>,
    capacity: Option<Res<StorageCapacity>>,
    farm_coverage: Option<Res<FarmCoverage>>,
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

        let scaled = scale_production(spec, definition.building.farm.as_ref(), id, farm_coverage.as_deref());
        let mut producer = production.producers.remove(&id).unwrap_or_default();
        advance_producer(&mut producer, &scaled, economy.as_ref(), &mut stock, minutes);
        production.producers.insert(id, producer);
    }

    // Ticket 080. Deliveries before dispatch, so a slot freed by an arrival
    // is usable on the same tick rather than one later — with
    // `concurrent_hauls: 1` the other order would idle the cart for a frame
    // between every stack.
    let capacity = warehouse::storage_capacity(capacity.as_deref());
    deliver_arrivals(&mut production, &mut stock, capacity, minutes);
    if let Some(coverage) = coverage {
        dispatch_hauls(&mut production, &city, &definitions, &coverage, economy.stack_size);
    }
}

/// Counts every in-flight shipment down and unloads the ones that have
/// arrived, blocking (not dropping) whatever the city has no room for — see
/// the module docs.
fn deliver_arrivals(production: &mut ProductionState, stock: &mut Stock, capacity: u64, minutes: f32) {
    let mut arrived: Vec<usize> = Vec::new();
    for (index, shipment) in production.shipments.iter_mut().enumerate() {
        if !shipment.blocked {
            shipment.remaining -= minutes;
        }
        if shipment.remaining <= 0.0 {
            arrived.push(index);
        }
    }

    for &index in &arrived {
        let shipment = &mut production.shipments[index];
        let overflow = stock.add_parcel_capped(&shipment.parcel, capacity);
        shipment.blocked = !overflow.is_empty();
        shipment.parcel = overflow;
    }

    // Only the ones that fully unloaded leave the road. A blocked shipment
    // keeps its slot, which is exactly the back-pressure the storage cap is
    // there to create.
    production.shipments.retain(|shipment| !(shipment.remaining <= 0.0 && !shipment.blocked));
}

/// Dispatches at most one stack per producer per tick, subject to its
/// warehouse's `concurrent_hauls` — see the module docs for what "a stack"
/// means and why a stalled producer may ship a partial one.
fn dispatch_hauls(
    production: &mut ProductionState,
    city: &City,
    definitions: &BuildingDefinitions,
    coverage: &Coverage,
    stack_size: u64,
) {
    // Deterministic order: two producers competing for the last slot of a
    // tier-1 warehouse should resolve the same way every frame, not by
    // whichever the hash map happened to yield first.
    let mut candidates: Vec<BuildingId> = production.producers.keys().copied().collect();
    candidates.sort();

    for producer_id in candidates {
        if production.is_hauling(producer_id) {
            continue;
        }
        let Some(served) = coverage.served(producer_id) else { continue };
        let Some(spec) = warehouse::warehouse_of(city, definitions, served.warehouse) else { continue };
        if production.hauls_to(served.warehouse) >= spec.concurrent_hauls as usize {
            continue;
        }

        let Some(producer) = production.producers.get_mut(&producer_id) else { continue };
        let Some((item, count)) = ready_stack(producer, stack_size) else { continue };

        producer.buffer.remove(&item, count);
        let mut parcel = Parcel::default();
        parcel.add(&item, count);
        production.shipments.push(Shipment {
            from: producer_id,
            to: served.warehouse,
            parcel,
            remaining: served.travel_minutes + spec.handling_minutes,
            blocked: false,
        });
    }
}

/// Which stack this producer is ready to send, if any: a full one, or — once
/// it has stalled — its largest partial. See the module docs for why the
/// second case exists.
///
/// Ties between two items of equal size go to the lower id, so a mixed
/// producer's dispatch order doesn't wander.
fn ready_stack(producer: &Producer, stack_size: u64) -> Option<(String, u64)> {
    let largest = producer.buffer.iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))?;
    let (item, held) = (largest.0.to_string(), largest.1);

    if held >= stack_size {
        Some((item, stack_size))
    } else if producer.state == ProducerState::BufferFull && held > 0 {
        Some((item, held))
    } else {
        None
    }
}

/// Applies a `farm` link's tile-count scaling (ticket 084) to `spec` —
/// `spec.clone()` unscaled when `farm` is `None`, or `farm` names a hub with
/// no tiles in range yet and [`FarmCoverage`] is missing entirely (a minimal
/// test `App` that never adds [`super::farm::FarmPlugin`], the same tolerant
/// `Option<Res<..>>` shape [`storage_capacity`](warehouse::storage_capacity)
/// documents its own reason for).
///
/// Returns an owned [`Production`] rather than taking `spec` by value or
/// mutating it in place: `spec` is borrowed out of [`BuildingDefinitions`],
/// which every other building's tick this frame is still reading.
///
/// **Both outputs and inputs scale, by the same ratio** (ticket 084's own
/// design) — an under-tiled hub runs at a fraction of its whole recipe
/// rather than paying full price for inputs it's only using part of. This
/// composes with `advance_producer`'s own "a full buffer stops everything"
/// rule rather than fighting it: at ratio 0 both lists are 0/min, so a
/// tile-less hub touches neither the stock nor its own buffer, the same
/// "genuinely idle" state a `production: None` building is already in.
fn scale_production(spec: &Production, farm: Option<&Farm>, hub: BuildingId, coverage: Option<&FarmCoverage>) -> Production {
    let Some(farm) = farm else { return spec.clone() };

    let tiles = coverage.map(|coverage| coverage.tiles_near(hub)).unwrap_or(0);
    let ratio = (tiles as f32 / farm.tiles_for_full_rate.max(1) as f32).min(1.0);

    let mut scaled = spec.clone();
    for item in scaled.outputs.iter_mut().chain(scaled.inputs.iter_mut()) {
        item.per_minute *= ratio;
    }
    scaled
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

/// The `logistics.ron` schema version this build writes. Version 2 (ticket
/// 080) added the in-flight shipments beside version 1's producers. See the
/// module docs for why a mismatch starts empty here rather than being refused
/// the way `city.ron`'s is — which is also why this bump costs a player a
/// buffer and a cart rather than a city.
pub const CURRENT_VERSION: u32 = 2;

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
    #[serde(default)]
    shipments: Vec<SavedShipment>,
}

/// [`Shipment`] with its ids as the bare `u64`s every other save file in this
/// crate writes.
#[derive(Debug, Serialize, Deserialize)]
struct SavedShipment {
    from: u64,
    to: u64,
    parcel: Parcel,
    /// Minutes *remaining*, not elapsed — a stack does not teleport home
    /// because the player closed the window, and it does not evaporate
    /// either.
    remaining: f32,
    #[serde(default)]
    blocked: bool,
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

    let shipments: Vec<SavedShipment> = production
        .shipments
        .iter()
        .map(|shipment| SavedShipment {
            from: shipment.from.as_u64(),
            to: shipment.to.as_u64(),
            parcel: shipment.parcel.clone(),
            remaining: shipment.remaining,
            blocked: shipment.blocked,
        })
        .collect();

    let save = LogisticsSave { version: CURRENT_VERSION, producers, shipments };
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
    production.shipments = save
        .shipments
        .into_iter()
        .map(|saved| Shipment {
            from: BuildingId::from_u64(saved.from),
            to: BuildingId::from_u64(saved.to),
            parcel: saved.parcel,
            remaining: saved.remaining,
            blocked: saved.blocked,
        })
        .collect();
    // Drops both orphaned producers and shipments whose producer or
    // warehouse is gone — see `retain_placed`.
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

    // --- farm-tile scaling (ticket 084) -------------------------------------

    fn farm_link(radius_blocks: u32, tiles_for_full_rate: u32) -> Farm {
        Farm { tile: "tile".to_string(), radius_blocks, tiles_for_full_rate }
    }

    #[test]
    fn no_farm_link_leaves_the_spec_unscaled() {
        let spec = spec(&[("minecraft:oak_log", 8.0)], &[], 4);
        let scaled = scale_production(&spec, None, BuildingId::from_u64(0), None);
        assert_eq!(scaled.outputs[0].per_minute, 8.0);
    }

    #[test]
    fn zero_tiles_scales_output_to_zero() {
        let spec = spec(&[("minecraft:oak_log", 8.0)], &[], 4);
        let farm = farm_link(16, 3);
        let mut coverage = FarmCoverage::default();
        coverage.set_tiles_near(BuildingId::from_u64(0), 0);

        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), Some(&coverage));
        assert_eq!(scaled.outputs[0].per_minute, 0.0);
    }

    #[test]
    fn a_farm_link_with_no_coverage_resource_scales_to_zero() {
        // The tolerant `Option<Res<FarmCoverage>>` shape: a minimal test App
        // that never adds `FarmPlugin` must not panic or silently give full
        // output to a building that declares a farm link.
        let spec = spec(&[("minecraft:oak_log", 8.0)], &[], 4);
        let farm = farm_link(16, 3);
        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), None);
        assert_eq!(scaled.outputs[0].per_minute, 0.0);
    }

    #[test]
    fn a_partial_tile_count_scales_output_linearly() {
        let spec = spec(&[("minecraft:oak_log", 9.0)], &[], 4);
        let farm = farm_link(16, 3);
        let mut coverage = FarmCoverage::default();
        coverage.set_tiles_near(BuildingId::from_u64(0), 1);

        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), Some(&coverage));
        assert!((scaled.outputs[0].per_minute - 3.0).abs() < 1e-5, "1 of 3 tiles is a third of the rate");
    }

    #[test]
    fn a_full_tile_count_scales_output_to_the_full_rate() {
        let spec = spec(&[("minecraft:oak_log", 8.0)], &[], 4);
        let farm = farm_link(16, 3);
        let mut coverage = FarmCoverage::default();
        coverage.set_tiles_near(BuildingId::from_u64(0), 3);

        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), Some(&coverage));
        assert_eq!(scaled.outputs[0].per_minute, 8.0);
    }

    #[test]
    fn tiles_beyond_the_cap_do_not_scale_output_past_the_full_rate() {
        let spec = spec(&[("minecraft:oak_log", 8.0)], &[], 4);
        let farm = farm_link(16, 3);
        let mut coverage = FarmCoverage::default();
        coverage.set_tiles_near(BuildingId::from_u64(0), 50);

        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), Some(&coverage));
        assert_eq!(scaled.outputs[0].per_minute, 8.0, "more tiles than needed must not exceed 100%");
    }

    #[test]
    fn scaling_applies_the_same_ratio_to_inputs_as_outputs() {
        let spec = spec(&[("minecraft:bread", 1.0)], &[("minecraft:wheat", 3.0)], 4);
        let farm = farm_link(16, 3);
        let mut coverage = FarmCoverage::default();
        coverage.set_tiles_near(BuildingId::from_u64(0), 1);

        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), Some(&coverage));
        assert!(
            (scaled.inputs[0].per_minute - 1.0).abs() < 1e-5,
            "1 of 3 tiles scales the input rate the same way it scales output"
        );
    }

    #[test]
    fn zero_tiles_scales_inputs_to_zero_too() {
        let spec = spec(&[("minecraft:bread", 1.0)], &[("minecraft:wheat", 3.0)], 4);
        let farm = farm_link(16, 3);
        let mut coverage = FarmCoverage::default();
        coverage.set_tiles_near(BuildingId::from_u64(0), 0);

        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), Some(&coverage));
        assert_eq!(scaled.inputs[0].per_minute, 0.0, "an idle hub touches neither the stock nor its own buffer");
    }

    /// End-to-end through [`advance_producer`]: a hub with no tiles nearby
    /// accrues nothing, even though its `spec` still lists a positive rate.
    #[test]
    fn a_scaled_zero_rate_advances_the_producer_to_nothing() {
        let spec = spec(&[("minecraft:oak_log", 8.0)], &[], 4);
        let farm = farm_link(16, 3);
        let scaled = scale_production(&spec, Some(&farm), BuildingId::from_u64(0), None);

        let mut producer = Producer::default();
        advance_producer(&mut producer, &scaled, &economy(), &mut Stock::default(), 1.0);
        assert_eq!(producer.buffer.get("minecraft:oak_log"), 0);
        assert_eq!(producer.state, ProducerState::Running, "zero tiles is not a stall, it's nothing to scale yet");
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

    // --- haulage (ticket 080) -----------------------------------------------

    use crate::city::definition::{Building, Category, FootprintSpec, Integrity, LoadedBuilding, Warehouse};
    use crate::city::road::RoadPieceVariant;
    use crate::city::road_definition::RoadTypes;
    use crate::city::state::ROAD_CELL_SIZE;
    use crate::city::warehouse::compute_coverage;

    fn definition(id: &str, warehouse: Option<Warehouse>, production: Option<Production>) -> LoadedBuilding {
        LoadedBuilding {
            id: id.to_string(),
            path: PathBuf::from(format!("{id}.ron")),
            building: Building {
                name: id.to_string(),
                blueprint: format!("{id}.nbt"),
                tier: 1,
                requires: Vec::new(),
                footprint: FootprintSpec::FromBlueprint,
                production,
                cost: Vec::new(),
                warehouse,
                farm: None,
                gatherer: None,
                category: Category::Production,
                ground_level: 0,
                integrity: Integrity { pristine_above: 0.95, ruined_below: 0.6 },
            },
            footprint: IVec2::ONE,
            catalogue_id: id.to_string(),
        }
    }

    /// One warehouse with `concurrent_hauls` slots, and a farm — the two
    /// definitions every haulage test below shares.
    fn haul_definitions(concurrent_hauls: u32) -> BuildingDefinitions {
        BuildingDefinitions::from_entries(vec![
            definition(
                "warehouse01",
                Some(Warehouse { radius_cells: 8, concurrent_hauls, handling_minutes: 0.0, storage: 100_000 }),
                None,
            ),
            definition("farm01", None, Some(spec(&[("minecraft:wheat", 60.0)], &[], 2))),
        ])
    }

    /// A warehouse at road cell 0 and `farms` farms strung along one road,
    /// each one cell further out.
    fn road_town(city: &mut City, farms: usize) -> (BuildingId, Vec<BuildingId>) {
        for x in 0..=(farms as i32) {
            city.add_road_cell(IVec2::new(x, 0), "dirt", 64, None, RoadPieceVariant::Surface).unwrap();
        }
        let at = |cell: i32| IVec3::new(cell * ROAD_CELL_SIZE, 64, -1);
        let warehouse = city
            .place_building("warehouse01", Some("warehouse01".to_string()), at(0), Rotation::Deg0, IVec2::ONE)
            .unwrap();
        let farms = (1..=farms as i32)
            .map(|cell| {
                city.place_building("farm01", Some("farm01".to_string()), at(cell), Rotation::Deg0, IVec2::ONE).unwrap()
            })
            .collect();
        (warehouse, farms)
    }

    /// Dispatches against a real [`Coverage`] rather than a hand-built one,
    /// so these exercise the same road-distance answer the game uses.
    fn dispatch(production: &mut ProductionState, city: &City, definitions: &BuildingDefinitions, stack_size: u64) {
        let coverage = compute_coverage(city, definitions, &RoadTypes::default());
        dispatch_hauls(production, city, definitions, &coverage, stack_size);
    }

    fn producer_holding(item: &str, count: u64, state: ProducerState) -> Producer {
        let mut producer = Producer::default();
        producer.buffer.add(item, count);
        producer.state = state;
        producer
    }

    #[test]
    fn a_full_stack_is_dispatched_to_the_serving_warehouse() {
        let mut city = City::default();
        let (warehouse, farms) = road_town(&mut city, 1);
        let definitions = haul_definitions(2);

        let mut production = ProductionState::default();
        production.insert(farms[0], producer_holding("minecraft:wheat", 70, ProducerState::Running));
        dispatch(&mut production, &city, &definitions, 64);

        assert_eq!(production.shipments().len(), 1);
        let shipment = &production.shipments()[0];
        assert_eq!(shipment.to, warehouse);
        assert_eq!(shipment.parcel.get("minecraft:wheat"), 64, "exactly one stack leaves");
        assert_eq!(production.get(farms[0]).unwrap().buffer.get("minecraft:wheat"), 6, "the rest waits");
        assert!(shipment.remaining > 0.0, "one road cell away is not instant");
    }

    #[test]
    fn less_than_a_stack_waits_while_the_producer_is_still_running() {
        let mut city = City::default();
        let (_, farms) = road_town(&mut city, 1);
        let definitions = haul_definitions(2);

        let mut production = ProductionState::default();
        production.insert(farms[0], producer_holding("minecraft:wheat", 63, ProducerState::Running));
        dispatch(&mut production, &city, &definitions, 64);

        assert!(production.shipments().is_empty());
    }

    /// The module docs' deadlock guard: a stalled producer whose outputs are
    /// mixed may never reach a full stack of any one of them.
    #[test]
    fn a_stalled_producer_ships_its_largest_partial_stack() {
        let mut city = City::default();
        let (_, farms) = road_town(&mut city, 1);
        let definitions = haul_definitions(2);

        let mut producer = producer_holding("minecraft:wheat", 40, ProducerState::BufferFull);
        producer.buffer.add("minecraft:carrot", 24);

        let mut production = ProductionState::default();
        production.insert(farms[0], producer);
        dispatch(&mut production, &city, &definitions, 64);

        assert_eq!(production.shipments().len(), 1);
        assert_eq!(production.shipments()[0].parcel.get("minecraft:wheat"), 40, "the largest partial goes first");
    }

    /// The throughput knob: a tier-1 warehouse with one cart serving two
    /// farms sends one stack, not two.
    #[test]
    fn concurrent_hauls_limits_what_a_warehouse_has_in_flight() {
        let mut city = City::default();
        let (warehouse, farms) = road_town(&mut city, 2);
        let definitions = haul_definitions(1);

        let mut production = ProductionState::default();
        for &farm in &farms {
            production.insert(farm, producer_holding("minecraft:wheat", 128, ProducerState::Running));
        }
        dispatch(&mut production, &city, &definitions, 64);

        assert_eq!(production.hauls_to(warehouse), 1, "one cart, one stack on the road");
    }

    #[test]
    fn a_producer_already_hauling_does_not_send_a_second_stack() {
        let mut city = City::default();
        let (_, farms) = road_town(&mut city, 1);
        let definitions = haul_definitions(4);

        let mut production = ProductionState::default();
        production.insert(farms[0], producer_holding("minecraft:wheat", 256, ProducerState::Running));
        dispatch(&mut production, &city, &definitions, 64);
        dispatch(&mut production, &city, &definitions, 64);

        assert_eq!(production.shipments().len(), 1);
    }

    /// An unserved producer — no road, or no warehouse in range — sends
    /// nothing however full it is. Ticket 079's whole point, from this side.
    #[test]
    fn an_unserved_producer_dispatches_nothing() {
        let mut city = City::default();
        road_town(&mut city, 1);
        let definitions = haul_definitions(2);
        let stranded = city
            .place_building("farm01", Some("farm01".to_string()), IVec3::new(900, 64, 900), Rotation::Deg0, IVec2::ONE)
            .unwrap();

        let mut production = ProductionState::default();
        production.insert(stranded, producer_holding("minecraft:wheat", 640, ProducerState::BufferFull));
        dispatch(&mut production, &city, &definitions, 64);

        assert!(production.shipments().is_empty());
    }

    // --- arrival ------------------------------------------------------------

    fn in_flight(from: BuildingId, to: BuildingId, count: u64, remaining: f32) -> Shipment {
        let mut parcel = Parcel::default();
        parcel.add("minecraft:wheat", count);
        Shipment { from, to, parcel, remaining, blocked: false }
    }

    #[test]
    fn a_shipment_arrives_when_its_time_runs_out() {
        let mut production = ProductionState::default();
        production.push_shipment(in_flight(BuildingId::from_u64(1), BuildingId::from_u64(0), 64, 2.0));
        let mut stock = Stock::default();

        deliver_arrivals(&mut production, &mut stock, u64::MAX, 1.0);
        assert_eq!(stock.count("minecraft:wheat"), 0, "still on the road");
        assert_eq!(production.shipments().len(), 1);

        deliver_arrivals(&mut production, &mut stock, u64::MAX, 1.5);
        assert_eq!(stock.count("minecraft:wheat"), 64);
        assert!(production.shipments().is_empty());
    }

    /// The loop the storage cap opens: a delivery the city has no room for
    /// waits at the warehouse holding its goods rather than dropping them,
    /// and keeps its hauler slot while it does.
    #[test]
    fn a_delivery_into_a_full_city_blocks_rather_than_losing_the_stack() {
        let warehouse = BuildingId::from_u64(0);
        let mut production = ProductionState::default();
        production.push_shipment(in_flight(BuildingId::from_u64(1), warehouse, 64, 0.5));

        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 100);

        deliver_arrivals(&mut production, &mut stock, 100, 1.0);
        assert_eq!(stock.count("minecraft:wheat"), 0);
        assert_eq!(production.shipments().len(), 1, "the shipment keeps its slot");
        assert!(production.shipments()[0].blocked);
        assert_eq!(production.blocked_hauls_to(warehouse), 1);
        assert_eq!(production.shipments()[0].parcel.get("minecraft:wheat"), 64, "and it still holds the goods");
    }

    #[test]
    fn a_blocked_delivery_lands_the_instant_room_appears() {
        let mut production = ProductionState::default();
        production.push_shipment(in_flight(BuildingId::from_u64(1), BuildingId::from_u64(0), 64, 0.0));
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 100);

        deliver_arrivals(&mut production, &mut stock, 100, 0.1);
        assert!(production.shipments()[0].blocked);

        stock.remove("minecraft:dirt", 100);
        deliver_arrivals(&mut production, &mut stock, 100, 0.1);

        assert_eq!(stock.count("minecraft:wheat"), 64);
        assert!(production.shipments().is_empty());
    }

    /// A partial unload keeps only the remainder — handing the whole stack
    /// back would duplicate materials on the next tick.
    #[test]
    fn a_partial_unload_keeps_only_what_did_not_fit() {
        let mut production = ProductionState::default();
        production.push_shipment(in_flight(BuildingId::from_u64(1), BuildingId::from_u64(0), 64, 0.0));
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 80);

        deliver_arrivals(&mut production, &mut stock, 100, 0.1);

        assert_eq!(stock.count("minecraft:wheat"), 20);
        assert_eq!(production.shipments()[0].parcel.get("minecraft:wheat"), 44);
        assert!(production.shipments()[0].blocked);
    }

    /// A blocked shipment's clock does not keep running — it has arrived, and
    /// a negative `remaining` would be meaningless in the panel.
    #[test]
    fn a_blocked_shipment_stops_counting_down() {
        let mut production = ProductionState::default();
        production.push_shipment(in_flight(BuildingId::from_u64(1), BuildingId::from_u64(0), 64, 0.0));
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 100);

        deliver_arrivals(&mut production, &mut stock, 100, 0.1);
        let after_block = production.shipments()[0].remaining;
        deliver_arrivals(&mut production, &mut stock, 100, 5.0);

        assert_eq!(production.shipments()[0].remaining, after_block);
    }

    /// A shipment whose warehouse was demolished mid-flight goes with it —
    /// there is nowhere left for it to arrive.
    #[test]
    fn a_shipment_to_a_demolished_warehouse_is_dropped() {
        let mut city = City::default();
        let (warehouse, farms) = road_town(&mut city, 1);

        let mut production = ProductionState::default();
        production.push_shipment(in_flight(farms[0], warehouse, 64, 1.0));
        city.remove_building(warehouse);
        production.retain_placed(&city);

        assert!(production.shipments().is_empty());
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
