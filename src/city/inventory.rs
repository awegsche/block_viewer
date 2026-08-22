//! The global material stock (ticket 072, roadmap H2): one city-wide pile
//! of materials, what goes in it, and how it reaches disk.
//!
//! ## Materials are Minecraft item ids
//!
//! [`Stock`] is keyed by `"minecraft:oak_planks"`, not by an invented
//! `"wood"`. That is the whole vocabulary decision of iteration 2's economy,
//! and three things fall out of it:
//!
//! - [`super::definition::Cost`] already spells its `block` field that way,
//!   so a building's cost needed no schema change to become spendable.
//! - The blocks a placement clears out of the terrain are already named in
//!   the same alphabet — [`super::drops`] is the only translation step, and
//!   it exists to say "stone gives cobblestone", not to bridge two
//!   vocabularies.
//! - Later production chains spell their goods the same way, so a sawmill's
//!   output and a house's cost are comparable without a lookup table.
//!
//! ## Whole units only
//!
//! Counts are `u64`. Production rates are per-minute floats, and a stock
//! holding 0.4 of an oak plank would make every number the player reads a
//! rounding artefact; the fractional accumulator belongs to the producing
//! building (a later ticket), and what reaches this pile is whole units.
//!
//! Two asymmetric operations, deliberately:
//!
//! - [`Stock::spend`] is **all-or-nothing**. A half-paid building must not
//!   exist, so a cost that can't be met removes nothing at all and reports
//!   the [`Shortfall`].
//! - [`Stock::remove`] **clamps at zero**. Ticket 073's debits (backfilling
//!   a demolished building's hole, undoing a placement's yield) are
//!   settlements of a world-state change that has already happened or is
//!   about to; a negative stock would be a debt no mechanic in this game can
//!   discharge, and refusing to demolish something because the player is
//!   short of dirt would leave the city state and the world unable to agree.
//!
//! No entry is ever stored at zero — [`Stock::add`] of nothing and a removal
//! down to nothing both leave the key absent, so [`Stock::iter`] never has
//! to filter and the saved file never accumulates junk.
//!
//! ## Why its own file
//!
//! [`save_stock`] writes `<save>/citybuilder/stock.ron`, not a field on
//! `city.ron`. That file is at version 6 with an equality check on load
//! ([`super::persistence`]'s own "no quiet defaults" note), so folding the
//! stock into it would refuse — and therefore discard — every city that
//! exists today, purely to add an empty ledger to it. A second file with its
//! own version is the shape `journal.ron` already established, and here
//! "there is no stock file yet" really does mean an empty stock rather than
//! a defaulted field whose true value is unknown.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

use super::definition::Cost;

// -------------------------------------------------------------------------------------------------
// ---- a parcel: some quantity of some materials ---------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A bundle of materials — item id to count, never zero.
///
/// One type for four jobs, because they are all "some quantity of some
/// materials" and nothing is gained by giving each its own newtype: what a
/// [`Stock::spend`] took, what clearing a patch of terrain yielded
/// ([`super::drops::DropTable::parcel_for`]), what a journal entry moved in
/// either direction (ticket 073's ledger), and what a hauler is carrying
/// between a production building and a warehouse (ticket 074+).
///
/// `BTreeMap`, not `HashMap`: every consumer either displays this or writes
/// it to a RON file, and both want a stable order.
/// `#[serde(transparent)]`: a parcel is written as the bare map it is
/// (`{"minecraft:dirt": 12}`) rather than nested under a field name nobody
/// reading `journal.ron` by hand wants to see.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Parcel {
    items: BTreeMap<String, u64>,
}

impl Parcel {
    /// Adds `amount` of `item`, saturating rather than wrapping — a count
    /// that has reached `u64::MAX` is already meaningless, and wrapping to
    /// zero would turn "unimaginably rich" into "bankrupt".
    pub fn add(&mut self, item: &str, amount: u64) {
        if amount == 0 {
            return;
        }
        let entry = self.items.entry(item.to_string()).or_insert(0);
        *entry = entry.saturating_add(amount);
    }

    /// Read by tests and by ticket 074's shipments; the panels and the write
    /// paths all go through [`iter`](Self::iter) or [`total`](Self::total).
    #[allow(dead_code)]
    pub fn get(&self, item: &str) -> u64 {
        self.items.get(item).copied().unwrap_or(0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.items.iter().map(|(item, &count)| (item.as_str(), count))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// How many *distinct* materials — not how many units; see
    /// [`total`](Self::total) for that.
    #[allow(dead_code)] // same "tests and 074" note `get` above carries
    pub fn distinct(&self) -> usize {
        self.items.len()
    }

    /// Every unit in the parcel added together, saturating like
    /// [`add`](Self::add).
    pub fn total(&self) -> u64 {
        self.items.values().fold(0u64, |sum, &count| sum.saturating_add(count))
    }

    /// Folds `other` into this parcel — used where one action's materials
    /// arrive in two pieces (ticket 074: a placement's cost *plus* what its
    /// conversions consumed both count as debited).
    /// Removes up to `amount` of `item`, clamped at zero, and returns how
    /// much was actually removed — [`Stock::remove`]'s shape, one level down.
    /// Ticket 080's dispatch takes a stack out of a producer's buffer this
    /// way.
    pub fn remove(&mut self, item: &str, amount: u64) -> u64 {
        let Some(held) = self.items.get_mut(item) else { return 0 };
        let taken = amount.min(*held);
        *held -= taken;
        if *held == 0 {
            self.items.remove(item);
        }
        taken
    }

    pub fn add_all(&mut self, other: &Parcel) {
        for (item, count) in other.iter() {
            self.add(item, count);
        }
    }

    /// A cost list as a parcel — the same summing an all-or-nothing
    /// [`Stock::spend`] has to do anyway, so a definition that lists the
    /// same block twice (`20 planks` and `20 planks` rather than `40`) costs
    /// forty rather than checking twenty twice and paying it once.
    pub fn from_costs(costs: &[Cost]) -> Parcel {
        let mut parcel = Parcel::default();
        for cost in costs {
            parcel.add(&cost.block, u64::from(cost.count));
        }
        parcel
    }
}

/// What a [`Stock::spend`] was short of — the *missing* amounts, not the
/// requested ones, so a message can read "needs 12 more oak_planks" without
/// the caller re-deriving the difference.
/// `12x dirt, 3x cobblestone` — what a parcel reads as in a log line or a
/// panel, in the ascending item order [`Parcel::iter`] gives, with the short
/// names every player-facing string in this crate uses. Empty renders as
/// `nothing`, for the same reason [`Shortfall`]'s does.
impl std::fmt::Display for Parcel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "nothing");
        }
        let mut first = true;
        for (item, count) in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{count}x {}", short_name(item))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shortfall {
    pub missing: Parcel,
}

impl std::fmt::Display for Shortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (item, count) in self.missing.iter() {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{count} more {}", short_name(item))?;
        }
        if first {
            // Unreachable through `Stock::spend`, which only builds a
            // `Shortfall` when something is actually missing — but a
            // `Display` that renders an empty string is worse than one that
            // says so.
            write!(f, "nothing")?;
        }
        Ok(())
    }
}

impl std::error::Error for Shortfall {}

/// `minecraft:oak_planks` -> `oak_planks`, for anything a player reads.
/// Duplicated from `city::ui::build_menu`'s own `short_name` rather than
/// shared: that one trims a *building definition's* display name, this one a
/// namespaced id, and they only look alike.
pub fn short_name(item: &str) -> &str {
    item.split_once(':').map(|(_, rest)| rest).unwrap_or(item)
}

// -------------------------------------------------------------------------------------------------
// ---- the stock itself ----------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The city's single global pile of materials. There is no per-warehouse
/// storage: warehouses (ticket 074+) decide *whether* a production building
/// can deliver here and how long it takes, not where the goods physically
/// live.
#[derive(Resource, Debug, Clone, Default, PartialEq, Eq)]
pub struct Stock {
    items: BTreeMap<String, u64>,
}

impl Stock {
    pub fn count(&self, item: &str) -> u64 {
        self.items.get(item).copied().unwrap_or(0)
    }

    /// Saturating, and a no-op for `amount == 0` — see the module docs' "no
    /// entry is ever stored at zero".
    pub fn add(&mut self, item: &str, amount: u64) {
        if amount == 0 {
            return;
        }
        let entry = self.items.entry(item.to_string()).or_insert(0);
        *entry = entry.saturating_add(amount);
    }

    pub fn add_parcel(&mut self, parcel: &Parcel) {
        for (item, count) in parcel.iter() {
            self.add(item, count);
        }
    }

    /// Everything currently held, across every material — the number
    /// [`add_parcel_capped`](Self::add_parcel_capped) measures against a
    /// warehouse capacity.
    pub fn total(&self) -> u64 {
        self.items.values().copied().fold(0, u64::saturating_add)
    }

    /// [`add_parcel`](Self::add_parcel) against a ceiling (ticket 079): adds
    /// what fits under `capacity` and **returns the overflow**, in the
    /// ascending item order [`Parcel::iter`] gives.
    ///
    /// The overflow is the caller's to deal with, never silently dropped
    /// here. Ticket 080's haulage holds it at the warehouse and retries;
    /// every other caller reports how much was lost. That is the whole
    /// reason this returns a [`Parcel`] rather than a `bool`.
    ///
    /// [`add`](Self::add)/[`add_parcel`](Self::add_parcel) stay uncapped as
    /// the primitive: a failed placement's refund is settling a spend that
    /// already fit, and bouncing it would take a player's materials for a
    /// building they never got.
    pub fn add_parcel_capped(&mut self, parcel: &Parcel, capacity: u64) -> Parcel {
        let mut room = capacity.saturating_sub(self.total());
        let mut overflow = Parcel::default();
        for (item, count) in parcel.iter() {
            let fits = count.min(room);
            self.add(item, fits);
            room -= fits;
            overflow.add(item, count - fits);
        }
        overflow
    }

    /// Removes up to `amount` of `item`, **clamped at zero**, and returns
    /// how much was actually removed — which is the caller's record of what
    /// it managed to settle (ticket 073 journals exactly that, so an undo of
    /// a clamped debit doesn't hand back materials that were never taken).
    pub fn remove(&mut self, item: &str, amount: u64) -> u64 {
        let Some(held) = self.items.get_mut(item) else { return 0 };
        let taken = amount.min(*held);
        *held -= taken;
        if *held == 0 {
            self.items.remove(item);
        }
        taken
    }

    /// [`remove`](Self::remove) over a whole parcel; the returned parcel is
    /// what was actually taken, which is `parcel` itself unless something
    /// was short.
    pub fn remove_parcel(&mut self, parcel: &Parcel) -> Parcel {
        let mut taken = Parcel::default();
        for (item, count) in parcel.iter() {
            taken.add(item, self.remove(item, count));
        }
        taken
    }

    /// Whether [`spend`](Self::spend) would succeed, without spending.
    ///
    /// The conversion-blind question: since ticket 074 both real pricing
    /// callers go through [`super::economy::plan_payment`] instead, which
    /// answers the same question *after* converting what the table allows.
    /// This stays as the primitive underneath it — and as the one to ask
    /// when there is no table in hand.
    #[allow(dead_code)]
    pub fn can_afford(&self, costs: &[Cost]) -> bool {
        self.shortfall(costs).missing.is_empty()
    }

    /// What `costs` is short by, given what's held. Empty when affordable.
    pub fn shortfall(&self, costs: &[Cost]) -> Shortfall {
        let mut missing = Parcel::default();
        for (item, needed) in Parcel::from_costs(costs).iter() {
            missing.add(item, needed.saturating_sub(self.count(item)));
        }
        Shortfall { missing }
    }

    /// All-or-nothing: removes the whole of `costs` and returns the
    /// [`Parcel`] it took, or removes **nothing** and reports the
    /// [`Shortfall`]. The returned parcel is what a rollback refunds — see
    /// the module docs.
    pub fn spend(&mut self, costs: &[Cost]) -> Result<Parcel, Shortfall> {
        let wanted = Parcel::from_costs(costs);
        let shortfall = self.shortfall(costs);
        if !shortfall.missing.is_empty() {
            return Err(shortfall);
        }
        Ok(self.remove_parcel(&wanted))
    }

    /// Every held material and its count, ascending by id — the order
    /// `BTreeMap` gives, which is what the city panel lists.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.items.iter().map(|(item, &count)| (item.as_str(), count))
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// How many distinct materials are held. The city panel's summary line;
    /// `iter().count()` for anyone who'd rather.
    pub fn distinct(&self) -> usize {
        self.items.len()
    }
}

// -------------------------------------------------------------------------------------------------
// ---- persistence ---------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The `stock.ron` schema version this build writes and reads. Version 1 is
/// the first; a mismatch is refused rather than guessed at, the same
/// contract [`super::persistence`] and [`super::journal`] use.
pub const CURRENT_VERSION: u32 = 1;

/// Where [`save_stock`]/[`load_stock`] look, relative to a save's root —
/// alongside `city.ron` and `journal.ron` in the same `citybuilder`
/// directory.
const STOCK_FILE: &str = "citybuilder/stock.ron";

fn stock_file_path(save_root: &Path) -> PathBuf {
    save_root.join(STOCK_FILE)
}

/// `<save_root>/citybuilder/stock.ron`, for [`super::run`]'s own log line.
pub(crate) fn stock_file_path_for_log(save_root: &Path) -> PathBuf {
    stock_file_path(save_root)
}

#[derive(Debug, Serialize, Deserialize)]
struct SavedStock {
    version: u32,
    items: BTreeMap<String, u64>,
}

/// Why [`save_stock`] or [`load_stock`] failed. No `Corrupt` variant: a
/// stock is a flat map with no derived state to disagree with itself, the
/// same reason [`super::journal::JournalError`] has none.
#[derive(Debug)]
pub enum StockError {
    Io(std::io::Error),
    Parse(String),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for StockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StockError::Io(err) => write!(f, "{err}"),
            StockError::Parse(msg) => write!(f, "{msg}"),
            StockError::UnsupportedVersion(version) => {
                write!(f, "stock file is version {version}, this build reads version {CURRENT_VERSION}")
            }
        }
    }
}

impl std::error::Error for StockError {}

/// Writes `stock` to `<save_root>/citybuilder/stock.ron`, creating the
/// `citybuilder` directory if it doesn't exist yet (shared with
/// [`super::persistence::save_city`] and [`super::journal::save_journal`]).
pub fn save_stock(stock: &Stock, save_root: &Path) -> Result<(), StockError> {
    let path = stock_file_path(save_root);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(StockError::Io)?;
    }

    let save = SavedStock {
        version: CURRENT_VERSION,
        items: stock.items.clone(),
    };
    let text = ron::ser::to_string_pretty(&save, ron::ser::PrettyConfig::default())
        .map_err(|err| StockError::Parse(err.to_string()))?;
    fs::write(&path, text).map_err(StockError::Io)
}

/// Reads `<save_root>/citybuilder/stock.ron`. **`Ok(None)` when there is no
/// such file** — not an empty [`Stock`], because ticket 074's founding grant
/// turns on exactly that distinction: a save that has never had a stock is a
/// new city and gets the grant, while one whose player spent everything has
/// an empty stock file and must not be refilled.
///
/// Zero counts in the file are dropped rather than loaded: nothing this
/// module writes can produce one, so a `0` is a hand edit, and carrying it
/// into memory would break the "no entry is ever stored at zero" invariant
/// the rest of the type relies on.
pub fn load_stock(save_root: &Path) -> Result<Option<Stock>, StockError> {
    let path = stock_file_path(save_root);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(StockError::Io(err)),
    };

    let save: SavedStock = ron::de::from_str(&text).map_err(|err| StockError::Parse(err.to_string()))?;
    if save.version != CURRENT_VERSION {
        return Err(StockError::UnsupportedVersion(save.version));
    }

    let mut stock = Stock::default();
    for (item, count) in save.items {
        stock.add(&item, count);
    }
    Ok(Some(stock))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- the storage cap (ticket 079) ---------------------------------------

    #[test]
    fn total_is_every_material_added_up() {
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 12);
        stock.add("minecraft:stone", 30);
        assert_eq!(stock.total(), 42);
    }

    #[test]
    fn a_capped_add_takes_what_fits_and_hands_back_the_rest() {
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 90);

        let mut parcel = Parcel::default();
        parcel.add("minecraft:stone", 20);
        let overflow = stock.add_parcel_capped(&parcel, 100);

        assert_eq!(stock.count("minecraft:stone"), 10);
        assert_eq!(stock.total(), 100);
        assert_eq!(overflow.get("minecraft:stone"), 10);
    }

    #[test]
    fn a_capped_add_into_a_full_stock_stores_nothing() {
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 100);

        let mut parcel = Parcel::default();
        parcel.add("minecraft:stone", 5);
        let overflow = stock.add_parcel_capped(&parcel, 100);

        assert_eq!(stock.count("minecraft:stone"), 0);
        assert_eq!(overflow.get("minecraft:stone"), 5);
    }

    /// A stock already over capacity — the player built a warehouse and then
    /// demolished it — accepts nothing more, and does not underflow working
    /// out how much room it has.
    #[test]
    fn a_stock_already_over_capacity_accepts_nothing_without_underflowing() {
        let mut stock = Stock::default();
        stock.add("minecraft:dirt", 500);

        let mut parcel = Parcel::default();
        parcel.add("minecraft:stone", 5);
        let overflow = stock.add_parcel_capped(&parcel, 100);

        assert_eq!(stock.count("minecraft:dirt"), 500, "what is already held is never confiscated");
        assert_eq!(overflow.get("minecraft:stone"), 5);
    }

    #[test]
    fn a_parcel_displays_as_short_names_in_id_order() {
        let mut parcel = Parcel::default();
        parcel.add("minecraft:stone", 3);
        parcel.add("minecraft:dirt", 12);
        assert_eq!(parcel.to_string(), "12x dirt, 3x stone");
        assert_eq!(Parcel::default().to_string(), "nothing");
    }

    fn cost(block: &str, count: u32) -> Cost {
        Cost { block: block.to_string(), count }
    }

    fn stock_with(items: &[(&str, u64)]) -> Stock {
        let mut stock = Stock::default();
        for &(item, count) in items {
            stock.add(item, count);
        }
        stock
    }

    // --- Parcel ------------------------------------------------------------

    #[test]
    fn adding_zero_stores_nothing() {
        let mut parcel = Parcel::default();
        parcel.add("minecraft:dirt", 0);
        assert!(parcel.is_empty());
    }

    #[test]
    fn adding_the_same_item_twice_sums_it() {
        let mut parcel = Parcel::default();
        parcel.add("minecraft:dirt", 3);
        parcel.add("minecraft:dirt", 4);
        assert_eq!(parcel.get("minecraft:dirt"), 7);
        assert_eq!(parcel.distinct(), 1);
        assert_eq!(parcel.total(), 7);
    }

    #[test]
    fn a_cost_listing_the_same_block_twice_sums_it() {
        let parcel = Parcel::from_costs(&[cost("minecraft:oak_planks", 20), cost("minecraft:oak_planks", 20)]);
        assert_eq!(parcel.get("minecraft:oak_planks"), 40);
    }

    #[test]
    fn parcels_iterate_in_id_order() {
        let mut parcel = Parcel::default();
        parcel.add("minecraft:stone", 1);
        parcel.add("minecraft:dirt", 1);
        let ids: Vec<&str> = parcel.iter().map(|(item, _)| item).collect();
        assert_eq!(ids, vec!["minecraft:dirt", "minecraft:stone"]);
    }

    // --- add / remove ------------------------------------------------------

    #[test]
    fn an_absent_item_counts_zero() {
        assert_eq!(Stock::default().count("minecraft:dirt"), 0);
    }

    #[test]
    fn removing_more_than_is_held_clamps_and_reports_what_it_took() {
        let mut stock = stock_with(&[("minecraft:dirt", 5)]);
        assert_eq!(stock.remove("minecraft:dirt", 9), 5);
        assert_eq!(stock.count("minecraft:dirt"), 0);
    }

    #[test]
    fn removing_everything_drops_the_key_rather_than_holding_a_zero() {
        let mut stock = stock_with(&[("minecraft:dirt", 5)]);
        stock.remove("minecraft:dirt", 5);
        assert!(stock.is_empty());
        assert_eq!(stock.distinct(), 0);
    }

    #[test]
    fn removing_from_an_absent_item_is_zero_not_a_panic() {
        let mut stock = Stock::default();
        assert_eq!(stock.remove("minecraft:dirt", 3), 0);
    }

    #[test]
    fn remove_parcel_returns_only_what_it_could_take() {
        let mut stock = stock_with(&[("minecraft:dirt", 2)]);
        let mut wanted = Parcel::default();
        wanted.add("minecraft:dirt", 5);
        wanted.add("minecraft:stone", 1);

        let taken = stock.remove_parcel(&wanted);

        assert_eq!(taken.get("minecraft:dirt"), 2);
        assert_eq!(taken.get("minecraft:stone"), 0);
        assert!(stock.is_empty());
    }

    // --- spend / shortfall -------------------------------------------------

    #[test]
    fn a_free_building_is_always_affordable() {
        assert!(Stock::default().can_afford(&[]));
    }

    #[test]
    fn spending_what_is_held_removes_exactly_it() {
        let mut stock = stock_with(&[("minecraft:oak_planks", 50), ("minecraft:cobblestone", 20)]);
        let spent = stock.spend(&[cost("minecraft:oak_planks", 40)]).expect("affordable");

        assert_eq!(spent.get("minecraft:oak_planks"), 40);
        assert_eq!(stock.count("minecraft:oak_planks"), 10);
        assert_eq!(stock.count("minecraft:cobblestone"), 20);
    }

    #[test]
    fn an_unaffordable_cost_removes_nothing_at_all() {
        // The all-or-nothing half: the planks are there, the cobblestone
        // isn't, and a half-paid building must not exist.
        let mut stock = stock_with(&[("minecraft:oak_planks", 40)]);
        let costs = [cost("minecraft:oak_planks", 40), cost("minecraft:cobblestone", 20)];

        let err = stock.spend(&costs).expect_err("cobblestone is missing");

        assert_eq!(err.missing.get("minecraft:cobblestone"), 20);
        assert_eq!(err.missing.get("minecraft:oak_planks"), 0);
        assert_eq!(stock.count("minecraft:oak_planks"), 40, "nothing may be taken on a refusal");
    }

    #[test]
    fn a_shortfall_reads_as_what_is_still_needed() {
        let stock = stock_with(&[("minecraft:oak_planks", 28)]);
        let shortfall = stock.shortfall(&[cost("minecraft:oak_planks", 40)]);
        assert_eq!(shortfall.to_string(), "12 more oak_planks");
    }

    #[test]
    fn an_exactly_affordable_cost_is_affordable() {
        let stock = stock_with(&[("minecraft:oak_planks", 40)]);
        assert!(stock.can_afford(&[cost("minecraft:oak_planks", 40)]));
    }

    #[test]
    fn short_names_drop_the_namespace() {
        assert_eq!(short_name("minecraft:oak_planks"), "oak_planks");
        assert_eq!(short_name("oak_planks"), "oak_planks");
    }

    // --- persistence -------------------------------------------------------

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("block_viewer_stock_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn a_missing_file_is_none_rather_than_an_empty_stock() {
        // The distinction ticket 074's founding grant turns on.
        let dir = temp_dir("missing");
        assert!(load_stock(&dir).expect("a missing file is not an error").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_saved_stock_loads_as_an_empty_stock_not_as_absent() {
        let dir = temp_dir("empty");
        save_stock(&Stock::default(), &dir).expect("save");

        let loaded = load_stock(&dir).expect("load").expect("the file is there, however empty");

        assert!(loaded.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stock_round_trips_through_disk() {
        let dir = temp_dir("round_trip");
        let stock = stock_with(&[("minecraft:oak_planks", 40), ("minecraft:dirt", 7)]);

        save_stock(&stock, &dir).expect("save");
        let loaded = load_stock(&dir).expect("load").expect("the file was just written");

        assert_eq!(loaded, stock);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_future_version_is_refused_rather_than_read() {
        let dir = temp_dir("version");
        let path = stock_file_path(&dir);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, format!("(version: {}, items: {{}})", CURRENT_VERSION + 1)).unwrap();

        let err = load_stock(&dir).expect_err("a newer file must not be read as this one");
        assert!(matches!(err, StockError::UnsupportedVersion(_)));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_hand_written_zero_is_dropped_on_load() {
        let dir = temp_dir("zero");
        let path = stock_file_path(&dir);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, format!("(version: {CURRENT_VERSION}, items: {{\"minecraft:dirt\": 0}})")).unwrap();

        let stock = load_stock(&dir).expect("load").expect("the file is there");
        assert!(stock.is_empty(), "a zero count is not an entry");
        let _ = fs::remove_dir_all(&dir);
    }
}
