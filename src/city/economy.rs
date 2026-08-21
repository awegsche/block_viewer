//! The economy's tunable numbers (ticket 074): what a new city is founded
//! with, and which materials turn into which on the way to paying for a
//! building.
//!
//! Both live in `assets/city/economy.ron`. One file rather than two: they're
//! a dozen knobs a player is expected to hand-tune, and one file means one
//! loader and one line in the errors panel. [`super::drops`] stays separate
//! — a mapping over hundreds of *block* names is a different kind of thing
//! from a handful of numbers.
//!
//! ## The founding grant
//!
//! [`EconomyConfig::start_stock`] is handed out when a save has **no**
//! `stock.ron` at all — not when it has one that happens to be empty. A
//! player who spent everything hasn't founded a new city and mustn't be
//! refilled; see [`super::inventory::load_stock`], which returns `None` for
//! "no file" precisely so this distinction can be made. A file that fails to
//! *load* isn't granted either: there was a stock there, and a corrupt file
//! must not become free materials.
//!
//! ## Conversions cover a shortfall — they don't run on their own
//!
//! Nothing here converts the whole pile up front. [`plan_payment`] prices one
//! cost against one stock and answers with the conversions that cost needs,
//! in whole runs: needing 2 planks with one log in the pile converts the
//! whole log and leaves the other 2 planks behind. Three rules fall out of
//! that:
//!
//! - **Placement only.** A demolition's backfill and a terraform's fill
//!   *debit*, and debits clamp (ticket 073) — converting a player's logs to
//!   satisfy a clamp would be the game spending their materials to fill a
//!   hole. A cost is the one place a shortfall is worth solving instead of
//!   absorbing.
//! - **Chains work.** Covering sticks may need planks, which may need logs,
//!   so [`plan_payment`] recurses through the table.
//! - **Cycles can't hang it.** The recursion carries a depth cap
//!   ([`MAX_CONVERSION_DEPTH`]) and the set of items already being covered,
//!   so a table saying `A -> B` and `B -> A` gives up instead of looping.
//!   `from.item == to.item` is refused at load time as the degenerate case
//!   of the same thing.
//!
//! ## One planner, two callers
//!
//! `city::ui::build_menu` prices a row with [`plan_payment`] and
//! `city::commit` pays with it. That is deliberate: a menu that computed
//! affordability its own way would eventually disagree with the click, and a
//! row reading red above a placement that succeeds is worse than either
//! answer alone.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use bevy::prelude::Resource;
use serde::Deserialize;

use super::definition::Cost;
use super::inventory::{Parcel, Shortfall, Stock};

/// How many conversion steps deep [`plan_payment`] will chase one item —
/// logs to planks to sticks is two, and nothing sensible is much longer.
/// This is a backstop against a pathological table, not a design limit worth
/// tuning.
const MAX_CONVERSION_DEPTH: usize = 4;

/// Some number of one material — a conversion's input or output.
#[derive(Debug, Clone, Deserialize)]
pub struct ItemStack {
    pub item: String,
    /// Defaults to 1, so the common `(item: "minecraft:oak_log")` side of a
    /// conversion doesn't have to say so.
    #[serde(default = "one")]
    pub count: u32,
}

fn one() -> u32 {
    1
}

/// One trivial transformation: `from` becomes `to`, in whole runs.
///
/// "Trivial" is the player's judgement, not the game's — the shipped table
/// is vanilla crafting and smelting recipes with no ambiguity about the
/// result. Nothing here models a workbench, a furnace or fuel; the point is
/// that a city holding logs shouldn't be unable to build with planks.
#[derive(Debug, Clone, Deserialize)]
pub struct Conversion {
    pub from: ItemStack,
    pub to: ItemStack,
}

/// The file's on-disk shape — `start_stock` as a plain map so it reads like
/// the stock file it seeds.
#[derive(Debug, Default, Deserialize)]
struct EconomyFile {
    #[serde(default)]
    start_stock: HashMap<String, u64>,
    #[serde(default)]
    conversions: Vec<Conversion>,
}

/// The loaded economy knobs. [`Default`] is the "no `economy.ron`" config:
/// no grant, no conversions — exactly the game ticket 073 shipped.
#[derive(Resource, Debug, Default)]
pub struct EconomyConfig {
    /// What a city with no `stock.ron` is founded with.
    pub start_stock: Parcel,
    /// Applied by [`plan_payment`], in file order — the first conversion
    /// that can produce a short item wins, so a table listing a cheap route
    /// before an expensive one gets the cheap one.
    pub conversions: Vec<Conversion>,
}

/// Why `economy.ron` didn't load. A missing file is **not** one of these —
/// see [`load_economy`].
#[derive(Debug)]
pub enum EconomyError {
    Read(std::io::Error),
    Parse(String),
    /// A conversion whose input or output count is zero — it would either
    /// produce nothing or cost nothing, and the second is an infinite
    /// material source.
    ZeroCount(String),
    /// A conversion with a blank item name on either side.
    EmptyItem(String),
    /// A conversion from a material to itself. Harmless-looking, and the
    /// degenerate cycle: it can only ever be a mistake in the file.
    SelfConversion(String),
}

impl std::fmt::Display for EconomyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EconomyError::Read(err) => write!(f, "{err}"),
            EconomyError::Parse(msg) => write!(f, "{msg}"),
            EconomyError::ZeroCount(what) => write!(f, "conversion {what} has a count of 0"),
            EconomyError::EmptyItem(what) => write!(f, "conversion {what} has a blank item name"),
            EconomyError::SelfConversion(item) => write!(f, "{item} is listed as converting to itself"),
        }
    }
}

impl std::error::Error for EconomyError {}

/// `stone` -> `minecraft:stone`, the same shorthand [`super::drops`] accepts
/// in its own table and [`crate::blueprint::BlockState`]'s `FromStr` accepts
/// everywhere else.
fn namespaced(name: &str) -> String {
    let name = name.trim();
    if name.contains(':') {
        name.to_string()
    } else {
        format!("minecraft:{name}")
    }
}

/// Reads `path` into an [`EconomyConfig`]. A **missing file is
/// `Ok(EconomyConfig::default())`** — no grant and no conversions is a
/// working game, the same tolerance [`super::drops::load_drop_table`] gives
/// a missing drop table. Anything else is an error for the caller to show in
/// the errors panel; there's no per-entry recovery, because a table half of
/// whose rules applied would be harder to reason about than one that didn't
/// load.
pub fn load_economy(path: &Path) -> Result<EconomyConfig, EconomyError> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(EconomyConfig::default()),
        Err(err) => return Err(EconomyError::Read(err)),
    };

    let file: EconomyFile = ron::de::from_str(&text).map_err(|err| EconomyError::Parse(err.to_string()))?;

    let mut start_stock = Parcel::default();
    for (item, count) in file.start_stock {
        start_stock.add(&namespaced(&item), count);
    }

    let mut conversions = Vec::with_capacity(file.conversions.len());
    for conversion in file.conversions {
        let from = namespaced(&conversion.from.item);
        let to = namespaced(&conversion.to.item);
        let what = format!("{from} -> {to}");
        if conversion.from.item.trim().is_empty() || conversion.to.item.trim().is_empty() {
            return Err(EconomyError::EmptyItem(what));
        }
        if conversion.from.count == 0 || conversion.to.count == 0 {
            return Err(EconomyError::ZeroCount(what));
        }
        if from == to {
            return Err(EconomyError::SelfConversion(from));
        }
        conversions.push(Conversion {
            from: ItemStack { item: from, count: conversion.from.count },
            to: ItemStack { item: to, count: conversion.to.count },
        });
    }

    Ok(EconomyConfig { start_stock, conversions })
}

// -------------------------------------------------------------------------------------------------
// ---- pricing a cost ------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The conversions one payment needs: what they take out of the stock, and
/// what they put back in. Both empty when the stock already holds the cost
/// outright, which is the common case.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversionPlan {
    pub consumed: Parcel,
    pub produced: Parcel,
}

impl ConversionPlan {
    #[allow(dead_code)] // read by this module's tests; the callers apply the plan rather than asking
    pub fn is_empty(&self) -> bool {
        self.consumed.is_empty() && self.produced.is_empty()
    }
}

/// What it would take to pay a cost out of a stock.
///
/// `shortfall` is what's *still* missing once the conversions in `conversion`
/// have been applied — empty means affordable. Both halves come back
/// together on purpose: the build menu shows the shortfall, `city::commit`
/// applies the conversion, and neither has to re-derive the other's answer.
#[derive(Debug, Clone)]
pub struct Payment {
    pub conversion: ConversionPlan,
    pub shortfall: Shortfall,
}

impl Payment {
    pub fn affordable(&self) -> bool {
        self.shortfall.missing.is_empty()
    }
}

/// A stock's counts, plus whatever a half-planned conversion has already
/// moved — the scratch space [`plan_payment`] works in so nothing is
/// committed until the whole cost is known to be payable.
struct Ledger<'a> {
    stock: &'a Stock,
    delta: HashMap<String, i128>,
}

impl<'a> Ledger<'a> {
    fn new(stock: &'a Stock) -> Self {
        Self { stock, delta: HashMap::new() }
    }

    fn available(&self, item: &str) -> u64 {
        let base = i128::from(self.stock.count(item));
        let delta = self.delta.get(item).copied().unwrap_or(0);
        u64::try_from((base + delta).max(0)).unwrap_or(u64::MAX)
    }

    fn take(&mut self, item: &str, amount: u64) {
        *self.delta.entry(item.to_string()).or_insert(0) -= i128::from(amount);
    }

    fn give(&mut self, item: &str, amount: u64) {
        *self.delta.entry(item.to_string()).or_insert(0) += i128::from(amount);
    }
}

/// Prices `costs` against `stock`, converting through `conversions` wherever
/// the stock is short — see the module docs for what "converting" is allowed
/// to mean here.
///
/// Pure: nothing is mutated, and the returned [`ConversionPlan`] is what the
/// caller should apply if it decides to go ahead. That's what lets the build
/// menu ask the same question the commit answers without either of them
/// touching the stock.
pub fn plan_payment(stock: &Stock, costs: &[Cost], conversions: &[Conversion]) -> Payment {
    let wanted = Parcel::from_costs(costs);
    let mut ledger = Ledger::new(stock);
    let mut plan = ConversionPlan::default();
    let mut missing = Parcel::default();

    for (item, needed) in wanted.iter() {
        let held = ledger.available(item);
        if held >= needed {
            ledger.take(item, needed);
            continue;
        }

        let mut covering = Vec::new();
        cover(item, needed - held, conversions, &mut ledger, &mut plan, &mut covering, 0);

        // Whatever the conversions managed, the cost itself is charged
        // against what's now there; anything still missing is reported.
        let available = ledger.available(item);
        let taken = available.min(needed);
        ledger.take(item, taken);
        missing.add(item, needed - taken);
    }

    Payment { conversion: plan, shortfall: Shortfall { missing } }
}

/// Tries to make `amount` more of `item` appear in `ledger` by running
/// conversions, recording each run in `plan`.
///
/// `covering` is the chain of items currently being covered — a conversion
/// whose input is already in it would be a cycle, and is skipped. Together
/// with `depth` against [`MAX_CONVERSION_DEPTH`] that's what makes a
/// hand-written table unable to hang the game.
///
/// Best-effort: it produces as much as the table and the stock allow and
/// says nothing about whether that was enough — [`plan_payment`] compares
/// what came out against what was needed.
fn cover(
    item: &str,
    amount: u64,
    conversions: &[Conversion],
    ledger: &mut Ledger,
    plan: &mut ConversionPlan,
    covering: &mut Vec<String>,
    depth: usize,
) {
    if amount == 0 || depth >= MAX_CONVERSION_DEPTH || covering.iter().any(|held| held == item) {
        return;
    }
    covering.push(item.to_string());

    let mut still_needed = amount;
    for conversion in conversions.iter().filter(|c| c.to.item == item) {
        if still_needed == 0 {
            break;
        }

        // A run produces `to.count`; round up, since half a run isn't a
        // thing — converting one log for two planks leaves two planks over.
        let runs_wanted = still_needed.div_ceil(u64::from(conversion.to.count));

        // The input may itself be short — chase it one level further before
        // giving up on this route.
        let input_wanted = runs_wanted.saturating_mul(u64::from(conversion.from.count));
        let input_held = ledger.available(&conversion.from.item);
        if input_held < input_wanted {
            cover(
                &conversion.from.item,
                input_wanted - input_held,
                conversions,
                ledger,
                plan,
                covering,
                depth + 1,
            );
        }

        let runs = runs_wanted.min(ledger.available(&conversion.from.item) / u64::from(conversion.from.count));
        if runs == 0 {
            continue;
        }

        let consumed = runs * u64::from(conversion.from.count);
        let produced = runs * u64::from(conversion.to.count);
        ledger.take(&conversion.from.item, consumed);
        ledger.give(item, produced);
        plan.consumed.add(&conversion.from.item, consumed);
        plan.produced.add(item, produced);
        still_needed = still_needed.saturating_sub(produced);
    }

    covering.pop();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stock_with(items: &[(&str, u64)]) -> Stock {
        let mut stock = Stock::default();
        for &(item, count) in items {
            stock.add(item, count);
        }
        stock
    }

    fn cost(block: &str, count: u32) -> Cost {
        Cost { block: block.to_string(), count }
    }

    fn conversion(from: (&str, u32), to: (&str, u32)) -> Conversion {
        Conversion {
            from: ItemStack { item: from.0.to_string(), count: from.1 },
            to: ItemStack { item: to.0.to_string(), count: to.1 },
        }
    }

    fn logs_to_planks() -> Vec<Conversion> {
        vec![conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4))]
    }

    // --- pricing without conversions ---------------------------------------

    #[test]
    fn a_cost_the_stock_covers_needs_no_conversion() {
        let stock = stock_with(&[("minecraft:oak_planks", 40)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &logs_to_planks());

        assert!(payment.affordable());
        assert!(payment.conversion.is_empty(), "the planks were already there");
    }

    #[test]
    fn a_free_cost_is_always_affordable() {
        let payment = plan_payment(&Stock::default(), &[], &[]);
        assert!(payment.affordable());
        assert!(payment.conversion.is_empty());
    }

    #[test]
    fn with_no_conversions_a_shortfall_is_just_a_shortfall() {
        let stock = stock_with(&[("minecraft:oak_log", 40)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &[]);

        assert!(!payment.affordable());
        assert_eq!(payment.shortfall.missing.get("minecraft:oak_planks"), 40);
        assert!(payment.conversion.is_empty());
    }

    // --- one step ----------------------------------------------------------

    #[test]
    fn logs_are_converted_to_cover_a_plank_cost() {
        let stock = stock_with(&[("minecraft:oak_log", 20)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &logs_to_planks());

        assert!(payment.affordable());
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 10, "40 planks is ten logs");
        assert_eq!(payment.conversion.produced.get("minecraft:oak_planks"), 40);
    }

    #[test]
    fn a_partial_run_converts_the_whole_input_and_leaves_the_rest_in_stock() {
        // Two planks wanted, four to a log: the log goes whole and the other
        // two planks stay in the pile.
        let stock = stock_with(&[("minecraft:oak_log", 1)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 2)], &logs_to_planks());

        assert!(payment.affordable());
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 1);
        assert_eq!(payment.conversion.produced.get("minecraft:oak_planks"), 4);
    }

    #[test]
    fn conversion_tops_up_what_the_stock_already_holds() {
        let stock = stock_with(&[("minecraft:oak_planks", 38), ("minecraft:oak_log", 4)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &logs_to_planks());

        assert!(payment.affordable());
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 1, "only the two missing planks are made");
    }

    #[test]
    fn not_enough_input_converts_what_it_can_and_reports_the_rest() {
        let stock = stock_with(&[("minecraft:oak_log", 2)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &logs_to_planks());

        assert!(!payment.affordable());
        assert_eq!(payment.shortfall.missing.get("minecraft:oak_planks"), 32, "8 planks out of 2 logs");
    }

    #[test]
    fn an_unrelated_material_is_left_alone() {
        let stock = stock_with(&[("minecraft:oak_log", 20), ("minecraft:cobblestone", 5)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 4)], &logs_to_planks());

        assert!(payment.conversion.consumed.get("minecraft:cobblestone") == 0);
    }

    // --- chains and cycles -------------------------------------------------

    #[test]
    fn a_chain_is_followed_through() {
        // sticks <- planks <- logs, with only logs in the pile.
        let table = vec![
            conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4)),
            conversion(("minecraft:oak_planks", 2), ("minecraft:stick", 4)),
        ];
        let stock = stock_with(&[("minecraft:oak_log", 4)]);

        let payment = plan_payment(&stock, &[cost("minecraft:stick", 8)], &table);

        assert!(payment.affordable(), "{:?}", payment.shortfall.missing);
        assert_eq!(payment.conversion.produced.get("minecraft:stick"), 8);
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_planks"), 4);
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 1);
    }

    #[test]
    fn a_cycle_gives_up_instead_of_hanging() {
        // The pathological table the depth cap and the `covering` set exist
        // for. If this test hangs, they don't work.
        let table = vec![
            conversion(("minecraft:a", 1), ("minecraft:b", 1)),
            conversion(("minecraft:b", 1), ("minecraft:a", 1)),
        ];
        let payment = plan_payment(&Stock::default(), &[cost("minecraft:a", 5)], &table);

        assert!(!payment.affordable());
        assert_eq!(payment.shortfall.missing.get("minecraft:a"), 5);
    }

    #[test]
    fn two_costs_cannot_both_spend_the_same_log() {
        // The whole reason the planner works against a running ledger rather
        // than pricing each item independently.
        let table = vec![
            conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4)),
            conversion(("minecraft:oak_log", 1), ("minecraft:stick", 4)),
        ];
        let stock = stock_with(&[("minecraft:oak_log", 1)]);

        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 4), cost("minecraft:stick", 4)], &table);

        assert!(!payment.affordable(), "one log can't pay for both");
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 1);
    }

    // --- loading -----------------------------------------------------------

    fn config_from(text: &str) -> Result<EconomyConfig, EconomyError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nth = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("block_viewer_economy_{}_{nth}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("economy.ron");
        fs::write(&path, text).expect("write");
        let result = load_economy(&path);
        let _ = fs::remove_dir_all(&dir);
        result
    }

    #[test]
    fn a_missing_file_is_an_empty_config_not_an_error() {
        let path = std::env::temp_dir().join("block_viewer_economy_definitely_not_here.ron");
        let _ = fs::remove_file(&path);
        let config = load_economy(&path).expect("a missing file is not an error");
        assert!(config.start_stock.is_empty());
        assert!(config.conversions.is_empty());
    }

    #[test]
    fn a_config_loads_its_grant_and_its_table() {
        let config = config_from(
            r#"(
                start_stock: {"dirt": 64, "minecraft:oak_log": 16},
                conversions: [(from: (item: "oak_log"), to: (item: "oak_planks", count: 4))],
            )"#,
        )
        .expect("loads");

        assert_eq!(config.start_stock.get("minecraft:dirt"), 64, "bare names are namespaced");
        assert_eq!(config.start_stock.get("minecraft:oak_log"), 16);
        assert_eq!(config.conversions.len(), 1);
        assert_eq!(config.conversions[0].from.count, 1, "an unstated input count is one");
        assert_eq!(config.conversions[0].to.item, "minecraft:oak_planks");
    }

    #[test]
    fn a_zero_count_is_refused() {
        let err = config_from(r#"(conversions: [(from: (item: "a", count: 0), to: (item: "b"))])"#).expect_err("zero");
        assert!(matches!(err, EconomyError::ZeroCount(_)), "{err}");
    }

    #[test]
    fn converting_something_to_itself_is_refused() {
        let err = config_from(r#"(conversions: [(from: (item: "stone"), to: (item: "minecraft:stone"))])"#)
            .expect_err("self conversion");
        assert!(matches!(err, EconomyError::SelfConversion(_)), "{err}");
    }

    #[test]
    fn nonsense_is_a_parse_error_not_a_panic() {
        let err = config_from("not ron at all").expect_err("parse");
        assert!(matches!(err, EconomyError::Parse(_)), "{err}");
    }

    #[test]
    fn the_shipped_config_loads_and_makes_planks_out_of_logs() {
        // The real asset: a typo in `assets/city/economy.ron` should fail
        // here, not at the player's first placement.
        let config = load_economy(Path::new("assets/city/economy.ron")).expect("assets/city/economy.ron loads");
        assert!(!config.start_stock.is_empty(), "a new city is founded with something");

        let stock = stock_with(&[("minecraft:oak_log", 10)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &config.conversions);
        assert!(payment.affordable(), "the shipped table should turn logs into planks");
    }
}
