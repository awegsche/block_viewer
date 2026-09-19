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
//! ## Groups: the same material under a different name (ticket 075)
//!
//! `birch_log` and `oak_log` are not a ratio, they're a synonym — and
//! spelling twelve plank types as mutually convertible would be 132 ordered
//! pairs in a hand-edited file. So a second list sits next to `conversions`:
//! [`EconomyConfig::groups`], where **every member converts to every other
//! 1:1**. Ratios stay in `conversions`; a ratio means the two aren't the
//! same material after all.
//!
//! Groups are expanded at *lookup* ([`routes_to`]), not at load. A
//! forty-eight-member group of every log, bark block and stripped variant
//! would otherwise become 2,256 `Conversion` structs, and `build_menu`
//! prices every visible row every frame. And expanded **once per chain**:
//! a group is a clique, so once a short item has offered its group-mates
//! as routes, chasing one of those mates only looks for *conversions* into
//! it, never its own group-mates again (ticket 125 — walking the clique
//! once per path through it was ~48⁴ calls per short log cost, a second
//! and a half of build menu per frame).
//!
//! This is what keeps the drop table honest: the pile goes on saying
//! `birch_log`, because that's what the world gave, and only the *payment*
//! stops caring. Collapsing wood to one id in `drops.ron` would have made a
//! building that genuinely wants birch impossible to express.
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
    /// Ticket 078. Defaulted rather than required, so an `economy.ron`
    /// written before production existed keeps working.
    #[serde(default = "default_stack_size")]
    stack_size: u64,
    /// Ticket 079 — the storage a city has before it builds a single
    /// warehouse. Defaulted for the same reason `stack_size` is.
    #[serde(default = "default_base_storage")]
    base_storage: u64,
    /// Ticket 128 — blocks per minute of game time dug out of a placement's
    /// volume (a building's or a road cell's) before its blueprint is
    /// written. Defaulted for the same reason `stack_size`/`base_storage`
    /// are: an `economy.ron` written before site clearing existed keeps
    /// working.
    #[serde(default = "default_site_clearing_rate")]
    site_clearing_blocks_per_minute: f32,
    #[serde(default)]
    start_stock: HashMap<String, u64>,
    #[serde(default)]
    conversions: Vec<Conversion>,
    /// Ticket 075 — each inner list is a set of materials that are the same
    /// material under different names.
    #[serde(default)]
    interchangeable: Vec<Vec<String>>,
}

/// Minecraft's own stack, and the unit a haul carries (ticket 078/080). A
/// tunable rather than a constant because it is a *balance* number — how big
/// a batch a farm accumulates before a cart comes for it — that happens to
/// coincide with the game's own.
fn default_stack_size() -> u64 {
    64
}

/// What a city can hold with no warehouse at all. Not zero, and not a
/// rounding of the founding grant: a fresh save has to be able to *hold* its
/// grant, or it would be over capacity before its first click.
fn default_base_storage() -> u64 {
    2048
}

/// The rate site clearing digs at with no `economy.ron` line for it — six
/// times the gatherer hut's own 20 blocks/minute: a build site is worked by
/// a whole crew, not one hut's worth of hands, and a road has to keep up
/// with a drag. See `city::construction`'s module docs.
fn default_site_clearing_rate() -> f32 {
    120.0
}

/// The loaded economy knobs. [`Default`] is the "no `economy.ron`" config:
/// no grant, no conversions, and a vanilla stack — exactly the game ticket
/// 073 shipped, plus the one number ticket 078 needs a value for whether or
/// not a file supplies it.
///
/// Hand-written rather than derived: a derived `Default` would give
/// `stack_size` a zero, and a stack of nothing is a divide-by-zero waiting in
/// every buffer calculation downstream.
#[derive(Resource, Debug)]
pub struct EconomyConfig {
    /// How many items make one stack — a producer's buffer is measured in
    /// these, and one is what a haul carries.
    pub stack_size: u64,
    /// The city's storage capacity before any warehouse adds to it (ticket
    /// 079) — see [`super::warehouse::StorageCapacity`].
    pub base_storage: u64,
    /// Blocks per minute of game time [`super::construction`] digs a
    /// placement's site out at, before its blueprint is written (ticket
    /// 128). Always `> 0.0` — [`load_economy`] refuses anything else, the
    /// same way it refuses a zero `stack_size`.
    pub site_clearing_blocks_per_minute: f32,
    /// What a city with no `stock.ron` is founded with.
    pub start_stock: Parcel,
    /// Applied by [`plan_payment`], in file order — the first conversion
    /// that can produce a short item wins, so a table listing a cheap route
    /// before an expensive one gets the cheap one.
    pub conversions: Vec<Conversion>,
    /// Sets of materials that convert to each other 1:1 (ticket 075) — see
    /// the module docs. Tried *after* [`Self::conversions`], so an explicit
    /// ratio always wins over a synonym.
    pub groups: Vec<Vec<String>>,
}

impl Default for EconomyConfig {
    fn default() -> Self {
        EconomyConfig {
            stack_size: default_stack_size(),
            base_storage: default_base_storage(),
            site_clearing_blocks_per_minute: default_site_clearing_rate(),
            start_stock: Parcel::default(),
            conversions: Vec::new(),
            groups: Vec::new(),
        }
    }
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
    /// An `interchangeable` group with fewer than two members — it says
    /// nothing, so it's a half-finished edit rather than a valid choice.
    GroupTooSmall(usize),
    /// A material listed twice in one group, or in two groups at once. The
    /// second is the real hazard: "which group wins" is not a question this
    /// file should be able to ask.
    DuplicateGroupMember(String),
    /// `stack_size: 0` (ticket 078) — a buffer measured in stacks of nothing
    /// holds nothing, so every producer would stall on its first tick.
    ZeroStackSize,
    /// `site_clearing_blocks_per_minute <= 0.0` (ticket 128) — a site that
    /// clears at zero or negative blocks per minute never finishes, and
    /// there's no "instant" spelling that number could reasonably mean.
    NonPositiveSiteClearingRate,
}

impl std::fmt::Display for EconomyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EconomyError::Read(err) => write!(f, "{err}"),
            EconomyError::Parse(msg) => write!(f, "{msg}"),
            EconomyError::ZeroCount(what) => write!(f, "conversion {what} has a count of 0"),
            EconomyError::EmptyItem(what) => write!(f, "conversion {what} has a blank item name"),
            EconomyError::SelfConversion(item) => write!(f, "{item} is listed as converting to itself"),
            EconomyError::ZeroStackSize => write!(f, "stack_size must be > 0"),
            EconomyError::NonPositiveSiteClearingRate => write!(f, "site_clearing_blocks_per_minute must be > 0"),
            EconomyError::GroupTooSmall(index) => {
                write!(f, "interchangeable group {index} has fewer than two materials in it")
            }
            EconomyError::DuplicateGroupMember(item) => {
                write!(f, "{item} appears more than once across the interchangeable groups")
            }
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

    let mut groups: Vec<Vec<String>> = Vec::with_capacity(file.interchangeable.len());
    let mut seen: HashMap<String, ()> = HashMap::new();
    for (index, group) in file.interchangeable.into_iter().enumerate() {
        let mut members = Vec::with_capacity(group.len());
        for member in group {
            if member.trim().is_empty() {
                return Err(EconomyError::EmptyItem(format!("interchangeable group {index}")));
            }
            let member = namespaced(&member);
            // Across groups as well as within one: two groups sharing a
            // member would make "what is this the same as" depend on which
            // group `routes_to` happened to look at first.
            if seen.insert(member.clone(), ()).is_some() {
                return Err(EconomyError::DuplicateGroupMember(member));
            }
            members.push(member);
        }
        if members.len() < 2 {
            return Err(EconomyError::GroupTooSmall(index));
        }
        groups.push(members);
    }

    if file.stack_size == 0 {
        return Err(EconomyError::ZeroStackSize);
    }
    if file.site_clearing_blocks_per_minute <= 0.0 {
        return Err(EconomyError::NonPositiveSiteClearingRate);
    }

    Ok(EconomyConfig {
        stack_size: file.stack_size,
        base_storage: file.base_storage,
        site_clearing_blocks_per_minute: file.site_clearing_blocks_per_minute,
        start_stock,
        conversions,
        groups,
    })
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

/// One way to make more of some material: `count` of `from` becomes
/// `produces` of it.
///
/// The two sources of supply — an explicit [`Conversion`] and a 1:1
/// group-mate (ticket 075) — flattened into one shape, so [`cover`] has a
/// single list to walk rather than two loops that have to agree with each
/// other about ordering and cycle rules.
struct Route<'a> {
    from: &'a str,
    count: u32,
    produces: u32,
}

/// Every route that ends in `item`: the explicit conversions first (a stated
/// ratio beats a synonym), then the item's group-mates at 1:1 — unless an
/// `ancestor` (an item further up the chain [`cover`] is working through)
/// is in the same group, in which case the synonyms are left out.
///
/// That exclusion is what keeps a group from being walked once per *path*
/// through it (ticket 125). A group is a clique: every member is already a
/// direct route of every other, so when `oak_planks` is short and `cover`
/// chases its synonym `birch_planks`, the only thing `birch_planks` can
/// contribute that `oak_planks`' own routes don't is a *conversion* into it
/// (`birch_log -> birch_planks`) — its own synonyms are the very list the
/// caller is already walking. Without the exclusion, an item short in the
/// 48-member log group recursed into 47 synonyms, each into 46, to the
/// depth cap: millions of [`cover`] calls per priced cost, which the build
/// menu does once per row per frame — a second and a half a frame with an
/// empty pile.
///
/// Built per call rather than cached: it's a filter over two small lists, it
/// happens once per short material rather than once per block, and a cached
/// index would be one more thing to keep in step with a hot-reloaded config.
fn routes_to<'a>(item: &str, economy: &'a EconomyConfig, ancestors: &[String]) -> Vec<Route<'a>> {
    let mut routes: Vec<Route<'a>> = economy
        .conversions
        .iter()
        .filter(|conversion| conversion.to.item == item)
        .map(|conversion| Route { from: &conversion.from.item, count: conversion.from.count, produces: conversion.to.count })
        .collect();

    // At most one group can hold `item` — `load_economy` refuses a name that
    // appears in two.
    if let Some(group) = economy.groups.iter().find(|group| group.iter().any(|member| member == item)) {
        let group_already_walked = ancestors.iter().any(|ancestor| group.iter().any(|member| member == ancestor));
        if !group_already_walked {
            routes.extend(
                group
                    .iter()
                    .filter(|member| member.as_str() != item)
                    .map(|member| Route { from: member.as_str(), count: 1, produces: 1 }),
            );
        }
    }

    routes
}

/// Prices `costs` against `stock`, converting through `economy`'s table and
/// groups wherever the stock is short — see the module docs for what
/// "converting" is allowed to mean here.
///
/// Pure: nothing is mutated, and the returned [`ConversionPlan`] is what the
/// caller should apply if it decides to go ahead. That's what lets the build
/// menu ask the same question the commit answers without either of them
/// touching the stock.
pub fn plan_payment(stock: &Stock, costs: &[Cost], economy: &EconomyConfig) -> Payment {
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
        cover(item, needed - held, economy, &mut ledger, &mut plan, &mut covering, 0);

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
    economy: &EconomyConfig,
    ledger: &mut Ledger,
    plan: &mut ConversionPlan,
    covering: &mut Vec<String>,
    depth: usize,
) {
    if amount == 0 || depth >= MAX_CONVERSION_DEPTH || covering.iter().any(|held| held == item) {
        return;
    }
    // The routes are chosen against the chain *above* `item` — see
    // `routes_to` for why its group-mates are left out when an ancestor
    // already walked that group.
    let routes = routes_to(item, economy, covering);
    covering.push(item.to_string());

    let mut still_needed = amount;
    for route in routes {
        if still_needed == 0 {
            break;
        }

        // A run produces `route.produces`; round up, since half a run isn't
        // a thing — converting one log for two planks leaves two planks over.
        let runs_wanted = still_needed.div_ceil(u64::from(route.produces));

        // The input may itself be short — chase it one level further before
        // giving up on this route.
        let input_wanted = runs_wanted.saturating_mul(u64::from(route.count));
        let input_held = ledger.available(route.from);
        if input_held < input_wanted {
            cover(route.from, input_wanted - input_held, economy, ledger, plan, covering, depth + 1);
        }

        let runs = runs_wanted.min(ledger.available(route.from) / u64::from(route.count));
        if runs == 0 {
            continue;
        }

        let consumed = runs * u64::from(route.count);
        let produced = runs * u64::from(route.produces);
        ledger.take(route.from, consumed);
        ledger.give(item, produced);
        plan.consumed.add(route.from, consumed);
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

    /// A config with a conversion table and no groups — what every ticket
    /// 074 test was written against.
    fn table(conversions: Vec<Conversion>) -> EconomyConfig {
        EconomyConfig { conversions, ..EconomyConfig::default() }
    }

    fn logs_to_planks() -> EconomyConfig {
        table(vec![conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4))])
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
        let payment = plan_payment(&Stock::default(), &[], &table(vec![]));
        assert!(payment.affordable());
        assert!(payment.conversion.is_empty());
    }

    #[test]
    fn with_no_conversions_a_shortfall_is_just_a_shortfall() {
        let stock = stock_with(&[("minecraft:oak_log", 40)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &table(vec![]));

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
        let chain = table(vec![
            conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4)),
            conversion(("minecraft:oak_planks", 2), ("minecraft:stick", 4)),
        ]);
        let stock = stock_with(&[("minecraft:oak_log", 4)]);

        let payment = plan_payment(&stock, &[cost("minecraft:stick", 8)], &chain);

        assert!(payment.affordable(), "{:?}", payment.shortfall.missing);
        assert_eq!(payment.conversion.produced.get("minecraft:stick"), 8);
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_planks"), 4);
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 1);
    }

    #[test]
    fn a_cycle_gives_up_instead_of_hanging() {
        // The pathological table the depth cap and the `covering` set exist
        // for. If this test hangs, they don't work.
        let cyclic = table(vec![
            conversion(("minecraft:a", 1), ("minecraft:b", 1)),
            conversion(("minecraft:b", 1), ("minecraft:a", 1)),
        ]);
        let payment = plan_payment(&Stock::default(), &[cost("minecraft:a", 5)], &cyclic);

        assert!(!payment.affordable());
        assert_eq!(payment.shortfall.missing.get("minecraft:a"), 5);
    }

    #[test]
    fn two_costs_cannot_both_spend_the_same_log() {
        // The whole reason the planner works against a running ledger rather
        // than pricing each item independently.
        let two_uses = table(vec![
            conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4)),
            conversion(("minecraft:oak_log", 1), ("minecraft:stick", 4)),
        ]);
        let stock = stock_with(&[("minecraft:oak_log", 1)]);

        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 4), cost("minecraft:stick", 4)], &two_uses);

        assert!(!payment.affordable(), "one log can't pay for both");
        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 1);
    }

    // --- groups (ticket 075) ------------------------------------------------

    /// The shipped shape: logs convert to their own planks at 4:1, and every
    /// plank type is the same plank.
    fn wood_economy() -> EconomyConfig {
        EconomyConfig {
            conversions: vec![
                conversion(("minecraft:oak_log", 1), ("minecraft:oak_planks", 4)),
                conversion(("minecraft:birch_log", 1), ("minecraft:birch_planks", 4)),
            ],
            groups: vec![vec!["minecraft:oak_planks".to_string(), "minecraft:birch_planks".to_string()]],
            ..EconomyConfig::default()
        }
    }

    #[test]
    fn a_group_mate_pays_a_cost_one_for_one() {
        let stock = stock_with(&[("minecraft:birch_planks", 40)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &wood_economy());

        assert!(payment.affordable());
        assert_eq!(payment.conversion.consumed.get("minecraft:birch_planks"), 40);
        assert_eq!(payment.conversion.produced.get("minecraft:oak_planks"), 40);
    }

    /// The user's actual question, end to end: a city that has only ever cut
    /// birch trees can build a house priced in oak planks.
    #[test]
    fn a_birch_forest_can_pay_for_an_oak_house() {
        let stock = stock_with(&[("minecraft:birch_log", 10)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &wood_economy());

        assert!(payment.affordable(), "{:?}", payment.shortfall.missing);
        assert_eq!(payment.conversion.consumed.get("minecraft:birch_log"), 10, "through birch planks");
        assert_eq!(payment.conversion.produced.get("minecraft:oak_planks"), 40);
    }

    #[test]
    fn what_the_stock_already_holds_beats_a_group_mate() {
        let stock = stock_with(&[("minecraft:oak_planks", 40), ("minecraft:birch_planks", 40)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &wood_economy());

        assert!(payment.affordable());
        assert!(payment.conversion.is_empty(), "no reason to touch the birch");
    }

    #[test]
    fn a_group_is_not_an_infinite_source() {
        // oak <-> birch is a cycle by construction; the visited set has to
        // stop it being a way to make planks out of nothing.
        let payment = plan_payment(&Stock::default(), &[cost("minecraft:oak_planks", 40)], &wood_economy());

        assert!(!payment.affordable());
        assert_eq!(payment.shortfall.missing.get("minecraft:oak_planks"), 40);
        assert!(payment.conversion.is_empty());
    }

    #[test]
    fn an_explicit_ratio_is_tried_before_a_group_mate() {
        // Both routes are open; the stated 1-log-makes-4 wins over trading
        // planks one for one, because it's the cheaper answer and the file
        // said so explicitly.
        let stock = stock_with(&[("minecraft:oak_log", 10), ("minecraft:birch_planks", 40)]);
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &wood_economy());

        assert_eq!(payment.conversion.consumed.get("minecraft:oak_log"), 10);
        assert_eq!(payment.conversion.consumed.get("minecraft:birch_planks"), 0);
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
        assert_eq!(config.stack_size, 64, "a missing file still has to name a stack size");
    }

    /// Ticket 078: an `economy.ron` written before production existed has no
    /// `stack_size` and must keep working, with the vanilla stack.
    #[test]
    fn a_file_with_no_stack_size_gets_the_vanilla_one() {
        let config = config_from(r#"(start_stock: {"dirt": 1})"#).expect("loads");
        assert_eq!(config.stack_size, 64);
    }

    #[test]
    fn a_stated_stack_size_is_read() {
        let config = config_from(r#"(stack_size: 16)"#).expect("loads");
        assert_eq!(config.stack_size, 16);
    }

    /// A buffer measured in stacks of nothing holds nothing, so every
    /// producer would stall on its first tick.
    #[test]
    fn a_zero_stack_size_is_refused() {
        let err = config_from(r#"(stack_size: 0)"#).expect_err("zero stack size");
        assert!(matches!(err, EconomyError::ZeroStackSize), "{err}");
    }

    /// Ticket 128: a file with no line for it keeps the default rate.
    #[test]
    fn a_file_with_no_site_clearing_rate_gets_the_default() {
        let config = config_from(r#"(start_stock: {"dirt": 1})"#).expect("loads");
        assert_eq!(config.site_clearing_blocks_per_minute, 120.0);
    }

    #[test]
    fn a_stated_site_clearing_rate_is_read() {
        let config = config_from(r#"(site_clearing_blocks_per_minute: 60.0)"#).expect("loads");
        assert_eq!(config.site_clearing_blocks_per_minute, 60.0);
    }

    #[test]
    fn a_non_positive_site_clearing_rate_is_refused() {
        let err = config_from(r#"(site_clearing_blocks_per_minute: 0.0)"#).expect_err("zero rate");
        assert!(matches!(err, EconomyError::NonPositiveSiteClearingRate), "{err}");
        let err = config_from(r#"(site_clearing_blocks_per_minute: -5.0)"#).expect_err("negative rate");
        assert!(matches!(err, EconomyError::NonPositiveSiteClearingRate), "{err}");
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
    fn a_group_loads_namespaced() {
        let config = config_from(r#"(interchangeable: [["oak_planks", "minecraft:birch_planks"]])"#).expect("loads");
        assert_eq!(config.groups, vec![vec!["minecraft:oak_planks".to_string(), "minecraft:birch_planks".to_string()]]);
    }

    #[test]
    fn a_one_member_group_is_refused() {
        let err = config_from(r#"(interchangeable: [["oak_planks"]])"#).expect_err("says nothing");
        assert!(matches!(err, EconomyError::GroupTooSmall(0)), "{err}");
    }

    #[test]
    fn a_material_in_two_groups_is_refused() {
        // "Which group wins" is not a question the file should be able to
        // ask — `routes_to` takes the first match.
        let err = config_from(
            r#"(interchangeable: [["oak_planks", "birch_planks"], ["oak_planks", "spruce_planks"]])"#,
        )
        .expect_err("ambiguous");
        assert!(matches!(err, EconomyError::DuplicateGroupMember(_)), "{err}");
    }

    #[test]
    fn a_material_listed_twice_in_one_group_is_refused() {
        let err = config_from(r#"(interchangeable: [["oak_planks", "minecraft:oak_planks"]])"#).expect_err("duplicate");
        assert!(matches!(err, EconomyError::DuplicateGroupMember(_)), "{err}");
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
        let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &config);
        assert!(payment.affordable(), "the shipped table should turn logs into planks");

        // Ticket 075, against the real file: the wood the map actually gave
        // you pays for a cost priced in oak.
        for wood in ["minecraft:birch_log", "minecraft:spruce_log", "minecraft:stripped_dark_oak_log", "minecraft:cherry_wood"] {
            let stock = stock_with(&[(wood, 10)]);
            let payment = plan_payment(&stock, &[cost("minecraft:oak_planks", 40)], &config);
            assert!(payment.affordable(), "{wood} should pay for an oak-planks cost: {:?}", payment.shortfall.missing);
        }
    }

    /// Ticket 125: a synonym group is walked once per chain, not once per
    /// path through it. Before `routes_to` learned to skip a group an
    /// ancestor already expanded, a short cost in a group this size recursed
    /// `n * (n-1) * (n-2) * (n-3)` ways to the depth cap — minutes here, and
    /// ~1.5 s a frame in the build menu against the shipped 48-log group.
    /// The answer is unchanged: with nothing in the pile there is nothing
    /// to convert, and with one synonym in it that synonym pays.
    #[test]
    fn a_large_group_is_priced_in_microseconds_not_minutes() {
        let members: Vec<String> = (0..80).map(|i| format!("minecraft:log_{i}")).collect();
        let config = EconomyConfig { groups: vec![members.clone()], ..EconomyConfig::default() };

        let started = std::time::Instant::now();
        let short = plan_payment(&Stock::default(), &[cost("minecraft:log_0", 4)], &config);
        let paid = plan_payment(&stock_with(&[("minecraft:log_79", 4)]), &[cost("minecraft:log_0", 4)], &config);
        let elapsed = started.elapsed();

        assert!(!short.affordable());
        assert!(paid.affordable());
        assert_eq!(paid.conversion.consumed.get("minecraft:log_79"), 4);
        assert!(elapsed < std::time::Duration::from_millis(200), "took {elapsed:?}");
    }

    /// Ticket 125, the actual case: every shipped building priced against
    /// an empty pile — what the build menu does once per row per frame —
    /// in well under a frame.
    #[test]
    fn the_shipped_buildings_price_against_an_empty_pile_within_a_frame() {
        let config = load_economy(Path::new("assets/city/economy.ron")).expect("shipped economy.ron should load");
        let catalogue = crate::blueprint::load_catalogue_dir(Path::new("assets/city/blueprints")).0;
        let (definitions, _) = super::super::definition::load_definitions_dir(Path::new("assets/city/buildings"), &catalogue);
        assert!(!definitions.is_empty(), "the shipped definitions should load");

        let started = std::time::Instant::now();
        for entry in definitions.iter() {
            plan_payment(&Stock::default(), &entry.building.cost, &config);
        }
        let elapsed = started.elapsed();
        assert!(elapsed < std::time::Duration::from_millis(50), "pricing every row took {elapsed:?}");
    }
}
