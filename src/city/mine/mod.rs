//! Mines (tickets 113–119, `MINES_DESIGN.md`): a surface complex plus
//! everything it digs underneath itself over time.
//!
//! This module is split across the ticket sequence rather than growing in
//! one file, so 114/115/116 don't fight over the same lines:
//!
//! - [`layout`] (ticket 114): pure integer geometry — the primary shaft's
//!   ring walk, flights and lining, a mining level's rows and galleries,
//!   and the [`layout::SliceGeometry`] a slice resolves to. No Bevy, no
//!   region cache, no `DecodedWorld`.
//! - [`progress`] (ticket 114): [`progress::MineProgress`], the small
//!   cursor that walks the design's "order of work" one slice at a time.
//! - Ticket 115 adds survey + slice planning (region-cache reads, block
//!   classification, the `WorldEdit` for one slice); ticket 116 adds the
//!   plugin that ticks it, budgets jobs and persists the cursor. Until
//!   then this module has no system and nothing wires it into `CityPlugin`
//!   — see `MINES_DESIGN.md`'s "The rule everything below follows" for why
//!   that split is deliberate rather than a gap.

// No non-test caller yet for most of either module's public surface — 115
// (survey + slice planning) and 116 (the tick) are what call into this,
// the same gap ticket 113's `Mine::is_valuable`/`valuables` sat in between
// 113 and 116. Whole-module rather than per-item: nearly everything here
// is in that position at once, unlike 113's one field and one method.
#[allow(dead_code)]
pub mod layout;
#[allow(dead_code)]
pub mod progress;
