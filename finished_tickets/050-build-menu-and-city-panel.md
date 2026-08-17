# 050 - Build menu and city panel

## Status
Done — implemented and tested (`cargo build`/`cargo test` clean, 439 tests,
up from 418). See the Resolution.

## Part of

Roadmap group G (`tickets/CITYBUILDER_ROADMAP.md`): G1 (build menu) and G2
(city panel), the citybuilder's first real UI. Depends on B4's catalogue
(ticket 039), C1/C2's definitions and tech tree (tickets 040-041), D1's city
state (ticket 042), D3's journal (ticket 044), and E1-E5's picking,
placement, commit and demolish (tickets 045-049) — everything G is a window
onto.

## Problem

Every citybuilder interaction so far is a keyboard stand-in: number keys
pick a catalogue entry, `R` rotates, `Delete` demolishes whatever tile is
hovered. `BuildingDefinitions` (tier, cost, production, `requires`) has no
reader at all. `Journal::undo_last` has existed, unused, since ticket 044.
Nothing shows how many buildings exist, whether the last write actually
landed, or where its backup went.

## Goal

Per the roadmap:

- **G1.** Catalogue grouped by tier, locked entries visible but disabled and
  showing what unlocks them, costs and production shown from C1's data even
  while inert.
- **G2.** Building counts, road length, and the write status — last write,
  dirty regions, backup location.

## Scope

- A second, citybuilder-only `EguiPlugin` registration (`city::ui`) — the
  viewer's `UiPlugin` carries five panels this game has no use for.
- The build menu: tier headings, one row per building (name, footprint,
  cost, production), clickable when unlocked, disabled with a "requires…"
  tooltip when not. Clicking sets the same `PlacementSelection` ticket 047's
  number keys already drive.
- **Unlocking has no prior definition anywhere in the codebase.** Iteration
  1 has no separate "researched techs" resource (C3's own words: "nothing
  simulates them yet"). Landed as: a `requires` id is met once at least one
  building of that type has actually been placed in `City` — the smallest
  rule that uses data already on hand rather than inventing a second piece
  of state.
- `LoadedBuilding` gains a `catalogue_id` field — `PlacementSelection`/
  `City::place_building` have always keyed a placement by the *catalogue*
  id (a blueprint's own filename stem), not a *definition* id (a `.ron`
  file's), and the two are only guaranteed equal by convention, not by type.
  A build menu built off definitions has to bridge them on every click.
- The city panel: building counts (total + per-definition), road tile
  count, the last commit/demolish/undo's own outcome (blocks, chunks,
  regions, backup paths, or a failure message) via a new `WriteStatus`
  resource.
- An "Undo" button, wired to `Journal::undo_last` through a new
  `city::undo` module — the caller `journal`'s own module docs have named
  since ticket 044.

## Out of scope

- Roadmap group I (damage) — `reconcile`/`repair_edit` still have no UI
  caller; not part of G2's own bullet list.
- F (streets), H (terraforming) — separate roadmap groups.
- Hot reload of definitions (C4).

## Done when

- `cargo build`/`cargo test` clean.
- Tests: requirement-satisfaction logic against a bare `City`; tier/id
  sort order; cost/production line formatting; building-count grouping;
  `WriteStatus` recording success/failure and replacing rather than
  accumulating; `UndoCommand`'s state machine including the documented gap
  (a write failure after `undo_last` has already run is reported, not
  rolled back) and the precondition ordering that keeps that gap rare (no
  save loaded leaves `City`/`Journal` untouched, retryable).
- Manual verification (open the build menu, place a locked and an unlocked
  building, demolish one, click Undo, watch the write-status line) goes in
  `../todo.md`.

## Resolution

Landed as scoped, in a new `city/ui/` (mirroring `viewer/ui/`'s own shape:
one `mod.rs` wiring two panel submodules into an `EguiPlugin`), plus two new
top-level `city` modules: `write_status.rs` and `undo.rs`.

**`city::ui::UiPlugin` is a second, independent `EguiPlugin` registration**,
not a reuse of `viewer::ui::UiPlugin` — that one carries a save picker, a
coordinate jump, a block inspector and a selection panel, four of which are
either explorer-only or built around `crate::selection::Selection`, which
`city::mod`'s own docs already name as deliberately unused here. `city::run`
wires it the same two-line way `viewer::run` wires its own:
`add_plugins(ui::UiPlugin)` plus `configure_sets(Update,
camera::CameraSet.after(ui::UiPanelSet))`, so the panels claim this frame's
pointer/keyboard input before `drive_camera` and the picking/placement/
commit/demolish/undo systems that already gate on `camera::EguiInputCapture`
read it.

**The catalogue-id/definition-id bridge turned out to be the real
prerequisite for G1, not a detail.** `PlacementSelection::catalogue_id` and
`City::place_building` have always named a `BuildingCatalogue` entry — the
raw blueprint shape — never a `BuildingDefinitions` entry. A menu built off
definitions (for tier/cost/production/`requires`) has to resolve one to the
other on every click. `LoadedBuilding` gained a `catalogue_id: String`
field, resolved once in `definition::load_entry` from `Building::blueprint`'s
own filename stem (the same computation that already existed there to find
the catalogue entry in the first place, just not previously kept), rather
than every caller re-deriving it from a path. `BuildingDefinitions::get`/
`is_empty` lost their `#[allow(dead_code)]` markers — this ticket is their
first real caller.

**Unlocking is defined for the first time here, deliberately the smallest
rule that fits.** `city::ui::build_menu::missing_requirements` checks
`Building::requires` against `state::City::buildings`'s own `definition`
field — a requirement is met once at least one building of that type has
actually been placed. No new resource, no persisted "researched" flag;
iteration 1 has nowhere else this data would live, and C3's own docs already
say production/tech data is parsed-but-inert. `entry_row` shows a locked
row disabled, with the missing requirements' *display names* (resolved
through `BuildingDefinitions::get`, falling back to the raw id for an
unreachable dangling reference — ticket 041's own validation already rules
those out of a real `BuildingDefinitions`) in both the disabled-hover
tooltip and a standing "Locked — requires…" line, since the roadmap wants
this visible, not just discoverable on hover.

**The keyboard stand-in stays, on purpose.** Ticket 047's own docs called
number-key selection an interim "until a real menu exists" — this is that
menu, but rotation (`R`), height (`PageUp`/`PageDown`/`Home`) and clearing
(`Escape`) are unchanged; a build menu only needed to replace "which
building," not reinvent the rest. The build menu's own `selected_line` and
a collapsed `key_legend` (mirroring `viewer::ui::selection_panel`'s own)
restate them so they're discoverable from the one panel a player is
actually looking at while placing something.

**`WriteStatus` decouples the city panel from three different write-path
error types.** `commit::poll_commit`, `demolish::poll_demolish` and the new
`undo::poll_undo` each already have a `Result<WriteSummary, WriteError>` (or,
for undo's own pre-write refusals, no `WriteError` at all) in hand at the
exact point they decide what happened; `WriteStatus::record_success`/
`record_failure` are one added call each, not a second tracking mechanism.
`record_failure` takes `impl Into<String>` rather than `&WriteError`
specifically — `city::undo`'s own failures (`UndoError::Occupied`, "nothing
to undo") aren't `WriteError`s, and this module has no reason to know the
difference between the three write paths' error types when each one's own
`Display` already says what happened. `WriteKind` gained a third variant,
`Undo`, alongside `Placed`/`Demolished` — which of the other two an undo
actually reversed is `undo::UndoneKind`'s own distinction, not
`WriteStatus`'s; the write itself produces no new journal entry either way,
so it doesn't borrow one of the other two labels.

**`city::undo` follows commit/demolish's `request`/`busy`/`state` shape,
with one real difference from both.** Neither commit nor demolish decides
what to write until a click supplies a target; undo's target is just
"whatever `Journal::undo_last` does" — and that call *is* the `City`-side
reversal, there's no way to preview it. So `start_undo` checks every
precondition it can *without* calling it first (an empty journal, no save
loaded, the write gate already held — the same three guards
`try_commit_placement`/`try_demolish` already check) and only calls
`undo_last` once all three are clear. From that point there is no rolling
back — `Journal::undo_last`'s own docs already say so ("this call has
already moved `city` and the journal on by the time it returns") — and a
write failure after it is reported, not undone; that's an accepted,
documented gap in D3's own contract, not one this ticket introduces.
Checking what can be checked first just keeps the gap as rare as possible.
`Journal::undo_last`/`UndoStep`/`entries`/`placement` all lost their
`#[allow(dead_code)]` markers — this is their first real caller, three
tickets after ticket 044 built them.

**The city panel** shows building counts (total, plus a per-definition
breakdown via a small `building_counts` grouping function), road tile count
(`City::roads().count()`), the write-status section above, and the undo
button/result — four `ui.heading` sections in one `egui::Window`, no new
resource beyond `WriteStatus` and `UndoCommand` themselves.

Testing: 21 new tests across `write_status` (4), `undo` (5), `ui::build_menu`
(9) and `ui::city_panel` (3) — 418 to 439, per `cargo test --lib`. The
pre-existing `journal`/`definition` suites are unchanged by the
`#[allow(dead_code)]` removals; those functions were already exercised by
their own tests, just without a non-test caller until now.

No manual verification recorded as done — `../todo.md` carries the
checklist (open the build menu, confirm tiers/locking/cost display, place
via a click instead of a number key, demolish, click Undo and confirm the
building reappears and the write-status line updates), same as every other
real-world check in this project.
