# 107 - Inspect panel: warehouse connection status, and two debug aids

## Why

Requested directly: the inspect panel should say whether a producer is
reachable by a warehouse (078/079/080 already compute this as `Coverage`, but
nothing displayed it per-building), and two things to make haulage easier to
debug by eye:

- a much smaller `stack_size` so a stack fills and a haul dispatches in
  seconds instead of minutes
- a manual "Clear buffer" button so a stalled/full producer can be reset
  without waiting out a haul or restarting the app

## Scope

- `city::ui::inspect_panel`: for a selected building with a `Producer`
  (production or gatherer), show `Coverage::served` — the serving warehouse's
  name and travel time, or a red "not connected to a warehouse" warning when
  `served` is `None`.
- Same panel: a "Clear buffer" button when the buffer holds anything, which
  empties `Producer::buffer` directly (`ProductionState::entry`). Debug-only;
  no journal entry, matching production's own "no journal, ever" rule.
- `assets/city/economy.ron`: `stack_size` 64 -> 8, purely for faster visual
  feedback while iterating on haulage. This also shrinks every buffer
  (`buffer_stacks * stack_size`), which is the point — a farm should visibly
  fill and dispatch in well under a minute of game time.

## Out of scope

Not touching `Coverage`, `dispatch_hauls`, or `deliver_arrivals` themselves —
this is display plus a debug control, not a mechanics change. Not adding a
setting to toggle `stack_size`; it's a single tunable in `economy.ron` and
changing it back later is a one-line revert if the smaller value turns out to
matter for real balance testing.

## Status

Done — `cargo test` green. Manual verification of the panel's new line and
button noted in `todo.md` per this repo's rule (Claude does not `cargo run` to
eyeball UI).
