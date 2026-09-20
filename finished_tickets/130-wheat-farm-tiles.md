# 130 - Wheat farm: give it farm tiles too

## Status
Done — `cargo test --lib` green (1174 passed; one failure seen on a
full-suite run, `inspect_panel::tests::site_clearing_line_...`, passes when
run in isolation — test-order flakiness in an unrelated module (site
clearing/inspect panel), not touched by this ticket's two files).

## Why
Ticket 084 built the `Building::farm` hub/tile mechanism (radius-scaled
production) and shipped it for `lumber.ron`/`lumber_farm_01.ron`, but
deliberately used `farm01.ron`'s flat-rate wheat production as the
*contrast* case — nothing about the mechanism itself is lumber-specific.
The wheat farm should scale the same way: zero tiles nearby, zero wheat;
enough tiles in range, full rate.

## What
- New tile building `farm01_tile.ron` ("Wheat Field"), `requires:
  ["farm01"]`, no `production`/`farm` of its own — mirrors
  `lumber_farm_01.ron` exactly. Placeholder geometry (`house01.nbt`,
  `Explicit(x: 5, z: 5)`), same convention `farm01.ron` itself already uses,
  since there's no `.nbt` export for a wheat field yet.
- `farm01.ron` gains `farm: Some(Farm(tile: "farm01_tile", radius_blocks:
  20, tiles_for_full_rate: 4))` — guessed tuning, same as every other
  knob in this file, scaled up a little from lumber's (16 blocks / 3
  tiles) since the hub's own footprint (9x9) is bigger.
- No code changes: `city::farm`, `production::scale_production`, the build
  menu's "Scales with ..." line and the city panel's `count/needed`
  fraction all already read `Building::farm` generically — 084 built this
  to be reusable by any hub, and nothing farm01-specific ever existed in
  the logic itself.

## Verification
`cargo test --lib` — in particular
`definition::tests::the_real_house01_fixture_loads`-style loader tests
that walk `assets/city/buildings` for real, plus `city::farm`'s own
coverage tests (unaffected, but the new file must parse and validate:
`InvalidFarm`, `DanglingFarmTile`, footprint checks all run over the real
shipped set).

Not verified here (needs a human watching the window, per this repo's
manual-verification convention): that a placed Wheat Field actually shows
up in the build menu's catalogue and scales `farm01`'s output in the city
panel the way `lumber_farm_01` already does. Noted in `todo.md`.
