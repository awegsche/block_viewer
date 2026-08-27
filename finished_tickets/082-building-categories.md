# 082 - Building categories and a categorized build menu

## Status
Done — `cargo test --lib` green (759 passed).

Landed as described below, with one deviation: a building row switching
`*tool` to `ActiveTool::Building` on click **did** land in this ticket after
all, not deferred to 083. It's symmetric with the Street section's own row
switching to `ActiveTool::Road` (same ticket, same mechanism), and
`ActiveTool::Building` already exists and is already the default today, so
it's a no-op until 083 changes what the default is — nothing to gain by
waiting.

## Depends on
040 (definitions), 050 (build menu).

## Why
The build menu today groups by tier only. Designing the tree farm's "Place
Farm Tile" button (083/084) surfaced a more basic gap first: there's no way
to browse the menu by *kind* of building, and the road tool has no menu
presence at all — `T` plus drag is the whole interface. Anno-style browsing
groups by category first, tier second.

## Schema
`definition::Building` gains:

```
category: Production | Residential | Street,
```

`#[serde(default)]` -> `Category::Production`, same treatment every field
added after the original ticket-040 founding set (`production`, `cost`,
`warehouse`) already gets — `category` doesn't gate any real validation the
way `blueprint`/`footprint` do, so requiring it outright would only mean
touching every one of `definition.rs`'s existing RON test fixtures to keep
them parsing, for no matching safety gained. All five shipped `.ron` files
still get an explicit one rather than leaning on the default:

- `house01` -> `Residential`
- `farm01`, `warehouse01`, `warehouse02`, `lumber` -> `Production`

`Street` is reserved for road styles, not a catalogue building (see below) —
no `.ron` uses it, but the enum needs the variant for the menu to have a
section to render.

## Build menu
`city::ui::build_menu` restructures from tier-only headings to type-outer,
tier-inner: one heading per `Category` (`Production`, `Residential`,
`Street`, in that order), tier sub-headings within each, same row rendering
otherwise.

The `Street` section does not list `BuildingDefinitions` entries — it lists
`RoadTypes`/`RoadCatalogue::styles()`, one row per loaded style. Clicking a
style row sets `ActiveTool::Road` and selects that style (mirroring
`road_build::RoadStyleSelection`, which keeps working as a `[`/`]` fallback).
An empty `Street` section (no style has a piece loaded — today's actual
state per ticket 060/063's own note) shows a placeholder line rather than
nothing.

## Not in scope here
Making `Inspect` the default tool, the per-building panel, and the farm-tile
mechanic itself are 083/084.
