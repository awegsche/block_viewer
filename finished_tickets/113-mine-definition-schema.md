# 113 - Mine: definition schema and three tier files

Design: `MINES_DESIGN.md` ("Coordinates and constants", "Tiers"). First of
the mine sequence (113 → 114 → 115 → 116); 114–116 have nothing to read
until this block exists, which is the iteration boundary 086's history
says a schema-first ticket is for.

## Scope

`definition::Building` gains an optional `mine` block, validated at load;
three `.ron` files ship against **placeholder geometry**. Nothing is
simulated yet — `is_producer`, the panel capacity lookups and the tick all
land in 116, so until then a placed Mine is an inert building with a
(correct) build-menu entry and cost.

## Schema

```rust
/// `Some` makes this building a mine (ticket 113, MINES_DESIGN.md) — the
/// specialised extraction building `Gatherer`'s docs have been contrasting
/// themselves against since 086. See `city::mine` (116) for the tick.
#[serde(default)]
pub mine: Option<Mine>,

#[derive(Debug, Clone, Deserialize)]
pub struct Mine {
    /// Min corner of the shaft square in the *unrotated* blueprint's (x, z).
    pub shaft: ShaftAt,                       // struct ShaftAt { x: i32, z: i32 }
    /// Outer edge of the stair ring. `level_spacing()` is derived from it.
    #[serde(default = "default_shaft_size")]  // 6
    pub shaft_size: u32,
    /// `floor_y - first_level_depth` is the first level's floor.
    pub first_level_depth: u32,
    /// The tier knob: no level floor below this world Y.
    pub min_level_y: i32,
    /// Each secondary arm's length (north and south).
    pub level_reach: u32,
    /// Each tertiary gallery's length (east and west of the secondary).
    pub gallery_length: u32,
    #[serde(default = "default_torch_spacing")]   // 8
    pub torch_spacing: u32,
    #[serde(default = "default_max_void_run")]    // 6
    pub max_void_run: u32,
    pub blocks_per_minute: f32,
    #[serde(default = "default_buffer_stacks")]   // the shared 4 — every shipped mine overrides it
    pub buffer_stacks: u32,
    #[serde(default)]
    pub haul_at_stacks: Option<u32>,
    /// Block names (bare or `minecraft:`-prefixed) scanned for in addition
    /// to every `*_ore`. Empty by default.
    #[serde(default)]
    pub valuables: Vec<String>,
}

impl Mine {
    pub fn level_spacing(&self) -> i32 { self.shaft_size as i32 - 2 }
    pub fn haul_threshold_stacks(&self) -> u32 { ... }   // same as Gatherer's
    pub fn is_valuable(&self, block_name: &str) -> bool  // `*_ore` or listed; normalises the prefix like drops.ron does
}
```

**Why no `ground`/`stairs`/`pillar` block fields.** The design names the
materials (cobblestone, oak stairs, oak log, stripped oak log). They are
constants in `city::mine::layout` (114) until a second mine style wants
different ones — the same call `road_definition` made before 059 gave roads
styles. Don't pre-build the knob.

## Validation (`validate`)

`DefinitionError::InvalidMine(&'static str)`, the `InvalidGatherer` shape,
plus one `HaulAtNotBelowBuffer { block: "mine", .. }` reuse:

- `shaft_size >= 4` — a 3-wide shaft has a 1-wide interior, which is not
  the 4-wide secondary the design is built around; the message says so.
- `first_level_depth >= level_spacing` and `first_level_depth %
  level_spacing == 0` — a level's landing must fall on a corner
  (`MINES_DESIGN.md`, "level_spacing is derived").
- `level_reach >= GALLERY_PITCH`, `gallery_length > 0`, `torch_spacing >
  0`, `max_void_run > 0`, `blocks_per_minute > 0`, `buffer_stacks > 0`.
- `haul_at_stacks < buffer_stacks` when given (`HaulAtNotBelowBuffer`).
- `min_level_y >= WORLD_MIN_Y + 8` — the deepest scan layer is the level
  floor itself; the bedrock band is `-64..=-60`, and 8 leaves a flight of
  slack under a `-56` floor. Message names the bound.

In `load_entry` (needs the resolved footprint, like `InvalidGroundLevel`):
`DefinitionError::ShaftOutsideFootprint { shaft, shaft_size, footprint }`
when the shaft square **plus its one-block lining** doesn't fit inside the
footprint: `1 <= shaft.x && shaft.x + shaft_size + 1 <= footprint.x`, same
on `z`. The lining is written from `floor_y - 1` down, so strictly it could
poke outside the footprint underground — but a neighbour's foundation or a
road's tunnel could be right there, and keeping it inside is free.

A definition with both `mine` and `gatherer`, or `mine` and `production`,
is refused (`InvalidMine("a mine is its own producer; drop the production/
gatherer block")`) — the panels resolve one capacity per building and the
shipped set never needs two.

## The three files

`assets/city/buildings/mine01.ron` / `mine02.ron` / `mine03.ron` per the
design's tier table (`Mine` / `Deep Mine` / `Deepslate Mine`; tiers 1/2/3;
`requires: []` / `["mine01"]` / `["mine02"]`; `min_level_y: 16 / -24 /
-56`). Shared numbers: `shaft: (x: 5, z: 5)`, `shaft_size: 6`,
`first_level_depth: 12`, `level_reach: 100`, `gallery_length: 200`,
`buffer_stacks: 512`, `haul_at_stacks: Some(64)`. Per tier:
`blocks_per_minute: 60 / 90 / 120`. Cost climbs with the tier — start from
`oak_planks 64, oak_log 32, cobblestone 64` and add `raw_iron`/`raw_gold`
to the higher tiers so the tree means something (a Deep Mine is paid for by
a Mine's output). `category: Production`.

**Placeholder geometry, the `warehouse01` way**: `blueprint: "lumber.nbt"`,
`footprint: Explicit(x: 16, z: 16)`, `ground_level: 2` (borrowed from
`lumber.ron`, describing the borrowed blueprint). The comment at the top of
each file says so and points at ticket 118. `todo.md` gets the "real model
pending" line 079/086 got.

## Also

- `Building`'s `gatherer` doc comment still says "a specialised quarry/mine
  (neither built yet)" — update it and `CITYBUILDER_ROADMAP.md`'s gatherer
  entry to point at `MINES_DESIGN.md`. Add a short **H3 — Mines** section to
  the roadmap after H2 with the ticket map.
- Hot reload (`city::hot_reload`) needs nothing: it reloads whole
  definitions.

## Tests

- A full `mine` block parses; every defaulted field defaults as documented.
- Each `validate` rule refuses with `InvalidMine` carrying its own message
  (one test per rule, the `gatherer_*` test shape); `haul_at_stacks ==
  buffer_stacks` refuses with `HaulAtNotBelowBuffer { block: "mine" }`.
- `ShaftOutsideFootprint` for a shaft at `(0, 5)` (no room for the lining)
  and for one whose far edge overruns.
- `first_level_depth: 10` with `shaft_size: 6` refuses; `12` passes.
- `is_valuable`: `minecraft:deepslate_iron_ore` yes, `iron_ore` yes,
  `ancient_debris` only when listed, `stone` no.
- The shipped directory loads with three more entries, and
  `resolve_requirements` chains mine01 → mine02 → mine03.
- `cargo check`, `cargo test --lib`, `cargo clippy` clean.
