# 115 - Mine: the survey read and the per-slice plan (no Bevy)

Design: `MINES_DESIGN.md` ("A mining level" — galleries, fluids, lighting;
"The rule everything below follows"). Depends on 114 (`SliceGeometry`,
`shaft_target`). Produces the `WorldEdit` for one slice from what is
actually underground; 116 strings these together into a job.

## The problem this ticket solves

`city::gatherer` reads terrain from `DecodedWorld`. A mine can't: in the
citybuilder that resource holds what is *rendered*, and everything below
030's render floor — the entire mine — is never decoded (030's own docs:
"anything asking what block is at (x,y,z) authoritatively must go through
the region cache — which is already what `blueprint::extract` does"). So
the mine reads through `blueprint::extract_blueprint`, the same way
`ranvil-cli get-area` does, one small box per slice.

## Module: `src/city/mine/plan.rs`

### The survey

```rust
/// Reads `bounds` out of the region cache — the mine's one and only read
/// path. A thin wrapper over `extract_blueprint` with a throwaway
/// `ExtractProgress`; `Blueprint::block_at` (dead code since 022) gets its
/// first real caller.
pub fn survey(bounds: &Box3, cache: &mut RegionCache) -> Result<Blueprint, ExtractError>;
```

`extract_blueprint` takes the `Arc<Mutex<RegionCache>>`; 116's job already
holds the lock for the write, so either give `extract_blueprint` a
`&mut RegionCache` core that the `Arc` version wraps (preferred — a
one-line refactor, no behaviour change, `ranvil-cli` untouched), or hold
and release around the read. Don't lock twice.

Cost note: `sample_column` decodes only the sections a box overlaps, so a
4×5×1 gallery survey is one section decode per chunk column touched —
cheap enough per slice. If profiling 116 ever says otherwise, batch the
survey per *face* (consecutive gallery slices are contiguous in `x`), not
per job (the job alternates east and west).

### Classification

```rust
pub trait BlockSampler { fn block(&self, at: IVec3) -> Option<&BlockState>; }
impl BlockSampler for Blueprint { .. }   // block_at; tests use a HashMap<IVec3, BlockState>

pub enum Material { Air, Fluid, Clutter, Solid, Ore, Bedrock, Unknown }
pub fn classify(state: Option<&BlockState>, mine: &Mine) -> Material;
```

- `Air`: `drops::AIR_NAMES` (make it `pub(crate)`) — `air`, `cave_air`,
  `void_air`.
- `Fluid`: `water`, `lava`, `bubble_column` — the same three `drops.ron`'s
  `nothing` list opens with; a small const here, not a read of that file.
- `Clutter`: `grid::is_clutter_name` (make it `pub(super)`) — torches,
  plants, glow lichen, hanging roots, cobwebs, rails; anything a wall torch
  can't hang on and a floor can't be.
- `Bedrock`: `minecraft:bedrock`. Never written, never counted.
- `Ore`: `mine.is_valuable(name)` (113).
- `Unknown`: `None` from the sampler — outside the box, or a column the
  extraction couldn't read. Treated as `Solid` for "is the wall solid?" and
  never written.
- `Solid`: everything else.

### The plan

```rust
pub struct SlicePlan {
    pub edit: WorldEdit,
    pub cost: u32,              // blocks *removed*: every excavated/ore block that wasn't Air or Fluid
    pub outcome: SliceOutcome,  // Dug { cost } | Void | Bedrock — 114's enum, minus Refused (that's the write's verdict)
}

pub fn plan_slice(
    geometry: &SliceGeometry,
    frame: &MineFrame,
    new_bottom: i32,                 // Sink: the bottom after this flight; otherwise the current one
    mine: &Mine,
    sampler: &impl BlockSampler,
) -> SlicePlan;
```

Rules, gallery slice (each a test):

1. **Bedrock anywhere in `excavate`** → `outcome: Bedrock`, empty edit.
2. **`excavate`** → air. `cost += 1` for `Solid`/`Ore`/`Clutter`; `Fluid`
   and `Air` cost nothing (a fluid *inside* the tunnel is still cleared —
   it's a source block that would refill the tunnel).
3. **Floor tiles** (`y = L`, the two under the gallery): `Solid` → keep;
   anything else → `GROUND` (`Ore` here costs 1 and is credited by the
   write's baseline like any removed block).
4. **Scan** (`survey` minus `excavate`, gallery only): `Ore` → air, or
   `GROUND` if `y == L`; `cost += 1`. `Fluid` → `GROUND`, cost 0 (the
   seal). Everything else keep.
5. **Torch**: wall block `Solid`/`Unknown`, *or* `Fluid` this slice seals →
   `WALL_TORCH[facing]` in the corridor block; else `TORCH` on the fallback
   floor tile, which rule 3 guarantees is solid. Nothing if the corridor
   block itself is `Unknown`.
6. **`Void`** when `cost == 0` and nothing but air was in `excavate` (a
   cave crossing or open air); `Dug { cost }` otherwise. A slice that only
   sealed fluid is `Dug { cost: 0 }` — not void, the face found *something*.

Secondary slice: rules 1, 2, 5-for-pillars (every `(pos, torch)` in
`geometry.pillars` → `PILLAR[axis=y]`, and `WALL_TORCH` on the corridor
side when `torch`), floor **always** `GROUND`, shell `Fluid` → `GROUND`,
no ore scan (the design scans galleries only).

Sink slice: every position in `geometry.survey` (the `(S+2)²` ×
`[new_bottom, old_bottom]` box) resolves through
`frame.shaft_target(new_bottom, torch_spacing, at)`:

| `ShaftBlock`      | write                                                          | cost |
|-------------------|----------------------------------------------------------------|------|
| `Air`, `Doorway`  | air, unless already `Air`                                      | 1 per `Solid`/`Ore`/`Clutter` |
| `Ground`          | `GROUND`                                                       | 1 if it replaces `Solid`/`Ore` |
| `Stair { facing }`| `STAIRS[facing=<facing>,half=bottom,shape=straight]`           | 1 if it replaces rock |
| `Pillar` / `Band` | `PILLAR` / `BAND` with `axis=y`                                | 0    |
| `SealIfNotSolid`  | `GROUND` only when `Air`/`Fluid`/`Clutter`                     | 0    |
| `WallTorch`       | `WALL_TORCH[facing]`                                           | 0    |
| `Untouched`       | nothing                                                        |      |

Bedrock anywhere in the ring or interior of that box → `Bedrock` (113's
`min_level_y` floor means this only happens on a definition someone edited
past it; the outcome is still honest rather than a panic).

### Idempotence — the property the whole design leans on

**A position whose sampled state already equals its target is not
written**, compared as a full `BlockState` (name and properties), so a
re-run over finished geometry produces an empty edit at zero cost. This is
what makes a lost `mines.ron` harmless (116) and what makes `Void` mean
"nothing there" rather than "already dug": an already-dug gallery slice
reads as all-air → `Void` → the face closes after `max_void_run`… which is
**wrong** for a resume. So `Void` must distinguish: `excavate` was all
`Air` *and* the floor tiles are natural (not `GROUND` under the exact
gallery footprint) → genuine void; floor tiles already `GROUND` → this is
our own tunnel → `Dug { cost: 0 }`. Test both.

### `BlockState` construction

`BlockState::new(name, &[(k, v)])` or whatever `blueprint::structure`'s
tests use for `torch[facing=east]` — reuse, don't add a second constructor.
Stairs: `facing`, `half=bottom`, `shape=straight`, `waterlogged=false`.
Logs: `axis=y`. Wall torch: `facing`. Standing torch: no properties.

## Tests

Every numbered rule above, against a `HashMap` sampler, plus:

- **Ore in the floor layer becomes `GROUND`, not air**; ore in the ceiling
  layer (`L+4`) becomes air.
- Water in the scan box is sealed with `GROUND` and costs 0; water inside
  `excavate` is cleared and costs 0.
- A gallery slice through solid stone: cost 6, edit is 6 airs (+ torch when
  due); the same slice with 3 ores in the scan: cost 9.
- Torch falls back to a standing torch when the north wall is a cave.
- Secondary slice: 12 airs, 4 floor `GROUND` (even when the rock under it
  was already solid stone), pillar + torch when due.
- Sink slice from `floor_y` on a 6-wide shaft through solid rock: the
  interior 4×4×3 goes to air, the north flight's 4 stairs face `west`, the
  NE landing is `GROUND` at `floor_y - 4`, four `PILLAR`s per y, lining
  otherwise untouched (all solid), no floor (not a level yet); the third
  sink (reaching level 0) writes the `S²` floor and the `Band`.
- Re-planning a slice against a sampler that already holds the previous
  plan's output → empty edit, `Dug { cost: 0 }`, never `Void`.
- `survey` against the fixture region used by `extract`'s own tests reads
  the same block `get-area` does at one coordinate.
- `cargo check`, `cargo test --lib`, `cargo clippy` clean.
