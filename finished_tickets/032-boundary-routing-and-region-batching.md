# 032 - Boundary routing and region batching (roadmap W5)

## Status
Done — implemented and tested. See the Resolution.

## Depends on

| needs | state |
|---|---|
| the chunk edit model, one region (W4) | done — ticket 031 |
| `ChunkRegion::is_dirty` / `dirty_chunks` (`ranvil` 010) | done |
| `ChunkRegion::save` (`ranvil` 009) | done — not called here, that's W6 |
| `RegionCache` (005-b) | exists, read-only |

## What this ticket is

Ticket 031 ends at one region file and refuses everything else
(`EditRefusal::OutsideRegion`). That refusal was deliberate — a position in a
neighbouring region would otherwise wrap into the region being edited via
`rem_euclid(512)` and put a building 512 blocks from where the user asked for
it. This ticket is what makes such an edit *legal*: route it to the right
files.

A 20x20 building straddles up to 4 chunks; placed near a region corner it
straddles up to 4 region files. The roadmap's requirements:

- group all changes by region file, apply every one of them, write each file
  **once**
- refuse (with a clear message, not a panic) any placement reaching into an
  ungenerated chunk — iteration 1 does not generate terrain
- `RegionCache` needs a mutable path and invalidation; the streaming pipeline
  must not go on serving pre-edit bytes

The first is routing, the second is mostly already true (031's
`ChunkNotGenerated`) and needs one new sibling for a region file that doesn't
exist at all, and the third is the `RegionCache` work.

## Design

### 1. Routing is a split, and it can't fail

```rust
pub fn route(edit: &WorldEdit) -> BTreeMap<(i32, i32), WorldEdit>
```

One sub-edit per region, each carrying the parent's `data_version`, each
keeping the parent's relative order so **last write wins** survives the split
(two edits at the same position are in the same region by construction, so the
rule needs no cross-region reasoning at all).

`route` validates nothing. Every refusal stays in `plan`, which now runs once
per region — a build-limit violation is a property of a position, not of the
routing, and duplicating the check would mean two places to keep in step.

### 2. The regions come from a trait, not from `RegionCache`

```rust
pub enum RegionUnavailable { NotGenerated, Unreadable(String) }

pub trait RegionSource {
    fn region_mut(&mut self, coord: (i32, i32)) -> Result<&mut ChunkRegion, RegionUnavailable>;
    fn discard(&mut self, coord: (i32, i32));
}
```

Two reasons this isn't just `&mut RegionCache`:

- The cache needs a real `SaveMeta` and a directory of `.mca` files. A trait
  lets the routing rules be tested against a `BTreeMap` of fixture regions with
  no save anywhere — the same instinct 031 followed when it built a synthetic
  region instead of copying the developer's world.
- The citybuilder and the viewer will not necessarily reach a save the same
  way, and W6 will want to wrap this with the backup/lock layer rather than
  teach `RegionCache` about either.

`discard` is on the trait because rollback needs it — see 4.

### 3. Two phases, because a half-applied *building* is worse than a refused one

`set_blocks` is all-or-nothing per call and 031's `apply` is all-or-nothing per
region. Neither gives all-or-nothing across four region files: region 1 applying
and region 4 refusing leaves three quarters of a building standing.

So `apply_routed` is:

1. **Plan every region.** Load each one, run 031's `plan`, collect the reports.
   Nothing is mutated, so a refusal here costs nothing.
2. **Apply every region.** Only reached if all of phase 1 passed.

Phase 1 can't make phase 2 infallible — `set_blocks` can still reject a batch
the preflight accepted (031's `EditRefusal::Rejected`) — so phase 2 keeps a
rollback: on a failure, **discard every region this transaction touched** from
the source, dropping the in-memory mutations. Nothing has been saved (`apply`
never calls `save`, by 031's design), so re-reading those regions from disk *is*
the rollback, and it's total in a way an undo log wouldn't be.

Note that a region evicted *between* the two phases is harmless: phase 2's
`apply` re-runs `plan` internally, so a re-loaded region is re-validated rather
than trusted. What must not happen is a region being evicted *during* phase 2
after it was mutated — which is exactly what the eviction guard below prevents.

### 4. The rollback only works if nothing else is unsaved

Discarding a region drops **every** unsaved change in it, not only this
transaction's. If a previous placement is still sitting dirty in the cache, the
rollback would silently destroy it too.

So the preflight refuses a target region that is already dirty:
`EditRefusal::RegionHasUnsavedChanges`. The contract this states — *one
transaction at a time, saved before the next* — is the one W6 is going to
implement anyway, and stating it here makes the rollback honest instead of
approximately correct. `EditPolicy::allow_dirty_regions` opts out for a caller
that knows it's batching (and accepts that a failure then rolls back further
than it caused).

Read by the routed entry points only; single-region `apply` keeps its current
behaviour, which several 031 tests rely on (the undo test applies twice).

### 5. `RegionCache`: a mutable path, an eviction guard, and invalidation

- **`get_or_load_mut`** — same load-on-miss path as `get_or_load`, returning
  `&mut`. Factored so there's one copy of the load, not two.
- **`evict_lru` skips dirty regions.** `ChunkRegion::is_dirty`'s doc comment
  upstream names this cache specifically; evicting an edited region silently
  drops the edit, and with a small capacity (`recommended_capacity(10)` is 9)
  a four-region building plus streaming traffic can hit that in one frame. If
  *every* resident region is dirty, capacity is exceeded rather than an edit
  lost — the overshoot is bounded by how much the user edited before saving.
- **`discard`** — drop a region, dirty or not. The rollback path, and loud in
  its doc comment about what that means.
- **`dirty_regions()`** — which regions have unsaved changes. W6 saves them;
  G2's city panel displays them.

**Invalidation** is smaller than it sounds: the cache *holds* the mutated
`ChunkRegion`, so every later `get_or_load` already serves post-edit NBT.
Nothing needs invalidating there. What is stale is `DecodedWorld`'s already-
decoded columns and their meshes — and that's W7, deliberately: the edit
reports the chunks it touched (`EditReport::chunks`) and W7 marks them dirty
through 005-f's existing re-mesh queue.

### 6. `EditReport::regions` comes back

031 dropped the field because a single-region apply has exactly one. Here it
means something. Merging the per-region reports: `blocks_written` sums (a
position belongs to exactly one region, so there's nothing to dedupe across
them), and `chunks` and `replaced` are concatenated **and re-sorted** —
per-region order is ascending within a region, but region-major order isn't
globally ascending once a region boundary is crossed on Z.

## Explicitly not in this ticket

- **Saving, backups, `session.lock`, atomicity** — W6. Nothing here writes to
  disk; regions are left dirty for the caller to save.
- **Re-meshing the edited chunks** — W7.
- **Any UI** — W8.
- **Terrain generation.** A placement reaching an ungenerated chunk *or an
  ungenerated region* is refused.

## Tests

Pure, against fixture regions in a `BTreeMap` (no save, no cache):

- `route` splits by region, keeps relative order per region, propagates
  `data_version`, and puts x = -1 in region -1 rather than region 0.
- A building straddling the four regions that meet at the world origin applies
  to all four, with the right blocks in each — the negative-coordinate case and
  the four-way case in one test.
- One bad region refuses the whole transaction and **no** region is mutated or
  dirtied.
- A `Rejected` failure in phase 2 rolls the already-applied regions back
  (discarded), so nothing half-applied survives.
- An already-dirty target region is refused, and `allow_dirty_regions` lets it
  through.
- The merged report: regions, chunks and `replaced` sorted, blocks summed.

Against a real `RegionCache` over a temp-directory save:

- `get_or_load_mut` edits are visible to a later `get_or_load`.
- Capacity 1, two regions edited: both stay resident (the guard) and both are
  still dirty.
- Once saved, the guard stops applying and normal LRU eviction resumes.
- `discard` drops unsaved changes: the next load re-reads the original blocks.
- A region the save doesn't have is `RegionNotGenerated`, not a panic.

## Done when

- An edit spanning four region files applies to all four, or to none.
- `RegionCache` has a mutable path and cannot evict unsaved work.
- Nothing in the viewer or the citybuilder behaves differently yet — W8 is the
  first caller, as it was for 031.

## Resolution

`src/edit/route.rs` (~240 lines with the docs), the `RegionCache` additions,
and 13 new tests in `src/edit/tests.rs`. `cargo test`: 222 pass. No new clippy
warnings.

### Deviations from the design above

**1. `RegionCache::is_resident`.** The eviction guard and `discard` are both
statements about what is *not* in the cache, and `len()` alone can't tell which
region went. One line, and W6 wants it anyway before deciding whether saving a
region needs loading it first.

**2. `EditReport::regions` is filled in by the single-region `plan` too**,
rather than only by the routed path. 031 dropped the field as noise; giving it
the one region it has makes merging uniform instead of special-casing an empty
vector, and it costs a `vec![]`.

### Confirmed while building it

- **The phase-2 failure is real, not hypothetical.** A section carrying a `Y`
  tag and no `block_states` passes `check_set_block` — which only asks whether
  a section covering that Y exists — and is refused by `set_blocks` when
  `edit_section` goes looking for the palette. So the rollback test needs no
  lying test double: `SaveFixture::corner_with_a_malformed_origin` is a
  legitimately malformed chunk, and it lands in region `(0,0)`, the last of the
  four the routed apply visits, with three regions already applied when it
  fails. This is also the concrete answer to "why does `EditRefusal::Rejected`
  exist if the preflight is thorough": the preflight validates what the *edit
  model* knows about, and the Anvil format has failure modes underneath it.
- **The eviction guard is load-bearing, not defensive.** With capacity 1 (the
  test uses exactly that), the second region of a two-region edit would evict
  the first — which is already dirty — and `apply_routed` would return success
  reporting two blocks written with one of them gone. Silent, and reported as a
  success, which is the worst shape a bug can have here.
- **A test flake worth the ten minutes it cost.** `dirty_regions()` iterates a
  `HashMap`, and `get_or_load_mut` touches the LRU order, so "save everything
  dirty" leaves the least-recently-used region up to hash order. The test now
  says which region it means instead of assuming; the same trap is waiting for
  W6's save-all loop, which should not care about order but will touch it.

### Next

W6: write safety — refuse a world Minecraft has open (`SaveMeta::is_locked` /
`SessionLock::acquire`, `ranvil` 016), back up each region file before its
first write of the session, and save the dirty regions this ticket leaves
behind. The contract W5 states — one transaction at a time, saved before the
next — is the one W6 implements; `RegionCache::dirty_regions` is its input.
