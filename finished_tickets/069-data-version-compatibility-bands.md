# 069 — `DataVersion` compatibility bands instead of exact equality

## The bug

Drawing a road at some positions in a real save refuses the whole drag with

```
chunk (-40, 12) is DataVersion 4438, the edit is 4903
```

and writes nothing — not even the cells over chunks that would have accepted.

## Why

`check_chunk_nbt` (`src/edit/mod.rs`) compares the edit's `DataVersion` to the
chunk's for **exact** equality. Both halves of that comparison are outside the
user's control and routinely differ in a save that has been played for a while:

- Every shipped `.nbt` asset (all seven `assets/city/roads/dirt/*.nbt`, and
  `assets/city/blueprints/house01.nbt`) was exported at `DataVersion` 4903, and
  `commit::blueprint_edit` stamps that onto the `WorldEdit`.
- A Minecraft world only rewrites the chunks it actually loads, so a world
  played across updates is a *patchwork* of versions. The `nbt_test` save's
  overworld: 9327 chunks at 4438, 12 at 4440, 9215 at 4903, spatially mixed —
  `r.-2.-1.mca` is 1024/1024 at 4438, `r.0.-2.mca` is 874/874 at 4903, and
  `r.0.0.mca` is 479 at 4438 against 383 at 4903.

So roughly half the world refuses every placement, in patches, which is the
reported "at certain positions". And because `edit::route::apply_routed` plans
every region before applying any, one refusing chunk anywhere along a drag
fails the entire transaction.

Exact equality is also stricter than the thing it protects against. The check
exists because *block names and properties migrate between versions* — 4438
and 4903 are both 1.21-era and rename nothing, so refusing there buys no
safety and costs the placement.

## The fix

Compare **compatibility bands**, not version numbers. A small, explicit,
sorted table of the `DataVersion`s at which block names/properties actually
migrated splits the version line into bands; two versions are compatible when
they land in the same band.

- 4438 vs 4903 → same band → the write proceeds, silently and correctly.
- 3953 (1.21) vs 2860 (1.18) → different bands → still refused.
- A 1.12 chunk vs any post-Flattening edit → still refused, loudly.

`EditPolicy::enforce_data_version` keeps its meaning and its override; only
the comparison inside it changes. `EditRefusal::DataVersionMismatch` keeps its
fields (the caller still needs to name the chunk and both versions) and gets a
message that says *incompatible* rather than merely *different*.

**Scope: not** partial-drag placement. A refused chunk still fails the whole
transaction — that is the roadmap's transaction model and it stays.

## Done when

- A boundary table lives in `src/edit`, documented with what each entry is and
  the rule for adding one when a new Minecraft release renames blocks.
- `check_chunk_nbt` refuses only cross-band mismatches.
- Tests cover: same band accepted, cross-band refused, either version missing
  accepted, the policy override, and the band function's edges (a version
  exactly on a boundary belongs to the newer band).
- `cargo check` and `cargo test` pass.
