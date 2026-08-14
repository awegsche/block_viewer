# 028 - follow `SaveMeta::region_dir` from ranvil

## Status
Done — see the Resolution at the bottom.

## Where
`src/lib.rs` (`empty_save`), `src/region_cache.rs` (tests),
`src/chunk_pipeline.rs` (tests).

## Problem
The latest Minecraft version writes the overworld's region files to
`<save>/dimensions/minecraft/overworld/region` instead of `<save>/region`,
so `mc_anvil::get_saves` skipped such saves entirely and the app came up
with "(no save loaded)". The fix belongs in `ranvil` (its ticket 026): it
resolves the region directory per layout and stores it on `SaveMeta` as a
new `region_dir` field.

This crate reads regions only through `SaveMeta::get_region_path`, so it
needs no path logic of its own — but it does build `SaveMeta` literals in
three places, which no longer compile without the new field.

## Suggested fix
Add `region_dir` to the three literals: the empty placeholder save
(`PathBuf::new()`, matching its empty `path`) and the two synthetic
"this save's files don't exist" test metas (`<path>/region`, the legacy
layout, which is what those paths were already implying).

## Tests
Covered by the existing suite — the literals are used by tests that already
assert the failure behavior they stage. The layout resolution itself is
tested in `ranvil` (`tests/save_tests.rs`), the crate that owns it.

## Resolution
Done alongside ranvil 026. `cargo test` passes here (181 tests).

Not verified by running the app: whether the app's save picker now lists and
loads the real `nbt_test` save is a look-at-the-window check — noted in
`todo.md`. `cargo run --example region_info` in `../ranvil` does read that
save's 29 regions.
