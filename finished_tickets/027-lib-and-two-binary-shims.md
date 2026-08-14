# 027 - `lib.rs` and two binary shims

Roadmap task **L1** (`tickets/CITYBUILDER_ROADMAP.md`). "Do this first and
alone" — it touches every file's module path, so overlapping it with real
work makes every later diff unreadable.

## Status
Done — implemented as described. `cargo check --all-targets` and
`cargo test` (181 tests) pass; both binaries build.

## Motivation

The citybuilder and the viewer are two games over one shared core (world
decode, meshing, atlas, tint, streaming, the region cache, the sky, the
camera). Today all of that lives under `src/main.rs`'s private module tree,
reachable only from the one binary. Nothing can grow a second entry point
without first turning the tree into a library.

This is milestone **M1**: the citybuilder binary opens a window on a real
save with the existing streaming and camera. Nothing else — no game code
lands in this ticket.

## The layout

```
src/
  lib.rs              module tree + the resources and startup wiring both
                      games share (LoadedSave, DecodedWorld, world_app())
  world/              unchanged
  blueprint/          unchanged
  selection/          unchanged
  camera.rs, sky/, streaming.rs, chunk_pipeline.rs, region_cache.rs,
  unload.rs           unchanged
  viewer/             mod.rs = today's main() body as `run()`, + ui/
  viewer/ui/          moved from src/ui/
  city/               NEW: mod.rs = `run()`, the M1 window
  bin/block_viewer.rs fn main() { block_viewer::viewer::run() }
  bin/citybuilder.rs  fn main() { block_viewer::city::run() }
```

`src/main.rs` goes away; `[[bin]]` entries in `Cargo.toml` replace it.

## The judgement call the roadmap leaves open

Which modules the citybuilder shares. The roadmap's starting suggestion is
"everything except `viewer::{selection, ui}`". `ui` moves as written. But
`selection` **stays at the crate root**, because the split falls out of the
existing dependencies rather than the sketch:

- `blueprint::extract` already takes a `SelectionBounds` — a shared module
  cannot depend on a viewer-only one.
- `SelectionBounds` is where 019 fixed the coordinate rules
  (`bevy.z = -mc.z`, inclusive bounds) that roadmap task W4 says to inherit
  rather than reinvent.
- Roadmap I2 reuses `blueprint::extract` — and therefore `SelectionBounds`
  — for the damage scan, from the citybuilder.

What *is* viewer-only about selection is its interaction (`gizmo`, `input`)
and the panel that drives it. Splitting the module along that line is a
later refactor if it earns itself; it is not this ticket's mechanical move.

`blueprint::BlueprintPlugin` stays root-level as a module but is only added
to the app by the viewer, since the only thing that starts an extraction
today is the selection panel.

## What `world_app()` owns

The shared half of today's `main()`, so `city::run()` is not a copy-paste of
`viewer::run()`:

- load the save (`load_real_save`), build the `App`, `DefaultPlugins`,
  frame-time diagnostics
- `camera`, `sky`, `streaming`, `chunk_pipeline`, `unload` plugins
- `LoadedSave` / `StartupIssue` / `DecodedWorld` resources
- the `Startup` system that builds the atlas and colormaps, inserts the
  region cache and terrain material, and spawns the camera, sun and sky
  scene

`viewer::run()` then adds `selection`, `blueprint`, `ui` and the
`CameraSet`/`SelectionInputSet` ordering against `UiPanelSet` (which only
matters where egui exists). `city::run()` adds nothing yet.

## Not in scope

- Any visibility churn. Both games live *inside* the lib, so `pub(crate)`
  keeps working throughout; only `viewer::run` and `city::run` need to be
  `pub`.
- The `mc_core` / `block_viewer` / `citybuilder` workspace split. Still
  available later, once the shared core stops changing shape.
- Any citybuilder gameplay.

## Verification

- `cargo check` and `cargo test` pass; both binaries build.
- Manual: `cargo run --bin block_viewer` behaves exactly as before, and
  `cargo run --bin citybuilder` opens a window streaming the same terrain
  with no UI panels. Noted in `todo.md` — Claude does not drive the app.

## Resolution notes

**Doc rot swept along the way.** Roughly thirty doc comments pointed at
`main.rs::setup`, `main.rs::load_real_save`, `main.rs::spawn_point` or
"`main.rs` orders this set". They now point at `lib.rs::setup_world`,
`lib.rs::*` and `viewer::run`. Two references to `main.rs` survive on
purpose, both historical: `camera.rs`'s note on where the old `orbit`
system lived, and `viewer/mod.rs`'s note on what `run()` used to be.

**A lint the move switched on.** Making the module tree `pub` turned every
"public doc links to a private item" into a rustdoc warning — 70 of them,
where a binary's private modules had produced none. This is an application
whose tree is `pub` only so two shims can reach `run()`, and whose docs are
written for `--document-private-items`, so `lib.rs` carries
`#![allow(rustdoc::private_intra_doc_links)]` with that reasoning next to
it. Six *genuinely* broken links surfaced underneath and were fixed for
real (`crate::sky::sync_sky_palette`, `SkyPlugin` from `sky::dome`,
`world::ChunkColumn` from `unload`, `camera::drive_camera` from two places,
`super::atlas::resolve_faces` from `world::tint`, and one redundant
explicit link target in `chunk_pipeline`). `cargo doc --no-deps
--document-private-items` is now warning-clean, which it wasn't before.

**Not run: `cargo fmt`.** The repo has pre-existing edition-2024 rustfmt
drift in ~30 files (import ordering, trailing semicolons in match arms).
Formatting as part of this ticket would have buried a mechanical move under
an unrelated whole-repo diff. Worth its own commit.
