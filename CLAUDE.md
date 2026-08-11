# CLAUDE.md

This file gives Claude Code guidance for working in this repository.

## Project overview

`block_viewer` is a Rust/Bevy (0.15) application for viewing Minecraft Anvil
saves. It uses:

- `bevy` — engine/ECS, windowing, rendering
- `bevy_egui` — immediate-mode UI overlay
- `mc_anvil` (crate `ranvil`, local path dependency `../ranvil`) — reading
  Minecraft save/region files
- `rnbt` (local path dependency `../rnbt`) — NBT parsing

`src/main.rs` is the entry point; it loads local saves via `mc_anvil::get_saves`,
sets up a Bevy `App`, and renders block meshes with an orbit camera
(`src/pan_orbit_camera_bundle.rs`).

Note: `ranvil` and `rnbt` are sibling local crates (`../ranvil`, `../rnbt`),
not published dependencies — changes may require editing those repos too.

## Build / run

```
cargo build
cargo run
cargo check
```

## Ticket workflow

This repo uses a simple file-based ticket protocol for tracking work:

- **New work**: before starting a task, create a ticket file in `./tickets/`
  describing the work (one file per ticket, short descriptive filename, e.g.
  `tickets/fix-camera-pitch-clamp.md`).
- **In progress**: work happens against the ticket in `./tickets/`.
- **Done**: when a ticket's work is complete (implemented, and building/passing
  checks), move the file from `./tickets/` to `./finished_tickets/` — don't
  delete it.

Both directories are created on demand if they don't exist yet.
