//! `model-exporter` (tickets 131–137, `MODEL_EXPORTER_ROADMAP.md`) — a
//! fourth, headless entry point that bridges the two ways a building model
//! gets made today: a human building it in Minecraft, or an agent writing it
//! directly with `ranvil-cli struct`. `assets/models/*.ron` is the shared
//! registry: one `world.ron` naming the *models world* (an empty save the
//! user provides) plus one `<name>.ron` per model, recording where its box
//! sits in that world.
//!
//! [`registry`] is this ticket (131): the RON schema and the loader that
//! validates a whole `assets/models` directory at once — no CLI, no world
//! I/O, nothing beyond reading and writing `.ron` files. `run()` and the
//! rest of the module tree the roadmap lays out (`cli`, `allocate`,
//! `markers`, `list`, `new`, `export`, `import`, `remove`) arrive starting
//! with ticket 132.

pub mod registry;
