//! `model-exporter` (tickets 131–137, `MODEL_EXPORTER_ROADMAP.md`) — a
//! fourth, headless entry point that bridges the two ways a building model
//! gets made today: a human building it in Minecraft, or an agent writing it
//! directly with `ranvil-cli struct`. `assets/models/*.ron` is the shared
//! registry: one `world.ron` naming the *models world* (an empty save the
//! user provides) plus one `<name>.ron` per model, recording where its box
//! sits in that world.
//!
//! [`registry`] is 131: the RON schema and the loader that validates a whole
//! `assets/models` directory at once — no CLI, no world I/O, nothing beyond
//! reading and writing `.ron` files.
//!
//! Ticket 132 added the bin shim, [`cli`] (the `clap` surface) and [`list`]
//! (`list`/`show`, both read-only over the registry) — the same shape
//! `ranvil_cli`'s own `run`/`dispatch` take, sharing its error/format
//! contract ([`crate::ranvil_cli::error::CliError`],
//! [`crate::ranvil_cli::format`]) rather than growing a second one.
//!
//! Ticket 133 added [`allocate`] (the first-fit slot allocator, a pure
//! function with its own tests) and [`new`] (the thin command wrapped around
//! it, until this ticket always registering a slot without touching the
//! world). Ticket 134 added [`markers`] — the first command that writes to
//! the models world: `marker_edit`/`marker_positions` (the ring and
//! corner-pillar geometry as a [`crate::edit::WorldEdit`]), `new` now
//! placing them via [`crate::ranvil_cli::edit::run_write`] unless
//! `--no-markers`, and `mark <name>` to re-place an already-registered
//! slot's markers. This ticket (135) adds [`export`] — the command the tool
//! is named for: every registered slot's world box read into its `.nbt` in
//! one call. The rest of the module tree the roadmap lays out (`import`,
//! `remove`) arrives in later tickets.

use std::process::ExitCode;

use clap::Parser;

pub mod allocate;
pub mod cli;
pub mod export;
pub mod list;
pub mod markers;
pub mod new;
pub mod registry;

use cli::{Cli, Command};
use crate::ranvil_cli::error::{report, CliError};
use crate::ranvil_cli::format::print;
use allocate::AllocateError;
use registry::RegistryError;

/// A bad registry is a bad argument, caught before anything is touched —
/// every [`RegistryError`] becomes [`CliError::Usage`], never
/// [`CliError::Data`].
impl From<RegistryError> for CliError {
    fn from(err: RegistryError) -> Self {
        CliError::Usage(err.to_string())
    }
}

/// [`AllocateError`] is always a bad request too: either the size was wrong
/// (caught before any read), or `world.area`/generated-chunk coverage can't
/// fit it right now — both are things the caller needs to change, not a read
/// that failed against ground that *was* found.
impl From<AllocateError> for CliError {
    fn from(err: AllocateError) -> Self {
        CliError::Usage(err.to_string())
    }
}

/// Parses `argv`, dispatches on [`Command`], and maps the result to an exit
/// code — the whole body of `src/bin/model_exporter.rs`'s `main`. Mirrors
/// [`crate::ranvil_cli::run`] exactly.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    let format = cli.format;

    match dispatch(&cli) {
        Ok(code) => code,
        Err(err) => report(&err, format),
    }
}

/// Routes to each subcommand's implementation and prints its result — the
/// same shape `ranvil_cli`'s own (private) `dispatch` takes.
fn dispatch(cli: &Cli) -> Result<ExitCode, CliError> {
    match &cli.command {
        Command::List(args) => {
            let result = list::list(cli, args)?;
            print(&result, cli.format);
            Ok(ExitCode::SUCCESS)
        }
        Command::Show(args) => {
            let result = list::show(cli, args)?;
            print(&result, cli.format);
            Ok(ExitCode::SUCCESS)
        }
        Command::New(args) => {
            let result = new::new(cli, args)?;
            print(&result, cli.format);
            Ok(ExitCode::SUCCESS)
        }
        Command::Mark(args) => {
            let result = markers::mark(cli, args)?;
            print(&result, cli.format);
            Ok(ExitCode::SUCCESS)
        }
        Command::Export(args) => {
            let result = export::export(cli, args)?;
            print(&result, cli.format);
            Ok(if result.any_failed() { ExitCode::from(1) } else { ExitCode::SUCCESS })
        }
    }
}
