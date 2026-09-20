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
//! This ticket (132) adds the bin shim, [`cli`] (the `clap` surface) and
//! [`list`] (`list`/`show`, both read-only over the registry) — the same
//! shape `ranvil_cli`'s own `run`/`dispatch` take, sharing its error/format
//! contract ([`crate::ranvil_cli::error::CliError`],
//! [`crate::ranvil_cli::format`]) rather than growing a second one. The rest
//! of the module tree the roadmap lays out (`allocate`, `markers`, `new`,
//! `export`, `import`, `remove`) arrives in later tickets.

use std::process::ExitCode;

use clap::Parser;

pub mod cli;
pub mod list;
pub mod registry;

use cli::{Cli, Command};
use crate::ranvil_cli::error::{report, CliError};
use crate::ranvil_cli::format::print;
use registry::RegistryError;

/// A bad registry is a bad argument, caught before anything is touched —
/// every [`RegistryError`] becomes [`CliError::Usage`], never
/// [`CliError::Data`].
impl From<RegistryError> for CliError {
    fn from(err: RegistryError) -> Self {
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
    }
}
