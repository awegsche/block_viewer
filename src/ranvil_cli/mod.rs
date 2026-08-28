//! `ranvil-cli` (ticket 087, `RANVIL_CLI_ROADMAP.md`) — a headless,
//! windowless entry point into this crate's save/blueprint/edit logic, for
//! an agent to drive without a Bevy `App`. This module is scaffolding: the
//! argument parser ([`cli`]), the `--format text|json|compact` contract
//! ([`mod@format`]), the error/exit-code contract ([`error`]), shared
//! coordinate parsing ([`coords`]), and one real command ([`save::saves`])
//! proving all four end to end. Every later ticket adds one more
//! subcommand to [`cli::Command`] and one more arm to [`run`]'s dispatch.
//!
//! [`chunk`] gained `chunk`/`chunks` in ticket 089. [`block`], [`edit`],
//! [`structure`] are still empty stubs — later tickets fill them in.

use std::process::ExitCode;

use clap::Parser;

pub mod block;
pub mod chunk;
pub mod cli;
pub mod coords;
pub mod edit;
pub mod error;
pub mod format;
pub mod save;
pub mod structure;

use cli::{Cli, Command};
use error::CliError;
use format::print;

/// Parses `argv`, dispatches on [`Command`], and maps the result to an
/// exit code — the whole body of `src/bin/ranvil_cli.rs`'s `main`.
///
/// A `clap` parse failure (unknown flag, missing subcommand, `--help`) goes
/// through `clap`'s own reporting and exit code, same as any other `clap`
/// binary — the JSON error envelope below covers errors from *this
/// module's* logic (a bad `--instance` path, later a bad coordinate), not
/// `clap`'s own argument-syntax diagnostics.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    let format = cli.format;

    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => error::report(&err, format),
    }
}

/// Routes to each subcommand's implementation and prints its result.
/// Resolution of `--save` against `--instance` happens lazily, inside each
/// command that actually needs a resolved save — `saves` doesn't, which is
/// deliberately why it's this ticket's only variant (see `save.rs`).
fn dispatch(cli: &Cli) -> Result<(), CliError> {
    match &cli.command {
        Command::Saves(args) => {
            let result = save::saves(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Info(args) => {
            let result = save::info(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Regions(args) => {
            let result = save::regions(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Lock(args) => {
            let result = save::lock(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Chunk(args) => {
            let result = chunk::chunk(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Chunks(args) => {
            let result = chunk::chunks(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
    }
}
