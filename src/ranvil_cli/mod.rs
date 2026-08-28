//! `ranvil-cli` (ticket 087, `RANVIL_CLI_ROADMAP.md`) — a headless,
//! windowless entry point into this crate's save/blueprint/edit logic, for
//! an agent to drive without a Bevy `App`. This module is scaffolding: the
//! argument parser ([`cli`]), the `--format text|json|compact` contract
//! ([`mod@format`]), the error/exit-code contract ([`error`]), shared
//! coordinate parsing ([`coords`]), and one real command ([`save::saves`])
//! proving all four end to end. Every later ticket adds one more
//! subcommand to [`cli::Command`] and one more arm to [`run`]'s dispatch.
//!
//! [`chunk`] gained `chunk`/`chunks` in ticket 089; [`heightmap`] adds
//! `heightmap` in ticket 090; [`block`] adds `get` in ticket 091, `get-area`
//! in ticket 092, and `column`/`scan` in ticket 093. Ticket 094 gave
//! [`edit`] the write substrate ([`edit::run_write`]) every write command
//! (095–097) and `struct import` (099) shares — lock, backup, dry-run,
//! force. Ticket 095 adds [`edit`]'s first two commands, `set`/`set-area`,
//! thin builders over `run_write` around `WorldEdit::new().set`/
//! `WorldEdit::fill` respectively. Ticket 096 adds `set-batch` (many
//! `x,y,z blockstate` lines from a file or stdin, parsed into one
//! `WorldEdit` before any of it is applied) and `replace` (a `get-area` scan
//! for a block name, rewritten to a new one) — both still one `run_write`
//! transaction, not a loop over 095's `set`. Ticket 097 adds `copy`:
//! `get_area`'s extraction over the source box, translated to a destination
//! and applied as one `run_write` transaction — the extraction finishes
//! before the write session opens, so an overlapping source/destination
//! reads the original blocks throughout.
//! Ticket 098 gives [`structure`] its first two commands, `struct info`/
//! `struct new` — the first `ranvil-cli` commands that read or write a
//! [`crate::blueprint::Blueprint`] on disk rather than a live save, so
//! unlike every command above they take no `--save`/`--instance` at all.
//! Ticket 099 adds `struct export`/`struct import`, the bridge back to a
//! live save: `export` is `get_area`'s box read written out through
//! `write_structure_file`; `import` is `read_structure_file` (optionally
//! rotated) turned into a `WorldEdit` and run through `run_write` — so
//! unlike `info`/`new` these two *do* take `--save`/`--instance`.
//! Ticket 100 adds `struct get`/`struct set`/`struct fill`, the first
//! commands that edit a [`crate::blueprint::Blueprint`] in memory outside of
//! extraction and rotation — no `--save`/`--instance` either, like `info`/
//! `new`, since everything they touch is a file on disk.
//! Ticket 101 adds `struct resize`: six independent per-face pads (positive
//! adds margin, negative crops), computed and applied in one pass over a
//! freshly allocated block array rather than reusing `struct fill`'s
//! in-place mutation — a resize changes the array's own shape, not just
//! positions within it.
//! Ticket 102 adds `struct rotate` (a thin wrapper over 038's
//! `rotate_blueprint`, already called by `struct import --rotate`) and
//! `struct diff` (the first `struct` command that only reads — two files in,
//! a per-position comparison out, refusing up front when their `size`s
//! disagree).

use std::process::ExitCode;

use clap::Parser;

pub mod block;
pub mod chunk;
pub mod cli;
pub mod coords;
pub mod edit;
pub mod error;
pub mod format;
pub mod heightmap;
pub mod save;
pub mod structure;

use cli::{Cli, Command, StructCommand};
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
        Command::Heightmap(args) => {
            let result = heightmap::heightmap(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Get(args) => {
            let result = block::get(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::GetArea(args) => {
            let result = block::get_area(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Column(args) => {
            let result = block::column(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Scan(args) => {
            let result = block::scan(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Set(args) => {
            let result = edit::set(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::SetArea(args) => {
            let result = edit::set_area(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::SetBatch(args) => {
            let result = edit::set_batch(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Replace(args) => {
            let result = edit::replace(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Copy(args) => {
            let result = edit::copy(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Info(args)) => {
            let result = structure::info(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::New(args)) => {
            let result = structure::new(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Export(args)) => {
            let result = structure::export(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Import(args)) => {
            let result = structure::import(cli, args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Get(args)) => {
            let result = structure::get(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Set(args)) => {
            let result = structure::set(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Fill(args)) => {
            let result = structure::fill(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Resize(args)) => {
            let result = structure::resize(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Rotate(args)) => {
            let result = structure::rotate(args)?;
            print(&result, cli.format);
            Ok(())
        }
        Command::Struct(StructCommand::Diff(args)) => {
            let result = structure::diff(args)?;
            print(&result, cli.format);
            Ok(())
        }
    }
}
