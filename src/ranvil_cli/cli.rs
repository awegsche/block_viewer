//! The `clap` argument surface (ticket 087): the top-level [`Cli`], the
//! [`Command`] enum every subcommand hangs off, and each subcommand's own
//! args struct. `save`/`instance`/`format` are declared `global = true` so
//! they parse whether typed before or after the subcommand
//! (`ranvil-cli --format json saves` and `ranvil-cli saves --format json`
//! both work).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use super::format::OutputFormat;

#[derive(Debug, Parser)]
#[command(
    name = "ranvil-cli",
    about = "Headless inspection and editing of Minecraft Anvil saves and structure files"
)]
pub struct Cli {
    /// A save name (under the resolved instance directory) or a path
    /// straight to one save. Not every command needs this resolved (e.g.
    /// `saves` itself doesn't) — see `save.rs`.
    #[arg(long, global = true)]
    pub save: Option<String>,

    /// The saves/instance directory to look in. Defaults to
    /// `dirs::config_dir()/.minecraft/saves`, same as `mc_anvil::get_saves`.
    #[arg(long, global = true)]
    pub instance: Option<PathBuf>,

    /// Output format — see `format.rs`.
    #[arg(long, global = true, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Command,
}

/// One variant per subcommand. This ticket adds exactly one (`Saves`);
/// every later `ranvil-cli` ticket adds one more, routing to its own
/// submodule the same way.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// List Minecraft saves under an instance directory.
    Saves(SavesArgs),
}

/// `saves` takes no arguments of its own — the instance directory it lists
/// comes from the global `--instance` above, since that flag is shared by
/// every later command that needs one too.
#[derive(Debug, Args)]
pub struct SavesArgs {}
