//! The `clap` argument surface (ticket 087): the top-level [`Cli`], the
//! [`Command`] enum every subcommand hangs off, and each subcommand's own
//! args struct. `save`/`instance`/`format` are declared `global = true` so
//! they parse whether typed before or after the subcommand
//! (`ranvil-cli --format json saves` and `ranvil-cli saves --format json`
//! both work).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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

/// One variant per subcommand. Ticket 087 added `Saves`; ticket 088 adds
/// `Info`/`Regions`/`Lock`, the first three commands that need `--save`
/// resolved (see `save::resolve_save`) rather than listing every save.
/// Every later `ranvil-cli` ticket adds one more, routing to its own
/// submodule the same way.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// List Minecraft saves under an instance directory.
    Saves(SavesArgs),
    /// Save-level summary: DataVersion, dimensions present, region count,
    /// locked.
    Info(InfoArgs),
    /// Region files present: coordinates, file size, chunk count.
    Regions(RegionsArgs),
    /// Whether Minecraft is holding this save's `session.lock` right now.
    Lock(LockArgs),
}

/// `saves` takes no arguments of its own — the instance directory it lists
/// comes from the global `--instance` above, since that flag is shared by
/// every later command that needs one too.
#[derive(Debug, Args)]
pub struct SavesArgs {}

/// `info` takes no arguments of its own — the save it summarizes comes from
/// the global `--save`/`--instance` above, resolved by `save::resolve_save`.
#[derive(Debug, Args)]
pub struct InfoArgs {}

/// `lock` takes no arguments of its own, same reason as [`InfoArgs`].
#[derive(Debug, Args)]
pub struct LockArgs {}

#[derive(Debug, Args)]
pub struct RegionsArgs {
    /// Rendering layout for `text`/`compact` output: `list` (default, one
    /// line per region, sorted by `(x, z)`) or `grid`
    /// (`SaveMeta::get_grid_view`'s ASCII map). `Option` rather than a
    /// defaulted value so `regions::regions` can tell "not given" (compatible
    /// with any `--format`) apart from an explicit `--layout grid`
    /// (incompatible with `--format json`, which is always the list shape as
    /// an array of objects) — see the roadmap's note on why `grid` is a
    /// `--layout` value rather than a fourth [`super::format::OutputFormat`].
    #[arg(long, value_enum)]
    pub layout: Option<RegionLayout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RegionLayout {
    List,
    Grid,
}
