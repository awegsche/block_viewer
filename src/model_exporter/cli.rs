//! The `clap` argument surface for `model-exporter` (ticket 132) — mirrors
//! `ranvil_cli::cli`'s shape (a [`Cli`] with global flags plus a [`Command`]
//! subcommand enum) so an agent driving both tools sees one convention, not
//! two. `--save`/`--instance` are optional overrides of
//! `assets/models/world.ron`'s own `save` field
//! ([`crate::ranvil_cli::save::resolve_save_from`]'s `save` argument is
//! `cli.save.as_deref().or(Some(world.save.as_str()))`, not `world.save`
//! alone) — most invocations never need them, since the registry already
//! names the models world.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::ranvil_cli::format::OutputFormat;

#[derive(Debug, Parser)]
#[command(
    name = "model-exporter",
    about = "The models world registry: what's registered, where it is, and the bridge to assets/city/blueprints"
)]
pub struct Cli {
    /// Directory holding `world.ron` and one `<name>.ron` per registered
    /// model.
    #[arg(long, global = true, default_value = "assets/models")]
    pub models_dir: PathBuf,

    /// Overrides `world.ron`'s own `save` field — same semantics as
    /// `ranvil-cli --save`: a save name under the resolved instance
    /// directory, or a path straight to one save.
    #[arg(long, global = true)]
    pub save: Option<String>,

    /// The saves/instance directory to look in. Defaults to
    /// `dirs::config_dir()/.minecraft/saves`, same as `ranvil-cli
    /// --instance`.
    #[arg(long, global = true)]
    pub instance: Option<PathBuf>,

    /// Output format — reuses `ranvil_cli::format::OutputFormat` directly so
    /// both CLIs' `--format json`/`compact` mean exactly the same thing.
    #[arg(long, global = true, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Command,
}

/// One variant per subcommand. Ticket 132 adds `List`/`Show`, the two
/// read-only commands that prove the format/error contract end to end;
/// later tickets add `New`, `Export`, `Import`, `Remove`, `Mark`.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// One row per registered slot: name, origin, size, box, `.nbt` status,
    /// `/tp` line. Does not open the models world — works with Minecraft
    /// running.
    List(ListArgs),
    /// One slot, in full: everything `list` shows plus the marker ring's
    /// geometry and the equivalent `ranvil-cli get-area` line.
    Show(ShowArgs),
    /// Finds a free spot, registers it, and prints where it is. Markers are
    /// ticket 134's job — until it lands, `new` always behaves as
    /// `--no-markers` and says so.
    New(NewArgs),
}

/// `list` takes no arguments of its own — the registry it reads comes from
/// the global `--models-dir` above.
#[derive(Debug, Args)]
pub struct ListArgs {}

/// `show <name>`.
#[derive(Debug, Args)]
pub struct ShowArgs {
    /// A registered model's name — its `.ron` file's stem.
    pub name: String,
}

/// `new <name> <width> <height> <depth> [--below N] [--no-markers]
/// [--dry-run]`. `width`/`height`/`depth` are the blueprint's `x`/`y`/`z`
/// extents — the same order `struct new --size x,y,z` takes.
#[derive(Debug, Args)]
pub struct NewArgs {
    /// The model's name: must match `[a-z0-9_]+` (it also becomes the
    /// `.ron`/`.nbt` file stem) and must not already be registered.
    pub name: String,
    /// The blueprint's `x` extent.
    pub width: i32,
    /// The blueprint's `y` extent.
    pub height: i32,
    /// The blueprint's `z` extent.
    pub depth: i32,
    /// Foundation layers under the ground surface: `origin.y = ground_y -
    /// below`. The default, `1`, is the `ground_level: 1` convention
    /// `gatherer_hut.ron`/`mine01.ron` already use (the blueprint's `y=0` is
    /// that foundation layer, the surface is `y=1`); `--below 0` gives
    /// `house01.ron`'s "y=0 is the surface" shape instead.
    #[arg(long, default_value_t = 1)]
    pub below: u32,
    /// Register the slot without placing marker blocks. This is the only
    /// mode available until ticket 134 lands — `new` behaves as if this were
    /// always set, and says so in its output.
    #[arg(long)]
    pub no_markers: bool,
    /// Compute the allocation and print it without writing the `.ron`.
    #[arg(long)]
    pub dry_run: bool,
}
