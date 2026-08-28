//! The `clap` argument surface (ticket 087): the top-level [`Cli`], the
//! [`Command`] enum every subcommand hangs off, and each subcommand's own
//! args struct. `save`/`instance`/`format` are declared `global = true` so
//! they parse whether typed before or after the subcommand
//! (`ranvil-cli --format json saves` and `ranvil-cli saves --format json`
//! both work).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use mc_anvil::heightmap::HeightmapKind;

use super::coords::{BlockPos, ChunkPos, ColumnPos};
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
/// Ticket 089 adds `Chunk`/`Chunks`, the first commands that decode a
/// chunk's own NBT rather than just a region's metadata. Ticket 090 adds
/// `Heightmap`, the first command that returns a per-column (rather than
/// per-chunk-summary) grid.
/// Ticket 091 adds `Get`, the first command that decodes a section's packed
/// block-state indices rather than just chunk/heightmap metadata. Ticket 092
/// adds `GetArea`, the same read over a box rather than one block — a
/// `SelectionBounds`/`extract_blueprint` call under a CLI wrapper (see
/// `block::get_area`), and the shared box-scan primitive later `copy`/
/// `struct export` tickets reuse rather than growing their own. Ticket 093
/// adds `Column`/`Scan`, both built on `get_area` rather than a fresh box
/// walk — `column` reads a single-column box top-to-bottom, `scan` filters a
/// box's blocks by name.
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
    /// One chunk: Status, DataVersion, section Y-range, inhabited time,
    /// block-entity/entity counts, biome list, a heightmap peek.
    Chunk(ChunkArgs),
    /// Bulk survey over a rectangle of chunk coordinates: counts by
    /// `Status`, ungenerated chunks, aggregate block-entity/entity counts.
    Chunks(ChunksArgs),
    /// One chunk's full 16×16 heightmap grid, for one of the four kinds.
    Heightmap(HeightmapArgs),
    /// One block's name + properties.
    Get(GetArgs),
    /// A box of blocks, as a palette + dense index array.
    GetArea(GetAreaArgs),
    /// One column's blocks, bottom-to-top by default, with consecutive
    /// identical blocks collapsed into ranges.
    Column(ColumnArgs),
    /// Every position in a box whose block matches a given name.
    Scan(ScanArgs),
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

/// `chunk <cx>,<cz> [--print]` (ticket 089).
#[derive(Debug, Args)]
pub struct ChunkArgs {
    /// Chunk coordinates, "cx,cz". `allow_hyphen_values`: clap's own
    /// negative-number heuristic doesn't fire for `-2,-2` (it isn't a bare
    /// negative integer), and the roadmap's own example invocations use
    /// negative chunk coordinates without a `--` separator.
    #[arg(allow_hyphen_values = true)]
    pub pos: ChunkPos,

    /// Accepted for compatibility with the invocation the roadmap and
    /// earlier planning discussion spelled (`chunk 0,0 --print --format
    /// json`) — it changes nothing. `chunk` always prints its result; this
    /// flag is a no-op kept so that exact invocation keeps working.
    #[arg(long)]
    pub print: bool,
}

/// `chunks <cx1>,<cz1> <cx2>,<cz2>` (ticket 089): a rectangle of chunk
/// coordinates, inclusive both ends, matching 019's inclusive-bounds
/// convention. Either corner may be given in either order — `chunk::chunks`
/// sorts them into min/max before walking.
#[derive(Debug, Args)]
pub struct ChunksArgs {
    /// One corner, "cx,cz". See [`ChunkArgs::pos`] on why `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub from: ChunkPos,
    /// The opposite corner, "cx,cz".
    #[arg(allow_hyphen_values = true)]
    pub to: ChunkPos,
}

/// `heightmap <cx>,<cz> [--kind world-surface|motion-blocking|
/// motion-blocking-no-leaves|ocean-floor]` (ticket 090, default `world-surface`).
#[derive(Debug, Args)]
pub struct HeightmapArgs {
    /// Chunk coordinates, "cx,cz". See [`ChunkArgs::pos`] on why
    /// `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub pos: ChunkPos,

    /// Which of the four heightmaps to read.
    #[arg(long, value_enum, default_value = "world-surface")]
    pub kind: HeightmapKindArg,
}

/// The `--kind` values `heightmap` accepts — a CLI-facing mirror of
/// [`HeightmapKind`], kept separate rather than making `HeightmapKind` itself
/// a `ValueEnum` since `ranvil` has no `clap` dependency of its own (the
/// architectural rule the roadmap sets: format-crate types don't grow
/// CLI-only derives). `ValueEnum`'s default `kebab-case` rename gives exactly
/// the roadmap's spelling (`world-surface`, `motion-blocking-no-leaves`, ...)
/// with no `#[value(rename_all = ...)]` needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum HeightmapKindArg {
    WorldSurface,
    MotionBlocking,
    MotionBlockingNoLeaves,
    OceanFloor,
}

/// `get <x>,<y>,<z>` (ticket 091).
#[derive(Debug, Args)]
pub struct GetArgs {
    /// Block coordinates, "x,y,z". See [`ChunkArgs::pos`] on why
    /// `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub pos: BlockPos,
}

/// `get-area <x1>,<y1>,<z1> <x2>,<y2>,<z2>` (ticket 092). Either corner may
/// be given in either order — [`super::block::get_area`] normalizes them via
/// [`crate::selection::SelectionBounds::from_corners`], the same as
/// [`ChunksArgs::from`]/[`ChunksArgs::to`] do for chunk rectangles.
#[derive(Debug, Args)]
pub struct GetAreaArgs {
    /// One corner, "x,y,z". See [`ChunkArgs::pos`] on why `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub from: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub to: BlockPos,
}

/// `column <x>,<z> [--from <y>] [--to <y>] [--top-down]` (ticket 093): every
/// block in one column, read via [`super::block::column`]. `--from`/`--to`
/// default to the world's build limits
/// ([`crate::selection::WORLD_MIN_Y`]/[`crate::selection::WORLD_MAX_Y`])
/// rather than a hardcoded `-64`/`320` — see the ticket on why that stopped
/// being universal.
#[derive(Debug, Args)]
pub struct ColumnArgs {
    /// The column, "x,z" — block coordinates, not a chunk position (see
    /// [`ColumnPos`]).
    #[arg(allow_hyphen_values = true)]
    pub pos: ColumnPos,

    /// Lowest Y to read. Defaults to the world's build-limit floor.
    #[arg(long, allow_hyphen_values = true)]
    pub from: Option<i32>,

    /// Highest Y to read. Defaults to the world's build-limit ceiling.
    #[arg(long, allow_hyphen_values = true)]
    pub to: Option<i32>,

    /// List from the highest Y down to the lowest instead of the default
    /// bottom-to-top order (the way a player reads a cave profile).
    #[arg(long)]
    pub top_down: bool,
}

/// `scan <x1,y1,z1> <x2,y2,z2> --block <name> [--limit N]` (ticket 093):
/// every position in the box whose block matches `--block` by exact
/// namespaced name (properties are not part of the match — see
/// [`super::block::scan`]).
#[derive(Debug, Args)]
pub struct ScanArgs {
    /// One corner, "x,y,z". See [`ChunkArgs::pos`] on why `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub from: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub to: BlockPos,

    /// Exact namespaced block name to match, e.g. `minecraft:oak_door` —
    /// matches every state of that block regardless of its properties.
    #[arg(long)]
    pub block: String,

    /// Caps how many matching positions are reported. Defaults to
    /// [`super::block::DEFAULT_SCAN_LIMIT`] — never unbounded, since a scan
    /// for a common block over a large box could otherwise print millions of
    /// positions.
    #[arg(long)]
    pub limit: Option<usize>,
}

impl HeightmapKindArg {
    /// The exact string this variant parses from on the command line —
    /// reused (rather than re-spelled) for `--format json`'s `"kind"` field
    /// and for error messages, so a caller never sees a name that doesn't
    /// round-trip back into `--kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            HeightmapKindArg::WorldSurface => "world-surface",
            HeightmapKindArg::MotionBlocking => "motion-blocking",
            HeightmapKindArg::MotionBlockingNoLeaves => "motion-blocking-no-leaves",
            HeightmapKindArg::OceanFloor => "ocean-floor",
        }
    }

    /// The `ranvil` type this CLI value maps to — the one place that mapping
    /// is spelled out.
    pub fn to_kind(self) -> HeightmapKind {
        match self {
            HeightmapKindArg::WorldSurface => HeightmapKind::WorldSurface,
            HeightmapKindArg::MotionBlocking => HeightmapKind::MotionBlocking,
            HeightmapKindArg::MotionBlockingNoLeaves => HeightmapKind::MotionBlockingNoLeaves,
            HeightmapKindArg::OceanFloor => HeightmapKind::OceanFloor,
        }
    }
}
