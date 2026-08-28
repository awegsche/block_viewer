//! The `clap` argument surface (ticket 087): the top-level [`Cli`], the
//! [`Command`] enum every subcommand hangs off, and each subcommand's own
//! args struct. `save`/`instance`/`format` are declared `global = true` so
//! they parse whether typed before or after the subcommand
//! (`ranvil-cli --format json saves` and `ranvil-cli saves --format json`
//! both work).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use mc_anvil::heightmap::HeightmapKind;

use crate::blueprint::BlockState;

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
/// box's blocks by name. Ticket 095 adds `Set`/`SetArea`, the first write
/// commands — thin builders over [`super::edit::run_write`] (094's write
/// substrate) around `WorldEdit::new().set`/`WorldEdit::fill` respectively.
/// Ticket 097 adds `Copy`, composing `get_area`'s read with `run_write`'s
/// write substrate — extract the source box, translate it, apply as one
/// transaction — rather than growing a third box-walking loop.
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
    /// Write one block.
    Set(SetArgs),
    /// Fill a box with one block.
    SetArea(SetAreaArgs),
    /// Apply many discrete `x,y,z blockstate` edits from a file or stdin as
    /// one transaction.
    SetBatch(SetBatchArgs),
    /// Find-and-replace by block name within a box, in one transaction.
    Replace(ReplaceArgs),
    /// Extract a box and re-apply it translated elsewhere in the same save.
    Copy(CopyArgs),
    /// Structure-file (`.nbt`) inspection and authoring — reads or writes a
    /// [`crate::blueprint::Blueprint`] on disk and, except `export`/`import`
    /// (099), never opens a save at all. Ticket 098 adds `info`/`new`;
    /// ticket 099 adds `export`/`import`, the bridge between a live save and
    /// a structure file; later tickets add `get`/`set`/`fill`/`resize`/
    /// `rotate`/`diff`/`validate` to [`StructCommand`].
    #[command(subcommand)]
    Struct(StructCommand),
}

/// `struct <cmd>` (tickets 098–103): one variant per structure-file
/// subcommand, routed the same way [`Command`] routes its own top-level
/// variants — see [`super::structure`]. Ticket 100 adds `get`/`set`/`fill`,
/// the first commands that edit a [`crate::blueprint::Blueprint`] in memory
/// outside of extraction and rotation.
#[derive(Debug, Subcommand)]
pub enum StructCommand {
    /// A structure file's own shape: size, origin, block count,
    /// `DataVersion`, palette.
    Info(StructInfoArgs),
    /// A blank structure file from scratch — air, or one state throughout.
    New(StructNewArgs),
    /// Extract a box out of a live save into a structure file.
    Export(StructExportArgs),
    /// Place a structure file's blocks into a live save.
    Import(StructImportArgs),
    /// Read one block inside a structure file, at a position relative to the
    /// structure's own `0..size` space.
    Get(StructGetArgs),
    /// Edit one block inside a structure file, in place or to a new file.
    Set(StructSetArgs),
    /// Fill a sub-box inside a structure file with one block.
    Fill(StructFillArgs),
    /// Add or trim margin on any of the six faces of a structure file.
    Resize(StructResizeArgs),
    /// Rotate a structure file about Y by 90/180/270 degrees.
    Rotate(StructRotateArgs),
    /// Per-position block differences between two same-size structure files.
    Diff(StructDiffArgs),
}

/// `struct info <file.nbt>` (ticket 098).
#[derive(Debug, Args)]
pub struct StructInfoArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,
}

/// `struct new --size <x,y,z> --out <file.nbt> [--fill <blockstate>]
/// [--force]` (ticket 098).
#[derive(Debug, Args)]
pub struct StructNewArgs {
    /// The new structure's size, "x,y,z". Reuses [`BlockPos`]'s "x,y,z"
    /// parser rather than growing a second one — `struct new` validates each
    /// component (at least 1, at most [`crate::blueprint::STRUCTURE_BLOCK_MAX_SIZE`])
    /// itself, the same way [`super::structure::new`] documents.
    #[arg(long, allow_hyphen_values = true)]
    pub size: BlockPos,

    /// Where to write the new structure file.
    #[arg(long)]
    pub out: PathBuf,

    /// The block every position is filled with.
    #[arg(long, default_value = "minecraft:air")]
    pub fill: BlockState,

    /// Overwrite `--out` if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// `struct export <x1>,<y1>,<z1> <x2>,<y2>,<z2> --out <file.nbt> [--force]`
/// (ticket 099) — [`super::block::get_area`]'s box read (092's primitive),
/// written to `out` via [`crate::blueprint::write_structure_file`] instead
/// of `get-area`'s stdout formatting. Reports the same summary `struct info`
/// would report on the freshly-written file, so a caller gets "extract this
/// building and tell me about it" in one command rather than `struct export`
/// followed by a separate `struct info` call.
#[derive(Debug, Args)]
pub struct StructExportArgs {
    /// One corner of the box, "x,y,z". See [`ChunkArgs::pos`] on why
    /// `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub from: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub to: BlockPos,

    /// Where to write the extracted structure file.
    #[arg(long)]
    pub out: PathBuf,

    /// Overwrite `--out` if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// `struct import <file.nbt> --at <x,y,z> [--rotate 90|180|270] [--dry-run]
/// [--force]` (ticket 099) — [`crate::blueprint::read_structure_file`],
/// optionally [`crate::blueprint::rotate_blueprint`], then a [`crate::edit::WorldEdit`]
/// writing every position (including air, matching `city::commit`'s own
/// `blueprint_edit` convention — a placement clears whatever terrain poked
/// into the footprint) offset by `--at`, run through
/// [`super::edit::run_write`].
///
/// `--at` names the structure's own origin (always `IVec3::ZERO`) mapped to
/// that world position — position `p` inside the structure lands at
/// `--at + p`, the same convention [`CopyArgs::dest`] uses for its source's
/// minimum corner.
#[derive(Debug, Args)]
pub struct StructImportArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,

    /// The world position the structure's own `(0, 0, 0)` lands at, "x,y,z".
    #[arg(long, allow_hyphen_values = true)]
    pub at: BlockPos,

    /// Rotate the structure about Y before placing it. `struct rotate`
    /// (ticket 102) is the one place rotation logic itself lives — this
    /// flag just calls it before building the write.
    #[arg(long, value_enum)]
    pub rotate: Option<RotateArg>,

    /// Report the plan without writing anything — see [`super::edit::run_write`].
    #[arg(long)]
    pub dry_run: bool,

    /// Write even though the save currently looks open in Minecraft.
    #[arg(long)]
    pub force: bool,
}

/// The `--rotate` values `struct import` accepts — a CLI-facing mirror of
/// [`crate::blueprint::Rotation`]'s three non-identity variants. `Deg0` isn't
/// offered here: omitting `--rotate` already means "no rotation", so a
/// fourth value that means the same thing would just be a second spelling of
/// "not given".
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RotateArg {
    #[value(name = "90")]
    Deg90,
    #[value(name = "180")]
    Deg180,
    #[value(name = "270")]
    Deg270,
}

impl RotateArg {
    /// The exact string this variant parses from on the command line —
    /// reused for `--format json`'s `"rotate"` field, same convention
    /// [`HeightmapKindArg::as_str`] follows.
    pub fn as_str(self) -> &'static str {
        match self {
            RotateArg::Deg90 => "90",
            RotateArg::Deg180 => "180",
            RotateArg::Deg270 => "270",
        }
    }

    /// The [`crate::blueprint::Rotation`] this CLI value maps to — the one
    /// place that mapping is spelled out.
    pub fn to_rotation(self) -> crate::blueprint::Rotation {
        match self {
            RotateArg::Deg90 => crate::blueprint::Rotation::Deg90,
            RotateArg::Deg180 => crate::blueprint::Rotation::Deg180,
            RotateArg::Deg270 => crate::blueprint::Rotation::Deg270,
        }
    }
}

/// `struct get <file.nbt> <x,y,z>` (ticket 100) — [`crate::blueprint::Blueprint::block_at`]'s
/// job, but with an in-bounds check that refuses instead of quietly reading
/// `None`. `pos` is relative to the structure's own `0..size` space, **not**
/// a world coordinate — the one exception among every other `get`-shaped
/// command in this CLI, per the ticket.
#[derive(Debug, Args)]
pub struct StructGetArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,

    /// Position inside the structure, "x,y,z" — `0..size` on each axis, not
    /// a world coordinate. See [`ChunkArgs::pos`] on why `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub pos: BlockPos,
}

/// `struct set <file.nbt> <x,y,z> <blockstate> [--out <file2.nbt>]
/// [--force]` (ticket 100): sets one position's palette index, inserting a
/// new palette entry if `state` isn't already in it. `pos` is relative to the
/// structure's own `0..size` space, same as [`StructGetArgs::pos`].
///
/// Without `--out`, overwrites `file` in place — refused without `--force`,
/// the same "editing a file in place is destructive too" gate `struct new`/
/// `struct export` apply to their own `--out`. With `--out`, the original is
/// untouched; `--out` pointing at an existing path follows the identical
/// `--force` gate rather than a special case, so the same rule covers both
/// "overwrite the source" and "overwrite something else already there".
#[derive(Debug, Args)]
pub struct StructSetArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,

    /// Position inside the structure, "x,y,z". See [`StructGetArgs::pos`].
    #[arg(allow_hyphen_values = true)]
    pub pos: BlockPos,

    /// The block to write, e.g. `minecraft:oak_stairs[facing=east,half=top]`.
    pub state: BlockState,

    /// Write the edited structure to a new file instead of `file` itself.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Overwrite the output path if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// `struct fill <file.nbt> <x1,y1,z1> <x2,y2,z2> <blockstate> [--out
/// <file2.nbt>] [--force]` (ticket 100): the same in-bounds box-fill as
/// `struct set`, batched — one palette-growth pass rather than one per block,
/// same reason [`ranvil::chunkregion::ChunkRegion::set_blocks`] exists over
/// calling `set_block` in a loop. Either corner may be given in either order,
/// same as [`GetAreaArgs`]. See [`StructSetArgs`] on `--out`/`--force`.
#[derive(Debug, Args)]
pub struct StructFillArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,

    /// One corner of the box, "x,y,z" — inside the structure's own `0..size`
    /// space, same as [`StructGetArgs::pos`].
    #[arg(allow_hyphen_values = true)]
    pub from: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub to: BlockPos,

    /// The block to fill the box with.
    pub state: BlockState,

    /// Write the edited structure to a new file instead of `file` itself.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Overwrite the output path if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// `struct resize <file.nbt> --out <file2.nbt> [--pad-y-top N]
/// [--pad-y-bottom N] [--pad-x-neg N] [--pad-x-pos N] [--pad-z-neg N]
/// [--pad-z-pos N] [--fill <blockstate>] [--force]` (ticket 101): adds or
/// trims margin on each of the six faces of the bounding box independently.
/// A positive pad on a face adds margin there, filled with `--fill`; a
/// negative pad crops that face instead — one command for both directions,
/// since a caller reaching for "make this 2 blocks shorter" shouldn't have
/// to know it's a different command from "make this 2 blocks taller". See
/// [`super::structure::resize`] for the mechanics and
/// [`super::structure::StructResizeArgs`]'s field docs for which face each
/// flag controls.
///
/// Unlike `struct set`/`struct fill`, `--out` is required rather than
/// defaulting to `file` itself — a resize changes the structure's own size
/// and every block's coordinate within it, not just a handful of positions,
/// so there is no "in place" convenience worth a default here.
///
/// `--pad-y-bottom`/`--pad-x-neg`/`--pad-z-neg` shift every remaining
/// block's coordinate by the pad amount (a new bottom layer at `y=0` pushes
/// the old `y=0` to `y=N`). A building's own `.ron` `ground_level`
/// (`city::definition`'s field) is **not** touched by this command — a
/// structure file carries no such field itself — so `--pad-y-bottom` on a
/// building already in `assets/city/buildings` likely needs its companion
/// `.ron`'s `ground_level` bumped by the same amount by hand, same as any
/// other structural edit to a building's blueprint would.
#[derive(Debug, Args)]
pub struct StructResizeArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,

    /// Where to write the resized structure file.
    #[arg(long)]
    pub out: PathBuf,

    /// Margin added above the structure's top layer (`+Y`). Negative crops
    /// that many layers off the top instead.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
    pub pad_y_top: i32,

    /// Margin added below the structure's bottom layer (`-Y`). Every
    /// existing block shifts up by this amount to make room. Negative crops
    /// that many layers off the bottom instead (shifting every remaining
    /// block down).
    #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
    pub pad_y_bottom: i32,

    /// Margin added on the `-X` face. Every existing block shifts by this
    /// amount on X to make room. Negative crops that face instead.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
    pub pad_x_neg: i32,

    /// Margin added on the `+X` face. Negative crops that face instead.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
    pub pad_x_pos: i32,

    /// Margin added on the `-Z` face. Every existing block shifts by this
    /// amount on Z to make room. Negative crops that face instead.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
    pub pad_z_neg: i32,

    /// Margin added on the `+Z` face. Negative crops that face instead.
    #[arg(long, allow_hyphen_values = true, default_value_t = 0)]
    pub pad_z_pos: i32,

    /// The block newly added margin is filled with. Irrelevant on an axis
    /// that only crops.
    #[arg(long, default_value = "minecraft:air")]
    pub fill: BlockState,

    /// Overwrite `--out` if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// `struct rotate <file.nbt> --by 90|180|270 --out <file2.nbt> [--force]`
/// (ticket 102): a direct wrapper over [`crate::blueprint::rotate_blueprint`]
/// (ticket 038's function, unchanged) — the CLI's job is argument parsing and
/// error mapping, not rotation logic (see [`super::structure::rotate`]).
/// Reuses [`RotateArg`] (already spelling `90`/`180`/`270` for `struct
/// import --rotate`) rather than a second enum for the same three values.
///
/// See [`StructResizeArgs`] on why `--out` is required here too — rotation
/// changes every block's coordinate (and, for 90°/270°, the structure's own
/// X/Z extents), not a handful of positions, so there is no "in place"
/// convenience worth a default.
#[derive(Debug, Args)]
pub struct StructRotateArgs {
    /// Path to a gzipped vanilla structure file.
    pub file: PathBuf,

    /// How far to rotate about Y, clockwise viewed from above.
    #[arg(long, value_enum)]
    pub by: RotateArg,

    /// Where to write the rotated structure file.
    #[arg(long)]
    pub out: PathBuf,

    /// Overwrite `--out` if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// `struct diff <a.nbt> <b.nbt> [--limit N]` (ticket 102): refuses
/// (`Usage`, exit 2) if `a`/`b` have different `size`s — a per-position diff
/// needs a shared coordinate space, so comparing two differently-sized
/// buildings is `struct info a.nbt` and `struct info b.nbt` side by side, not
/// this command's job. Otherwise walks every position and reports where the
/// two disagree, properties included (unlike `scan`/`replace`'s name-only
/// matching) — see [`super::structure::diff`].
#[derive(Debug, Args)]
pub struct StructDiffArgs {
    /// Path to the first gzipped vanilla structure file.
    pub a: PathBuf,
    /// Path to the second gzipped vanilla structure file — must be the same
    /// `size` as `a`.
    pub b: PathBuf,

    /// Caps how many differing positions `text`/`compact` list. `json`
    /// always reports the full list regardless of this flag. Defaults to
    /// [`super::block::DEFAULT_SCAN_LIMIT`], the same "a few hundred, never
    /// unbounded" default `scan --limit` uses.
    #[arg(long)]
    pub limit: Option<usize>,
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

/// `set <x>,<y>,<z> <blockstate> [--dry-run] [--force]` (ticket 095).
/// `state` is parsed by [`BlockState`]'s own `FromStr` (ticket 035's
/// paint-tool parser) — the same `name[key=value,...]` syntax `get` already
/// prints back, so a `get`'s output pastes straight into this argument.
#[derive(Debug, Args)]
pub struct SetArgs {
    /// Block coordinates, "x,y,z". See [`ChunkArgs::pos`] on why
    /// `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub pos: BlockPos,

    /// The block to write, e.g. `minecraft:oak_stairs[facing=east,half=top]`.
    pub state: BlockState,

    /// Report the plan (blocks, regions touched) without writing anything —
    /// see [`super::edit::run_write`].
    #[arg(long)]
    pub dry_run: bool,

    /// Write even though the save currently looks open in Minecraft — see
    /// [`super::edit::run_write`].
    #[arg(long)]
    pub force: bool,
}

/// `set-area <x1>,<y1>,<z1> <x2>,<y2>,<z2> <blockstate> [--dry-run]
/// [--force]` (ticket 095) — `WorldEdit::fill` over the box, the same
/// function `viewer::paint` calls for the same job (ticket 035). Either
/// corner may be given in either order, same as [`GetAreaArgs`].
#[derive(Debug, Args)]
pub struct SetAreaArgs {
    /// One corner, "x,y,z". See [`ChunkArgs::pos`] on why `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub from: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub to: BlockPos,

    /// The block to fill the box with.
    pub state: BlockState,

    /// Report the plan without writing anything — see [`super::edit::run_write`].
    #[arg(long)]
    pub dry_run: bool,

    /// Write even though the save currently looks open in Minecraft.
    #[arg(long)]
    pub force: bool,
}

/// `set-batch <file|-> [--dry-run] [--force]` (ticket 096): reads
/// `x,y,z blockstate` lines (blank lines and `#`-prefixed lines ignored)
/// from `file`, or from stdin when `file` is `-`, and applies every one as a
/// single [`super::edit::run_write`] transaction — see
/// [`super::edit::set_batch`].
#[derive(Debug, Args)]
pub struct SetBatchArgs {
    /// A path to a file of "x,y,z blockstate" lines, or `-` to read from
    /// stdin.
    pub file: String,

    /// Report the plan without writing anything — see [`super::edit::run_write`].
    #[arg(long)]
    pub dry_run: bool,

    /// Write even though the save currently looks open in Minecraft.
    #[arg(long)]
    pub force: bool,
}

/// `replace <x1,y1,z1> <x2,y2,z2> --from <name> --to <blockstate>
/// [--dry-run] [--force]` (ticket 096): every position in the box whose
/// block's name (properties ignored, same convention [`ScanArgs::block`]
/// uses) matches `--from` gets written to `--to`, as one transaction — see
/// [`super::edit::replace`]. The box corners are positional (either order,
/// same as [`GetAreaArgs`]); `--from`/`--to` name the match/replacement
/// block so they can't be confused with the box's own corners.
#[derive(Debug, Args)]
pub struct ReplaceArgs {
    /// One corner of the box, "x,y,z". See [`ChunkArgs::pos`] on why
    /// `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub corner1: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub corner2: BlockPos,

    /// Exact namespaced block name to match, e.g. `minecraft:oak_log` —
    /// matches every state of that block regardless of its properties, same
    /// as `scan --block`.
    #[arg(long = "from")]
    pub from_block: String,

    /// The block to write at every matched position.
    #[arg(long = "to")]
    pub to_state: BlockState,

    /// Report the plan without writing anything — see [`super::edit::run_write`].
    #[arg(long)]
    pub dry_run: bool,

    /// Write even though the save currently looks open in Minecraft.
    #[arg(long)]
    pub force: bool,
}

/// `copy <x1>,<y1>,<z1> <x2>,<y2>,<z2> --to <x>,<y>,<z> [--include-air]
/// [--dry-run] [--force]` (ticket 097): extracts the source box via
/// [`super::block::get_area`] (092's primitive — the same read `scan`/
/// `replace` reuse rather than growing a fresh box walk) and re-applies it
/// translated so its min corner lands at `--to`, as one
/// [`super::edit::run_write`] transaction — see [`super::edit::copy`].
///
/// Source and destination may overlap (a same-region nudge is a legitimate
/// use): the whole source box is read before `run_write` opens a session or
/// touches the destination, so an overlap always reads the *original*
/// blocks, never a partially-copied one.
///
/// There is no `--rotate` here — `struct rotate` (ticket 102) is the one
/// place rotation logic lives. A rotated in-place copy is `struct export`
/// the source, `struct rotate` it, then `struct import` at the destination.
#[derive(Debug, Args)]
pub struct CopyArgs {
    /// One corner of the source box, "x,y,z". See [`ChunkArgs::pos`] on why
    /// `allow_hyphen_values`.
    #[arg(allow_hyphen_values = true)]
    pub corner1: BlockPos,
    /// The opposite corner, "x,y,z".
    #[arg(allow_hyphen_values = true)]
    pub corner2: BlockPos,

    /// The destination's minimum corner, "x,y,z" — matching how the source
    /// box's own minimum corner is read (`SelectionBounds::from_corners`'s
    /// existing convention).
    #[arg(long = "to", allow_hyphen_values = true)]
    pub dest: BlockPos,

    /// Also write the source's air at the destination, overwriting whatever
    /// stands there. Without this flag, air positions inside the source box
    /// are skipped — copying a tree-shaped selection shouldn't punch an
    /// air-shaped hole through whatever already stands at the destination.
    #[arg(long)]
    pub include_air: bool,

    /// Report the plan without writing anything — see [`super::edit::run_write`].
    #[arg(long)]
    pub dry_run: bool,

    /// Write even though the save currently looks open in Minecraft.
    #[arg(long)]
    pub force: bool,
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
