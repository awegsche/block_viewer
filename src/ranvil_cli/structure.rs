//! `ranvil-cli struct info`/`struct new` (ticket 098) — the first `struct`
//! subcommands: everything they read or write is a [`Blueprint`] on disk, so
//! unlike every command in [`super::block`]/[`super::edit`] neither one
//! opens a save or takes `--save`/`--instance` at all (per
//! `RANVIL_CLI_ROADMAP.md`'s ordering advice — group E needs only ticket
//! 087's scaffolding).
//!
//! `info` is a thin wrapper over [`read_structure_file`]: everything it
//! reports is a [`Blueprint`] field, computed once by the reader rather than
//! re-derived here. `new` is the inverse — a solid-fill [`Blueprint`] built
//! from nothing and handed to [`write_structure_file`] — and is deliberately
//! stricter than the writer it calls: [`STRUCTURE_BLOCK_MAX_SIZE`] is only a
//! warning inside [`write_structure_file`] (see that module's docs), but a
//! file `struct new` builds from scratch has no data worth keeping past that
//! cap, so it refuses outright instead.
//!
//! `export`/`import` (ticket 099) are the bridge to a live save — the only
//! two commands in this module that take `--save`/`--instance` at all.
//! `export` is [`super::block::get_area`]'s box read
//! ([`crate::blueprint::extract_blueprint`], 092's primitive) written out
//! through [`write_structure_file`] instead of `get-area`'s stdout
//! formatting; `import` is [`read_structure_file`] followed by a
//! [`crate::edit::WorldEdit`] run through [`super::edit::run_write`] (094's
//! write substrate) — the same read-then-write shape [`super::edit::copy`]
//! (097) uses, just with a file instead of a second box as the source.
//!
//! `get`/`set`/`fill` (ticket 100) are the first commands that edit a
//! [`Blueprint`] in memory outside of extraction and rotation, scoped
//! tightly to their world-side `get`/`set`/`set-area` equivalents: no
//! `--save`/`--instance` (a structure file is the whole story), positions
//! relative to the structure's own `0..size` space rather than a world
//! coordinate, and out-of-bounds is [`CliError::Usage`] rather than
//! [`CliError::Data`] — a structure file's extent is fully known up front
//! from its own `size`, unlike a live save's "maybe it's just ungenerated"
//! ambiguity. `set`/`fill` share one palette-growth rule
//! ([`palette_index_for`]), mirroring — rather than reusing — the "does the
//! palette need to grow" logic [`ranvil::chunkregion::ChunkRegion::set_blocks`]
//! performs for a live section: `Blueprint`'s dense `Vec<u16>` and a
//! section's packed-bits array are different representations of the same
//! idea.
//!
//! `resize` (ticket 101) is the first command that changes a [`Blueprint`]'s
//! own `size` rather than editing positions within a fixed one: six
//! independent per-face pads (positive adds margin filled with `--fill`,
//! negative crops that face) computed into a new size up front, then one
//! pass copying every source position that still lands inside it into a
//! freshly allocated block array — a crop simply never copies the part that
//! falls outside. `--pad-y-bottom`/`--pad-x-neg`/`--pad-z-neg` are the three
//! pads that shift every remaining block's coordinate (a new bottom layer at
//! `y=0` pushes the old `y=0` to `y=N`); the other three only change where
//! the far face sits. Unlike `set`/`fill`, `--out` is required rather than
//! defaulting to `file` itself (see [`super::cli::StructResizeArgs`]'s docs).
//!
//! `rotate`/`diff` (ticket 102) are unrelated to each other except sharing a
//! module: `rotate` is a thin wrapper over [`rotate_blueprint`] (038's
//! function — `struct import --rotate` already called it, so this is the
//! CLI's first *direct* exposure of it, not new logic), while `diff` is the
//! first `struct` command that reads two structure files and never writes
//! one — a per-position [`BlockState`] comparison (properties included, not
//! just names) that refuses up front when the two `size`s don't match, since
//! there's no shared coordinate space to walk otherwise. `diff`'s `text`/
//! `compact` cap their listing at `--limit` the same way `scan` does; `json`
//! deliberately does not, per the ticket — see [`StructDiffResult`]'s docs.
//!
//! `validate` (ticket 103, last in the plan) runs
//! [`crate::blueprint::run_checks`] — the exact size/palette checks
//! `blueprint::catalogue`'s loader already applies silently to every file it
//! scans, pulled out so this command and that loader share one
//! implementation rather than two copies of the same numeric limits (see
//! that function's own docs). Unlike every other `struct` command, a file
//! that parses but fails a check is not an error: [`StructValidateResult`]
//! is `Ok` either way, and [`StructValidateResult::all_passed`] is what
//! [`super::run`]'s dispatch reads to choose exit `1` over `0`, since the
//! per-check listing is the answer the ticket asked for, not something to
//! discard in favor of an error envelope.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use serde_json::{json, Value};

use crate::blueprint::{
    extract_blueprint, read_structure_file, rotate_blueprint, run_checks, write_structure_file,
    BlockState, Blueprint, BlueprintCheck, ExtractProgress, FALLBACK_DATA_VERSION,
    LOGGED_PALETTE_ENTRIES, MAX_BLOCKS, STRUCTURE_BLOCK_MAX_SIZE,
};
use crate::edit::WorldEdit;
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;
use crate::world::SECTION_SIZE;

use super::block::position_of;
use super::chunk::region_span;
use super::cli::{
    Cli, RotateArg, StructDiffArgs, StructExportArgs, StructFillArgs, StructGetArgs,
    StructImportArgs, StructInfoArgs, StructNewArgs, StructResizeArgs, StructRotateArgs,
    StructSetArgs, StructValidateArgs,
};
use super::edit::{outcome_json_fields, outcome_summary, run_write, WriteOutcome};
use super::error::CliError;
use super::format::Render;
use super::save::resolve_save;

// -------------------------------------------------------------------------------------------------
// ---- info -----------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct info`'s result: every [`Blueprint`] field a structure file
/// carries.
#[derive(Debug)]
pub struct StructInfoResult {
    pub path: PathBuf,
    pub size: IVec3,
    /// Always [`IVec3::ZERO`] — a structure file never carries an origin
    /// (see [`crate::blueprint::read_structure`]'s "The reader's
    /// decisions"). Reported anyway, so this field is present here the same
    /// way it is in `get-area`/`struct export`'s output, rather than
    /// silently missing for this command alone.
    pub origin: IVec3,
    pub blocks: usize,
    pub data_version: i32,
    pub palette: Vec<BlockState>,
}

/// Runs `struct info`: [`read_structure_file`] on `args.file`, then reports
/// the [`Blueprint`] it hands back. A malformed or oversized file surfaces as
/// [`CliError::Data`] carrying [`StructureReadError`](crate::blueprint::StructureReadError)'s
/// own message, not a panic — [`read_structure_file`] already turns every
/// failure mode (missing file, non-gzip, truncated NBT, a `blocks` list that
/// doesn't cover `size`) into an `Err` rather than one.
pub fn info(args: &StructInfoArgs) -> Result<StructInfoResult, CliError> {
    let blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    Ok(StructInfoResult {
        path: args.file.clone(),
        size: blueprint.size,
        origin: blueprint.origin,
        blocks: blueprint.volume(),
        data_version: blueprint.data_version,
        palette: blueprint.palette,
    })
}

impl StructInfoResult {
    fn summary_line(&self) -> String {
        format!(
            "struct info {}: size {}x{}x{} ({} blocks), origin {}, DataVersion {}, {} distinct states",
            self.path.display(),
            self.size.x,
            self.size.y,
            self.size.z,
            self.blocks,
            self.origin,
            self.data_version,
            self.palette.len(),
        )
    }
}

impl Render for StructInfoResult {
    /// The summary line plus the palette listing, capped the same way
    /// [`blueprint's finished-extraction log`](crate::blueprint) caps its
    /// own console output ([`LOGGED_PALETTE_ENTRIES`]) — unlike `get-area`
    /// (whose palette can come from a save-spanning box), a structure file's
    /// palette is small enough to list by default rather than requiring
    /// `--format json`.
    fn render_text(&self) -> String {
        let mut lines = vec![self.summary_line()];
        for (index, state) in self.palette.iter().take(LOGGED_PALETTE_ENTRIES).enumerate() {
            lines.push(format!("  [{index}] {state}"));
        }
        let rest = self.palette.len().saturating_sub(LOGGED_PALETTE_ENTRIES);
        if rest > 0 {
            lines.push(format!("  ... and {rest} more"));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "path": self.path.display().to_string(),
            "size": [self.size.x, self.size.y, self.size.z],
            "origin": [self.origin.x, self.origin.y, self.origin.z],
            "blocks": self.blocks,
            "data_version": self.data_version,
            "palette_size": self.palette.len(),
            "palette": self.palette.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })
    }

    /// The summary line alone, no palette listing — same "how big / how many
    /// distinct blocks" role `get-area`'s compact plays for its own palette.
    fn render_compact(&self) -> String {
        self.summary_line()
    }
}

// -------------------------------------------------------------------------------------------------
// ---- new ------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct new`'s result: where the file went and what it was filled with.
#[derive(Debug)]
pub struct StructNewResult {
    pub path: PathBuf,
    pub size: IVec3,
    pub fill: BlockState,
}

/// Runs `struct new`: builds a `size`-shaped [`Blueprint`], every position
/// `fill`, and writes it to `out` via [`write_structure_file`], replacing
/// whatever was there.
///
/// `size`'s components are checked against [`STRUCTURE_BLOCK_MAX_SIZE`] —
/// each at least 1, none over it — before anything is allocated, so a size a
/// later `write_structure_file` would only warn about never gets that far
/// here (see the module docs on why `struct new` is stricter than the
/// writer it calls). `out` refuses to be overwritten without `--force`, the
/// same convention every other file-writing `struct` command (100–102)
/// follows.
pub fn new(args: &StructNewArgs) -> Result<StructNewResult, CliError> {
    let size = args.size.0;
    for (component, axis) in [(size.x, 'x'), (size.y, 'y'), (size.z, 'z')] {
        if component < 1 || component > STRUCTURE_BLOCK_MAX_SIZE {
            return Err(CliError::Usage(format!(
                "--size {axis} component {component} must be between 1 and {STRUCTURE_BLOCK_MAX_SIZE}"
            )));
        }
    }

    if args.out.exists() && !args.force {
        return Err(CliError::Usage(format!(
            "{} already exists — pass --force to overwrite",
            args.out.display()
        )));
    }

    let volume = size.x as usize * size.y as usize * size.z as usize;
    let blueprint = Blueprint {
        size,
        origin: IVec3::ZERO,
        palette: vec![args.fill.clone()],
        // Every position indexes the one-entry palette above — `struct new`
        // has no source to sample a second state from.
        blocks: vec![0u16; volume],
        // No world to sample a real `DataVersion` from (unlike
        // `extract_blueprint`'s per-chunk sampling) — the same "nothing to
        // copy it from" fallback extraction itself uses.
        data_version: FALLBACK_DATA_VERSION,
        failed_columns: 0,
    };

    write_structure_file(&args.out, &blueprint).map_err(|err| {
        CliError::Data(format!("could not write {}: {err}", args.out.display()))
    })?;

    Ok(StructNewResult { path: args.out.clone(), size, fill: args.fill.clone() })
}

impl Render for StructNewResult {
    fn render_text(&self) -> String {
        format!(
            "struct new {}: size {}x{}x{}, filled with {}",
            self.path.display(),
            self.size.x,
            self.size.y,
            self.size.z,
            self.fill,
        )
    }

    fn render_json(&self) -> Value {
        json!({
            "path": self.path.display().to_string(),
            "size": [self.size.x, self.size.y, self.size.z],
            "fill": self.fill.to_string(),
        })
    }
}

// -------------------------------------------------------------------------------------------------
// ---- export -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct export`'s result: the source box, where the file went, plus
/// everything [`StructInfoResult`] would report on it — size, block count,
/// `DataVersion`, palette — and the box read's own [`Blueprint::failed_columns`],
/// which a structure file never carries but a live-save read can produce.
#[derive(Debug)]
pub struct StructExportResult {
    pub save_name: String,
    pub from: IVec3,
    pub to: IVec3,
    pub out: PathBuf,
    pub size: IVec3,
    pub blocks: usize,
    pub data_version: i32,
    pub palette: Vec<BlockState>,
    pub failed_columns: usize,
}

/// Runs `struct export`: [`extract_blueprint`] over `args.from`/`args.to`
/// (the same [`SelectionBounds`]/region-cache-sizing shape
/// [`super::block::get_area`] builds — see that function's docs), then
/// [`write_structure_file`] instead of `get-area`'s stdout formatting.
///
/// The [`MAX_BLOCKS`] check and the `out`-exists check both happen before
/// [`resolve_save`] runs, same reasoning [`super::block::get_area`]'s own
/// volume check uses: a bad request is caught before anything is touched,
/// regardless of which save it names.
pub fn export(cli: &Cli, args: &StructExportArgs) -> Result<StructExportResult, CliError> {
    let bounds = SelectionBounds::from_corners(args.from.0, args.from.0, args.to.0);

    let volume = bounds.volume();
    if volume > MAX_BLOCKS {
        return Err(CliError::Usage(format!(
            "selection ({}) to ({}) is {volume} blocks — over the {MAX_BLOCKS}-block struct export limit",
            args.from.0, args.to.0
        )));
    }

    if args.out.exists() && !args.force {
        return Err(CliError::Usage(format!(
            "{} already exists — pass --force to overwrite",
            args.out.display()
        )));
    }

    let meta = resolve_save(cli)?;

    // Sized the same way `get_area` sizes its own cache: exactly the regions
    // this box's chunk columns span.
    let size = SECTION_SIZE as i32;
    let (min_cx, min_cz) = (bounds.min.x.div_euclid(size), bounds.min.z.div_euclid(size));
    let (max_cx, max_cz) = (bounds.max.x.div_euclid(size), bounds.max.z.div_euclid(size));
    let capacity = region_span(min_cx, max_cx, min_cz, max_cz);

    let cache = Arc::new(Mutex::new(RegionCache::new(meta.clone(), capacity)));
    let progress = ExtractProgress::default();
    let blueprint = extract_blueprint(bounds, &cache, &progress).map_err(|e| {
        CliError::Data(format!(
            "could not extract ({}) to ({}): {e}",
            args.from.0, args.to.0
        ))
    })?;

    write_structure_file(&args.out, &blueprint).map_err(|err| {
        CliError::Data(format!("could not write {}: {err}", args.out.display()))
    })?;

    Ok(StructExportResult {
        save_name: meta.name,
        from: bounds.min,
        to: bounds.max,
        out: args.out.clone(),
        size: blueprint.size,
        blocks: blueprint.volume(),
        data_version: blueprint.data_version,
        palette: blueprint.palette,
        failed_columns: blueprint.failed_columns,
    })
}

impl StructExportResult {
    fn summary_line(&self) -> String {
        format!(
            "struct export {} to {} in {} -> {}: size {}x{}x{} ({} blocks), DataVersion {}, \
             {} distinct states, {} failed columns",
            self.from,
            self.to,
            self.save_name,
            self.out.display(),
            self.size.x,
            self.size.y,
            self.size.z,
            self.blocks,
            self.data_version,
            self.palette.len(),
            self.failed_columns,
        )
    }
}

impl Render for StructExportResult {
    /// The summary line plus the palette listing, capped the same way
    /// [`StructInfoResult::render_text`] caps its own.
    fn render_text(&self) -> String {
        let mut lines = vec![self.summary_line()];
        for (index, state) in self.palette.iter().take(LOGGED_PALETTE_ENTRIES).enumerate() {
            lines.push(format!("  [{index}] {state}"));
        }
        let rest = self.palette.len().saturating_sub(LOGGED_PALETTE_ENTRIES);
        if rest > 0 {
            lines.push(format!("  ... and {rest} more"));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "from": [self.from.x, self.from.y, self.from.z],
            "to": [self.to.x, self.to.y, self.to.z],
            "out": self.out.display().to_string(),
            "size": [self.size.x, self.size.y, self.size.z],
            "blocks": self.blocks,
            "data_version": self.data_version,
            "palette_size": self.palette.len(),
            "palette": self.palette.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "failed_columns": self.failed_columns,
        })
    }

    fn render_compact(&self) -> String {
        self.summary_line()
    }
}

// -------------------------------------------------------------------------------------------------
// ---- import -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct import`'s result: the file, where it landed, whether it was
/// rotated first, plus what [`run_write`] did (or would do).
#[derive(Debug)]
pub struct StructImportResult {
    pub save_name: String,
    pub file: PathBuf,
    pub at: IVec3,
    pub rotate: Option<RotateArg>,
    pub size: IVec3,
    pub outcome: WriteOutcome,
}

/// Runs `struct import`: [`read_structure_file`], then (if `args.rotate` is
/// given) [`rotate_blueprint`] — 102's function, called rather than
/// reimplemented, since it already exists (ticket 038) even though `struct
/// rotate` the CLI command doesn't yet — then a [`WorldEdit`] writing every
/// position (air included, matching `city::commit`'s own `blueprint_edit`
/// convention — see that module's docs on why a placement writes air rather
/// than skipping it) offset by `args.at`, run through [`run_write`].
///
/// `size` is the *rotated* blueprint's size (when `--rotate` was given) —
/// what actually gets written, not the file's own on-disk size.
pub fn import(cli: &Cli, args: &StructImportArgs) -> Result<StructImportResult, CliError> {
    let blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    let blueprint = match args.rotate {
        None => blueprint,
        Some(rotate) => rotate_blueprint(&blueprint, rotate.to_rotation())
            .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?,
    };

    let at = args.at.0;
    let size = blueprint.size;
    let data_version = blueprint.data_version;
    let palette = blueprint.palette;
    let blocks = blueprint.blocks;

    let meta = resolve_save(cli)?;

    let outcome = run_write(&meta, args.dry_run, args.force, move |_cache| {
        let mut edit = WorldEdit::new().with_data_version(data_version);
        for (index, &palette_index) in blocks.iter().enumerate() {
            let state = palette[palette_index as usize].clone();
            let local = position_of(IVec3::ZERO, size, index);
            edit.set(at + local, state);
        }
        Ok(edit)
    })?;

    Ok(StructImportResult {
        save_name: meta.name,
        file: args.file.clone(),
        at,
        rotate: args.rotate,
        size,
        outcome,
    })
}

impl Render for StructImportResult {
    fn render_text(&self) -> String {
        let rotate_suffix = self
            .rotate
            .map(|r| format!(" rotated {}", r.as_str()))
            .unwrap_or_default();
        let prefix = format!(
            "struct import {} ({}x{}x{}) at {} in {}{rotate_suffix}",
            self.file.display(),
            self.size.x,
            self.size.y,
            self.size.z,
            self.at,
            self.save_name,
        );
        let mut lines = vec![outcome_summary(prefix, &self.outcome)];
        if !self.outcome.dry_run {
            lines.push(format!("  backup: {}", self.outcome.backup_dir.display()));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("save".to_string(), json!(self.save_name));
        map.insert("file".to_string(), json!(self.file.display().to_string()));
        map.insert("at".to_string(), json!([self.at.x, self.at.y, self.at.z]));
        map.insert("rotate".to_string(), json!(self.rotate.map(RotateArg::as_str)));
        map.insert("size".to_string(), json!([self.size.x, self.size.y, self.size.z]));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

// -------------------------------------------------------------------------------------------------
// ---- get / set / fill (ticket 100) ----------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The flat index into [`Blueprint::blocks`] for a position already known to
/// be in `0..size` on every axis — [`SelectionBounds::index_of`]'s Y-outer/
/// Z-middle/X-inner order with an implicit min corner of [`IVec3::ZERO`],
/// spelled out here rather than routed through [`SelectionBounds`]: that
/// type's own [`SelectionBounds::from_corners`] clamps Y into the world's
/// build limits, which is the wrong bound for a structure's own
/// `0..size`-tall space (bounded instead by [`STRUCTURE_BLOCK_MAX_SIZE`],
/// comfortably inside the world's own Y range but a different constraint
/// with a different owner).
fn flat_index(size: IVec3, pos: IVec3) -> usize {
    (pos.y as usize) * (size.z as usize) * (size.x as usize)
        + (pos.z as usize) * (size.x as usize)
        + (pos.x as usize)
}

/// Bounds-checks `pos` against `size`'s `0..size` structure-local space and
/// returns its [`flat_index`]. Out of bounds is [`CliError::Usage`] naming
/// the file's actual size — the ticket's own call: unlike a live save, a
/// structure file's extent is fully known up front (from `struct info`),
/// so there's no "maybe it's just ungenerated" ambiguity worth preserving
/// with a [`CliError::Data`] here.
fn index_in_blueprint(size: IVec3, pos: IVec3) -> Result<usize, CliError> {
    if pos.cmplt(IVec3::ZERO).any() || pos.cmpge(size).any() {
        return Err(CliError::Usage(format!(
            "{pos} is outside this structure's bounds (size {}x{}x{})",
            size.x, size.y, size.z
        )));
    }
    Ok(flat_index(size, pos))
}

/// Where `struct set`/`struct fill` write: `file` itself when `out` is
/// `None`, refused without `force` — the same "editing a file in place is
/// destructive too" gate `struct new`/`struct export` apply to their own
/// `--out`, applied here to the *default* output path instead. A given
/// `--out` that already names an existing file follows the identical gate
/// rather than a special case, so "overwrite the source" and "overwrite
/// something else already sitting at `--out`" are one rule, not two.
fn resolve_struct_out(file: &std::path::Path, out: &Option<PathBuf>, force: bool) -> Result<PathBuf, CliError> {
    let target = out.clone().unwrap_or_else(|| file.to_path_buf());
    if target.exists() && !force {
        return Err(CliError::Usage(format!(
            "{} already exists — pass --force to overwrite",
            target.display()
        )));
    }
    Ok(target)
}

/// Finds `state`'s index in `palette`, inserting it as a new entry if it
/// isn't already present — the palette-growth question [`ranvil::chunkregion::ChunkRegion::set_blocks`]
/// answers for a live section's packed palette (see the module docs),
/// mirrored here for `Blueprint`'s plain `Vec<BlockState>`. Growth is capped
/// at exactly one entry per distinct new state, never a duplicate: an
/// already-present state (matched by `PartialEq`, i.e. name plus its already-
/// sorted properties — see [`BlockState`]'s own docs on why the sort makes
/// this comparison correct) reuses its existing index.
fn palette_index_for(palette: &mut Vec<BlockState>, state: BlockState) -> u16 {
    match palette.iter().position(|entry| entry == &state) {
        Some(index) => index as u16,
        None => {
            palette.push(state);
            (palette.len() - 1) as u16
        }
    }
}

/// `struct get`'s result: the block at one position inside a structure file.
#[derive(Debug)]
pub struct StructGetResult {
    pub file: PathBuf,
    /// Relative to the structure's own `0..size` space — see
    /// [`StructGetArgs::pos`].
    pub pos: IVec3,
    pub state: BlockState,
}

/// Runs `struct get`: [`read_structure_file`], then [`index_in_blueprint`]
/// against `args.pos` — the same lookup [`Blueprint::block_at`] performs,
/// except a position outside the structure refuses with
/// [`CliError::Usage`] rather than `block_at`'s silent `None`, per the
/// ticket.
pub fn get(args: &StructGetArgs) -> Result<StructGetResult, CliError> {
    let blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    let pos = args.pos.0;
    let index = index_in_blueprint(blueprint.size, pos)?;
    let state = blueprint.palette[blueprint.blocks[index] as usize].clone();

    Ok(StructGetResult { file: args.file.clone(), pos, state })
}

impl Render for StructGetResult {
    /// The same copy-pasteable block-state string [`super::block::GetResult::render_text`]
    /// prints for the world-side `get` — straight into `struct set`'s own
    /// `state` argument.
    fn render_text(&self) -> String {
        self.state.to_string()
    }

    fn render_json(&self) -> Value {
        let mut properties = serde_json::Map::new();
        for (key, value) in &self.state.properties {
            properties.insert(key.clone(), json!(value));
        }
        json!({
            "file": self.file.display().to_string(),
            "pos": [self.pos.x, self.pos.y, self.pos.z],
            "name": self.state.name,
            "properties": properties,
        })
    }
}

/// `struct set`'s result: the position and block written, where it landed,
/// and whether the palette grew to hold it.
#[derive(Debug)]
pub struct StructSetResult {
    pub file: PathBuf,
    pub out: PathBuf,
    pub pos: IVec3,
    pub state: BlockState,
    pub palette_grew: bool,
}

/// Runs `struct set`: [`read_structure_file`], [`index_in_blueprint`] on
/// `args.pos`, [`palette_index_for`] to find or grow the palette entry for
/// `args.state`, then [`write_structure_file`] to [`resolve_struct_out`]'s
/// target.
///
/// Every check — the position's bounds, the output path's `--force` gate —
/// happens before the in-memory [`Blueprint`] is touched, so a refused call
/// leaves both `args.file` and any existing `--out` exactly as they were.
pub fn set(args: &StructSetArgs) -> Result<StructSetResult, CliError> {
    let mut blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    let pos = args.pos.0;
    let index = index_in_blueprint(blueprint.size, pos)?;
    let out = resolve_struct_out(&args.file, &args.out, args.force)?;

    let before = blueprint.palette.len();
    let palette_index = palette_index_for(&mut blueprint.palette, args.state.clone());
    blueprint.blocks[index] = palette_index;
    let palette_grew = blueprint.palette.len() > before;

    write_structure_file(&out, &blueprint)
        .map_err(|err| CliError::Data(format!("could not write {}: {err}", out.display())))?;

    Ok(StructSetResult { file: args.file.clone(), out, pos, state: args.state.clone(), palette_grew })
}

impl StructSetResult {
    fn summary_line(&self) -> String {
        format!(
            "struct set {} {} = {} -> {}{}",
            self.file.display(),
            self.pos,
            self.state,
            self.out.display(),
            if self.palette_grew { " (new palette entry)" } else { "" },
        )
    }
}

impl Render for StructSetResult {
    fn render_text(&self) -> String {
        self.summary_line()
    }

    fn render_json(&self) -> Value {
        json!({
            "file": self.file.display().to_string(),
            "out": self.out.display().to_string(),
            "pos": [self.pos.x, self.pos.y, self.pos.z],
            "block": self.state.to_string(),
            "palette_grew": self.palette_grew,
        })
    }
}

/// `struct fill`'s result: the box and block filled, where it landed, how
/// many positions were rewritten, and whether the palette grew.
#[derive(Debug)]
pub struct StructFillResult {
    pub file: PathBuf,
    pub out: PathBuf,
    /// The box's min/max corners, already normalized — either order in
    /// `args.from`/`args.to` lands here the same way.
    pub from: IVec3,
    pub to: IVec3,
    pub state: BlockState,
    pub blocks_filled: usize,
    pub palette_grew: bool,
}

/// Runs `struct fill`: [`read_structure_file`], normalizes `args.from`/
/// `args.to` into a min/max box and bounds-checks both corners via
/// [`index_in_blueprint`] (sufficient for an axis-aligned box: every position
/// between two in-bounds corners is itself in bounds), [`palette_index_for`]
/// once for `args.state` — one palette-growth pass rather than one per
/// block, mirroring why [`ranvil::chunkregion::ChunkRegion::set_blocks`]
/// exists over calling `set_block` in a loop — then writes every position in
/// the box to that one index before a single [`write_structure_file`].
pub fn fill(args: &StructFillArgs) -> Result<StructFillResult, CliError> {
    let mut blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    let min = args.from.0.min(args.to.0);
    let max = args.from.0.max(args.to.0);
    index_in_blueprint(blueprint.size, min)?;
    index_in_blueprint(blueprint.size, max)?;

    let out = resolve_struct_out(&args.file, &args.out, args.force)?;

    let before = blueprint.palette.len();
    let palette_index = palette_index_for(&mut blueprint.palette, args.state.clone());
    let size = blueprint.size;

    let mut blocks_filled = 0usize;
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                let index = flat_index(size, IVec3::new(x, y, z));
                blueprint.blocks[index] = palette_index;
                blocks_filled += 1;
            }
        }
    }
    let palette_grew = blueprint.palette.len() > before;

    write_structure_file(&out, &blueprint)
        .map_err(|err| CliError::Data(format!("could not write {}: {err}", out.display())))?;

    Ok(StructFillResult { file: args.file.clone(), out, from: min, to: max, state: args.state.clone(), blocks_filled, palette_grew })
}

impl StructFillResult {
    fn summary_line(&self) -> String {
        format!(
            "struct fill {} {} to {} = {} -> {}: {} block{} filled{}",
            self.file.display(),
            self.from,
            self.to,
            self.state,
            self.out.display(),
            self.blocks_filled,
            if self.blocks_filled == 1 { "" } else { "s" },
            if self.palette_grew { " (new palette entry)" } else { "" },
        )
    }
}

impl Render for StructFillResult {
    fn render_text(&self) -> String {
        self.summary_line()
    }

    fn render_json(&self) -> Value {
        json!({
            "file": self.file.display().to_string(),
            "out": self.out.display().to_string(),
            "from": [self.from.x, self.from.y, self.from.z],
            "to": [self.to.x, self.to.y, self.to.z],
            "block": self.state.to_string(),
            "blocks_filled": self.blocks_filled,
            "palette_grew": self.palette_grew,
        })
    }
}

// -------------------------------------------------------------------------------------------------
// ---- resize (ticket 101) ---------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct resize`'s result: the six pads applied, the size before and
/// after, and where the resized file landed.
#[derive(Debug)]
pub struct StructResizeResult {
    pub file: PathBuf,
    pub out: PathBuf,
    pub old_size: IVec3,
    pub new_size: IVec3,
    pub pad_y_top: i32,
    pub pad_y_bottom: i32,
    pub pad_x_neg: i32,
    pub pad_x_pos: i32,
    pub pad_z_neg: i32,
    pub pad_z_pos: i32,
    pub fill: BlockState,
}

/// Runs `struct resize`: [`read_structure_file`], computes `new_size` from
/// the six independent pads, refuses (before writing anything) if any axis
/// would collapse to zero or less, then allocates a fresh block array of
/// `new_size` pre-filled with `args.fill` and copies every source position
/// that still lands inside it at its shifted coordinate — a crop simply
/// never copies the part that falls outside.
///
/// `--pad-y-bottom`/`--pad-x-neg`/`--pad-z-neg` are the only three pads that
/// shift a copied position's coordinate (`shift` below); `--pad-y-top`/
/// `--pad-x-pos`/`--pad-z-pos` only change where the far face of `new_size`
/// sits, so they never appear in `shift` at all. That split is what makes
/// the six pads independent of each other and of application order — see
/// the ticket's own order-independence test.
///
/// The output palette is rebuilt from scratch (starting with `args.fill` at
/// index 0, same as [`new`]) rather than reusing the source's palette
/// verbatim: a crop can drop every position of a source palette entry, and
/// starting fresh means the resized file's palette only ever lists states
/// that actually still appear in it.
pub fn resize(args: &StructResizeArgs) -> Result<StructResizeResult, CliError> {
    let blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    let old_size = blueprint.size;
    let new_size = IVec3::new(
        old_size.x + args.pad_x_neg + args.pad_x_pos,
        old_size.y + args.pad_y_bottom + args.pad_y_top,
        old_size.z + args.pad_z_neg + args.pad_z_pos,
    );

    for (component, axis) in [(new_size.x, 'x'), (new_size.y, 'y'), (new_size.z, 'z')] {
        if component < 1 {
            return Err(CliError::Usage(format!(
                "resulting {axis} size would be {component} — the pads on that axis crop away the whole structure"
            )));
        }
    }

    if args.out.exists() && !args.force {
        return Err(CliError::Usage(format!(
            "{} already exists — pass --force to overwrite",
            args.out.display()
        )));
    }

    // Where a source position's coordinate lands in `new_size`'s space —
    // only the three pads that grow/shrink the *min* corner of an axis
    // shift anything; the three that grow/shrink the *max* corner leave a
    // surviving position's coordinate exactly where it was.
    let shift = IVec3::new(args.pad_x_neg, args.pad_y_bottom, args.pad_z_neg);

    let volume = new_size.x as usize * new_size.y as usize * new_size.z as usize;
    let mut palette = vec![args.fill.clone()];
    let mut blocks = vec![0u16; volume];

    for z in 0..old_size.z {
        for y in 0..old_size.y {
            for x in 0..old_size.x {
                let old_pos = IVec3::new(x, y, z);
                let new_pos = old_pos + shift;
                if new_pos.cmplt(IVec3::ZERO).any() || new_pos.cmpge(new_size).any() {
                    continue; // cropped away
                }
                let old_index = flat_index(old_size, old_pos);
                let state = blueprint.palette[blueprint.blocks[old_index] as usize].clone();
                let palette_index = palette_index_for(&mut palette, state);
                blocks[flat_index(new_size, new_pos)] = palette_index;
            }
        }
    }

    let resized = Blueprint {
        size: new_size,
        origin: IVec3::ZERO,
        palette,
        blocks,
        data_version: blueprint.data_version,
        // A resize is one in-memory transform, all-or-nothing, same as
        // `struct import`'s read — there's no partial result to count
        // failures against (see `Blueprint::failed_columns`'s own docs).
        failed_columns: 0,
    };

    write_structure_file(&args.out, &resized).map_err(|err| {
        CliError::Data(format!("could not write {}: {err}", args.out.display()))
    })?;

    Ok(StructResizeResult {
        file: args.file.clone(),
        out: args.out.clone(),
        old_size,
        new_size,
        pad_y_top: args.pad_y_top,
        pad_y_bottom: args.pad_y_bottom,
        pad_x_neg: args.pad_x_neg,
        pad_x_pos: args.pad_x_pos,
        pad_z_neg: args.pad_z_neg,
        pad_z_pos: args.pad_z_pos,
        fill: args.fill.clone(),
    })
}

impl StructResizeResult {
    fn summary_line(&self) -> String {
        format!(
            "struct resize {} ({}x{}x{}) -> {} ({}x{}x{}): pad y[{}..{}] x[{}..{}] z[{}..{}], fill {}",
            self.file.display(),
            self.old_size.x,
            self.old_size.y,
            self.old_size.z,
            self.out.display(),
            self.new_size.x,
            self.new_size.y,
            self.new_size.z,
            self.pad_y_bottom,
            self.pad_y_top,
            self.pad_x_neg,
            self.pad_x_pos,
            self.pad_z_neg,
            self.pad_z_pos,
            self.fill,
        )
    }
}

impl Render for StructResizeResult {
    fn render_text(&self) -> String {
        self.summary_line()
    }

    fn render_json(&self) -> Value {
        json!({
            "file": self.file.display().to_string(),
            "out": self.out.display().to_string(),
            "old_size": [self.old_size.x, self.old_size.y, self.old_size.z],
            "new_size": [self.new_size.x, self.new_size.y, self.new_size.z],
            "pad_y_top": self.pad_y_top,
            "pad_y_bottom": self.pad_y_bottom,
            "pad_x_neg": self.pad_x_neg,
            "pad_x_pos": self.pad_x_pos,
            "pad_z_neg": self.pad_z_neg,
            "pad_z_pos": self.pad_z_pos,
            "fill": self.fill.to_string(),
        })
    }
}

// -------------------------------------------------------------------------------------------------
// ---- rotate (ticket 102) ---------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct rotate`'s result: the file, where the rotated copy landed, the
/// rotation applied, and the size before/after (unchanged for 180°, X/Z
/// swapped for 90°/270°).
#[derive(Debug)]
pub struct StructRotateResult {
    pub file: PathBuf,
    pub out: PathBuf,
    pub by: RotateArg,
    pub old_size: IVec3,
    pub new_size: IVec3,
}

/// Runs `struct rotate`: [`read_structure_file`], [`rotate_blueprint`] (038's
/// function — see the module docs on `struct import --rotate` calling the
/// exact same one), then [`write_structure_file`] to `args.out`.
///
/// The `--out`-exists check happens before [`rotate_blueprint`] runs, same
/// "before anything is touched" ordering [`new`]/[`export`] apply to their
/// own `--out`. A [`RotationError::UnrotatableProperty`] surfaces as
/// [`CliError::Data`] carrying that error's own `Display` — already naming
/// the offending block, property key and value — rather than a generic
/// message, per the ticket.
pub fn rotate(args: &StructRotateArgs) -> Result<StructRotateResult, CliError> {
    let blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;

    if args.out.exists() && !args.force {
        return Err(CliError::Usage(format!(
            "{} already exists — pass --force to overwrite",
            args.out.display()
        )));
    }

    let old_size = blueprint.size;
    let rotated = rotate_blueprint(&blueprint, args.by.to_rotation())
        .map_err(|err| CliError::Data(format!("{}: {err}", args.file.display())))?;
    let new_size = rotated.size;

    write_structure_file(&args.out, &rotated).map_err(|err| {
        CliError::Data(format!("could not write {}: {err}", args.out.display()))
    })?;

    Ok(StructRotateResult { file: args.file.clone(), out: args.out.clone(), by: args.by, old_size, new_size })
}

impl StructRotateResult {
    fn summary_line(&self) -> String {
        format!(
            "struct rotate {} by {} -> {}: size {}x{}x{} -> {}x{}x{}",
            self.file.display(),
            self.by.as_str(),
            self.out.display(),
            self.old_size.x,
            self.old_size.y,
            self.old_size.z,
            self.new_size.x,
            self.new_size.y,
            self.new_size.z,
        )
    }
}

impl Render for StructRotateResult {
    fn render_text(&self) -> String {
        self.summary_line()
    }

    fn render_json(&self) -> Value {
        json!({
            "file": self.file.display().to_string(),
            "out": self.out.display().to_string(),
            "by": self.by.as_str(),
            "old_size": [self.old_size.x, self.old_size.y, self.old_size.z],
            "new_size": [self.new_size.x, self.new_size.y, self.new_size.z],
        })
    }
}

// -------------------------------------------------------------------------------------------------
// ---- diff (ticket 102) -----------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// One position where two structure files' blocks disagree — properties
/// included, per the ticket ("a diff's whole purpose is catching exactly the
/// kind of change name-only matching would hide, like a rotated door nobody
/// noticed").
#[derive(Debug, Clone)]
pub struct StructDiffEntry {
    pub pos: IVec3,
    pub a: BlockState,
    pub b: BlockState,
}

/// `struct diff`'s result: every differing position, in full — `text`/
/// `compact` cap their own listing at [`Self::limit`] (see
/// [`Render::render_text`]/[`Render::render_compact`] below), but
/// [`Self::differences`] itself always holds the complete list, since
/// `render_json` reports it uncapped per the ticket.
#[derive(Debug)]
pub struct StructDiffResult {
    pub a_file: PathBuf,
    pub b_file: PathBuf,
    pub size: IVec3,
    /// How many differences `text`/`compact` list before truncating —
    /// `args.limit` or [`super::block::DEFAULT_SCAN_LIMIT`], the same default
    /// `scan --limit` uses. Does not affect `render_json`.
    pub limit: usize,
    pub differences: Vec<StructDiffEntry>,
}

/// Runs `struct diff`: [`read_structure_file`] on both `a` and `b`, refuses
/// (`Usage`, exit 2) if their `size`s differ — a per-position diff needs a
/// shared coordinate space — then walks both blueprints' dense `blocks`
/// arrays position-by-position (they're guaranteed the same length once
/// `size` matches, since [`read_structure_file`] already rejects a `blocks`
/// list that doesn't cover its own `size`), recording every position whose
/// resolved [`BlockState`] (name and properties both) differs.
pub fn diff(args: &StructDiffArgs) -> Result<StructDiffResult, CliError> {
    let a = read_structure_file(&args.a)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.a.display())))?;
    let b = read_structure_file(&args.b)
        .map_err(|err| CliError::Data(format!("{}: {err}", args.b.display())))?;

    if a.size != b.size {
        return Err(CliError::Usage(format!(
            "{} is {}x{}x{} but {} is {}x{}x{} — struct diff needs matching sizes \
             (compare with struct info on each instead)",
            args.a.display(),
            a.size.x,
            a.size.y,
            a.size.z,
            args.b.display(),
            b.size.x,
            b.size.y,
            b.size.z,
        )));
    }

    let size = a.size;
    let limit = args.limit.unwrap_or(super::block::DEFAULT_SCAN_LIMIT);

    let mut differences = Vec::new();
    for (index, (&a_index, &b_index)) in a.blocks.iter().zip(b.blocks.iter()).enumerate() {
        let state_a = &a.palette[a_index as usize];
        let state_b = &b.palette[b_index as usize];
        if state_a != state_b {
            let pos = super::block::position_of(IVec3::ZERO, size, index);
            differences.push(StructDiffEntry { pos, a: state_a.clone(), b: state_b.clone() });
        }
    }

    Ok(StructDiffResult { a_file: args.a.clone(), b_file: args.b.clone(), size, limit, differences })
}

impl StructDiffResult {
    fn summary_line(&self) -> String {
        let count = self.differences.len();
        format!(
            "struct diff {} {}: size {}x{}x{}, {count} difference{}",
            self.a_file.display(),
            self.b_file.display(),
            self.size.x,
            self.size.y,
            self.size.z,
            if count == 1 { "" } else { "s" },
        )
    }

    /// [`Self::differences`] capped at [`Self::limit`] — the sample `text`/
    /// `compact` share, and how many were left out beyond it.
    fn sample(&self) -> (&[StructDiffEntry], usize) {
        let shown = self.differences.len().min(self.limit);
        (&self.differences[..shown], self.differences.len() - shown)
    }
}

impl Render for StructDiffResult {
    /// The summary line plus one line per differing position, capped at
    /// `limit` — same "count plus a capped sample list" convention
    /// [`super::block::ScanResult::render_text`] follows for its own matches.
    fn render_text(&self) -> String {
        let (sample, rest) = self.sample();
        let mut lines = vec![self.summary_line()];
        for entry in sample {
            lines.push(format!("  {}: {} -> {}", entry.pos, entry.a, entry.b));
        }
        if rest > 0 {
            lines.push(format!("  ... and {rest} more (raise with --limit)"));
        }
        lines.join("\n")
    }

    /// The full, uncapped list — `"count"` reports the true total even when
    /// it exceeds `limit`, since `limit` only bounds `text`/`compact` here.
    fn render_json(&self) -> Value {
        json!({
            "a": self.a_file.display().to_string(),
            "b": self.b_file.display().to_string(),
            "size": [self.size.x, self.size.y, self.size.z],
            "count": self.differences.len(),
            "differences": self.differences.iter().map(|entry| json!({
                "pos": [entry.pos.x, entry.pos.y, entry.pos.z],
                "a": entry.a.to_string(),
                "b": entry.b.to_string(),
            })).collect::<Vec<_>>(),
        })
    }

    /// One line: the summary plus the same capped sample `render_text` uses,
    /// semicolon-separated — the same "pack the capped listing into one
    /// line" shape [`super::block::ColumnResult::render_compact`] uses for
    /// its own ranges.
    fn render_compact(&self) -> String {
        let (sample, rest) = self.sample();
        if sample.is_empty() {
            return self.summary_line();
        }
        let entries: Vec<String> = sample
            .iter()
            .map(|entry| format!("{}: {} -> {}", entry.pos, entry.a, entry.b))
            .collect();
        let trailer = if rest > 0 { format!("; ... and {rest} more") } else { String::new() };
        format!("{}: {}{trailer}", self.summary_line(), entries.join("; "))
    }
}

// -------------------------------------------------------------------------------------------------
// ---- validate (ticket 103) --------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// `struct validate`'s result: every named check [`run_checks`] ran, in
/// order, against the size ceiling actually used (whichever `--max-size`
/// resolved to). Reaching this struct at all already means `args.file`
/// parsed as a structure — a parse failure never gets this far, see
/// [`validate`]'s own docs on why that's a different, earlier error instead.
#[derive(Debug)]
pub struct StructValidateResult {
    pub file: PathBuf,
    pub max_size: IVec3,
    pub checks: Vec<BlueprintCheck>,
}

impl StructValidateResult {
    /// Whether every check passed. [`super::run`]'s dispatch reads this to
    /// choose exit `0` vs `1` per the ticket's contract, rather than folding
    /// that choice into a `CliError` the way every other `struct` command's
    /// failure path does — see [`validate`]'s docs on why this one result is
    /// always `Ok`.
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|check| check.pass)
    }
}

/// Runs `struct validate`: [`read_structure_file`] on `args.file`, then
/// [`run_checks`] — the exact per-file checks
/// [`crate::blueprint::catalogue::load_catalogue_dir`] (ticket 039) applies
/// silently to every building it scans, against `args.max_size` (or
/// [`STRUCTURE_BLOCK_MAX_SIZE`] splatted across all three axes when not
/// given, the same limit the catalogue loader enforces).
///
/// A file that doesn't parse as a structure at all is [`CliError::Usage`]
/// (exit `2`), not [`CliError::Data`] (exit `1`, every other `struct`
/// command's convention for a read failure) — per the ticket, there's no
/// file here to run checks against and report on, so this is closer to a
/// bad request than a well-formed one that came back with a real, negative
/// answer. A file that *does* parse but fails one or more checks is always
/// `Ok`: the per-check listing in [`Self::checks`] is itself the answer, not
/// a error to replace with one, so [`Self::all_passed`] — read by
/// [`super::run`]'s dispatch — is what picks exit `1` over `0`, not this
/// `Result`.
pub fn validate(args: &StructValidateArgs) -> Result<StructValidateResult, CliError> {
    let blueprint = read_structure_file(&args.file)
        .map_err(|err| CliError::Usage(format!("{}: {err}", args.file.display())))?;

    let max_size = args
        .max_size
        .map(|pos| pos.0)
        .unwrap_or(IVec3::splat(STRUCTURE_BLOCK_MAX_SIZE));
    let checks = run_checks(&blueprint, max_size);

    Ok(StructValidateResult { file: args.file.clone(), max_size, checks })
}

impl StructValidateResult {
    fn summary_line(&self) -> String {
        let passed = self.checks.iter().filter(|check| check.pass).count();
        format!(
            "struct validate {}: {}/{} checks passed",
            self.file.display(),
            passed,
            self.checks.len(),
        )
    }
}

impl Render for StructValidateResult {
    /// The summary line plus one `pass`/`FAIL` line per check, in
    /// [`run_checks`]'s own order.
    fn render_text(&self) -> String {
        let mut lines = vec![self.summary_line()];
        for check in &self.checks {
            lines.push(format!(
                "  [{}] {}: {}",
                if check.pass { "pass" } else { "FAIL" },
                check.name,
                check.detail,
            ));
        }
        lines.join("\n")
    }

    /// `"checks"` lists every check's `name`/`pass`/`detail` — per the
    /// ticket, so an agent can see *which* check failed rather than
    /// re-deriving it from `"pass"` alone. `"pass"` at the top level is
    /// [`Self::all_passed`], the same overall verdict the exit code encodes.
    fn render_json(&self) -> Value {
        json!({
            "file": self.file.display().to_string(),
            "max_size": [self.max_size.x, self.max_size.y, self.max_size.z],
            "pass": self.all_passed(),
            "checks": self.checks.iter().map(|check| json!({
                "name": check.name,
                "pass": check.pass,
                "detail": check.detail,
            })).collect::<Vec<_>>(),
        })
    }

    /// The summary line, plus the names of any failing checks — a passing
    /// file is one line with nothing more to say, mirroring
    /// [`StructDiffResult::render_compact`]'s "nothing to list" case.
    fn render_compact(&self) -> String {
        let failing: Vec<&str> = self
            .checks
            .iter()
            .filter(|check| !check.pass)
            .map(|check| check.name)
            .collect();
        if failing.is_empty() {
            self.summary_line()
        } else {
            format!("{}: failed {}", self.summary_line(), failing.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::coords::BlockPos;

    fn temp_path(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("block_viewer-ranvil-cli-struct-new-{label}-{nanos}.nbt"))
    }

    fn new_args(size: IVec3, out: PathBuf, fill: BlockState, force: bool) -> StructNewArgs {
        StructNewArgs { size: BlockPos(size), out, fill, force }
    }

    // -----------------------------------------------------------------------------------------
    // ---- info -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    fn state(name: &str) -> BlockState {
        BlockState { name: name.to_string(), properties: Vec::new() }
    }

    fn sample_blueprint() -> Blueprint {
        Blueprint {
            size: IVec3::new(2, 1, 1),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state("minecraft:stone")],
            blocks: vec![0, 1],
            data_version: 3953,
            failed_columns: 0,
        }
    }

    /// The ticket's own "Done when": `struct info` against a real file
    /// reports exactly the `Blueprint` fields the writer put in — size,
    /// origin (always zero), block count, `DataVersion`, palette.
    #[test]
    fn info_reports_a_written_structures_own_fields() {
        let path = temp_path("info-roundtrip");
        let blueprint = sample_blueprint();
        write_structure_file(&path, &blueprint).expect("should write");

        let result = info(&StructInfoArgs { file: path.clone() }).expect("should read back");
        std::fs::remove_file(&path).ok();

        assert_eq!(result.size, blueprint.size);
        assert_eq!(result.origin, IVec3::ZERO);
        assert_eq!(result.blocks, blueprint.volume());
        assert_eq!(result.data_version, blueprint.data_version);
        assert_eq!(result.palette, blueprint.palette);
    }

    /// A missing file is a `Data` error (exit 1) carrying the reader's own
    /// message — not a panic, and not a `Usage` error (the path itself was
    /// well-formed; it's the read that failed).
    #[test]
    fn info_on_a_missing_file_is_a_data_error() {
        let path = temp_path("info-missing");
        let err = info(&StructInfoArgs { file: path }).unwrap_err();
        match err {
            CliError::Data(_) => {}
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }
    }

    /// `render_text` lists every palette entry when the palette is under
    /// `LOGGED_PALETTE_ENTRIES`, with no "... and N more" trailer.
    #[test]
    fn render_text_lists_the_whole_palette_when_its_small() {
        let result = StructInfoResult {
            path: PathBuf::from("test.nbt"),
            size: IVec3::new(2, 1, 1),
            origin: IVec3::ZERO,
            blocks: 2,
            data_version: 3953,
            palette: vec![BlockState::air(), state("minecraft:stone")],
        };
        let text = result.render_text();
        assert!(text.contains("minecraft:air"), "{text}");
        assert!(text.contains("minecraft:stone"), "{text}");
        assert!(!text.contains("more"), "{text}");
    }

    /// A palette over `LOGGED_PALETTE_ENTRIES` is capped, with a trailer
    /// naming exactly how many entries were left out — the same contract
    /// `blueprint::log_blueprint` follows for its own console log.
    #[test]
    fn render_text_caps_a_large_palette_with_a_trailer() {
        let extra = 5;
        let palette: Vec<BlockState> = (0..LOGGED_PALETTE_ENTRIES + extra)
            .map(|i| state(&format!("minecraft:x{i}")))
            .collect();
        let result = StructInfoResult {
            path: PathBuf::from("test.nbt"),
            size: IVec3::new(1, 1, 1),
            origin: IVec3::ZERO,
            blocks: 1,
            data_version: 3953,
            palette,
        };
        let text = result.render_text();
        assert!(text.contains(&format!("... and {extra} more")), "{text}");
        assert!(text.contains("minecraft:x63"), "index 63 is the last one under the cap: {text}");
        assert!(!text.contains("minecraft:x64"), "index 64 is past the cap: {text}");
    }

    /// `render_compact` is the summary line alone — no palette listing, even
    /// for a small palette.
    #[test]
    fn render_compact_has_no_palette_listing() {
        let result = StructInfoResult {
            path: PathBuf::from("test.nbt"),
            size: IVec3::new(2, 1, 1),
            origin: IVec3::ZERO,
            blocks: 2,
            data_version: 3953,
            palette: vec![BlockState::air(), state("minecraft:stone")],
        };
        let compact = result.render_compact();
        assert!(!compact.contains("stone"), "{compact}");
        assert!(compact.contains("2 distinct states"), "{compact}");
    }

    #[test]
    fn render_json_matches_the_documented_shape() {
        let result = StructInfoResult {
            path: PathBuf::from("house.nbt"),
            size: IVec3::new(2, 1, 1),
            origin: IVec3::ZERO,
            blocks: 2,
            data_version: 3953,
            palette: vec![BlockState::air(), state("minecraft:stone")],
        };
        assert_eq!(
            result.render_json(),
            json!({
                "path": "house.nbt",
                "size": [2, 1, 1],
                "origin": [0, 0, 0],
                "blocks": 2,
                "data_version": 3953,
                "palette_size": 2,
                "palette": ["minecraft:air", "minecraft:stone"],
            })
        );
    }

    // -----------------------------------------------------------------------------------------
    // ---- new ------------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    /// The ticket's own "Done when": `struct new --size 5,5,5` produces a
    /// file `struct info` reports as `5x5x5`, all-air, palette size 1.
    #[test]
    fn new_with_default_fill_produces_an_all_air_structure() {
        let path = temp_path("new-default");
        let result = new(&new_args(IVec3::splat(5), path.clone(), BlockState::air(), false))
            .expect("a valid size should write");

        assert_eq!(result.size, IVec3::splat(5));
        assert_eq!(result.fill, BlockState::air());

        let info_result = info(&StructInfoArgs { file: path.clone() }).expect("should read back");
        std::fs::remove_file(&path).ok();

        assert_eq!(info_result.size, IVec3::splat(5));
        assert_eq!(info_result.blocks, 125);
        assert_eq!(info_result.palette, vec![BlockState::air()]);
    }

    /// `--fill minecraft:stone` produces a palette of size 1, all stone —
    /// the ticket's other "Done when" for `struct new`.
    #[test]
    fn new_with_a_fill_produces_a_single_entry_palette_of_that_block() {
        let path = temp_path("new-fill");
        let stone = state("minecraft:stone");
        new(&new_args(IVec3::new(3, 3, 3), path.clone(), stone.clone(), false))
            .expect("a valid size should write");

        let info_result = info(&StructInfoArgs { file: path.clone() }).expect("should read back");
        std::fs::remove_file(&path).ok();

        assert_eq!(info_result.palette, vec![stone]);
        assert_eq!(info_result.blocks, 27);
    }

    /// Writing onto an existing path without `--force` refuses, and touches
    /// nothing — the same convention every other file-writing `struct`
    /// command follows.
    #[test]
    fn new_onto_an_existing_path_without_force_refuses() {
        let path = temp_path("new-no-force");
        new(&new_args(IVec3::ONE, path.clone(), BlockState::air(), false)).expect("first write");
        let before = std::fs::read(&path).expect("read the first write");

        let err = new(&new_args(IVec3::splat(2), path.clone(), state("minecraft:stone"), false))
            .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("--force"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(std::fs::read(&path).expect("still readable"), before, "must not have overwritten");
        std::fs::remove_file(&path).ok();
    }

    /// With `--force`, the same path overwrites.
    #[test]
    fn new_onto_an_existing_path_with_force_overwrites() {
        let path = temp_path("new-force");
        new(&new_args(IVec3::ONE, path.clone(), BlockState::air(), false)).expect("first write");

        let stone = state("minecraft:stone");
        new(&new_args(IVec3::splat(2), path.clone(), stone.clone(), true)).expect("forced overwrite");

        let info_result = info(&StructInfoArgs { file: path.clone() }).expect("should read back");
        std::fs::remove_file(&path).ok();

        assert_eq!(info_result.size, IVec3::splat(2));
        assert_eq!(info_result.palette, vec![stone]);
    }

    /// Each axis is checked individually against `STRUCTURE_BLOCK_MAX_SIZE`
    /// — a size one over the cap on any single axis refuses as `Usage`
    /// (exit 2), before any file is touched.
    #[test]
    fn new_rejects_a_size_over_the_structure_block_cap() {
        let path = temp_path("new-too-big");
        let err = new(&new_args(
            IVec3::new(STRUCTURE_BLOCK_MAX_SIZE + 1, 1, 1),
            path.clone(),
            BlockState::air(),
            false,
        ))
        .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("x component"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert!(!path.exists(), "a rejected size must not write a file");
    }

    /// Zero and negative components are rejected the same way — "at least
    /// 1" is part of the same check, not a separate one.
    #[test]
    fn new_rejects_a_zero_or_negative_size_component() {
        let path = temp_path("new-zero");
        for size in [IVec3::new(0, 1, 1), IVec3::new(1, -1, 1), IVec3::new(1, 1, -5)] {
            let err = new(&new_args(size, path.clone(), BlockState::air(), false)).unwrap_err();
            assert!(matches!(err, CliError::Usage(_)), "size {size} should be rejected");
            assert!(!path.exists());
        }
    }

    // -----------------------------------------------------------------------------------------
    // ---- export / import (ticket 099) ----------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use mc_anvil::SaveMeta;
    use rnbt::{NbtField, NbtList, NbtValue};

    use super::super::cli::{Command, SavesArgs};
    use super::super::format::OutputFormat;

    /// The `DataVersion` the fixture's one chunk claims — a 1.21 release,
    /// same as [`super::super::edit::tests`]'s own fixture.
    const FIXTURE_DATA_VERSION: i32 = 4438;

    /// A finished chunk: one all-stone section at `Y = 0`, `Status =
    /// minecraft:full`, the given `DataVersion` — the same shape
    /// [`super::super::edit::tests::Fixture`] builds, copied rather than
    /// shared since it's private to that module.
    fn full_chunk(data_version: i32) -> NbtField {
        let palette = NbtList::Compound(vec![NbtField::new_compound(
            "",
            vec![NbtField::new_string("Name", "minecraft:stone")],
        )]);
        let section = NbtField::new_compound(
            "",
            vec![
                NbtField { name: "Y".to_string(), value: NbtValue::Byte(0) },
                NbtField::new_compound("block_states", vec![NbtField::new_list("palette", palette)]),
            ],
        );
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_list("sections", NbtList::Compound(vec![section])),
                NbtField::new_i32("xPos", 0),
                NbtField::new_i32("zPos", 0),
                NbtField::new_i32("yPos", -4),
                NbtField::new_i32("DataVersion", data_version),
                NbtField::new_string("Status", "minecraft:full"),
                NbtField { name: "isLightOn".to_string(), value: NbtValue::Byte(1) },
            ],
        )
    }

    /// A single-region, single-chunk fixture save in a temp directory,
    /// removed on drop.
    struct Fixture {
        dir: PathBuf,
        meta: SaveMeta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("block_viewer-ranvil-cli-struct-export-import-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk(FIXTURE_DATA_VERSION)));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = SaveMeta {
                name: "struct-fixture".to_string(),
                path: dir.clone(),
                region_dir,
                regions: vec![(0, 0)],
            };
            Self { dir, meta }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn cli_for(fixture: &Fixture) -> Cli {
        Cli {
            save: Some(fixture.meta.path.to_string_lossy().to_string()),
            instance: Some(PathBuf::from("does-not-exist")),
            format: OutputFormat::Json,
            command: Command::Saves(SavesArgs {}),
        }
    }

    fn block_state_at(meta: &SaveMeta, at: IVec3) -> BlockState {
        let address = crate::edit::address_of(at);
        let mut cache = RegionCache::new(meta.clone(), 1);
        let entry = cache
            .get_or_load(address.region)
            .expect("resident")
            .get_block(address.local_x, address.y, address.local_z)
            .expect("a populated chunk");
        BlockState::from_palette_entry(entry).expect("valid palette entry")
    }

    /// Plants a small, non-uniform pattern at `origin`/`origin + (1, 0, 1)`
    /// on top of the fixture's all-stone background: a multi-property
    /// stairs, a dirt, and an explicit air, leaving one corner as the
    /// fixture's own stone — four distinct states across a 2x1x2 box.
    fn plant_pattern(meta: &SaveMeta, origin: IVec3) {
        let stairs = BlockState {
            name: "minecraft:oak_stairs".to_string(),
            properties: vec![
                ("facing".to_string(), "east".to_string()),
                ("half".to_string(), "top".to_string()),
            ],
        };
        let dirt = BlockState { name: "minecraft:dirt".to_string(), properties: Vec::new() };

        run_write(meta, false, false, move |_cache| {
            let mut edit = WorldEdit::new();
            edit.set(origin, stairs.clone());
            edit.set(origin + IVec3::new(1, 0, 0), dirt.clone());
            edit.set(origin + IVec3::new(0, 0, 1), BlockState::air());
            Ok(edit)
        })
        .expect("planting the fixture pattern should succeed");
    }

    fn export_args(from: IVec3, to: IVec3, out: PathBuf, force: bool) -> StructExportArgs {
        StructExportArgs { from: BlockPos(from), to: BlockPos(to), out, force }
    }

    fn import_args(
        file: PathBuf,
        at: IVec3,
        rotate: Option<RotateArg>,
        dry_run: bool,
        force: bool,
    ) -> StructImportArgs {
        StructImportArgs { file, at: BlockPos(at), rotate, dry_run, force }
    }

    /// The ticket's headline round trip: `struct export` over a box,
    /// followed by `struct import` of the resulting file back at the *same*
    /// coordinates on a separate fixture save (standing in for "a copy of
    /// the fixture save"), reproduces the original blocks exactly.
    #[test]
    fn export_then_import_at_the_same_coordinates_reproduces_the_original_blocks() {
        let source = Fixture::new("roundtrip-source");
        let origin = IVec3::new(1, 5, 1);
        plant_pattern(&source.meta, origin);
        let far = origin + IVec3::new(1, 0, 1);

        let out = temp_path("export-roundtrip");
        let export_result = export(&cli_for(&source), &export_args(origin, far, out.clone(), false))
            .expect("export should succeed");
        assert_eq!(export_result.size, IVec3::new(2, 1, 2));
        assert_eq!(export_result.blocks, 4);
        assert_eq!(export_result.data_version, FIXTURE_DATA_VERSION);
        assert_eq!(export_result.failed_columns, 0);

        let dest = Fixture::new("roundtrip-dest");
        let import_result = import(&cli_for(&dest), &import_args(out.clone(), origin, None, false, false))
            .expect("import should succeed");
        std::fs::remove_file(&out).ok();

        assert!(!import_result.outcome.dry_run);
        assert_eq!(import_result.outcome.report.blocks_written, 4);

        for dx in 0..2 {
            for dz in 0..2 {
                let pos = origin + IVec3::new(dx, 0, dz);
                assert_eq!(
                    block_state_at(&dest.meta, pos),
                    block_state_at(&source.meta, pos),
                    "at {pos}"
                );
            }
        }
    }

    /// `struct import --at <elsewhere>` places the structure at the new
    /// location only, leaving the coordinates it was exported from
    /// untouched (the destination save never had that pattern to begin
    /// with — it's still the fixture's own stone).
    #[test]
    fn import_at_elsewhere_places_only_the_new_location() {
        let source = Fixture::new("elsewhere-source");
        let origin = IVec3::new(1, 5, 1);
        plant_pattern(&source.meta, origin);
        let far = origin + IVec3::new(1, 0, 1);

        let out = temp_path("import-elsewhere");
        export(&cli_for(&source), &export_args(origin, far, out.clone(), false)).expect("export");

        let dest = Fixture::new("elsewhere-dest");
        let elsewhere = IVec3::new(8, 5, 8);
        import(&cli_for(&dest), &import_args(out.clone(), elsewhere, None, false, false))
            .expect("import should succeed");
        std::fs::remove_file(&out).ok();

        for dx in 0..2 {
            for dz in 0..2 {
                let offset = IVec3::new(dx, 0, dz);
                assert_eq!(
                    block_state_at(&dest.meta, elsewhere + offset),
                    block_state_at(&source.meta, origin + offset),
                    "at the destination, offset {offset}"
                );
                assert_eq!(
                    block_state_at(&dest.meta, origin + offset).name,
                    "minecraft:stone",
                    "the coordinates the structure was exported from must stay untouched"
                );
            }
        }
    }

    /// `--dry-run` on `struct import` leaves the destination save
    /// byte-identical.
    #[test]
    fn import_dry_run_leaves_the_destination_untouched() {
        let source = Fixture::new("dry-run-source");
        let origin = IVec3::new(1, 5, 1);
        plant_pattern(&source.meta, origin);
        let far = origin + IVec3::new(1, 0, 1);

        let out = temp_path("import-dry-run");
        export(&cli_for(&source), &export_args(origin, far, out.clone(), false)).expect("export");

        let dest = Fixture::new("dry-run-dest");
        let region_path = dest.meta.get_region_path(0, 0);
        let before_bytes = std::fs::read(&region_path).expect("read the fixture region");

        let result = import(&cli_for(&dest), &import_args(out.clone(), origin, None, true, false))
            .expect("a valid plan");
        std::fs::remove_file(&out).ok();

        assert!(result.outcome.dry_run);
        assert_eq!(result.outcome.report.blocks_written, 4);
        assert_eq!(
            std::fs::read(&region_path).expect("read the fixture region"),
            before_bytes,
            "dry-run must not touch the region file's bytes"
        );
        assert_eq!(block_state_at(&dest.meta, origin).name, "minecraft:stone");
    }

    /// `struct import --rotate 90` agrees with rotating the same blueprint
    /// by hand through [`rotate_blueprint`] — the function `struct rotate`
    /// (ticket 102) will itself call once it exists, so this is the "the two
    /// paths agree" cross-check the ticket asks for, without depending on
    /// that command's own CLI surface landing first.
    #[test]
    fn import_rotate_90_agrees_with_rotating_the_blueprint_by_hand() {
        let source = Fixture::new("rotate-source");
        let origin = IVec3::new(1, 5, 1);
        plant_pattern(&source.meta, origin);
        let far = origin + IVec3::new(1, 0, 1);

        let out = temp_path("import-rotate");
        export(&cli_for(&source), &export_args(origin, far, out.clone(), false)).expect("export");

        let at = IVec3::new(8, 5, 8);

        // The path `struct import --rotate` itself takes.
        let dest_a = Fixture::new("rotate-dest-a");
        let result = import(
            &cli_for(&dest_a),
            &import_args(out.clone(), at, Some(RotateArg::Deg90), false, false),
        )
        .expect("rotated import should succeed");

        // The same rotation, done by hand against a second, independent
        // destination.
        let blueprint = read_structure_file(&out).expect("should read back");
        std::fs::remove_file(&out).ok();
        let rotated = rotate_blueprint(&blueprint, crate::blueprint::Rotation::Deg90).expect("should rotate");
        assert_eq!(rotated.size, result.size, "import's own rotation should agree on the resulting size");
        let size = rotated.size;

        let dest_b = Fixture::new("rotate-dest-b");
        run_write(&dest_b.meta, false, false, move |_cache| {
            let mut edit = WorldEdit::new().with_data_version(rotated.data_version);
            for (index, &palette_index) in rotated.blocks.iter().enumerate() {
                let state = rotated.palette[palette_index as usize].clone();
                let local = position_of(IVec3::ZERO, rotated.size, index);
                edit.set(at + local, state);
            }
            Ok(edit)
        })
        .expect("manual rotated write should succeed");

        for dx in 0..size.x {
            for dz in 0..size.z {
                let pos = at + IVec3::new(dx, 0, dz);
                assert_eq!(
                    block_state_at(&dest_a.meta, pos),
                    block_state_at(&dest_b.meta, pos),
                    "at {pos}"
                );
            }
        }
    }

    /// `struct export` over `MAX_BLOCKS` is a `Usage` error (exit 2), raised
    /// before `resolve_save` runs — same convention `get-area`/`set-area`'s
    /// own volume checks follow.
    #[test]
    fn export_over_max_blocks_is_a_usage_error_before_touching_a_save() {
        let cli = Cli {
            save: None,
            instance: Some(PathBuf::from("does-not-exist")),
            format: OutputFormat::Json,
            command: Command::Saves(SavesArgs {}),
        };
        let args = export_args(
            IVec3::new(0, crate::selection::WORLD_MIN_Y, 0),
            IVec3::new(9999, crate::selection::WORLD_MAX_Y, 9999),
            temp_path("export-too-big"),
            false,
        );

        let err = export(&cli, &args).unwrap_err();
        match err {
            CliError::Usage(message) => {
                assert!(message.contains(&MAX_BLOCKS.to_string()), "{message}");
                assert!(
                    !message.contains("instance directory"),
                    "should fail on the volume check, not on resolving a save: {message}"
                );
            }
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    /// `struct export --out` refuses to overwrite an existing file without
    /// `--force`, and touches nothing — same convention `struct new`'s own
    /// `out`-exists check follows.
    #[test]
    fn export_onto_an_existing_path_without_force_refuses() {
        let source = Fixture::new("export-no-force-source");
        let origin = IVec3::new(1, 5, 1);
        plant_pattern(&source.meta, origin);
        let far = origin + IVec3::new(1, 0, 1);

        let out = temp_path("export-no-force");
        std::fs::write(&out, b"not a structure file").expect("seed an existing file");

        let err = export(&cli_for(&source), &export_args(origin, far, out.clone(), false)).unwrap_err();
        match err {
            CliError::Usage(message) => assert!(message.contains("--force"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(
            std::fs::read(&out).expect("still readable"),
            b"not a structure file",
            "must not have overwritten"
        );
        std::fs::remove_file(&out).ok();
    }

    // -----------------------------------------------------------------------------------------
    // ---- get / set / fill (ticket 100) -----------------------------------------------------
    // -----------------------------------------------------------------------------------------

    fn write_sample(label: &str) -> PathBuf {
        let path = temp_path(label);
        write_structure_file(&path, &sample_blueprint()).expect("should write");
        path
    }

    fn get_args(file: PathBuf, pos: IVec3) -> StructGetArgs {
        StructGetArgs { file, pos: BlockPos(pos) }
    }

    fn set_args(file: PathBuf, pos: IVec3, state: BlockState, out: Option<PathBuf>, force: bool) -> StructSetArgs {
        StructSetArgs { file, pos: BlockPos(pos), state, out, force }
    }

    fn fill_args(
        file: PathBuf,
        from: IVec3,
        to: IVec3,
        state: BlockState,
        out: Option<PathBuf>,
        force: bool,
    ) -> StructFillArgs {
        StructFillArgs { file, from: BlockPos(from), to: BlockPos(to), state, out, force }
    }

    /// `struct get` reads back exactly what [`sample_blueprint`] put at each
    /// of its two positions.
    #[test]
    fn get_reads_the_blueprints_own_blocks() {
        let path = write_sample("get-basic");

        let air = get(&get_args(path.clone(), IVec3::new(0, 0, 0))).expect("in bounds").state;
        let stone = get(&get_args(path.clone(), IVec3::new(1, 0, 0))).expect("in bounds").state;
        std::fs::remove_file(&path).ok();

        assert_eq!(air, BlockState::air());
        assert_eq!(stone, state("minecraft:stone"));
    }

    /// The ticket's own "Done when": `struct set` then `struct get` on the
    /// same file agree.
    #[test]
    fn set_then_get_on_the_same_file_round_trips() {
        let path = write_sample("set-roundtrip-same-file");
        let stairs = state("minecraft:oak_stairs");

        set(&set_args(path.clone(), IVec3::new(0, 0, 0), stairs.clone(), None, true))
            .expect("in-place set with --force");

        let read_back = get(&get_args(path.clone(), IVec3::new(0, 0, 0))).expect("in bounds").state;
        std::fs::remove_file(&path).ok();

        assert_eq!(read_back, stairs);
    }

    /// The same round trip via `--out`: the original is untouched, and the
    /// new file carries the edit.
    #[test]
    fn set_with_out_leaves_the_original_untouched_and_writes_the_edit_to_the_new_file() {
        let path = write_sample("set-roundtrip-out-source");
        let out = temp_path("set-roundtrip-out-dest");
        let stairs = state("minecraft:oak_stairs");

        set(&set_args(path.clone(), IVec3::new(0, 0, 0), stairs.clone(), Some(out.clone()), false))
            .expect("a fresh --out path needs no --force");

        let original_at_zero = get(&get_args(path.clone(), IVec3::new(0, 0, 0))).expect("in bounds").state;
        let edited_at_zero = get(&get_args(out.clone(), IVec3::new(0, 0, 0))).expect("in bounds").state;
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&out).ok();

        assert_eq!(original_at_zero, BlockState::air(), "the source file must be untouched");
        assert_eq!(edited_at_zero, stairs);
    }

    /// The ticket's own "Done when": a blockstate not already in the palette
    /// grows it by exactly one entry.
    #[test]
    fn set_with_a_new_blockstate_grows_the_palette_by_exactly_one() {
        let path = write_sample("set-grows-palette");
        let before = info(&StructInfoArgs { file: path.clone() }).expect("read").palette.len();

        let result = set(&set_args(path.clone(), IVec3::new(0, 0, 0), state("minecraft:dirt"), None, true))
            .expect("in-place set");
        let after = info(&StructInfoArgs { file: path.clone() }).expect("read").palette.len();
        std::fs::remove_file(&path).ok();

        assert!(result.palette_grew);
        assert_eq!(after, before + 1);
    }

    /// The ticket's other "Done when": setting to a state already present
    /// does not duplicate it.
    #[test]
    fn set_with_an_existing_blockstate_does_not_duplicate_it() {
        let path = write_sample("set-no-duplicate");
        let before = info(&StructInfoArgs { file: path.clone() }).expect("read").palette.len();

        // Position (0, 0, 0) is already air; setting it to the *other*
        // existing entry (stone) should still not grow the palette.
        let result = set(&set_args(path.clone(), IVec3::new(0, 0, 0), state("minecraft:stone"), None, true))
            .expect("in-place set");
        let after = info(&StructInfoArgs { file: path.clone() }).expect("read").palette.len();
        std::fs::remove_file(&path).ok();

        assert!(!result.palette_grew);
        assert_eq!(after, before);
    }

    /// `struct set` without `--out` and without `--force` refuses, and
    /// touches nothing — the "editing in place is destructive too" gate.
    #[test]
    fn set_in_place_without_force_refuses_and_touches_nothing() {
        let path = write_sample("set-no-force");
        let before = std::fs::read(&path).expect("read the original");

        let err = set(&set_args(path.clone(), IVec3::new(0, 0, 0), state("minecraft:dirt"), None, false))
            .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("--force"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(std::fs::read(&path).expect("still readable"), before, "must not have overwritten");
        std::fs::remove_file(&path).ok();
    }

    /// `struct set --out` onto an existing path without `--force` refuses the
    /// same way — the gate covers "overwrite the source" and "overwrite
    /// something else at `--out`" identically.
    #[test]
    fn set_with_out_onto_an_existing_path_without_force_refuses() {
        let path = write_sample("set-out-no-force-source");
        let out = temp_path("set-out-no-force-dest");
        std::fs::write(&out, b"not a structure file").expect("seed an existing file");

        let err = set(&set_args(path.clone(), IVec3::new(0, 0, 0), state("minecraft:dirt"), Some(out.clone()), false))
            .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("--force"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(
            std::fs::read(&out).expect("still readable"),
            b"not a structure file",
            "must not have overwritten"
        );
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&out).ok();
    }

    /// The ticket's own "Done when" for `struct fill`: a fill over a sub-box
    /// changes only that sub-box — verified against `struct info`'s block
    /// count before/after (unchanged, fill replaces rather than adds) and
    /// sampled `struct get` calls both inside and outside the filled box.
    #[test]
    fn fill_changes_only_the_sub_box() {
        let path = temp_path("fill-sub-box");
        let blueprint = Blueprint {
            size: IVec3::new(4, 1, 4),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air()],
            blocks: vec![0u16; 16],
            data_version: 3953,
            failed_columns: 0,
        };
        write_structure_file(&path, &blueprint).expect("should write");
        let before_blocks = info(&StructInfoArgs { file: path.clone() }).expect("read").blocks;

        let result = fill(&fill_args(
            path.clone(),
            IVec3::new(1, 0, 1),
            IVec3::new(2, 0, 2),
            state("minecraft:stone"),
            None,
            true,
        ))
        .expect("in-bounds fill");

        let after_info = info(&StructInfoArgs { file: path.clone() }).expect("read");

        assert_eq!(result.blocks_filled, 4);
        assert!(result.palette_grew);
        assert_eq!(after_info.blocks, before_blocks, "fill replaces, never adds blocks");

        // Inside the filled sub-box: stone everywhere.
        for (x, z) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            let got = get(&get_args(path.clone(), IVec3::new(x, 0, z))).expect("in bounds").state;
            assert_eq!(got, state("minecraft:stone"), "at ({x}, 0, {z})");
        }
        // Outside the filled sub-box: still air.
        for (x, z) in [(0, 0), (3, 0), (0, 3), (3, 3)] {
            let got = get(&get_args(path.clone(), IVec3::new(x, 0, z))).expect("in bounds").state;
            assert_eq!(got, BlockState::air(), "at ({x}, 0, {z})");
        }

        std::fs::remove_file(&path).ok();
    }

    /// A corner given in the opposite order (`from` greater than `to` on
    /// every axis) still fills the same box — corners normalize like every
    /// other box command in this CLI.
    #[test]
    fn fill_normalizes_reversed_corners() {
        let path = temp_path("fill-reversed");
        let blueprint = Blueprint {
            size: IVec3::new(3, 1, 3),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air()],
            blocks: vec![0u16; 9],
            data_version: 3953,
            failed_columns: 0,
        };
        write_structure_file(&path, &blueprint).expect("should write");

        let result = fill(&fill_args(
            path.clone(),
            IVec3::new(2, 0, 2),
            IVec3::new(0, 0, 0),
            state("minecraft:stone"),
            None,
            true,
        ))
        .expect("reversed corners should normalize");
        std::fs::remove_file(&path).ok();

        assert_eq!(result.blocks_filled, 9);
        assert_eq!(result.from, IVec3::ZERO);
        assert_eq!(result.to, IVec3::new(2, 0, 2));
    }

    /// The ticket's own "Done when": out-of-bounds coordinates on `struct
    /// get` exit 2 (`Usage`) with the file's actual size named.
    #[test]
    fn get_out_of_bounds_is_a_usage_error_naming_the_actual_size() {
        let path = write_sample("get-out-of-bounds");
        let err = get(&get_args(path.clone(), IVec3::new(5, 0, 0))).unwrap_err();
        std::fs::remove_file(&path).ok();

        match err {
            CliError::Usage(message) => assert!(message.contains("2x1x1"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    /// Same contract for `struct set` — refused before anything is written.
    #[test]
    fn set_out_of_bounds_is_a_usage_error_and_touches_nothing() {
        let path = write_sample("set-out-of-bounds");
        let before = std::fs::read(&path).expect("read the original");

        let err = set(&set_args(path.clone(), IVec3::new(-1, 0, 0), state("minecraft:dirt"), None, true))
            .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("2x1x1"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(std::fs::read(&path).expect("still readable"), before, "must not have written");
        std::fs::remove_file(&path).ok();
    }

    /// Same contract for `struct fill` — a box that partly spills outside
    /// the structure is refused in full, not partially applied.
    #[test]
    fn fill_out_of_bounds_is_a_usage_error_and_touches_nothing() {
        let path = write_sample("fill-out-of-bounds");
        let before = std::fs::read(&path).expect("read the original");

        let err = fill(&fill_args(
            path.clone(),
            IVec3::new(0, 0, 0),
            IVec3::new(5, 0, 0),
            state("minecraft:dirt"),
            None,
            true,
        ))
        .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("2x1x1"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(std::fs::read(&path).expect("still readable"), before, "must not have written");
        std::fs::remove_file(&path).ok();
    }

    /// `struct get`/`struct set`/`struct fill` on a missing file are `Data`
    /// errors (exit 1), same as `struct info`'s own missing-file test — the
    /// path was well-formed, it's the read that failed.
    #[test]
    fn missing_file_is_a_data_error_for_get_set_and_fill() {
        let path = temp_path("missing-for-get-set-fill");

        let get_err = get(&get_args(path.clone(), IVec3::ZERO)).unwrap_err();
        assert!(matches!(get_err, CliError::Data(_)), "get: {get_err:?}");

        let set_err = set(&set_args(path.clone(), IVec3::ZERO, state("minecraft:dirt"), None, true)).unwrap_err();
        assert!(matches!(set_err, CliError::Data(_)), "set: {set_err:?}");

        let fill_err = fill(&fill_args(path, IVec3::ZERO, IVec3::ZERO, state("minecraft:dirt"), None, true))
            .unwrap_err();
        assert!(matches!(fill_err, CliError::Data(_)), "fill: {fill_err:?}");
    }

    // -----------------------------------------------------------------------------------------
    // ---- resize (ticket 101) ---------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn resize_args(
        file: PathBuf,
        out: PathBuf,
        pad_y_top: i32,
        pad_y_bottom: i32,
        pad_x_neg: i32,
        pad_x_pos: i32,
        pad_z_neg: i32,
        pad_z_pos: i32,
        fill: BlockState,
        force: bool,
    ) -> StructResizeArgs {
        StructResizeArgs {
            file,
            out,
            pad_y_top,
            pad_y_bottom,
            pad_x_neg,
            pad_x_pos,
            pad_z_neg,
            pad_z_pos,
            fill,
            force,
        }
    }

    /// A structure whose every position holds a distinct, coordinate-named
    /// block state — precise enough to check "everything below is identical
    /// to the source" and "every remaining block shifted by exactly N"
    /// against concrete positions, rather than just a block count.
    fn labeled_blueprint(size: IVec3) -> Blueprint {
        let mut palette = Vec::new();
        let mut blocks = Vec::new();
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    palette.push(state(&format!("minecraft:pos_{x}_{y}_{z}")));
                    blocks.push((palette.len() - 1) as u16);
                }
            }
        }
        Blueprint { size, origin: IVec3::ZERO, palette, blocks, data_version: 3953, failed_columns: 0 }
    }

    /// Every block state in `file` (which must be exactly `size`-shaped), in
    /// a fixed scan order — used to compare two resized files position by
    /// position without depending on their (independently rebuilt) palette
    /// orders.
    fn block_names(file: &PathBuf, size: IVec3) -> Vec<String> {
        let mut names = Vec::new();
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    let got = get(&get_args(file.clone(), IVec3::new(x, y, z))).expect("in bounds").state;
                    names.push(got.to_string());
                }
            }
        }
        names
    }

    /// The ticket's own "Done when": `--pad-y-top 3` on a known structure
    /// produces a file 3 taller, with the top 3 layers filled per `--fill`
    /// and everything below identical to the source.
    #[test]
    fn pad_y_top_adds_height_at_the_top_and_leaves_everything_below_identical() {
        let path = temp_path("resize-pad-y-top-source");
        write_structure_file(&path, &labeled_blueprint(IVec3::new(2, 2, 2))).expect("should write");

        let out = temp_path("resize-pad-y-top-dest");
        let fill_state = state("minecraft:glass");
        let result = resize(&resize_args(path.clone(), out.clone(), 3, 0, 0, 0, 0, 0, fill_state.clone(), false))
            .expect("valid resize");
        std::fs::remove_file(&path).ok();

        assert_eq!(result.old_size, IVec3::new(2, 2, 2));
        assert_eq!(result.new_size, IVec3::new(2, 5, 2));

        for y in 0..2 {
            for z in 0..2 {
                for x in 0..2 {
                    let got = get(&get_args(out.clone(), IVec3::new(x, y, z))).expect("in bounds").state;
                    assert_eq!(got.name, format!("minecraft:pos_{x}_{y}_{z}"), "at ({x},{y},{z})");
                }
            }
        }
        for y in 2..5 {
            for z in 0..2 {
                for x in 0..2 {
                    let got = get(&get_args(out.clone(), IVec3::new(x, y, z))).expect("in bounds").state;
                    assert_eq!(got, fill_state, "at ({x},{y},{z})");
                }
            }
        }
        std::fs::remove_file(&out).ok();
    }

    /// The ticket's other "Done when": `--pad-y-bottom 2` shifts every
    /// existing block up by 2 and fills the new bottom 2 layers.
    #[test]
    fn pad_y_bottom_shifts_existing_blocks_up_and_fills_the_new_bottom_layers() {
        let path = temp_path("resize-pad-y-bottom-source");
        write_structure_file(&path, &labeled_blueprint(IVec3::new(2, 2, 2))).expect("should write");

        let out = temp_path("resize-pad-y-bottom-dest");
        let fill_state = state("minecraft:glass");
        let result = resize(&resize_args(path.clone(), out.clone(), 0, 2, 0, 0, 0, 0, fill_state.clone(), false))
            .expect("valid resize");
        std::fs::remove_file(&path).ok();

        assert_eq!(result.new_size, IVec3::new(2, 4, 2));

        for y in 0..2 {
            for z in 0..2 {
                for x in 0..2 {
                    let got = get(&get_args(out.clone(), IVec3::new(x, y, z))).expect("in bounds").state;
                    assert_eq!(got, fill_state, "at ({x},{y},{z})");
                }
            }
        }
        for y in 0..2 {
            for z in 0..2 {
                for x in 0..2 {
                    let got = get(&get_args(out.clone(), IVec3::new(x, y + 2, z))).expect("in bounds").state;
                    assert_eq!(got.name, format!("minecraft:pos_{x}_{y}_{z}"), "at ({x},{y},{z}) shifted up by 2");
                }
            }
        }
        std::fs::remove_file(&out).ok();
    }

    /// A negative pad crops correctly, including cropping away part of a
    /// non-air region — data loss is allowed, this is a deliberate crop.
    #[test]
    fn negative_pad_crops_and_allows_dropping_non_air_data() {
        let path = temp_path("resize-crop-source");
        write_structure_file(&path, &labeled_blueprint(IVec3::new(4, 1, 1))).expect("should write");

        let out = temp_path("resize-crop-dest");
        let result = resize(&resize_args(path.clone(), out.clone(), 0, 0, 0, -2, 0, 0, BlockState::air(), false))
            .expect("valid crop");
        std::fs::remove_file(&path).ok();

        assert_eq!(result.new_size, IVec3::new(2, 1, 1));
        for x in 0..2 {
            let got = get(&get_args(out.clone(), IVec3::new(x, 0, 0))).expect("in bounds").state;
            assert_eq!(got.name, format!("minecraft:pos_{x}_0_0"), "at ({x},0,0)");
        }
        std::fs::remove_file(&out).ok();
    }

    /// The ticket's own refusal case: a crop that removes every block on
    /// some axis is a `Usage` error naming the resulting non-positive
    /// dimension, refused before writing anything.
    #[test]
    fn a_crop_that_collapses_an_axis_is_a_usage_error_and_touches_nothing() {
        let path = temp_path("resize-collapse-source");
        write_structure_file(&path, &labeled_blueprint(IVec3::new(3, 1, 1))).expect("should write");

        let out = temp_path("resize-collapse-dest");
        let err = resize(&resize_args(path.clone(), out.clone(), 0, 0, 0, -3, 0, 0, BlockState::air(), false))
            .unwrap_err();
        std::fs::remove_file(&path).ok();

        match err {
            CliError::Usage(message) => assert!(message.contains("x size would be 0"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert!(!out.exists(), "a refused resize must not write a file");
    }

    /// The ticket's own order-independence test: combining pads on multiple
    /// faces in one call produces the same result as applying them one at a
    /// time, in an order different from how the flags are listed — the six
    /// pads must not interact.
    #[test]
    fn combining_pads_in_one_call_matches_applying_them_one_at_a_time_in_a_different_order() {
        let source = temp_path("resize-order-source");
        write_structure_file(&source, &labeled_blueprint(IVec3::new(3, 3, 3))).expect("should write");
        let fill_state = state("minecraft:glass");

        // All six pads in one call: +2 top, +1 bottom, +1 x-neg, -1 x-pos
        // (crop), -1 z-neg (crop), +2 z-pos.
        let combined = temp_path("resize-order-combined");
        let combined_result = resize(&resize_args(
            source.clone(),
            combined.clone(),
            2,
            1,
            1,
            -1,
            -1,
            2,
            fill_state.clone(),
            false,
        ))
        .expect("combined resize");

        // The same six pads, applied one at a time, deliberately in a
        // different order (z-pos, x-neg, y-bottom, x-pos, z-neg, y-top).
        let step1 = temp_path("resize-order-step1");
        resize(&resize_args(source.clone(), step1.clone(), 0, 0, 0, 0, 0, 2, fill_state.clone(), false))
            .expect("step 1: pad-z-pos");
        let step2 = temp_path("resize-order-step2");
        resize(&resize_args(step1.clone(), step2.clone(), 0, 0, 1, 0, 0, 0, fill_state.clone(), false))
            .expect("step 2: pad-x-neg");
        let step3 = temp_path("resize-order-step3");
        resize(&resize_args(step2.clone(), step3.clone(), 0, 1, 0, 0, 0, 0, fill_state.clone(), false))
            .expect("step 3: pad-y-bottom");
        let step4 = temp_path("resize-order-step4");
        resize(&resize_args(step3.clone(), step4.clone(), 0, 0, 0, -1, 0, 0, fill_state.clone(), false))
            .expect("step 4: pad-x-pos crop");
        let step5 = temp_path("resize-order-step5");
        resize(&resize_args(step4.clone(), step5.clone(), 0, 0, 0, 0, -1, 0, fill_state.clone(), false))
            .expect("step 5: pad-z-neg crop");
        let sequential = temp_path("resize-order-sequential");
        let sequential_result = resize(&resize_args(
            step5.clone(),
            sequential.clone(),
            2,
            0,
            0,
            0,
            0,
            0,
            fill_state.clone(),
            false,
        ))
        .expect("step 6: pad-y-top");

        for path in [&source, &step1, &step2, &step3, &step4, &step5] {
            std::fs::remove_file(path).ok();
        }

        assert_eq!(combined_result.new_size, sequential_result.new_size, "the two paths must agree on the resulting size");
        let size = combined_result.new_size;

        let combined_names = block_names(&combined, size);
        let sequential_names = block_names(&sequential, size);
        std::fs::remove_file(&combined).ok();
        std::fs::remove_file(&sequential).ok();

        assert_eq!(combined_names, sequential_names, "the six pads must not interact");
    }

    /// `struct resize --out` refuses to overwrite an existing file without
    /// `--force`, and touches nothing — same convention every other
    /// file-writing `struct` command follows.
    #[test]
    fn resize_onto_an_existing_out_without_force_refuses() {
        let path = temp_path("resize-no-force-source");
        write_structure_file(&path, &labeled_blueprint(IVec3::new(2, 2, 2))).expect("should write");

        let out = temp_path("resize-no-force-dest");
        std::fs::write(&out, b"not a structure file").expect("seed an existing file");

        let err = resize(&resize_args(path.clone(), out.clone(), 1, 0, 0, 0, 0, 0, BlockState::air(), false))
            .unwrap_err();

        match err {
            CliError::Usage(message) => assert!(message.contains("--force"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(
            std::fs::read(&out).expect("still readable"),
            b"not a structure file",
            "must not have overwritten"
        );
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&out).ok();
    }

    /// `struct resize` on a missing file is a `Data` error (exit 1), same as
    /// every other `struct` command's own missing-file test.
    #[test]
    fn resize_on_a_missing_file_is_a_data_error() {
        let path = temp_path("resize-missing");
        let out = temp_path("resize-missing-out");
        let err = resize(&resize_args(path, out, 1, 0, 0, 0, 0, 0, BlockState::air(), false)).unwrap_err();
        assert!(matches!(err, CliError::Data(_)), "{err:?}");
    }

    // -----------------------------------------------------------------------------------------
    // ---- rotate / diff (ticket 102) --------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    fn rotate_args(file: PathBuf, by: RotateArg, out: PathBuf, force: bool) -> StructRotateArgs {
        StructRotateArgs { file, by, out, force }
    }

    fn diff_args(a: PathBuf, b: PathBuf, limit: Option<usize>) -> StructDiffArgs {
        StructDiffArgs { a, b, limit }
    }

    /// A blueprint carrying a property `blueprint::rotate`'s table has no
    /// rewrite rule for — the same fixture that module's own tests (and
    /// `city::placement`'s) use to exercise `UnrotatableProperty`.
    fn unrotatable_blueprint() -> Blueprint {
        let unrotatable = BlockState {
            name: "minecraft:made_up_block".to_string(),
            properties: vec![("orientation".to_string(), "north_up".to_string())],
        };
        Blueprint {
            size: IVec3::ONE,
            origin: IVec3::ZERO,
            palette: vec![unrotatable],
            blocks: vec![0],
            data_version: 3953,
            failed_columns: 0,
        }
    }

    /// The ticket's own "Done when": rotating 180° twice reproduces the
    /// original — verified with `struct diff` reporting zero differences,
    /// not just an eyeballed size match.
    #[test]
    fn rotating_180_twice_reproduces_the_original_per_struct_diff() {
        let source = temp_path("rotate-180-twice-source");
        write_structure_file(&source, &labeled_blueprint(IVec3::new(2, 1, 3))).expect("should write");

        let once = temp_path("rotate-180-twice-once");
        rotate(&rotate_args(source.clone(), RotateArg::Deg180, once.clone(), false)).expect("first rotation");
        let twice = temp_path("rotate-180-twice-twice");
        let result = rotate(&rotate_args(once.clone(), RotateArg::Deg180, twice.clone(), false))
            .expect("second rotation");
        std::fs::remove_file(&once).ok();

        assert_eq!(result.new_size, IVec3::new(2, 1, 3), "180 degrees twice must restore the original size");

        let diff_result = diff(&diff_args(source.clone(), twice.clone(), None)).expect("should diff");
        std::fs::remove_file(&source).ok();
        std::fs::remove_file(&twice).ok();

        assert_eq!(diff_result.differences.len(), 0, "{:?}", diff_result.differences);
    }

    /// `struct rotate` on a blueprint carrying a property the rotation table
    /// doesn't know is `CliError::Data` carrying that error's own message
    /// (naming the block, property key and value), not a generic failure —
    /// and no `--out` file is written.
    #[test]
    fn rotate_on_an_unrotatable_property_carries_the_errors_own_message() {
        let source = temp_path("rotate-unrotatable-source");
        write_structure_file(&source, &unrotatable_blueprint()).expect("should write");

        let out = temp_path("rotate-unrotatable-out");
        let err = rotate(&rotate_args(source.clone(), RotateArg::Deg90, out.clone(), false)).unwrap_err();
        std::fs::remove_file(&source).ok();

        match err {
            CliError::Data(message) => {
                assert!(message.contains("minecraft:made_up_block"), "{message}");
                assert!(message.contains("orientation"), "{message}");
                assert!(message.contains("north_up"), "{message}");
            }
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }
        assert!(!out.exists(), "a refused rotation must not write a file");
    }

    /// `struct rotate --out` refuses to overwrite an existing file without
    /// `--force`, same convention every other file-writing `struct` command
    /// follows.
    #[test]
    fn rotate_onto_an_existing_out_without_force_refuses() {
        let source = temp_path("rotate-no-force-source");
        write_structure_file(&source, &labeled_blueprint(IVec3::new(2, 1, 2))).expect("should write");

        let out = temp_path("rotate-no-force-out");
        std::fs::write(&out, b"not a structure file").expect("seed an existing file");

        let err = rotate(&rotate_args(source.clone(), RotateArg::Deg90, out.clone(), false)).unwrap_err();
        std::fs::remove_file(&source).ok();

        match err {
            CliError::Usage(message) => assert!(message.contains("--force"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(
            std::fs::read(&out).expect("still readable"),
            b"not a structure file",
            "must not have overwritten"
        );
        std::fs::remove_file(&out).ok();
    }

    /// `struct rotate` on a missing file is a `Data` error (exit 1), same as
    /// every other `struct` command's own missing-file test.
    #[test]
    fn rotate_on_a_missing_file_is_a_data_error() {
        let path = temp_path("rotate-missing");
        let out = temp_path("rotate-missing-out");
        let err = rotate(&rotate_args(path, RotateArg::Deg90, out, false)).unwrap_err();
        assert!(matches!(err, CliError::Data(_)), "{err:?}");
    }

    /// The ticket's own "Done when": a copy of a file with exactly one block
    /// changed reports exactly one difference, with the correct position and
    /// before/after values.
    #[test]
    fn diff_between_a_file_and_a_one_block_edit_reports_exactly_one_difference() {
        let a = temp_path("diff-one-block-a");
        write_structure_file(&a, &labeled_blueprint(IVec3::new(2, 1, 2))).expect("should write a");

        let b = temp_path("diff-one-block-b");
        let changed_pos = IVec3::new(1, 0, 0);
        set(&set_args(a.clone(), changed_pos, state("minecraft:glass"), Some(b.clone()), false))
            .expect("should write b as an edited copy");

        let result = diff(&diff_args(a.clone(), b.clone(), None)).expect("should diff");
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();

        assert_eq!(result.differences.len(), 1, "{:?}", result.differences);
        let entry = &result.differences[0];
        assert_eq!(entry.pos, changed_pos);
        assert_eq!(entry.a.name, "minecraft:pos_1_0_0");
        assert_eq!(entry.b.name, "minecraft:glass");
    }

    /// Two structure files with no differences at all report zero, not a
    /// missing/empty-list ambiguity.
    #[test]
    fn diff_between_identical_files_reports_zero_differences() {
        let a = temp_path("diff-identical-a");
        write_structure_file(&a, &labeled_blueprint(IVec3::new(2, 2, 2))).expect("should write a");
        let b = temp_path("diff-identical-b");
        write_structure_file(&b, &labeled_blueprint(IVec3::new(2, 2, 2))).expect("should write b");

        let result = diff(&diff_args(a.clone(), b.clone(), None)).expect("should diff");
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();

        assert_eq!(result.differences.len(), 0);
        assert!(!result.render_text().contains("more"));
    }

    /// The ticket's own refusal case: differently-sized files exit 2 rather
    /// than attempting a partial comparison.
    #[test]
    fn diff_between_differently_sized_files_is_a_usage_error() {
        let a = temp_path("diff-size-mismatch-a");
        write_structure_file(&a, &labeled_blueprint(IVec3::new(2, 1, 1))).expect("should write a");
        let b = temp_path("diff-size-mismatch-b");
        write_structure_file(&b, &labeled_blueprint(IVec3::new(3, 1, 1))).expect("should write b");

        let err = diff(&diff_args(a.clone(), b.clone(), None)).unwrap_err();
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();

        match err {
            CliError::Usage(message) => {
                assert!(message.contains("2x1x1"), "{message}");
                assert!(message.contains("3x1x1"), "{message}");
            }
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    /// `struct diff` on a missing file is a `Data` error (exit 1), same as
    /// every other `struct` command's own missing-file test.
    #[test]
    fn diff_on_a_missing_file_is_a_data_error() {
        let a = temp_path("diff-missing-a");
        let b = temp_path("diff-missing-b");
        write_structure_file(&b, &labeled_blueprint(IVec3::ONE)).expect("should write b");

        let err = diff(&diff_args(a, b.clone(), None)).unwrap_err();
        std::fs::remove_file(&b).ok();
        assert!(matches!(err, CliError::Data(_)), "{err:?}");
    }

    /// `text`/`compact` cap their listing at `--limit`, but `render_json`
    /// always reports the full, uncapped list — the ticket's own split
    /// between the two.
    #[test]
    fn text_and_compact_cap_the_listing_but_json_stays_full() {
        let a = temp_path("diff-limit-a");
        write_structure_file(&a, &labeled_blueprint(IVec3::new(4, 1, 1))).expect("should write a");
        let b = temp_path("diff-limit-b");
        set(&set_args(a.clone(), IVec3::new(0, 0, 0), state("minecraft:changed_0"), Some(b.clone()), false))
            .expect("should seed b as an edited copy");
        for x in 1..4 {
            set(&set_args(b.clone(), IVec3::new(x, 0, 0), state(&format!("minecraft:changed_{x}")), None, true))
                .expect("should edit b in place");
        }

        let result = diff(&diff_args(a.clone(), b.clone(), Some(2))).expect("should diff");
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();

        assert_eq!(result.differences.len(), 4, "the full list is always collected");

        let text = result.render_text();
        assert!(text.contains("4 differences"), "{text}");
        assert!(text.contains("... and 2 more"), "{text}");

        let compact = result.render_compact();
        assert!(compact.contains("... and 2 more"), "{compact}");

        let json = result.render_json();
        assert_eq!(json["count"], 4);
        assert_eq!(json["differences"].as_array().expect("array").len(), 4, "json is never capped by --limit");
    }

    // -----------------------------------------------------------------------------------------
    // ---- validate (ticket 103) ---------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    fn validate_args(file: PathBuf, max_size: Option<IVec3>) -> StructValidateArgs {
        StructValidateArgs { file, max_size: max_size.map(BlockPos) }
    }

    /// An all-air structure — what a failed/empty extraction looks like, per
    /// the `non_air` check's own docs. Mirrors `catalogue.rs`'s own
    /// `air_only` test fixture.
    fn air_only(size: IVec3) -> Blueprint {
        let volume = (size.x * size.y * size.z) as usize;
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air()],
            blocks: vec![0; volume],
            data_version: 3953,
            failed_columns: 0,
        }
    }

    /// A structure with exactly one non-air block — enough to pass the
    /// `non_air` check on its own, so a test built on this fixture is only
    /// ever exercising the `size_limit` check.
    fn one_stone(size: IVec3) -> Blueprint {
        let volume = (size.x * size.y * size.z) as usize;
        let mut blocks = vec![0u16; volume];
        blocks[0] = 1;
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state("minecraft:stone")],
            blocks,
            data_version: 3953,
            failed_columns: 0,
        }
    }

    fn check<'a>(result: &'a StructValidateResult, name: &str) -> &'a BlueprintCheck {
        result
            .checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("no {name:?} check in {:?}", result.checks))
    }

    /// The ticket's own "Done when": a real shipped blueprint passes every
    /// check.
    #[test]
    fn validate_against_the_real_house01_fixture_passes_every_check() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/city/blueprints/house01.nbt");
        let result = validate(&validate_args(path, None)).expect("house01.nbt should parse");
        assert!(result.all_passed(), "{:?}", result.checks);
        assert!(result.checks.iter().all(|check| check.pass), "{:?}", result.checks);
    }

    /// The ticket's own "Done when": an all-air structure (`struct new`'s
    /// default fill) fails the `non_air` check specifically, leaving
    /// `size_limit` unaffected.
    #[test]
    fn an_all_air_structure_fails_non_air_only() {
        let path = temp_path("validate-all-air");
        write_structure_file(&path, &air_only(IVec3::new(4, 4, 4))).expect("should write");

        let result = validate(&validate_args(path.clone(), None)).expect("should parse");
        std::fs::remove_file(&path).ok();

        assert!(!result.all_passed());
        assert!(!check(&result, "non_air").pass);
        assert!(check(&result, "size_limit").pass);
    }

    /// The ticket's own "Done when": a structure built past the catalogue's
    /// real size limit fails `size_limit`, with the actual and allowed sizes
    /// both in the message. `write_structure_file` (unlike `struct new`,
    /// which refuses outright — see the module docs) only warns on an
    /// oversized blueprint, so writing one directly is how this fixture
    /// exists at all.
    #[test]
    fn an_oversized_structure_fails_size_limit_with_actual_and_allowed_sizes_in_the_message() {
        let side = STRUCTURE_BLOCK_MAX_SIZE + 1;
        let path = temp_path("validate-oversized");
        write_structure_file(&path, &one_stone(IVec3::splat(side))).expect("should write");

        let result = validate(&validate_args(path.clone(), None)).expect("should parse");
        std::fs::remove_file(&path).ok();

        assert!(!result.all_passed());
        let size_check = check(&result, "size_limit");
        assert!(!size_check.pass);
        assert!(size_check.detail.contains(&side.to_string()), "{}", size_check.detail);
        assert!(
            size_check.detail.contains(&STRUCTURE_BLOCK_MAX_SIZE.to_string()),
            "{}",
            size_check.detail
        );
        assert!(check(&result, "non_air").pass);
    }

    /// `--max-size` overrides the default cap — a structure well inside
    /// [`STRUCTURE_BLOCK_MAX_SIZE`] still fails `size_limit` when validated
    /// against a smaller ceiling, and the reported `max_size` reflects the
    /// override rather than the default.
    #[test]
    fn max_size_overrides_the_default_cap() {
        let path = temp_path("validate-max-size-override");
        write_structure_file(&path, &one_stone(IVec3::new(4, 4, 4))).expect("should write");

        let result = validate(&validate_args(path.clone(), Some(IVec3::new(2, 2, 2))))
            .expect("should parse");
        std::fs::remove_file(&path).ok();

        assert!(!result.all_passed());
        assert!(!check(&result, "size_limit").pass);
        assert_eq!(result.max_size, IVec3::new(2, 2, 2));
    }

    /// A file that doesn't parse at all is a `Usage` error (exit `2`), not a
    /// `Data` error (exit `1`, every other `struct` command's convention) —
    /// per the ticket, there's no file to report per-check results about.
    #[test]
    fn a_file_that_does_not_parse_is_a_usage_error() {
        let path = temp_path("validate-missing");
        let err = validate(&validate_args(path, None)).unwrap_err();
        match err {
            CliError::Usage(_) => {}
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    /// `render_json`'s shape: `"pass"` mirrors [`StructValidateResult::all_passed`],
    /// and `"checks"` names every check individually, per the ticket's own
    /// spelled-out example.
    #[test]
    fn render_json_reports_pass_and_every_named_check() {
        let path = temp_path("validate-json-shape");
        write_structure_file(&path, &air_only(IVec3::new(2, 2, 2))).expect("should write");

        let result = validate(&validate_args(path.clone(), None)).expect("should parse");
        std::fs::remove_file(&path).ok();

        let json = result.render_json();
        assert_eq!(json["pass"], false);
        let checks = json["checks"].as_array().expect("array");
        assert!(checks.iter().any(|c| c["name"] == "size_limit" && c["pass"] == true));
        assert!(checks.iter().any(|c| c["name"] == "non_air" && c["pass"] == false));
    }
}
