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

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use serde_json::{json, Value};

use crate::blueprint::{
    extract_blueprint, read_structure_file, rotate_blueprint, write_structure_file, BlockState,
    Blueprint, ExtractProgress, FALLBACK_DATA_VERSION, LOGGED_PALETTE_ENTRIES, MAX_BLOCKS,
    STRUCTURE_BLOCK_MAX_SIZE,
};
use crate::edit::WorldEdit;
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;
use crate::world::SECTION_SIZE;

use super::block::position_of;
use super::chunk::region_span;
use super::cli::{
    Cli, RotateArg, StructExportArgs, StructFillArgs, StructGetArgs, StructImportArgs,
    StructInfoArgs, StructNewArgs, StructSetArgs,
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
}
