//! `run_write` (ticket 094): the write substrate every `ranvil-cli` write
//! command (095–097) and `struct import` (099) is built on — lock, backup,
//! dry-run, force.
//!
//! [`crate::edit::session::WriteSession`] already does the right things
//! (hold `session.lock` for the session, back up every touched region before
//! any is saved, name exactly what did and didn't write on a mid-write
//! failure), but it's shaped around the game's flow: one session opened
//! around a manual Save action, edits accumulated over a play session,
//! flushed once. A `ranvil-cli` invocation is a single edit (or batch) start
//! to finish in one process, so [`run_write`] opens a session, builds one
//! edit, plans or commits it, and lets the session drop — all inside one
//! call.
//!
//! Distinct from the crate-root [`crate::edit`] module (the viewer/
//! citybuilder write path this wraps) — same name, different module path,
//! no collision.

use std::io::Read as _;
use std::path::PathBuf;

use bevy::math::IVec3;
use mc_anvil::SaveMeta;
use serde_json::{json, Value};

use crate::blueprint::{BlockState, MAX_BLOCKS};
use crate::edit::{EditPolicy, EditReport, WorldEdit, WriteSession};
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;

use super::block::{get_area, matching_positions, position_of};
use super::cli::{Cli, CopyArgs, GetAreaArgs, ReplaceArgs, SetAreaArgs, SetArgs, SetBatchArgs};
use super::coords::BlockPos;
use super::error::CliError;
use super::format::Render;
use super::save::resolve_save;

/// The [`RegionCache`] capacity a one-shot CLI write gets.
///
/// Correctness never depends on this: [`RegionCache::evict_lru`] skips
/// dirty regions, so a cache too small for what an edit touches just
/// reloads a clean region instead of losing anything. Four is the most a
/// single edit can span (a building placed on a region corner, per
/// [`crate::edit::route`]); doubling that leaves room for a `set-batch`
/// or `replace` that fans out a little further before it starts reloading.
const WRITE_CACHE_CAPACITY: usize = 8;

/// What [`run_write`] hands back on success: [`edit::plan`](crate::edit::plan)/
/// [`apply`](crate::edit::apply)'s own [`EditReport`] (blocks written,
/// chunks, regions touched), plus what only a [`WriteSession`] knows —
/// whether this was `--dry-run`, and where backups went (or would go).
#[derive(Debug, Clone)]
pub struct WriteOutcome {
    pub report: EditReport,
    /// `true` for `--dry-run` — nothing else in this struct describes
    /// something that actually happened on disk.
    pub dry_run: bool,
    /// This session's backup directory. Only exists on disk once a real
    /// write has backed something up into it; for `--dry-run` it names
    /// where a real run's backups *would* go.
    pub backup_dir: PathBuf,
    /// Region files actually saved this call. Empty for `--dry-run`.
    pub regions_written: Vec<(i32, i32)>,
    /// Backups taken by this call. Empty for `--dry-run`, and empty on a
    /// real write too if every touched region already had one from earlier
    /// in this same session.
    pub backups: Vec<PathBuf>,
}

/// The write substrate every `ranvil-cli` write command shares (ticket 094):
/// resolves the write policy, holds `session.lock` for the duration of the
/// call, plans and — unless `dry_run` — applies, backs up and saves.
///
/// `build` gets a scratch [`RegionCache`] scoped to this one call to read
/// from while constructing its [`WorldEdit`] (e.g. `replace` needs to read
/// the blocks it's about to overwrite).
///
/// The policy is fixed here rather than taken as a parameter: `Status` stays
/// gated to `minecraft:full` chunks only, `DataVersion` compatibility stays
/// enforced, and heightmaps are handled the same way every other policy in
/// this crate leaves them by default — the same rules the game itself
/// writes under, so a `ranvil-cli set` into an ungenerated chunk refuses
/// exactly like the game does rather than carving out a new exception.
///
/// # `force`
///
/// Without it, a save [`SaveMeta::is_locked`] currently reports as open in
/// Minecraft refuses before anything is touched — [`CliError::Data`], per
/// the roadmap's write-command contract. With it, that pre-check is skipped
/// and the real `session.lock` acquisition is attempted regardless:
/// `--force` overrides the friendly probe, not the acquisition itself, so a
/// save that is genuinely open still refuses — the OS lock can't be forced.
///
/// # `dry_run`
///
/// Plans the edit and reports what it *would* do — regions, chunks, block
/// count, or whatever refusal [`edit::plan`](crate::edit::plan) would raise
/// — without applying, backing up, or saving anything.
pub fn run_write(
    save: &SaveMeta,
    dry_run: bool,
    force: bool,
    build: impl FnOnce(&mut RegionCache) -> Result<WorldEdit, CliError>,
) -> Result<WriteOutcome, CliError> {
    if !force {
        let locked = save.is_locked().map_err(|err| {
            CliError::Data(format!("could not check lock state for {}: {err}", save.name))
        })?;
        if locked {
            return Err(CliError::Data(format!(
                "\"{}\" is open in Minecraft — pass --force to write anyway",
                save.name
            )));
        }
    }

    let mut session = WriteSession::open(save).map_err(|err| CliError::Data(err.to_string()))?;
    let mut cache = RegionCache::new(save.clone(), WRITE_CACHE_CAPACITY);
    let edit = build(&mut cache)?;
    let policy = EditPolicy::default();

    if dry_run {
        let plan = session
            .plan(&edit, &mut cache, &policy)
            .map_err(|refusal| CliError::Data(refusal.to_string()))?;

        return Ok(WriteOutcome {
            report: plan.report,
            dry_run: true,
            backup_dir: session.backup_dir().to_path_buf(),
            regions_written: Vec::new(),
            backups: Vec::new(),
        });
    }

    let summary = session
        .commit(&edit, &mut cache, &policy)
        .map_err(|err| CliError::Data(err.to_string()))?;

    Ok(WriteOutcome {
        report: summary.report,
        dry_run: false,
        backup_dir: session.backup_dir().to_path_buf(),
        regions_written: summary.regions_written,
        backups: summary.backups,
    })
}

// -------------------------------------------------------------------------------------------------
// ---- set / set-area (ticket 095) -----------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// The [`WriteOutcome`] fields every write command's `render_json` shares —
/// `status` first and unconditionally, so a `--dry-run`'s JSON reads
/// unmistakably as a plan rather than a completed write even on a skim, per
/// the ticket ("a script can't mistake a dry run's JSON for a real one").
fn outcome_json_fields(outcome: &WriteOutcome) -> Vec<(&'static str, Value)> {
    let region_json = |coords: &[(i32, i32)]| {
        json!(coords.iter().map(|&(x, z)| json!([x, z])).collect::<Vec<_>>())
    };

    vec![
        ("status", json!(if outcome.dry_run { "dry-run" } else { "applied" })),
        ("blocks_written", json!(outcome.report.blocks_written)),
        ("chunks", region_json(&outcome.report.chunks)),
        ("regions", region_json(&outcome.report.regions)),
        ("regions_written", region_json(&outcome.regions_written)),
        (
            "backups",
            json!(outcome.backups.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()),
        ),
        ("backup_dir", json!(outcome.backup_dir.display().to_string())),
    ]
}

/// The one summary line every write command's `text`/`compact` rendering
/// shares: `[dry-run]` up front when nothing was actually written — visible
/// in *every* format, not just `json`, per the ticket.
fn outcome_summary(prefix: String, outcome: &WriteOutcome) -> String {
    let verb = if outcome.dry_run { "would write" } else { "wrote" };
    let blocks = outcome.report.blocks_written;
    let regions = outcome.report.regions.len();
    let label = if outcome.dry_run { "[dry-run] " } else { "" };
    format!(
        "{label}{prefix}: {verb} {blocks} block{} across {regions} region{}",
        if blocks == 1 { "" } else { "s" },
        if regions == 1 { "" } else { "s" },
    )
}

/// `set`'s result: the position and block written, plus what [`run_write`]
/// did (or would do).
#[derive(Debug)]
pub struct SetResult {
    pub save_name: String,
    pub pos: IVec3,
    pub state: BlockState,
    pub outcome: WriteOutcome,
}

/// Runs `set`: one [`WorldEdit::set`] handed to [`run_write`].
pub fn set(cli: &Cli, args: &SetArgs) -> Result<SetResult, CliError> {
    let meta = resolve_save(cli)?;
    let pos = args.pos.0;
    let state = args.state.clone();

    let outcome = run_write(&meta, args.dry_run, args.force, |_cache| {
        let mut edit = WorldEdit::new();
        edit.set(pos, state.clone());
        Ok(edit)
    })?;

    Ok(SetResult {
        save_name: meta.name,
        pos,
        state: args.state.clone(),
        outcome,
    })
}

impl Render for SetResult {
    fn render_text(&self) -> String {
        let prefix = format!("set {} = {} in {}", self.pos, self.state, self.save_name);
        let mut lines = vec![outcome_summary(prefix, &self.outcome)];
        if !self.outcome.dry_run {
            lines.push(format!("  backup: {}", self.outcome.backup_dir.display()));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("save".to_string(), json!(self.save_name));
        map.insert("pos".to_string(), json!([self.pos.x, self.pos.y, self.pos.z]));
        map.insert("block".to_string(), json!(self.state.to_string()));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

/// `set-area`'s result: the box and block filled, plus what [`run_write`]
/// did (or would do).
#[derive(Debug)]
pub struct SetAreaResult {
    pub save_name: String,
    pub from: IVec3,
    pub to: IVec3,
    pub state: BlockState,
    pub outcome: WriteOutcome,
}

/// Runs `set-area`: [`WorldEdit::fill`] over `args.from`/`args.to` (either
/// corner in either order — `from_corners` normalizes, same as
/// [`super::block::get_area`]), handed to [`run_write`]. The box is capped
/// at [`MAX_BLOCKS`], the same limit `get-area` enforces, before
/// [`resolve_save`] even runs — a selection over the cap is a bad request
/// regardless of which save it names.
pub fn set_area(cli: &Cli, args: &SetAreaArgs) -> Result<SetAreaResult, CliError> {
    let bounds = SelectionBounds::from_corners(args.from.0, args.from.0, args.to.0);

    let volume = bounds.volume();
    if volume > MAX_BLOCKS {
        return Err(CliError::Usage(format!(
            "selection ({}) to ({}) is {volume} blocks — over the {MAX_BLOCKS}-block set-area limit",
            args.from.0, args.to.0
        )));
    }

    let meta = resolve_save(cli)?;
    let state = args.state.clone();

    let outcome = run_write(&meta, args.dry_run, args.force, move |_cache| {
        Ok(WorldEdit::fill(bounds, state))
    })?;

    Ok(SetAreaResult {
        save_name: meta.name,
        from: bounds.min,
        to: bounds.max,
        state: args.state.clone(),
        outcome,
    })
}

impl Render for SetAreaResult {
    fn render_text(&self) -> String {
        let prefix = format!(
            "set-area {} to {} = {} in {}",
            self.from, self.to, self.state, self.save_name
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
        map.insert("from".to_string(), json!([self.from.x, self.from.y, self.from.z]));
        map.insert("to".to_string(), json!([self.to.x, self.to.y, self.to.z]));
        map.insert("block".to_string(), json!(self.state.to_string()));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

// -------------------------------------------------------------------------------------------------
// ---- set-batch / replace (ticket 096) --------------------------------------------------------
// -------------------------------------------------------------------------------------------------
//
// Both build on 094's `run_write` the same way `set`/`set-area` do, so they
// get the same all-or-nothing guarantee across the whole batch/box rather
// than per line or per position: one `WorldEdit`, one `run_write` call.

/// Reads `set-batch`'s input: the file at `path`, or stdin when `path` is
/// `-` — the shape an agent's own generated diff arrives in without needing
/// a temp file.
fn read_batch_input(path: &str) -> Result<String, CliError> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| CliError::Usage(format!("could not read stdin: {e}")))?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| CliError::Usage(format!("could not read \"{path}\": {e}")))
    }
}

/// Parses `set-batch`'s line format: `x,y,z blockstate`, one edit per line.
/// Blank lines and `#`-prefixed lines are skipped; every other line is
/// `<position> <blockstate>`, split on the first run of whitespace so a
/// blockstate's own `[key=value,...]` never gets mistaken for a second
/// column.
///
/// Every line is parsed before any edit is attempted — the first malformed
/// line anywhere in `input` aborts the whole batch with its 1-based line
/// number (comments and blanks counted, so the number always points at the
/// line the caller sees in their own file), matching `edit::route`'s "plan
/// everything before applying anything" discipline. Duplicate positions are
/// not rejected here: last write wins, the same as calling `WorldEdit::set`
/// twice on one `WorldEdit` already behaves.
fn parse_batch(input: &str) -> Result<Vec<(IVec3, BlockState)>, CliError> {
    let mut edits = Vec::new();

    for (index, raw_line) in input.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.splitn(2, char::is_whitespace);
        let (pos_part, state_part) = match (parts.next(), parts.next()) {
            (Some(pos), Some(state)) if !state.trim().is_empty() => (pos, state.trim()),
            _ => {
                return Err(CliError::Usage(format!(
                    "line {line_number}: expected \"x,y,z blockstate\", got \"{line}\""
                )));
            }
        };

        let pos: BlockPos = pos_part
            .parse()
            .map_err(|e| CliError::Usage(format!("line {line_number}: {e}")))?;
        let state: BlockState = state_part
            .parse()
            .map_err(|e| CliError::Usage(format!("line {line_number}: {e}")))?;

        edits.push((pos.0, state));
    }

    Ok(edits)
}

/// `set-batch`'s result: how many lines were applied, plus what
/// [`run_write`] did (or would do).
#[derive(Debug)]
pub struct SetBatchResult {
    pub save_name: String,
    pub file: String,
    /// Lines successfully parsed into an edit — a raw line count, not the
    /// count of distinct positions written (a batch with a correction for
    /// one position parses as two lines but writes one block).
    pub lines_applied: usize,
    pub outcome: WriteOutcome,
}

/// Runs `set-batch`: reads and [`parse_batch`]es every line of `args.file`
/// (or stdin) into one [`WorldEdit`] before touching a save, then hands it
/// to [`run_write`] — so a malformed line anywhere refuses the whole batch
/// (nothing written) rather than applying a prefix of it.
pub fn set_batch(cli: &Cli, args: &SetBatchArgs) -> Result<SetBatchResult, CliError> {
    let input = read_batch_input(&args.file)?;
    let edits = parse_batch(&input)?;
    if edits.is_empty() {
        return Err(CliError::Usage(format!(
            "\"{}\" has no edits — every line is blank or a comment",
            args.file
        )));
    }
    let lines_applied = edits.len();

    let meta = resolve_save(cli)?;

    let outcome = run_write(&meta, args.dry_run, args.force, move |_cache| {
        let mut edit = WorldEdit::new();
        for (pos, state) in edits {
            edit.set(pos, state);
        }
        Ok(edit)
    })?;

    Ok(SetBatchResult {
        save_name: meta.name,
        file: args.file.clone(),
        lines_applied,
        outcome,
    })
}

impl Render for SetBatchResult {
    fn render_text(&self) -> String {
        let prefix = format!(
            "set-batch {} ({} line{}) in {}",
            self.file,
            self.lines_applied,
            if self.lines_applied == 1 { "" } else { "s" },
            self.save_name
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
        map.insert("file".to_string(), json!(self.file));
        map.insert("lines_applied".to_string(), json!(self.lines_applied));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

/// `replace`'s result: the box, the match/replacement blocks, how many
/// positions matched, plus what [`run_write`] did (or would do).
#[derive(Debug)]
pub struct ReplaceResult {
    pub save_name: String,
    pub from: IVec3,
    pub to: IVec3,
    pub from_block: String,
    pub to_state: BlockState,
    /// Positions in the box whose block matched `from_block` — distinct from
    /// `set-area`'s "every position in the box" count, and (barring a
    /// scan/write race on a live save) equal to `outcome.report.blocks_written`.
    pub matched: usize,
    pub outcome: WriteOutcome,
}

/// Runs `replace`: extracts the box via [`get_area`] (092's primitive),
/// filters it down to positions whose block matches `args.from_block` via
/// [`matching_positions`] (the same by-name-only rule `scan` uses), and
/// writes `args.to_state` at each as one [`WorldEdit`] handed to
/// [`run_write`].
///
/// The box is capped at [`MAX_BLOCKS`] before [`resolve_save`] even runs,
/// same as `set-area`/`get-area`. A box with no matches reaches `run_write`
/// with an empty `WorldEdit`, which surfaces the same
/// [`crate::edit::EditRefusal::Empty`] refusal any other empty edit does —
/// not a special-cased no-op, since "nothing matched" and "nothing to write"
/// are the same fact here.
pub fn replace(cli: &Cli, args: &ReplaceArgs) -> Result<ReplaceResult, CliError> {
    let bounds = SelectionBounds::from_corners(args.corner1.0, args.corner1.0, args.corner2.0);

    let volume = bounds.volume();
    if volume > MAX_BLOCKS {
        return Err(CliError::Usage(format!(
            "selection ({}) to ({}) is {volume} blocks — over the {MAX_BLOCKS}-block replace limit",
            args.corner1.0, args.corner2.0
        )));
    }

    let area = get_area(cli, &GetAreaArgs { from: args.corner1, to: args.corner2 })?;
    let (positions, _truncated) = matching_positions(&area, &args.from_block, usize::MAX);
    let matched = positions.len();
    let to_state = args.to_state.clone();

    let meta = resolve_save(cli)?;
    let outcome = run_write(&meta, args.dry_run, args.force, move |_cache| {
        let mut edit = WorldEdit::new();
        for pos in positions {
            edit.set(pos, to_state.clone());
        }
        Ok(edit)
    })?;

    Ok(ReplaceResult {
        save_name: meta.name,
        from: bounds.min,
        to: bounds.max,
        from_block: args.from_block.clone(),
        to_state: args.to_state.clone(),
        matched,
        outcome,
    })
}

impl Render for ReplaceResult {
    fn render_text(&self) -> String {
        let prefix = format!(
            "replace {} with {} in {} to {} in {}: {} matched",
            self.from_block, self.to_state, self.from, self.to, self.save_name, self.matched
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
        map.insert("from".to_string(), json!([self.from.x, self.from.y, self.from.z]));
        map.insert("to".to_string(), json!([self.to.x, self.to.y, self.to.z]));
        map.insert("from_block".to_string(), json!(self.from_block));
        map.insert("to_block".to_string(), json!(self.to_state.to_string()));
        map.insert("matched".to_string(), json!(self.matched));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

// -------------------------------------------------------------------------------------------------
// ---- copy (ticket 097) -----------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
//
// `copy` composes 092's read primitive ([`get_area`]) with 094's write
// substrate ([`run_write`]) rather than adding a second box-walking loop.
// The whole source box is read into `area` — via `get_area`'s own
// `Arc<Mutex<RegionCache>>`, unrelated to the one `run_write` opens — before
// `run_write` is even called, which is what makes an overlapping
// source/destination safe: the read finishes in full before a session or a
// write-side cache is opened, so an overlap always sees the *original*
// blocks, never a partially-copied one.

/// `copy`'s result: the source box, the destination's min corner, whether
/// air was included, how many positions were written, plus what
/// [`run_write`] did (or would do).
#[derive(Debug)]
pub struct CopyResult {
    pub save_name: String,
    pub from: IVec3,
    pub to: IVec3,
    pub dest: IVec3,
    pub include_air: bool,
    /// Positions written: every source position with `--include-air`, or
    /// only the non-air ones without it.
    pub copied: usize,
    pub outcome: WriteOutcome,
}

/// Runs `copy`: [`get_area`]'s extraction over `args.corner1`/`args.corner2`
/// (finished in full before [`run_write`] is even called — see the module
/// note above), translated so the source box's min corner lands at
/// `args.dest.0`, and handed to [`run_write`] as one [`WorldEdit`].
///
/// Air positions inside the source box are skipped by default — copying a
/// tree-shaped selection shouldn't punch an air-shaped hole through whatever
/// already stands at the destination — unless `args.include_air` asks for an
/// exact clone, blank spots included.
pub fn copy(cli: &Cli, args: &CopyArgs) -> Result<CopyResult, CliError> {
    let bounds = SelectionBounds::from_corners(args.corner1.0, args.corner1.0, args.corner2.0);

    let volume = bounds.volume();
    if volume > MAX_BLOCKS {
        return Err(CliError::Usage(format!(
            "selection ({}) to ({}) is {volume} blocks — over the {MAX_BLOCKS}-block copy limit",
            args.corner1.0, args.corner2.0
        )));
    }

    let area = get_area(cli, &GetAreaArgs { from: args.corner1, to: args.corner2 })?;
    let include_air = args.include_air;
    let dest_min = args.dest.0;
    let size = area.size;

    // Built entirely from `area` — already a finished read — before
    // `run_write` opens anything, so nothing here re-reads the destination.
    let mut edit = WorldEdit::new();
    for (index, &palette_index) in area.blocks.iter().enumerate() {
        let state = &area.palette[palette_index as usize];
        if !include_air && state.name == BlockState::AIR {
            continue;
        }
        let local = position_of(IVec3::ZERO, size, index);
        edit.set(dest_min + local, state.clone());
    }
    let copied = edit.len();

    let meta = resolve_save(cli)?;
    let outcome = run_write(&meta, args.dry_run, args.force, move |_cache| Ok(edit))?;

    Ok(CopyResult {
        save_name: meta.name,
        from: bounds.min,
        to: bounds.max,
        dest: dest_min,
        include_air,
        copied,
        outcome,
    })
}

impl Render for CopyResult {
    fn render_text(&self) -> String {
        let prefix = format!(
            "copy {} to {} -> {} in {}{}: {} block{} copied",
            self.from,
            self.to,
            self.dest,
            self.save_name,
            if self.include_air { " (including air)" } else { "" },
            self.copied,
            if self.copied == 1 { "" } else { "s" },
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
        map.insert("from".to_string(), json!([self.from.x, self.from.y, self.from.z]));
        map.insert("to".to_string(), json!([self.to.x, self.to.y, self.to.z]));
        map.insert("dest".to_string(), json!([self.dest.x, self.dest.y, self.dest.z]));
        map.insert("include_air".to_string(), json!(self.include_air));
        map.insert("copied".to_string(), json!(self.copied));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::IVec3;
    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use rnbt::{NbtField, NbtList, NbtValue};

    use crate::blueprint::BlockState;

    use super::*;

    /// The `DataVersion` the fixture's one chunk claims — a 1.21 release,
    /// same as every other fixture in this crate.
    const FIXTURE_DATA_VERSION: i32 = 4438;

    fn dirt() -> BlockState {
        BlockState { name: "minecraft:dirt".to_string(), properties: Vec::new() }
    }

    /// A finished chunk: one all-stone section at `Y = 0`, `Status =
    /// minecraft:full`, the given `DataVersion`. Heightmaps are left out —
    /// `EditPolicy::default()`'s `HeightmapPolicy::Delete` only removes the
    /// compound if one is present, and `viewer::paint`'s own fixture skips
    /// it the same way.
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
    /// removed on drop — the same shape `viewer::paint`'s own test fixture
    /// builds, copied rather than shared since it's private to that module.
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
            let dir = std::env::temp_dir().join(format!("block_viewer-ranvil-cli-write-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk(FIXTURE_DATA_VERSION)));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = SaveMeta {
                name: "write-fixture".to_string(),
                path: dir.clone(),
                region_dir,
                regions: vec![(0, 0)],
            };
            Self { dir, meta }
        }

        fn region_path(&self) -> PathBuf {
            self.meta.get_region_path(0, 0)
        }

        fn bytes(&self) -> Vec<u8> {
            std::fs::read(self.region_path()).expect("read the fixture region")
        }

        fn mtime(&self) -> std::time::SystemTime {
            std::fs::metadata(self.region_path())
                .expect("stat the fixture region")
                .modified()
                .expect("mtime")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn block_name_at(meta: &SaveMeta, at: IVec3) -> String {
        let address = crate::edit::address_of(at);
        let mut cache = RegionCache::new(meta.clone(), 1);
        cache
            .get_or_load(address.region)
            .expect("resident")
            .get_block(address.local_x, address.y, address.local_z)
            .expect("a populated chunk")
            .get_string("Name")
            .expect("a palette entry")
            .clone()
    }

    fn one_block_edit(at: IVec3, state: BlockState) -> WorldEdit {
        let mut edit = WorldEdit::new();
        edit.set(at, state);
        edit
    }

    #[test]
    fn a_successful_edit_backs_up_and_writes() {
        let fixture = Fixture::new("commit");
        let at = IVec3::new(1, 5, 1);

        let outcome = run_write(&fixture.meta, false, false, |_cache| Ok(one_block_edit(at, dirt())))
            .expect("a valid write");

        assert!(!outcome.dry_run);
        assert_eq!(outcome.report.blocks_written, 1);
        assert_eq!(outcome.regions_written, vec![(0, 0)]);
        assert_eq!(outcome.backups.len(), 1);
        assert!(outcome.backup_dir.exists());
        assert_eq!(block_name_at(&fixture.meta, at), "minecraft:dirt");
    }

    #[test]
    fn dry_run_reports_the_plan_and_touches_nothing() {
        let fixture = Fixture::new("dry-run");
        let before_bytes = fixture.bytes();
        let before_mtime = fixture.mtime();
        let at = IVec3::new(1, 5, 1);

        let outcome = run_write(&fixture.meta, true, false, |_cache| Ok(one_block_edit(at, dirt())))
            .expect("a valid plan");

        assert!(outcome.dry_run);
        assert_eq!(outcome.report.blocks_written, 1);
        assert_eq!(outcome.report.regions, vec![(0, 0)]);
        assert!(outcome.regions_written.is_empty());
        assert!(outcome.backups.is_empty());
        assert!(!outcome.backup_dir.exists(), "dry-run must not create the backup dir");

        assert_eq!(fixture.bytes(), before_bytes, "dry-run must not touch the region file's bytes");
        assert_eq!(fixture.mtime(), before_mtime, "dry-run must not touch the region file's mtime");
        assert_eq!(block_name_at(&fixture.meta, at), "minecraft:stone");
    }

    /// Windows only, and deliberately — see `edit::tests`'
    /// `a_world_that_is_open_in_minecraft_is_refused`: a POSIX record lock
    /// never conflicts with its own owner, so on Unix this would pass
    /// without proving anything unless the lock were staged from a child
    /// process.
    #[test]
    #[cfg(windows)]
    fn a_locked_save_without_force_refuses() {
        let fixture = Fixture::new("locked");
        let at = IVec3::new(1, 5, 1);
        let held = mc_anvil::SessionLock::acquire(&fixture.meta)
            .expect("the lock file is reachable")
            .expect("nobody else holds it");

        let err = run_write(&fixture.meta, false, false, |_cache| Ok(one_block_edit(at, dirt())))
            .unwrap_err();

        match err {
            CliError::Data(message) => {
                assert!(message.contains("--force"), "{message}");
                assert!(message.contains("open in Minecraft"), "{message}");
            }
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }

        drop(held);
        assert_eq!(block_name_at(&fixture.meta, at), "minecraft:stone");
    }

    /// `--force` skips the friendly pre-check, but the real `session.lock`
    /// acquisition still fails when the save genuinely is open — the OS
    /// lock can't be forced, only the early, cheaper refusal can be skipped.
    #[test]
    #[cfg(windows)]
    fn force_skips_the_pre_check_but_not_a_genuinely_held_lock() {
        let fixture = Fixture::new("forced-open");
        let at = IVec3::new(1, 5, 1);
        let held = mc_anvil::SessionLock::acquire(&fixture.meta)
            .expect("the lock file is reachable")
            .expect("nobody else holds it");

        let err = run_write(&fixture.meta, false, true, |_cache| Ok(one_block_edit(at, dirt())))
            .unwrap_err();

        match err {
            CliError::Data(message) => assert!(message.contains("open in Minecraft"), "{message}"),
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }

        drop(held);
        // ...and once the world is closed, the same call succeeds.
        let outcome = run_write(&fixture.meta, false, true, |_cache| Ok(one_block_edit(at, dirt())))
            .expect("the save is no longer open");
        assert!(!outcome.dry_run);
    }

    #[test]
    fn a_data_version_incompatible_edit_refuses() {
        let fixture = Fixture::new("dataversion");
        let at = IVec3::new(1, 5, 1);

        // 1000 and the fixture's 4438 sit in different `BLOCK_FORMAT_BOUNDARIES`
        // bands (pre-1.13 versus 1.21) — the same refusal `edit::plan` raises
        // for the game.
        let err = run_write(&fixture.meta, false, false, |_cache| {
            let mut edit = WorldEdit::new().with_data_version(1000);
            edit.set(at, dirt());
            Ok(edit)
        })
        .unwrap_err();

        match err {
            CliError::Data(message) => assert!(message.contains("DataVersion"), "{message}"),
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }
        assert_eq!(block_name_at(&fixture.meta, at), "minecraft:stone");
    }

    // -----------------------------------------------------------------------------------------
    // ---- set / set-area (ticket 095) ---------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    use super::super::cli::{Command, SavesArgs};
    use super::super::coords::BlockPos;
    use super::super::format::OutputFormat;

    fn dummy_cli(save: Option<String>) -> Cli {
        Cli {
            save,
            instance: Some(PathBuf::from("does-not-exist")),
            format: OutputFormat::Json,
            command: Command::Saves(SavesArgs {}),
        }
    }

    fn cli_for(fixture: &Fixture) -> Cli {
        dummy_cli(Some(fixture.meta.path.to_string_lossy().to_string()))
    }

    fn oak_stairs() -> BlockState {
        BlockState {
            name: "minecraft:oak_stairs".to_string(),
            properties: vec![
                ("facing".to_string(), "east".to_string()),
                ("half".to_string(), "top".to_string()),
            ],
        }
    }

    fn set_args(pos: IVec3, state: BlockState, dry_run: bool, force: bool) -> SetArgs {
        SetArgs { pos: BlockPos(pos), state, dry_run, force }
    }

    fn set_area_args(from: IVec3, to: IVec3, state: BlockState, dry_run: bool, force: bool) -> SetAreaArgs {
        SetAreaArgs { from: BlockPos(from), to: BlockPos(to), state, dry_run, force }
    }

    /// Like [`block_name_at`], but the whole [`BlockState`] (name plus
    /// properties) — what a multi-property roundtrip (a stair's `facing`/
    /// `half`) needs that a bare name can't confirm.
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

    /// The ticket's headline "Done when": `set` writes the block, and reading
    /// it straight back (including a multi-property state's properties, not
    /// just its name) matches exactly.
    #[test]
    fn set_writes_a_block_and_reads_back_identically_including_properties() {
        let fixture = Fixture::new("set-roundtrip");
        let at = IVec3::new(1, 5, 1);
        let stairs = oak_stairs();

        let result = set(&cli_for(&fixture), &set_args(at, stairs.clone(), false, false))
            .expect("a valid set");

        assert!(!result.outcome.dry_run);
        assert_eq!(result.outcome.report.blocks_written, 1);
        assert_eq!(result.outcome.regions_written, vec![(0, 0)]);
        assert_eq!(block_state_at(&fixture.meta, at), stairs);
    }

    /// `set-area` over a small box fills exactly those positions and nothing
    /// outside it — the same box `scan --block <name>` (093) would confirm
    /// end to end.
    #[test]
    fn set_area_fills_exactly_the_box() {
        let fixture = Fixture::new("set-area-fill");
        let (from, to) = (IVec3::new(0, 0, 0), IVec3::new(2, 2, 2));
        let target = dirt();

        let result = set_area(&cli_for(&fixture), &set_area_args(from, to, target.clone(), false, false))
            .expect("a valid set-area");

        assert_eq!(result.outcome.report.blocks_written, 27);
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    let pos = IVec3::new(x, y, z);
                    assert_eq!(block_state_at(&fixture.meta, pos), target, "at {pos}");
                }
            }
        }
        // Just outside the box on every axis, the fixture's stone survives.
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(3, 0, 0)), "minecraft:stone");
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(0, 3, 0)), "minecraft:stone");
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(0, 0, 3)), "minecraft:stone");
    }

    /// `--dry-run` on `set` leaves the save's files byte-identical.
    #[test]
    fn set_dry_run_touches_nothing() {
        let fixture = Fixture::new("set-dry-run");
        let (before_bytes, before_mtime) = (fixture.bytes(), fixture.mtime());
        let at = IVec3::new(2, 2, 2);

        let result = set(&cli_for(&fixture), &set_args(at, dirt(), true, false)).expect("a valid plan");

        assert!(result.outcome.dry_run);
        assert_eq!(result.outcome.report.blocks_written, 1);
        assert!(result.outcome.regions_written.is_empty());
        assert_eq!(fixture.bytes(), before_bytes);
        assert_eq!(fixture.mtime(), before_mtime);
        assert_eq!(block_name_at(&fixture.meta, at), "minecraft:stone");
    }

    /// `--dry-run` on `set-area` leaves the save's files byte-identical.
    #[test]
    fn set_area_dry_run_touches_nothing() {
        let fixture = Fixture::new("set-area-dry-run");
        let (before_bytes, before_mtime) = (fixture.bytes(), fixture.mtime());
        let (from, to) = (IVec3::new(0, 0, 0), IVec3::new(1, 1, 1));

        let result = set_area(&cli_for(&fixture), &set_area_args(from, to, dirt(), true, false))
            .expect("a valid plan");

        assert!(result.outcome.dry_run);
        assert_eq!(result.outcome.report.blocks_written, 8);
        assert!(result.outcome.regions_written.is_empty());
        assert_eq!(fixture.bytes(), before_bytes);
        assert_eq!(fixture.mtime(), before_mtime);
        assert_eq!(block_name_at(&fixture.meta, from), "minecraft:stone");
    }

    /// A `set` into a chunk the fixture never generated refuses with the same
    /// [`crate::edit::EditRefusal`] message the game's own write path raises
    /// — exit 1, not a panic or a silent write.
    #[test]
    fn set_into_an_ungenerated_chunk_refuses() {
        let fixture = Fixture::new("set-ungenerated");
        // Chunk index 0 (chunk (0, 0)) is the fixture's only populated chunk;
        // x = 20 lands in chunk (1, 0), which the region never got a payload
        // for.
        let at = IVec3::new(20, 5, 0);

        let err = set(&cli_for(&fixture), &set_args(at, dirt(), false, false)).unwrap_err();

        match err {
            CliError::Data(message) => assert!(message.contains("has not been generated"), "{message}"),
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }
    }

    /// Same refusal, reached through `set-area` when any part of the box
    /// spills into ungenerated territory.
    #[test]
    fn set_area_into_an_ungenerated_chunk_refuses() {
        let fixture = Fixture::new("set-area-ungenerated");
        let (from, to) = (IVec3::new(0, 5, 0), IVec3::new(20, 5, 0));

        let err = set_area(&cli_for(&fixture), &set_area_args(from, to, dirt(), false, false)).unwrap_err();

        match err {
            CliError::Data(message) => assert!(message.contains("has not been generated"), "{message}"),
            CliError::Usage(message) => panic!("expected Data (exit 1), got Usage: {message}"),
        }
    }

    /// `set-area`'s own volume cap (mirrors `get-area`'s [`MAX_BLOCKS`]): a
    /// box over the limit is a `Usage` error (exit 2) raised before
    /// `resolve_save` even runs, same as `get-area`'s own test proves.
    #[test]
    fn set_area_over_max_blocks_is_a_usage_error_before_touching_a_save() {
        let args = set_area_args(
            IVec3::new(0, crate::selection::WORLD_MIN_Y, 0),
            IVec3::new(9999, crate::selection::WORLD_MAX_Y, 9999),
            dirt(),
            false,
            false,
        );

        let err = set_area(&dummy_cli(None), &args).unwrap_err();

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

    fn sample_outcome(dry_run: bool) -> WriteOutcome {
        WriteOutcome {
            report: EditReport {
                blocks_written: 1,
                chunks: vec![(0, 0)],
                regions: vec![(0, 0)],
                replaced: None,
            },
            dry_run,
            backup_dir: PathBuf::from("backups/stamp"),
            regions_written: if dry_run { Vec::new() } else { vec![(0, 0)] },
            backups: if dry_run { Vec::new() } else { vec![PathBuf::from("backups/stamp/r.0.0.mca")] },
        }
    }

    /// The ticket's own contract: a `--dry-run`'s JSON is shaped identically
    /// to a real write's but labeled as a plan, so a script can't mistake one
    /// for the other by skimming.
    #[test]
    fn set_render_json_labels_a_dry_run_as_a_plan_not_a_completed_write() {
        let dry = SetResult {
            save_name: "world".to_string(),
            pos: IVec3::new(1, 2, 3),
            state: dirt(),
            outcome: sample_outcome(true),
        };
        let applied = SetResult {
            save_name: "world".to_string(),
            pos: IVec3::new(1, 2, 3),
            state: dirt(),
            outcome: sample_outcome(false),
        };

        assert_eq!(dry.render_json()["status"], json!("dry-run"));
        assert_eq!(dry.render_json()["regions_written"], json!([]));
        assert_eq!(applied.render_json()["status"], json!("applied"));
        assert_eq!(applied.render_json()["regions_written"], json!([[0, 0]]));

        // Same shape either way — every field present in both, not a
        // dry-run-only or applied-only key that would make the two JSON
        // documents structurally different.
        let (dry_json, applied_json) = (dry.render_json(), applied.render_json());
        let mut dry_keys: Vec<&String> = dry_json.as_object().unwrap().keys().collect();
        let mut applied_keys: Vec<&String> = applied_json.as_object().unwrap().keys().collect();
        dry_keys.sort();
        applied_keys.sort();
        assert_eq!(dry_keys, applied_keys);
    }

    /// The same labeling shows up in `text`, not just `json` — a human
    /// skimming a terminal can't miss a dry run either.
    #[test]
    fn set_render_text_prefixes_dry_run_visibly() {
        let dry = SetResult {
            save_name: "world".to_string(),
            pos: IVec3::new(1, 2, 3),
            state: dirt(),
            outcome: sample_outcome(true),
        };
        assert!(dry.render_text().starts_with("[dry-run]"), "{}", dry.render_text());
        assert!(!dry.render_text().contains("backup:"), "dry-run has no real backup to report");
    }

    // -----------------------------------------------------------------------------------------
    // ---- set-batch / replace (ticket 096) -----------------------------------------------------
    // -----------------------------------------------------------------------------------------

    use super::super::cli::{ReplaceArgs, SetBatchArgs};

    fn oak_slab_top() -> BlockState {
        BlockState {
            name: "minecraft:oak_slab".to_string(),
            properties: vec![("type".to_string(), "top".to_string())],
        }
    }

    #[test]
    fn parse_batch_skips_blank_and_comment_lines() {
        let edits = parse_batch(
            "\n# a comment\n1,2,3 minecraft:dirt\n   \n# another\n4,5,6 minecraft:stone\n",
        )
        .expect("a valid batch");

        assert_eq!(
            edits,
            vec![
                (IVec3::new(1, 2, 3), BlockState { name: "minecraft:dirt".to_string(), properties: vec![] }),
                (IVec3::new(4, 5, 6), BlockState { name: "minecraft:stone".to_string(), properties: vec![] }),
            ]
        );
    }

    /// Duplicate positions parse fine — not deduped, not rejected — so a
    /// caller regenerating a batch that includes a correction never has to
    /// dedupe first (last write wins downstream, in `WorldEdit`/`set_blocks`).
    #[test]
    fn parse_batch_keeps_duplicate_positions_in_order() {
        let edits = parse_batch("1,1,1 minecraft:dirt\n1,1,1 minecraft:stone\n").expect("a valid batch");
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0].0, edits[1].0);
        assert_eq!(edits[1].1.name, "minecraft:stone");
    }

    /// A blockstate carrying `[key=value,...]` still splits correctly on the
    /// first run of whitespace — the bracket isn't mistaken for a second
    /// column.
    #[test]
    fn parse_batch_reads_a_blockstate_with_properties() {
        let edits = parse_batch("1,2,3 minecraft:oak_stairs[facing=east,half=top]\n").expect("a valid batch");
        assert_eq!(edits, vec![(IVec3::new(1, 2, 3), oak_stairs())]);
    }

    /// The ticket's own contract: a malformed line is reported with its
    /// 1-based line number, counting blank and comment lines too, so the
    /// number always matches the line the caller sees in their own file.
    #[test]
    fn parse_batch_reports_the_malformed_lines_number() {
        let err = parse_batch("# header\n1,2,3 minecraft:dirt\nnot a line\n").unwrap_err();
        match err {
            CliError::Usage(message) => assert!(message.contains("line 3"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    /// A line with a position but no blockstate is malformed too, not
    /// silently ignored.
    #[test]
    fn parse_batch_rejects_a_line_missing_its_blockstate() {
        assert!(parse_batch("1,2,3\n").is_err());
        assert!(parse_batch("1,2,3   \n").is_err());
    }

    fn write_batch_file(label: &str, contents: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("block_viewer-ranvil-cli-set-batch-{label}-{nanos}.txt"));
        std::fs::write(&path, contents).expect("write batch file");
        path
    }

    fn set_batch_args(file: String, dry_run: bool, force: bool) -> SetBatchArgs {
        SetBatchArgs { file, dry_run, force }
    }

    /// The ticket's headline "Done when": every line of a small fixture batch
    /// writes correctly and reads back via `get` (here, `block_state_at`).
    #[test]
    fn set_batch_writes_every_line_and_reads_back() {
        let fixture = Fixture::new("set-batch-roundtrip");
        let file = write_batch_file(
            "roundtrip",
            "# a batch\n\n1,5,1 minecraft:dirt\n2,5,1 minecraft:oak_stairs[facing=east,half=top]\n",
        );

        let result = set_batch(&cli_for(&fixture), &set_batch_args(file.to_string_lossy().to_string(), false, false))
            .expect("a valid batch");
        std::fs::remove_file(&file).ok();

        assert_eq!(result.lines_applied, 2);
        assert_eq!(result.outcome.report.blocks_written, 2);
        assert_eq!(block_state_at(&fixture.meta, IVec3::new(1, 5, 1)), dirt());
        assert_eq!(block_state_at(&fixture.meta, IVec3::new(2, 5, 1)), oak_stairs());
    }

    /// A correction in the batch (the same position written twice) leaves the
    /// later line's block in place — last write wins, and the deduped
    /// `blocks_written` count reflects one block in the world, not two lines.
    #[test]
    fn set_batch_last_write_wins_for_a_duplicate_position() {
        let fixture = Fixture::new("set-batch-duplicate");
        let file = write_batch_file(
            "duplicate",
            "1,5,1 minecraft:dirt\n1,5,1 minecraft:gold_block\n",
        );

        let result = set_batch(&cli_for(&fixture), &set_batch_args(file.to_string_lossy().to_string(), false, false))
            .expect("a valid batch");
        std::fs::remove_file(&file).ok();

        assert_eq!(result.lines_applied, 2);
        assert_eq!(result.outcome.report.blocks_written, 1);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(1, 5, 1)), "minecraft:gold_block");
    }

    /// The ticket's other headline "Done when": a batch with one malformed
    /// line writes nothing and reports the offending line number — parsed
    /// (and refused) before `run_write` is even reached.
    #[test]
    fn set_batch_with_a_malformed_line_writes_nothing_and_reports_its_line_number() {
        let fixture = Fixture::new("set-batch-malformed");
        let (before_bytes, before_mtime) = (fixture.bytes(), fixture.mtime());
        let file = write_batch_file(
            "malformed",
            "1,5,1 minecraft:dirt\nnot a valid line\n2,5,1 minecraft:stone\n",
        );

        let err = set_batch(&cli_for(&fixture), &set_batch_args(file.to_string_lossy().to_string(), false, false))
            .unwrap_err();
        std::fs::remove_file(&file).ok();

        match err {
            CliError::Usage(message) => assert!(message.contains("line 2"), "{message}"),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
        assert_eq!(fixture.bytes(), before_bytes, "a malformed batch must not touch the region file's bytes");
        assert_eq!(fixture.mtime(), before_mtime);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(1, 5, 1)), "minecraft:stone");
    }

    /// A batch of only blank/comment lines is refused before `resolve_save`
    /// even runs — an empty batch is a bad request regardless of which save
    /// it names, the same reasoning `set-area`'s `MAX_BLOCKS` check uses.
    #[test]
    fn set_batch_with_no_edits_is_a_usage_error() {
        let file = write_batch_file("empty", "# nothing here\n\n");

        let err = set_batch(&dummy_cli(None), &set_batch_args(file.to_string_lossy().to_string(), false, false))
            .unwrap_err();
        std::fs::remove_file(&file).ok();

        match err {
            CliError::Usage(message) => assert!(
                !message.contains("instance directory"),
                "should fail on the empty-batch check, not on resolving a save: {message}"
            ),
            CliError::Data(message) => panic!("expected Usage (exit 2), got Data: {message}"),
        }
    }

    fn replace_args(
        corner1: IVec3,
        corner2: IVec3,
        from_block: &str,
        to_state: BlockState,
        dry_run: bool,
        force: bool,
    ) -> ReplaceArgs {
        ReplaceArgs {
            corner1: BlockPos(corner1),
            corner2: BlockPos(corner2),
            from_block: from_block.to_string(),
            to_state,
            dry_run,
            force,
        }
    }

    /// The ticket's own "Done when": `replace` over a box holding a mix of
    /// blocks changes only the matching positions.
    #[test]
    fn replace_changes_only_the_matching_positions() {
        let fixture = Fixture::new("replace-mix");
        // The fixture is all stone; plant one dirt block and one stair inside
        // the box `replace` will scan, so it's a real mix rather than
        // uniform stone.
        set(&cli_for(&fixture), &set_args(IVec3::new(1, 0, 0), dirt(), false, false)).expect("plant dirt");
        set(&cli_for(&fixture), &set_args(IVec3::new(2, 0, 0), oak_stairs(), false, false))
            .expect("plant a stair");

        let result = replace(
            &cli_for(&fixture),
            &replace_args(
                IVec3::new(0, 0, 0),
                IVec3::new(2, 0, 0),
                "minecraft:stone",
                oak_slab_top(),
                false,
                false,
            ),
        )
        .expect("a valid replace");

        // Only (0, 0, 0) was stone; the dirt and the stair are untouched.
        assert_eq!(result.matched, 1);
        assert_eq!(result.outcome.report.blocks_written, 1);
        assert_eq!(block_state_at(&fixture.meta, IVec3::new(0, 0, 0)), oak_slab_top());
        assert_eq!(block_state_at(&fixture.meta, IVec3::new(1, 0, 0)), dirt());
        assert_eq!(block_state_at(&fixture.meta, IVec3::new(2, 0, 0)), oak_stairs());
    }

    /// `replace` matches by name only — properties never factor in, same
    /// convention `scan --block` uses (an oriented stair still counts as
    /// `minecraft:oak_stairs`).
    #[test]
    fn replace_matches_by_name_ignoring_properties() {
        let fixture = Fixture::new("replace-by-name");
        set(&cli_for(&fixture), &set_args(IVec3::new(0, 0, 0), oak_stairs(), false, false))
            .expect("plant a stair");

        let result = replace(
            &cli_for(&fixture),
            &replace_args(IVec3::new(0, 0, 0), IVec3::new(0, 0, 0), "minecraft:oak_stairs", dirt(), false, false),
        )
        .expect("a valid replace");

        assert_eq!(result.matched, 1);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(0, 0, 0)), "minecraft:dirt");
    }

    /// `--dry-run` on `replace` leaves the save's files byte-identical, same
    /// contract every other write command holds to.
    #[test]
    fn replace_dry_run_touches_nothing() {
        let fixture = Fixture::new("replace-dry-run");
        let (before_bytes, before_mtime) = (fixture.bytes(), fixture.mtime());

        let result = replace(
            &cli_for(&fixture),
            &replace_args(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), "minecraft:stone", dirt(), true, false),
        )
        .expect("a valid plan");

        assert!(result.outcome.dry_run);
        assert_eq!(result.matched, 2);
        assert_eq!(result.outcome.report.blocks_written, 2);
        assert_eq!(fixture.bytes(), before_bytes);
        assert_eq!(fixture.mtime(), before_mtime);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(0, 0, 0)), "minecraft:stone");
    }

    /// `replace`'s own volume cap (mirrors `get-area`/`set-area`'s
    /// [`MAX_BLOCKS`]): a box over the limit is a `Usage` error (exit 2)
    /// raised before `resolve_save` even runs.
    #[test]
    fn replace_over_max_blocks_is_a_usage_error_before_touching_a_save() {
        let args = replace_args(
            IVec3::new(0, crate::selection::WORLD_MIN_Y, 0),
            IVec3::new(9999, crate::selection::WORLD_MAX_Y, 9999),
            "minecraft:stone",
            dirt(),
            false,
            false,
        );

        let err = replace(&dummy_cli(None), &args).unwrap_err();

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

    // -----------------------------------------------------------------------------------------
    // ---- copy (ticket 097) -----------------------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    use super::super::cli::CopyArgs;

    fn copy_args(
        corner1: IVec3,
        corner2: IVec3,
        dest: IVec3,
        include_air: bool,
        dry_run: bool,
        force: bool,
    ) -> CopyArgs {
        CopyArgs {
            corner1: BlockPos(corner1),
            corner2: BlockPos(corner2),
            dest: BlockPos(dest),
            include_air,
            dry_run,
            force,
        }
    }

    /// The ticket's headline "Done when": a small structure with a few
    /// distinct blocks (one with properties) reproduces exactly at the
    /// destination, confirmed the same way the ticket asks — reading each
    /// position back and comparing it to its source, the same
    /// [`block_state_at`] every other write test's roundtrip check uses.
    #[test]
    fn copy_reproduces_a_small_structure_exactly_at_the_destination() {
        let fixture = Fixture::new("copy-roundtrip");
        let cli = cli_for(&fixture);
        set(&cli, &set_args(IVec3::new(0, 0, 0), dirt(), false, false)).expect("plant dirt");
        set(&cli, &set_args(IVec3::new(1, 0, 0), oak_stairs(), false, false)).expect("plant a stair");
        // (2, 0, 0) is left as the fixture's own stone.

        let result = copy(
            &cli,
            &copy_args(IVec3::new(0, 0, 0), IVec3::new(2, 0, 0), IVec3::new(5, 0, 0), false, false, false),
        )
        .expect("a valid copy");

        assert_eq!(result.copied, 3);
        assert_eq!(result.outcome.report.blocks_written, 3);
        for dx in 0..3 {
            let src = IVec3::new(dx, 0, 0);
            let dest = IVec3::new(5 + dx, 0, 0);
            assert_eq!(block_state_at(&fixture.meta, dest), block_state_at(&fixture.meta, src), "at offset {dx}");
        }
    }

    /// The ticket's other headline "Done when": an overlapping
    /// source/destination — a checkerboard shifted by one — reads the
    /// *original* blocks throughout, not a partially-copied one.
    #[test]
    fn copy_with_an_overlapping_destination_reads_the_original_blocks() {
        let fixture = Fixture::new("copy-overlap");
        let cli = cli_for(&fixture);
        let pattern = [dirt(), oak_slab_top(), dirt(), oak_slab_top()];
        for (dx, state) in pattern.iter().enumerate() {
            set(&cli, &set_args(IVec3::new(dx as i32, 5, 0), state.clone(), false, false))
                .expect("plant the checkerboard");
        }

        let result = copy(
            &cli,
            &copy_args(IVec3::new(0, 5, 0), IVec3::new(3, 5, 0), IVec3::new(1, 5, 0), false, false, false),
        )
        .expect("a valid overlapping copy");

        assert_eq!(result.copied, 4);
        for (dx, state) in pattern.iter().enumerate() {
            let dest = IVec3::new(1 + dx as i32, 5, 0);
            assert_eq!(block_state_at(&fixture.meta, dest), *state, "at dest offset {dx}");
        }
    }

    /// `--include-air` overwrites the destination where the source was air;
    /// without it, that position is untouched.
    #[test]
    fn include_air_controls_whether_air_overwrites_the_destination() {
        let fixture = Fixture::new("copy-include-air");
        let cli = cli_for(&fixture);
        set(&cli, &set_args(IVec3::new(0, 0, 0), BlockState::air(), false, false)).expect("plant air");
        set(&cli, &set_args(IVec3::new(1, 0, 0), dirt(), false, false)).expect("plant dirt");
        // A distinctive prior state at the destination, so "untouched" is
        // unambiguous.
        set(&cli, &set_args(IVec3::new(5, 0, 0), oak_stairs(), false, false)).expect("prefill dest");

        let without = copy(
            &cli,
            &copy_args(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(5, 0, 0), false, false, false),
        )
        .expect("a valid copy");
        assert_eq!(without.copied, 1, "only the non-air source position is copied");
        assert_eq!(
            block_state_at(&fixture.meta, IVec3::new(5, 0, 0)),
            oak_stairs(),
            "the air source position leaves the destination untouched"
        );
        assert_eq!(block_state_at(&fixture.meta, IVec3::new(6, 0, 0)), dirt());

        // Re-prefill for a clean "with `--include-air`" comparison.
        set(&cli, &set_args(IVec3::new(5, 0, 0), oak_stairs(), false, false)).expect("prefill dest again");

        let with = copy(
            &cli,
            &copy_args(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(5, 0, 0), true, false, false),
        )
        .expect("a valid copy");
        assert_eq!(with.copied, 2);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(5, 0, 0)), "minecraft:air");
    }

    /// `--dry-run` on `copy` leaves the save's files byte-identical, same
    /// contract every other write command holds to.
    #[test]
    fn copy_dry_run_touches_nothing() {
        let fixture = Fixture::new("copy-dry-run");
        let cli = cli_for(&fixture);
        set(&cli, &set_args(IVec3::new(0, 0, 0), dirt(), false, false)).expect("plant dirt");
        let (before_bytes, before_mtime) = (fixture.bytes(), fixture.mtime());

        let result = copy(
            &cli,
            &copy_args(IVec3::new(0, 0, 0), IVec3::new(0, 0, 0), IVec3::new(5, 0, 0), false, true, false),
        )
        .expect("a valid plan");

        assert!(result.outcome.dry_run);
        assert_eq!(result.copied, 1);
        assert_eq!(fixture.bytes(), before_bytes);
        assert_eq!(fixture.mtime(), before_mtime);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(5, 0, 0)), "minecraft:stone");
    }

    /// `copy`'s own volume cap (mirrors `get-area`/`set-area`/`replace`'s
    /// [`MAX_BLOCKS`]): a box over the limit is a `Usage` error (exit 2)
    /// raised before `resolve_save` even runs.
    #[test]
    fn copy_over_max_blocks_is_a_usage_error_before_touching_a_save() {
        let args = copy_args(
            IVec3::new(0, crate::selection::WORLD_MIN_Y, 0),
            IVec3::new(9999, crate::selection::WORLD_MAX_Y, 9999),
            IVec3::new(0, 0, 0),
            false,
            false,
            false,
        );

        let err = copy(&dummy_cli(None), &args).unwrap_err();

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

    #[test]
    fn copy_render_json_matches_the_documented_shape() {
        let result = CopyResult {
            save_name: "world".to_string(),
            from: IVec3::new(0, 60, 0),
            to: IVec3::new(1, 60, 0),
            dest: IVec3::new(10, 60, 0),
            include_air: false,
            copied: 2,
            outcome: sample_outcome(false),
        };

        let json = result.render_json();
        assert_eq!(json["save"], json!("world"));
        assert_eq!(json["from"], json!([0, 60, 0]));
        assert_eq!(json["to"], json!([1, 60, 0]));
        assert_eq!(json["dest"], json!([10, 60, 0]));
        assert_eq!(json["include_air"], json!(false));
        assert_eq!(json["copied"], json!(2));
        assert_eq!(json["status"], json!("applied"));
    }

    /// The dry-run label shows up in `text` too, same as every other write
    /// command.
    #[test]
    fn copy_render_text_prefixes_dry_run_visibly() {
        let dry = CopyResult {
            save_name: "world".to_string(),
            from: IVec3::new(0, 60, 0),
            to: IVec3::new(1, 60, 0),
            dest: IVec3::new(10, 60, 0),
            include_air: false,
            copied: 2,
            outcome: sample_outcome(true),
        };
        assert!(dry.render_text().starts_with("[dry-run]"), "{}", dry.render_text());
    }
}
