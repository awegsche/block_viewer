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

use std::path::PathBuf;

use bevy::math::IVec3;
use serde_json::{json, Value};

use crate::blueprint::{
    read_structure_file, write_structure_file, BlockState, Blueprint, FALLBACK_DATA_VERSION,
    LOGGED_PALETTE_ENTRIES, STRUCTURE_BLOCK_MAX_SIZE,
};

use super::cli::{StructInfoArgs, StructNewArgs};
use super::error::CliError;
use super::format::Render;

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
}
