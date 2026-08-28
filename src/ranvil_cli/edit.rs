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

use std::path::PathBuf;

use bevy::math::IVec3;
use mc_anvil::SaveMeta;
use serde_json::{json, Value};

use crate::blueprint::{BlockState, MAX_BLOCKS};
use crate::edit::{EditPolicy, EditReport, WorldEdit, WriteSession};
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;

use super::cli::{Cli, SetAreaArgs, SetArgs};
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
}
