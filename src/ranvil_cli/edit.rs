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

use mc_anvil::SaveMeta;

use crate::edit::{EditPolicy, EditReport, WorldEdit, WriteSession};
use crate::region_cache::RegionCache;

use super::error::CliError;

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
}
