//! "Save world" (ticket 051): the manual write [`super::commit`]'s module
//! docs point at — the moment every dirty region the shared [`RegionCache`]
//! is holding actually gets backed up and written to disk.
//!
//! ## Following `city::undo`'s shape
//!
//! Same `request`/`busy`/`state` surface [`crate::viewer::paint::PaintCommand`]
//! established: a click sets [`SaveCommand::requested`], [`start_save`]
//! dispatches onto [`AsyncComputeTaskPool`], [`poll_save`] polls it with the
//! same `block_on(poll_once(..))` pattern every other task in this crate
//! uses.
//!
//! Unlike commit/demolish/undo, this is the *only* place in the citybuilder
//! that opens a [`crate::edit::session::WriteSession`] — see
//! `city::commit`'s module docs' "Applied to memory, not written to disk"
//! for why the other three stopped. [`WriteSession::flush`] already exists
//! (ticket 033) and was unused until now: it writes every region the shared
//! [`RegionCache`] currently has dirty, backing each one up once per session,
//! regardless of which of commit/demolish/undo made it dirty. `flush` locks
//! the same `Arc<Mutex<RegionCache>>` commit/demolish/undo already share for
//! their own applies, so a save landing mid-placement simply blocks on the
//! mutex like any other contended access — no new coordination needed.
//!
//! ## One save in flight at a time
//!
//! [`SaveState::pending`] is this module's own single slot, the same
//! backpressure shape `city::commit::CommitState`/`city::demolish::DemolishState`
//! use. A second click while a save is already writing is ignored —
//! `city::ui::city_panel` disables the button then, the same backstop those
//! two modules' own docs describe.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};

use crate::chunk_pipeline::SharedRegionCache;
use crate::edit::session::{WriteError, WriteSession, WriteSummary};
use crate::region_cache::RegionCache;
use crate::LoadedSave;

use super::loading::GameplaySet;
use super::write_status::WriteStatus;

/// A save's flush, in flight.
struct PendingSave {
    task: Task<Result<WriteSummary, WriteError>>,
}

/// What a save command is doing, or last did — sticky terminal states, the
/// same reasoning [`crate::viewer::paint::PaintState`]/`city::undo::UndoState`
/// give.
#[derive(Default)]
pub(super) enum SaveState {
    #[default]
    Idle,
    Saving,
    Done {
        regions: usize,
        backups: usize,
    },
    Failed {
        message: String,
    },
}

/// One "Save world" request, from the click to the flush — see the module
/// docs.
#[derive(Resource, Default)]
pub(super) struct SaveCommand {
    requested: bool,
    pending: Option<PendingSave>,
    state: SaveState,
}

impl SaveCommand {
    /// Asks for every dirty region to be flushed to disk. Ignored (returning
    /// `false`) while already busy — `city::ui::city_panel` disables its
    /// button then, so this is a backstop rather than the normal path, the
    /// same contract [`super::undo::UndoCommand::request`] documents.
    pub(super) fn request(&mut self) -> bool {
        if self.busy() {
            return false;
        }
        self.requested = true;
        true
    }

    pub(super) fn busy(&self) -> bool {
        self.requested || self.pending.is_some()
    }

    pub(super) fn state(&self) -> &SaveState {
        &self.state
    }
}

pub struct SavePlugin;

impl Plugin for SavePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SaveCommand>()
            // Idempotent-either-order, the same shape `WriteStatus` already
            // uses across `city::commit`/`city::demolish`/`city::undo`.
            .init_resource::<WriteStatus>()
            .add_systems(Update, (start_save, poll_save).chain().in_set(GameplaySet));
    }
}

/// Dispatches a requested save onto [`AsyncComputeTaskPool`]: opens a
/// [`WriteSession`] and flushes every dirty region the shared
/// [`RegionCache`] is holding.
fn start_save(mut save: ResMut<SaveCommand>, loaded_save: Option<Res<LoadedSave>>, region_cache: Option<Res<SharedRegionCache>>) {
    if !std::mem::take(&mut save.requested) {
        return;
    }

    let (Some(loaded_save), Some(region_cache)) = (loaded_save, region_cache) else {
        save.state = SaveState::Failed { message: "no save is loaded".to_string() };
        return;
    };

    let save_meta = loaded_save.0.meta.clone();
    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        let mut session = WriteSession::open(&save_meta)?;
        session.flush(&mut *cache)
    });

    save.pending = Some(PendingSave { task });
    save.state = SaveState::Saving;
}

/// Single non-blocking poll of the in-flight save — same
/// `block_on(poll_once(..))` pattern every other task in this crate uses.
fn poll_save(mut save: ResMut<SaveCommand>, mut write_status: ResMut<WriteStatus>) {
    let result = {
        let Some(pending) = &mut save.pending else { return };
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            return; // Still saving.
        };
        result
    };
    save.pending = None;

    match result {
        Ok(summary) => {
            println!(
                "block_viewer: saved {} region file(s) ({} new backup(s))",
                summary.regions_written.len(),
                summary.backups.len()
            );
            save.state = SaveState::Done { regions: summary.regions_written.len(), backups: summary.backups.len() };
            write_status.record_save_success(&summary);
        }
        Err(err) => {
            println!("block_viewer: save failed: {err}");
            save.state = SaveState::Failed { message: err.to_string() };
            write_status.record_save_failure(err.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::{EditPolicy, WorldEdit};
    use bevy::math::IVec3;
    use bevy::tasks::TaskPool;
    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use mc_anvil::SaveMeta as Meta;
    use rnbt::{NbtField, NbtList, NbtValue};

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(SavePlugin);
        app
    }

    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if !app.world().resource::<SaveCommand>().busy() {
                return;
            }
        }
        panic!("save never settled");
    }

    #[test]
    fn requesting_without_a_save_loaded_fails() {
        let mut app = app();
        app.world_mut().resource_mut::<SaveCommand>().request();
        run_until_settled(&mut app);

        let save = app.world().resource::<SaveCommand>();
        assert!(matches!(save.state(), SaveState::Failed { message } if message.contains("no save is loaded")));
    }

    #[test]
    fn a_request_is_refused_while_another_is_pending() {
        let mut save = SaveCommand::default();
        assert!(!save.busy());
        assert!(save.request());
        assert!(save.busy());
        assert!(!save.request());
    }

    #[test]
    fn a_finished_save_reports_done_and_records_write_status() {
        let mut app = app();
        app.world_mut().resource_mut::<SaveCommand>().pending = Some(PendingSave {
            task: pool().spawn(async {
                Ok(WriteSummary {
                    report: crate::edit::EditReport { regions: vec![(0, 0)], ..Default::default() },
                    regions_written: vec![(0, 0)],
                    backups: vec![std::path::PathBuf::from("world/block_viewer_backups/x/r.0.0.mca")],
                })
            }),
        });
        app.world_mut().resource_mut::<SaveCommand>().state = SaveState::Saving;

        run_until_settled(&mut app);

        let save = app.world().resource::<SaveCommand>();
        assert!(matches!(save.state(), SaveState::Done { regions: 1, backups: 1 }));

        let status = app.world().resource::<WriteStatus>();
        assert!(status.last_save().is_some());
    }

    #[test]
    fn a_failed_save_is_reported() {
        let mut app = app();
        app.world_mut().resource_mut::<SaveCommand>().pending = Some(PendingSave {
            task: pool().spawn(async { Err(WriteError::WorldIsOpen { save: "world".to_string() }) }),
        });
        app.world_mut().resource_mut::<SaveCommand>().state = SaveState::Saving;

        run_until_settled(&mut app);

        let save = app.world().resource::<SaveCommand>();
        assert!(matches!(save.state(), SaveState::Failed { message } if message.contains("open in Minecraft")));
    }

    // --- a real flush, end to end -------------------------------------------
    //
    // Two edits applied in memory (mirroring `city::commit::apply_building_edit`),
    // then a real `WriteSession::flush` — proving ticket 051's whole point:
    // nothing hits disk until this call, and it picks up every dirty region
    // regardless of which caller dirtied it.

    struct Fixture {
        dir: std::path::PathBuf,
        meta: Meta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let dir = std::env::temp_dir().join(format!("block_viewer-save-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk()));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = Meta { name: "save-fixture".to_string(), path: dir.clone(), region_dir, regions: vec![(0, 0)] };
            Self { dir, meta }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn full_chunk() -> NbtField {
        let palette = NbtList::Compound(vec![NbtField::new_compound("", vec![NbtField::new_string("Name", "minecraft:stone")])]);
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
                NbtField::new_i32("DataVersion", 4438),
                NbtField::new_string("Status", "minecraft:full"),
                NbtField { name: "isLightOn".to_string(), value: NbtValue::Byte(1) },
            ],
        )
    }

    fn block_name_at(cache: &mut RegionCache, at: IVec3) -> String {
        let address = crate::edit::address_of(at);
        cache
            .get_or_load(address.region)
            .expect("resident")
            .get_block(address.local_x, address.y, address.local_z)
            .expect("a populated chunk")
            .get_string("Name")
            .expect("a palette entry")
            .clone()
    }

    #[test]
    fn flush_writes_every_edit_batched_in_memory_and_clears_the_dirty_set() {
        let fixture = Fixture::new("flush");
        let mut cache = RegionCache::new(fixture.meta.clone(), 4);
        let policy = EditPolicy { allow_dirty_regions: true, ..EditPolicy::default() };

        let mut first = WorldEdit::new();
        first.set(IVec3::new(1, 5, 1), "minecraft:dirt".parse().unwrap());
        crate::edit::apply_routed(&first, &mut cache, &policy).expect("first edit");

        let mut second = WorldEdit::new();
        second.set(IVec3::new(4, 5, 4), "minecraft:dirt".parse().unwrap());
        crate::edit::apply_routed(&second, &mut cache, &policy).expect("second edit, same region, still dirty");

        assert_eq!(cache.dirty_regions().count(), 1, "batched in memory — no save call yet");

        let mut session = WriteSession::open_with(&fixture.meta, crate::edit::session::WriteSafety { require_session_lock: false, back_up: true })
            .expect("open a write session");
        let summary = session.flush(&mut cache).expect("flush every dirty region");
        assert_eq!(summary.regions_written, vec![(0, 0)]);
        assert_eq!(cache.dirty_regions().count(), 0, "flushed — nothing left dirty");

        // A fresh cache over the same directory reads the flushed edits back
        // from disk, proving the flush actually landed there and not just in
        // the cache that made it.
        let mut reloaded = RegionCache::new(fixture.meta.clone(), 4);
        assert_eq!(block_name_at(&mut reloaded, IVec3::new(1, 5, 1)), "minecraft:dirt");
        assert_eq!(block_name_at(&mut reloaded, IVec3::new(4, 5, 4)), "minecraft:dirt");
    }
}
