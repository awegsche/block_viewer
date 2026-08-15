//! The paint/fill command (ticket 035, roadmap W8): fill the current
//! [`crate::selection::Selection`] with one block, writing for real into the
//! save's region files. Driven from [`crate::viewer::ui::selection_panel`],
//! the existing selection panel, per the roadmap's instruction — no new
//! window.
//!
//! This is the first thing in the whole project that turns W1-W7 into a
//! button a person can click: [`crate::edit::WorldEdit::fill`] builds the
//! edit, [`crate::edit::session::WriteSession`] opens, backs up and commits
//! it (W6, on top of W4's model and W5's region routing), and a successful
//! commit fires [`crate::chunk_pipeline::ChunksEdited`] so the chunks just
//! written re-decode and re-mesh on screen (W7) without a restart. Manual
//! verification — that the edit is also correct when the same world is
//! opened in Minecraft — is a `todo.md` item per this repo's rule on visual
//! checks, not something this ticket can itself confirm.
//!
//! ## Why a task, and why it holds the region-cache lock throughout
//!
//! [`crate::blueprint::extract`] takes the shared region-cache lock **per
//! chunk column** so a multi-second extraction doesn't stall terrain
//! streaming. A commit can't do that: [`WriteSession::commit`] is one
//! transaction — plan every region, then apply every region, then back up
//! and save every region — and releasing the lock partway through would let
//! a streaming load or another edit observe (or evict) a region mid-edit,
//! which is exactly what 032's "all or nothing" guarantee depends on not
//! happening. So this holds the lock for the whole commit. Streaming stalls
//! for as long as the write takes, same as any other write to a shared
//! resource; acceptable because painting is a deliberate, occasional click,
//! not a continuous background operation the way streaming is.
//!
//! ## Following [`crate::blueprint::export`]'s shape
//!
//! One state machine, one slot, the same `request`/`busy`/`state` surface
//! and the same `start_*`/`poll_*` system pair polled with
//! `block_on(poll_once(..))` — deliberately, so a reader of one already
//! knows how to read the other.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use mc_anvil::SaveMeta;

use crate::blueprint::BlockState;
use crate::chunk_pipeline::{ChunksEdited, SharedRegionCache};
use crate::edit::session::{WriteError, WriteSession, WriteSummary};
use crate::edit::{EditPolicy, WorldEdit};
use crate::region_cache::RegionCache;
use crate::selection::SelectionBounds;
use crate::LoadedSave;

/// One paint command, from the click to the write.
///
/// One at a time, like [`crate::blueprint::BlueprintExport`] — a second fill
/// while one is committing would race it over the same region files.
#[derive(Resource, Default)]
pub(crate) struct PaintCommand {
    /// A click waiting for [`start_paint`] to pick it up on the same frame.
    requested: Option<(SelectionBounds, BlockState)>,
    state: PaintState,
}

/// What a paint command is doing, or last did. Sticky terminal states, the
/// same reasoning [`crate::blueprint::export::ExportState`] gives: a result
/// that vanishes after one frame tells the user nothing.
#[derive(Default)]
pub(crate) enum PaintState {
    /// Nothing in flight. The button is enabled.
    #[default]
    Idle,
    /// The commit is running: [`WriteSession::open`], then
    /// [`WriteSession::commit`], off the main thread.
    Writing { blocks: usize, task: Task<Result<WriteSummary, WriteError>> },
    /// The last paint wrote `blocks` blocks across `chunks` chunk(s) and
    /// `regions` region file(s).
    Done { blocks: usize, chunks: usize, regions: usize },
    /// The last paint failed. Also sticky, for the same reason.
    Failed { message: String },
}

impl PaintState {
    /// The three middle states — also exactly when the panel disables its
    /// button. [`Self::Done`]/[`Self::Failed`] are the *previous* paint's
    /// result sitting on screen, not this one's progress.
    fn in_flight(&self) -> bool {
        matches!(self, PaintState::Writing { .. })
    }
}

impl PaintCommand {
    /// Asks for `bounds` to be filled with `state`. Ignored (returning
    /// `false`) while a paint is already in flight — the panel disables its
    /// button then, so this is a backstop rather than the normal path.
    pub(crate) fn request(&mut self, bounds: SelectionBounds, state: BlockState) -> bool {
        if self.busy() {
            return false;
        }
        self.requested = Some((bounds, state));
        true
    }

    /// Whether a click is pending or a commit is running.
    pub(crate) fn busy(&self) -> bool {
        self.requested.is_some() || self.state.in_flight()
    }

    /// What to report. The panel matches on this.
    pub(crate) fn state(&self) -> &PaintState {
        &self.state
    }
}

/// Adds [`PaintCommand`] and the two systems that drive it.
///
/// Registers [`ChunksEdited`] itself rather than relying on
/// [`crate::chunk_pipeline::ChunkLoadPipelinePlugin`] having run first —
/// `add_event` is idempotent, and this way `PaintPlugin` doesn't silently
/// depend on plugin registration order (today it's always added after that
/// one, through [`crate::world_app`], but nothing enforces it).
pub(crate) struct PaintPlugin;

impl Plugin for PaintPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PaintCommand>()
            .add_event::<ChunksEdited>()
            .add_systems(Update, (poll_paint, start_paint).chain());
    }
}

/// Opens a write session on `save` and commits `edit` through `cache` — the
/// task body, pulled out on its own so it can be tested directly and
/// synchronously rather than through a real [`AsyncComputeTaskPool`] task,
/// the same way [`crate::chunk_pipeline::load_and_mesh_chunk`] is both a
/// task body and the function its own tests call.
fn commit_fill(
    save: &SaveMeta,
    cache: &mut RegionCache,
    edit: &WorldEdit,
    policy: &EditPolicy,
) -> Result<WriteSummary, WriteError> {
    let mut session = WriteSession::open(save)?;
    session.commit(edit, cache, policy)
}

/// Dispatches a requested paint onto [`AsyncComputeTaskPool`].
///
/// Mirrors [`crate::blueprint::start_extraction`]: no save/region cache is a
/// reportable failure (ticket 008's empty-save startup can leave the app in
/// exactly that state) rather than a silently dropped click.
fn start_paint(
    mut paint: ResMut<PaintCommand>,
    loaded_save: Option<Res<LoadedSave>>,
    region_cache: Option<Res<SharedRegionCache>>,
) {
    let Some((bounds, block_state)) = paint.requested.take() else {
        return;
    };
    let (Some(loaded_save), Some(region_cache)) = (loaded_save, region_cache) else {
        paint.state = PaintState::Failed { message: "no save is loaded".to_string() };
        return;
    };

    let edit = WorldEdit::fill(bounds, block_state);
    let blocks = edit.len();
    let save_meta = loaded_save.0.meta.clone();
    let cache: Arc<Mutex<RegionCache>> = region_cache.0.clone();

    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut cache = cache.lock().expect("region cache mutex poisoned");
        commit_fill(&save_meta, &mut cache, &edit, &EditPolicy::default())
    });

    paint.state = PaintState::Writing { blocks, task };
}

/// Single non-blocking poll of the in-flight commit, the same
/// `block_on(poll_once(..))` pattern every other task in this crate uses. On
/// success, fires [`ChunksEdited`] with the written chunks so 034's reload
/// queue picks them up (W7) — the whole reason this command proves W1-W7
/// rather than just W1-W6.
fn poll_paint(mut paint: ResMut<PaintCommand>, mut edited: EventWriter<ChunksEdited>) {
    let PaintState::Writing { task, .. } = &mut paint.state else {
        return;
    };
    let Some(result) = block_on(poll_once(task)) else {
        return; // Still writing.
    };

    paint.state = match result {
        Ok(summary) => {
            let blocks = summary.report.blocks_written;
            let chunks = summary.report.chunks.len();
            let regions = summary.regions_written.len();
            println!(
                "block_viewer: painted {blocks} block(s) across {chunks} chunk(s), \
                 {regions} region file(s)"
            );
            edited.send(ChunksEdited(summary.report.chunks));
            PaintState::Done { blocks, chunks, regions }
        }
        Err(err) => {
            println!("block_viewer: paint failed: {err}");
            PaintState::Failed { message: err.to_string() }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::tasks::TaskPool;
    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use rnbt::{NbtField, NbtList, NbtValue};

    /// A single-region, single-chunk fixture save: chunk (0,0), one section
    /// of stone at `Y = 0`, `Status = minecraft:full`. Just enough for
    /// [`commit_fill`] to have somewhere to write — a slimmed-down, private
    /// copy of `edit::tests`' `RegionFixture`/`SaveFixture`, which this
    /// module can't reach since they're private to that one.
    struct Fixture {
        dir: std::path::PathBuf,
        meta: SaveMeta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("block_viewer-paint-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk()));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = SaveMeta {
                name: "paint-fixture".to_string(),
                path: dir.clone(),
                region_dir,
                regions: vec![(0, 0)],
            };
            Self { dir, meta }
        }

        fn cache(&self) -> RegionCache {
            RegionCache::new(self.meta.clone(), 4)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A finished chunk with one all-stone section at `Y = 0` (world Y
    /// 0..15) — the same shape `edit::tests::full_chunk` builds, copied
    /// rather than shared for the reason [`Fixture`]'s docs give.
    fn full_chunk() -> NbtField {
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

    fn dirt() -> BlockState {
        BlockState { name: "minecraft:dirt".to_string(), properties: Vec::new() }
    }

    fn bounds() -> SelectionBounds {
        SelectionBounds::from_corners(
            IVec3::new(1, 5, 1),
            IVec3::new(1, 5, 1),
            IVec3::new(3, 5, 3),
        )
    }

    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(PaintPlugin);
        app
    }

    /// Ticks `app` until `PaintCommand` is no longer busy, or gives up.
    ///
    /// A task spawned directly onto [`AsyncComputeTaskPool`] (as these tests
    /// do, rather than going through `start_paint`) can complete on its
    /// worker thread before or after any particular `app.update()` — a real
    /// race, not a hypothetical one; a single `app.update()` was observed to
    /// see the task still in flight. Looping with a generous cap is the
    /// standard fix for polling a background task from a synchronous test.
    fn run_until_settled(app: &mut App) {
        for _ in 0..200 {
            app.update();
            if !app.world().resource::<PaintCommand>().busy() {
                return;
            }
        }
        panic!("paint command never settled");
    }

    // -----------------------------------------------------------------------------------------
    // ---- commit_fill: the actual write, tested directly and synchronously --------------------
    // -----------------------------------------------------------------------------------------

    #[test]
    fn commit_fill_writes_every_block_and_reports_it() {
        let fixture = Fixture::new("commit");
        let mut cache = fixture.cache();
        let edit = WorldEdit::fill(bounds(), dirt());

        let summary =
            commit_fill(&fixture.meta, &mut cache, &edit, &EditPolicy::default()).expect("a valid fill");

        assert_eq!(summary.report.blocks_written, 9);
        assert_eq!(summary.report.chunks, vec![(0, 0)]);
        assert_eq!(summary.regions_written, vec![(0, 0)]);

        for at in bounds().iter_blocks() {
            assert_eq!(block_name_at(&mut cache, at), "minecraft:dirt");
        }
        // Just outside the box, untouched.
        assert_eq!(block_name_at(&mut cache, IVec3::new(4, 5, 4)), "minecraft:stone");
    }

    #[test]
    fn commit_fill_refuses_ungenerated_terrain_and_writes_nothing() {
        let fixture = Fixture::new("refuse");
        let mut cache = fixture.cache();
        // Chunk (5, 0) is outside the fixture's one generated chunk.
        let out_of_bounds = SelectionBounds::from_anchor(IVec3::new(5 * 16, 5, 0));
        let edit = WorldEdit::fill(out_of_bounds, dirt());

        let err = commit_fill(&fixture.meta, &mut cache, &edit, &EditPolicy::default()).unwrap_err();
        assert!(matches!(
            err,
            WriteError::Refused(crate::edit::EditRefusal::ChunkNotGenerated { chunk: (5, 0) })
        ));
        assert_eq!(cache.dirty_regions().count(), 0);
    }

    // -----------------------------------------------------------------------------------------
    // ---- PaintCommand: the state machine -------------------------------------------------------
    // -----------------------------------------------------------------------------------------

    #[test]
    fn a_request_is_refused_while_another_is_pending() {
        let mut paint = PaintCommand::default();
        assert!(!paint.busy());
        assert!(paint.request(bounds(), dirt()));
        assert!(paint.busy());
        assert!(!paint.request(bounds(), dirt()));
    }

    /// Without a save loaded there's nothing to write to, and a click should
    /// say so rather than vanish (mirrors
    /// `blueprint::a_request_without_a_region_cache_reports_no_save_loaded`).
    #[test]
    fn a_request_without_a_save_reports_no_save_loaded() {
        let mut app = app();
        app.world_mut().resource_mut::<PaintCommand>().request(bounds(), dirt());
        app.update();

        let paint = app.world().resource::<PaintCommand>();
        assert!(!paint.busy(), "the request should have been consumed");
        assert!(matches!(paint.state(), PaintState::Failed { message } if message == "no save is loaded"));
    }

    /// The plumbing around a completed commit: a successful `WriteSummary`
    /// becomes `Done` and fires `ChunksEdited` with the written chunks — the
    /// hand-off into 034's reload queue that makes this ticket W7's proof as
    /// well as W1-W6's. The write itself is `commit_fill`'s test, above; this
    /// checks `poll_paint`'s transition and event, the same level
    /// `blueprint::export`'s `a_finished_write_reports_the_path_and_the_block_count`
    /// checks its own poller at.
    #[test]
    fn a_finished_commit_reports_done_and_fires_chunks_edited() {
        let mut app = app();
        app.world_mut().resource_mut::<PaintCommand>().state = PaintState::Writing {
            blocks: 9,
            task: pool().spawn(async {
                Ok(WriteSummary {
                    report: crate::edit::EditReport {
                        blocks_written: 9,
                        chunks: vec![(0, 0), (1, 0)],
                        regions: vec![(0, 0)],
                        replaced: None,
                    },
                    regions_written: vec![(0, 0)],
                    backups: vec![],
                })
            }),
        };
        run_until_settled(&mut app);

        let paint = app.world().resource::<PaintCommand>();
        assert!(matches!(
            paint.state(),
            PaintState::Done { blocks: 9, chunks: 2, regions: 1 }
        ));
        assert!(!paint.busy());

        let fired: Vec<_> =
            app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().collect();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].0, vec![(0, 0), (1, 0)]);
    }

    /// A write that fails (the world open in Minecraft, a refused edit, a
    /// disk error) reports and re-enables, and does *not* fire
    /// `ChunksEdited` — nothing changed, so nothing needs re-meshing.
    #[test]
    fn a_failed_commit_reports_and_does_not_fire_chunks_edited() {
        let mut app = app();
        app.world_mut().resource_mut::<PaintCommand>().state = PaintState::Writing {
            blocks: 9,
            task: pool().spawn(async {
                Err(WriteError::WorldIsOpen { save: "world".to_string() })
            }),
        };
        run_until_settled(&mut app);

        let paint = app.world().resource::<PaintCommand>();
        let PaintState::Failed { message } = paint.state() else {
            panic!("expected Failed, got another state");
        };
        assert!(message.contains("open in Minecraft"));
        assert!(!paint.busy());

        let fired = app.world_mut().resource_mut::<Events<ChunksEdited>>().drain().count();
        assert_eq!(fired, 0);
    }
}
