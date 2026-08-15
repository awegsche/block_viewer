//! The export flow (ticket 024): the "Export…" button to a file on disk.
//!
//! Ticket 022 built the extraction and 023 the writer, but nothing joined
//! them — the button ran an extraction, logged its palette, and dropped the
//! blueprint. The console line it printed ("extracted 5x5x5 at …") reads like
//! a successful export, so the visible symptom was a click that claimed to do
//! something and wrote no file anywhere. This module is the missing half.
//!
//! ## One state, not a scatter of `Option<Task<_>>`
//!
//! An export is four sequential steps, three of which can fail or be
//! cancelled, so it's modelled as one [`ExportState`] rather than a handful
//! of independent option fields that can disagree about what's happening:
//!
//! ```text
//! Idle --request--> Choosing --Some(path)--> Extracting --> Writing --> Done
//!                      |                          |            |
//!                      +--None (cancelled)--> Idle |            +--> Failed
//!                                                  +----------------> Failed
//! ```
//!
//! The panel renders one line per state, and "the button is enabled in
//! exactly the states that aren't in flight" is a single `matches!` rather
//! than a rule spread across four fields.
//!
//! ## Why the dialog is async
//!
//! [`rfd::FileDialog::save_file`] is a blocking modal: called from an
//! `Update` system it stalls the render loop for as long as the dialog is
//! open, so the window stops repainting behind it and Windows may mark it
//! unresponsive. [`AsyncFileDialog`] hands back a future instead, which goes
//! on [`AsyncComputeTaskPool`] and gets polled with the same
//! `block_on(poll_once(..))` the extraction and the chunk pipeline use.
//!
//! (`rfd` also has platform threading rules — on macOS the dialog must run
//! on the main thread, which `AsyncFileDialog` handles internally. Windows is
//! the target here and wouldn't care, but sticking to the async API means the
//! constraint never has to be revisited.)
//!
//! ## Extract after the filename, not before
//!
//! Cancelling a save dialog is common and extraction is the expensive half,
//! so nothing is read out of the save until there's somewhere to put it.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use rfd::AsyncFileDialog;

use crate::selection::SelectionBounds;
use crate::LoadedSave;

use super::{write_structure_file, BlueprintExtraction, ExtractOutcome};

/// Where the game looks for structure files, relative to a save's root — and
/// so the dialog's starting directory when the save has one.
///
/// Only ever *checked*, never created: a viewer has no business making
/// folders inside someone's save, and `rfd`'s own dialog will happily create
/// one if the user wants it.
const STRUCTURES_SUBDIR: [&str; 3] = ["generated", "minecraft", "structures"];

/// The extension the dialog filters on and the one the default name carries.
const EXTENSION: &str = "nbt";

/// One export, from the click to the file.
///
/// [`Task`]s live in the variants that own them, so a state change is also
/// the point at which the previous step's task is dropped.
#[derive(Default)]
pub enum ExportState {
    /// Nothing in flight. The button is enabled.
    #[default]
    Idle,
    /// The save dialog is open. `bounds` is held across it so the extraction
    /// works from what was selected when the button was clicked, not from
    /// wherever the box has been moved to while the dialog sat open.
    Choosing {
        bounds: SelectionBounds,
        task: Task<Option<PathBuf>>,
    },
    /// A filename is chosen and [`BlueprintExtraction`] is reading the world.
    /// Progress comes from that resource, not this one.
    Extracting { path: PathBuf },
    /// The blueprint is being written to `path`.
    Writing { path: PathBuf, blocks: usize, task: Task<Result<(), String>> },
    /// The last export wrote `blocks` blocks to `path`. Sticks around until
    /// the next export starts — a status line that vanishes after one frame
    /// tells the user nothing.
    Done { path: PathBuf, blocks: usize },
    /// The last export failed. Also sticky, for the same reason.
    Failed { message: String },
}

impl ExportState {
    /// Whether an export is in flight. The three middle states, which is
    /// also exactly when the panel disables its button.
    ///
    /// [`Self::Done`] and [`Self::Failed`] are *not* busy: they're last
    /// export's result sitting on screen, not this one's progress.
    fn in_flight(&self) -> bool {
        matches!(
            self,
            ExportState::Choosing { .. } | ExportState::Extracting { .. } | ExportState::Writing { .. }
        )
    }
}

/// The one export slot, and the panel's whole view of the feature.
///
/// One at a time, like [`BlueprintExtraction`] — and more strictly, since two
/// native save dialogs at once isn't a thing anyone wants.
#[derive(Resource, Default)]
pub struct BlueprintExport {
    /// A click waiting for [`start_dialog`] to pick it up on the same frame.
    requested: Option<SelectionBounds>,
    state: ExportState,
}

impl BlueprintExport {
    /// Asks for `bounds` to be exported: opens the dialog on the next
    /// [`start_dialog`] run. Ignored (returning `false`) while an export is
    /// already in flight — the panel disables its button then, so this is a
    /// backstop rather than the normal path.
    pub fn request(&mut self, bounds: SelectionBounds) -> bool {
        if self.busy() {
            return false;
        }
        self.requested = Some(bounds);
        true
    }

    /// Whether a click is pending or an export is running.
    pub fn busy(&self) -> bool {
        self.requested.is_some() || self.state.in_flight()
    }

    /// What to report. The panel matches on this.
    pub fn state(&self) -> &ExportState {
        &self.state
    }
}

/// Adds [`BlueprintExport`] and the four systems that walk it through
/// [`ExportState`].
///
/// The chain is one frame's worth of the whole pipeline, ordered so a step
/// that finishes early in the frame is picked up by the next step in the
/// *same* frame rather than a frame later. It has to interleave with
/// [`super::BlueprintPlugin`]'s two systems, so both are registered here:
///
/// 1. `poll_extraction` — a finished extraction publishes its blueprint,
/// 2. [`drive_write`] — which this picks up and starts writing,
/// 3. [`poll_write`] — a finished write becomes `Done`/`Failed`,
/// 4. [`start_dialog`] — a click this frame opens its dialog,
/// 5. [`poll_dialog`] — a chosen filename requests an extraction,
/// 6. `start_extraction` — which this dispatches, same frame.
pub struct BlueprintExportPlugin;

impl Plugin for BlueprintExportPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BlueprintExport>().add_systems(
            Update,
            (
                super::poll_extraction,
                drive_write,
                poll_write,
                start_dialog,
                poll_dialog,
                super::start_extraction,
            )
                .chain(),
        );
    }
}

/// Opens the native save dialog for a requested export.
///
/// The dialog's configuration is the only place this feature knows anything
/// about the loaded save, which is why [`LoadedSave`] is read here rather
/// than being threaded through the panel.
fn start_dialog(
    mut export: ResMut<BlueprintExport>,
    loaded_save: Option<Res<LoadedSave>>,
) {
    let Some(bounds) = export.requested.take() else {
        return;
    };

    let mut dialog = AsyncFileDialog::new()
        .add_filter("Minecraft structure", &[EXTENSION])
        .set_file_name(default_file_name(bounds.min));
    // No save loaded is not an error here — the extraction will say so in a
    // moment. The dialog just opens wherever `rfd` would have anyway.
    if let Some(directory) = loaded_save
        .as_ref()
        .and_then(|save| default_directory(&save.0.meta.path))
    {
        dialog = dialog.set_directory(directory);
    }

    let file = dialog.save_file();
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { file.await.map(|handle| handle.path().to_path_buf()) });

    export.state = ExportState::Choosing { bounds, task };
}

/// Polls the open dialog. A chosen path starts the extraction; a cancel goes
/// quietly back to [`ExportState::Idle`] without leaving a status line
/// behind, because cancelling is a decision rather than a result.
fn poll_dialog(
    mut export: ResMut<BlueprintExport>,
    mut extraction: ResMut<BlueprintExtraction>,
) {
    let ExportState::Choosing { bounds, task } = &mut export.state else {
        return;
    };
    let Some(chosen) = block_on(poll_once(task)) else {
        return; // Dialog still open.
    };

    let bounds = *bounds;
    export.state = match chosen {
        None => ExportState::Idle,
        Some(path) => {
            if extraction.request(bounds) {
                ExportState::Extracting { path }
            } else {
                // Only reachable if something else drove the extraction
                // directly while the dialog was open; the panel can't.
                ExportState::Failed {
                    message: "another extraction is already running".to_string(),
                }
            }
        }
    };
}

/// Hands a finished extraction to the writer.
///
/// Waits on [`BlueprintExtraction`] rather than owning the extraction task
/// itself: extraction is a whole feature with its own progress reporting and
/// its own one-at-a-time rule, and duplicating that here would be a second
/// copy of both.
fn drive_write(
    mut export: ResMut<BlueprintExport>,
    mut extraction: ResMut<BlueprintExtraction>,
) {
    let ExportState::Extracting { path } = &export.state else {
        return;
    };
    if extraction.busy() {
        return;
    }
    let path = path.clone();

    let Some(blueprint) = extraction.take_blueprint() else {
        // No blueprint and not running means it failed. The extraction's own
        // error is the useful message; the fallback is for the case where
        // something consumed the outcome first, which nothing does today.
        let message = match extraction.last() {
            Some(ExtractOutcome::Failed(err)) => err.to_string(),
            _ => "the extraction produced nothing".to_string(),
        };
        println!("block_viewer: export failed: {message}");
        export.state = ExportState::Failed { message };
        return;
    };

    let blocks = blueprint.volume();
    let write_path = path.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move {
        // The writer's `io::Error` alone says "Access is denied" without
        // saying to what, and the path is the half the user can act on.
        write_structure_file(&write_path, &blueprint)
            .map_err(|err| format!("{}: {err}", write_path.display()))
    });

    export.state = ExportState::Writing { path, blocks, task };
}

/// Single non-blocking poll of the write, the last step.
fn poll_write(mut export: ResMut<BlueprintExport>) {
    let ExportState::Writing { path, blocks, task } = &mut export.state else {
        return;
    };
    let Some(result) = block_on(poll_once(task)) else {
        return; // Still writing.
    };

    let (path, blocks) = (path.clone(), *blocks);
    export.state = match result {
        Ok(()) => {
            println!(
                "block_viewer: exported {blocks} blocks to {}",
                path.display()
            );
            ExportState::Done { path, blocks }
        }
        Err(message) => {
            println!("block_viewer: export failed: {message}");
            ExportState::Failed { message }
        }
    };
}

/// The filename the dialog opens with: `blueprint_<x>_<y>_<z>.nbt` from the
/// selection's minimum corner, so consecutive exports don't all default to
/// the same name and overwrite each other by reflex.
///
/// Minecraft coordinates go negative, and `-` is the only character this can
/// produce beyond digits and `_` — all three are fine in a Windows filename,
/// which `:`, `<`, `>` and the rest are not.
fn default_file_name(min: IVec3) -> String {
    format!("blueprint_{}_{}_{}.{EXTENSION}", min.x, min.y, min.z)
}

/// Where the dialog opens: the save's own `generated/minecraft/structures/`
/// if the game has made one, else the save directory, else nowhere in
/// particular (`rfd` then uses its own default).
///
/// Checks rather than creates, deliberately — see [`STRUCTURES_SUBDIR`]. The
/// empty-path case is [`crate::empty_save`]'s, which has no directory at all.
fn default_directory(save_root: &Path) -> Option<PathBuf> {
    if save_root.as_os_str().is_empty() {
        return None;
    }

    let structures = STRUCTURES_SUBDIR
        .iter()
        .fold(save_root.to_path_buf(), |path, part| path.join(part));
    if structures.is_dir() {
        return Some(structures);
    }
    save_root.is_dir().then(|| save_root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::tasks::TaskPool;
    use std::fs;

    /// A unique scratch directory, the same way `structure.rs`'s round-trip
    /// tests get one.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_export_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn bounds() -> SelectionBounds {
        SelectionBounds::from_anchor(IVec3::new(1, 64, -3))
    }

    /// The tasks below are spawned by hand rather than by the systems, so the
    /// pool has to exist without `TaskPoolPlugin` having run.
    fn pool() -> &'static AsyncComputeTaskPool {
        AsyncComputeTaskPool::get_or_init(TaskPool::default)
    }

    fn app() -> App {
        let mut app = App::new();
        // The whole feature, since the export's four systems and the
        // extraction's two are one chain — see `BlueprintExportPlugin`.
        app.add_plugins(super::super::BlueprintPlugin);
        app
    }

    #[test]
    fn the_default_name_carries_the_selections_corner() {
        assert_eq!(
            default_file_name(IVec3::new(12, 64, 30)),
            "blueprint_12_64_30.nbt"
        );
    }

    /// Coordinates below zero and below Y 0 are ordinary in Minecraft, and
    /// the name they produce has to stay a legal filename.
    #[test]
    fn the_default_name_survives_negative_coordinates() {
        let name = default_file_name(IVec3::new(-1500, -59, -30000000));
        assert_eq!(name, "blueprint_-1500_-59_-30000000.nbt");
        assert!(
            !name.contains(['<', '>', ':', '"', '/', '\\', '|', '?', '*']),
            "{name} is not a legal Windows filename"
        );
    }

    /// The structures folder wins when the game has made one.
    #[test]
    fn the_dialog_opens_in_the_saves_structures_folder_when_it_exists() {
        let root = scratch("structures");
        let structures = root.join("generated").join("minecraft").join("structures");
        fs::create_dir_all(&structures).unwrap();

        assert_eq!(default_directory(&root), Some(structures));
    }

    /// Most saves have never had a structure block saved in them, so the
    /// common case is the fallback.
    #[test]
    fn without_a_structures_folder_it_opens_in_the_save_itself() {
        let root = scratch("no_structures");
        assert_eq!(default_directory(&root), Some(root.clone()));

        // And having looked, it must not have made one — a viewer doesn't
        // write folders into someone's save.
        assert!(!root.join("generated").exists());
    }

    /// `empty_save`'s path, and a save directory that has since been deleted:
    /// neither is a directory to open a dialog in.
    #[test]
    fn an_absent_save_directory_leaves_the_dialog_to_its_own_default() {
        assert_eq!(default_directory(Path::new("")), None);

        let missing = scratch("missing").join("gone");
        assert_eq!(default_directory(&missing), None);
    }

    #[test]
    fn a_request_is_refused_while_an_export_is_in_flight() {
        let mut export = BlueprintExport::default();
        assert!(!export.busy());
        assert!(export.request(bounds()));
        assert!(export.busy(), "a pending click counts as busy");
        assert!(!export.request(bounds()));
    }

    /// Cancelling is a decision, not a result: back to `Idle` with nothing
    /// written, nothing extracted and the button enabled again.
    #[test]
    fn cancelling_the_dialog_returns_to_idle() {
        let mut app = app();
        app.world_mut().resource_mut::<BlueprintExport>().state = ExportState::Choosing {
            bounds: bounds(),
            task: pool().spawn(async { None }),
        };
        app.update();

        let export = app.world().resource::<BlueprintExport>();
        assert!(matches!(export.state(), ExportState::Idle));
        assert!(!export.busy(), "the button must be enabled again");
        assert!(!app.world().resource::<BlueprintExtraction>().busy());
    }

    /// A chosen filename is what starts the extraction — the expensive half
    /// deliberately doesn't run until there's somewhere to put it.
    #[test]
    fn choosing_a_filename_starts_the_extraction() {
        let path = scratch("chosen").join("blueprint.nbt");
        let mut app = app();
        app.world_mut().resource_mut::<BlueprintExport>().state = ExportState::Choosing {
            bounds: bounds(),
            task: pool().spawn({
                let path = path.clone();
                async move { Some(path) }
            }),
        };
        app.update();

        let export = app.world().resource::<BlueprintExport>();
        assert!(
            matches!(export.state(), ExportState::Extracting { path: p } if *p == path),
            "a chosen path moves to Extracting, holding onto the path"
        );
        assert!(export.busy());
    }

    /// With no save loaded the extraction fails immediately, and that failure
    /// has to surface as the *export's* failure rather than stranding the
    /// state machine in `Extracting` forever.
    #[test]
    fn an_extraction_failure_becomes_an_export_failure() {
        let mut app = app();
        app.world_mut().resource_mut::<BlueprintExport>().state = ExportState::Choosing {
            bounds: bounds(),
            task: pool().spawn(async { Some(PathBuf::from("unused.nbt")) }),
        };

        // Frame one chooses the path and dispatches the extraction, which
        // fails on the spot for want of a region cache; frame two picks that
        // up.
        app.update();
        app.update();

        let export = app.world().resource::<BlueprintExport>();
        let ExportState::Failed { message } = export.state() else {
            panic!("expected Failed, got another state");
        };
        assert_eq!(message, "no save is loaded");
        assert!(!export.busy(), "a failure re-enables the button");
    }

    /// A write that fails (a read-only location, a full disk) reports and
    /// re-enables, rather than panicking the way ticket 008 ruled out.
    #[test]
    fn a_write_failure_reports_and_re_enables_the_button() {
        let mut app = app();
        app.world_mut().resource_mut::<BlueprintExport>().state = ExportState::Writing {
            path: PathBuf::from("C:\\nope\\blueprint.nbt"),
            blocks: 27,
            task: pool().spawn(async { Err("C:\\nope\\blueprint.nbt: Access is denied".to_string()) }),
        };
        app.update();

        let export = app.world().resource::<BlueprintExport>();
        let ExportState::Failed { message } = export.state() else {
            panic!("expected Failed, got another state");
        };
        assert!(message.contains("Access is denied"));
        assert!(!export.busy());
    }

    /// The success end of the same path — and the block count survives to the
    /// status line, which is the number the user checks the export by.
    #[test]
    fn a_finished_write_reports_the_path_and_the_block_count() {
        let path = scratch("done").join("blueprint.nbt");
        let mut app = app();
        app.world_mut().resource_mut::<BlueprintExport>().state = ExportState::Writing {
            path: path.clone(),
            blocks: 27,
            task: pool().spawn(async { Ok(()) }),
        };
        app.update();

        let export = app.world().resource::<BlueprintExport>();
        assert!(
            matches!(export.state(), ExportState::Done { path: p, blocks: 27 } if *p == path)
        );
        assert!(!export.busy(), "a finished export re-enables the button");
    }

    /// Both terminal states are sticky: the result stays on screen until the
    /// next export replaces it, rather than flashing for one frame.
    #[test]
    fn a_finished_export_stays_reported_across_frames() {
        let mut app = app();
        app.world_mut().resource_mut::<BlueprintExport>().state = ExportState::Done {
            path: PathBuf::from("blueprint.nbt"),
            blocks: 27,
        };
        for _ in 0..3 {
            app.update();
        }
        assert!(matches!(
            app.world().resource::<BlueprintExport>().state(),
            ExportState::Done { .. }
        ));
    }
}
