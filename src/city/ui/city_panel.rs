//! The city panel (ticket 050, roadmap G2): "building counts, road length,
//! and the write status — last write, dirty regions, backup location. That
//! last part matters more than it sounds: the user needs to know whether
//! what they see has actually reached the world" — the roadmap's own
//! justification, quoted because it's the reason this panel exists rather
//! than a nice-to-have.
//!
//! Also carries the "Undo" button [`super::super::journal`]'s own module
//! docs have been pointing at since ticket 044 — [`super::super::undo`] is
//! the write path behind it; this panel is only the button and the status
//! line.
//!
//! Ticket 051 split "write status" into two sections: [`write_status_section`]
//! now shows the last placement/demolition/undo *applied to memory*, and
//! [`save_section`] is new — the "Save world" button and the last actual
//! disk write's own result, plus how many regions are currently dirty (read
//! straight off the shared [`crate::region_cache::RegionCache`] via a
//! `try_lock`, since it's also being written to by whichever of
//! commit/demolish/undo/save is mid-apply; a UI count one frame stale from a
//! contended lock is harmless, so this skips the count that frame rather
//! than blocking on it).

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::chunk_pipeline::SharedRegionCache;

use super::super::journal::Journal;
use super::super::save::{SaveCommand, SaveState};
use super::super::state;
use super::super::undo::{UndoCommand, UndoState};
use super::super::write_status::{LastSave, LastWrite, WriteKind, WriteStatus};

/// A written file's status line is green rather than the default text
/// colour — the same call `viewer::ui::selection_panel`'s own `WROTE_COLOR`
/// makes for the same reason, duplicated rather than shared across two
/// otherwise-independent UI modules for one constant.
const WROTE_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 220, 120);

/// How many buildings of each definition id are currently placed, sorted by
/// id — [`state::City::buildings`] has no grouping of its own, and a menu
/// this small doesn't need a second resource to hold one. A plain function,
/// not inlined into [`city_panel`], so the grouping is directly testable
/// without an `egui::Context` in the loop.
fn building_counts(city: &state::City) -> Vec<(String, usize)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, building) in city.buildings() {
        *counts.entry(building.definition.as_str()).or_insert(0) += 1;
    }
    let mut counts: Vec<(String, usize)> = counts.into_iter().map(|(id, n)| (id.to_string(), n)).collect();
    counts.sort_by(|a, b| a.0.cmp(&b.0));
    counts
}

/// [`WriteKind`] as the verb its status line reads with — "Placed
/// house01: 40 blocks...", not "WriteKind::Placed".
fn kind_verb(kind: WriteKind) -> &'static str {
    match kind {
        WriteKind::Placed => "Placed",
        WriteKind::Demolished => "Demolished",
        WriteKind::Undo => "Undid",
        WriteKind::Road => "Built",
        WriteKind::Terraform => "Shaped",
    }
}

/// The "Last edit" section: what happened, how many blocks/chunks/regions —
/// applied to the shared region cache, not yet written to disk (ticket 051;
/// see [`save_section`] for the disk side). `None` (nothing applied this
/// session yet) is a plain message, the same tone every other empty state in
/// this crate's panels uses.
fn write_status_section(ui: &mut egui::Ui, last: Option<&LastWrite>) {
    let Some(last) = last else {
        ui.label("(nothing built or demolished yet this session)");
        return;
    };

    match last {
        LastWrite::Success(record) => {
            ui.colored_label(
                WROTE_COLOR,
                format!(
                    "{} {}: {} block(s) across {} chunk(s), {} region file(s) — not yet saved to disk",
                    kind_verb(record.kind),
                    record.building,
                    record.blocks,
                    record.chunks,
                    record.regions.len(),
                ),
            );
        }
        LastWrite::Failed { kind, building, message } => {
            ui.colored_label(egui::Color32::RED, format!("{} {} failed: {message}", kind_verb(*kind), building));
        }
    }
}

/// The "World save" section: the roadmap's own "dirty regions, backup
/// location" half, plus the button that actually flushes them. The dirty
/// count is read straight off the shared region cache, `try_lock`ed rather
/// than blocked on — a save or an in-flight commit/demolish/undo can hold
/// the same mutex, and a stale count for one frame is harmless where
/// blocking the UI thread on it would not be.
fn save_section(ui: &mut egui::Ui, region_cache: Option<&SharedRegionCache>, last_save: Option<&LastSave>, save: &mut SaveCommand) {
    let dirty = region_cache.and_then(|cache| cache.0.try_lock().ok().map(|cache| cache.dirty_regions().count()));

    match dirty {
        Some(0) => ui.label("(nothing unsaved)"),
        Some(n) => ui.label(format!("{n} region file(s) with unsaved changes")),
        None => ui.label("(unsaved-region count unavailable right now)"),
    };

    let response = ui.add_enabled(!save.busy(), egui::Button::new("Save world"));
    if response.clicked() {
        save.request();
    }
    response.on_disabled_hover_text(if save.busy() { "Already saving." } else { "" });

    match save.state() {
        SaveState::Idle => {}
        SaveState::Saving => {
            ui.label("Saving…");
            ui.spinner();
        }
        SaveState::Done { regions, backups } => {
            ui.colored_label(WROTE_COLOR, format!("Saved {regions} region file(s), {backups} new backup(s)."));
        }
        SaveState::Failed { message } => {
            ui.colored_label(egui::Color32::RED, format!("Save failed: {message}"));
        }
    }

    // `SaveState` above already says whether *this* click succeeded or
    // failed; `last_save` is the roadmap's own "dirty regions, backup
    // location" — which files the most recent successful save actually
    // touched and where the originals went, which stays worth showing even
    // once `SaveState` has moved on to a later click's own `Saving`/`Failed`.
    match last_save {
        Some(LastSave::Success(record)) => {
            ui.label(format!("Last save wrote {} region file(s).", record.regions.len()));
            if !record.backups.is_empty() {
                ui.label("Backup location(s):");
                for path in &record.backups {
                    ui.add(egui::Label::new(egui::RichText::new(path.display().to_string()).monospace().small()).wrap());
                }
            }
        }
        Some(LastSave::Failed { message }) => {
            ui.label(format!("Last save attempt: {message}"));
        }
        None => {}
    }
}

/// The "Undo" section: a button naming how many actions are in the journal,
/// disabled while a write of any kind is in flight, plus the last undo's own
/// result.
fn undo_section(ui: &mut egui::Ui, journal: &Journal, undo: &mut UndoCommand) {
    if journal.is_empty() {
        ui.label("(nothing to undo)");
    } else {
        let response =
            ui.add_enabled(!undo.busy(), egui::Button::new(format!("Undo ({} action(s) in the journal)", journal.len())));
        if response.clicked() {
            undo.request();
        }
        response.on_disabled_hover_text(if undo.busy() { "Already undoing, or another write is in progress." } else { "" });
    }

    match undo.state() {
        UndoState::Idle => {}
        UndoState::Writing => {
            ui.label("Undoing…");
            ui.spinner();
        }
        UndoState::Done { definition, kind } => {
            ui.colored_label(WROTE_COLOR, format!("Undid {kind} {definition}."));
        }
        UndoState::Failed { message } => {
            ui.colored_label(egui::Color32::RED, format!("Undo failed: {message}"));
        }
    }
}

/// Egui window: buildings, roads, write status, world save, undo. See the
/// module docs.
pub(super) fn city_panel(
    mut contexts: EguiContexts,
    city: Res<state::City>,
    journal: Res<Journal>,
    write_status: Res<WriteStatus>,
    region_cache: Option<Res<SharedRegionCache>>,
    mut save: ResMut<SaveCommand>,
    mut undo: ResMut<UndoCommand>,
) {
    egui::Window::new("City").show(contexts.ctx_mut(), |ui| {
        ui.heading("Buildings");
        if city.is_empty() {
            ui.label("(nothing built yet)");
        } else {
            ui.label(format!("Total: {}", city.len()));
            for (definition, count) in building_counts(&city) {
                ui.label(format!("  {definition}: {count}"));
            }
        }

        ui.separator();
        ui.heading("Roads");
        ui.label(format!("{} cell(s)", city.road_cells().count()));

        ui.separator();
        ui.heading("Last edit");
        write_status_section(ui, write_status.last());

        ui.separator();
        ui.heading("World save");
        save_section(ui, region_cache.as_deref(), write_status.last_save(), &mut save);

        ui.separator();
        ui.heading("Undo");
        undo_section(ui, &journal, &mut undo);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::Rotation;
    use bevy::math::{IVec2, IVec3};

    #[test]
    fn no_buildings_is_an_empty_list() {
        let city = state::City::default();
        assert!(building_counts(&city).is_empty());
    }

    #[test]
    fn buildings_are_grouped_by_definition_and_sorted() {
        let mut city = state::City::default();
        city.place_building("house01", IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        city.place_building("house01", IVec3::new(2, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
        city.place_building("carpenter", IVec3::new(4, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();

        let counts = building_counts(&city);
        assert_eq!(counts, vec![("carpenter".to_string(), 1), ("house01".to_string(), 2)]);
    }

    #[test]
    fn kind_verbs_read_as_a_sentence() {
        assert_eq!(kind_verb(WriteKind::Placed), "Placed");
        assert_eq!(kind_verb(WriteKind::Demolished), "Demolished");
        assert_eq!(kind_verb(WriteKind::Undo), "Undid");
        assert_eq!(kind_verb(WriteKind::Road), "Built");
        assert_eq!(kind_verb(WriteKind::Terraform), "Shaped");
    }
}
