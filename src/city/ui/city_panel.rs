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

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::super::journal::Journal;
use super::super::state;
use super::super::undo::{UndoCommand, UndoState};
use super::super::write_status::{LastWrite, WriteKind, WriteStatus};

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
    }
}

/// The "last write" section: what happened, how many blocks/chunks/regions,
/// and where the backups went — the roadmap's own "last write, dirty
/// regions, backup location" in one block. `None` (nothing written this
/// session yet) is a plain message, the same tone every other empty state in
/// this crate's panels uses.
fn write_status_section(ui: &mut egui::Ui, last: Option<&LastWrite>) {
    let Some(last) = last else {
        ui.label("(nothing written to the world yet this session)");
        return;
    };

    match last {
        LastWrite::Success(record) => {
            ui.colored_label(
                WROTE_COLOR,
                format!(
                    "{} {}: {} block(s) across {} chunk(s), {} region file(s)",
                    kind_verb(record.kind),
                    record.building,
                    record.blocks,
                    record.chunks,
                    record.regions.len(),
                ),
            );
            if record.backups.is_empty() {
                ui.label("No new backups (already backed up earlier this session, or backups are off).");
            } else {
                ui.label(format!("Backed up {} region file(s):", record.backups.len()));
                for path in &record.backups {
                    ui.add(egui::Label::new(egui::RichText::new(path.display().to_string()).monospace().small()).wrap());
                }
            }
        }
        LastWrite::Failed { kind, building, message } => {
            ui.colored_label(egui::Color32::RED, format!("{} {} failed: {message}", kind_verb(*kind), building));
        }
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

/// Egui window: buildings, roads, write status, undo. See the module docs.
pub(super) fn city_panel(
    mut contexts: EguiContexts,
    city: Res<state::City>,
    journal: Res<Journal>,
    write_status: Res<WriteStatus>,
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
        ui.label(format!("{} tile(s)", city.roads().count()));

        ui.separator();
        ui.heading("Write status");
        write_status_section(ui, write_status.last());

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
    }
}
