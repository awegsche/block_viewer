//! Selection panel (ticket 021): what's selected, as six editable
//! coordinates, plus the size/volume readout that decides whether exporting
//! it is a good idea — and the legend for ticket 020's keys, which are
//! undiscoverable otherwise.
//!
//! This is the one place [`crate::ui`] and [`crate::selection`] meet: panels
//! live here, selection state lives there, and this module reads and writes
//! [`Selection`] without either side growing a dependency on the other's
//! internals.
//!
//! The bounds fields follow [`super::navigate`]'s typed-coordinate pattern:
//! [`String`] scratch committed on `Enter` or a button, never parsed straight
//! into the bounds every frame — otherwise a half-typed `-` or an emptied
//! field would jump the box to zero mid-keystroke. The scratch is re-filled
//! from [`Selection`] whenever the box moves by any other route (a click, a
//! face key, `Escape`), so the readout stays live without the fields fighting
//! what's being typed into them.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::blueprint::{BlueprintExtraction, ExtractOutcome, MAX_BLOCKS};
use crate::selection::{Face, Selection, SelectionBounds, CHUNK_STEP};

/// Above this many blocks the volume line is coloured and the panel says an
/// export will be slow: 1,000,000 = 100x100x100.
///
/// Conservative against ticket 022's measured extraction cost — 2,097,152
/// blocks over 64 chunk columns take ~175 ms including the region load
/// (`cargo test measure_large_extraction -- --nocapture`), so a million
/// blocks is under a tenth of a second, not "slow". Left where it is until
/// ticket 023 measures the *write* half, which is the other thing this
/// warning is about; the panel now reports each extraction's elapsed time,
/// so the number to re-tune against is on screen.
const VOLUME_WARN: u64 = 1_000_000;

/// Above this many blocks the export button is disabled outright.
///
/// The extraction's own hard limit, not a second copy of it: ticket 022
/// refuses anything larger, and a panel that offered a button for a
/// selection the extractor would reject outright would be lying about it.
const VOLUME_CAP: u64 = MAX_BLOCKS;

/// 48x48x48 = 110,592 blocks: the largest structure a *vanilla* structure
/// block can load. Well below [`VOLUME_WARN`] on purpose — this app's writer
/// (ticket 023) will happily emit a bigger `.nbt` than Minecraft will load
/// back, so it's worth saying out loud in the panel rather than warning on.
const STRUCTURE_BLOCK_LIMIT: u64 = 48 * 48 * 48;

/// Wide enough for `-30000000` at egui's default body font.
const FIELD_WIDTH: f32 = 72.0;

/// Where a selection's volume falls against the two thresholds above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VolumeClass {
    /// Small enough that the export is unremarkable.
    Fine,
    /// Exportable, but slow enough to warn about.
    Slow,
    /// Past [`VOLUME_CAP`] — the export button is disabled.
    OverCap,
}

/// Classifies `volume` against [`VOLUME_WARN`]/[`VOLUME_CAP`]. Both
/// thresholds are *exclusive*: a selection of exactly 1,000,000 blocks is
/// still fine, and one of exactly 16,000,000 still exports.
fn classify_volume(volume: u64) -> VolumeClass {
    if volume > VOLUME_CAP {
        VolumeClass::OverCap
    } else if volume > VOLUME_WARN {
        VolumeClass::Slow
    } else {
        VolumeClass::Fine
    }
}

/// The six coordinate text fields, plus the bounds they were last filled
/// from so [`Self::sync`] can tell "the box moved under us" (re-fill) from
/// "someone is typing" (leave alone).
#[derive(Default)]
pub(crate) struct BoundsDraft {
    min: [String; 3],
    max: [String; 3],
    /// What [`Selection`] held when the fields were last filled. `None` both
    /// before the first sync and while nothing is selected — the two want the
    /// same behaviour (empty fields), so they don't need telling apart.
    synced: Option<SelectionBounds>,
}

impl BoundsDraft {
    /// Re-fills the fields from `selection` if it has changed since they were
    /// last filled — which includes the frame *after* a commit, so typed
    /// values come back normalized and Y-clamped rather than as typed.
    fn sync(&mut self, selection: &Selection) {
        if self.synced == selection.0 {
            return;
        }
        self.synced = selection.0;
        match selection.0 {
            Some(bounds) => {
                self.min = axis_strings(bounds.min);
                self.max = axis_strings(bounds.max);
            }
            None => {
                self.min = Default::default();
                self.max = Default::default();
            }
        }
    }
}

fn axis_strings(v: IVec3) -> [String; 3] {
    [v.x.to_string(), v.y.to_string(), v.z.to_string()]
}

/// Parses one corner's three fields. `None` if any of them isn't a whole
/// number — including empty, which is what a field mid-edit looks like.
fn parse_corner(fields: &[String; 3]) -> Option<IVec3> {
    let mut parsed = [0i32; 3];
    for (out, field) in parsed.iter_mut().zip(fields) {
        *out = field.trim().parse().ok()?;
    }
    Some(IVec3::from(parsed))
}

/// Turns the six typed fields into new bounds, keeping `anchor` where it was
/// (the anchor is the *clicked* block — typing bounds doesn't re-click
/// anything). `None` if any field fails to parse, in which case the caller
/// leaves the selection untouched.
///
/// Goes through [`SelectionBounds::from_corners`], so an inverted range
/// (`min` typed above `max` on some axis) normalizes itself per axis instead
/// of producing an invalid box, and Y is clamped to the build limits.
fn commit_typed_bounds(
    anchor: IVec3,
    min: &[String; 3],
    max: &[String; 3],
) -> Option<SelectionBounds> {
    Some(SelectionBounds::from_corners(
        anchor,
        parse_corner(min)?,
        parse_corner(max)?,
    ))
}

/// `1234567` as `1,234,567`. Volumes are the numbers this panel exists to
/// make judgeable, and seven unbroken digits aren't.
fn format_blocks(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Egui window: the selection's bounds (editable), its size and volume, the
/// anchor, and the export affordance the rest of the group builds toward.
pub(crate) fn selection_panel(
    mut contexts: EguiContexts,
    mut selection: ResMut<Selection>,
    mut extraction: ResMut<BlueprintExtraction>,
    mut draft: Local<BoundsDraft>,
) {
    egui::Window::new("Selection").show(contexts.ctx_mut(), |ui| {
        draft.sync(&selection);

        match selection.0 {
            // Same tone as the block inspector's "(hover the world to
            // inspect a block)" — the gesture, not an error.
            None => {
                ui.label("(click a block in the world to start a selection)");
            }
            Some(bounds) => {
                let mut commit = false;
                ui.horizontal(|ui| {
                    ui.label("Min");
                    commit |= coord_fields(ui, &mut draft.min);
                });
                ui.horizontal(|ui| {
                    ui.label("Max");
                    commit |= coord_fields(ui, &mut draft.max);
                });
                if parse_corner(&draft.min).is_none() || parse_corner(&draft.max).is_none() {
                    ui.colored_label(egui::Color32::RED, "Bounds must be whole numbers.");
                }

                ui.horizontal(|ui| {
                    commit |= ui.button("Set bounds").clicked();
                    // Same effect as ticket 020's `Escape`, for a hand that's
                    // on the mouse.
                    if ui.button("Clear").clicked() {
                        selection.0 = None;
                    }
                });

                if commit {
                    // A failed parse deliberately does nothing at all: the
                    // red label above already says why, and silently moving
                    // the box to whatever *did* parse would be worse.
                    if let Some(typed) =
                        commit_typed_bounds(bounds.anchor, &draft.min, &draft.max)
                    {
                        selection.0 = Some(typed);
                    }
                }

                // Re-read rather than using `bounds`: a commit or a Clear
                // this frame should show its own result, not the state the
                // panel opened the frame with.
                if let Some(bounds) = selection.0 {
                    ui.separator();
                    readout(ui, &bounds, &mut extraction);
                }
            }
        }

        // Outside the `match`: an extraction keeps running (and keeps
        // reporting) even if the selection it was started from is cleared or
        // moved while it's in flight — it works from a snapshot of the bounds.
        extraction_status(ui, &extraction);

        ui.separator();
        key_legend(ui);
    });
}

/// One corner's `X`/`Y`/`Z` fields. Returns whether `Enter` was pressed in
/// any of them — egui reports that as "lost focus this frame *and* Enter went
/// down", there being no `TextEdit` submit event.
fn coord_fields(ui: &mut egui::Ui, fields: &mut [String; 3]) -> bool {
    let mut entered = false;
    for (label, field) in ["X", "Y", "Z"].into_iter().zip(fields.iter_mut()) {
        ui.label(label);
        let response = ui.add(egui::TextEdit::singleline(field).desired_width(FIELD_WIDTH));
        entered |= response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
    }
    entered
}

/// Size, volume, anchor, and the export button — the read-only half.
fn readout(ui: &mut egui::Ui, bounds: &SelectionBounds, extraction: &mut BlueprintExtraction) {
    let size = bounds.size();
    ui.label(format!("Size: {} x {} x {}", size.x, size.y, size.z));

    let volume = bounds.volume();
    let class = classify_volume(volume);
    let volume_text = format!("Volume: {} blocks", format_blocks(volume));
    let volume_label = match class {
        VolumeClass::Fine => ui.label(volume_text),
        VolumeClass::Slow => ui.colored_label(egui::Color32::YELLOW, volume_text),
        VolumeClass::OverCap => ui.colored_label(egui::Color32::RED, volume_text),
    };
    volume_label.on_hover_text(format!(
        "A vanilla structure block loads at most 48 x 48 x 48 = {} blocks. \
         Bigger blueprints still write, but Minecraft won't load them back.",
        format_blocks(STRUCTURE_BLOCK_LIMIT)
    ));

    match class {
        VolumeClass::Fine => {}
        VolumeClass::Slow => {
            ui.colored_label(
                egui::Color32::YELLOW,
                format!(
                    "Over {} blocks — exporting this will be slow.",
                    format_blocks(VOLUME_WARN)
                ),
            );
        }
        VolumeClass::OverCap => {
            ui.colored_label(
                egui::Color32::RED,
                format!(
                    "Over {} blocks — too large to export.",
                    format_blocks(VOLUME_CAP)
                ),
            );
        }
    }

    let anchor = bounds.anchor;
    ui.label(format!("Anchor: {}, {}, {}", anchor.x, anchor.y, anchor.z));

    // Ticket 022 makes this run the extraction and log what it found; 023
    // gives it a file format and 024 a save dialog to pick a filename with.
    let busy = extraction.busy();
    let response = ui.add_enabled(
        class != VolumeClass::OverCap && !busy,
        egui::Button::new("Export…"),
    );
    if response.clicked() {
        extraction.request(*bounds);
    }
    response
        .on_hover_text("Extract these blocks and log the palette (no file yet — ticket 024).")
        .on_disabled_hover_text(if busy {
            "An extraction is already running."
        } else {
            "Selection is over the export cap."
        });
}

/// The running extraction's progress, or the last one's result. Nothing at
/// all before the first export — an idle line here would just be noise in a
/// panel that already has plenty.
fn extraction_status(ui: &mut egui::Ui, extraction: &BlueprintExtraction) {
    if let Some((done, total)) = extraction.progress() {
        ui.separator();
        ui.label(format!("Extracting… {done} / {total} chunk columns"));
        // The whole reason the task publishes a column count: a
        // multi-second extraction with no feedback reads as a hung window.
        ui.add(egui::ProgressBar::new(extraction.fraction().unwrap_or(0.0)).show_percentage());
        return;
    }

    let Some(outcome) = extraction.last() else {
        return;
    };
    ui.separator();
    match outcome {
        ExtractOutcome::Done(summary) => {
            ui.label(format!(
                "Extracted {} blocks, {} distinct states in {:.2?}",
                format_blocks(summary.blocks as u64),
                summary.palette,
                summary.elapsed
            ));
            ui.label(format!(
                "  {} x {} x {} at {}, {}, {} — palette logged to the console",
                summary.size.x,
                summary.size.y,
                summary.size.z,
                summary.origin.x,
                summary.origin.y,
                summary.origin.z,
            ));
            if summary.failed_columns > 0 {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!(
                        "{} chunk column(s) could not be read — those blocks are air.",
                        summary.failed_columns
                    ),
                );
            }
        }
        ExtractOutcome::Failed(err) => {
            ui.colored_label(egui::Color32::RED, format!("Export failed: {err}"));
        }
    }
}

/// Ticket 020's bindings, generated from [`Face`] itself. A legend retyped as
/// strings drifts from the code and is then worse than no legend; this one
/// can only go wrong by a face being removed.
///
/// Collapsed by default: useful exactly once per person, in the way.
fn key_legend(ui: &mut egui::Ui) {
    egui::CollapsingHeader::new("Keys").show(ui, |ui| {
        egui::Grid::new("selection_key_legend")
            .num_columns(2)
            .show(ui, |ui| {
                for face in Face::ALL {
                    ui.label(face.key_label());
                    ui.label(format!("push the {} face outward", face.label()));
                    ui.end_row();
                }
                ui.label("Alt +");
                ui.label("pull that face back in instead");
                ui.end_row();
                ui.label("Ctrl +");
                ui.label(format!("step {CHUNK_STEP} blocks (one chunk) at a time"));
                ui.end_row();
                ui.label("Esc");
                ui.label("clear the selection");
                ui.end_row();
            });
        ui.label("Directions are world-absolute — ↑ is north whichever way the camera faces.");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(a: i32, b: i32, c: i32) -> [String; 3] {
        [a.to_string(), b.to_string(), c.to_string()]
    }

    /// The ticket's headline edit case: type a `min` above a `max` and the
    /// pair sorts itself out per axis instead of producing an inverted box.
    #[test]
    fn committing_an_inverted_pair_normalizes_it() {
        let anchor = IVec3::new(1, 64, 1);
        // Inverted on X and Z, correct on Y — a whole-vector swap would
        // "fix" X and Z at Y's expense.
        let bounds = commit_typed_bounds(anchor, &fields(10, 60, 8), &fields(2, 70, -4)).unwrap();

        assert_eq!(bounds.min, IVec3::new(2, 60, -4));
        assert_eq!(bounds.max, IVec3::new(10, 70, 8));
        assert_eq!(bounds.anchor, anchor, "typing bounds must not move the anchor");
    }

    #[test]
    fn committing_a_normal_pair_keeps_it_as_typed() {
        let bounds =
            commit_typed_bounds(IVec3::ZERO, &fields(-5, 0, -5), &fields(5, 10, 5)).unwrap();
        assert_eq!(bounds.min, IVec3::new(-5, 0, -5));
        assert_eq!(bounds.max, IVec3::new(5, 10, 5));
        assert_eq!(bounds.size(), IVec3::new(11, 11, 11));
    }

    /// Anything that isn't a whole number leaves the caller with nothing to
    /// apply, so the previous bounds stand.
    #[test]
    fn non_numeric_input_commits_nothing() {
        let good = fields(0, 64, 0);
        for bad in ["", " ", "-", "12.5", "1e3", "abc", "12a", "+"] {
            let mut corner = good.clone();
            corner[1] = bad.to_string();
            assert!(
                commit_typed_bounds(IVec3::ZERO, &corner, &good).is_none(),
                "min {bad:?} should not commit"
            );
            assert!(
                commit_typed_bounds(IVec3::ZERO, &good, &corner).is_none(),
                "max {bad:?} should not commit"
            );
        }
    }

    /// Surrounding whitespace is a paste artefact, not a typo — the
    /// coordinate-jump panel is equally forgiving about it.
    #[test]
    fn surrounding_whitespace_still_parses() {
        let padded = [" 1".to_string(), "64 ".to_string(), "\t-3".to_string()];
        assert_eq!(parse_corner(&padded), Some(IVec3::new(1, 64, -3)));
    }

    #[test]
    fn committed_bounds_are_y_clamped_like_every_other_constructor() {
        let bounds =
            commit_typed_bounds(IVec3::ZERO, &fields(0, -9999, 0), &fields(0, 9999, 0)).unwrap();
        assert_eq!(bounds.min.y, crate::selection::WORLD_MIN_Y);
        assert_eq!(bounds.max.y, crate::selection::WORLD_MAX_Y);
    }

    /// Both thresholds are exclusive, so each boundary value belongs to the
    /// *lower* class — the case an off-by-one here would flip.
    #[test]
    fn the_volume_classifier_at_each_boundary() {
        assert_eq!(classify_volume(0), VolumeClass::Fine);
        assert_eq!(classify_volume(1), VolumeClass::Fine);
        assert_eq!(classify_volume(VOLUME_WARN - 1), VolumeClass::Fine);
        assert_eq!(classify_volume(VOLUME_WARN), VolumeClass::Fine);
        assert_eq!(classify_volume(VOLUME_WARN + 1), VolumeClass::Slow);
        assert_eq!(classify_volume(VOLUME_CAP - 1), VolumeClass::Slow);
        assert_eq!(classify_volume(VOLUME_CAP), VolumeClass::Slow);
        assert_eq!(classify_volume(VOLUME_CAP + 1), VolumeClass::OverCap);
        // `SelectionBounds::volume` saturates rather than wrapping, so this
        // is a value the panel can genuinely be handed.
        assert_eq!(classify_volume(u64::MAX), VolumeClass::OverCap);
    }

    /// A single block, a vanilla structure block's biggest load, and the two
    /// thresholds, all classified the way the panel claims.
    #[test]
    fn the_sizes_the_panel_talks_about_land_where_expected() {
        // The biggest thing a structure block can load is comfortably below
        // the warning threshold — this panel must never nag about a
        // selection that vanilla itself would accept.
        assert_eq!(classify_volume(STRUCTURE_BLOCK_LIMIT), VolumeClass::Fine);

        let big = SelectionBounds::from_corners(
            IVec3::ZERO,
            IVec3::ZERO,
            IVec3::new(299, 255, 299), // 300 x 256 x 300 = 23,040,000
        );
        assert_eq!(classify_volume(big.volume()), VolumeClass::OverCap);
    }

    #[test]
    fn volumes_are_grouped_into_thousands() {
        assert_eq!(format_blocks(0), "0");
        assert_eq!(format_blocks(7), "7");
        assert_eq!(format_blocks(999), "999");
        assert_eq!(format_blocks(1_000), "1,000");
        assert_eq!(format_blocks(110_592), "110,592");
        assert_eq!(format_blocks(16_000_000), "16,000,000");
    }

    /// The fields track the box while it's moved by other means (a click, a
    /// face key), and stop tracking it only while it hasn't changed — which
    /// is what leaves a half-typed field alone.
    #[test]
    fn the_draft_refills_when_the_selection_moves_underneath_it() {
        let mut draft = BoundsDraft::default();
        let mut selection = Selection(None);

        draft.sync(&selection);
        assert_eq!(draft.min, <[String; 3]>::default(), "nothing selected");

        selection.0 = Some(SelectionBounds::from_anchor(IVec3::new(1, 64, -3)));
        draft.sync(&selection);
        assert_eq!(draft.min, fields(1, 64, -3));
        assert_eq!(draft.max, fields(1, 64, -3));

        // Mid-edit: an unchanged selection must not overwrite what's typed.
        draft.min[0] = "-".to_string();
        draft.sync(&selection);
        assert_eq!(draft.min[0], "-");

        // A face key moves the box; now it must.
        let mut moved = selection.0.unwrap();
        crate::selection::move_face(&mut moved, Face::East, 4);
        selection.0 = Some(moved);
        draft.sync(&selection);
        assert_eq!(draft.max, fields(5, 64, -3));

        // And clearing empties them rather than stranding the last box's
        // numbers in the fields.
        selection.0 = None;
        draft.sync(&selection);
        assert_eq!(draft.max, <[String; 3]>::default());
    }

    /// A commit's own result comes back through the fields normalized, not as
    /// typed — the round trip the panel relies on instead of writing the
    /// fields back itself.
    #[test]
    fn a_commit_round_trips_through_the_fields_normalized() {
        let mut draft = BoundsDraft::default();
        let mut selection = Selection(Some(SelectionBounds::from_anchor(IVec3::new(0, 64, 0))));
        draft.sync(&selection);

        draft.min = fields(10, 70, 0);
        draft.max = fields(2, 60, 0);
        selection.0 = commit_typed_bounds(IVec3::new(0, 64, 0), &draft.min, &draft.max);

        draft.sync(&selection);
        assert_eq!(draft.min, fields(2, 60, 0));
        assert_eq!(draft.max, fields(10, 70, 0));
    }
}
