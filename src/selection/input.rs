//! Driving the selection from the mouse and keyboard (ticket 020): click a
//! block to anchor a 1x1x1 selection on it, then push each of the box's six
//! faces around with the arrow keys and `PageUp`/`PageDown`.
//!
//! ## Why these keys
//!
//! [`crate::camera::drive_camera`] already owns `W`/`A`/`S`/`D`, `Q`/`E`,
//! `ShiftLeft`, `Tab` and both mouse buttons, so nothing here may reuse any
//! of them. What's left, and what this module claims:
//!
//! | Key | Moves |
//! | --- | --- |
//! | `→` / `←` | the +X (east) / −X (west) face outward |
//! | `↑` / `↓` | the −Z (north) / +Z (south) face outward |
//! | `PageUp` / `PageDown` | the +Y (up) / −Y (down) face outward |
//! | + `AltLeft` | pull that face back in instead of pushing it out |
//! | + `ControlLeft` | a step of 16 (one chunk) instead of 1 |
//! | `Escape` | clear the selection |
//!
//! **One key moves one face**, not "grow along an axis". Per-face is what
//! makes repeated presses read like dragging a wall, and it's the only
//! meaning "extend in each dimension" has once the anchor is somewhere other
//! than the middle of the box.
//!
//! Directions are **world-absolute**, never camera-relative: `↑` is north =
//! MC `−Z` whichever way the camera happens to be pointing. Camera-relative
//! arrows would need a "snap the yaw to one of four quadrants" rule and get
//! confusing exactly at 45°. If absolute turns out to be disorienting at the
//! window (it's in `todo.md` to judge), the fix is a remap in front of
//! [`move_face`] — which is why that function takes a [`Face`] and not a
//! [`KeyCode`].

use bevy::{input::mouse::AccumulatedMouseMotion, prelude::*, window::PrimaryWindow};

use super::{Selection, SelectionBounds, WORLD_MAX_Y, WORLD_MIN_Y};
use crate::camera::{self, CameraRig, CameraSettings, EguiInputCapture};
use crate::DecodedWorld;

/// A step of one chunk, for `ControlLeft`-modified presses — the unit a
/// Minecraft build is actually laid out in, and the difference between
/// growing a 48-block box in 3 presses and in 48.
///
/// Public so ticket 021's key legend can state the step size rather than
/// retyping `16` into a UI string that would then drift from this.
pub const CHUNK_STEP: i32 = 16;

/// How long a key must be held before it starts repeating, and how fast it
/// repeats after that. Bevy's [`ButtonInput`] has no auto-repeat of its own,
/// and tap-only would make a 40-block box forty keystrokes — the ticket
/// calls that the difference between "usable" and "demo".
const REPEAT_DELAY: f32 = 0.3;
const REPEAT_INTERVAL: f32 = 0.05;

/// Ceiling on how many repeats one frame may fire. A long frame (a chunk
/// batch completing, a debugger breakpoint) would otherwise cash in its
/// whole elapsed time at once and fling a face hundreds of blocks out.
const MAX_STEPS_PER_FRAME: u32 = 8;

/// How far (window pixels) the cursor may travel between press and release
/// and still count as a click rather than a drag. Left mouse is orbit-mode
/// look, so without this every look-drag that happened to end over terrain
/// would re-anchor the selection.
const CLICK_DRAG_TOLERANCE: f32 = 4.0;

/// One of the selection box's six faces, named by the world direction it
/// faces. Key reads turn into one of these immediately, so the geometry
/// below never sees a [`KeyCode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    /// +X.
    East,
    /// −X.
    West,
    /// +Y.
    Up,
    /// −Y.
    Down,
    /// **−Z** — north is negative Z in Minecraft coordinates.
    North,
    /// +Z.
    South,
}

impl Face {
    /// Every face, in axis order (X, Y, Z), each axis's `max` end first.
    ///
    /// Public so ticket 021's key legend is generated from the bindings
    /// themselves rather than retyped as UI strings that would then drift.
    /// The order is also the order [`FaceRepeats`] indexes its timers in, so
    /// it may be reordered but not shortened.
    pub const ALL: [Face; 6] = [
        Face::East,
        Face::West,
        Face::Up,
        Face::Down,
        Face::North,
        Face::South,
    ];

    /// Which component of [`SelectionBounds`]'s `min`/`max` this face lives
    /// on: 0 = X, 1 = Y, 2 = Z.
    fn axis(self) -> usize {
        match self {
            Face::East | Face::West => 0,
            Face::Up | Face::Down => 1,
            Face::North | Face::South => 2,
        }
    }

    /// Whether this face is the `max` end of its axis (and so moves outward
    /// by *increasing* its coordinate) or the `min` end (outward =
    /// decreasing). Note `North` is the `min` end despite `↑` reading as
    /// "forward": north is −Z.
    fn is_max_end(self) -> bool {
        matches!(self, Face::East | Face::Up | Face::South)
    }

    /// The key that pushes this face outward.
    pub fn key(self) -> KeyCode {
        match self {
            Face::East => KeyCode::ArrowRight,
            Face::West => KeyCode::ArrowLeft,
            Face::Up => KeyCode::PageUp,
            Face::Down => KeyCode::PageDown,
            Face::North => KeyCode::ArrowUp,
            Face::South => KeyCode::ArrowDown,
        }
    }

    /// How [`Self::key`] is written in ticket 021's legend. Deliberately
    /// right next to `key` itself, and in the same match order, so a rebinding
    /// that forgets the label is a one-line diff away from being obvious —
    /// the legend is generated from these, never from a separate table.
    pub fn key_label(self) -> &'static str {
        match self {
            Face::East => "→ (Right)",
            Face::West => "← (Left)",
            Face::Up => "PgUp",
            Face::Down => "PgDn",
            Face::North => "↑ (Up)",
            Face::South => "↓ (Down)",
        }
    }

    /// The face's world direction, named the way the legend reads it out:
    /// compass name plus the signed axis, since "north" and "−Z" are both
    /// things a reader may be thinking in.
    pub fn label(self) -> &'static str {
        match self {
            Face::East => "east (+X)",
            Face::West => "west (−X)",
            Face::Up => "up (+Y)",
            Face::Down => "down (−Y)",
            Face::North => "north (−Z)",
            Face::South => "south (+Z)",
        }
    }
}

/// Pushes `face` outward by `blocks`, or pulls it inward when `blocks` is
/// negative. Nothing but this face's bound changes.
///
/// The two clamps are the whole reason this is a function rather than three
/// lines in the system:
///
/// - **A face never crosses its opposite face.** Retracting stops at size 1
///   on that axis, so the selection can shrink to a single block but never
///   invert or become empty — [`SelectionBounds`]'s `min <= max` invariant
///   holds by construction rather than by a later `normalized` call.
/// - **Y is additionally clamped to the world's build limits**
///   ([`WORLD_MIN_Y`]/[`WORLD_MAX_Y`]), matching every other constructor in
///   the parent module.
pub fn move_face(bounds: &mut SelectionBounds, face: Face, blocks: i32) {
    let axis = face.axis();
    let (world_min, world_max) = axis_limits(axis);

    if face.is_max_end() {
        // Outward is +. The lower clamp is the opposite face: a retraction
        // big enough to cross it lands exactly on it, i.e. size 1.
        let moved = bounds.max[axis].saturating_add(blocks);
        bounds.max[axis] = moved.clamp(bounds.min[axis], world_max);
    } else {
        // Outward is −, so a positive `blocks` *subtracts* here.
        let moved = bounds.min[axis].saturating_sub(blocks);
        bounds.min[axis] = moved.clamp(world_min, bounds.max[axis]);
    }
}

/// The hard limits on an axis. Y is bounded by the world's build height; X
/// and Z are effectively unbounded (the parent module's `clamp_y` makes the
/// same split, and ticket 021 guards horizontally on volume instead).
fn axis_limits(axis: usize) -> (i32, i32) {
    if axis == 1 {
        (WORLD_MIN_Y, WORLD_MAX_Y)
    } else {
        (i32::MIN, i32::MAX)
    }
}

/// How far one press moves a face, given the modifiers held with it:
/// `ControlLeft` steps a chunk, `AltLeft` turns the push into a pull.
fn signed_step(chunk_step: bool, retract: bool) -> i32 {
    let step = if chunk_step { CHUNK_STEP } else { 1 };
    if retract {
        -step
    } else {
        step
    }
}

/// Auto-repeat state for one key: fires once on press, then again every
/// [`REPEAT_INTERVAL`] once it's been held for [`REPEAT_DELAY`].
///
/// A countdown rather than an accumulating elapsed-time comparison, so that
/// capping at [`MAX_STEPS_PER_FRAME`] can *drop* the leftover time instead of
/// carrying a debt that pays itself out over the following frames.
#[derive(Default, Clone, Copy)]
struct RepeatTimer {
    held: bool,
    cooldown: f32,
}

impl RepeatTimer {
    /// How many steps this key should fire this frame.
    fn tick(&mut self, pressed: bool, dt: f32) -> u32 {
        if !pressed {
            *self = Self::default();
            return 0;
        }
        if !self.held {
            self.held = true;
            self.cooldown = REPEAT_DELAY;
            return 1;
        }

        self.cooldown -= dt;
        let mut steps = 0;
        while self.cooldown <= 0.0 && steps < MAX_STEPS_PER_FRAME {
            self.cooldown += REPEAT_INTERVAL;
            steps += 1;
        }
        // The loop only exits with a non-positive cooldown when the cap cut
        // it short, so this is the "we forfeited a backlog" case: resume at
        // the normal rate rather than banking the rest, which would pay out
        // over the following frames and fling the face out anyway.
        if self.cooldown <= 0.0 {
            self.cooldown = REPEAT_INTERVAL;
        }
        steps
    }
}

/// One [`RepeatTimer`] per face, so holding `↑` and `→` together grows the
/// box diagonally instead of one key starving the other.
#[derive(Default)]
struct FaceRepeats([RepeatTimer; 6]);

/// Press/drag bookkeeping for the left mouse button, between the frame it
/// goes down and the frame it comes up.
///
/// `None` means "this press is not a candidate" — either no button is down,
/// or the press landed on an egui panel and so was never ours to interpret.
#[derive(Default)]
struct ClickTracker(Option<f32>);

/// [`SystemSet`] for this module's systems. `main.rs` orders it
/// `.after(ui::UiPanelSet)`, exactly as it does [`camera::CameraSet`], so
/// [`EguiInputCapture`] holds *this* frame's value when these systems read it
/// rather than lagging a frame behind — otherwise the first click on a new
/// panel button would also re-anchor the selection to the terrain behind the
/// panel.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelectionInputSet;

pub struct SelectionInputPlugin;

impl Plugin for SelectionInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (anchor_selection_on_click, drive_selection_keys).in_set(SelectionInputSet),
        );
    }
}

/// Left click on terrain replaces the selection with a fresh 1x1x1 box on
/// the block clicked.
///
/// Fires on *release*, not press, and only if the cursor barely moved in
/// between: left mouse is also orbit-mode look, and a look-drag that happens
/// to end over terrain must not re-anchor. A press that started over an egui
/// panel is discarded outright.
///
/// A click that hits no loaded terrain does **nothing** — deliberately not
/// "clears the selection". A missed click at the sky shouldn't destroy a box
/// that took twenty keystrokes to build; `Escape` is the way to clear.
#[allow(clippy::too_many_arguments)]
fn anchor_selection_on_click(
    mut selection: ResMut<Selection>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    egui_input: Res<EguiInputCapture>,
    camera_query: Query<(&Camera, &GlobalTransform), With<CameraRig>>,
    settings: Res<CameraSettings>,
    decoded_world: Res<DecodedWorld>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut click: Local<ClickTracker>,
) {
    if mouse_buttons.just_pressed(MouseButton::Left) {
        // A press egui claimed is not a candidate at all, and can't become
        // one by the pointer leaving the panel before release.
        click.0 = (!egui_input.pointer).then_some(0.0);
    }
    if let Some(travel) = click.0.as_mut() {
        *travel += mouse_motion.delta.length();
    }
    if !mouse_buttons.just_released(MouseButton::Left) {
        return;
    }

    let Some(travel) = click.0.take() else { return };
    if travel > CLICK_DRAG_TOLERANCE || egui_input.pointer {
        return;
    }

    let Ok((camera, camera_transform)) = camera_query.get_single() else {
        return;
    };
    let Some(cursor) = windows.get_single().ok().and_then(Window::cursor_position) else {
        return;
    };

    // The same DDA ray-march the block inspector (007) and orbit re-aiming
    // (006) use — deliberately not a second raycast, so all three agree on
    // which block the cursor is over.
    if let Some(block) = camera::block_under_cursor(
        camera,
        camera_transform,
        cursor,
        settings.max_ray_distance,
        &decoded_world,
    ) {
        selection.0 = Some(SelectionBounds::from_anchor(block));
    }
}

/// Turns held keys into [`move_face`] calls, and `Escape` into a cleared
/// selection.
///
/// Every read is gated on `!egui_input.keyboard`: without that, typing a
/// coordinate into the jump panel (007) or a bounds field (021) would also
/// walk the selection's faces around.
fn drive_selection_keys(
    mut selection: ResMut<Selection>,
    keys: Res<ButtonInput<KeyCode>>,
    egui_input: Res<EguiInputCapture>,
    time: Res<Time>,
    mut repeats: Local<FaceRepeats>,
) {
    // Treat every key as released while egui has the keyboard, rather than
    // returning early: that also resets the repeat timers, so a key held as
    // a panel takes focus doesn't resume mid-repeat when focus comes back.
    let captured = egui_input.keyboard;
    let pressed = |key: KeyCode| !captured && keys.pressed(key);

    if !captured && keys.just_pressed(KeyCode::Escape) {
        selection.0 = None;
    }

    let step = signed_step(pressed(KeyCode::ControlLeft), pressed(KeyCode::AltLeft));
    let dt = time.delta_secs();

    for (face, timer) in Face::ALL.into_iter().zip(repeats.0.iter_mut()) {
        let steps = timer.tick(pressed(face.key()), dt);
        // Timers tick even with nothing selected, so a key already held when
        // a click anchors a box doesn't fire a burst of banked repeats.
        let Some(bounds) = selection.0.as_mut() else {
            continue;
        };
        for _ in 0..steps {
            move_face(bounds, face, step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(min: IVec3, max: IVec3) -> SelectionBounds {
        SelectionBounds::from_corners(min, min, max)
    }

    /// Each key pushes exactly one face outward, and leaves the other five
    /// bounds — and the anchor — alone.
    #[test]
    fn each_face_moves_its_own_bound_outward_and_nothing_else() {
        let start = bounds(IVec3::new(0, 64, 0), IVec3::new(2, 66, 4));

        for (face, expected) in [
            (Face::East, bounds(IVec3::new(0, 64, 0), IVec3::new(5, 66, 4))),
            (Face::West, bounds(IVec3::new(-3, 64, 0), IVec3::new(2, 66, 4))),
            (Face::Up, bounds(IVec3::new(0, 64, 0), IVec3::new(2, 69, 4))),
            (Face::Down, bounds(IVec3::new(0, 61, 0), IVec3::new(2, 66, 4))),
            (Face::North, bounds(IVec3::new(0, 64, -3), IVec3::new(2, 66, 4))),
            (Face::South, bounds(IVec3::new(0, 64, 0), IVec3::new(2, 66, 7))),
        ] {
            let mut moved = start;
            move_face(&mut moved, face, 3);
            assert_eq!(moved.min, expected.min, "{face:?} min");
            assert_eq!(moved.max, expected.max, "{face:?} max");
            assert_eq!(moved.anchor, start.anchor, "{face:?} must not move the anchor");
        }
    }

    /// The one this ticket is most likely to get wrong, given `bevy.z =
    /// -mc.z` elsewhere: `↑` is north, north is −Z, and extending north
    /// therefore *decreases* `min.z` rather than increasing anything.
    #[test]
    fn extending_north_decreases_min_z() {
        let mut b = bounds(IVec3::new(0, 64, 10), IVec3::new(0, 64, 10));
        move_face(&mut b, Face::North, 4);

        assert_eq!(b.min.z, 6);
        assert_eq!(b.max.z, 10, "the south face must not move");
        assert_eq!(Face::North.key(), KeyCode::ArrowUp);
        assert_eq!(Face::South.key(), KeyCode::ArrowDown);
    }

    #[test]
    fn retracting_shrinks_the_box_from_the_face_pressed() {
        let mut b = bounds(IVec3::new(0, 64, 0), IVec3::new(9, 64, 0));
        move_face(&mut b, Face::East, -4);
        assert_eq!(b.max.x, 5);
        assert_eq!(b.min.x, 0);

        move_face(&mut b, Face::West, -2);
        assert_eq!(b.min.x, 2);
        assert_eq!(b.max.x, 5);
    }

    /// Retraction stops at a single block rather than inverting the box or
    /// producing a zero-volume selection — from either end of the axis.
    #[test]
    fn retracting_clamps_at_size_one_instead_of_inverting() {
        let mut from_max = bounds(IVec3::new(0, 64, 0), IVec3::new(3, 64, 0));
        move_face(&mut from_max, Face::East, -100);
        assert_eq!(from_max.min.x, 0);
        assert_eq!(from_max.max.x, 0);
        assert_eq!(from_max.size().x, 1);
        assert_eq!(from_max.volume(), 1);

        let mut from_min = bounds(IVec3::new(0, 64, 0), IVec3::new(3, 64, 0));
        move_face(&mut from_min, Face::West, -100);
        assert_eq!(from_min.min.x, 3);
        assert_eq!(from_min.max.x, 3);
        assert_eq!(from_min.size().x, 1);
    }

    /// The legend ticket 021 draws is generated from these, so a face with a
    /// duplicated key (or a label copy-pasted and not edited) would show up
    /// there as two rows claiming the same thing.
    #[test]
    fn every_face_has_its_own_key_and_its_own_labels() {
        let mut keys: Vec<KeyCode> = Face::ALL.iter().map(|f| f.key()).collect();
        keys.sort_by_key(|k| format!("{k:?}"));
        keys.dedup();
        assert_eq!(keys.len(), Face::ALL.len());

        let mut labels: Vec<&str> = Face::ALL
            .iter()
            .flat_map(|f| [f.key_label(), f.label()])
            .collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 2 * Face::ALL.len());
    }

    #[test]
    fn control_steps_a_chunk_in_both_directions_and_with_alt() {
        assert_eq!(signed_step(false, false), 1);
        assert_eq!(signed_step(false, true), -1);
        assert_eq!(signed_step(true, false), CHUNK_STEP);
        assert_eq!(signed_step(true, true), -CHUNK_STEP);

        let mut b = bounds(IVec3::new(0, 64, 0), IVec3::new(0, 64, 0));
        move_face(&mut b, Face::East, signed_step(true, false));
        assert_eq!(b.max.x, 16);
        move_face(&mut b, Face::East, signed_step(true, true));
        assert_eq!(b.max.x, 0, "a chunk out then a chunk back is where it started");

        // And a chunk-sized retraction still can't cross the opposite face.
        move_face(&mut b, Face::East, signed_step(true, true));
        assert_eq!(b.max.x, 0);
    }

    #[test]
    fn y_clamps_at_both_build_limits() {
        let mut up = bounds(IVec3::new(0, 300, 0), IVec3::new(0, 300, 0));
        move_face(&mut up, Face::Up, 1000);
        assert_eq!(up.max.y, WORLD_MAX_Y);
        assert_eq!(up.min.y, 300, "clamping the top must not drag the bottom");

        let mut down = bounds(IVec3::new(0, -60, 0), IVec3::new(0, -60, 0));
        move_face(&mut down, Face::Down, 1000);
        assert_eq!(down.min.y, WORLD_MIN_Y);
        assert_eq!(down.max.y, -60);
    }

    /// X and Z are unclamped, so a long key-repeat can walk a face toward
    /// `i32`'s range — it must saturate rather than wrap into a box on the
    /// far side of the world.
    #[test]
    fn horizontal_faces_saturate_rather_than_wrapping() {
        let mut b = bounds(IVec3::new(0, 64, 0), IVec3::new(i32::MAX - 2, 64, 0));
        move_face(&mut b, Face::East, CHUNK_STEP);
        assert_eq!(b.max.x, i32::MAX);

        let mut west = bounds(IVec3::new(i32::MIN + 2, 64, 0), IVec3::new(0, 64, 0));
        move_face(&mut west, Face::West, CHUNK_STEP);
        assert_eq!(west.min.x, i32::MIN);
    }

    #[test]
    fn a_press_fires_once_and_only_repeats_after_the_hold_delay() {
        let mut timer = RepeatTimer::default();
        assert_eq!(timer.tick(true, 0.0), 1, "the initial press fires immediately");

        // Held, but not yet long enough.
        let mut fired = 0;
        let mut elapsed = 0.0;
        while elapsed < REPEAT_DELAY - 0.02 {
            fired += timer.tick(true, 0.01);
            elapsed += 0.01;
        }
        assert_eq!(fired, 0, "nothing should repeat inside the hold delay");

        // Past the delay it repeats at roughly REPEAT_INTERVAL.
        let mut repeats = 0;
        for _ in 0..100 {
            repeats += timer.tick(true, 0.01);
        }
        let expected = (1.0 / REPEAT_INTERVAL) as u32;
        assert!(
            repeats.abs_diff(expected) <= 2,
            "expected ~{expected} repeats in a second of holding, got {repeats}"
        );
    }

    #[test]
    fn releasing_resets_the_repeat_so_the_next_press_starts_over() {
        let mut timer = RepeatTimer::default();
        timer.tick(true, 0.0);
        for _ in 0..100 {
            timer.tick(true, 0.01);
        }
        assert_eq!(timer.tick(false, 0.01), 0);
        assert_eq!(timer.tick(true, 0.0), 1, "a fresh press fires once");
        assert_eq!(timer.tick(true, 0.01), 0, "and then waits out the delay again");
    }

    /// A frame long enough to owe hundreds of repeats must not cash them all
    /// in — and must not bank the remainder for the following frames either.
    #[test]
    fn a_long_frame_is_capped_and_does_not_bank_the_remainder() {
        let mut timer = RepeatTimer::default();
        timer.tick(true, 0.0);
        assert_eq!(timer.tick(true, 10.0), MAX_STEPS_PER_FRAME);
        assert_eq!(
            timer.tick(true, 0.0),
            0,
            "the forfeited time must not pay out on the next frame"
        );
        // It picks back up at the ordinary repeat rate, not stalled.
        assert_eq!(timer.tick(true, REPEAT_INTERVAL), 1);
    }

    /// Everything above is pure; these two ride an `App` because the gating
    /// they check (`EguiInputCapture`, `Escape`) only exists in the system.
    fn key_app() -> App {
        let mut app = App::new();
        app.init_resource::<Selection>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<EguiInputCapture>()
            .init_resource::<Time>()
            .add_systems(Update, drive_selection_keys);
        app
    }

    fn press(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
    }

    #[test]
    fn escape_clears_the_selection() {
        let mut app = key_app();
        app.world_mut().resource_mut::<Selection>().0 =
            Some(SelectionBounds::from_anchor(IVec3::new(1, 64, 1)));

        press(&mut app, KeyCode::Escape);
        app.update();

        assert!(app.world().resource::<Selection>().0.is_none());
    }

    /// Typing into an egui field must move nothing — neither a face nor the
    /// selection's existence.
    #[test]
    fn keys_do_nothing_while_egui_has_the_keyboard() {
        let mut app = key_app();
        let start = SelectionBounds::from_anchor(IVec3::new(1, 64, 1));
        app.world_mut().resource_mut::<Selection>().0 = Some(start);
        app.world_mut().resource_mut::<EguiInputCapture>().keyboard = true;

        press(&mut app, KeyCode::ArrowRight);
        press(&mut app, KeyCode::Escape);
        app.update();

        assert_eq!(app.world().resource::<Selection>().0, Some(start));
    }

    #[test]
    fn an_arrow_press_moves_the_face_its_key_owns() {
        let mut app = key_app();
        app.world_mut().resource_mut::<Selection>().0 =
            Some(SelectionBounds::from_anchor(IVec3::new(1, 64, 1)));

        press(&mut app, KeyCode::ArrowRight);
        app.update();

        let bounds = app.world().resource::<Selection>().0.unwrap();
        assert_eq!(bounds.max.x, 2);
        assert_eq!(bounds.min, IVec3::new(1, 64, 1));
    }

    /// Pressing a face key with nothing selected must not bank repeats that
    /// then fire all at once the moment a click anchors a box.
    #[test]
    fn holding_a_key_with_nothing_selected_does_not_bank_steps() {
        let mut app = key_app();
        press(&mut app, KeyCode::ArrowRight);
        for _ in 0..30 {
            app.update();
        }

        app.world_mut().resource_mut::<Selection>().0 =
            Some(SelectionBounds::from_anchor(IVec3::new(1, 64, 1)));
        app.update();

        let bounds = app.world().resource::<Selection>().0.unwrap();
        assert_eq!(
            bounds.size(),
            IVec3::ONE,
            "a key held before the selection existed should not have banked steps"
        );
    }
}
