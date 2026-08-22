//! The game clock (ticket 077, roadmap H2): the time base the economy runs
//! on, with pause and speed controls.
//!
//! The roadmap decided this rather than deferring it — "the economy runs on a
//! game clock with pause and speed controls, not on raw wall-clock time" —
//! and the reason is that two things have to agree about how much time has
//! passed. A producer's accumulator (ticket 078) integrates a `per_minute`
//! rate; a haul's remaining travel time (ticket 080) counts one down. If each
//! read [`Time::delta`] for itself, "4x" would have to be applied identically
//! in two places, and a player who paused to read the build menu would come
//! back to a minute of farm output.
//!
//! So: [`GameClock::delta`] is the only time anything in the economy reads,
//! and it is zero while paused.
//!
//! ## Why an enum and not an `f32`
//!
//! [`GameSpeed`] is `Paused | Normal | Fast | Fastest`. The UI offers exactly
//! those four, so those four are exactly what the rest of the game has to be
//! correct at; a hand-set 0.37x is not a state worth defending against, and a
//! bare multiplier would invite one. Pause is a *speed*, not a separate flag,
//! which is what keeps "is the game paused" from being answerable two ways.
//!
//! ## The catch-up clamp
//!
//! A frame's real delta can be enormous — a chunk-load stall, a dragged
//! window, a debugger breakpoint — and Bevy hands the whole of it over. Left
//! alone, a two-second hitch at 4x would deliver eight seconds of production
//! in one tick, and, worse, land a haul that should still have been on the
//! road. [`MAX_FRAME_ADVANCE`] caps what one frame can advance and the
//! remainder is dropped, not banked: the economy losing a fraction of a
//! second of output across a hitch is invisible, where a stack teleporting
//! home because the window was dragged is not.
//!
//! ## Not persisted, deliberately
//!
//! [`GameClock::elapsed`] starts at zero every run and [`GameSpeed`] at
//! [`GameSpeed::Normal`]. Nothing downstream reads an absolute time — 078's
//! buffers and 080's *remaining* travel times are the state that has to
//! survive a quit, and those live in their own file. A save that reopened
//! paused would look broken, and an elapsed counter nothing reads would be a
//! number kept for its own sake.

use std::time::Duration;

use bevy::prelude::*;

use crate::camera;

/// The most game time one frame may advance, whatever the real delta was —
/// see the module docs. A quarter of a second is one 4x frame at 15fps: slow
/// enough that no ordinary frame is ever clamped, short enough that a hitch
/// can't move a haul a meaningful distance.
const MAX_FRAME_ADVANCE: Duration = Duration::from_millis(250);

/// How fast game time runs relative to real time. Pause is one of these
/// rather than a separate flag — see the module docs.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GameSpeed {
    Paused,
    #[default]
    Normal,
    Fast,
    Fastest,
}

impl GameSpeed {
    /// Real seconds to game seconds. `Paused` is `0.0`, which is what makes
    /// [`advance_clock`] need no special case for it.
    pub fn multiplier(self) -> f32 {
        match self {
            GameSpeed::Paused => 0.0,
            GameSpeed::Normal => 1.0,
            GameSpeed::Fast => 2.0,
            GameSpeed::Fastest => 4.0,
        }
    }

    /// The button face in the city panel, and the label anything else uses
    /// when it has to name a speed.
    pub fn label(self) -> &'static str {
        match self {
            GameSpeed::Paused => "||",
            GameSpeed::Normal => "1x",
            GameSpeed::Fast => "2x",
            GameSpeed::Fastest => "4x",
        }
    }

    /// Every speed, in the order the panel lays them out — so the UI doesn't
    /// carry its own list that could fall out of step with the enum.
    pub const ALL: [GameSpeed; 4] = [GameSpeed::Paused, GameSpeed::Normal, GameSpeed::Fast, GameSpeed::Fastest];
}

/// Game time: how much has passed in total, and how much passed this frame.
///
/// [`delta`](Self::delta) is what the economy reads; [`elapsed`](Self::elapsed)
/// exists so the city panel can distinguish "my farm has produced nothing"
/// from "the game has been paused for ten minutes", which is a real support
/// question and a cheap one to answer.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct GameClock {
    pub elapsed: Duration,
    /// Zero while paused, and never larger than [`MAX_FRAME_ADVANCE`].
    pub delta: Duration,
}

impl GameClock {
    /// This frame's advance in *minutes*, the unit every rate in
    /// [`super::definition`] is already written in (`per_minute`), so no
    /// caller does its own division by 60.
    #[allow(dead_code)] // ticket 078's production tick is the first caller
    pub fn delta_minutes(&self) -> f32 {
        self.delta.as_secs_f32() / 60.0
    }

    /// `elapsed` as `h:mm:ss`, for the panel. Hours are shown from zero
    /// rather than appearing once the first one passes, so the field doesn't
    /// change width under the reader.
    pub fn elapsed_label(&self) -> String {
        let total = self.elapsed.as_secs();
        format!("{}:{:02}:{:02}", total / 3600, (total % 3600) / 60, total % 60)
    }
}

pub struct ClockPlugin;

impl Plugin for ClockPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GameSpeed>()
            .init_resource::<GameClock>()
            // `First`, so every `Update` system this frame — production,
            // haulage, and the panel that reports on both — sees the same
            // already-advanced clock rather than racing the system that
            // advances it.
            .add_systems(First, advance_clock)
            .add_systems(Update, toggle_pause);
    }
}

/// Advances [`GameClock`] by this frame's real delta, scaled by
/// [`GameSpeed`] and clamped by [`MAX_FRAME_ADVANCE`] — see the module docs.
fn advance_clock(time: Res<Time>, speed: Res<GameSpeed>, mut clock: ResMut<GameClock>) {
    let advance = time.delta().mul_f32(speed.multiplier()).min(MAX_FRAME_ADVANCE);
    clock.delta = advance;
    clock.elapsed += advance;
}

/// `Space` toggles pause, gated on [`camera::EguiInputCapture`] the same way
/// every other keyboard binding in this game is, so hitting space while
/// typing in a panel doesn't also stop the world.
///
/// Toggling *back* goes to [`GameSpeed::Normal`] rather than to whatever
/// speed was running before. Remembering the previous speed would mean the
/// same keypress does different things depending on state the player can't
/// see; the panel's own buttons are how you get back to 4x.
fn toggle_pause(keys: Res<ButtonInput<KeyCode>>, egui_input: Res<camera::EguiInputCapture>, mut speed: ResMut<GameSpeed>) {
    if egui_input.keyboard || !keys.just_pressed(KeyCode::Space) {
        return;
    }
    *speed = if *speed == GameSpeed::Paused { GameSpeed::Normal } else { GameSpeed::Paused };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No `TimePlugin`: a bare [`Time`] resource driven by
    /// [`advance`] below, so a test asserts on an exact number of
    /// milliseconds rather than on however long the frame happened to take.
    fn app() -> App {
        let mut app = App::new();
        app.init_resource::<Time>();
        app.init_resource::<camera::EguiInputCapture>();
        app.init_resource::<ButtonInput<KeyCode>>();
        app.add_plugins(ClockPlugin);
        app
    }

    fn advance(app: &mut App, real: Duration) {
        app.world_mut().resource_mut::<Time>().advance_by(real);
        app.update();
    }

    fn press_space(app: &mut App) {
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::Space);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().release(KeyCode::Space);
    }

    #[test]
    fn normal_speed_passes_real_time_through() {
        let mut app = app();
        advance(&mut app, Duration::from_millis(100));
        assert_eq!(app.world().resource::<GameClock>().delta, Duration::from_millis(100));
        assert_eq!(app.world().resource::<GameClock>().elapsed, Duration::from_millis(100));
    }

    #[test]
    fn a_faster_speed_multiplies_the_advance() {
        let mut app = app();
        *app.world_mut().resource_mut::<GameSpeed>() = GameSpeed::Fastest;
        advance(&mut app, Duration::from_millis(50));
        assert_eq!(app.world().resource::<GameClock>().delta, Duration::from_millis(200));
    }

    #[test]
    fn paused_advances_nothing_at_all() {
        let mut app = app();
        advance(&mut app, Duration::from_millis(100));
        *app.world_mut().resource_mut::<GameSpeed>() = GameSpeed::Paused;
        advance(&mut app, Duration::from_secs(10));

        let clock = *app.world().resource::<GameClock>();
        assert_eq!(clock.delta, Duration::ZERO, "a paused frame advances nothing");
        assert_eq!(clock.elapsed, Duration::from_millis(100), "and elapsed stays where it was");
    }

    /// The module docs' catch-up clamp: a two-second hitch must not deliver
    /// two seconds of economy in one tick, and the remainder is dropped
    /// rather than banked for the next frame.
    #[test]
    fn a_huge_real_delta_is_clamped_and_the_rest_is_dropped() {
        let mut app = app();
        advance(&mut app, Duration::from_secs(2));
        assert_eq!(app.world().resource::<GameClock>().delta, MAX_FRAME_ADVANCE);

        advance(&mut app, Duration::from_millis(16));
        assert_eq!(
            app.world().resource::<GameClock>().elapsed,
            MAX_FRAME_ADVANCE + Duration::from_millis(16),
            "the clamped remainder must not reappear on a later frame"
        );
    }

    /// Constructed directly rather than driven through [`advance`]: any
    /// delta big enough to make a round number of minutes is past
    /// [`MAX_FRAME_ADVANCE`], so going through the app would only re-test the
    /// clamp. This is the unit conversion on its own.
    #[test]
    fn delta_minutes_is_the_unit_production_rates_are_written_in() {
        let clock = GameClock { elapsed: Duration::ZERO, delta: Duration::from_secs(30) };
        assert!((clock.delta_minutes() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn space_toggles_pause_and_back_to_normal() {
        let mut app = app();
        press_space(&mut app);
        assert_eq!(*app.world().resource::<GameSpeed>(), GameSpeed::Paused);

        press_space(&mut app);
        assert_eq!(*app.world().resource::<GameSpeed>(), GameSpeed::Normal);
    }

    /// Unpausing returns to `Normal`, not to the speed that was running
    /// before — see [`toggle_pause`]'s own docs.
    #[test]
    fn unpausing_returns_to_normal_not_to_the_previous_speed() {
        let mut app = app();
        *app.world_mut().resource_mut::<GameSpeed>() = GameSpeed::Fastest;
        press_space(&mut app);
        press_space(&mut app);

        assert_eq!(*app.world().resource::<GameSpeed>(), GameSpeed::Normal);
    }

    #[test]
    fn space_is_ignored_while_egui_has_the_keyboard() {
        let mut app = app();
        app.world_mut().resource_mut::<camera::EguiInputCapture>().keyboard = true;
        press_space(&mut app);
        assert_eq!(*app.world().resource::<GameSpeed>(), GameSpeed::Normal, "typing in a panel must not pause the game");
    }

    #[test]
    fn elapsed_label_pads_minutes_and_seconds() {
        let clock = GameClock { elapsed: Duration::from_secs(3 * 3600 + 4 * 60 + 5), delta: Duration::ZERO };
        assert_eq!(clock.elapsed_label(), "3:04:05");
    }
}
