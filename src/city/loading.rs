//! The startup loading phase (ticket 124): the citybuilder holds every
//! piece of game logic back until the first disc of chunks around the
//! camera is in, and shows a loading screen meanwhile
//! (`ui::loading_screen`).
//!
//! ## What "loaded" means
//!
//! The camera spawns at a fixed point (`lib.rs::setup_world`), and the first
//! `Update` frame's `streaming::update_pending_chunk_work` publishes the
//! whole load disc as [`PendingChunkWork::to_load`], which `chunk_pipeline`
//! drains into [`InFlightChunkLoads`]. A coordinate with no chunk behind it
//! — no region file, not fully generated — resolves to `None` and never
//! enters `DecodedWorld`, so "every desired chunk is decoded" is a condition
//! that need never hold. The one that does: the streaming diff has run at
//! least once ([`LastCameraChunk`] is set) and nothing is queued or in
//! flight. Between the first recompute and that point there is no frame
//! where both are empty while work remains — `to_load` is only empty after
//! `start_chunk_loads` drained it into tasks, and the tasks are in flight
//! until they're polled complete. The camera is frozen for the duration
//! (`city::run` gates `camera::CameraSet` on [`CityPhase::Playing`]), so the
//! disc never moves under the measurement.
//!
//! ## Why a `States` enum and a set, not a flag
//!
//! Game logic is spread over every plugin in `city` — ~18 of them, all in
//! `Update`, plus the clock in `First`. A `bool` resource each of them
//! checked would be one forgotten check away from a farm producing under the
//! loading screen. Instead every one of them puts its systems in
//! [`GameplaySet`], and `city::run` configures that set once, per schedule,
//! to run only in [`CityPhase::Playing`]. [`GameClock`](super::clock::GameClock)
//! then needs nothing of its own: its only writer is held back, so its
//! `delta` stays at the `Default` zero and nothing downstream can see time
//! pass.

use bevy::prelude::*;

use crate::camera;
use crate::chunk_pipeline::InFlightChunkLoads;
use crate::streaming::{self, ChunkPreload, LastCameraChunk, PendingChunkWork, RenderDistance};

/// Which half of the run the citybuilder is in. Starts in `Loading`;
/// [`track_initial_load`] moves it to `Playing` exactly once, and nothing
/// moves it back.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CityPhase {
    #[default]
    Loading,
    Playing,
}

/// Every city plugin's game-logic systems sit in this set, and `city::run`
/// configures it to run only in [`CityPhase::Playing`] — in `First` for the
/// clock and in `Update` for everything else. A plugin spun up on its own
/// (the tests do this) leaves the set unconfigured, which is to say it runs
/// unconditionally, so nothing outside `city::run` needs to know.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GameplaySet;

/// How far the initial load has got — what the loading screen draws its
/// bar from. Measured fresh every frame by [`track_initial_load`].
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InitialLoad {
    /// Whether the streaming diff has published the disc yet. Before it
    /// has, `outstanding` is zero for the wrong reason.
    pub started: bool,
    /// Chunks in the load disc around the camera.
    pub total: usize,
    /// Chunks still queued or in flight.
    pub outstanding: usize,
}

impl InitialLoad {
    /// Chunks that have resolved — loaded, or found not to exist. Zero
    /// until the disc has actually been published, whatever `outstanding`
    /// says.
    pub fn resolved(&self) -> usize {
        if self.started { self.total.saturating_sub(self.outstanding) } else { 0 }
    }

    /// `resolved / total` in `0..=1`; zero for an empty disc rather than a
    /// NaN the progress bar would choke on.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.resolved() as f32 / self.total as f32 }
    }

    /// The module docs' condition: the disc has been published and nothing
    /// is left queued or in flight.
    pub fn is_complete(&self) -> bool {
        self.started && self.outstanding == 0
    }
}

pub struct LoadingPlugin;

impl Plugin for LoadingPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<CityPhase>()
            .init_resource::<InitialLoad>()
            .add_systems(Update, track_initial_load.run_if(in_state(CityPhase::Loading)));
    }
}

/// Measures [`InitialLoad`] against the load disc around the camera and
/// flips to [`CityPhase::Playing`] once it's complete. Order relative to
/// the streaming and pipeline systems is deliberately unconstrained: on any
/// frame, `to_load + in_flight` is the outstanding count whether or not
/// `start_chunk_loads` has drained the one into the other yet, and a frame's
/// lag on the bar is invisible.
fn track_initial_load(
    camera: Query<&Transform, With<camera::CameraRig>>,
    render_distance: Res<RenderDistance>,
    preload: Res<ChunkPreload>,
    last_chunk: Res<LastCameraChunk>,
    pending: Res<PendingChunkWork>,
    in_flight: Res<InFlightChunkLoads>,
    mut progress: ResMut<InitialLoad>,
    mut next_phase: ResMut<NextState<CityPhase>>,
) {
    let Ok(transform) = camera.get_single() else {
        return; // Not spawned yet — `Startup` hasn't finished.
    };
    let center = streaming::camera_chunk_coord(transform.translation);
    let radius = streaming::load_radius(&render_distance, &preload);

    *progress = InitialLoad {
        started: last_chunk.0.is_some(),
        total: streaming::desired_chunks(center, radius).len(),
        outstanding: pending.to_load.len() + in_flight.len(),
    };
    if progress.is_complete() {
        println!("Chunk streaming: initial {} chunks resolved, starting the game", progress.total);
        next_phase.set(CityPhase::Playing);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::state::app::StatesPlugin;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(StatesPlugin)
            .insert_resource(RenderDistance(1))
            .insert_resource(ChunkPreload(0))
            .init_resource::<LastCameraChunk>()
            .init_resource::<PendingChunkWork>()
            .init_resource::<InFlightChunkLoads>()
            .add_plugins(LoadingPlugin);
        let eye = Vec3::new(8.0, 100.0, -8.0);
        app.world_mut().spawn((Transform::from_translation(eye), camera::CameraRig::looking_at(eye, Vec3::ZERO)));
        app
    }

    fn phase(app: &App) -> CityPhase {
        *app.world().resource::<State<CityPhase>>().get()
    }

    #[test]
    fn nothing_outstanding_before_the_first_recompute_does_not_count_as_done() {
        let mut app = app();
        app.update();
        app.update();

        assert_eq!(phase(&app), CityPhase::Loading);
        let progress = *app.world().resource::<InitialLoad>();
        assert!(!progress.started);
        assert_eq!(progress.resolved(), 0);
        assert_eq!(progress.fraction(), 0.0);
    }

    /// The disc is the radius-1 disc around chunk (0, 0): five chunks.
    #[test]
    fn progress_counts_queued_and_in_flight_against_the_disc() {
        let mut app = app();
        app.world_mut().resource_mut::<LastCameraChunk>().0 = Some((0, 0));
        app.world_mut().resource_mut::<PendingChunkWork>().to_load = vec![(0, 0), (1, 0), (0, 1)];
        app.update();

        let progress = *app.world().resource::<InitialLoad>();
        assert_eq!(progress, InitialLoad { started: true, total: 5, outstanding: 3 });
        assert_eq!(progress.resolved(), 2);
        assert!((progress.fraction() - 0.4).abs() < 1e-6);
        assert_eq!(phase(&app), CityPhase::Loading);
    }

    #[test]
    fn the_phase_flips_to_playing_once_nothing_is_outstanding() {
        let mut app = app();
        app.world_mut().resource_mut::<LastCameraChunk>().0 = Some((0, 0));
        app.world_mut().resource_mut::<PendingChunkWork>().to_load = vec![(0, 0)];
        app.update();
        assert_eq!(phase(&app), CityPhase::Loading);

        app.world_mut().resource_mut::<PendingChunkWork>().to_load.clear();
        app.update(); // `NextState` set this frame …
        app.update(); // … applied by `StateTransition` on the next.
        assert_eq!(phase(&app), CityPhase::Playing);
        assert!(app.world().resource::<InitialLoad>().is_complete());
    }

    /// The set is what `city::run` gates; a system in it must not run under
    /// the loading screen and must run once the phase flips.
    #[test]
    fn gameplay_set_runs_only_while_playing() {
        #[derive(Resource, Default)]
        struct Ticks(u32);
        fn tick(mut ticks: ResMut<Ticks>) {
            ticks.0 += 1;
        }

        let mut app = app();
        app.init_resource::<Ticks>()
            .add_systems(Update, tick.in_set(GameplaySet))
            .configure_sets(Update, GameplaySet.run_if(in_state(CityPhase::Playing)));

        app.update();
        app.update();
        assert_eq!(app.world().resource::<Ticks>().0, 0, "no game logic under the loading screen");

        app.world_mut().resource_mut::<LastCameraChunk>().0 = Some((0, 0));
        app.update();
        app.update();
        assert_eq!(phase(&app), CityPhase::Playing);
        assert_eq!(app.world().resource::<Ticks>().0, 1);
    }

    #[test]
    fn an_empty_disc_has_a_zero_fraction_not_a_nan() {
        let progress = InitialLoad { started: true, total: 0, outstanding: 0 };
        assert_eq!(progress.fraction(), 0.0);
        assert!(progress.is_complete());
    }
}
