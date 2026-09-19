# 124 - Citybuilder: loading screen until the first chunks are in

## Status
Done — `cargo test --lib`: 1141 passed (was 1135; new: `city::loading`'s
progress/completion/transition/set-gating tests and the loading screen's
status line). Manual check pending, see `todo.md`.

## Ask

"For the citybuilder, when starting up, show a loading screen until all
initial chunks are loaded (the first batch of chunks around the camera such
that we can actually start playing). During this loading time no game logic
is happening and no time is passing in-game."

## What "initial chunks loaded" means

The camera spawns at a fixed point (`lib.rs::setup_world`) and the first
`Update` frame's `streaming::update_pending_chunk_work` publishes the whole
load disc (`RenderDistance + ChunkPreload`, ~1000 chunks at the
citybuilder's defaults) as `PendingChunkWork::to_load`; `chunk_pipeline`
drains that into `InFlightChunkLoads`. A coordinate with no chunk behind it
(no region file, not fully generated) resolves to `None` and never enters
`DecodedWorld`, so "every desired chunk is decoded" is *not* a condition
that's guaranteed to ever hold. The one that is: the streaming diff has run
at least once (`LastCameraChunk` is set) and nothing is queued or in flight.
Between the first recompute and that point there is no frame where both are
empty while work remains — `to_load` is only empty between `start_chunk_loads`
draining it and the next recompute, and by then the tasks are in flight.

The camera is frozen during loading, so the disc it measures against never
moves.

## Change

- `city::loading` — `CityPhase` (`Loading` -> `Playing`, a Bevy `States`),
  `GameplaySet` (the set every city plugin's game-logic systems sit in), an
  `InitialLoad` progress resource (`total` = the disc, `outstanding` =
  queued + in flight), and `track_initial_load`, which measures it every
  frame in `Loading` and flips to `Playing` once complete.
- `city::run` configures `GameplaySet` in both `First` (the clock) and
  `Update` to `run_if(in_state(CityPhase::Playing))`, and `camera::CameraSet`
  the same way — no input moves the camera under the loading screen.
- Every city plugin's `add_systems` gets `.in_set(GameplaySet)`: clock,
  hot reload, warehouse/farm coverage, production, gatherer, mine, tool,
  picking, placement, commit, demolish, road build, terraform, work area,
  undo, save, and the UI panels. The `Last` exit-save systems stay ungated
  (quitting mid-load must still not lose anything).
- `city::ui::loading_screen` — a full-window opaque egui `CentralPanel`
  with the save name and a progress bar, drawn only in `Loading`.

`GameClock` is a plain resource whose only writer is `advance_clock`; with
that system held back its `delta` stays at the `Default` zero, so nothing
that reads the clock can see time pass. `TimeOfDay` (the sky) defaults to
paused and the citybuilder has no control for it, so nothing to do there.

## Done when

- `cargo test --lib` passes, with tests for `InitialLoad`'s progress /
  completion rules and the `Loading -> Playing` transition.
- Manual: `cargo run --bin citybuilder` shows the loading screen, the bar
  fills, the game appears with the clock at `0:00:00` and no production
  having happened. See `todo.md`.
