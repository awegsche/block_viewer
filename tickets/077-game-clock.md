# 077 - The game clock: pause and speed

## Status
Open

## Why

Roadmap H2's last line, decided rather than deferred:

> **Decided**: the economy runs on a game clock with pause and speed
> controls, not on raw wall-clock time.

Production (078) and haulage (080) both integrate a rate over time. Neither
should read `Time::delta` directly: a player who pauses to look at the build
menu should not come back to a minute of farm output, and "4x" has to mean
the same thing to a producer's accumulator and to a cart's remaining travel
time or the two drift apart within one session.

## Work

`city::clock`, a small module with no dependencies on anything else in `city`:

- **`GameSpeed`** — `Paused | Normal | Fast | Fastest` (1x / 2x / 4x), a
  resource. An enum, not a bare `f32`: the UI offers exactly these, and a
  hand-set 0.37x is not a state anything should have to be correct at.
- **`GameClock`** — `elapsed: Duration` (game time, not wall time) and
  `delta: Duration` (this frame's game-time advance, zero while paused).
  One system in `First` advances it from `Time::delta` scaled by `GameSpeed`.
- **`GameClock::delta_minutes() -> f32`** — the unit every `per_minute` rate
  in `city::definition` is already written in, so no caller does its own
  60.0 arithmetic.
- **A clamp on catch-up.** A frame whose real delta is huge (a stalled
  chunk-load, a dragged window, a breakpoint) must not deliver a minute of
  production in one tick. Cap the per-frame game-time advance at
  `MAX_FRAME_ADVANCE` (0.25s of game time, i.e. one 4x frame at 15fps) and
  drop the rest on the floor. The economy losing a tenth of a second of
  output on a hitch is invisible; a haul teleporting because the window was
  dragged is not.

## Controls

- `Space` toggles pause, gated on `camera::EguiInputCapture` like every other
  keyboard binding in this game.
- The city panel grows a speed row (`|| 1x 2x 4x`) — the real control, the
  keyboard being the stand-in as usual.
- The panel shows elapsed game time, so "my farm has produced nothing" can be
  distinguished from "the game has been paused for ten minutes".

## Not persisted

`GameClock::elapsed` starts at zero every run, and that is correct rather than
a shortcut: nothing in 078-080 reads an absolute time, only `delta`. A
producer's buffer and a haul's *remaining* time are the state that has to
survive a quit, and those are 078's and 080's own (`logistics.ron`).
`GameSpeed` likewise starts at `Normal` — a save that reopens paused looks
broken.
