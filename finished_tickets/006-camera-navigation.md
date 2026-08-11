# 006 - A camera you can actually explore with

## Status
Open

## Depends on
Nothing hard; most useful once 003 renders something.

## Problem

The current `orbit` system orbits a fixed `Vec3::ZERO` target: you can look
around the origin and zoom, but you cannot *go* anywhere. For a world that is
thousands of blocks across, that's unusable.

Meanwhile `src/pan_orbit_camera_bundle.rs` (8.6 KB — `PanOrbitCameraBundle`,
`PanOrbitState`, `PanOrbitSettings`) is **never declared as a module** in
`main.rs`, so it isn't compiled at all. Decide its fate in this ticket:
either wire it up and delete the inline `orbit`, or delete the file. Two
half-camera implementations is the actual bug.

## Goal

Free movement through the world, plus a way to return to a known spot.

## Scope

- **Fly camera**: WASD + Q/E (or Space/Shift) for vertical, mouse-look on
  right-drag or with cursor grab, Shift-to-sprint, scroll to adjust speed.
  Movement speed should scale sensibly — crossing a region at 5 blocks/s is
  not exploring.
- Keep an **orbit/inspect mode** if it's wanted for looking at a build, but
  make the target settable (orbit what's under the cursor) rather than
  hardcoded to the origin — otherwise it has no use once the camera moves.
- Far clip plane and fog tuned to the render distance from 005 — Bevy's
  default projection far plane will clip distant terrain otherwise.
- Sensible initial camera placement: put it above the terrain surface near
  the save's spawn or the centre of the loaded region, not at
  `(10, 5, 10)` where it may well be underground.

## Cleanup included

- `CameraSettings.is_roatatin` (typo, and never read) and `orbit_distance`
  (never read) currently produce dead-code warnings on every build. Remove or
  use them.

## Out of scope

- Collision, gravity, player physics — this is a viewer, fly through walls.
- UI for camera settings (007).

## Done when

- The camera can fly anywhere in the world at a usable speed, with no dead
  code warnings from `CameraSettings`.
- There is exactly one camera implementation in the repo.
- Distant terrain is not clipped by the far plane at the default render
  distance.
