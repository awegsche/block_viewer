# 106 - Rts pan speed: untie from zoom level

## Problem

`CameraMode::Rts` pan speed (`W`/`A`/`S`/`D`) is computed as
`orbit_radius * rts_pan_speed_factor` (`src/camera.rs::drive_camera`). This
means panning slows down dramatically when zoomed in (small
`orbit_radius`), which feels broken — the camera becomes crawlingly slow
to move at high zoom.

## Fix

Replace the zoom-scaled pan speed with a constant blocks/sec speed, set to
20% faster than what the old formula produced at the rig's default spawn
zoom level (`orbit_radius` ≈ 39.4, from `lib.rs::setup_world`'s
`eye = target + Vec3::new(-24.0, 20.0, 24.0)` offset).

- Old speed at default zoom: `39.395 * 1.2 ≈ 47.27` blocks/sec.
- New constant speed: `47.27 * 1.2 ≈ 56.73` blocks/sec.

Remove `CameraSettings::rts_pan_speed_factor` and replace with
`CameraSettings::rts_pan_speed: f32` (flat blocks/sec, no longer
multiplied by `orbit_radius`).
