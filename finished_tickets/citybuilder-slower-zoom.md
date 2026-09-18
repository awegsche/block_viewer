# Citybuilder camera zoom is too fast

`CameraMode::Rts` scroll-zoom shared `CameraSettings::orbit_zoom_sensitivity`
(0.15) with the block viewer's `Orbit` mode. Because the zoom update is
multiplicative (`orbit_radius *= 1 - scroll * sensitivity`), a single
accumulated wheel notch could swing the radius 30-45%, and the absolute
jump gets bigger the further zoomed out the camera already is — matching
the reported "zooming out suddenly jumps way out."

## Fix

Added a separate `rts_zoom_sensitivity` (0.045, vs 0.15 for `Orbit`) so the
citybuilder's zoom can be tuned independently of the block viewer's, and
lowered it to slow zoom overall, including at large `orbit_radius`.

`cargo test --lib camera::` passes (11/11).
