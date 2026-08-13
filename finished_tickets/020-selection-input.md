# 020 - Selection input: click to anchor, keystrokes to extend

## Status
Done (see Resolution)

## Depends on
019 (`Selection` / `SelectionBounds` — this ticket only writes them).

## Goal

Click a block to start a selection, then grow or shrink each of the box's
six faces with the keyboard. This is the interaction the whole feature is
judged on, and its main risk is not the maths — it's colliding with the
camera controller, which already owns most of the keyboard.

## What the camera already owns

From `src/camera.rs::drive_camera`, do not reuse any of these:

| Binding | Used for |
| --- | --- |
| `W`/`A`/`S`/`D` | fly movement |
| `Q`/`E` | fly down/up |
| `ShiftLeft` | sprint multiplier |
| `Tab` | fly ↔ orbit mode toggle |
| right mouse | fly look + cursor grab |
| left mouse | orbit look drag |
| scroll wheel | fly speed / orbit zoom |

Free and used by this ticket: the arrow keys, `PageUp`/`PageDown`,
`ControlLeft`, `AltLeft`, `Escape`.

## The scheme

**Each key pushes one face of the box outward.** Not "grow on an axis" —
per-face is what makes repeated presses feel like dragging a wall, and it's
what "extend in each dimension" actually means when the anchor isn't in the
middle.

| Key | Moves |
| --- | --- |
| `→` | the +X (east) face outward |
| `←` | the −X (west) face outward |
| `↑` | the −Z (north) face outward |
| `↓` | the +Z (south) face outward |
| `PageUp` | the +Y (up) face outward |
| `PageDown` | the −Y (down) face outward |
| + `AltLeft` | pull that same face back in instead |
| + `ControlLeft` | step of 16 (a chunk) instead of 1 |
| `Escape` | clear the selection |

Note `↑` is **north** = MC `−Z` (see `src/world/mesh.rs`'s axis mapping);
`↑`/`↓` map to screen-forward/back only when the camera happens to face
north. See "Camera-relative arrows" below.

Rules:

- Directions are **world-absolute**, not camera-relative.
- Retracting clamps at size 1 on that axis — a face can never cross its
  opposite face, and the selection never becomes empty.
- `PageUp`/`PageDown` additionally clamp to 019's world build limits.
- Keys **repeat while held** (Bevy's `ButtonInput` has no auto-repeat;
  either use `just_pressed` and require tapping, or add a small
  `Timer`-driven repeat after a ~300 ms hold). Recommend implementing
  repeat: growing a 40-block box one tap at a time is the difference
  between "usable" and "demo".

## Click to anchor

Left click on terrain → `Selection.0 = Some(SelectionBounds::from_anchor(
block))`, replacing whatever was there.

Reuse `camera::block_under_cursor(camera, camera_transform, cursor,
settings.max_ray_distance, &decoded_world)` — the same DDA ray-march the
block inspector (007) and orbit re-aiming (006) use, returning Minecraft
block coordinates. Do not write a second raycast; if this one needs a
change, change it in `camera.rs` so all three callers move together.

Three gates, all necessary:

1. **`!egui_input.pointer`** (`camera::EguiInputCapture`) — otherwise
   clicking a button in the new selection panel (021) also re-anchors the
   selection to whatever terrain is behind the panel.
2. **Not a drag.** Left mouse is orbit-mode look. Fire on *release*, and
   only if the cursor moved less than a few pixels since press (accumulate
   `AccumulatedMouseMotion` between press and release). This also stops a
   fly-mode look-drag that happens to end over terrain from re-anchoring.
3. **Ordering.** This system must run in a set ordered the same way
   `camera::CameraSet` is — `.after(ui::UiPanelSet)` — so `EguiInputCapture`
   is this frame's value, not last frame's. See `ui::UiPanelSet`'s docs.

If the ray hits nothing loaded, do nothing (and leave any existing selection
alone) rather than clearing — a missed click over sky shouldn't destroy a
box the user spent twenty keystrokes building.

### Keyboard gating

Every key read above must be gated on `!egui_input.keyboard`, or typing a
coordinate into 007's jump panel (or a bounds field in 021) will also move
the selection faces.

## Camera-relative arrows

Deliberately *not* done here. Absolute directions are predictable and
testable; camera-relative ones need a "snap the camera yaw to the nearest of
four quadrants" rule, and get confusing exactly when the camera is at 45°.
If it turns out world-absolute is disorienting in practice (a real
possibility — note it in `todo.md` as something to judge at the window), the
fix is a small remap in front of the same face-moving functions, which is
why those functions should take a face/direction, not a `KeyCode`.

## Structure

Keep the pure part testable: a function like

```rust
fn move_face(bounds: &mut SelectionBounds, face: Face, blocks: i32)
```

with the system doing nothing but key reads → `(Face, i32)` → this call.
Every test below then runs without an `App`.

## Tests

- Each of the six faces moves the right bound in the right direction, and
  only that bound.
- `Alt` (retract) shrinks, and clamps at size 1 rather than inverting the
  box or producing a zero-volume selection.
- `Ctrl` gives a step of 16, in both directions, including combined with
  `Alt`.
- Y retraction/extension clamps at both build limits.
- Extending north (`↑`) *decreases* `min.z` — the one that catches an
  inverted Z (the ticket's most likely bug, given `bevy.z = -mc.z`).
- Clearing leaves `Selection.0 == None`.

## Done when

- Clicking a block shows a 1x1x1 box on it, and the six keys grow it into a
  region whose faces land where you'd expect from the direction you pressed.
- Typing in an egui field or dragging a slider never moves the selection;
  dragging the camera with left mouse never re-anchors it.
- `cargo test` passes.
- `todo.md` gets a manual check: anchor on a recognisable block, extend each
  of the six directions in turn and confirm the box grows on the side you
  pressed (facing north, then again facing south — that's the check for
  world-absolute arrows being usable at all, and for the Z sign); confirm
  `Ctrl` steps a chunk's worth; confirm holding a key repeats at a
  comfortable rate; confirm `Escape` clears it.

## Resolution

`src/selection/input.rs` (`SelectionInputPlugin`, added by 019's
`SelectionPlugin`), with `selection::SelectionInputSet` ordered
`.after(ui::UiPanelSet)` in `main.rs` alongside `camera::CameraSet`. 17
tests; every binding, clamp and gate in the ticket is covered, and all but
three of them run without an `App`.

Implemented as specified. Decisions worth recording:

- **`Face` owns its own key**, via `Face::key()`, rather than the system
  matching a `KeyCode` to a face. That keeps the "the remap for
  camera-relative arrows goes in *front* of `move_face`" property the ticket
  asked for, and it means the North/`↑` pairing — the ticket's flagged
  most-likely bug — is asserted in the same test as the `min.z` decrease.
- **Per-face repeat timers, not one.** Six independent `RepeatTimer`s (one
  per face, held in a `Local`), so holding `↑` and `→` together grows the box
  diagonally instead of one key starving the other. The ticket only asked for
  repeat at all; per-face fell out of the same array for free.
- **Repeat is a countdown, not an accumulated elapsed time.** A frame long
  enough to owe hundreds of repeats (a chunk batch completing, a breakpoint)
  is capped at `MAX_STEPS_PER_FRAME` = 8 and then *forfeits* the remainder
  rather than banking it — an accumulator would pay the backlog out over the
  following frames and fling the face out anyway. Tested both halves: the cap
  holds, and the timer resumes at the ordinary rate rather than stalling.
- **Timers tick even with no selection.** Otherwise a key already held when a
  click anchors a box would fire its whole banked repeat immediately, and the
  first click would produce a 20-block box. Tested.
- **egui keyboard capture zeroes the keys rather than returning early**, so
  the repeat timers reset too: a key held as a panel takes focus doesn't
  resume mid-repeat when focus comes back.
- **A left press that starts over an egui panel is discarded outright**, not
  re-tested at release — the pointer leaving the panel mid-drag must not turn
  a panel interaction into an anchor. That's a third gate beyond the ticket's
  press-time `!egui_input.pointer` and the drag-distance check, both of which
  are also there (`CLICK_DRAG_TOLERANCE` = 4 px, accumulated from
  `AccumulatedMouseMotion` between press and release).
- **Horizontal faces saturate.** X/Z are unclamped by 019's design, so a long
  repeat can walk a face toward `i32`'s range; `saturating_add`/`_sub` keep it
  from wrapping into a box on the far side of the world. Tested at both ends.

### Not done, deliberately

Camera-relative arrows, per the ticket's own reasoning — and face
highlighting for "which face moves next" (019 left that to this ticket if the
extrusion turned out hard to follow; that's a judgement that needs the window,
so it's noted in `todo.md` rather than guessed at here).
