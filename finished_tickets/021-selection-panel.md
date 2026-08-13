# 021 - Selection panel: bounds readout, typed edits, and the export button

## Status
Done (see Resolution)

## Depends on
019 (`Selection`). Usable without 020 (type the bounds in by hand), and
gains the "Export…" button from 024.

## Goal

An egui window showing exactly what is selected, letting the bounds be typed
in directly, and hosting the export affordance the rest of the group builds
toward. It's also where the keyboard scheme gets documented for a human —
020's bindings are undiscoverable otherwise.

## Where it goes

`src/ui/selection_panel.rs`, added to `ui::UiPlugin`'s `UiPanelSet` tuple
alongside the four existing panels. It reads/writes `Selection`; it does
**not** get its own plugin — panels live in `src/ui/`, selection state lives
in `src/selection/`, and this ticket is the only place they meet.

Follow `src/ui/navigate.rs` for the typed-coordinate pattern: `String`
scratch fields committed on a button/Enter rather than parsed every frame,
so a half-typed `-` doesn't jump the bounds to zero.

## Contents

- **No selection:** a one-line hint naming the click-to-anchor gesture,
  matching the block inspector's `"(hover the world to inspect a block)"`
  tone.
- **Bounds:** `min` and `max` as six editable integer fields (Minecraft
  coordinates — same convention as the block inspector's `Block: x, y, z`
  readout, so the two can be cross-read). Committing a typed pair goes
  through `SelectionBounds::normalized`, so entering an inverted range
  fixes itself instead of producing an invalid box.
- **Size and volume:** `size` as `W x H x D` and `volume` as a block count.
  This is the number that decides whether an export is a good idea.
- **Anchor:** the clicked block, read-only.
- **Volume warning:** above a threshold (start at 1,000,000 blocks —
  100x100x100), colour the volume line and say the export will be slow.
  Above a hard cap (start at 16,000,000), disable the export button
  entirely. Both as named `const`s with a comment, tunable once 022 has
  real timings. Vanilla structure blocks cap at 48x48x48 = 110,592 blocks —
  worth mentioning in the panel's hover text, since a blueprint above that
  can't be loaded back by a structure block even though our writer will
  happily emit it.
- **Clear** button (same effect as 020's `Escape`).
- **The key legend:** the six extrusion keys plus the `Alt`/`Ctrl`
  modifiers, in a collapsible `egui::CollapsingHeader` so it isn't in the
  way once learned. Generate it from the same `Face` enum 020 defines rather
  than retyping the bindings as strings — a legend that drifts from the code
  is worse than none.
- **Export…** button — inert in this ticket (or gated behind
  `#[allow(unused)]` until 024 wires the dialog to it). 022 gives it a
  block count to report; 023 gives it a file to write; 024 gives it a
  filename. Landing the button here means 024 is a small, purely
  plumbing-shaped ticket.

## Interactions to get right

- Typing in any field sets `EguiInputCapture::keyboard`, which 020 already
  gates on — but *verify* it, because this is the panel most likely to be
  focused while someone reaches for an arrow key.
- The panel must not re-anchor the selection when clicked over terrain; 020
  gates on `EguiInputCapture::pointer` for exactly this, and this panel is
  the thing that tests it.

## Tests

Panel bodies are hard to test without an egui harness; put the logic in free
functions and test those:

- Committing a typed min/max pair normalizes an inverted range.
- Rejecting non-numeric input leaves the previous bounds untouched.
- The volume-threshold classifier (ok / warn / over cap) at each boundary.

## Done when

- The panel shows a live, correct readout as 020's keys move the box.
- Typing bounds in moves the box; typing garbage does nothing.
- `cargo test` passes.
- `todo.md` gets a manual check: with the panel focused, confirm arrow keys
  type into the field rather than moving the selection; confirm clicking
  the panel over terrain doesn't re-anchor; confirm the volume warning
  appears at a large selection and the export button greys out past the cap.

## Resolution

`src/ui/selection_panel.rs`, added to `UiPlugin`'s `UiPanelSet` tuple
alongside the four existing panels — no plugin of its own, as specified. 11
tests (10 here, 1 in `selection::input`), none of which need an egui harness.

Implemented as specified. Decisions worth recording:

- **The text fields are re-filled from `Selection`, never written back to
  it directly.** `BoundsDraft` keeps the `SelectionBounds` it was last filled
  from; each frame it re-fills only if `Selection` has changed since. That one
  rule covers all four cases at once: a face key or a click moves the box and
  the fields follow it; an unchanged selection leaves a half-typed `-` alone;
  a commit comes back through the fields *normalized and Y-clamped* rather
  than as typed, because the commit itself changes `Selection`; and `Escape`
  empties them instead of stranding the last box's numbers. Tested as a
  round trip rather than by asserting on the strings the panel writes.
- **A failed parse commits nothing at all**, rather than applying whichever
  of the six fields did parse — a partial commit would move the box somewhere
  nobody asked for. The red "Bounds must be whole numbers." label is live
  (drawn whenever a field doesn't parse) rather than raised on commit, so it
  needs no error state of its own; that's `navigate.rs`'s behaviour too.
- **Both volume thresholds are exclusive**, so exactly 1,000,000 blocks is
  still "fine" and exactly 16,000,000 still exports. Tested at every boundary,
  and at `u64::MAX` — `SelectionBounds::volume` saturates, so the panel can
  genuinely be handed that.
- **The structure-block limit is stated, not enforced.** 48x48x48 = 110,592
  is well below the warn threshold and sits in the volume line's hover text:
  our writer (023) will emit a bigger `.nbt` than Minecraft will load back,
  which is worth knowing but isn't this app's problem to prevent.
- **The legend is generated from `Face`.** `Face::ALL`, `key_label()` and
  `label()` became `pub` for it, and `CHUNK_STEP` too so "step 16 blocks"
  isn't retyped. A test asserts all six faces have distinct keys and distinct
  labels, which is what a copy-pasted match arm would trip. Key labels are
  written as `→ (Right)` / `PgUp` rather than bare arrows, so a missing glyph
  in egui's default font degrades to something still readable.
- **`Export…` is inert**, disabled past the cap, with hover text naming 024.
  It needed no `#[allow(unused)]` — the response is consumed by
  `on_hover_text`, so nothing is unused.
- **No new input plumbing.** 020 already gates its keys on
  `EguiInputCapture::keyboard` and its clicks on `::pointer`, and this panel
  is inside the same `UiPanelSet` those are ordered after, so both gates cover
  it by construction. That they *actually* hold with a text field focused is
  in `todo.md` — it's an egui-focus behaviour, not something the app's own
  code can assert.
