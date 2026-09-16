# 117 - Mine: inspect-panel section

Design: `MINES_DESIGN.md` ("What the citybuilder shows"). Depends on 116.
Everything a mine does is underground and deliberately invisible to the
citybuilder, so the inspect panel is the *only* place its progress can be
read in-game. Small ticket; the point is that the numbers exist.

## The section

In `city::ui::inspect_panel`, after the producer lines (buffer, state,
warehouse — those already work for a mine via 116), for a building whose
definition has `mine`:

```
Mine
  Level:      2 of 9  (floor Y 44)          // 1-based; total = levels from level_floor(0) down to min_level_y
  Shaft:      bottom Y 44, 20 blocks deep
  Phase:      mining — north arm 36 m, south arm 32 m, row 8 of 25 (north), galleries 15 of 100 closed
              | sinking to Y 40 (1 flight left)
              | mined out
  Job:        digging (96 blocks budget)    // only while a job is pending — the same "in flight" hint gatherer has none of; useful here because a job can take a moment
```

Each line is a pure `fn(..) -> String` next to `work_area_line`, tested the
same way. "Closed" counts galleries whose face is closed for any reason
(length, void run, refusal, bedrock) — the panel doesn't distinguish; the
console log from 116 does.

A **`Rebuild shaft & levels`** debug button is *not* added — 115's
idempotence means "delete `mines.ron`" already is that button, and the
panel shouldn't grow a second way to do it.

## City panel

No new lines: the producer lines already list a mine's buffer and state.
Check that `ProducerState::MinedOut`'s label reads sensibly in that
compact list (`"mined out"`).

## Tests

- Each line fn against a hand-built `MineProgress` in each phase.
- Level total for `first_level_depth: 12`, `floor_y: 64`, `spacing: 4`,
  `min_level_y: 16` is 10 (`52, 48, …, 16`).
- The section doesn't render for a non-mine building.
- `cargo check`, `cargo test --lib`, `cargo clippy` clean.

## Manual check (todo.md)

Select a running mine: the numbers move as jobs land; `sinking` shows
between levels; a mine with `min_level_y` set just under its first level
reaches `mined out` and stays there without the state flickering back.
