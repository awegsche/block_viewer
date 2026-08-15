# 029 - The solid selection pass isn't depth tested in practice

## Status
Done — pending the look call in `todo.md`

## Depends on
026 (the two-pass gizmo). This is a bug in the constant 026 picked.

## The report

> the selection frame gizmo is still hard to understand (I don't see which
> part is underneath the surface, is the double pass rendering working at
> all?)

Both passes are running. The problem is that `SOLID_DEPTH_BIAS = -0.02`
pulls the "depth tested" pass so far toward the camera that terrain stops
occluding it at ordinary viewing distances — so both passes draw everywhere,
and the box looks exactly like the single always-in-front pass ticket 019
had.

## Why -0.02 is enormous

026's doc comment treats the perspective-correct bias as a convenience
("the nudge grows with distance, so one value covers the whole view").
It's the opposite: the nudge grows *proportionally*, so one value is a
fixed percentage of the view distance, and 2% of the view distance is
metres.

From `bevy_gizmos-0.15.0/src/lines.wgsl`, for a negative bias:

```wgsl
depth = clip.z * exp2(-depth_bias * log2(clip.w / clip.z - EPSILON));
```

Bevy uses a reverse-Z infinite perspective projection, so `clip.z = near`
and `clip.w = d` (the view-space distance). With `k = -depth_bias`:

```
depth' = near * (d / near)^k        →  the line renders as if at  d / (d/near)^k
```

`PerspectiveProjection::default()` has `near = 0.1`, and `camera.rs` uses
the default. So the fraction of the distance the line is pulled forward is
`1 - (d/0.1)^-k ≈ k * ln(d / 0.1)`:

| distance | `k = 0.02` (now) | `k = 0.0002` (proposed) |
| --- | --- | --- |
| 10 blocks  | 0.9 blocks | 9 mm |
| 20 blocks  | 2.0 blocks | 2 cm |
| 50 blocks  | 5.8 blocks | 6 cm |
| 200 blocks | 24 blocks  | 24 cm |

A selection box is normally built around terrain within a few blocks of the
surface, and the whole buried part of it is within the 1–6 blocks the bias
punches through. That is exactly the region the two passes were supposed to
tell apart.

## The fix

`SOLID_DEPTH_BIAS = -0.0002`, and a doc comment that says the bias is
proportional rather than implying it self-corrects.

That is still ~1000x more than float32 reverse-Z needs to win a coplanar
depth test (relative depth resolution is ~1e-7), so the z-fighting the bias
exists for stays fixed: a selection boundary sitting exactly on a block face
still resolves in the gizmo's favour.

## The second half: the dots are 2px

Once the solid pass is genuinely occluded, the buried pass has to read as a
different thing on its own. Right now it barely does.

`fragment_dotted` in the same shader computes the dash pattern in units of
`line_width`, period 2:

```wgsl
alpha = 1 - floor((in.uv * in.position.w) % 2.0);
```

so a dash is `line_width` pixels long and the gap after it is
`line_width` pixels. At `BURIED_LINE_WIDTH = 2.0` that's a 2px-on/2px-off
stipple, which at any real viewing distance reads as a solid line at about
half brightness rather than as a dashed one. The dash length cannot be set
independently of the width in Bevy 0.15 — the only knob is the width.

Raise `BURIED_LINE_WIDTH` to `4.0`: 4px dashes with 4px gaps, unmistakably
a dashed line. It makes the buried pass *wider* than the 2.5px solid one,
which is the wrong way round for the overlap in open air, so drop
`BURIED_ALPHA` from 0.55 to 0.4 to compensate — in open air the buried pass
then reads as a faint broken halo around a crisp solid line, and underground
it's all there is.

Both numbers are still guesses by someone who can't look at the window; the
manual check names which one to turn for each way it can be wrong.

## Tests

The bias and the width are single constants feeding a shader — there is
nothing to assert headlessly that isn't a tautology, and `slice_heights` /
`box_transform` (the parts that *are* arithmetic) already have tests. So:
no new tests, and `cargo test` must stay green.

## Done when

- `SOLID_DEPTH_BIAS` is `-0.0002` with a doc comment that explains the
  proportionality, and `BURIED_LINE_WIDTH` / `BURIED_ALPHA` are 4.0 / 0.4.
- `cargo test` passes.
- `todo.md` gets the look call, superseding 026's: build a box running from
  open air down into a hillside and confirm the buried edges are dashed and
  the open-air ones solid, and that the boundary tracks the surface as the
  camera orbits. If the box now z-fights where it's coplanar with a flat
  surface, `SOLID_DEPTH_BIAS` is too small; if edges still punch through
  terrain, it's too large. If the dashes read as clutter, lower
  `BURIED_LINE_WIDTH`; if the buried part is invisible at distance, raise
  `BURIED_ALPHA`.

---

## Resolution

Three constants in `src/selection/gizmo.rs`, and the doc comments that got
them wrong.

- `SOLID_DEPTH_BIAS`: `-0.02` -> `-0.0002`. The old doc comment presented the
  perspective-correct bias as a convenience; it now says out loud that the
  bias is a *fraction of the view distance* and that this is why the number
  has to be tiny, with the arithmetic that gets from Bevy's shader to "~6 cm
  at 50 blocks".
- `BURIED_LINE_WIDTH`: `2.0` -> `4.0`, documented as being the dash length as
  well as the width, since `fragment_dotted` measures its pattern in units of
  `line_width` and there is no separate spacing knob in Bevy 0.15.
- `BURIED_ALPHA`: `0.55` -> `0.4`, paying for the buried pass now being wider
  than the solid one where the two overlap in open air.

No new tests: all three feed a shader, and the parts of the module that are
arithmetic (`slice_heights`, `slice_step`, `box_transform`) already have
theirs. `cargo test` is 193 passing.

### Why 026 couldn't have caught this

026's own manual check asked the right question — "if the whole box looks
solid the depth test isn't happening" — and the answer came back as "I can't
tell". That's the failure mode of a look call whose two states differ by an
amount the bug erased: with the solid pass punching several blocks through
terrain, the only difference left between the passes was a 2px stipple at
0.55 alpha drawn directly on top of a solid line. Both halves of this ticket
exist to make that check answerable next time.
