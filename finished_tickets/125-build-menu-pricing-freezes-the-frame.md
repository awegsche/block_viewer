# 125 - Build menu pricing walks a synonym group once per path: ~1.5 s a frame

## Status
Done — `cargo test --lib`: 1143 passed (was 1141; new: the large-group and shipped-buildings pricing bounds). Measured in the running game (see
below): 1 fps -> 60 fps.

## Symptom

Right after ticket 124's loading screen: "the initial loading is super
quick but then the game is very laggy." And, with hindsight, ticket 123's
"chunk loading seems extremely slow" — loads are polled once per frame, so
a 1.5 s frame is ~1 chunk/s however fast the tasks are.

## Finding it

The new `Streaming health` console line (one a second while any chunk
work is queued or in flight: phase, fps, frame time, every queue depth)
showed the frame time going from 18 ms under the loading screen to
**1500-1800 ms** the moment `CityPhase::Playing` began — while the re-mesh
queue, my first guess, drained at the same rate in both. Running with the
gating disabled put the 1.5 s frames at frame 1, so the lag predates 124;
the loading screen only exposed it by letting the pipeline run at 55 fps
first.

`cargo build --features bevy/trace_chrome` and a span summary of the trace
made it unambiguous: `city::ui::build_menu::build_menu_panel` was
**8.67 s of an 8.68 s `Update`** over five frames, 1.8 s a call.

## Cause

`economy::plan_payment` prices every menu row every frame, and `cover`
chases a short item's routes recursively. A synonym group (ticket 075,
"every log is a log", 48 members) is a clique, and the cycle guard only
excluded items on the *current chain* — so a short `oak_log` recursed into
47 synonyms, each into 46, each into 45, to the depth cap of 4: ~5 million
`cover` calls per priced cost, each allocating a `Vec` of routes. A probe
against the shipped definitions with an empty pile: 6 ms for a
planks-and-cobble row, **230 ms** for each row costing `oak_log`
(the three mines and the lumber farm), ~1 s per pass of the menu.

## Fix

A group is expanded **once per chain**. `routes_to` now takes the chain
above the item and leaves the group-mates out when an ancestor is in the
same group — chasing a synonym then only looks for *conversions* into it
(`birch_log -> birch_planks`), never its own synonyms, which are exactly
the list the caller is already walking. Same answers (every group-mate is
a direct route of every other, and conversions into mates are still
followed); the per-cost work drops to ~10-35 µs.

Also kept from the investigation:

- `city::loading::StreamingHealthPlugin` — the health line above, silent
  when idle. It's what makes "the game is laggy" answerable from the
  console next time.
- `len()` on the re-mesh/reload queues, for it.

## Measured

Same save, same run shape, before -> after:

```
@2.2s [Playing] fps 1 (1751.6 ms): remeshes 0+583
@3.2s [Playing] fps 1 (1500.7 ms): remeshes 0+567
```
```
@2.0s [Playing] fps 69 (15.1 ms): remeshes 0+508
@3.0s [Playing] fps 60 (17.0 ms): remeshes 0+260
@4.2s [Playing] fps 60 (16.8 ms): ... — idle
```

The ~500 re-meshes left when the screen drops (each landed chunk queues
its already-loaded neighbours to close the seam) drain in two seconds at
60 fps and are not visible. Not worth a two-pass startup load.

## Noticed, not done

`Chunk streaming: … (166 to load …)` recurs every 2 s: the 166 coordinates
in the disc with no chunk behind them get re-dispatched by ticket 062's
periodic retry, each a task that hits the (cached) region and returns
`None`. Cheap, but it's 166 tasks every two seconds forever while the
camera sits still. A "known missing" set with a longer retry would stop
it.
