# 005-d - Manual verification checklist

Automated checks already confirmed: `InFlightChunkLoads::cancel_out_of_range`
drops a task whose coordinate left the desired set while leaving one still
inside it running (`src/chunk_pipeline.rs`), and `cargo build`/`cargo test`
are clean (38 tests). The two items below are the ticket's actual "Done
when" criteria and both need an actual human at the window — `cargo run`
and confirm them.

- [ ] **Memory stays flat while flying** — run a debug build (`cargo run`)
      against the real save and fly the camera in one direction for a few
      minutes, past several render-distance widths of terrain. Watch RSS
      (Task Manager or similar) over that time: it should climb initially
      as the working set fills, then roughly plateau rather than climb
      monotonically once chunks are both loading ahead and unloading
      behind.
- [ ] **Reversing near the loading edge doesn't spawn stale chunks** — fly
      toward the edge of render distance so new chunks start loading
      (watch the console for `to_load` pickups), then reverse direction
      before they finish. Confirm no chunk mesh pops in noticeably behind
      the camera's current render-distance ring right after the reversal —
      that would mean an in-flight load finished and spawned despite the
      coordinate no longer being desired.
