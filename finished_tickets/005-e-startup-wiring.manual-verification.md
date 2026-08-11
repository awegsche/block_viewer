# 005-e - Manual verification checklist

Automated checks already confirmed: `cargo build`/`cargo test` are clean (38
tests) with the eager pre-`App::run()` load and the per-column spawn loop
both deleted. The items below need an actual human at the window —
`cargo run` and confirm them.

- [ ] **Window opens immediately** — run `cargo run` against the real save
      and confirm the window appears right away, with no multi-second
      blank/frozen startup (nothing blocks before `App::run()` anymore).
- [ ] **Terrain streams in visibly** — after the window opens, confirm
      chunk mesh entities appear around the camera's fixed startup position
      within a few seconds (watch the console for `chunk streaming: camera
      entered chunk ...` / pipeline `to_load` messages), rather than an
      empty world.
- [ ] **Flying behaves per the parent ticket's "Done when"** — fly in one
      direction for a few minutes: terrain keeps loading ahead and unloading
      behind, RSS stays roughly flat rather than growing monotonically, and
      there's no frame stutter attributable to chunk loading at the default
      render distance. This overlaps 005-c's and 005-d's own still-open
      manual-verification items (`005-c-async-pipeline.manual-verification.md`,
      `005-d-unload-and-budget.manual-verification.md`) — checking it off
      here can also close those out if they hold up under the same run.
