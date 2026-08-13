# 005-c - Manual verification checklist

Automated checks already confirmed: `local_chunk_index` (region-relative
chunk position), a load against a region the save doesn't have failing
cleanly (`None`, no panic), and a full decode+mesh of a real chunk from the
dev machine's save — all in `src/chunk_pipeline.rs`, plus `cargo
build`/`cargo test` are clean (37 tests). The item below needs an actual
human at the window — `cargo run` and confirm it.

- [ ] **Chunks stream in without stutter** — run a debug build
      (`cargo run`) against the real save, fly the camera away from the
      eagerly-loaded startup region (past its edges) for a minute or two,
      and confirm: new chunk mesh entities appear within a few frames of
      entering render distance (watch the console for the pipeline picking
      up `to_load` coordinates), the frame rate doesn't visibly hitch when
      a batch of chunks completes, and nothing panics.
