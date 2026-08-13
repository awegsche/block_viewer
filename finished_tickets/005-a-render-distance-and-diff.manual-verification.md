# 005-a - Manual verification checklist

Automated checks already confirmed: `desired_chunks` (radius/count),
`diff_chunks` (entering/leaving/unchanged buckets), and the Bevy-to-Minecraft
chunk-coordinate conversion are covered by unit tests in `src/streaming.rs`,
and `cargo build`/`cargo test` are clean. The item below needs an actual
human at the window — `cargo run` and confirm it.

- [ ] **Moving the camera changes the computed delta** — fly (or orbit) the
      camera across a chunk boundary and confirm a new
      `Chunk streaming: camera entered chunk (...)  (N to load, M to unload)`
      line prints each time the camera's chunk coordinate changes, and stays
      quiet while it sits still within one chunk.
