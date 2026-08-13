# 005-f - Manual verification checklist

Automated checks already confirmed: `remesh_chunk_column_closes_a_seam_once_the_neighbor_is_available`
exercises the re-mesh task function directly (a lone block's east face goes
from rendered to culled once the east neighbour column is passed in), the
`cancel_out_of_range`/dedupe unit tests for `InFlightChunkRemeshes` and
`PendingChunkRemeshes` pass, and `cargo build`/`cargo test` are clean (42
tests). The item below is the ticket's actual "Done when" criterion and
needs an actual human at the window — `cargo run` and confirm it.

- [ ] **No lingering seam at the loading frontier** — run a debug build
      (`cargo run`) against the real save and fly past the loading frontier
      at normal fly speed (the direction terrain is actively streaming in
      ahead of the camera). Watch the boundary between a chunk that just
      finished loading and its already-loaded neighbour: a seam (a visible
      gap/flicker where the two meshes meet, from the neighbour's stale
      "missing = air" face) should close within the one frame the new
      chunk is legitimately still in flight — not persist or visibly
      trail the camera the way it would before this ticket.
