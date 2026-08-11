# To-do

Manual/visual checks that need a human at the window — Claude doesn't run
these itself (see CLAUDE.md's "Manual/visual verification").

- [ ] **005-c async pipeline: chunks stream in without stutter.**
  `cargo run` (debug build) against the real save, fly the camera away
  from the eagerly-loaded startup region (past its edges) for a minute or
  two. Confirm: new chunk mesh entities appear within a few frames of
  entering render distance (watch the console for the pipeline picking up
  `to_load` coordinates), the frame rate doesn't visibly hitch when a
  batch of chunks completes, and nothing panics. Checklist:
  `finished_tickets/005-c-async-pipeline.manual-verification.md`.
