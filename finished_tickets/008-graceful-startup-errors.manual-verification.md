# 008 - Manual verification checklist

Automated checks already confirmed: `cargo build`/`cargo check`/`cargo clippy`
are clean (only pre-existing warnings, unrelated to this ticket) in both
`bevy_minecraft` and `../ranvil`, and `cargo test` passes in both (48 tests in
`bevy_minecraft`, up from 42 — new coverage for the startup-directory
fallback, the empty-directory/missing-directory error paths, and a region
that fails to load being remembered rather than retried forever; 6 tests in
`../ranvil`'s `save_tests.rs`, up from 4 — `get_saves_from_instance` no
longer failing its whole listing over one malformed entry). The items below
are the ticket's "Done when" criteria and need an actual human at the
window — `cargo run` and confirm them.

- [ ] **No `.minecraft` directory at all: window still opens.** Run with a
  saves directory argument pointing at a path that doesn't exist, e.g.
  `cargo run -- C:\definitely\does\not\exist`. Confirm: the window opens
  normally (empty world, camera at the origin), the console prints a single
  `block_viewer: could not read saves directory ...` line (no panic/
  backtrace), and the "Save" egui window shows that same reason in red at
  the top.
- [ ] **Empty `saves/` folder: window still opens.** Point it at an empty
  directory, e.g. `mkdir` a scratch folder and `cargo run -- <that folder>`.
  Confirm: window opens, console prints `block_viewer: no Minecraft saves
  found under ...`, and the "Save" window shows the same message in red.
- [ ] **Normal startup is unaffected.** `cargo run` with no arguments against
  the real `.minecraft/saves` directory. Confirm it loads the first save
  exactly as before — no red message in the "Save" window, terrain streams
  in normally.
- [ ] **A corrupted `.mca` file is skipped, not fatal.** Pick one region file
  under the active save's `region/` folder that's away from where you're
  about to fly (back up the original first!) and truncate or zero part of
  it (e.g. `type nul > path\to\r.X.Z.mca` to make it empty, or truncate to a
  few hundred bytes). Fly the camera toward that region. Confirm: the
  console prints exactly one `block_viewer: skipping region (X, Z) — failed
  to load ...` line (not one per chunk), the rest of the world keeps
  streaming in around the hole normally, and nothing panics. Restore the
  original file afterward.
- [ ] **Loading a save at runtime after a failed startup clears the
  message.** Start the app pointed at an empty/missing directory (as
  above) so the red startup message shows, then use the "Save" window's
  picker to load a real save (needs a way to reach one — e.g. run with no
  CLI arg redirection and instead temporarily empty the real `saves/`
  folder, or test this against a second save if two are available).
  Confirm the red message disappears once a real save is loaded and doesn't
  come back.

Checklist file: `finished_tickets/008-graceful-startup-errors.manual-verification.md`.
