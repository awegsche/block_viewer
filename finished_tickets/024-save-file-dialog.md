# 024 - Save-file dialog: pick a filename, run the export

## Status
Done — pending the manual check in `todo.md`

## Depends on
021 (the "Export…" button), 022 (the extraction task), 023 (the writer).
Landing last wires the three together.

## Goal

Clicking "Export…" opens a native save-file dialog; picking a filename runs
the extraction and writes the file; the panel reports what happened. This is
plumbing — the interesting work is all in 022 and 023 — but it's the ticket
that makes the feature usable by a human.

## The dialog

`rfd` (Rust File Dialog) — a native Windows save dialog with an extension
filter, ~one new dependency and no window handling of our own.

- **Why not an in-app egui file browser?** No new dependency, and it could
  default to the save's own folder — but directory navigation, filtering,
  overwrite confirmation, and drive switching are all then ours to build and
  maintain. `rfd` gets the platform's own dialog, including the
  overwrite prompt, for free.

### It must not block the main thread

`rfd::FileDialog::save_file()` is a blocking modal. Called from an `Update`
system it stalls the render loop for as long as the dialog is open — the
window stops repainting and Windows may mark it unresponsive.

Use `rfd::AsyncFileDialog` and poll the returned future the same way
`src/chunk_pipeline.rs` polls chunk loads:

- spawn on `AsyncComputeTaskPool`, keep the `Task<Option<FileHandle>>` in a
  resource,
- poll with `block_on(poll_once(&mut task))` in an `Update` system,
- `None` means the user cancelled — clear the task and do nothing else.

Note `rfd` has platform threading rules (on macOS the dialog must run on the
main thread; `AsyncFileDialog` handles this internally). Windows is the
target here, but stick to the async API rather than the blocking one so the
constraint never has to be revisited.

### Dialog configuration

- Filter: `("Minecraft structure", &["nbt"])`.
- Default filename from the selection: something like
  `blueprint_<x>_<y>_<z>.nbt` using `Blueprint::origin` / the selection's
  `min`, so consecutive exports don't all default to the same name.
- Default directory: the loaded save's `generated/minecraft/structures/`
  if it exists (that's where the game looks for structure files), else the
  save directory, else whatever `rfd` defaults to. `LoadedSave.0.meta.path`
  has the save root. Check the directory exists rather than creating it —
  a viewer shouldn't be making folders inside someone's save.

## The flow

```text
Export… clicked
  -> AsyncFileDialog task            (this ticket)
  -> Some(path)  -> extraction task  (022)
                 -> write            (023)
                 -> report to panel  (021)
  -> None        -> back to idle
```

Model it as an explicit state on the export resource — `Idle | Choosing |
Extracting { progress } | Writing | Done { path, blocks } | Failed { error }`
— rather than a scatter of `Option<Task<_>>` fields. The panel then renders
one match arm per state, and it's obvious that the export button is disabled
in every state but `Idle`.

Do the extraction **after** the filename is chosen, not before: extraction
is the expensive half, and cancelling the dialog is common.

## Reporting

The panel (021) shows the current state: a progress readout while
extracting, then either the written path and block count, or the error.
Keep the result visible until the next export starts — a status line that
vanishes after one frame tells the user nothing. Log the same to the console.

Errors that need real messages, not `unwrap`s (this app's ticket 008 set the
precedent that nothing user-triggerable panics): the chosen path isn't
writable, the disk is full, the extraction hit unreadable region files.

## Tests

Dialog code can't be tested headlessly. Test what's around it:

- The default-filename generator for a given selection origin, including
  negative coordinates (no path-hostile characters in the result).
- Default-directory resolution: picks `generated/minecraft/structures/` when
  it exists, falls back cleanly when it doesn't, and never creates it.
- The state machine's transitions, including cancel-from-`Choosing` and
  failure-from-`Writing`, both returning to a state where the button is
  enabled again.

## Done when

- "Export…" opens a real Windows save dialog, cancelling it leaves the app
  in a working state, and choosing a name writes a file that ticket 023's
  reader test would accept.
- The window keeps repainting while the dialog is open and while a large
  extraction runs.
- `cargo test` passes.
- `todo.md` gets a manual check: export with the dialog open for a while and
  confirm the app stays responsive behind it; cancel and confirm nothing is
  written and the button re-enables; export a large selection and watch the
  progress readout advance rather than the UI freezing; export to a
  read-only location and confirm an error message rather than a crash;
  confirm the default filename and starting directory are sensible.

---

## Resolution

New module `src/blueprint/export.rs`; the panel and `blueprint/mod.rs`
changed around it, nothing else in the app did.

### What was built

- `BlueprintExport`, one resource holding one `ExportState`
  (`Idle | Choosing | Extracting | Writing | Done | Failed`) exactly as the
  ticket asked, driven by four systems chained *through* the extraction's
  two so that a whole export can advance several steps in one frame:
  `poll_extraction -> drive_write -> poll_write -> start_dialog ->
  poll_dialog -> start_extraction`. `BlueprintPlugin` now adds
  `BlueprintExportPlugin` rather than registering the extraction systems
  itself, because an extraction is the middle of an export, not a parallel
  feature.
- `rfd::AsyncFileDialog` on `AsyncComputeTaskPool`, polled with the same
  `block_on(poll_once(..))` as everything else. Filter
  `("Minecraft structure", ["nbt"])`, default name
  `blueprint_<x>_<y>_<z>.nbt` from the selection's minimum corner, default
  directory `generated/minecraft/structures/` -> save root -> `rfd`'s own.
- `BlueprintExtraction::take_blueprint`. The blueprint used to be logged and
  dropped at the end of `poll_extraction`; it's now parked in a `finished`
  field for `drive_write` to collect. Cleared when the next extraction
  starts, so an uncollected one isn't left holding megabytes.
- The panel's `extraction_status` became `export_status`, one arm per state,
  ending in a green "Wrote N blocks to:", the **full path** under it, and a
  Copy path button.

### Decisions worth knowing about

- **The export layers on the extraction rather than absorbing it.** The
  ticket's state list implies one machine owning everything, but
  `BlueprintExtraction` already has its own progress reporting and its own
  one-at-a-time rule, and re-implementing those inside `ExportState` would
  have been a second copy of both. `Extracting` therefore holds only the
  chosen path and reads progress from the other resource.
- **The bounds are captured in `Choosing`,** not re-read when the dialog
  closes: a native dialog can sit open for a minute, and the box the user
  clicked Export on is the one they meant.
- **Cancelling leaves no status line.** `Choosing -> Idle` with nothing
  reported, because a cancel is a decision rather than a result. Only `Done`
  and `Failed` are sticky.
- **The extraction's failures are reported by the export, not twice.**
  `extraction_summary` renders the `Done` arm only; `ExtractOutcome::Failed`
  reaches the user through `ExportState::Failed`, so one problem is one red
  line.
- **No `set_parent` on the dialog.** It would make the dialog properly owned
  by the game window, but rfd wants a `HasWindowHandle` and Bevy 0.15 only
  offers one behind an `unsafe` main-thread-only accessor plus a direct
  `raw-window-handle` dependency. Skipped as speculative — the panel's
  "Choosing a file…" line is the cheap mitigation, and `todo.md` asks
  whether the dialog can actually get lost behind the window.
- **No progress figure while writing.** The writer streams into a gzip
  encoder and has no counter to publish, so it's a spinner. A million blocks
  is ~1.25 s (023's `measure_large_write`); a fake percentage would be worse
  than an honest spinner.

### Tests

12 new in `export.rs`, all of the "what's around the dialog" kind the ticket
asked for: the default name (including negative coordinates, checked against
the Windows illegal-character set), directory resolution in all four cases
(structures folder, save-root fallback, empty path, deleted directory) plus
an assertion that looking never *creates* the folder, and the state machine
driven through a real `App` with hand-spawned tasks standing in for the
dialog — cancel-to-`Idle`, path-to-`Extracting`, extraction-failure-to-
`Failed`, write-failure-to-`Failed`, write-success-to-`Done`, and `Done`
surviving three more frames.

`cargo test` is 193 passing; no new clippy warnings.
