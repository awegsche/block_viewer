# 061 - Definition hot reload and an error panel

## Status
Done — implemented and tested (`cargo build`/`cargo test --lib` clean, 552
tests, all passing). Picked up from `tickets/CITYBUILDER_ROADMAP.md`'s
group C: C1 (schema/loader), C2 (tech tree) and — functionally, via ticket
050's build menu already displaying cost/production — C3 were all done;
C4 ("hot reload and an error panel") was the one remaining, non-deliberately-
deferred item left in the group.

## Scope, per the roadmap's own C4 wording

"Definition files reload on change; errors go to an egui panel rather than
a panic or a console line nobody sees. This is what makes balancing
tolerable later." Scoped to the two RON "game data" directories the C group
is actually about — `assets/city/buildings` (`city::definition::
BuildingDefinitions`) and `assets/city/road_types` (`city::road_definition::
RoadTypes`) — not the `.nbt` geometry catalogues (`assets/city/blueprints`,
`assets/city/roads`), which are shape, not the numbers/references schema C1
describes.

## The change

New `city::hot_reload` module:

- `dir_snapshot(dir) -> HashMap<PathBuf, SystemTime>` — every `*.ron` file's
  last-modified time, non-recursive. A missing directory or an unreadable
  file is silently absent, same "not fatal" contract every loader in this
  crate already uses.
- `DefinitionHotReloadPlugin` adds one `Update` system,
  `poll_definition_reload`, gated by a 1-second repeating `Timer` (no new
  dependency — a filesystem-watcher crate would need a thread/channel bridge
  into Bevy's `Resource` world for two directories that hold a handful of
  hand-edited files; a stat() per file per second costs nothing next to the
  terrain streaming already running every frame). When a directory's
  snapshot differs from the one stored last tick, that directory's loader
  (`definition::load_definitions_dir`/`road_definition::load_road_types_dir`)
  re-runs and its resource (`BuildingDefinitions`/`RoadTypes`) is replaced
  outright — safe because neither resource's only readers (the build menu,
  and this ticket's own error panel) ever hold a reference across frames,
  every read is a fresh `.get(id)` off the current `Res`.
- `DefinitionErrors` resource: `{ buildings: Vec<(PathBuf, String)>,
  road_types: Vec<(PathBuf, String)> }`, replaced wholesale by whichever
  directory's reload just ran. `city::run()` seeds both the snapshots and
  this resource from the *startup* load's own `skipped` list — the first
  `Update` tick sees no change and does nothing, rather than silently
  redoing (and re-printing) the load a moment after startup's own log lines.
  `load_building_definitions`/`load_road_types` in `city::mod` now return
  `(resource, skipped)` instead of discarding `skipped` after printing it,
  for exactly this.

New `city::ui::definition_errors` panel — a third egui window alongside
ticket 050's build menu/city panel, always registered (empty state: "(no
problems)"), collapsed by default only when there's nothing to show. The
window's title stays the fixed string `"Definition Errors"` rather than
growing a live count in the title text: egui keys a window's
position/open/collapsed state off its title by default, so a title that
changes every reload would make each one look like a brand new window and
reset wherever the player had moved or collapsed it — the count goes in the
body instead, where it's free to change every tick.

## Explicitly not in scope here

- **Watching `.nbt` catalogues.** A blueprint/road-piece file changing
  mid-session is a bigger problem (a placed building's mesh assumes its
  blueprint's shape hasn't moved) than this ticket's data-file reload, and
  the roadmap's C4 wording is about *definitions*, not geometry.
- **A sub-second reload.** 1 second was picked as "instant enough for
  someone alt-tabbing back to the game after saving a file," not measured
  against anything — the knob to revisit if it turns out to matter is
  `hot_reload::POLL_INTERVAL_SECS`.
- **Undoing an in-progress placement/ghost when its definition disappears
  mid-edit.** `PlacementSelection::catalogue_id` names a `BuildingCatalogue`
  (geometry) entry, not a `BuildingDefinitions` one — this ticket doesn't
  touch geometry, so a ghost stays valid; only the build menu's cost/
  production/lock display for that entry would go stale for up to a second,
  same as any other data changing under a running frame.

## Done when

- `cargo build`/`cargo test --lib` clean.
- Editing, adding, or removing a `.ron` under `assets/city/buildings` or
  `assets/city/road_types` while the game is running reloads that
  directory's resource within ~1 second, without a restart.
- A `.ron` file with a real problem (parse error, unknown blueprint/style,
  invalid range) shows up in the new "Definition Errors" window rather than
  only a console line — both at startup and after a live edit.

## Resolution

Landed as scoped. `city::hot_reload::{DefinitionHotReloadPlugin,
DefinitionSnapshot, RoadTypeSnapshot, DefinitionErrors, dir_snapshot,
poll_definition_reload}`; `city::ui::definition_errors::
definition_errors_panel`, registered in `ui::UiPlugin` alongside 050's two
existing windows. `city::run()` wires the snapshots/error resource in
right after the existing catalogue/definition startup loads.

One correctness note worth recording: the first draft's own test for
"editing a file changes its snapshot entry" failed on Windows —
`File::set_modified` needs a handle opened for *write*, and `File::open`
opens read-only, which Windows refuses permission for on a metadata write
even though the same call succeeds fine on Linux/macOS's more permissive
handling. Fixed by opening with `OpenOptions::new().write(true)` instead.

Hit one pre-existing, unrelated test flake while running the full suite
(`road_build::tests::bracket_right_cycles_forward_and_wraps`, fails only
under `cargo test`'s default parallel execution, passes both alone and with
`--test-threads=1`) — confirmed unrelated to this change and left as is; not
a hot-reload regression.

No manual-verification checklist item existed for this before — added
alongside this ticket's entry in `../todo.md`, since "does a live edit
actually get picked up, and does a bad file actually show up in the panel"
needs an editor and a running window, the same reasoning every other
startup/live-behavior checklist item in that file already uses.

Summarized on `CITYBUILDER_ROADMAP.md` under C4.
