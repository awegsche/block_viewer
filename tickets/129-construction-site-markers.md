# 129 - Construction sites are invisible while they clear: markers for both kinds

## The problem

Ticket 128 made a placement over rough ground into a **site** — claimed in
`City`, journaled (for a building), dug clear by `city::construction`'s tick
over real time — but deliberately shipped with no visual marker for one at
all. Right now a site reads as: terrain quietly disappearing block by block,
and (for a building, if the inspect panel happens to be open on it) a
"Clearing site: N of M block(s)" line. There is nothing standing over the
hole to say *a building is coming here*, and nothing at all for a road cell
site — a piece of ground just started digging itself out with no marker
whatsoever, building or road. This is the gap 128's own "Seeing it" section
called for and 128's `todo.md` entry flags as known-missing.

## Design

### Building sites: the ghost, reused

A site marker is `placement::ghost_mesh`'s cached mesh for
`(catalogue_id, rotation)`, at the site's own `origin`, using the ghost's
*valid* material (translucent, green-tinted) — not a new mesh path, the same
one ticket 047 already built and ticket 048's commit already resolves at
click time. `city::construction`'s tick (or a small new system in the same
module) spawns one such entity when a building enters `under_construction`
and despawns it on completion (`City::complete_building`'s caller) or
cancellation (`city::demolish`'s site-reversal path). One entity per site,
tracked the same way `placement::GhostState`/`road_build::RoadPreviewState`
track theirs — a `HashMap<BuildingId, Entity>` is enough, no pool needed
since sites are rare and short-lived compared to a drag preview redrawn
every frame.

Reusing `ghost_mesh` means `placement`'s own mesh cache (or an equivalent one
scoped to `construction`) has to be reachable from a system that isn't
`placement::update_ghost_preview` — either factor `ghost_mesh`/`ensure_materials`
out to somewhere both modules can call (a small shared submodule, or `pub(super)`
on `placement`'s own versions), or accept a second, `construction`-local cache
keyed the same way. Whichever way settle it, don't recompute the mesh: the
rotation/meshing cost is the whole reason `placement`'s own cache exists in
the first place.

### Road cell sites: the piece mesh, reused

Same idea, one level down: `road_build::preview_mesh` (currently private to
that module, and gated on `RoadPreviewState`'s cache) resolves the mesh for
`(style, kind, variant, rotation)` — make it `pub(super)` alongside
`cell_transform` (also currently private), and spawn/despawn an entity per
road cell site the same way. A cell with no catalogue piece for its
`(style, kind, variant)` never becomes a site at all (128's own rule — see
`road_build::road_site_box`), so every site this ticket needs to mark
*does* have a real piece to preview; there is no quad-fallback case to worry
about here the way the live drag preview has to.

### What both markers show, and don't

- Both use the *valid* (green) tint, never red — a site is already
  confirmed buildable; there is no invalid state for one to be in.
- Neither marker changes as the dig progresses — it's a static "this is
  what's coming," not a progress bar (the inspect panel's "N of M" line, and
  a manual watch of the terrain shrinking, are the progress feedback).
- A site loaded from a save (`city.ron`'s `under_construction: true`) needs
  its marker spawned on load too, not only when one is freshly entered this
  session — the spawn system should scan `City::placements()`/`road_cells_with_data()`
  for existing sites on startup (or simply run its "spawn a marker for any
  site missing one" pass every tick, which is simplest and cheap given how
  few sites exist at once).

## Out of scope

- Any change to the clearing mechanic itself (rate, order, credit) — this is
  rendering only, on top of 128's finished mechanic.
- An in-progress visual (a wireframe cage shrinking, a percentage overlay) —
  the ticket only asks for "there is a marker," not a new progress affordance.
  A future ticket can revisit if the static ghost isn't legible enough once
  it's actually looked at.

## Files

- `src/city/construction.rs` — the marker spawn/despawn systems and their
  tracking maps, added to `ConstructionPlugin`.
- `src/city/placement.rs` — `ghost_mesh`/`ensure_materials` (or equivalent),
  made reachable from `construction` if not already `pub(super)`.
- `src/city/road_build.rs` — `preview_mesh`/`cell_transform` made
  `pub(super)`.

## Tests

- A building site's marker spawns the tick it's entered (or found on load)
  and despawns on completion and on cancellation.
- A road cell site's marker spawns and despawns the same way, per cell.
- Loading a save with an existing site (either kind) spawns its marker
  without waiting for a fresh commit.
- `cargo check`, `cargo test`, `cargo clippy` clean.

## Manual check (todo.md)

Place a building against a hillside and confirm a translucent green
building-shaped ghost stands over the site from the moment it's entered,
in the same spot and rotation the real building lands in once clearing
finishes, and disappears the instant the real blueprint is written. Cancel
one mid-clearing (`Delete`) and confirm the marker disappears with it. Drag
a road into the same hillside and confirm each new cell shows its piece's
ghost (or the tunnel/connected variant, whichever the cell resolved to)
while it clears, disappearing exactly when that cell's real piece lands.
Save and reload mid-clearing (both kinds) and confirm the marker reappears
without needing a fresh click.
