## TODO

- [ ] **134 model-exporter markers, in the real models world.** Back the
  real models world save up first (or run against a scratch copy). With
  `assets/models/world.ron` pointed at it, run
  `cargo run --bin model-exporter -- new ring_test 5 4 5` and confirm the
  printed origin/box/tp line, then `/tp` there in Minecraft and check: an
  orange terracotta ring flush with the ground, one block outside the 5x4x5
  footprint on every side; four pillars, three blocks tall, standing on the
  ring's four corners; nothing placed inside the footprint itself. Then run
  `model-exporter mark ring_test` again and confirm nothing visibly
  changes (idempotent re-placement). Once 135 (`export`) exists, export
  `ring_test` and confirm the ring/pillars are *not* in the resulting
  `.nbt` — only the footprint's own blocks are.
- [ ] **130 wheat farm tiles.** Load the citybuilder, unlock and place a
  `farm01` (Wheat Farm), then a `farm01_tile` (Wheat Field) — the build
  menu's `farm01` catalogue row should read "Scales with Wheat Field
  nearby (4 needed, within 20 blocks)" before either is placed. After
  placing one to four tiles within 20 blocks of the hub, confirm the city
  panel's `farm01` line shows a growing `N/4 tiles` fraction and that
  wheat actually starts accruing in its buffer once `N` is above zero
  (nothing before that). Both buildings are still placeholder geometry
  (`house01.nbt`), so this is only checking the mechanism, not how it
  looks.
- [ ] **126 gallery torches hang on the north wall on both arms.** Place a
  fresh mine (or let an existing one open a new gallery row on the *north*
  arm), open the world in Minecraft and walk a north-arm gallery: every
  torch should sit on the wall at head height, none floating mid-corridor.
  South-arm galleries were already right. Galleries dug before this fix
  keep their floating torches — slices aren't re-run — so check a new row,
  not an old one.
- [ ] **127 shaft torches hang on the lining, not in it.** Place a fresh
  mine and walk the primary shaft's spiral stair in Minecraft: every
  torch should be on the shaft wall at head height above a step, flush
  with the wall — no one-block niches, no torch missing its wall behind a
  cave, and none in front of a level's doorway. Shafts sunk before this
  fix keep their niche torches; check a new mine.
- [ ] **128 site clearing before placement: the hillside actually peels
  away over time, not instantly.** The automated suite covers the scan
  order, the carry/dispatch math, the journal merge, and the `City`/
  `warehouse` filtering directly against synthetic fixtures — what it can't
  cover is whether the tick actually reads as "digging" at the window, or
  whether the (deliberately unimplemented — see below) missing site marker
  makes it confusing in practice. **Back the world up first** (or run this
  against a scratch copy) — this writes to a real save.

  `cargo run --bin citybuilder` against the real save, with a building
  whose footprint is a few tiles (a house is fine) and a hillside or
  otherwise uneven patch of terrain. Place it against the hillside and
  confirm: (1) the building does **not** appear immediately — the console
  should print a "... entered as a site — N block(s) to clear ..." line
  instead of a "placed ..." line; (2) over the next tens of seconds, watch
  the terrain in the footprint's volume visibly shrink from the top down
  (no restart needed to see it — `ChunksEdited` still fires per dig); (3)
  the city panel's stock gains the dirt/stone the dig is crediting, ticking
  up as it goes; (4) once the volume is fully clear, the building's
  blueprint actually appears, and only then; (5) place the same building on
  flat, already-clear ground and confirm it appears instantly, unchanged
  from before this ticket (the "nothing to clear" fast path).

  Then the road half: drag a road into the same hillside (or a new one) and
  confirm the piece markers/quads for the newly added cells sit there while
  the ground under them clears in parallel (cells cleared in parallel, not
  one at a time), each piece landing the moment its own cell empties out —
  a tunnel cell should land last, once the terrain that made it a tunnel
  has been dug away from around it, not through it.

  Then `Delete` mid-clearing on the site: confirm the partial hole gets
  filled back in with what was actually dug (not the whole original
  hillside snapping back if only part of it was cleared), the planks are
  refunded, and the building's row disappears from the city panel's count.
  Try `Delete` again immediately after clicking (while a dig write might
  still be in flight) and confirm it's refused with a message rather than
  racing the dig.

  Finally, save and reload mid-clearing (quit while a site is still
  digging, reopen the same save): confirm the site is still there, still
  clearing, and the terrain hasn't snapped back to its pre-dig state.

  **Known gap, spun off as ticket 129**: the ticket's own "Seeing it"
  section calls for a translucent site marker (a ghost-tinted mesh at a
  building site's origin, and piece meshes for road cell sites) so a site
  reads as "a building is coming" rather than just a hole. That marker
  was **not implemented** in this pass — a site is currently only visible
  as bare terrain being dug and the inspect panel's "Clearing site: N of M"
  line, with no ghost standing over it. See
  `tickets/129-construction-site-markers.md`.
