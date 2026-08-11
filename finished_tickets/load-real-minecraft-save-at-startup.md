# Load a real Minecraft save at startup

## Goal

On startup, instead of only listing save metadata, actually load one real
save from `C:\Users\andiw\AppData\Roaming\.minecraft\saves` (via
`dirs::config_dir()`, which already resolves to `AppData\Roaming` on
Windows) and make it available to the Bevy app as a resource.

## Scope

- Pick a save from the real saves directory (first one found).
- Load its region list and parse the chunks of at least one region so we
  know real chunk NBT data is reachable.
- Store the loaded `mc_anvil::Save` as a Bevy `Resource` so later systems
  (e.g. terrain meshing) can use it.
- Out of scope: converting real block/chunk data into the rendered mesh —
  `ChunkRegion::get_block` in `ranvil` is still a debug/WIP function that
  always returns `None`, so real terrain rendering is a follow-up ticket.
  The placeholder test-cube mesh stays for now.

## Done when

- App startup loads a real save from the saves folder above (no panics on
  a missing/empty folder — handled gracefully).
- Loaded save (name, region count, loaded chunk count) is visible in the
  startup log.
