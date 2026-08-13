# Roadmap — from "loads a save" to "explore a save"

Not a work item; an index of the numbered tickets in this directory and how
they fit together. Numbered tickets are the actual unit of work and move to
`../finished_tickets/` when done (see `CLAUDE.md`).

## Where we are

001–008 have landed. The app opens a window immediately, streams chunks
around a free-flying camera out to a tunable render distance, decodes real
`block_states`, meshes each column with neighbour-aware face culling and a
packed texture atlas, re-seams the loading frontier, unloads behind the
camera, and has an egui shell (save picker, coordinate jump, block
inspector, status panel) plus non-panicking startup.

What's missing is everything about how it *looks*: the world is untinted
(so grass, leaves and water render grey — those textures ship as neutral
masks), lit by a single leftover `PointLight` near spawn, under a default
clear colour with no sky, and with no shadows.

## The shape of the work

```
001  bit-width bug (upstream, in ../ranvil)         done
002  decode layer: NBT -> BlockState grids          done
003  mesher: BlockState grids -> Bevy meshes        done
004  textures: block name -> atlas UV               done
005  streaming: load/mesh chunks around camera      done (a–f)
006  camera: navigation fit for exploring           done
007  egui: save picker, coord jump, block readout   done
008  no-panic startup + error surfacing             done
      |
009  performance pass                               open
010  visual fidelity: block models, transparency, AO, baked light
      |
      colour & atmosphere (this group):
      |
      011  mesher: per-vertex colour channel   <- shared substrate
      |     |
      |     012  biome decode (NBT -> 4x4x4 biome grids)
      |     013  biome tint (colormaps -> vertex colours)
      |     014  alpha-masked cutouts + grass side overlay
      |
      015  sun rig + SkyPalette (one source of sky colour)
            |
            016  skybox: gradient dome, sun, moon
            017  shadows: cascades, bias, a toggle
            018  day/night cycle: TimeOfDay -> palette + sun angle
```

## The colour & atmosphere group (011–018)

Two tracks that don't touch each other and can run in parallel or in either
order.

**Track A — colour (011 → 012 → 013 → 014).** Why grass is grey: modern
vanilla ships `grass_block_top`, `oak_leaves` and `water_still` as greyscale
masks tinted per biome at render time. 011 adds the vertex colour channel
they need; 012 decodes the biome data (a different NBT encoding from
`block_states` in three ways — read the ticket before writing the bit
width); 013 does the actual tinting; 014 cleans up the two things tinting
exposes (leaves have alpha holes that currently render solid, and a grass
block's green sides are a separate overlay texture).

**Track B — light and sky (015 → 016/017/018).** 015 is the keystone: one
`SkyPalette` resource owning sun direction, ambient, sky colour and fog
colour, replacing the placeholder `PointLight` and the hardcoded fog colour
in `camera::atmosphere_fog`. 016 draws a sky from it, 017 adds shadows to
its sun, 018 animates it. Each of the three is small *because* 015 put the
plumbing in one place; doing any of them first means doing 015 badly inside
it.

### Ordering advice

- **011 first, always.** Three separate features multiply into that one
  channel (biome tint, 010's baked `SkyLight`/`BlockLight`, 010's AO). It's
  an hour of work and it's the only ticket in the group that changes the
  vertex layout.
- **015 is the best value per hour in the whole group.** Deleting the
  point light and putting a real sun and a coherent sky/fog colour in costs
  little and changes how everything reads.
- **017 wants 009's numbers.** Shadows can multiply draw calls by up to 5x
  on a renderer that currently does one un-batched entity per chunk column.
  It's the only ticket here that can plausibly halve the frame rate.
- **013 and 018 are much easier to tune with each other's work available**
  (biome colours under a moving sun), but neither blocks the other.

### Deliberately not in this group

**Baked `SkyLight`/`BlockLight`** — the per-section light nibbles in chunk
NBT, which are what actually make caves dark and give Minecraft its
characteristic flat-lit look. It stays in 010 because it's a different
mechanism from real-time sun lighting, but it writes the *same* vertex
colour channel 011 adds and multiplies with 013's tint. Whoever picks it up
should read 011's composition rule first. Arguably the highest-value item
left in 010 once this group lands.

## Suggested milestones

1. **M1 — "it's my world"**: 001, 002, 003. ✔
2. **M2 — "it looks right"**: 004. ✔
3. **M3 — "I can explore it"**: 005, 006, 008. ✔
4. **M4 — "it's a tool"**: 007, 009.
5. **M5 — "it looks like Minecraft"**: 011–018. Track A makes the world the
   right colour; track B gives it a sky and a sun. A good stopping point
   after 013 + 015 if the rest gets deprioritised — those two alone cover
   most of the visible gap.
