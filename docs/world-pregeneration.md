# Pre-generating a world with Chunky

`block_viewer` reads existing Anvil saves — it doesn't generate terrain. To
have a sizable world to point it at, generate one ahead of time with the
**Chunky** mod/plugin, then load the resulting save (or copy it into a local
`saves/` folder) for `block_viewer` to open.

## Why Chunky

Vanilla Minecraft only generates chunks as a player walks near them, which
makes producing a large world tedious (fly around for hours, or script chunk
loading). Chunky is a dedicated world pre-generator:

- Generates chunks directly, without needing a player to explore them.
- Supports region shapes (`square`, `circle`, `star`, etc.) and precise
  radius/center control, so you generate exactly the area you want instead
  of an arbitrary blob.
- Tracks progress and is resumable — you can pause, restart the server, and
  continue later without redoing work.
- Reports progress/ETA so long generations (multi-thousand-block radius) are
  observable instead of a black box.
- Available both as a Fabric/Forge mod (for a singleplayer world) and as a
  Paper/Spigot server plugin (for a dedicated server), covering either
  workflow.

It's distributed on [Modrinth](https://modrinth.com/) and
[CurseForge](https://www.curseforge.com/) — search "Chunky" (by `pop4959`).
Grab the build matching your setup:

| Setup | Chunky build |
|---|---|
| Singleplayer world, Fabric | Chunky (Fabric) + Fabric API |
| Singleplayer world, Forge/NeoForge | Chunky (Forge) |
| Dedicated/local server | Chunky (Paper/Spigot plugin) |

For generating a large world, the **server plugin** route is recommended:
it runs headless (no client rendering overhead), can be given more RAM, and
survives you closing the game. The instructions below cover that path, with
the singleplayer-mod path noted as an alternative.

## Option A: server plugin (recommended for large worlds)

1. **Set up a local Paper server.**
   - Download a Paper server jar for the Minecraft version you want
     (https://papermc.io/downloads), e.g. `paper-1.21.1-XXX.jar`.
   - Put it in an empty folder (e.g. `C:\mc-pregen\`) and bootstrap it:
     ```
     java -jar paper-1.21.1-XXX.jar --nogui
     ```
     This generates `eula.txt` and `server.properties`, then exits (or
     refuses to start, complaining about the EULA).
   - Open `eula.txt` and change `eula=false` to `eula=true`.
   - Optionally edit `server.properties` before the first real start if you
     want a specific world, e.g. `level-seed=...`, `level-type=...`.
   - Start it for real, with more heap than the default:
     ```
     java -Xmx8G -Xms8G -jar paper-1.21.1-XXX.jar --nogui
     ```
     This generates spawn chunks and drops you into the server console
     (`>` prompt), where the `chunky ...` commands below are run. A
     `world/` folder appears next to the jar — that's the save you'll copy
     into your Minecraft `saves/` directory once generation is done.

2. **Install Chunky.**
   - Download the **Bukkit/Spigot/Paper** build specifically — Modrinth and
     CurseForge list separate Fabric, Forge/NeoForge, and Bukkit builds
     under the same "Chunky" listing, and grabbing the wrong one (e.g.
     `Chunky-NeoForge-*.jar`) makes Paper fail to load it with `does not
     contain a paper-plugin.yml or plugin.yml!`.
   - Drop the Chunky plugin `.jar` into the server's `plugins/` folder.
   - Start the server so it loads the plugin, then stop it (or leave it
     running — Chunky works with the server live).

3. **Configure the generation task.** From the server console (or in-game
   with op permissions):
   ```
   chunky world world          # select which world/dimension to target
   chunky center 0 0           # generation center, in block coordinates
   chunky shape circle         # circle | square | star | ... 
   chunky radius 4000          # radius in blocks
   ```
   A radius of 4000 blocks is ~250×250 chunks (~62k chunks) — adjust to
   taste; generation time scales roughly with area.

4. **Start it.**
   ```
   chunky start
   ```
   Chunky reports periodic progress (`chunks/s`, percent complete, ETA) in
   the console.

5. **Pause/resume as needed.**
   ```
   chunky pause
   chunky continue
   ```
   Progress is persisted, so stopping the whole server and starting it again
   later also resumes correctly — just run `chunky continue` after restart.

6. **Let it finish**, then stop the server cleanly (`stop`) so the region
   files are flushed to disk.

### Tuning for speed

- Give the server more heap: `java -Xmx8G -Xms8G -jar paper-*.jar ...`
  (adjust to available RAM).
- Set `view-distance` low in `server.properties` — Chunky doesn't need it
  high, and a lower value reduces memory pressure while chunks are held for
  writing.
- Consider `chunky silent true` if you don't want per-tick chatter in the
  console, and check progress with `chunky status`/`chunky progress`
  instead.

## Option B: singleplayer mod

If you'd rather not stand up a server:

1. Install Fabric Loader + Fabric API (or Forge) for your target Minecraft
   version, then drop the matching Chunky build into `mods/`.
2. Create/open the singleplayer world you want to pre-generate.
3. Open chat/console and run the same commands as above (`chunky center`,
   `chunky shape`, `chunky radius`, `chunky start`).
4. Chunky generates in the background while the world is loaded; progress
   shows in chat/HUD. You can leave the world and come back — progress is
   saved per-world.

This keeps the client (and its rendering cost) running for the whole
generation, so it's noticeably slower than the server route for large radii.

## Getting the save into `block_viewer`

`block_viewer` discovers saves via `mc_anvil::get_saves`, which reads the
standard Minecraft `saves/` directory layout (a folder per world, each
containing `region/`, `level.dat`, etc.).

- **Singleplayer route:** the world already lives under your Minecraft
  `saves/<world name>/` — nothing to copy.
- **Server route:** copy the server's world folder (by default named
  `world/`, alongside `world_nether/` and `world_the_end/` for the other
  dimensions) into your Minecraft `saves/` directory, e.g.
  ```
  saves/<new-name>/
  ```
  Rename it to whatever you want it to appear as. Make sure the server is
  stopped first so files aren't mid-write.

Afterwards, run `cargo run` and the pre-generated world should show up
alongside your other saves.

## Sanity-checking the result

Before loading a multi-thousand-chunk world in `block_viewer`, it's worth
confirming Chunky actually covered the area you asked for:

- `chunky status` (or `chunky progress`) while running, or the final summary
  line when a task completes, reports chunks generated vs. total for the
  requested shape/radius.
- The `region/` folder should contain `.mca` files covering the expected
  coordinate range (each region file spans 32×32 chunks, named
  `r.<x>.<z>.mca`).
