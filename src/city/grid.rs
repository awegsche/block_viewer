//! Tile grid sampling and terrain fit (ticket 046, roadmap E2): whether a
//! footprint's ground is buildable, and at what height to place it.
//!
//! [`fit_footprint`] is the answer E3's ghost preview and E4's commit both
//! need before a placement can go ahead. It does no writing itself — see
//! "No auto-level" below — and reads only [`DecodedWorld`], the same
//! in-memory decode E1's picking already reads through
//! `camera::block_under_cursor`. The tile grid itself is
//! [`super::state::footprint_tiles`], already built for D1's occupancy
//! grid; this module doesn't reinvent it, only feeds it ground heights.
//!
//! ## The height read
//!
//! [`ground_height_at`] is [`world::ChunkColumn::topmost_non_air`] — the
//! primitive that type's own doc comment names this as the eventual caller
//! of. It answers straight off [`DecodedWorld::columns`], no `RegionCache`
//! lock and no I/O, which is also why a tile whose chunk hasn't streamed in
//! yet reads as [`FitError::NotLoaded`] rather than blocking to fetch it:
//! the citybuilder's own streaming radius decides what ground is known, the
//! same way it already decides what the camera can see.
//!
//! "Ground" here is **not** [`world::is_solid`]'s plain not-air — ticket 052
//! found that reading a tree, fence post or tall grass as "ground" refuses a
//! placement several blocks (ten, for a tree) away from real, flat terrain,
//! even though [`crate::city::commit`]'s own write already clears the
//! obstruction without complaint. [`is_ground`] narrows the predicate: still
//! not `ranvil::heightmap`'s `blocks_motion` (a lake's surface still counts
//! as ground the way a stone hilltop does — iteration 1 doesn't distinguish,
//! and that refinement really is roadmap I3's later job), but clutter —
//! vegetation, decoration, anything the write path's own "air is a block"
//! policy would already bulldoze — no longer reads as the footprint's floor.
//!
//! ## No auto-level, and no slope refusal either (ticket 058)
//!
//! The roadmap asks this ticket to decide whether uneven ground under a
//! footprint gets levelled or refused. **Neither, any more.** Levelling
//! means writing blocks, and writing blocks is W4/W5's job through a real
//! [`crate::edit::WorldEdit`] — folding it into a read-only fit check would
//! make "is this buildable" secretly depend on the write path, and every
//! future caller (E3's every-frame ghost preview included) would pay for it.
//! [`fit_footprint`] originally refused past a 1-block tolerance
//! (`MAX_FOOTPRINT_STEP`), but a real Minecraft world is inherently uneven,
//! and a hard height-difference cap restricts where a player can build far
//! more than it's worth — ticket 058 removed it. `base_y`
//! ([`FootprintFit::Fits`]) is now purely the footprint's *lowest* sampled
//! point, used only as the placement's initial suggested height: a higher
//! corner clips a little into the building's own foundation (better than a
//! gap floating over a low corner, given nothing here fills terrain in), and
//! nothing about how steep the rest of the footprint is stops the
//! placement. `city::placement::PlacementSelection::y_offset` (`Page Up`/
//! `Page Down`/`Home`) is the player's own override on top of that
//! suggestion — always was, and is now the *only* way a steep site's height
//! gets adjusted, since this module no longer has an opinion beyond "here's
//! the lowest point."
//!
//! What a steep placement should *cost* — the roadmap floated build time (or
//! some future resource) scaling with how many solid blocks a placement
//! needs to clear, so burying a building in a hillside is expensive rather
//! than blocked or free — is deliberately not this module's job either, and
//! isn't implemented anywhere yet; see `CITYBUILDER_ROADMAP.md`'s note under
//! E2/H2. Terraforming (H1) remains the way a player can flatten a site by
//! hand if they'd rather not pay whatever that eventual cost turns out to
//! be, through the write path this module deliberately doesn't touch.
//!
//! [`fit_footprint`]'s first real caller is `city::placement` (ticket 047,
//! roadmap E3)'s ghost preview, called once per frame at the hovered tile.
//! Occupancy (is a tile already claimed by another building or a road) is a
//! separate question this module still doesn't answer — that's
//! [`super::state::City::is_tile_free`], which `placement` checks alongside
//! this module's terrain fit rather than the two being folded into one.

use bevy::math::{IVec2, IVec3};

use super::state::footprint_tiles;
use crate::blueprint::Rotation;
use crate::world;
use crate::DecodedWorld;

/// Why [`fit_footprint`] refused a placement. `NotLoaded` is the only
/// reason left as of ticket 058 — see the module docs' "No auto-level, and
/// no slope refusal either".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    /// `tile`'s chunk isn't decoded in [`DecodedWorld`] yet (outside the
    /// streamed radius), or its column has no solid block anywhere in
    /// what's decoded. Both collapse to one "no answer available" case —
    /// same call [`super::picking::HoveredBlock`] already makes for its own
    /// `None`.
    NotLoaded { tile: IVec2 },
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::NotLoaded { tile } => {
                write!(f, "ground at ({}, {}) isn't loaded yet", tile.x, tile.y)
            }
        }
    }
}

impl std::error::Error for FitError {}

/// Whether a footprint's ground supports a placement, and at what height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FootprintFit {
    /// `base_y` is the footprint's lowest sampled ground point — the
    /// *suggested* world Y for a placed building's floor, freely overridable
    /// by [`super::placement::PlacementSelection::y_offset`] (see the module
    /// docs' "No auto-level, and no slope refusal either"). Ticket 058: this
    /// is returned regardless of how steep the rest of the footprint is.
    Fits { base_y: i32 },
    Refused(FitError),
}

/// Block name families [`is_ground`] excludes from "the footprint's floor" —
/// vegetation, decoration, and anything else the write path's own "air is a
/// block" policy (`city::commit`'s `blueprint_edit`) already clears without
/// complaint. See ticket 052 and the module docs.
///
/// Deliberately a blocklist, not an allowlist: the overwhelming majority of
/// registered names really are terrain (every stone/dirt/sand/ore/deepslate
/// variant among them), so naming the small set of exceptions is far shorter
/// than trying to enumerate "ground." Suffix matches catch a whole wood/
/// redstone-component family at once, the same way `world::tint`'s `_leaves`
/// heuristic does for a different purpose; the exact-match list below is
/// everything else that isn't itself a suffix family.
fn is_clutter_name(name: &str) -> bool {
    const CLUTTER_SUFFIXES: &[&str] = &[
        "_leaves",
        "_log",
        "_wood",
        "_stem",
        "_hyphae",
        "_sapling",
        "_fence",
        "_fence_gate",
        "_sign",
        "_hanging_sign",
        "_banner",
        "_carpet",
        "_pressure_plate",
        "_button",
        "_door",
        "_trapdoor",
    ];
    if CLUTTER_SUFFIXES.iter().any(|suffix| name.ends_with(suffix)) {
        return true;
    }

    const CLUTTER_NAMES: &[&str] = &[
        "short_grass",
        "tall_grass",
        "fern",
        "large_fern",
        "dead_bush",
        "vine",
        "glow_lichen",
        "lily_pad",
        "sugar_cane",
        "cactus",
        "bamboo",
        "kelp",
        "kelp_plant",
        "seagrass",
        "tall_seagrass",
        "dandelion",
        "poppy",
        "blue_orchid",
        "allium",
        "azure_bluet",
        "red_tulip",
        "orange_tulip",
        "white_tulip",
        "pink_tulip",
        "oxeye_daisy",
        "cornflower",
        "lily_of_the_valley",
        "wither_rose",
        "torchflower",
        "pitcher_plant",
        "sunflower",
        "lilac",
        "rose_bush",
        "peony",
        "brown_mushroom",
        "red_mushroom",
        "snow",
        "cobweb",
        "ladder",
        "torch",
        "wall_torch",
        "soul_torch",
        "soul_wall_torch",
        "redstone_wire",
        "redstone_torch",
        "redstone_wall_torch",
        "tripwire",
        "tripwire_hook",
        "lever",
        "rail",
        "powered_rail",
        "detector_rail",
        "activator_rail",
    ];
    CLUTTER_NAMES.contains(&name)
}

/// Whether `id` counts as *terrain* for [`ground_height_at`] to sample —
/// [`world::is_solid`]'s not-air check, narrowed by [`is_clutter_name`]. Not
/// a per-[`world::BlockId`] table the way `world::tint`'s tables are: a
/// footprint fit samples a few dozen tiles at most, not once per emitted
/// mesh face, so there's no hot loop here for a table to earn its keep — the
/// same "name function first, plain lookup second" shape
/// `world::mesh::is_solid_name`/`is_solid` use for the same reason.
fn is_ground(id: world::BlockId, registry: &world::BlockRegistry) -> bool {
    if id == world::BlockRegistry::AIR {
        return false;
    }
    let full_name = registry.name(id);
    let name = full_name.strip_prefix("minecraft:").unwrap_or(full_name);
    !is_clutter_name(name)
}

/// The world Y one above the topmost [`is_ground`] block at `tile`, via
/// [`world::ChunkColumn::topmost_matching`] against `world.columns` directly
/// — or `None` if `tile`'s chunk isn't decoded, or nothing counts as ground
/// in what's decoded (including a footprint standing on clutter all the way
/// down — see the module docs). Locks `world.registry` once per call, not
/// once per block: cheap enough at "a few dozen tiles, once a frame."
pub fn ground_height_at(tile: IVec2, world: &DecodedWorld) -> Option<i32> {
    let size = world::SECTION_SIZE as i32;
    let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
    let column = world.columns.get(&chunk)?;
    let local_x = tile.x.rem_euclid(size) as usize;
    let local_z = tile.y.rem_euclid(size) as usize;
    let registry = world.registry.lock().unwrap();
    column
        .topmost_matching(local_x, local_z, |id| is_ground(id, &registry))
        .map(|(y, _id)| y + 1)
}

/// The block at Minecraft coordinates `block`, or `None` if its chunk column
/// isn't decoded at all — the deliberate distinction
/// [`ground_height_at`] already draws between "nothing there" and "not
/// loaded". An absent *section* inside a decoded column is uniform air (see
/// [`world::ChunkColumn::sections`]), so it reads back as
/// [`world::BlockRegistry::AIR`] rather than `None`.
///
/// Ticket 071 is the caller: `city::road_build` counts how many of a road
/// cell's 36 columns are roofed over at one exact Y, which is a question
/// [`ground_height_at`]'s top-down scan can't answer. `camera` has a private
/// near-twin of this for its own ray-marching, which reads an unloaded
/// column as air; that suits a mesher's neighbour lookups and doesn't suit a
/// decision about whether to carve a tunnel, so this one keeps the `Option`.
pub fn block_at(block: IVec3, world: &DecodedWorld) -> Option<world::BlockId> {
    let size = world::SECTION_SIZE as i32;
    let column = world.columns.get(&(block.x.div_euclid(size), block.z.div_euclid(size)))?;
    let section_y = block.y.div_euclid(size) as i8;
    let Some(section) = column.sections.iter().find(|s| s.y == section_y) else {
        return Some(world::BlockRegistry::AIR);
    };
    Some(section.get(block.x.rem_euclid(size) as usize, block.y.rem_euclid(size) as usize, block.z.rem_euclid(size) as usize))
}

/// Samples every tile [`footprint_tiles`] covers for `footprint` placed at
/// `origin`/`rotation`, and decides whether it's buildable — see the module
/// docs for the height read and the "No auto-level, and no slope refusal
/// either" note — as of ticket 058, the only way this refuses at all is
/// [`FitError::NotLoaded`].
///
/// Fails fast on the first unresolvable tile — a footprint reaching off the
/// streamed edge is refused before every other tile is even sampled, the
/// same "stop once the answer is already no" shape
/// [`super::state::City::place_building`] uses for occupancy.
pub fn fit_footprint(
    origin: IVec3,
    footprint: IVec2,
    rotation: Rotation,
    world: &DecodedWorld,
) -> FootprintFit {
    // Only the minimum is tracked — nothing reads a maximum any more now
    // that steepness can't refuse a placement (ticket 058). `Option` rather
    // than an `i32::MAX` sentinel: a zero-extent footprint (not producible
    // by the catalogue, but not guarded against by `footprint_tiles`
    // either) samples no tiles at all.
    let mut min_y: Option<i32> = None;

    for tile in footprint_tiles(origin, footprint, rotation) {
        let Some(height) = ground_height_at(tile, world) else {
            return FootprintFit::Refused(FitError::NotLoaded { tile });
        };
        min_y = Some(match min_y {
            Some(min) => min.min(height),
            None => height,
        });
    }

    let Some(min_y) = min_y else {
        // No tiles to check — nothing constrains the height, so leave the
        // placement at whatever Y it was asked for rather than inventing one.
        return FootprintFit::Fits { base_y: origin.y };
    };

    FootprintFit::Fits { base_y: min_y }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::world::{BiomeRegistry, BlockId, BlockRegistry, ChunkColumn, ChunkSection};

    fn registry_with_stone() -> (BlockRegistry, BlockId) {
        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");
        (registry, stone)
    }

    /// A `DecodedWorld` whose only structure is one stone block at each
    /// `(tile, y)` in `ground` — everything else in the world is air.
    /// Chunks that cover none of `ground`'s tiles are simply absent, so a
    /// tile in one reads as [`FitError::NotLoaded`], not as flat ground at
    /// the world bottom.
    fn world_with_ground(ground: &[(IVec2, i32)]) -> DecodedWorld {
        let (registry, stone) = registry_with_stone();
        let size = world::SECTION_SIZE as i32;
        let mut columns: HashMap<(i32, i32), ChunkColumn> = HashMap::new();

        for &(tile, y) in ground {
            let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
            let (local_x, local_z) = (tile.x.rem_euclid(size) as usize, tile.y.rem_euclid(size) as usize);
            let section_y = y.div_euclid(size) as i8;
            let local_y = y.rem_euclid(size) as usize;

            let column = columns.entry(chunk).or_insert_with(|| ChunkColumn {
                x: chunk.0,
                z: chunk.1,
                sections: Vec::new(),
                floor_y: world::WORLD_MIN_Y,
            });
            let section = match column.sections.iter().position(|s| s.y == section_y) {
                Some(index) => index,
                None => {
                    column.sections.push(ChunkSection {
                        y: section_y,
                        blocks: Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]),
                        biomes: Box::new([BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]),
                    });
                    column.sections.len() - 1
                }
            };
            column.sections[section].blocks[ChunkSection::index(local_x, local_y, local_z)] = stone;
        }

        DecodedWorld {
            registry: Arc::new(Mutex::new(registry)),
            biomes: Arc::new(Mutex::new(BiomeRegistry::new())),
            columns,
        }
    }

    /// A single chunk's worth of flat ground at `y`, covering `0..16` on
    /// both axes.
    fn flat_chunk(y: i32) -> DecodedWorld {
        let ground: Vec<(IVec2, i32)> = (0..16)
            .flat_map(|x| (0..16).map(move |z| (IVec2::new(x, z), y)))
            .collect();
        world_with_ground(&ground)
    }

    /// A single decoded, entirely-air chunk column at chunk `(0, 0)` — for
    /// tests that build clutter up from nothing via [`add_block`] rather
    /// than starting from [`world_with_ground`]'s stone floor.
    fn empty_chunk() -> DecodedWorld {
        let (registry, _stone) = registry_with_stone();
        let mut columns: HashMap<(i32, i32), ChunkColumn> = HashMap::new();
        columns.insert((0, 0), ChunkColumn { x: 0, z: 0, sections: Vec::new(), floor_y: world::WORLD_MIN_Y });
        DecodedWorld {
            registry: Arc::new(Mutex::new(registry)),
            biomes: Arc::new(Mutex::new(BiomeRegistry::new())),
            columns,
        }
    }

    /// Interns `name` (if not already interned) and sets the block at
    /// `tile`/`y` in `world` to it — `tile`'s chunk must already exist
    /// (built by [`world_with_ground`]/[`empty_chunk`]), the same
    /// "no ungenerated chunks" assumption [`ground_height_at`] itself makes.
    /// Used to stack clutter (a tree trunk, a fence post) onto ground a
    /// prior helper already built, without duplicating its section-building
    /// logic.
    fn add_block(world: &mut DecodedWorld, tile: IVec2, y: i32, name: &str) {
        let id = world.registry.lock().unwrap().intern(name);
        let size = world::SECTION_SIZE as i32;
        let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
        let (local_x, local_z) = (tile.x.rem_euclid(size) as usize, tile.y.rem_euclid(size) as usize);
        let section_y = y.div_euclid(size) as i8;
        let local_y = y.rem_euclid(size) as usize;

        let column = world.columns.get_mut(&chunk).expect("add_block: tile's chunk must already exist");
        let section = match column.sections.iter().position(|s| s.y == section_y) {
            Some(index) => index,
            None => {
                column.sections.push(ChunkSection {
                    y: section_y,
                    blocks: Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]),
                    biomes: Box::new([BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]),
                });
                column.sections.len() - 1
            }
        };
        column.sections[section].blocks[ChunkSection::index(local_x, local_y, local_z)] = id;
    }

    #[test]
    fn ground_height_is_one_above_the_topmost_solid_block() {
        let world = world_with_ground(&[(IVec2::new(5, 5), 63)]);
        assert_eq!(ground_height_at(IVec2::new(5, 5), &world), Some(64));
    }

    #[test]
    fn ground_height_is_none_for_a_tile_whose_chunk_is_not_decoded() {
        let world = world_with_ground(&[(IVec2::new(5, 5), 63)]);
        assert_eq!(ground_height_at(IVec2::new(500, 500), &world), None);
    }

    #[test]
    fn flat_ground_fits_at_the_sampled_height() {
        let world = flat_chunk(64);
        let fit = fit_footprint(IVec3::new(2, 0, 2), IVec2::new(3, 4), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 });
    }

    #[test]
    fn uneven_ground_fits_at_its_lowest_point() {
        // A one-block step under the footprint's middle column.
        let mut ground: Vec<(IVec2, i32)> = (0..3).flat_map(|x| (0..3).map(move |z| (IVec2::new(x, z), 64))).collect();
        for entry in ground.iter_mut() {
            if entry.0 == IVec2::new(1, 1) {
                entry.1 = 65;
            }
        }
        let world = world_with_ground(&ground);

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 }, "base_y is the lowest sampled point, one above y=64");
    }

    #[test]
    fn a_steep_step_still_fits_at_its_lowest_point_ticket_058() {
        // A genuine 6-block terrain step under one corner — before ticket
        // 058 this refused as `TooSteep`; now it just fits at the lowest
        // sampled point, same as `uneven_ground_fits_at_its_lowest_point`'s
        // one-block step. A real Minecraft world is inherently uneven, so
        // fit_footprint no longer has an opinion on how steep is "too"
        // steep — see the module docs.
        let mut ground: Vec<(IVec2, i32)> = (0..3).flat_map(|x| (0..3).map(move |z| (IVec2::new(x, z), 64))).collect();
        for entry in ground.iter_mut() {
            if entry.0 == IVec2::new(2, 2) {
                entry.1 = 70; // 6 blocks higher
            }
        }
        let world = world_with_ground(&ground);

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 }, "fits at the lowest point (64), not refused for the 70-high corner");
    }

    #[test]
    fn a_footprint_reaching_unloaded_ground_is_refused_without_needing_every_tile() {
        // Only a 2x2 patch of ground exists; a 4x4 footprint over it reaches
        // tiles with no decoded chunk at all.
        let ground: Vec<(IVec2, i32)> = (0..2).flat_map(|x| (0..2).map(move |z| (IVec2::new(x, z), 64))).collect();
        let world = world_with_ground(&ground);

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(4, 4), Rotation::Deg0, &world);
        assert!(matches!(fit, FootprintFit::Refused(FitError::NotLoaded { .. })));
    }

    #[test]
    fn a_90_degree_rotation_samples_the_rotated_rectangle() {
        // Ground only exists for x 0..4 (not 4..5) — unrotated width is 3,
        // so x=4 is outside an unrotated 3x5 footprint but inside a
        // rotated one's swapped 5x3 extent. Ticket 058 removed the
        // `TooSteep` refusal this test used to prove rotation with, so the
        // proof now goes through `NotLoaded` at that same column instead —
        // still the same "does the sampled rectangle actually rotate"
        // question, since the only other choice is out-of-bounds terrain.
        let ground: Vec<(IVec2, i32)> = (0..4).flat_map(|x| (0..5).map(move |z| (IVec2::new(x, z), 64))).collect();
        let world = world_with_ground(&ground);

        // Unrotated: 3 wide (x 0..3), 5 deep — never touches x=4.
        let unrotated = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 5), Rotation::Deg0, &world);
        assert_eq!(unrotated, FootprintFit::Fits { base_y: 65 });

        // Rotated 90°: occupied rectangle becomes 5 wide (x 0..5), 3 deep —
        // reaches x=4, which has no ground at all.
        let rotated = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 5), Rotation::Deg90, &world);
        assert!(
            matches!(rotated, FootprintFit::Refused(FitError::NotLoaded { .. })),
            "a rotation bug that keeps sampling the unrotated rectangle would still report Fits here"
        );
    }

    #[test]
    fn a_footprint_straddling_a_chunk_boundary_reads_both_columns() {
        // Flat ground at y=64 across two adjacent chunks (x 0..16 and 16..32).
        let ground: Vec<(IVec2, i32)> = (14..18).flat_map(|x| (0..2).map(move |z| (IVec2::new(x, z), 64))).collect();
        let world = world_with_ground(&ground);

        let fit = fit_footprint(IVec3::new(14, 0, 0), IVec2::new(4, 2), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 });
    }

    #[test]
    fn a_zero_extent_footprint_does_not_panic_and_fits_at_the_origins_own_height() {
        let world = flat_chunk(64);
        let fit = fit_footprint(IVec3::new(5, 80, 5), IVec2::new(0, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 80 });
    }

    // -- ticket 052: clutter shouldn't read as ground --------------------

    #[test]
    fn is_clutter_name_covers_common_suffix_and_exact_families() {
        assert!(is_clutter_name("oak_log"));
        assert!(is_clutter_name("stripped_oak_log"));
        assert!(is_clutter_name("oak_leaves"));
        assert!(is_clutter_name("oak_fence"));
        assert!(is_clutter_name("torch"));
        assert!(is_clutter_name("short_grass"));
        assert!(!is_clutter_name("stone"));
        assert!(!is_clutter_name("oak_planks"));
        assert!(!is_clutter_name("cobblestone_wall"), "walls are structural, not clutter");
    }

    #[test]
    fn a_tree_standing_on_otherwise_flat_ground_does_not_read_as_a_cliff() {
        let mut world = flat_chunk(64);
        // A six-log trunk plus leaves in the middle of the footprint — read
        // as ground, this would be a seven-block discrepancy against the
        // flat ground everywhere else. Since ticket 058, an unexcluded
        // discrepancy like that couldn't refuse the placement any more
        // either way — see `clutter_on_the_lowest_tile_does_not_inflate_base_y`
        // for the case where clutter exclusion actually changes `base_y`.
        for y in 65..71 {
            add_block(&mut world, IVec2::new(1, 1), y, "minecraft:oak_log");
        }
        add_block(&mut world, IVec2::new(1, 1), 71, "minecraft:oak_leaves");

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 }, "the tree doesn't crash the fit or otherwise disturb it");
    }

    #[test]
    fn a_fence_post_on_flat_ground_is_skipped_too() {
        let mut world = flat_chunk(64);
        add_block(&mut world, IVec2::new(2, 2), 65, "minecraft:oak_fence");

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 });
    }

    #[test]
    fn clutter_on_the_lowest_tile_does_not_inflate_base_y() {
        // Ticket 058 made `base_y` (the lowest sampled point) the only thing
        // `is_ground`'s clutter exclusion still affects — with no refusal
        // left to test against, this replaces the old "still refuses" check
        // with a direct assertion on that value. Every tile's true terrain
        // is high (70) except one (1,1), whose true terrain is low (64) but
        // has a solid fence post sitting on it. If clutter weren't excluded,
        // that tile would sample as the fence's height (66) rather than the
        // ground beneath it (65), one block too high.
        let mut ground: Vec<(IVec2, i32)> = (0..3).flat_map(|x| (0..3).map(move |z| (IVec2::new(x, z), 70))).collect();
        for entry in ground.iter_mut() {
            if entry.0 == IVec2::new(1, 1) {
                entry.1 = 64;
            }
        }
        let mut world = world_with_ground(&ground);
        add_block(&mut world, IVec2::new(1, 1), 65, "minecraft:oak_fence");

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 }, "base_y should read the true ground under the fence, not the fence itself");
    }

    #[test]
    fn a_footprint_standing_on_clutter_all_the_way_down_is_refused_not_crashed() {
        let mut world = empty_chunk();
        for x in 0..2 {
            for z in 0..2 {
                add_block(&mut world, IVec2::new(x, z), 64, "minecraft:oak_log");
            }
        }

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(2, 2), Rotation::Deg0, &world);
        assert!(
            matches!(fit, FootprintFit::Refused(FitError::NotLoaded { .. })),
            "no ground anywhere in a decoded column should refuse, not fit at a nonsensical height"
        );
    }

    // -- block_at (ticket 071) ------------------------------------------------

    /// The three answers [`block_at`] draws apart, and the one that isn't
    /// obvious: an *absent section* inside a decoded column is uniform air,
    /// so it reads back as air rather than as "not loaded". Only the column
    /// being missing entirely is `None`.
    #[test]
    fn block_at_tells_air_apart_from_an_undecoded_column() {
        let mut world = flat_chunk(64);
        let air = world::BlockRegistry::AIR;

        // The ground block itself — `flat_chunk` puts one stone block *at*
        // the y it is given, which is why `ground_height_at` answers 65 here.
        assert_ne!(block_at(IVec3::new(3, 64, 3), &world), Some(air));
        // Air directly above it, in a section that exists.
        assert_eq!(block_at(IVec3::new(3, 65, 3), &world), Some(air));
        // Air far above it, in a section that does not exist at all.
        assert_eq!(block_at(IVec3::new(3, 200, 3), &world), Some(air));
        // A column that was never decoded is the one case that isn't an answer.
        assert_eq!(block_at(IVec3::new(500, 64, 500), &world), None);

        add_block(&mut world, IVec2::new(3, 3), 70, "minecraft:oak_leaves");
        let leaves = block_at(IVec3::new(3, 70, 3), &world).expect("the column is decoded");
        assert_eq!(world.registry.lock().unwrap().name(leaves), "minecraft:oak_leaves");
    }

    /// [`block_at`] reads *exactly* the Y it is asked for — it is not
    /// [`ground_height_at`]'s top-down scan, which is the whole reason
    /// `city::road_build` needs it to count what roofs a road cell.
    #[test]
    fn block_at_reads_one_exact_y_not_the_topmost_block() {
        let mut world = empty_chunk();
        add_block(&mut world, IVec2::new(0, 0), 100, "minecraft:stone");

        let air = world::BlockRegistry::AIR;
        assert_eq!(block_at(IVec3::new(0, 99, 0), &world), Some(air));
        assert_ne!(block_at(IVec3::new(0, 100, 0), &world), Some(air));
        assert_eq!(block_at(IVec3::new(0, 101, 0), &world), Some(air));
    }
}
