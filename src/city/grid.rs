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
//! placement `MAX_FOOTPRINT_STEP` (or ten blocks, for a tree) away from real,
//! flat terrain, even though [`crate::city::commit`]'s own write already
//! clears the obstruction without complaint. [`is_ground`] narrows the
//! predicate: still not `ranvil::heightmap`'s `blocks_motion` (a lake's
//! surface still counts as ground the way a stone hilltop does — iteration 1
//! doesn't distinguish, and that refinement really is roadmap I3's later
//! job), but clutter — vegetation, decoration, anything the write path's own
//! "air is a block" policy would already bulldoze — no longer reads as the
//! footprint's floor.
//!
//! ## No auto-level
//!
//! The roadmap asks this ticket to decide whether uneven ground under a
//! footprint gets levelled or refused. **Refused.** Levelling means writing
//! blocks, and writing blocks is W4/W5's job through a real
//! [`crate::edit::WorldEdit`] — folding it into a read-only fit check would
//! make "is this buildable" secretly depend on the write path, and every
//! future caller (E3's every-frame ghost preview included) would pay for
//! it. [`MAX_FOOTPRINT_STEP`] draws the line: within it, the footprint sits
//! at its *lowest* sampled point ([`FootprintFit::Fits`]'s `base_y`) and
//! higher corners clip a little into the building's own foundation, which
//! reads better than a gap floating over a low corner given nothing here
//! fills it in. Past the tolerance, placement is refused outright.
//! Terraforming (H1) is what would later let a player fix a steeper site by
//! hand, through the write path this module deliberately doesn't touch.
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

/// Height difference (world Y) tolerated across a footprint's sampled
/// ground before [`fit_footprint`] refuses the placement outright — see the
/// module docs' "No auto-level" note. `1`: enough to absorb a single stair
/// step in the terrain without papering over a real slope.
pub const MAX_FOOTPRINT_STEP: i32 = 1;

/// Why [`fit_footprint`] refused a placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    /// `tile`'s chunk isn't decoded in [`DecodedWorld`] yet (outside the
    /// streamed radius), or its column has no solid block anywhere in
    /// what's decoded. Both collapse to one "no answer available" case —
    /// same call [`super::picking::HoveredBlock`] already makes for its own
    /// `None`.
    NotLoaded { tile: IVec2 },
    /// The sampled ground varies by more than [`MAX_FOOTPRINT_STEP`] across
    /// the footprint.
    TooSteep { min_y: i32, max_y: i32 },
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::NotLoaded { tile } => {
                write!(f, "ground at ({}, {}) isn't loaded yet", tile.x, tile.y)
            }
            FitError::TooSteep { min_y, max_y } => write!(
                f,
                "ground varies from y={min_y} to y={max_y}, more than the {MAX_FOOTPRINT_STEP}-block limit"
            ),
        }
    }
}

impl std::error::Error for FitError {}

/// Whether a footprint's ground supports a placement, and at what height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FootprintFit {
    /// The footprint's sampled ground is level within [`MAX_FOOTPRINT_STEP`]
    /// — `base_y` is its lowest point, the world Y a placed building's
    /// floor belongs at (see the module docs' "No auto-level" note).
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

/// Samples every tile [`footprint_tiles`] covers for `footprint` placed at
/// `origin`/`rotation`, and decides whether it's buildable — see the module
/// docs for the height read and the no-auto-level rule.
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
    // `Option<(min, max)>` rather than `i32::MAX`/`i32::MIN` sentinels: a
    // zero-extent footprint (not producible by the catalogue, but not
    // guarded against by `footprint_tiles` either) samples no tiles at all,
    // and subtracting unset sentinels would overflow rather than just being
    // wrong.
    let mut bounds: Option<(i32, i32)> = None;

    for tile in footprint_tiles(origin, footprint, rotation) {
        let Some(height) = ground_height_at(tile, world) else {
            return FootprintFit::Refused(FitError::NotLoaded { tile });
        };
        bounds = Some(match bounds {
            Some((min, max)) => (min.min(height), max.max(height)),
            None => (height, height),
        });
    }

    let Some((min_y, max_y)) = bounds else {
        // No tiles to check — nothing constrains the height, so leave the
        // placement at whatever Y it was asked for rather than inventing one.
        return FootprintFit::Fits { base_y: origin.y };
    };

    if max_y - min_y > MAX_FOOTPRINT_STEP {
        return FootprintFit::Refused(FitError::TooSteep { min_y, max_y });
    }

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
    fn ground_within_tolerance_fits_at_its_lowest_point() {
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
    fn ground_past_tolerance_is_refused_as_too_steep() {
        let mut ground: Vec<(IVec2, i32)> = (0..3).flat_map(|x| (0..3).map(move |z| (IVec2::new(x, z), 64))).collect();
        for entry in ground.iter_mut() {
            if entry.0 == IVec2::new(2, 2) {
                entry.1 = 70; // 6 blocks higher, well past MAX_FOOTPRINT_STEP
            }
        }
        let world = world_with_ground(&ground);

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Refused(FitError::TooSteep { min_y: 65, max_y: 71 }));
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
        // Asymmetric terrain: a ridge along x=4 (unrotated width is 3, so
        // x=4 is outside an unrotated 3x5 footprint but inside a rotated
        // one's swapped 5x3 extent).
        let mut ground: Vec<(IVec2, i32)> = (0..5).flat_map(|x| (0..5).map(move |z| (IVec2::new(x, z), 64))).collect();
        for entry in ground.iter_mut() {
            if entry.0.x == 4 {
                entry.1 = 70;
            }
        }
        let world = world_with_ground(&ground);

        // Unrotated: 3 wide (x 0..3), 5 deep — never touches the ridge at x=4.
        let unrotated = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 5), Rotation::Deg0, &world);
        assert_eq!(unrotated, FootprintFit::Fits { base_y: 65 });

        // Rotated 90°: occupied rectangle becomes 5 wide (x 0..5), 3 deep —
        // reaches x=4's ridge, which is far past MAX_FOOTPRINT_STEP.
        let rotated = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 5), Rotation::Deg90, &world);
        assert!(
            matches!(rotated, FootprintFit::Refused(FitError::TooSteep { .. })),
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
        // flat ground everywhere else, well past MAX_FOOTPRINT_STEP.
        for y in 65..71 {
            add_block(&mut world, IVec2::new(1, 1), y, "minecraft:oak_log");
        }
        add_block(&mut world, IVec2::new(1, 1), 71, "minecraft:oak_leaves");

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 }, "a tree should not refuse an otherwise-flat placement");
    }

    #[test]
    fn a_fence_post_on_flat_ground_is_skipped_too() {
        let mut world = flat_chunk(64);
        add_block(&mut world, IVec2::new(2, 2), 65, "minecraft:oak_fence");

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert_eq!(fit, FootprintFit::Fits { base_y: 65 });
    }

    #[test]
    fn a_real_slope_still_refuses_even_with_clutter_ignored() {
        // A genuine 6-block terrain step (not clutter) under one corner —
        // ignoring clutter must not also start ignoring real elevation.
        let mut ground: Vec<(IVec2, i32)> = (0..3).flat_map(|x| (0..3).map(move |z| (IVec2::new(x, z), 64))).collect();
        for entry in ground.iter_mut() {
            if entry.0 == IVec2::new(2, 2) {
                entry.1 = 70;
            }
        }
        let world = world_with_ground(&ground);

        let fit = fit_footprint(IVec3::new(0, 0, 0), IVec2::new(3, 3), Rotation::Deg0, &world);
        assert!(matches!(fit, FootprintFit::Refused(FitError::TooSteep { .. })));
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
}
