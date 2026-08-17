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
//! "Ground" here is [`world::is_solid`]'s notion — not air, the same
//! predicate the mesher culls faces against — not `ranvil::heightmap`'s
//! `blocks_motion`. A lake's surface counts as ground the way a stone
//! hilltop does; iteration 1 doesn't distinguish. That's exactly the kind
//! of thing roadmap I3's block-classification table will eventually refine
//! once damage detection needs a real "is this actually a foundation"
//! answer — nothing here waits on that.
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

/// The world Y one above the topmost solid (non-air, [`world::is_solid`])
/// block at `tile`, via [`world::ChunkColumn::topmost_non_air`] against
/// `world.columns` directly — or `None` if `tile`'s chunk isn't decoded, or
/// nothing solid is in what's decoded. See the module docs.
#[allow(dead_code)] // no caller outside this module's own tests yet — see the module docs
pub fn ground_height_at(tile: IVec2, world: &DecodedWorld) -> Option<i32> {
    let size = world::SECTION_SIZE as i32;
    let chunk = (tile.x.div_euclid(size), tile.y.div_euclid(size));
    let column = world.columns.get(&chunk)?;
    let local_x = tile.x.rem_euclid(size) as usize;
    let local_z = tile.y.rem_euclid(size) as usize;
    column.topmost_non_air(local_x, local_z).map(|(y, _id)| y + 1)
}

/// Samples every tile [`footprint_tiles`] covers for `footprint` placed at
/// `origin`/`rotation`, and decides whether it's buildable — see the module
/// docs for the height read and the no-auto-level rule.
///
/// Fails fast on the first unresolvable tile — a footprint reaching off the
/// streamed edge is refused before every other tile is even sampled, the
/// same "stop once the answer is already no" shape
/// [`super::state::City::place_building`] uses for occupancy.
#[allow(dead_code)] // no caller yet — E3's ghost preview / E4's commit are the eventual readers
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
}
