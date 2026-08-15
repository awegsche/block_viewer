//! Tests for the edit model (ticket 031).
//!
//! Two layers, deliberately. The rules — coordinates, build limits, `Status`,
//! `DataVersion` — are pure functions over a chunk's NBT and are tested
//! without a file anywhere. Everything that actually writes runs against a
//! **synthetic region file** built in a temp directory by
//! [`RegionFixture`], rather than against the developer's real save: the
//! interesting cases (an ungenerated chunk, a half-generated one, a chunk from
//! the wrong Minecraft version) are ones a real save mostly doesn't have where
//! you need them, and a test that edits a copy of somebody's world is a test
//! that behaves differently on every machine.
//!
//! `mc_anvil::region::Region::write` takes chunk NBT and produces a real
//! `.mca`, so the fixture is byte-accurate without this module knowing
//! anything about sector tables or zlib.

use super::*;

use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION, REGION_WIDTH_IN_CHUNKS};
use mc_anvil::SaveMeta;
use rnbt::{NbtField, NbtList, NbtValue};

use crate::region_cache::RegionCache;

/// The `DataVersion` the fixtures claim: a 1.21 release, the same one the real
/// save carries.
const FIXTURE_DATA_VERSION: i32 = 4438;

/// The section the fixtures populate, and so the Y range the tests write in:
/// `Y = 0`, world Y 0..15. Deliberately not the bottommost one — an edit model
/// that only ever worked at the world bottom would hide a section-index bug.
const FIXTURE_SECTION_Y: i8 = 0;

fn stone() -> BlockState {
    BlockState {
        name: "minecraft:stone".to_string(),
        properties: Vec::new(),
    }
}

fn dirt() -> BlockState {
    BlockState {
        name: "minecraft:dirt".to_string(),
        properties: Vec::new(),
    }
}

/// A block with properties, to prove they survive the conversion into
/// `mc_anvil`'s `BlockState` and back out of the palette.
fn stairs() -> BlockState {
    BlockState {
        name: "minecraft:oak_stairs".to_string(),
        properties: vec![
            ("facing".to_string(), "east".to_string()),
            ("half".to_string(), "top".to_string()),
        ],
    }
}

// -------------------------------------------------------------------------------------------------
// ---- coordinates --------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn addresses_are_region_relative_and_round_towards_the_world_bottom() {
    let at = address_of(IVec3::new(0, 5, 0));
    assert_eq!(at.chunk, (0, 0));
    assert_eq!(at.region, (0, 0));
    assert_eq!((at.local_x, at.local_z), (0, 0));

    // The last block of region (0,0).
    let at = address_of(IVec3::new(511, 5, 511));
    assert_eq!(at.region, (0, 0));
    assert_eq!(at.chunk, (31, 31));
    assert_eq!((at.local_x, at.local_z), (511, 511));

    // One further is the next region's first block.
    let at = address_of(IVec3::new(512, 5, 512));
    assert_eq!(at.region, (1, 1));
    assert_eq!(at.chunk, (32, 32));
    assert_eq!((at.local_x, at.local_z), (0, 0));
}

#[test]
fn negative_coordinates_do_not_truncate_towards_zero() {
    // The bug every world editor writes once: `/` and `%` put x = -1 in chunk
    // 0 at local -1, when it belongs to chunk -1 at local 511.
    let at = address_of(IVec3::new(-1, 5, -1));
    assert_eq!(at.chunk, (-1, -1));
    assert_eq!(at.region, (-1, -1));
    assert_eq!((at.local_x, at.local_z), (511, 511));

    let at = address_of(IVec3::new(-512, 5, -513));
    assert_eq!(at.region, (-1, -2));
    assert_eq!((at.local_x, at.local_z), (0, 511));

    // ...and the same rounding for the section a Y falls in.
    assert_eq!(section_y_of(0), 0);
    assert_eq!(section_y_of(15), 0);
    assert_eq!(section_y_of(-1), -1);
    assert_eq!(section_y_of(-64), -4);
    assert_eq!(section_y_of(319), 19);
}

// -------------------------------------------------------------------------------------------------
// ---- the per-chunk rules, without a region file --------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A chunk root carrying only the tags the rules read.
fn chunk_root(status: Option<&str>, data_version: Option<i32>) -> NbtField {
    let mut fields = Vec::new();
    if let Some(status) = status {
        fields.push(NbtField::new_string("Status", status));
    }
    if let Some(version) = data_version {
        fields.push(NbtField::new_i32("DataVersion", version));
    }
    NbtField::new_compound("", fields)
}

#[test]
fn a_half_generated_chunk_is_refused() {
    let nbt = chunk_root(Some("minecraft:features"), Some(FIXTURE_DATA_VERSION));
    let refusal = check_chunk_nbt((3, 4), &nbt, &EditPolicy::default(), None).unwrap_err();

    assert_eq!(
        refusal,
        EditRefusal::StatusNotFull {
            chunk: (3, 4),
            status: "minecraft:features".to_string()
        }
    );

    // ...and a chunk with no `Status` at all is refused too, rather than
    // assumed finished. We're about to write into it.
    let nbt = chunk_root(None, Some(FIXTURE_DATA_VERSION));
    assert!(matches!(
        check_chunk_nbt((3, 4), &nbt, &EditPolicy::default(), None),
        Err(EditRefusal::StatusNotFull { .. })
    ));
}

#[test]
fn a_data_version_mismatch_is_refused_and_can_be_overridden() {
    let nbt = chunk_root(Some("minecraft:full"), Some(FIXTURE_DATA_VERSION));

    assert_eq!(
        check_chunk_nbt((0, 0), &nbt, &EditPolicy::default(), Some(3953)).unwrap_err(),
        EditRefusal::DataVersionMismatch {
            chunk: (0, 0),
            save: FIXTURE_DATA_VERSION,
            edit: 3953
        }
    );

    // The same version passes...
    assert!(
        check_chunk_nbt(
            (0, 0),
            &nbt,
            &EditPolicy::default(),
            Some(FIXTURE_DATA_VERSION)
        )
        .is_ok()
    );

    // ...and so does an edit that makes no version claim at all, which is what
    // an edit built from block names in code looks like.
    assert!(check_chunk_nbt((0, 0), &nbt, &EditPolicy::default(), None).is_ok());

    // The override is for a UI offering "do it anyway".
    let lenient = EditPolicy {
        enforce_data_version: false,
        ..EditPolicy::default()
    };
    assert!(check_chunk_nbt((0, 0), &nbt, &lenient, Some(3953)).is_ok());
}

// -------------------------------------------------------------------------------------------------
// ---- the region fixture -------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A synthetic region file in a temp directory, removed on drop.
struct RegionFixture {
    dir: std::path::PathBuf,
    path: std::path::PathBuf,
    coord: (i32, i32),
}

impl RegionFixture {
    /// Region (0,0) with four chunks in it:
    ///
    /// - **(0,0)** and **(1,0)**: ordinary finished chunks, one section of
    ///   stone at `Y = 0`, `Status = minecraft:full`, `isLightOn = 1`,
    ///   `DataVersion` 4438, and `Heightmaps` claiming a flat surface.
    /// - **(2,0)**: finished, but written by an older Minecraft
    ///   (`DataVersion` 3953).
    /// - **(3,0)**: still generating (`Status = minecraft:features`).
    /// - every other slot is empty, i.e. ungenerated terrain.
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("block_viewer-edit-{label}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let path = write_region(
            &dir,
            (0, 0),
            &[
                ((0, 0), full_chunk(0, 0, FIXTURE_DATA_VERSION)),
                ((1, 0), full_chunk(1, 0, FIXTURE_DATA_VERSION)),
                ((2, 0), full_chunk(2, 0, 3953)),
                ((3, 0), unfinished_chunk(3, 0)),
            ],
        );

        Self {
            dir,
            path,
            coord: (0, 0),
        }
    }

    fn load(&self) -> ChunkRegion {
        let mut region: ChunkRegion = Region::new(self.coord.0, self.coord.1, &self.path).into();
        region.load_chunks().expect("load the fixture region");
        region
    }

    fn bytes(&self) -> Vec<u8> {
        std::fs::read(&self.path).expect("read the fixture region")
    }
}

impl Drop for RegionFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Writes one `r.<x>.<z>.mca` into `dir`, with `chunks` given by **world**
/// chunk coordinate — the region-local slot each one lands in comes from the
/// same `local_chunk_index` the read path uses, so a fixture for a negative
/// region doesn't need the arithmetic done by hand. Returns its path.
fn write_region(
    dir: &std::path::Path,
    region_coord: (i32, i32),
    chunks: &[((i32, i32), NbtField)],
) -> std::path::PathBuf {
    let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
    for (chunk, nbt) in chunks {
        let (local_x, local_z) = local_chunk_index(*chunk, region_coord);
        payloads[local_z * REGION_WIDTH_IN_CHUNKS + local_x] =
            Some(ChunkPayload::Nbt(nbt.clone()));
    }

    let (rx, rz) = region_coord;
    let path = dir.join(format!("r.{rx}.{rz}.mca"));
    Region::new(rx, rz, &path)
        .write(&payloads)
        .expect("write the fixture region");
    path
}

/// A finished chunk: one all-stone section at `Y = 0`, plus the chunk-root
/// tags the edit model and `mc_anvil` read.
fn full_chunk(x: i32, z: i32, data_version: i32) -> NbtField {
    let mut fields = chunk_fields(x, z, data_version);
    fields.push(NbtField::new_string("Status", "minecraft:full"));
    NbtField::new_compound("", fields)
}

/// A chunk the generator hasn't finished with.
fn unfinished_chunk(x: i32, z: i32) -> NbtField {
    let mut fields = chunk_fields(x, z, FIXTURE_DATA_VERSION);
    fields.push(NbtField::new_string("Status", "minecraft:features"));
    NbtField::new_compound("", fields)
}

fn chunk_fields(x: i32, z: i32, data_version: i32) -> Vec<NbtField> {
    // A one-entry palette and no packed `data`, which is how Minecraft stores
    // a uniform section — `set_blocks` has to grow both.
    let palette = NbtList::Compound(vec![NbtField::new_compound(
        "",
        vec![NbtField::new_string("Name", "minecraft:stone")],
    )]);
    let section = NbtField::new_compound(
        "",
        vec![
            NbtField {
                name: "Y".to_string(),
                value: NbtValue::Byte(FIXTURE_SECTION_Y as u8),
            },
            NbtField::new_compound("block_states", vec![NbtField::new_list("palette", palette)]),
        ],
    );

    // Stone to the top of the one section: world Y 15, so every column's
    // heightmap value is `15 + 1 - (-64)` = 80.
    let heights = [80u16; 256];
    let mut longs = vec![0i64; 37];
    for (column, value) in heights.iter().enumerate() {
        longs[column / 7] |= (*value as i64) << ((column % 7) * 9);
    }
    let heightmaps = NbtField::new_compound(
        "Heightmaps",
        [
            "MOTION_BLOCKING",
            "MOTION_BLOCKING_NO_LEAVES",
            "OCEAN_FLOOR",
            "WORLD_SURFACE",
        ]
        .iter()
        .map(|key| NbtField::new_long_array(*key, longs.clone()))
        .collect::<Vec<_>>(),
    );

    vec![
        NbtField::new_list("sections", NbtList::Compound(vec![section])),
        NbtField::new_i32("xPos", x),
        NbtField::new_i32("zPos", z),
        NbtField::new_i32("yPos", -4),
        NbtField::new_i32("DataVersion", data_version),
        NbtField {
            name: "isLightOn".to_string(),
            value: NbtValue::Byte(1),
        },
        heightmaps,
    ]
}

/// The block name at a world position, read back through the region.
fn block_name_at(region: &ChunkRegion, at: IVec3) -> String {
    let address = address_of(at);
    region
        .get_block(address.local_x, address.y, address.local_z)
        .expect("a populated chunk")
        .get_string("Name")
        .expect("a palette entry")
        .clone()
}

fn light_flag(region: &ChunkRegion, chunk: (i32, i32)) -> Option<u8> {
    region
        .get_chunk(chunk.0 as usize, chunk.1 as usize)
        .expect("a populated chunk")
        .get_byte("isLightOn")
}

/// A single-block edit at a position inside the fixture's stone.
fn one_block(at: IVec3, state: BlockState) -> WorldEdit {
    let mut edit = WorldEdit::new();
    edit.set(at, state);
    edit
}

// -------------------------------------------------------------------------------------------------
// ---- planning -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn an_empty_edit_is_refused_rather_than_reported_as_success() {
    let fixture = RegionFixture::new("empty");
    let region = fixture.load();

    assert_eq!(
        plan(&WorldEdit::new(), &region, &EditPolicy::default()).unwrap_err(),
        EditRefusal::Empty
    );
}

#[test]
fn the_build_limits_are_the_selections() {
    let fixture = RegionFixture::new("limits");
    let region = fixture.load();

    for y in [WORLD_MAX_Y + 1, WORLD_MIN_Y - 1] {
        let at = IVec3::new(4, y, 4);
        assert_eq!(
            plan(&one_block(at, stone()), &region, &EditPolicy::default()).unwrap_err(),
            EditRefusal::OutsideBuildLimits { at }
        );
    }
}

#[test]
fn a_position_in_another_region_is_refused_not_wrapped() {
    // The trap this exists for: region-local coordinates are `rem_euclid(512)`,
    // so x = 512 would silently land at local 0 of *this* region — a building
    // placed 512 blocks from where the user asked for it, in a file they
    // weren't editing.
    let fixture = RegionFixture::new("region");
    let region = fixture.load();
    let at = IVec3::new(512, 5, 5);

    assert_eq!(
        plan(&one_block(at, stone()), &region, &EditPolicy::default()).unwrap_err(),
        EditRefusal::OutsideRegion { at, region: (0, 0) }
    );
}

#[test]
fn an_ungenerated_chunk_is_refused() {
    let fixture = RegionFixture::new("ungenerated");
    let region = fixture.load();

    // Chunk (5,0) is one of the 1020 empty slots.
    assert_eq!(
        plan(
            &one_block(IVec3::new(5 * 16, 5, 0), stone()),
            &region,
            &EditPolicy::default()
        )
        .unwrap_err(),
        EditRefusal::ChunkNotGenerated { chunk: (5, 0) }
    );
}

#[test]
fn a_missing_section_is_refused() {
    let fixture = RegionFixture::new("section");
    let region = fixture.load();

    // The fixture chunks carry one section, `Y = 0`. Y = 100 is inside the
    // build limits and inside a generated chunk, and still has nowhere to go —
    // writes never create sections.
    assert_eq!(
        plan(
            &one_block(IVec3::new(4, 100, 4), stone()),
            &region,
            &EditPolicy::default()
        )
        .unwrap_err(),
        EditRefusal::SectionMissing {
            chunk: (0, 0),
            section_y: 6
        }
    );
}

#[test]
fn planning_counts_blocks_and_chunks_and_writes_nothing() {
    let fixture = RegionFixture::new("plan");
    let before = fixture.bytes();
    let mut region = fixture.load();

    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(1, 5, 1), dirt());
    edit.set(IVec3::new(2, 5, 1), dirt());
    // Same position twice: one block in the world, so counted once.
    edit.set(IVec3::new(2, 5, 1), stone());
    // ...and one in the neighbouring chunk.
    edit.set(IVec3::new(16, 5, 1), dirt());

    let report = plan(&edit, &region, &EditPolicy::default()).expect("a valid edit");
    assert_eq!(report.blocks_written, 3);
    assert_eq!(report.chunks, vec![(0, 0), (1, 0)]);
    assert_eq!(report.replaced, None);

    // The dry run is a dry run: nothing dirtied, nothing on disk changed.
    assert!(!region.is_dirty());
    region.save().expect("a no-op save");
    assert_eq!(fixture.bytes(), before);
}

// -------------------------------------------------------------------------------------------------
// ---- applying -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn an_edit_lands_in_the_world_and_survives_a_save_and_reload() {
    let fixture = RegionFixture::new("apply");
    let mut region = fixture.load();

    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(1, 5, 1), dirt());
    edit.set(IVec3::new(2, 5, 1), stairs());

    let report = apply(&edit, &mut region, &EditPolicy::default()).expect("a valid edit");
    assert_eq!(report.blocks_written, 2);
    assert!(region.is_dirty());

    region.save().expect("save");
    let reloaded = fixture.load();

    assert_eq!(
        block_name_at(&reloaded, IVec3::new(1, 5, 1)),
        "minecraft:dirt"
    );
    assert_eq!(
        block_name_at(&reloaded, IVec3::new(2, 5, 1)),
        "minecraft:oak_stairs"
    );
    // Untouched blocks are still what they were.
    assert_eq!(
        block_name_at(&reloaded, IVec3::new(3, 5, 1)),
        "minecraft:stone"
    );

    // The properties survived the round trip through the palette — a stair
    // that comes back with none is the failure ticket 022 exists to prevent,
    // one direction over.
    let address = address_of(IVec3::new(2, 5, 1));
    let entry = reloaded
        .get_block(address.local_x, address.y, address.local_z)
        .expect("the stairs");
    assert_eq!(
        entry
            .get_compound("Properties")
            .expect("stairs have properties")
            .len(),
        2
    );
}

#[test]
fn a_refused_edit_changes_absolutely_nothing() {
    // The single most important test here: one bad position in a batch must
    // not leave the good ones written.
    let fixture = RegionFixture::new("refused");
    let before = fixture.bytes();
    let mut region = fixture.load();

    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(1, 5, 1), dirt());
    edit.set(IVec3::new(2, 5, 1), dirt());
    // Chunk (3,0) is still generating, so the whole edit is refused.
    edit.set(IVec3::new(3 * 16 + 1, 5, 1), dirt());

    assert!(matches!(
        apply(&edit, &mut region, &EditPolicy::default()),
        Err(EditRefusal::StatusNotFull { chunk: (3, 0), .. })
    ));

    assert_eq!(
        block_name_at(&region, IVec3::new(1, 5, 1)),
        "minecraft:stone"
    );
    assert!(
        !region.is_dirty(),
        "a refused edit must not dirty the region"
    );

    region.save().expect("a no-op save");
    assert_eq!(fixture.bytes(), before, "the file must be byte-identical");
}

#[test]
fn an_edit_hands_the_lighting_back_to_the_game_for_the_chunks_it_touched() {
    // `mc_anvil` clears `isLightOn` inside `set_blocks` (its ticket 014). This
    // is the cross-crate assertion that it's actually wired in — nothing in
    // this repo checked it before.
    let fixture = RegionFixture::new("light");
    let mut region = fixture.load();

    apply(
        &one_block(IVec3::new(1, 5, 1), dirt()),
        &mut region,
        &EditPolicy::default(),
    )
    .expect("a valid edit");

    assert_eq!(light_flag(&region, (0, 0)), Some(0));
    assert_eq!(light_flag(&region, (1, 0)), Some(1), "an untouched chunk");
}

// -------------------------------------------------------------------------------------------------
// ---- heightmap policy ----------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A stand-in for the block registry a real caller brings — enough for the
/// fixture's three blocks. The real one is a table built against
/// `world::BlockRegistry`, and isn't worth building until the in-game check
/// says `Delete` isn't good enough.
fn classify(state: &AnvilBlockState) -> HeightmapClass {
    if state.is_air() {
        HeightmapClass::AIR
    } else {
        HeightmapClass::SOLID
    }
}

fn surface_at(region: &ChunkRegion, chunk: (i32, i32), dx: usize, dz: usize) -> i32 {
    region
        .heightmap(
            chunk.0 as usize,
            chunk.1 as usize,
            mc_anvil::HeightmapKind::WorldSurface,
        )
        .expect("the chunk has heightmaps")[mc_anvil::heightmap::column_index(dx, dz)]
}

#[test]
fn the_default_policy_deletes_the_heightmaps_of_the_touched_chunks_only() {
    let fixture = RegionFixture::new("hm-delete");
    let mut region = fixture.load();

    apply(
        &one_block(IVec3::new(1, 5, 1), dirt()),
        &mut region,
        &EditPolicy::default(),
    )
    .expect("a valid edit");

    assert!(
        region
            .get_chunk(0, 0)
            .expect("populated")
            .get("Heightmaps")
            .is_none()
    );
    assert!(
        region
            .get_chunk(1, 0)
            .expect("populated")
            .get("Heightmaps")
            .is_some(),
        "an untouched chunk keeps its heightmaps"
    );
}

#[test]
fn recomputing_follows_the_blocks_and_leaving_them_alone_goes_stale() {
    let fixture = RegionFixture::new("hm-recompute");

    // Digging the top two blocks out of one column lowers its surface from 16
    // (one above the section's top block, Y = 15) to 14.
    let air = BlockState::air();
    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(1, 15, 1), air.clone());
    edit.set(IVec3::new(1, 14, 1), air);

    let mut region = fixture.load();
    apply(
        &edit,
        &mut region,
        &EditPolicy {
            heightmaps: HeightmapPolicy::Recompute(classify),
            ..EditPolicy::default()
        },
    )
    .expect("a valid edit");

    assert_eq!(surface_at(&region, (0, 0), 1, 1), 14);
    assert_eq!(surface_at(&region, (0, 0), 2, 1), 16, "an untouched column");

    // The same edit with `Leave` keeps the pre-edit claim — which is what
    // "stale" looks like, and why the default isn't `Leave`.
    let mut region = fixture.load();
    apply(
        &edit,
        &mut region,
        &EditPolicy {
            heightmaps: HeightmapPolicy::Leave,
            ..EditPolicy::default()
        },
    )
    .expect("a valid edit");

    assert_eq!(surface_at(&region, (0, 0), 1, 1), 16);
}

// -------------------------------------------------------------------------------------------------
// ---- the as-built baseline -----------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn capturing_the_replaced_blocks_gives_an_undo() {
    let fixture = RegionFixture::new("baseline");
    let mut region = fixture.load();

    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(1, 5, 1), dirt());
    edit.set(IVec3::new(2, 5, 1), stairs());

    let policy = EditPolicy {
        capture_replaced: true,
        ..EditPolicy::default()
    };
    let report = apply(&edit, &mut region, &policy).expect("a valid edit");

    let replaced = report.replaced.expect("asked for it");
    assert_eq!(
        replaced,
        vec![
            (IVec3::new(1, 5, 1), stone()),
            (IVec3::new(2, 5, 1), stone()),
        ]
    );

    // Which is exactly an undo: play it back and the world is as it was.
    let undo: WorldEdit = replaced
        .into_iter()
        .map(|(at, state)| BlockEdit { at, state })
        .collect();
    apply(&undo, &mut region, &EditPolicy::default()).expect("the undo applies");

    assert_eq!(
        block_name_at(&region, IVec3::new(1, 5, 1)),
        "minecraft:stone"
    );
    assert_eq!(
        block_name_at(&region, IVec3::new(2, 5, 1)),
        "minecraft:stone"
    );
}

#[test]
fn the_baseline_is_off_by_default_because_it_costs_a_read_per_block() {
    let fixture = RegionFixture::new("baseline-off");
    let mut region = fixture.load();

    let report = apply(
        &one_block(IVec3::new(1, 5, 1), dirt()),
        &mut region,
        &EditPolicy::default(),
    )
    .expect("a valid edit");

    assert_eq!(report.replaced, None);
}

// -------------------------------------------------------------------------------------------------
// ---- routing across region files (ticket 032) ----------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// A synthetic *save*: a `region/` directory of `.mca` files and the
/// [`SaveMeta`] that lists them, so the routed edit can be driven both against
/// bare regions and against a real [`RegionCache`].
struct SaveFixture {
    dir: std::path::PathBuf,
    meta: SaveMeta,
}

impl SaveFixture {
    /// The four regions that meet at the world origin — `(-1,-1)`, `(-1,0)`,
    /// `(0,-1)` and `(0,0)` — each with exactly the one chunk that touches the
    /// corner generated, and nothing else.
    ///
    /// The origin corner is the four-way junction, so a small building placed
    /// across it exercises the maximum fan-out *and* negative coordinates in
    /// the same edit — which is the pair of things that go wrong together.
    fn corner(label: &str) -> Self {
        Self::build(label, full_chunk(0, 0, FIXTURE_DATA_VERSION))
    }

    /// [`SaveFixture::corner`] with the origin chunk's section malformed, so
    /// the preflight accepts it and `set_blocks` refuses it — the phase-2
    /// failure the rollback exists for. Chunk `(0,0)` is the *last* region the
    /// routed apply visits, so three regions are already applied when it
    /// happens.
    fn corner_with_a_malformed_origin(label: &str) -> Self {
        Self::build(label, malformed_chunk(0, 0))
    }

    fn build(label: &str, origin_chunk: NbtField) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("block_viewer-route-{label}-{nanos}"));
        let region_dir = dir.join("region");
        std::fs::create_dir_all(&region_dir).expect("create temp dir");

        for chunk in [(-1, -1), (-1, 0), (0, -1)] {
            write_region(
                &region_dir,
                chunk_to_region_coord(chunk),
                &[(chunk, full_chunk(chunk.0, chunk.1, FIXTURE_DATA_VERSION))],
            );
        }
        write_region(&region_dir, (0, 0), &[((0, 0), origin_chunk)]);

        let meta = SaveMeta {
            name: "route-fixture".to_string(),
            path: dir.clone(),
            region_dir,
            regions: vec![(-1, -1), (-1, 0), (0, -1), (0, 0)],
        };
        Self { dir, meta }
    }

    fn load(&self, coord: (i32, i32)) -> ChunkRegion {
        load_region_file(&self.meta.get_region_path(coord.0, coord.1), coord)
    }

    fn region_bytes(&self, coord: (i32, i32)) -> Vec<u8> {
        std::fs::read(self.meta.get_region_path(coord.0, coord.1)).expect("read the region file")
    }

    /// Every region loaded up front, as a [`RegionSource`] with no cache and
    /// no eviction in it.
    fn regions(&self) -> FixtureRegions {
        FixtureRegions {
            regions: self.meta.regions.iter().map(|c| (*c, self.load(*c))).collect(),
            discarded: Vec::new(),
        }
    }

    fn cache(&self, capacity: usize) -> RegionCache {
        RegionCache::new(self.meta.clone(), capacity)
    }
}

impl Drop for SaveFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Any `.mca` file, loaded as the region at `coord` — the fixture's own files
/// and, in the write-safety tests below, the backup copies of them.
fn load_region_file(path: &std::path::Path, coord: (i32, i32)) -> ChunkRegion {
    let mut region: ChunkRegion = Region::new(coord.0, coord.1, path).into();
    region.load_chunks().expect("load the region file");
    region
}

/// A chunk that passes every preflight check and still can't be written to:
/// finished, current, with a section at `Y = 0` that has no `block_states` at
/// all. `check_set_block` finds the section and says yes; `set_blocks` needs
/// the palette and says no.
fn malformed_chunk(x: i32, z: i32) -> NbtField {
    let section = NbtField::new_compound(
        "",
        vec![NbtField {
            name: "Y".to_string(),
            value: NbtValue::Byte(FIXTURE_SECTION_Y as u8),
        }],
    );
    NbtField::new_compound(
        "",
        vec![
            NbtField::new_list("sections", NbtList::Compound(vec![section])),
            NbtField::new_i32("xPos", x),
            NbtField::new_i32("zPos", z),
            NbtField::new_i32("yPos", -4),
            NbtField::new_i32("DataVersion", FIXTURE_DATA_VERSION),
            NbtField::new_string("Status", "minecraft:full"),
        ],
    )
}

/// A [`RegionSource`] over regions already in memory: no save, no cache, no
/// eviction — only the routing rules under test. Records what was discarded,
/// which is how the rollback is observed at this level; that discarding
/// actually restores the pre-edit blocks is the *cache's* behaviour and is
/// tested through one.
struct FixtureRegions {
    regions: BTreeMap<(i32, i32), ChunkRegion>,
    discarded: Vec<(i32, i32)>,
}

impl RegionSource for FixtureRegions {
    fn region_mut(&mut self, coord: (i32, i32)) -> Result<&mut ChunkRegion, RegionUnavailable> {
        self.regions
            .get_mut(&coord)
            .ok_or(RegionUnavailable::NotGenerated)
    }

    fn discard(&mut self, coord: (i32, i32)) {
        self.regions.remove(&coord);
        self.discarded.push(coord);
    }

    fn dirty_regions(&self) -> Vec<(i32, i32)> {
        self.regions
            .iter()
            .filter(|(_, region)| region.is_dirty())
            .map(|(coord, _)| *coord)
            .collect()
    }
}

/// The 4x4 footprint centred on the world origin: 16 blocks, four blocks in
/// each of the four regions that meet there.
fn corner_building(state: BlockState) -> WorldEdit {
    let mut edit = WorldEdit::new();
    for x in -2..2 {
        for z in -2..2 {
            edit.set(IVec3::new(x, 5, z), state.clone());
        }
    }
    edit
}

// ---- the split -----------------------------------------------------------------------------------

#[test]
fn routing_splits_an_edit_by_region_file() {
    let mut edit = WorldEdit::new().with_data_version(FIXTURE_DATA_VERSION);
    edit.set(IVec3::new(-1, 5, -1), stone());
    edit.set(IVec3::new(0, 5, 0), dirt());
    // The last block of region (0,0), and the first of region (1,0).
    edit.set(IVec3::new(511, 5, 0), stone());
    edit.set(IVec3::new(512, 5, 0), dirt());

    let routed = route(&edit);

    assert_eq!(
        routed.keys().copied().collect::<Vec<_>>(),
        vec![(-1, -1), (0, 0), (1, 0)]
    );
    assert_eq!(routed[&(0, 0)].len(), 2);
    assert_eq!(routed[&(-1, -1)].len(), 1);
    // Every sub-edit inherits the parent's version claim, or nothing would be
    // refused on a mismatch once it had been split.
    assert_eq!(
        routed[&(1, 0)].data_version(),
        Some(FIXTURE_DATA_VERSION),
        "the split has to carry the DataVersion claim onto every piece"
    );
}

#[test]
fn routing_keeps_last_write_wins_within_a_region() {
    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(0, 5, 0), dirt());
    // A write to another region in between, which must not disturb the order
    // of the two that share one.
    edit.set(IVec3::new(600, 5, 0), stone());
    edit.set(IVec3::new(0, 5, 0), stairs());

    let routed = route(&edit);
    let origin = &routed[&(0, 0)];

    assert_eq!(origin.edits().len(), 2);
    assert_eq!(origin.edits()[1].state, stairs());
}

// ---- applying across regions ---------------------------------------------------------------------

#[test]
fn a_building_on_a_region_corner_lands_in_all_four_files() {
    let save = SaveFixture::corner("corner");
    let mut source = save.regions();

    let report = apply_routed(
        &corner_building(dirt()),
        &mut source,
        &EditPolicy::default(),
    )
    .expect("all four regions are generated");

    assert_eq!(report.blocks_written, 16);
    assert_eq!(report.regions, vec![(-1, -1), (-1, 0), (0, -1), (0, 0)]);
    // One chunk per region here, and the chunk coordinates happen to be the
    // same four numbers — the corner is where both grids meet.
    assert_eq!(report.chunks, vec![(-1, -1), (-1, 0), (0, -1), (0, 0)]);

    for x in -2..2 {
        for z in -2..2 {
            let at = IVec3::new(x, 5, z);
            let region = &source.regions[&address_of(at).region];
            assert_eq!(
                block_name_at(region, at),
                "minecraft:dirt",
                "the block at ({x}, 5, {z}) should have been written"
            );
        }
    }
    assert!(
        source.regions.values().all(|region| region.is_dirty()),
        "all four region files need saving now"
    );
}

#[test]
fn one_refused_region_leaves_every_other_region_untouched() {
    let save = SaveFixture::corner("refused");
    let mut source = save.regions();

    let mut edit = corner_building(dirt());
    // Chunk (-1, 2) is in region (-1, 0), which exists — but that chunk was
    // never generated, and iteration 1 doesn't generate terrain.
    edit.set(IVec3::new(-2, 5, 40), dirt());

    assert_eq!(
        apply_routed(&edit, &mut source, &EditPolicy::default()).unwrap_err(),
        EditRefusal::ChunkNotGenerated { chunk: (-1, 2) }
    );

    // Three quarters of a building is worse than none: the preflight covers
    // every region before any of them is written.
    assert!(
        source.regions.values().all(|region| !region.is_dirty()),
        "a refused transaction must not dirty a single region"
    );
    assert_eq!(
        block_name_at(&source.regions[&(-1, -1)], IVec3::new(-2, 5, -2)),
        "minecraft:stone"
    );
}

#[test]
fn a_failure_after_the_preflight_rolls_the_applied_regions_back() {
    let save = SaveFixture::corner_with_a_malformed_origin("rollback");
    let mut source = save.regions();

    let refusal = apply_routed(
        &corner_building(dirt()),
        &mut source,
        &EditPolicy::default(),
    )
    .unwrap_err();

    assert!(
        matches!(refusal, EditRefusal::Rejected { .. }),
        "the preflight can't model a malformed section; set_blocks refuses it: {refusal:?}"
    );
    // The three regions that did apply were thrown away rather than left
    // holding a quarter of a building each.
    assert_eq!(source.discarded, vec![(-1, -1), (-1, 0), (0, -1)]);
    assert!(
        source.regions.values().all(|region| !region.is_dirty()),
        "nothing dirty survives a rolled-back transaction"
    );
}

#[test]
fn a_region_the_save_never_generated_is_refused_rather_than_generated() {
    let save = SaveFixture::corner("no-region");
    let mut source = save.regions();

    assert_eq!(
        apply_routed(
            &one_block(IVec3::new(1000, 5, 0), dirt()),
            &mut source,
            &EditPolicy::default()
        )
        .unwrap_err(),
        EditRefusal::RegionNotGenerated { region: (1, 0) }
    );
}

#[test]
fn a_region_with_unsaved_changes_is_refused_unless_the_policy_allows_it() {
    let save = SaveFixture::corner("dirty");
    let mut source = save.regions();

    apply_routed(
        &one_block(IVec3::new(0, 5, 0), dirt()),
        &mut source,
        &EditPolicy::default(),
    )
    .expect("the first transaction");

    // The rollback discards a whole region, so a second transaction over the
    // same one would put the first at risk. One transaction at a time, saved
    // in between — which is the contract W6 implements.
    let second = one_block(IVec3::new(1, 5, 1), stairs());
    assert_eq!(
        apply_routed(&second, &mut source, &EditPolicy::default()).unwrap_err(),
        EditRefusal::RegionHasUnsavedChanges { region: (0, 0) }
    );

    let batching = EditPolicy {
        allow_dirty_regions: true,
        ..EditPolicy::default()
    };
    apply_routed(&second, &mut source, &batching).expect("the caller took the risk knowingly");
    assert_eq!(
        block_name_at(&source.regions[&(0, 0)], IVec3::new(1, 5, 1)),
        "minecraft:oak_stairs"
    );
}

#[test]
fn the_merged_baseline_is_sorted_across_regions() {
    let save = SaveFixture::corner("baseline");
    let mut source = save.regions();

    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(0, 5, 0), dirt());
    edit.set(IVec3::new(-1, 5, -1), dirt());

    let report = apply_routed(
        &edit,
        &mut source,
        &EditPolicy {
            capture_replaced: true,
            ..EditPolicy::default()
        },
    )
    .expect("a valid edit");

    // Region-major order isn't position order, so the merge sorts: the undo
    // record has to look the same however the edit was split.
    assert_eq!(
        report.replaced.expect("asked for it"),
        vec![
            (IVec3::new(-1, 5, -1), stone()),
            (IVec3::new(0, 5, 0), stone()),
        ]
    );
}

// ---- the cache as a region source ------------------------------------------------------------------

/// An edit spanning the two diagonally opposite regions of the corner
/// fixture — two files, four regions' worth of cache pressure between them.
fn two_region_edit() -> WorldEdit {
    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(-1, 5, -1), dirt());
    edit.set(IVec3::new(0, 5, 0), dirt());
    edit
}

#[test]
fn the_cache_will_not_evict_a_region_with_unsaved_changes() {
    let save = SaveFixture::corner("cache-guard");
    // Capacity 1: without the guard, applying the second region would evict
    // the first and the edit in it would vanish silently.
    let mut cache = save.cache(1);

    apply_routed(&two_region_edit(), &mut cache, &EditPolicy::default()).expect("a valid edit");

    assert_eq!(cache.len(), 2, "capacity is exceeded rather than an edit lost");
    assert_eq!(
        cache.dirty_regions().collect::<BTreeSet<_>>(),
        BTreeSet::from([(-1, -1), (0, 0)])
    );
    // And the cache serves post-edit blocks: the streaming pipeline reads
    // through this same entry, so it can't go on handing out the old ones.
    assert_eq!(
        block_name_at(cache.get_or_load((0, 0)).expect("resident"), IVec3::new(0, 5, 0)),
        "minecraft:dirt"
    );
}

#[test]
fn saving_lets_ordinary_eviction_resume() {
    let save = SaveFixture::corner("cache-save");
    let mut cache = save.cache(1);

    apply_routed(&two_region_edit(), &mut cache, &EditPolicy::default()).expect("a valid edit");
    for coord in cache.dirty_regions().collect::<Vec<_>>() {
        cache
            .get_or_load_mut(coord)
            .expect("resident")
            .save()
            .expect("save the region");
    }
    assert_eq!(cache.dirty_regions().count(), 0);

    // `dirty_regions` comes back in no particular order, and saving touches
    // what it returns — so say which region is most-recently-used rather than
    // depending on which order the two got saved in.
    cache.get_or_load((0, 0)).expect("resident");

    // (-1,-1) is now the least-recently-used clean region, so a third region
    // takes its place instead of pushing the cache further over capacity.
    cache.get_or_load((-1, 0)).expect("a generated region");
    assert!(!cache.is_resident((-1, -1)));
    assert_eq!(cache.len(), 2);

    // ...and what was saved is on disk, not just in memory.
    assert_eq!(
        block_name_at(&save.load((-1, -1)), IVec3::new(-1, 5, -1)),
        "minecraft:dirt"
    );
}

#[test]
fn discarding_a_region_throws_the_unsaved_edit_away() {
    let save = SaveFixture::corner("cache-discard");
    let mut cache = save.cache(4);

    apply_routed(
        &one_block(IVec3::new(0, 5, 0), dirt()),
        &mut cache,
        &EditPolicy::default(),
    )
    .expect("a valid edit");
    assert_eq!(cache.dirty_regions().count(), 1);

    assert!(cache.discard((0, 0)));
    assert!(!cache.is_resident((0, 0)));

    // Which is the rollback: nothing was saved, so re-reading the file is an
    // exact undo of everything that had been applied to it.
    assert_eq!(
        block_name_at(cache.get_or_load((0, 0)).expect("reloaded"), IVec3::new(0, 5, 0)),
        "minecraft:stone"
    );
    assert_eq!(cache.dirty_regions().count(), 0);
}

#[test]
fn a_region_outside_the_save_is_refused_through_the_cache_too() {
    let save = SaveFixture::corner("cache-missing");
    let mut cache = save.cache(4);

    // The cache answers "not found" for a region the save never had and for
    // one that won't load; only it knows which, so only it can tell them apart.
    assert_eq!(
        apply_routed(
            &one_block(IVec3::new(5000, 5, 0), dirt()),
            &mut cache,
            &EditPolicy::default()
        )
        .unwrap_err(),
        EditRefusal::RegionNotGenerated { region: (9, 0) }
    );
}

// -------------------------------------------------------------------------------------------------
// ---- write safety (ticket 033) -------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

use crate::edit::session::{session_stamp, WriteError, WriteSafety, WriteSession};

/// A write session on a fixture save. The fixture's temp directory has no
/// `session.lock` until this creates one, which is itself part of what's under
/// test — a world nobody has opened is a world nobody has locked.
fn write_session(save: &SaveFixture) -> WriteSession {
    WriteSession::open(&save.meta).expect("nobody has the fixture world open")
}

#[test]
fn a_committed_edit_reaches_the_files_and_leaves_nothing_dirty() {
    let save = SaveFixture::corner("commit");
    let mut cache = save.cache(4);
    let mut session = write_session(&save);

    let summary = session
        .commit(
            &corner_building(dirt()),
            &mut cache,
            &EditPolicy::default(),
        )
        .expect("a valid edit on an unlocked world");

    assert_eq!(summary.report.blocks_written, 16);
    assert_eq!(
        summary.regions_written,
        vec![(-1, -1), (-1, 0), (0, -1), (0, 0)]
    );
    assert_eq!(
        cache.dirty_regions().count(),
        0,
        "committing is what 031 and 032 stopped short of: the regions are saved"
    );

    // Read back from disk rather than from the cache — the cache would show
    // the in-memory edit whether or not it was written.
    for (region, at) in [
        ((-1, -1), IVec3::new(-1, 5, -1)),
        ((-1, 0), IVec3::new(-1, 5, 0)),
        ((0, -1), IVec3::new(0, 5, -1)),
        ((0, 0), IVec3::new(0, 5, 0)),
    ] {
        assert_eq!(
            block_name_at(&save.load(region), at),
            "minecraft:dirt",
            "region {region:?} should have been written"
        );
    }
}

#[test]
fn the_first_write_of_a_session_backs_the_region_file_up_and_the_second_does_not() {
    let save = SaveFixture::corner("backup");
    let mut cache = save.cache(4);
    let mut session = write_session(&save);

    let first = session
        .commit(
            &one_block(IVec3::new(0, 5, 0), dirt()),
            &mut cache,
            &EditPolicy::default(),
        )
        .expect("a valid edit");

    assert_eq!(first.backups.len(), 1);
    let backup = first.backups[0].clone();
    assert!(backup.starts_with(session.backup_dir()));
    assert_eq!(
        block_name_at(&load_region_file(&backup, (0, 0)), IVec3::new(0, 5, 0)),
        "minecraft:stone",
        "the backup is the world as it was before we touched it"
    );

    let second = session
        .commit(
            &one_block(IVec3::new(1, 5, 1), stairs()),
            &mut cache,
            &EditPolicy::default(),
        )
        .expect("the region was saved, so a second transaction is allowed");

    assert!(
        second.backups.is_empty(),
        "backing up again would overwrite the only copy of the pre-edit state with our own output"
    );
    assert_eq!(
        block_name_at(&load_region_file(&backup, (0, 0)), IVec3::new(0, 5, 0)),
        "minecraft:stone",
        "...which is what this asserts: the backup still holds the original"
    );
    // ...and the second edit did land, on top of the first.
    let on_disk = save.load((0, 0));
    assert_eq!(block_name_at(&on_disk, IVec3::new(0, 5, 0)), "minecraft:dirt");
    assert_eq!(
        block_name_at(&on_disk, IVec3::new(1, 5, 1)),
        "minecraft:oak_stairs"
    );
}

#[test]
fn a_refused_edit_writes_nothing_and_backs_nothing_up() {
    let save = SaveFixture::corner("commit-refused");
    let mut cache = save.cache(4);
    let mut session = write_session(&save);
    let before = save.region_bytes((0, 0));

    let mut edit = corner_building(dirt());
    edit.set(IVec3::new(1000, 5, 0), dirt());

    assert!(matches!(
        session
            .commit(&edit, &mut cache, &EditPolicy::default())
            .unwrap_err(),
        WriteError::Refused(EditRefusal::RegionNotGenerated { region: (1, 0) })
    ));

    assert!(
        !session.backup_dir().exists(),
        "a session that never wrote anything leaves nothing on disk, not even a directory"
    );
    assert_eq!(save.region_bytes((0, 0)), before);
    assert_eq!(cache.dirty_regions().count(), 0);
}

#[test]
fn a_backup_that_cannot_be_taken_rolls_the_transaction_back_instead_of_writing_unprotected() {
    let save = SaveFixture::corner("backup-fails");
    let mut cache = save.cache(4);
    // A plain file where the backup directory needs to go: `create_dir_all`
    // can't have it, and no backup means no write.
    std::fs::write(save.meta.path.join(crate::edit::BACKUP_DIR), b"not a directory")
        .expect("stage the obstruction");
    let mut session = write_session(&save);
    let before = save.region_bytes((0, 0));

    let err = session
        .commit(
            &corner_building(dirt()),
            &mut cache,
            &EditPolicy::default(),
        )
        .unwrap_err();
    assert!(
        matches!(err, WriteError::Backup { .. }),
        "expected a backup failure, got {err:?}"
    );

    // The backup pass runs to completion before the first save, so this fails
    // with the save untouched — and 032's discard makes that a full rollback.
    assert_eq!(save.region_bytes((0, 0)), before);
    assert_eq!(
        cache.dirty_regions().count(),
        0,
        "the in-memory mutations were discarded too, or the next transaction would be refused"
    );
    assert_eq!(
        block_name_at(
            cache.get_or_load((0, 0)).expect("re-read from disk"),
            IVec3::new(0, 5, 0)
        ),
        "minecraft:stone"
    );
}

#[test]
fn flushing_saves_what_a_batch_left_dirty_and_is_a_no_op_afterwards() {
    let save = SaveFixture::corner("flush");
    let mut cache = save.cache(4);
    let batching = EditPolicy {
        allow_dirty_regions: true,
        ..EditPolicy::default()
    };

    // Two transactions applied without saving in between — the case
    // `allow_dirty_regions` exists for, and the one `flush` finishes.
    apply_routed(
        &one_block(IVec3::new(0, 5, 0), dirt()),
        &mut cache,
        &batching,
    )
    .expect("a valid edit");
    apply_routed(
        &one_block(IVec3::new(-1, 5, -1), dirt()),
        &mut cache,
        &batching,
    )
    .expect("a valid edit");
    assert_eq!(cache.dirty_regions().count(), 2);

    let mut session = write_session(&save);
    let summary = session.flush(&mut cache).expect("both regions save");

    assert_eq!(summary.regions_written, vec![(-1, -1), (0, 0)]);
    assert_eq!(summary.backups.len(), 2);
    assert_eq!(cache.dirty_regions().count(), 0);
    assert_eq!(
        block_name_at(&save.load((0, 0)), IVec3::new(0, 5, 0)),
        "minecraft:dirt"
    );
    assert_eq!(
        block_name_at(&save.load((-1, -1)), IVec3::new(-1, 5, -1)),
        "minecraft:dirt"
    );

    let again = session.flush(&mut cache).expect("nothing to do");
    assert!(again.regions_written.is_empty());
    assert!(again.backups.is_empty());
}

#[test]
fn the_dry_run_names_the_files_it_would_write_and_touches_none_of_them() {
    let save = SaveFixture::corner("dry-run");
    let mut cache = save.cache(4);
    let session = write_session(&save);
    let before = save.region_bytes((0, 0));

    let plan = session
        .plan(
            &corner_building(dirt()),
            &mut cache,
            &EditPolicy::default(),
        )
        .expect("a valid edit");

    assert_eq!(plan.report.blocks_written, 16);
    assert_eq!(plan.regions.len(), 4);
    assert_eq!(plan.regions[0].coord, (-1, -1));
    assert_eq!(plan.regions[0].path, save.meta.get_region_path(-1, -1));
    assert!(
        plan.regions.iter().all(|region| !region.backed_up
            && region
                .backup
                .as_ref()
                .is_some_and(|backup| backup.starts_with(session.backup_dir()))),
        "nothing is backed up yet, and every backup would go in this session's directory"
    );
    // The text a UI shows before committing.
    let rendered = plan.to_string();
    assert!(rendered.contains("16 block(s)"), "{rendered}");
    assert!(rendered.contains("r.0.0.mca"), "{rendered}");

    // A dry run is dry: no edit, no save, no directory.
    assert_eq!(cache.dirty_regions().count(), 0);
    assert!(!session.backup_dir().exists());
    assert_eq!(save.region_bytes((0, 0)), before);
}

#[test]
fn the_backups_can_be_turned_off_for_a_caller_that_has_its_own() {
    let save = SaveFixture::corner("no-backup");
    let mut cache = save.cache(4);
    let mut session = WriteSession::open_with(
        &save.meta,
        WriteSafety {
            back_up: false,
            ..WriteSafety::default()
        },
    )
    .expect("nobody has the fixture world open");

    let summary = session
        .commit(
            &one_block(IVec3::new(0, 5, 0), dirt()),
            &mut cache,
            &EditPolicy::default(),
        )
        .expect("a valid edit");

    assert!(summary.backups.is_empty());
    assert!(!session.backup_dir().exists());
    assert_eq!(
        block_name_at(&save.load((0, 0)), IVec3::new(0, 5, 0)),
        "minecraft:dirt"
    );
}

/// Windows only, and deliberately: a POSIX record lock never conflicts with
/// its own owner, so the Unix version of this would pass without proving
/// anything unless the lock were staged from a child process. `mc_anvil`'s
/// ticket 016 test suite does exactly that for the lock itself; what's under
/// test *here* is only that this layer refuses when the lock is refused.
#[test]
#[cfg(windows)]
fn a_world_that_is_open_in_minecraft_is_refused() {
    let save = SaveFixture::corner("locked");
    let held = mc_anvil::SessionLock::acquire(&save.meta)
        .expect("the lock file is reachable")
        .expect("nobody else holds it");

    let err = WriteSession::open(&save.meta).unwrap_err();
    assert!(
        matches!(err, WriteError::WorldIsOpen { .. }),
        "expected the world to read as open, got {err:?}"
    );

    // ...and closing the world gives it back.
    drop(held);
    assert!(WriteSession::open(&save.meta).is_ok());
}

#[test]
fn the_backup_directory_is_named_after_the_moment_the_session_opened() {
    let at = |seconds: u64| {
        session_stamp(std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds))
    };

    assert_eq!(at(0), "1970-01-01T00-00-00Z");
    assert_eq!(at(1_755_264_000), "2025-08-15T13-20-00Z");
    // The two leap-year cases a hand-rolled civil date gets wrong: an ordinary
    // leap year, and the century that is one anyway.
    assert_eq!(at(1_709_209_845), "2024-02-29T12-30-45Z");
    assert_eq!(at(951_825_600), "2000-02-29T12-00-00Z");
}
