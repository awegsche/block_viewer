//! Tests for the journal (ticket 044).

use super::*;

use std::collections::BTreeMap as StdBTreeMap;

use bevy::math::IVec2;
use mc_anvil::chunkregion::ChunkRegion;
use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
use rnbt::{NbtField, NbtList, NbtValue};

use crate::edit::{apply, EditPolicy};

fn stone() -> BlockState {
    BlockState { name: "minecraft:stone".to_string(), properties: Vec::new() }
}

fn dirt() -> BlockState {
    BlockState { name: "minecraft:dirt".to_string(), properties: Vec::new() }
}

fn placed(definition: &str, origin: IVec3, footprint: IVec2) -> PlacedBuilding {
    PlacedBuilding { catalogue_id: definition.to_string(), definition_id: None, origin, rotation: Rotation::Deg0, footprint }
}

// -------------------------------------------------------------------------------------------------
// ---- the baseline ---------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn baseline_capture_lines_up_written_and_previous_by_position() {
    let mut edit = WorldEdit::new().with_data_version(4438);
    edit.set(IVec3::new(2, 5, 1), stone());
    edit.set(IVec3::new(1, 5, 1), dirt());

    let report = EditReport {
        blocks_written: 2,
        chunks: vec![(0, 0)],
        regions: vec![(0, 0)],
        // Deliberately in the same ascending order `capture_replaced` sorts
        // to, but built by hand — this test doesn't need a real region.
        replaced: Some(vec![
            (IVec3::new(1, 5, 1), BlockState::air()),
            (IVec3::new(2, 5, 1), BlockState::air()),
        ]),
    };

    let baseline = Baseline::capture(&edit, &report).expect("replaced was captured");
    assert_eq!(
        baseline.written,
        vec![(IVec3::new(1, 5, 1), dirt()), (IVec3::new(2, 5, 1), stone())]
    );
    assert_eq!(
        baseline.previous,
        vec![(IVec3::new(1, 5, 1), BlockState::air()), (IVec3::new(2, 5, 1), BlockState::air())]
    );
    assert_eq!(baseline.data_version, Some(4438));
}

#[test]
fn baseline_capture_is_none_without_a_captured_replaced_set() {
    let mut edit = WorldEdit::new();
    edit.set(IVec3::new(1, 5, 1), dirt());
    let report = EditReport { replaced: None, ..EditReport::default() };

    assert!(Baseline::capture(&edit, &report).is_none());
}

fn sample_baseline() -> Baseline {
    Baseline {
        written: vec![(IVec3::new(0, 64, 0), stone()), (IVec3::new(1, 64, 0), stone())],
        previous: vec![(IVec3::new(0, 64, 0), dirt()), (IVec3::new(1, 64, 0), dirt())],
        data_version: Some(4438),
    }
}

// -------------------------------------------------------------------------------------------------
// ---- recording and undo ----------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn recording_and_undoing_a_placement_removes_the_building_and_restores_the_previous_blocks() {
    let mut city = City::default();
    let id = city
        .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 1))
        .unwrap();
    let snapshot = city.building(id).unwrap().clone();

    let mut journal = Journal::default();
    journal.record_placement(id, snapshot, sample_baseline(), Ledger::default());
    assert_eq!(journal.len(), 1);

    let step = journal.undo_last(&mut city).expect("a placement can be undone");
    assert_eq!(step.building, id);
    assert_eq!(
        step.edit.edits().iter().map(|e| (e.at, e.state.clone())).collect::<Vec<_>>(),
        sample_baseline().previous
    );

    assert!(city.building(id).is_none(), "undo removed the building from city state");
    assert!(city.is_tile_free(IVec2::new(0, 0)));
    assert!(journal.is_empty(), "the entry is popped once undone");
}

#[test]
fn undo_on_an_empty_journal_is_an_error() {
    let mut journal = Journal::default();
    let mut city = City::default();
    assert!(matches!(journal.undo_last(&mut city), Err(UndoError::Empty)));
}

#[test]
fn recording_and_undoing_a_demolition_reinserts_the_building_under_its_original_id() {
    let mut city = City::default();
    let id = city
        .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 1))
        .unwrap();
    let snapshot = city.remove_building(id).unwrap();

    let mut journal = Journal::default();
    journal.record_demolition(id, snapshot, sample_baseline(), Ledger::default());

    let step = journal.undo_last(&mut city).expect("a demolition can be undone");
    assert_eq!(step.building, id);
    // Undoing a demolition puts the building's own blocks back — `previous`
    // on a demolition's baseline is what the building was, not the terrain.
    assert_eq!(
        step.edit.edits().iter().map(|e| (e.at, e.state.clone())).collect::<Vec<_>>(),
        sample_baseline().previous
    );

    let restored = city.building(id).expect("undo put the building back under its own id");
    assert_eq!(restored.catalogue_id, "house01");
    assert!(journal.is_empty());
}

#[test]
fn undoing_a_demolition_is_refused_when_the_tile_is_occupied_now() {
    let mut city = City::default();
    let id = city
        .place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(2, 1))
        .unwrap();
    let snapshot = city.remove_building(id).unwrap();

    let mut journal = Journal::default();
    journal.record_demolition(id, snapshot, sample_baseline(), Ledger::default());

    // Something else claims the freed tile before the undo runs.
    city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::new(1, 1))
        .unwrap();

    let err = journal.undo_last(&mut city).unwrap_err();
    assert!(matches!(err, UndoError::Occupied(_)));
    // Neither side moved: the failed undo can be retried later.
    assert_eq!(journal.len(), 1);
    assert!(city.building(id).is_none());
}

#[test]
fn placement_baseline_returns_the_most_recent_record_for_a_building() {
    let mut journal = Journal::default();
    let id = BuildingId::from_u64(0);
    let first = Baseline { written: vec![(IVec3::ZERO, stone())], previous: vec![(IVec3::ZERO, dirt())], data_version: None };
    let second = sample_baseline();

    journal.record_placement(id, placed("house01", IVec3::ZERO, IVec2::ONE), first, Ledger::default());
    assert_eq!(journal.placement_baseline(id).unwrap().written[0].1, stone());

    // A later Placed record for the same id (a fresh placement after a
    // demolition, say) supersedes the earlier one.
    journal.record_placement(id, placed("house01", IVec3::ZERO, IVec2::ONE), second.clone(), Ledger::default());
    assert_eq!(journal.placement_baseline(id).unwrap().written, second.written);

    assert!(journal.placement_baseline(BuildingId::from_u64(99)).is_none());
}

// -------------------------------------------------------------------------------------------------
// ---- a minimal region fixture for reconciliation ---------------------------------------------------
// -------------------------------------------------------------------------------------------------

const FIXTURE_DATA_VERSION: i32 = 4438;

/// One finished chunk with a single all-stone section at `Y = 0` — just
/// enough for [`reconcile`] to have something real to read.
fn full_chunk(x: i32, z: i32) -> NbtField {
    let palette = NbtList::Compound(vec![NbtField::new_compound(
        "",
        vec![NbtField::new_string("Name", "minecraft:stone")],
    )]);
    let section = NbtField::new_compound(
        "",
        vec![
            NbtField { name: "Y".to_string(), value: NbtValue::Byte(0) },
            NbtField::new_compound("block_states", vec![NbtField::new_list("palette", palette)]),
        ],
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
            NbtField { name: "isLightOn".to_string(), value: NbtValue::Byte(1) },
        ],
    )
}

/// A synthetic region file in a temp directory, removed on drop — the same
/// technique `edit::tests::RegionFixture` uses, trimmed to the one chunk
/// reconciliation's tests need.
struct RegionFixture {
    dir: std::path::PathBuf,
    path: std::path::PathBuf,
}

impl RegionFixture {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("block_viewer-journal-{label}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
        payloads[0] = Some(ChunkPayload::Nbt(full_chunk(0, 0)));
        let path = dir.join("r.0.0.mca");
        Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

        Self { dir, path }
    }

    fn load(&self) -> ChunkRegion {
        let mut region: ChunkRegion = Region::new(0, 0, &self.path).into();
        region.load_chunks().expect("load the fixture region");
        region
    }
}

impl Drop for RegionFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A [`RegionSource`] over regions already in memory — only region `(0, 0)`
/// (the fixture's own) is ever present; every other coordinate reads as
/// [`RegionUnavailable::NotGenerated`], which is exactly what reconciliation
/// needs to exercise the region-level `unknown` path.
struct FixtureRegions {
    regions: StdBTreeMap<(i32, i32), ChunkRegion>,
}

impl RegionSource for FixtureRegions {
    fn region_mut(&mut self, coord: (i32, i32)) -> Result<&mut ChunkRegion, crate::edit::RegionUnavailable> {
        self.regions.get_mut(&coord).ok_or(crate::edit::RegionUnavailable::NotGenerated)
    }

    fn discard(&mut self, coord: (i32, i32)) {
        self.regions.remove(&coord);
    }

    fn dirty_regions(&self) -> Vec<(i32, i32)> {
        Vec::new()
    }
}

/// A `City` with one building placed at `origin` and a matching journal
/// entry whose baseline says `at` should hold `expected`.
fn city_and_journal_with_one_building(id_seed: IVec3, at: IVec3, expected: BlockState) -> (City, Journal) {
    let mut city = City::default();
    let id = city.place_building("house01", None, id_seed, Rotation::Deg0, IVec2::ONE).unwrap();
    let building = city.building(id).unwrap().clone();

    let mut journal = Journal::default();
    journal.record_placement(
        id,
        building,
        Baseline { written: vec![(at, expected)], previous: vec![(at, BlockState::air())], data_version: None },
        Ledger::default(),
    );
    (city, journal)
}

// -------------------------------------------------------------------------------------------------
// ---- reconciliation ---------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

#[test]
fn reconcile_reports_no_mismatch_when_the_world_matches_the_baseline() {
    let fixture = RegionFixture::new("clean");
    let mut source = FixtureRegions { regions: StdBTreeMap::from([((0, 0), fixture.load())]) };

    let (city, journal) = city_and_journal_with_one_building(IVec3::new(50, 64, 50), IVec3::new(1, 0, 1), stone());

    let report = reconcile(&journal, &city, &mut source);
    assert!(report.mismatched.is_empty());
    assert!(report.unknown.is_empty());
}

#[test]
fn reconcile_reports_a_mismatch_when_a_block_was_changed_in_the_world() {
    let fixture = RegionFixture::new("changed");
    let mut region = fixture.load();
    apply(&{
        let mut e = WorldEdit::new();
        e.set(IVec3::new(1, 0, 1), dirt());
        e
    }, &mut region, &EditPolicy::default())
    .expect("stage a player edit");

    let mut source = FixtureRegions { regions: StdBTreeMap::from([((0, 0), region)]) };
    let (city, journal) = city_and_journal_with_one_building(IVec3::new(50, 64, 50), IVec3::new(1, 0, 1), stone());

    let report = reconcile(&journal, &city, &mut source);
    assert_eq!(report.mismatched, vec![Mismatch { at: IVec3::new(1, 0, 1), expected: stone(), actual: dirt() }]);
    assert!(report.unknown.is_empty());
}

#[test]
fn reconcile_reports_unknown_for_a_region_the_source_does_not_have() {
    let mut source = FixtureRegions { regions: StdBTreeMap::new() };
    // Region (0, 0) is never inserted into the source at all.
    let (city, journal) = city_and_journal_with_one_building(IVec3::new(50, 64, 50), IVec3::new(1, 0, 1), stone());

    let report = reconcile(&journal, &city, &mut source);
    assert!(report.mismatched.is_empty());
    assert_eq!(report.unknown.len(), 1);
    assert_eq!(report.unknown[0].0, IVec3::new(1, 0, 1));
}

#[test]
fn reconcile_reports_unknown_for_an_ungenerated_chunk_inside_a_present_region() {
    let fixture = RegionFixture::new("half");
    let mut source = FixtureRegions { regions: StdBTreeMap::from([((0, 0), fixture.load())]) };

    // Chunk (5, 0) is one of the region's ungenerated slots.
    let (city, journal) =
        city_and_journal_with_one_building(IVec3::new(50, 64, 50), IVec3::new(5 * 16, 0, 0), stone());

    let report = reconcile(&journal, &city, &mut source);
    assert!(report.mismatched.is_empty());
    assert_eq!(report.unknown.len(), 1);
}

#[test]
fn a_building_with_no_placement_record_is_skipped_rather_than_reported() {
    // Simulates a save from before this ticket: `City` has a building, the
    // journal has nothing for it. Reconciliation can't say anything about a
    // building it has no baseline for, so it says nothing rather than
    // treating the whole thing as unknown or mismatched.
    let mut city = City::default();
    city.place_building("house01", None, IVec3::new(0, 64, 0), Rotation::Deg0, IVec2::ONE).unwrap();
    let journal = Journal::default();
    let mut source = FixtureRegions { regions: StdBTreeMap::new() };

    let report = reconcile(&journal, &city, &mut source);
    assert!(report.mismatched.is_empty());
    assert!(report.unknown.is_empty());
}

#[test]
fn repair_edit_writes_every_mismatch_back_and_is_none_when_there_is_nothing_to_repair() {
    assert!(repair_edit(&ReconcileReport::default()).is_none());

    let report = ReconcileReport {
        mismatched: vec![Mismatch { at: IVec3::new(1, 0, 1), expected: stone(), actual: dirt() }],
        unknown: Vec::new(),
    };
    let edit = repair_edit(&report).expect("one mismatch to repair");
    assert_eq!(edit.len(), 1);
    assert_eq!(edit.edits()[0].at, IVec3::new(1, 0, 1));
    assert_eq!(edit.edits()[0].state, stone());
}

// -------------------------------------------------------------------------------------------------
// ---- persistence -----------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

fn temp_dir(name: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join(format!("block_viewer_test_journal_{name}_{}_{unique}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("should create temp dir");
    dir
}

#[test]
fn a_missing_journal_file_loads_as_empty() {
    let dir = temp_dir("missing");
    let journal = load_journal(&dir).expect("a missing journal file is not an error");
    assert!(journal.is_empty());
}

#[test]
fn an_empty_journal_round_trips() {
    let dir = temp_dir("empty");
    save_journal(&Journal::default(), &dir).unwrap();
    assert!(load_journal(&dir).unwrap().is_empty());
}

#[test]
fn placements_and_demolitions_round_trip_exactly() {
    let dir = temp_dir("full");
    let mut journal = Journal::default();
    let placed_id = BuildingId::from_u64(0);
    let demolished_id = BuildingId::from_u64(1);

    journal.record_placement(
        placed_id,
        placed("house01", IVec3::new(10, 64, 20), IVec2::new(3, 2)),
        sample_baseline(),
        Ledger::default(),
    );
    journal.record_demolition(
        demolished_id,
        placed("house01", IVec3::new(0, 70, 0), IVec2::new(3, 5)),
        Baseline {
            written: vec![(IVec3::new(0, 70, 0), dirt())],
            previous: vec![(IVec3::new(0, 70, 0), stone())],
            data_version: Some(3953),
        },
        Ledger::default(),
    );

    save_journal(&journal, &dir).unwrap();
    let loaded = load_journal(&dir).unwrap();

    assert_eq!(loaded.len(), 2);
    match &loaded.entries()[0] {
        JournalEntry::Placed { building, placement, baseline, .. } => {
            assert_eq!(*building, placed_id);
            assert_eq!(placement.origin, IVec3::new(10, 64, 20));
            assert_eq!(*baseline, sample_baseline());
        }
        other => panic!("expected a Placed entry, got {other:?}"),
    }
    match &loaded.entries()[1] {
        JournalEntry::Demolished { building, placement, baseline, .. } => {
            assert_eq!(*building, demolished_id);
            assert_eq!(placement.footprint, IVec2::new(3, 5));
            assert_eq!(baseline.data_version, Some(3953));
        }
        other => panic!("expected a Demolished entry, got {other:?}"),
    }
}

#[test]
fn a_version_mismatch_is_refused() {
    let dir = temp_dir("version");
    fs::create_dir_all(dir.join("citybuilder")).unwrap();
    fs::write(dir.join("citybuilder/journal.ron"), "(version: 999, entries: [])").unwrap();

    assert!(matches!(load_journal(&dir).unwrap_err(), JournalError::UnsupportedVersion(999)));
}

// -------------------------------------------------------------------------------------------------
// ---- the ledger (ticket 073) ------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

fn parcel(items: &[(&str, u64)]) -> super::super::inventory::Parcel {
    let mut parcel = super::super::inventory::Parcel::default();
    for &(item, count) in items {
        parcel.add(item, count);
    }
    parcel
}

#[test]
fn a_ledger_round_trips_through_disk() {
    let dir = temp_dir("ledger");
    let mut journal = Journal::default();
    let ledger = Ledger {
        credited: parcel(&[("minecraft:dirt", 12), ("minecraft:cobblestone", 3)]),
        debited: parcel(&[("minecraft:oak_planks", 40)]),
    };
    journal.record_placement(BuildingId::from_u64(0), placed("house01", IVec3::ZERO, IVec2::ONE), sample_baseline(), ledger.clone());

    save_journal(&journal, &dir).unwrap();
    let loaded = load_journal(&dir).unwrap();

    assert_eq!(*loaded.entries()[0].ledger(), ledger);
}

/// The reason [`MIN_READABLE_VERSION`] is a band rather than an equality
/// check: a version-1 journal's as-built baselines are unrecoverable if this
/// file is refused, and the ledger those entries lack is provably empty —
/// they were written before anything could be charged.
#[test]
fn a_version_1_journal_still_loads_with_empty_ledgers() {
    let dir = temp_dir("v1");
    fs::create_dir_all(dir.join("citybuilder")).unwrap();
    fs::write(
        dir.join("citybuilder/journal.ron"),
        r#"(
            version: 1,
            entries: [
                Placed(
                    building: 7,
                    placement: (catalogue_id: "house01", origin: (1, 64, 2), rotation: Deg0, footprint: (3, 3)),
                    baseline: (
                        written: [((1, 64, 2), (name: "minecraft:stone", properties: []))],
                        previous: [((1, 64, 2), (name: "minecraft:dirt", properties: []))],
                        data_version: Some(3953),
                    ),
                ),
            ],
        )"#,
    )
    .unwrap();

    let loaded = load_journal(&dir).expect("a version-1 journal is still readable");

    assert_eq!(loaded.len(), 1, "the entry — and its baseline — survives");
    let entry = &loaded.entries()[0];
    assert_eq!(entry.building(), BuildingId::from_u64(7));
    assert_eq!(entry.baseline().previous[0].1, dirt());
    assert!(entry.ledger().is_empty(), "an entry from before the economy moved no materials");
}

#[test]
fn undo_last_hands_back_the_entrys_own_ledger() {
    // Undo settles what was settled — not a fresh reading of a definition
    // that may have been edited since.
    let mut city = City::default();
    let id = city.place_building("house01", None, IVec3::ZERO, Rotation::Deg0, IVec2::ONE).unwrap();
    let building = city.building(id).unwrap().clone();
    let ledger = Ledger { credited: parcel(&[("minecraft:dirt", 9)]), debited: parcel(&[("minecraft:oak_planks", 40)]) };

    let mut journal = Journal::default();
    journal.record_placement(id, building, sample_baseline(), ledger.clone());

    let step = journal.undo_last(&mut city).expect("undoes");

    assert_eq!(step.ledger, ledger);
}

#[test]
fn garbage_ron_is_a_parse_error_not_a_panic() {
    let dir = temp_dir("garbage");
    fs::create_dir_all(dir.join("citybuilder")).unwrap();
    fs::write(dir.join("citybuilder/journal.ron"), b"not valid ron {{{").unwrap();

    assert!(matches!(load_journal(&dir).unwrap_err(), JournalError::Parse(_)));
}
