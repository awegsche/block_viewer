//! Marker-block geometry as a [`WorldEdit`] (ticket 134,
//! `MODEL_EXPORTER_ROADMAP.md` "Coordinates and markers") — the first
//! `model-exporter` command that writes to the models world. Every write
//! goes through [`crate::ranvil_cli::edit::run_write`]; this module only
//! builds the [`WorldEdit`] geometry hands it.
//!
//! [`ring_y`]/[`corner`] are the same arithmetic ticket 132's `show`
//! (`super::list`) reports, factored out here so `show`'s printed geometry
//! and the blocks [`marker_edit`] actually places agree by construction
//! rather than each rederiving `world.ground_y`/`Footprint::expanded(1)`.

use bevy::math::{IVec2, IVec3};
use serde_json::{json, Value};

use crate::edit::WorldEdit;
use crate::ranvil_cli::edit::{outcome_json_fields, outcome_summary, run_write, WriteOutcome};
use crate::ranvil_cli::error::CliError;
use crate::ranvil_cli::format::Render;
use crate::ranvil_cli::save::resolve_save_from;

use super::cli::{Cli, MarkArgs};
use super::registry::{load_registry, ModelSlot, ModelWorld};

/// The marker ring's `y` — flush with `world.ground_y`, the layer the ring
/// itself replaces.
pub fn ring_y(world: &ModelWorld) -> i32 {
    world.ground_y
}

/// One of the marker ring's four corners (where a pillar stands): the
/// ring's bounding rectangle — `slot.footprint().expanded(1)` — picked by
/// which end of each axis to take. [`Footprint`](super::registry::Footprint)'s
/// `y` is world `z`, not a typo.
pub fn corner(slot: &ModelSlot, max_x: bool, max_z: bool) -> IVec2 {
    let ring = slot.footprint().expanded(1);
    IVec2::new(
        if max_x { ring.max.x } else { ring.min.x },
        if max_z { ring.max.y } else { ring.min.y },
    )
}

/// Every position [`marker_edit`] sets, in `world.marker` (ticket 134's
/// geometry, exactly): a ring at `y = ring_y(world)` one block outside the
/// slot's footprint on every side, plus corner pillars on the four ring
/// corners from `ring_y(world) + 1` up to the slot's own `max.y` — skipped
/// entirely when the box never rises above the ring (`max.y <= ring_y`).
/// Never a position inside the slot's own box.
///
/// [`super::remove`] (ticket 137) reuses this list to know exactly which
/// positions a `--clear` should put ground/air back at.
pub fn marker_positions(slot: &ModelSlot, world: &ModelWorld) -> Vec<IVec3> {
    let min = slot.min();
    let max = slot.max();
    let y = ring_y(world);
    let mut positions = Vec::new();

    for z in (min.z - 1)..=(max.z + 1) {
        for x in (min.x - 1)..=(max.x + 1) {
            let inside_footprint = (min.x..=max.x).contains(&x) && (min.z..=max.z).contains(&z);
            if inside_footprint {
                continue;
            }
            positions.push(IVec3::new(x, y, z));
        }
    }

    if max.y > y {
        for &max_x in &[false, true] {
            for &max_z in &[false, true] {
                let c = corner(slot, max_x, max_z);
                for py in (y + 1)..=max.y {
                    positions.push(IVec3::new(c.x, py, c.y));
                }
            }
        }
    }

    positions
}

/// [`marker_positions`] as a [`WorldEdit`], every position set to
/// `world.marker`.
pub fn marker_edit(slot: &ModelSlot, world: &ModelWorld) -> WorldEdit {
    let mut edit = WorldEdit::new();
    for pos in marker_positions(slot, world) {
        edit.set(pos, world.marker.clone());
    }
    edit
}

/// `mark`'s result: the outcome of re-placing one slot's markers.
#[derive(Debug)]
pub struct MarkResult {
    pub name: String,
    pub save_name: String,
    pub outcome: WriteOutcome,
}

/// Runs `mark <name>`: re-places an already-registered slot's markers —
/// for a `.ron` edited by hand (a bigger `size`), after `remove --clear`
/// (ticket 137), or a world reset. Unknown `name` → [`CliError::Usage`].
pub fn mark(cli: &Cli, args: &MarkArgs) -> Result<MarkResult, CliError> {
    let registry = load_registry(&cli.models_dir)?;
    let slot = registry
        .slots
        .iter()
        .find(|slot| slot.name == args.name)
        .ok_or_else(|| {
            CliError::Usage(format!(
                "no model named {:?} registered under {}",
                args.name,
                registry.dir.display()
            ))
        })?
        .clone();

    let save = cli.save.as_deref().or(Some(registry.world.save.as_str()));
    let meta = resolve_save_from(save, cli.instance.as_deref())?;

    let world = registry.world.clone();
    let outcome = run_write(&meta, args.dry_run, args.force, move |_cache| {
        Ok(marker_edit(&slot, &world))
    })?;

    Ok(MarkResult {
        name: args.name.clone(),
        save_name: meta.name,
        outcome,
    })
}

impl Render for MarkResult {
    fn render_text(&self) -> String {
        let prefix = format!("mark {} in {}", self.name, self.save_name);
        let mut lines = vec![outcome_summary(prefix, &self.outcome)];
        if !self.outcome.dry_run {
            lines.push(format!("  backup: {}", self.outcome.backup_dir.display()));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("name".to_string(), json!(self.name));
        map.insert("save".to_string(), json!(self.save_name));
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::*;
    use crate::blueprint::BlockState;
    use crate::model_exporter::registry::{Area, Point};

    fn test_world(ground_y: i32) -> ModelWorld {
        ModelWorld {
            save: "models".to_string(),
            ground_y,
            area: Area { min: Point { x: 0, z: 0 }, max: Point { x: 63, z: 63 } },
            gap: 3,
            grid: 8,
            marker: BlockState::from_str("minecraft:orange_terracotta").expect("valid block state"),
            blueprints_dir: PathBuf::from("assets/city/blueprints"),
        }
    }

    fn test_slot(origin: (i32, i32, i32), size: (i32, i32, i32)) -> ModelSlot {
        ModelSlot {
            name: "test".to_string(),
            origin: IVec3::new(origin.0, origin.1, origin.2),
            size: IVec3::new(size.0, size.1, size.2),
            out: None,
        }
    }

    /// The ticket's headline "Done when": ring count, pillar count, none
    /// inside the box, pillar tops at `max.y`.
    #[test]
    fn marker_positions_ring_and_pillar_counts_match_the_formula() {
        let world = test_world(5);
        // origin (2,3,2) size (3,4,3): min (2,3,2), max (4,6,4). ground_y=5
        // cuts through the box's vertical span; pillars run from
        // ground_y+1=6 up to max.y=6 — one layer, on each of the four
        // corners.
        let slot = test_slot((2, 3, 2), (3, 4, 3));

        let positions = marker_positions(&slot, &world);
        // ring: 2*(sx+2) + 2*sz = 2*(3+2) + 2*3 = 16; pillars: 4*(max.y -
        // ground_y) = 4*(6-5) = 4.
        assert_eq!(positions.len(), 20);

        let (min, max) = (slot.min(), slot.max());
        assert!(
            positions.iter().all(|p| !((min.x..=max.x).contains(&p.x)
                && (min.z..=max.z).contains(&p.z)
                && (min.y..=max.y).contains(&p.y))),
            "no marker position may sit inside the slot's own box"
        );

        assert_eq!(positions.iter().filter(|p| p.y == 5).count(), 16);

        let pillar_tops: Vec<IVec3> = positions.iter().copied().filter(|p| p.y == 6).collect();
        assert_eq!(pillar_tops.len(), 4, "pillar tops must be at max.y");
        for (x, z) in [(1, 1), (1, 5), (5, 1), (5, 5)] {
            assert!(pillar_tops.contains(&IVec3::new(x, 6, z)), "missing pillar top at ({x},{z})");
        }
    }

    /// A slot with `below = 0` and `size.y = 1` (`origin.y == ground_y`,
    /// `max.y == ground_y`) has a ring and no pillars.
    #[test]
    fn a_flat_slot_flush_with_the_ground_has_a_ring_and_no_pillars() {
        let world = test_world(5);
        let slot = test_slot((2, 5, 2), (3, 1, 3));

        let positions = marker_positions(&slot, &world);
        assert_eq!(positions.len(), 16);
        assert!(positions.iter().all(|p| p.y == 5));
    }
}

#[cfg(test)]
mod mark_tests {
    use std::path::{Path, PathBuf};

    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use mc_anvil::SaveMeta;
    use rnbt::{NbtField, NbtList, NbtValue};

    use super::*;
    use crate::ranvil_cli::format::OutputFormat;
    use crate::region_cache::RegionCache;
    use super::super::cli::{Cli, Command, ListArgs};

    /// The `DataVersion` the fixture's one chunk claims — a 1.21 release,
    /// same as `ranvil_cli::edit`'s own write-path fixture.
    const FIXTURE_DATA_VERSION: i32 = 4438;

    /// A finished chunk: one all-stone section at `Y = 0` (covers block `y`
    /// 0..15), `Status = minecraft:full` — the same shape
    /// `ranvil_cli::edit::tests::full_chunk` builds, copied rather than
    /// shared since it's private to that module.
    fn full_chunk(data_version: i32) -> NbtField {
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
                NbtField::new_i32("xPos", 0),
                NbtField::new_i32("zPos", 0),
                NbtField::new_i32("yPos", -4),
                NbtField::new_i32("DataVersion", data_version),
                NbtField::new_string("Status", "minecraft:full"),
                NbtField { name: "isLightOn".to_string(), value: NbtValue::Byte(1) },
            ],
        )
    }

    /// A single-region, single-chunk fixture save in a temp directory,
    /// removed on drop.
    struct Fixture {
        dir: PathBuf,
        meta: SaveMeta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("block_viewer-model-exporter-markers-{label}-{nanos}"));
            let region_dir = dir.join("region");
            std::fs::create_dir_all(&region_dir).expect("create temp dir");

            let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
            payloads[0] = Some(ChunkPayload::Nbt(full_chunk(FIXTURE_DATA_VERSION)));
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

            let meta = SaveMeta {
                name: "mark-fixture".to_string(),
                path: dir.clone(),
                region_dir,
                regions: vec![(0, 0)],
            };
            Self { dir, meta }
        }

        fn region_path(&self) -> PathBuf {
            self.meta.get_region_path(0, 0)
        }

        fn bytes(&self) -> Vec<u8> {
            std::fs::read(self.region_path()).expect("read the fixture region")
        }

        fn mtime(&self) -> std::time::SystemTime {
            std::fs::metadata(self.region_path())
                .expect("stat the fixture region")
                .modified()
                .expect("mtime")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn block_name_at(meta: &SaveMeta, at: IVec3) -> String {
        let address = crate::edit::address_of(at);
        let mut cache = RegionCache::new(meta.clone(), 1);
        cache
            .get_or_load(address.region)
            .expect("resident")
            .get_block(address.local_x, address.y, address.local_z)
            .expect("a populated chunk")
            .get_string("Name")
            .expect("a palette entry")
            .clone()
    }

    fn temp_models_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_model_exporter_markers_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    fn write_world(dir: &Path, save_dir: &Path, ground_y: i32) {
        std::fs::write(
            dir.join("world.ron"),
            format!(
                r#"ModelWorld(
    save: "{}",
    ground_y: {ground_y},
    area: (min: (x: 0, z: 0), max: (x: 15, z: 15)),
    gap: 1,
    grid: 8,
    marker: "minecraft:orange_terracotta",
    blueprints_dir: "assets/city/blueprints",
)"#,
                save_dir.display().to_string().replace('\\', "/")
            ),
        )
        .expect("write world.ron");
    }

    fn write_slot(dir: &Path, name: &str, origin: (i32, i32, i32), size: (i32, i32, i32)) {
        std::fs::write(
            dir.join(format!("{name}.ron")),
            format!(
                "ModelSlot(name: \"{name}\", origin: ({}, {}, {}), size: ({}, {}, {}), out: None)",
                origin.0, origin.1, origin.2, size.0, size.1, size.2
            ),
        )
        .expect("write slot file");
    }

    fn cli(dir: &Path) -> Cli {
        Cli {
            models_dir: dir.to_path_buf(),
            save: None,
            instance: None,
            format: OutputFormat::Text,
            command: Command::List(ListArgs {}),
        }
    }

    fn mark_args(name: &str, dry_run: bool, force: bool) -> MarkArgs {
        MarkArgs { name: name.to_string(), dry_run, force }
    }

    /// The ticket's headline "Done when": `mark` against the single-chunk
    /// fixture save writes the expected block count, and leaves the
    /// interior (and everything outside the ring/pillars) untouched.
    #[test]
    fn mark_writes_the_ring_and_pillars_and_leaves_the_interior_alone() {
        let fixture = Fixture::new("write");
        let models_dir = temp_models_dir("write");
        write_world(&models_dir, &fixture.dir, 5);
        write_slot(&models_dir, "test", (2, 3, 2), (3, 4, 3));

        let result =
            mark(&cli(&models_dir), &mark_args("test", false, false)).expect("a valid mark");

        assert!(!result.outcome.dry_run);
        assert_eq!(result.outcome.report.blocks_written, 20);
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(1, 5, 1)), "minecraft:orange_terracotta");
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(1, 6, 1)), "minecraft:orange_terracotta");
        // Inside the slot's own box: untouched.
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(3, 5, 3)), "minecraft:stone");
    }

    /// `--dry-run` leaves the region file byte-identical.
    #[test]
    fn mark_dry_run_leaves_the_region_file_byte_identical() {
        let fixture = Fixture::new("dry-run");
        let models_dir = temp_models_dir("dry-run");
        write_world(&models_dir, &fixture.dir, 5);
        write_slot(&models_dir, "test", (2, 3, 2), (3, 4, 3));
        let (before_bytes, before_mtime) = (fixture.bytes(), fixture.mtime());

        let result =
            mark(&cli(&models_dir), &mark_args("test", true, false)).expect("a valid plan");

        assert!(result.outcome.dry_run);
        assert_eq!(result.outcome.report.blocks_written, 20);
        assert_eq!(fixture.bytes(), before_bytes);
        assert_eq!(fixture.mtime(), before_mtime);
    }

    #[test]
    fn mark_of_an_unknown_name_is_a_usage_error() {
        let fixture = Fixture::new("unknown");
        let models_dir = temp_models_dir("unknown");
        write_world(&models_dir, &fixture.dir, 5);

        let err = mark(&cli(&models_dir), &mark_args("nope", false, false)).unwrap_err();
        assert!(matches!(err, CliError::Usage(_)));
    }
}
