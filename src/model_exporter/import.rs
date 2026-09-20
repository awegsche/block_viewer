//! `model-exporter import <name> [--from <file.nbt>]` (ticket 136,
//! `MODEL_EXPORTER_ROADMAP.md` "Bridging both pipelines") — the reverse
//! bridge: an agent-built `.nbt` (or an older export) goes *into* the models
//! world so a human can walk through it and fix it in game, then `export`
//! brings it back.
//!
//! Two shapes, chosen by whether `name` is already registered:
//!
//! - **Registered.** `--from` defaults to the slot's own
//!   [`ModelSlot::out_path`] (re-import the last export). The file's `size`
//!   must equal the slot's own `size` — the box is the contract, and
//!   silently importing a bigger file would spill over the ring — checked
//!   before anything is written. Only the blocks are touched; the `.ron` and
//!   markers are left exactly as they were.
//! - **Unregistered.** `--from` is required. [`allocate`] finds a slot sized
//!   from the file (`--below` picks the origin, same default and meaning as
//!   [`super::new::new`]'s own flag), [`save_slot`] registers it, and the
//!   markers ([`super::markers::marker_positions`]) plus the blocks
//!   ([`blueprint_edit`]) go into **one** [`run_write`] call, so a refused
//!   write can never leave a marked-but-empty slot, or a `.ron` whose
//!   markers never landed. If that write is refused, the `.ron` [`save_slot`]
//!   just wrote is removed again — nothing this call did survives it.
//!
//! Either way the block write is exactly
//! [`crate::ranvil_cli::structure::import`]'s own: [`blueprint_edit`] — every
//! position, air included — offset by the slot's `origin`.

use std::path::PathBuf;

use bevy::math::IVec3;
use mc_anvil::SaveMeta;
use serde_json::{json, Value};

use crate::blueprint::{read_structure_file, Blueprint};
use crate::ranvil_cli::edit::{outcome_json_fields, outcome_summary, run_write, WriteOutcome};
use crate::ranvil_cli::error::CliError;
use crate::ranvil_cli::format::Render;
use crate::ranvil_cli::save::resolve_save_from;
use crate::ranvil_cli::structure::blueprint_edit;

use super::allocate::allocate;
use super::cli::{Cli, ImportArgs};
use super::list::{box_text, tp_position, tp_text};
use super::markers::marker_positions;
use super::new::{chunk_generated_predicate, is_valid_name};
use super::registry::{load_registry, save_slot, ModelSlot, ModelWorld, Registry};

/// Whether [`import`] wrote into a slot that already existed, or allocated
/// (and registered) a fresh one this call.
#[derive(Debug)]
pub enum ImportSlot {
    /// `name` was already registered; only the blocks were (or would be)
    /// written.
    Existing,
    /// A slot was allocated and registered this call — the same fields
    /// [`super::new::NewResult`] reports, so `import`'s `text` output can
    /// print the identical "origin/box/tp" block for it.
    Allocated { origin: IVec3, min: IVec3, max: IVec3, tp: IVec3, file: PathBuf },
}

/// `import`'s result.
#[derive(Debug)]
pub struct ImportResult {
    pub name: String,
    pub save_name: String,
    pub from: PathBuf,
    pub size: IVec3,
    pub slot: ImportSlot,
    pub outcome: WriteOutcome,
}

/// [`read_structure_file`], mapping any failure — missing file, non-gzip,
/// truncated NBT — to [`CliError::Usage`]: `import` refuses before touching
/// the registry or the world rather than reporting a read failure mid-write.
fn read_from(path: &PathBuf) -> Result<Blueprint, CliError> {
    read_structure_file(path).map_err(|err| CliError::Usage(format!("{}: {err}", path.display())))
}

/// Runs `import <name> [--from <file.nbt>]`. See the module docs for the
/// registered-vs-unregistered split.
pub fn import(cli: &Cli, args: &ImportArgs) -> Result<ImportResult, CliError> {
    let registry = load_registry(&cli.models_dir)?;

    let save = cli.save.as_deref().or(Some(registry.world.save.as_str()));
    let meta = resolve_save_from(save, cli.instance.as_deref())?;

    match registry.slots.iter().find(|slot| slot.name == args.name).cloned() {
        Some(slot) => import_into_existing(args, &meta, slot, &registry.world),
        None => import_new(args, &meta, &registry),
    }
}

/// The registered-name shape: [`ModelSlot::size`] gates the file, then a
/// single-slot [`blueprint_edit`] through [`run_write`]. Neither the `.ron`
/// nor the markers are touched.
fn import_into_existing(
    args: &ImportArgs,
    meta: &SaveMeta,
    slot: ModelSlot,
    world: &ModelWorld,
) -> Result<ImportResult, CliError> {
    let from = args.from.clone().unwrap_or_else(|| slot.out_path(world));
    let blueprint = read_from(&from)?;

    if blueprint.size != slot.size {
        return Err(CliError::Usage(format!(
            "{} is {}x{}x{} but {:?}'s slot is {}x{}x{} — the box is the contract: edit the \
             .ron's size and run `model-exporter mark {:?}` if it should grow, then retry, or \
             import a file that matches the slot as it stands",
            from.display(),
            blueprint.size.x,
            blueprint.size.y,
            blueprint.size.z,
            slot.name,
            slot.size.x,
            slot.size.y,
            slot.size.z,
            slot.name,
        )));
    }

    let size = slot.size;
    let origin = slot.origin;
    let outcome = run_write(meta, args.dry_run, args.force, move |_cache| {
        Ok(blueprint_edit(&blueprint, origin))
    })?;

    Ok(ImportResult {
        name: slot.name,
        save_name: meta.name.clone(),
        from,
        size,
        slot: ImportSlot::Existing,
        outcome,
    })
}

/// The unregistered-name shape: allocate, register, then one [`run_write`]
/// carrying both the markers and the blocks. On a refused write, the `.ron`
/// [`save_slot`] wrote is removed again so no half-registered slot survives.
fn import_new(
    args: &ImportArgs,
    meta: &SaveMeta,
    registry: &Registry,
) -> Result<ImportResult, CliError> {
    if !is_valid_name(&args.name) {
        return Err(CliError::Usage(format!(
            "{:?} is not a valid model name — must match [a-z0-9_]+",
            args.name
        )));
    }

    let from = args.from.clone().ok_or_else(|| {
        CliError::Usage(format!(
            "{:?} is not registered under {} — pass --from <file.nbt> to allocate a slot for it",
            args.name,
            registry.dir.display(),
        ))
    })?;
    let blueprint = read_from(&from)?;
    let size = blueprint.size;

    let chunk_generated = chunk_generated_predicate(meta);
    let origin = allocate(&registry.world, &registry.slots, size, args.below, chunk_generated)?;

    let slot = ModelSlot { name: args.name.clone(), origin, size, out: None };
    let file = registry.dir.join(format!("{}.ron", args.name));

    if !args.dry_run {
        save_slot(&registry.dir, &slot)
            .map_err(|e| CliError::Data(format!("could not write {}: {e}", file.display())))?;
    }

    let world = registry.world.clone();
    let slot_for_edit = slot.clone();
    let write_result = run_write(meta, args.dry_run, args.force, move |_cache| {
        let mut edit = blueprint_edit(&blueprint, slot_for_edit.origin);
        for pos in marker_positions(&slot_for_edit, &world) {
            edit.set(pos, world.marker.clone());
        }
        Ok(edit)
    });

    let outcome = match write_result {
        Ok(outcome) => outcome,
        Err(err) => {
            if !args.dry_run {
                // Nothing else this call did survives a refused write — the
                // `.ron` above was written this call, so it comes back off
                // too, per the ticket ("a refused write leaves neither a
                // marked slot without blocks nor a `.ron` without markers").
                let _ = std::fs::remove_file(&file);
            }
            return Err(err);
        }
    };

    Ok(ImportResult {
        name: slot.name.clone(),
        save_name: meta.name.clone(),
        from,
        size,
        slot: ImportSlot::Allocated {
            origin: slot.origin,
            min: slot.min(),
            max: slot.max(),
            tp: tp_position(&slot, &registry.world),
            file,
        },
        outcome,
    })
}

impl Render for ImportResult {
    fn render_text(&self) -> String {
        let mut lines = Vec::new();
        if let ImportSlot::Allocated { origin, min, max, tp, file } = &self.slot {
            lines.push(format!(
                "{}: origin ({},{},{}) size {}x{}x{}  box {}",
                self.name,
                origin.x,
                origin.y,
                origin.z,
                self.size.x,
                self.size.y,
                self.size.z,
                box_text(*min, *max),
            ));
            lines.push(format!("  registered at {}", file.display()));
            lines.push(format!("  {}", tp_text(*tp)));
        }

        let prefix = format!("import {} into {}", self.from.display(), self.name);
        lines.push(outcome_summary(prefix, &self.outcome));
        if !self.outcome.dry_run {
            lines.push(format!("  backup: {}", self.outcome.backup_dir.display()));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("name".to_string(), json!(self.name));
        map.insert("save".to_string(), json!(self.save_name));
        map.insert("from".to_string(), json!(self.from.display().to_string()));
        map.insert("size".to_string(), json!([self.size.x, self.size.y, self.size.z]));
        match &self.slot {
            ImportSlot::Allocated { origin, min, max, tp, file } => {
                map.insert("allocated".to_string(), json!(true));
                map.insert("origin".to_string(), json!([origin.x, origin.y, origin.z]));
                map.insert("min".to_string(), json!([min.x, min.y, min.z]));
                map.insert("max".to_string(), json!([max.x, max.y, max.z]));
                map.insert("tp".to_string(), json!([tp.x, tp.y, tp.z]));
                map.insert("file".to_string(), json!(file.display().to_string()));
            }
            ImportSlot::Existing => {
                map.insert("allocated".to_string(), json!(false));
            }
        }
        for (key, value) in outcome_json_fields(&self.outcome) {
            map.insert(key.to_string(), value);
        }
        Value::Object(map)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use rnbt::{NbtField, NbtList, NbtValue};

    use super::*;
    use super::super::cli::Command;
    use crate::blueprint::{write_structure_file, BlockState};
    use crate::ranvil_cli::structure::extract_box;

    /// The `ground_y` every fixture's [`full_chunk`] is built around: block
    /// `y = 15` is the topmost row of the dirt section, `y = 16` the first
    /// row of the (pre-existing, air) section above it.
    const GROUND_Y: i32 = 15;

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_model_exporter_import_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// A superflat-shaped finished chunk: dirt at `Y=0` (block y 0..15, the
    /// ground and everything below it) and air at `Y=1` (block y 16..31,
    /// everything above the surface) — the second section is present (not
    /// omitted) so a write can still land in it. Same shape
    /// `model_exporter::export`'s own fixture builds, copied rather than
    /// shared since it's private to that module.
    fn full_chunk() -> NbtField {
        fn section(y: i8, block: &str) -> NbtField {
            let palette = NbtList::Compound(vec![NbtField::new_compound(
                "",
                vec![NbtField::new_string("Name", block)],
            )]);
            NbtField::new_compound(
                "",
                vec![
                    NbtField { name: "Y".to_string(), value: NbtValue::Byte(y as u8) },
                    NbtField::new_compound("block_states", vec![NbtField::new_list("palette", palette)]),
                ],
            )
        }
        NbtField::new_compound(
            "",
            vec![
                NbtField::new_list(
                    "sections",
                    NbtList::Compound(vec![section(0, "minecraft:dirt"), section(1, "minecraft:air")]),
                ),
                NbtField::new_i32("DataVersion", 4438),
                NbtField::new_string("Status", "minecraft:full"),
            ],
        )
    }

    /// A fixture save with every chunk column in region (0,0) generated —
    /// same shape `export`'s own fixture builds.
    struct Fixture {
        dir: PathBuf,
        meta: SaveMeta,
    }

    impl Fixture {
        fn new(label: &str) -> Self {
            let dir = temp_dir(label);
            let region_dir = dir.join("region");
            fs::create_dir_all(&region_dir).expect("create region dir");

            let payloads: Vec<Option<ChunkPayload>> =
                (0..CHUNKS_PER_REGION).map(|_| Some(ChunkPayload::Nbt(full_chunk()))).collect();
            let path = region_dir.join("r.0.0.mca");
            Region::new(0, 0, &path).write(&payloads).expect("write fixture region");

            let meta = SaveMeta {
                name: "import-fixture".to_string(),
                path: dir.clone(),
                region_dir,
                regions: vec![(0, 0)],
            };
            Self { dir, meta }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn models_dir(name: &str) -> PathBuf {
        temp_dir(&format!("{name}_registry"))
    }

    /// `world.area.min` starts at `(4, 4)` rather than `(0, 0)`: a slot
    /// allocated at `area.min` has its marker ring one block *outside* its
    /// footprint, and this fixture's only region is `(0, 0)` — starting the
    /// area away from the axes keeps every ring/pillar position (and every
    /// "just outside the box" position these tests probe) inside that one
    /// region rather than spilling into a neighbour the fixture never wrote.
    fn write_world(dir: &Path, save_dir: &Path) {
        fs::write(
            dir.join("world.ron"),
            format!(
                r#"ModelWorld(
    save: "{}",
    ground_y: {GROUND_Y},
    area: (min: (x: 4, z: 4), max: (x: 63, z: 63)),
    gap: 1,
    grid: 8,
    marker: "minecraft:orange_terracotta",
    blueprints_dir: "{}",
)"#,
                save_dir.display().to_string().replace('\\', "/"),
                dir.join("blueprints").display().to_string().replace('\\', "/"),
            ),
        )
        .expect("write world.ron");
    }

    fn write_slot(dir: &Path, name: &str, origin: (i32, i32, i32), size: (i32, i32, i32)) {
        fs::write(
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
            format: crate::ranvil_cli::format::OutputFormat::Text,
            command: Command::List(super::super::cli::ListArgs {}),
        }
    }

    fn import_args(name: &str, from: Option<&Path>, below: u32, dry_run: bool, force: bool) -> ImportArgs {
        ImportArgs {
            name: name.to_string(),
            from: from.map(Path::to_path_buf),
            below,
            dry_run,
            force,
        }
    }

    fn write_dirt_file(path: &Path, size: (i32, i32, i32)) {
        let blueprint = Blueprint {
            size: IVec3::new(size.0, size.1, size.2),
            origin: IVec3::ZERO,
            palette: vec![BlockState { name: "minecraft:dirt".to_string(), properties: Vec::new() }],
            blocks: vec![0u16; (size.0 * size.1 * size.2) as usize],
            data_version: 4438,
            failed_columns: 0,
        };
        write_structure_file(path, &blueprint).expect("write fixture .nbt");
    }

    fn block_name_at(meta: &SaveMeta, at: IVec3) -> String {
        let address = crate::edit::address_of(at);
        let mut cache = crate::region_cache::RegionCache::new(meta.clone(), 1);
        cache
            .get_or_load(address.region)
            .expect("resident")
            .get_block(address.local_x, address.y, address.local_z)
            .expect("a populated chunk")
            .get_string("Name")
            .expect("a palette entry")
            .clone()
    }

    /// The ticket's headline round-trip: `import` of a `dirt`-filled file
    /// into a registered slot writes exactly the box's blocks and nothing
    /// outside it, and a subsequent `export` of that slot is `unchanged`
    /// against the same file.
    #[test]
    fn import_into_a_registered_slot_writes_exactly_the_box_and_round_trips_with_export() {
        let fixture = Fixture::new("roundtrip");
        let dir = models_dir("roundtrip");
        write_world(&dir, &fixture.dir);
        // origin.y = ground_y + 1 = 16: the first (air) row above ground —
        // the existing-slot import path places no markers, so nothing else
        // needs to be inside `world.area` here.
        write_slot(&dir, "built", (4, GROUND_Y + 1, 4), (2, 2, 2));

        let file = dir.join("source.nbt");
        write_dirt_file(&file, (2, 2, 2));

        let result = import(&cli(&dir), &import_args("built", Some(&file), 1, false, false))
            .expect("a valid import");
        assert!(matches!(result.slot, ImportSlot::Existing));
        assert_eq!(result.outcome.report.blocks_written, 8);

        // Every position inside the box is now dirt.
        for x in 4..6 {
            for y in (GROUND_Y + 1)..(GROUND_Y + 3) {
                for z in 4..6 {
                    assert_eq!(block_name_at(&fixture.meta, IVec3::new(x, y, z)), "minecraft:dirt");
                }
            }
        }
        // Just outside the box, at the same height: still the pristine air
        // the fixture started with.
        for pos in [
            IVec3::new(3, GROUND_Y + 1, 4),
            IVec3::new(6, GROUND_Y + 1, 4),
            IVec3::new(4, GROUND_Y + 1, 3),
            IVec3::new(4, GROUND_Y + 1, 6),
            IVec3::new(4, GROUND_Y + 3, 4),
        ] {
            assert_eq!(block_name_at(&fixture.meta, pos), "minecraft:air", "{pos} must be untouched");
        }
        // The ground layer right underneath: untouched dirt.
        assert_eq!(block_name_at(&fixture.meta, IVec3::new(4, GROUND_Y, 4)), "minecraft:dirt");

        // export of the same slot afterwards is unchanged against the file.
        let slot = super::super::registry::load_registry(&dir).expect("reload").slots[0].clone();
        let blueprint = extract_box(&fixture.meta, slot.bounds()).expect("extract");
        let on_disk = crate::blueprint::read_structure_file(&file).expect("read fixture file");
        assert!(
            crate::ranvil_cli::structure::blueprints_equal(&blueprint, &on_disk),
            "import -> export must round-trip"
        );
    }

    /// A registered slot whose file's size doesn't match the box's is
    /// refused before anything is written.
    #[test]
    fn a_size_mismatch_against_a_registered_slot_is_a_usage_error() {
        let fixture = Fixture::new("mismatch");
        let dir = models_dir("mismatch");
        write_world(&dir, &fixture.dir);
        write_slot(&dir, "built", (4, GROUND_Y + 1, 4), (2, 2, 2));

        let file = dir.join("wrong_size.nbt");
        write_dirt_file(&file, (3, 2, 2));

        let err = import(&cli(&dir), &import_args("built", Some(&file), 1, false, false))
            .expect_err("should refuse");
        assert!(matches!(err, CliError::Usage(_)));
    }

    /// `import newname --from file.nbt` with no `.ron` allocates one (visible
    /// to `list`), places markers and blocks in one write; a subsequent
    /// identical call overwrites in place rather than allocating a second
    /// slot.
    #[test]
    fn import_of_an_unregistered_name_allocates_marks_and_writes_then_overwrites_in_place() {
        let fixture = Fixture::new("allocate");
        let dir = models_dir("allocate");
        write_world(&dir, &fixture.dir);

        let file = dir.join("source.nbt");
        write_dirt_file(&file, (2, 2, 2));

        // below = 1 (the default), size.y = 2: origin.y = ground_y - 1, so
        // the box's own max.y lands exactly on ground_y — a ring with no
        // pillars, the same shape `new`'s own default-`below` tests use.
        let result = import(&cli(&dir), &import_args("fresh", Some(&file), 1, false, false))
            .expect("a valid import");
        let (origin, box_file) = match &result.slot {
            ImportSlot::Allocated { origin, file, .. } => (*origin, file.clone()),
            ImportSlot::Existing => panic!("expected an allocated slot"),
        };
        assert_eq!(origin, IVec3::new(4, GROUND_Y - 1, 4));
        assert!(box_file.is_file(), "the .ron must have been written");
        // 8 blocks (2x2x2) plus a 12-block ring ((2+2)*(2+2) - 2*2), no
        // pillars.
        assert_eq!(result.outcome.report.blocks_written, 20);

        let listed = super::super::list::list(&cli(&dir), &super::super::cli::ListArgs {})
            .expect("should list");
        assert_eq!(listed.entries.len(), 1);
        assert_eq!(listed.entries[0].name, "fresh");

        // Marker just outside the footprint, on the ground layer.
        assert_eq!(
            block_name_at(&fixture.meta, IVec3::new(origin.x - 1, GROUND_Y, origin.z - 1)),
            "minecraft:orange_terracotta"
        );
        // Inside the box: dirt from the imported file.
        assert_eq!(block_name_at(&fixture.meta, origin), "minecraft:dirt");

        // A second, identical call re-imports into the same slot rather than
        // allocating a new one, overwriting in place.
        let second = import(&cli(&dir), &import_args("fresh", Some(&file), 1, false, false))
            .expect("second import should overwrite in place");
        assert!(matches!(second.slot, ImportSlot::Existing));
        let listed_again = super::super::list::list(&cli(&dir), &super::super::cli::ListArgs {})
            .expect("should list");
        assert_eq!(listed_again.entries.len(), 1, "no second slot should be allocated");
    }

    /// A size change between the first unregistered `import --from` and a
    /// later one against the now-registered name is refused, just like the
    /// already-registered case.
    #[test]
    fn a_changed_file_size_on_a_repeat_unregistered_import_is_refused() {
        let fixture = Fixture::new("size-change");
        let dir = models_dir("size-change");
        write_world(&dir, &fixture.dir);

        let file = dir.join("source.nbt");
        write_dirt_file(&file, (2, 2, 2));
        import(&cli(&dir), &import_args("fresh", Some(&file), 1, false, false))
            .expect("first import should allocate");

        write_dirt_file(&file, (3, 2, 2));
        let err = import(&cli(&dir), &import_args("fresh", Some(&file), 1, false, false))
            .expect_err("should refuse the size change");
        assert!(matches!(err, CliError::Usage(_)));
    }

    /// `--dry-run` against an unregistered name writes neither the `.ron`
    /// nor the region, while still reporting what it would do.
    #[test]
    fn dry_run_writes_neither_the_ron_nor_the_region() {
        let fixture = Fixture::new("dry-run");
        let dir = models_dir("dry-run");
        write_world(&dir, &fixture.dir);

        let file = dir.join("source.nbt");
        write_dirt_file(&file, (2, 2, 2));

        let result = import(&cli(&dir), &import_args("fresh", Some(&file), 1, true, false))
            .expect("a valid plan");
        let (origin, box_file) = match &result.slot {
            ImportSlot::Allocated { origin, file, .. } => (*origin, file.clone()),
            ImportSlot::Existing => panic!("expected an allocated slot"),
        };
        assert!(!box_file.is_file(), "dry-run must not write the .ron");
        assert_eq!(result.outcome.report.blocks_written, 20, "still reports what it would write");

        let registry = super::super::registry::load_registry(&dir).expect("should load");
        assert!(registry.slots.is_empty());
        assert_eq!(
            block_name_at(&fixture.meta, origin),
            "minecraft:dirt",
            "dry-run must not touch the world (this is the pre-existing ground layer)"
        );
    }

    /// An unregistered name with no `--from` is a usage error before
    /// anything is read.
    #[test]
    fn an_unregistered_name_with_no_from_is_a_usage_error() {
        let fixture = Fixture::new("no-from");
        let dir = models_dir("no-from");
        write_world(&dir, &fixture.dir);

        let err = import(&cli(&dir), &import_args("fresh", None, 1, false, false))
            .expect_err("should refuse");
        assert!(matches!(err, CliError::Usage(_)));
    }

    /// A registered name with no `--from` and no prior export is a usage
    /// error naming the missing file.
    #[test]
    fn a_registered_name_with_no_from_and_no_prior_export_is_a_usage_error() {
        let fixture = Fixture::new("no-export-yet");
        let dir = models_dir("no-export-yet");
        write_world(&dir, &fixture.dir);
        write_slot(&dir, "built", (4, GROUND_Y + 1, 4), (2, 2, 2));

        let err = import(&cli(&dir), &import_args("built", None, 1, false, false))
            .expect_err("should refuse");
        assert!(matches!(err, CliError::Usage(_)));
    }
}
