//! `model-exporter export [<name>...]` (ticket 135, `MODEL_EXPORTER_ROADMAP.md`
//! "export") — the command the tool is named for: every registered model's
//! world box, in one call, with no corners typed and no selection dragged.
//!
//! Per slot this is exactly [`crate::ranvil_cli::structure::export`]'s own
//! read/write pair — [`extract_blueprint`] then [`write_structure_file`] —
//! with the box looked up from the slot's `.ron` instead of `--from`/`--to`.
//! Unlike that single-box command, a whole-run export shares one
//! [`RegionCache`] across every slot rather than opening (and re-reading
//! region files into) a fresh one per slot: [`export`] sizes it once, up
//! front, to the union of every target slot's chunk columns — see
//! [`crate::ranvil_cli::structure::extract_box`]'s own docs on why that
//! function isn't the right thing to call once per slot here.
//!
//! Three checks decide a slot's status before anything is written:
//!
//! 1. [`Blueprint::failed_columns`] — an ungenerated chunk inside the box —
//!    is `failed`, file untouched.
//! 2. [`slot_is_still_empty`] — nobody has built here yet, whether because
//!    the box is pure air or because the only non-air blocks are the
//!    pre-existing ground layer a `--below` foundation leaves behind — is
//!    `empty`, file untouched. A registered slot nobody has built in must
//!    never overwrite a real `.nbt`.
//! 3. Otherwise, [`crate::ranvil_cli::structure::blueprints_equal`] against
//!    whatever is already at `slot.out_path()` decides `new` (nothing
//!    there), `unchanged` (identical — the file's bytes and mtime are left
//!    alone), or `updated`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::math::IVec3;
use serde_json::{json, Value};

use crate::blueprint::{
    extract_blueprint, read_structure_file, run_checks, write_structure_file, Blueprint,
    ExtractProgress, STRUCTURE_BLOCK_MAX_SIZE,
};
use crate::ranvil_cli::chunk::region_span;
use crate::ranvil_cli::error::CliError;
use crate::ranvil_cli::format::Render;
use crate::ranvil_cli::save::resolve_save_from;
use crate::ranvil_cli::structure::blueprints_equal;
use crate::region_cache::RegionCache;
use crate::world::SECTION_SIZE;

use super::cli::{Cli, ExportArgs};
use super::registry::{load_registry, ModelSlot, ModelWorld};

/// What [`export_slot`] decided for one slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportStatus {
    /// `slot.out_path()` didn't exist; it does now (or would, without
    /// `--dry-run`).
    New,
    /// `slot.out_path()` existed and differed; it's been rewritten (or
    /// would be).
    Updated,
    /// `slot.out_path()` existed and already matched — left untouched, byte
    /// for byte, so its mtime and git status stay clean.
    Unchanged,
    /// The box is still nothing but pre-existing ground (or pure air) —
    /// see [`slot_is_still_empty`]. File untouched either way.
    Empty,
    /// [`Blueprint::failed_columns`] was nonzero: an ungenerated chunk sits
    /// inside the box. File untouched.
    Failed,
}

impl ExportStatus {
    fn as_str(self) -> &'static str {
        match self {
            ExportStatus::New => "new",
            ExportStatus::Updated => "updated",
            ExportStatus::Unchanged => "unchanged",
            ExportStatus::Empty => "empty",
            ExportStatus::Failed => "failed",
        }
    }
}

/// One slot's export outcome.
#[derive(Debug)]
pub struct SlotExport {
    pub name: String,
    pub status: ExportStatus,
    pub out: PathBuf,
    pub size: IVec3,
    pub blocks: usize,
    pub palette_size: usize,
    pub failed_columns: usize,
}

/// `export`'s result: every target slot's outcome, in the order they were
/// processed (registry order with no names given, the given order
/// otherwise).
#[derive(Debug)]
pub struct ExportResult {
    pub save_name: String,
    pub dry_run: bool,
    pub slots: Vec<SlotExport>,
}

/// The five statuses' totals, plus `exported` (`new + updated`) — what the
/// trailer line and `json`'s counts both report.
struct Counts {
    new: usize,
    updated: usize,
    unchanged: usize,
    empty: usize,
    failed: usize,
}

impl ExportResult {
    /// Whether any slot came back [`ExportStatus::Failed`] — [`super::run`]'s
    /// dispatch reads this to choose exit `1` over `0`, the same "the
    /// per-slot listing is the answer, not an error to replace it with"
    /// contract [`crate::ranvil_cli::structure::StructValidateResult::all_passed`]
    /// follows for `struct validate`.
    pub fn any_failed(&self) -> bool {
        self.slots.iter().any(|slot| slot.status == ExportStatus::Failed)
    }

    fn counts(&self) -> Counts {
        let mut counts = Counts { new: 0, updated: 0, unchanged: 0, empty: 0, failed: 0 };
        for slot in &self.slots {
            match slot.status {
                ExportStatus::New => counts.new += 1,
                ExportStatus::Updated => counts.updated += 1,
                ExportStatus::Unchanged => counts.unchanged += 1,
                ExportStatus::Empty => counts.empty += 1,
                ExportStatus::Failed => counts.failed += 1,
            }
        }
        counts
    }

    fn trailer(&self) -> String {
        let c = self.counts();
        format!(
            "{} exported ({} new, {} updated), {} unchanged, {} empty, {} failed",
            c.new + c.updated,
            c.new,
            c.updated,
            c.unchanged,
            c.empty,
            c.failed,
        )
    }
}

fn slot_line(slot: &SlotExport) -> String {
    format!(
        "{}  {}  {}x{}x{}  {} state{} -> {}",
        slot.name,
        slot.status.as_str(),
        slot.size.x,
        slot.size.y,
        slot.size.z,
        slot.palette_size,
        if slot.palette_size == 1 { "" } else { "s" },
        slot.out.display(),
    )
}

/// Whether nobody has built inside `slot`'s box yet: either
/// [`run_checks`]'s `non_air` check already says the whole box is air, or
/// every non-air block sits at or below `world.ground_y` (world `y <
/// ground_y + 1`).
///
/// The second half matters because a `--below` foundation
/// ([`super::new::new`]'s own convention) puts real, pre-existing blocks —
/// dug-in dirt/stone, and the ground surface itself — under a slot before a
/// human ever builds there. Comparing against `world.ground_y` directly
/// (rather than against the slot's own `origin`) is what makes this correct
/// for any `--below` value, not just the `--below 1` case the roadmap's
/// example walks through: a registered slot nobody has built in must never
/// overwrite a real `.nbt`.
fn slot_is_still_empty(blueprint: &Blueprint, slot: &ModelSlot, world: &ModelWorld) -> bool {
    let non_air = run_checks(blueprint, IVec3::splat(STRUCTURE_BLOCK_MAX_SIZE))
        .into_iter()
        .find(|check| check.name == "non_air")
        .is_some_and(|check| check.pass);
    if !non_air {
        return true;
    }

    let size = blueprint.size;
    let layer_len = size.x as usize * size.z as usize;
    for local_y in 0..size.y {
        let world_y = slot.origin.y + local_y;
        if world_y <= world.ground_y {
            continue; // foundation fill or the ground surface itself
        }
        let start = local_y as usize * layer_len;
        // Index 0 is always `minecraft:air` for a freshly extracted
        // blueprint (`Accumulator::new`) — the same convention
        // `run_checks`'s own `non_air` check relies on.
        if blueprint.blocks[start..start + layer_len].iter().any(|&index| index != 0) {
            return false;
        }
    }
    true
}

/// One slot's export: reads its box out of `cache`, decides its
/// [`ExportStatus`], and (unless `dry_run` or the status leaves the file
/// untouched) writes it.
fn export_slot(
    slot: &ModelSlot,
    world: &ModelWorld,
    cache: &Arc<Mutex<RegionCache>>,
    dry_run: bool,
) -> Result<SlotExport, CliError> {
    let progress = ExtractProgress::default();
    let blueprint = extract_blueprint(slot.bounds(), cache, &progress)
        .map_err(|err| CliError::Data(format!("{}: could not read its box: {err}", slot.name)))?;

    let out = slot.out_path(world);
    let size = blueprint.size;
    let palette_size = blueprint.palette.len();

    if blueprint.failed_columns > 0 {
        return Ok(SlotExport {
            name: slot.name.clone(),
            status: ExportStatus::Failed,
            out,
            size,
            blocks: blueprint.volume(),
            palette_size,
            failed_columns: blueprint.failed_columns,
        });
    }

    if slot_is_still_empty(&blueprint, slot, world) {
        return Ok(SlotExport {
            name: slot.name.clone(),
            status: ExportStatus::Empty,
            out,
            size,
            blocks: blueprint.volume(),
            palette_size,
            failed_columns: 0,
        });
    }

    let status = match read_structure_file(&out) {
        Ok(existing) if blueprints_equal(&existing, &blueprint) => ExportStatus::Unchanged,
        _ if out.is_file() => ExportStatus::Updated,
        _ => ExportStatus::New,
    };

    if !dry_run && status != ExportStatus::Unchanged {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| CliError::Data(format!("could not create {}: {err}", parent.display())))?;
        }
        write_structure_file(&out, &blueprint)
            .map_err(|err| CliError::Data(format!("could not write {}: {err}", out.display())))?;
    }

    Ok(SlotExport {
        name: slot.name.clone(),
        status,
        out,
        size,
        blocks: blueprint.volume(),
        palette_size,
        failed_columns: 0,
    })
}

/// Runs `export [<name>...]`: no names exports every registered slot in
/// registry order; one or more names exports only those, and an unknown one
/// refuses the whole command ([`CliError::Usage`]) before any slot is
/// touched.
///
/// The save's lock gate — [`crate::ranvil_cli::edit::run_write`]'s own
/// pre-check, applied here to a read rather than a write, per the module
/// docs' reasoning — refuses the whole command too, before the shared
/// [`RegionCache`] is even built.
pub fn export(cli: &Cli, args: &ExportArgs) -> Result<ExportResult, CliError> {
    let registry = load_registry(&cli.models_dir)?;

    let targets: Vec<ModelSlot> = if args.names.is_empty() {
        registry.slots.clone()
    } else {
        args.names
            .iter()
            .map(|name| {
                registry
                    .slots
                    .iter()
                    .find(|slot| &slot.name == name)
                    .cloned()
                    .ok_or_else(|| {
                        CliError::Usage(format!(
                            "no model named {:?} registered under {}",
                            name,
                            registry.dir.display()
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let save = cli.save.as_deref().or(Some(registry.world.save.as_str()));
    let meta = resolve_save_from(save, cli.instance.as_deref())?;

    if !args.force {
        let locked = meta.is_locked().map_err(|err| {
            CliError::Data(format!("could not check lock state for {}: {err}", meta.name))
        })?;
        if locked {
            return Err(CliError::Data(format!(
                "{:?} is open in Minecraft — save & quit, or pass --force to read what's on disk anyway",
                meta.name
            )));
        }
    }

    // One cache for the whole run, sized to the union of every target
    // slot's chunk columns, rather than one per slot — see the module docs.
    let chunk_size = SECTION_SIZE as i32;
    let mut span: Option<(i32, i32, i32, i32)> = None;
    for slot in &targets {
        let bounds = slot.bounds();
        let (min_cx, min_cz) = (bounds.min.x.div_euclid(chunk_size), bounds.min.z.div_euclid(chunk_size));
        let (max_cx, max_cz) = (bounds.max.x.div_euclid(chunk_size), bounds.max.z.div_euclid(chunk_size));
        span = Some(match span {
            None => (min_cx, max_cx, min_cz, max_cz),
            Some((s_min_cx, s_max_cx, s_min_cz, s_max_cz)) => (
                s_min_cx.min(min_cx),
                s_max_cx.max(max_cx),
                s_min_cz.min(min_cz),
                s_max_cz.max(max_cz),
            ),
        });
    }
    let capacity = span
        .map(|(min_cx, max_cx, min_cz, max_cz)| region_span(min_cx, max_cx, min_cz, max_cz))
        .unwrap_or(1);
    let cache = Arc::new(Mutex::new(RegionCache::new(meta.clone(), capacity)));

    let mut slots = Vec::with_capacity(targets.len());
    for slot in &targets {
        slots.push(export_slot(slot, &registry.world, &cache, args.dry_run)?);
    }

    Ok(ExportResult { save_name: meta.name, dry_run: args.dry_run, slots })
}

impl Render for ExportResult {
    /// One [`slot_line`] per slot plus [`ExportResult::trailer`] — identical
    /// whether or not `--dry-run` was given, per the ticket ("`--dry-run`
    /// ... prints the same statuses").
    fn render_text(&self) -> String {
        let mut lines: Vec<String> = self.slots.iter().map(slot_line).collect();
        lines.push(self.trailer());
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        let c = self.counts();
        json!({
            "save": self.save_name,
            "dry_run": self.dry_run,
            "slots": self.slots.iter().map(|slot| json!({
                "name": slot.name,
                "status": slot.status.as_str(),
                "out": slot.out.display().to_string(),
                "size": [slot.size.x, slot.size.y, slot.size.z],
                "blocks": slot.blocks,
                "palette_size": slot.palette_size,
                "failed_columns": slot.failed_columns,
            })).collect::<Vec<_>>(),
            "counts": {
                "exported": c.new + c.updated,
                "new": c.new,
                "updated": c.updated,
                "unchanged": c.unchanged,
                "empty": c.empty,
                "failed": c.failed,
            },
        })
    }

    fn render_compact(&self) -> String {
        if self.slots.is_empty() {
            return self.trailer();
        }
        let entries: Vec<String> = self
            .slots
            .iter()
            .map(|slot| format!("{}:{}", slot.name, slot.status.as_str()))
            .collect();
        format!("{}: {}", self.trailer(), entries.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
    use mc_anvil::SaveMeta;
    use rnbt::{NbtField, NbtList, NbtValue};

    use super::*;
    use super::super::cli::Command;
    use crate::ranvil_cli::edit::run_write;
    use crate::blueprint::BlockState;
    use crate::edit::WorldEdit;

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_model_exporter_export_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// A superflat-shaped finished chunk: dirt at `Y=0` (block y 0..15, the
    /// ground and everything below it) and air at `Y=1` (block y 16..31,
    /// everything above the surface) — the second section is present (not
    /// omitted) so a write can still land in it, per `edit::WriteSession`'s
    /// "writes never create sections" rule. `DataVersion` a 1.21 release,
    /// `Status = minecraft:full`.
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

    /// A fixture save with every chunk column in region (0,0) generated
    /// (dirt/dirt, per [`full_chunk`]), so a small slot anywhere near the
    /// origin reads as fully generated ground.
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
                name: "export-fixture".to_string(),
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

    fn write_world(dir: &Path, save_dir: &Path, ground_y: i32) {
        fs::write(
            dir.join("world.ron"),
            format!(
                r#"ModelWorld(
    save: "{}",
    ground_y: {ground_y},
    area: (min: (x: 0, z: 0), max: (x: 31, z: 31)),
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

    fn export_args(names: &[&str], force: bool, dry_run: bool) -> ExportArgs {
        ExportArgs {
            names: names.iter().map(|n| n.to_string()).collect(),
            force,
            dry_run,
        }
    }

    fn build_in_slot(meta: &SaveMeta, at: IVec3) {
        let mut edit = WorldEdit::new();
        edit.set(at, BlockState { name: "minecraft:stone_bricks".to_string(), properties: Vec::new() });
        run_write(meta, false, false, move |_cache| Ok(edit)).expect("a valid write");
    }

    /// The ticket's headline "Done when": a slot with planted blocks exports
    /// as `new`; an untouched slot (ground only) reports `empty` and writes
    /// nothing; a second run of the planted slot reports `unchanged` with
    /// the file's bytes and mtime identical; a further edit makes it
    /// `updated`.
    #[test]
    fn export_reports_new_empty_unchanged_and_updated_across_runs() {
        let fixture = Fixture::new("lifecycle");
        let dir = models_dir("lifecycle");
        // ground_y = 15: block y 16 (Y-section 1) is the first layer above
        // ground, so origin.y = 16 puts the slot's own y=0 right at the
        // first "built" layer.
        write_world(&dir, &fixture.dir, 15);
        write_slot(&dir, "built", (0, 16, 0), (2, 2, 2));
        write_slot(&dir, "untouched", (4, 16, 0), (2, 2, 2));
        build_in_slot(&fixture.meta, IVec3::new(0, 16, 0));

        let result = export(&cli(&dir), &export_args(&[], false, false)).expect("a valid export");
        let built = result.slots.iter().find(|s| s.name == "built").expect("built");
        let untouched = result.slots.iter().find(|s| s.name == "untouched").expect("untouched");
        assert_eq!(built.status, ExportStatus::New);
        assert_eq!(untouched.status, ExportStatus::Empty);
        assert!(built.out.is_file());
        assert!(!untouched.out.is_file(), "an empty slot must not write a file");

        let before_bytes = fs::read(&built.out).expect("read the exported file");
        let before_mtime = fs::metadata(&built.out).expect("stat").modified().expect("mtime");

        let second = export(&cli(&dir), &export_args(&["built"], false, false)).expect("a valid export");
        assert_eq!(second.slots[0].status, ExportStatus::Unchanged);
        assert_eq!(fs::read(&built.out).expect("read again"), before_bytes);
        assert_eq!(
            fs::metadata(&built.out).expect("stat again").modified().expect("mtime again"),
            before_mtime,
            "an unchanged export must not touch the file's mtime"
        );

        build_in_slot(&fixture.meta, IVec3::new(1, 16, 0));
        let third = export(&cli(&dir), &export_args(&["built"], false, false)).expect("a valid export");
        assert_eq!(third.slots[0].status, ExportStatus::Updated);
    }

    /// A `--below`-style foundation (ground-layer dirt below the slot's own
    /// `y=0`, per `slot_is_still_empty`) is still `empty` even though the
    /// palette has more than air in it — the roadmap's own worked case.
    #[test]
    fn a_slot_whose_only_non_air_blocks_are_at_or_below_ground_is_empty() {
        let fixture = Fixture::new("foundation");
        let dir = models_dir("foundation");
        // ground_y = 15 (block y 15, still within the dirt section):
        // origin.y = 14 puts local y=0 one below ground (dirt, pre-existing)
        // and local y=1 exactly at ground_y (also dirt, pre-existing) — both
        // non-air, neither "built".
        write_world(&dir, &fixture.dir, 15);
        write_slot(&dir, "foundation_only", (0, 14, 0), (2, 2, 2));

        let result = export(&cli(&dir), &export_args(&[], false, false)).expect("a valid export");
        assert_eq!(result.slots[0].status, ExportStatus::Empty);
        assert!(!result.slots[0].out.is_file());
    }

    /// `--dry-run` computes and reports every status without writing
    /// anything.
    #[test]
    fn dry_run_writes_nothing() {
        let fixture = Fixture::new("dry-run");
        let dir = models_dir("dry-run");
        write_world(&dir, &fixture.dir, 15);
        write_slot(&dir, "built", (0, 16, 0), (2, 2, 2));
        build_in_slot(&fixture.meta, IVec3::new(0, 16, 0));

        let result = export(&cli(&dir), &export_args(&[], false, true)).expect("a valid plan");
        assert_eq!(result.slots[0].status, ExportStatus::New);
        assert!(!result.slots[0].out.is_file(), "dry-run must not write the file");
    }

    #[test]
    fn an_unknown_name_is_a_usage_error_before_anything_runs() {
        let fixture = Fixture::new("unknown");
        let dir = models_dir("unknown");
        write_world(&dir, &fixture.dir, 15);
        write_slot(&dir, "built", (0, 16, 0), (2, 2, 2));

        let err = export(&cli(&dir), &export_args(&["nope"], false, false)).unwrap_err();
        assert!(matches!(err, CliError::Usage(_)));
        assert!(!dir.join("blueprints").exists(), "no export may run once one name is unknown");
    }

    #[test]
    fn exporting_no_slots_reports_nothing_and_succeeds() {
        let fixture = Fixture::new("no-slots");
        let dir = models_dir("no-slots");
        write_world(&dir, &fixture.dir, 15);

        let result = export(&cli(&dir), &export_args(&[], false, false)).expect("a valid, empty export");
        assert!(result.slots.is_empty());
        assert!(!result.any_failed());
    }
}
