//! `model-exporter new` (ticket 133, `MODEL_EXPORTER_ROADMAP.md` "Coordinates
//! and markers") — the thin command wrapped around [`super::allocate::allocate`]:
//! validate the name, build a `chunk_generated` predicate from the real save
//! (the same per-chunk decode [`crate::ranvil_cli::chunk::chunks`] uses),
//! allocate, then write the `.ron` via [`super::registry::save_slot`].
//!
//! Marker placement is ticket 134's job — the first command that writes to
//! the models world. Until it lands, `new` always behaves as if
//! `--no-markers` were given and says so in its output; `--dry-run` writes
//! nothing at all.

use std::cell::RefCell;
use std::path::PathBuf;

use bevy::math::{IVec2, IVec3};
use serde_json::{json, Value};

use crate::ranvil_cli::chunk::load_chunk_nbt;
use crate::ranvil_cli::error::CliError;
use crate::ranvil_cli::format::Render;
use crate::ranvil_cli::save::resolve_save_from;
use crate::region_cache::RegionCache;

use super::allocate::allocate;
use super::cli::{Cli, NewArgs};
use super::list::{box_text, tp_position, tp_text};
use super::registry::{load_registry, save_slot, ModelSlot};

/// A valid model name: non-empty, `[a-z0-9_]+` — the same charset a `.ron`/
/// `.nbt` file stem and `blueprint::catalogue`'s name matching both need.
fn is_valid_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// What `new` did about marker blocks — always [`MarkersStatus::DryRun`] or
/// [`MarkersStatus::Skipped`] until ticket 134 adds [`MarkersStatus::Placed`]'s
/// real caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkersStatus {
    Placed,
    Skipped,
    DryRun,
}

impl MarkersStatus {
    fn as_str(self) -> &'static str {
        match self {
            MarkersStatus::Placed => "placed",
            MarkersStatus::Skipped => "skipped",
            MarkersStatus::DryRun => "dry-run",
        }
    }
}

#[derive(Debug)]
pub struct NewResult {
    pub name: String,
    pub origin: IVec3,
    pub size: IVec3,
    pub min: IVec3,
    pub max: IVec3,
    pub tp: IVec3,
    pub file: PathBuf,
    pub markers: MarkersStatus,
}

/// Runs `new <name> <width> <height> <depth>`: validates `name`, allocates a
/// box via [`allocate`], and — unless `--dry-run` — writes `<name>.ron`.
///
/// Resolves the save (`cli.save` overriding `world.ron`'s own `save` field,
/// per [`super::cli`]'s doc comment) only to answer `chunk_generated`;
/// nothing in the models world is written here. Works with Minecraft open,
/// same as `list`/`show`.
pub fn new(cli: &Cli, args: &NewArgs) -> Result<NewResult, CliError> {
    if !is_valid_name(&args.name) {
        return Err(CliError::Usage(format!(
            "{:?} is not a valid model name — must match [a-z0-9_]+",
            args.name
        )));
    }

    let registry = load_registry(&cli.models_dir)?;
    if registry.slots.iter().any(|slot| slot.name == args.name) {
        return Err(CliError::Usage(format!(
            "{:?} is already registered under {}",
            args.name,
            registry.dir.display()
        )));
    }

    let size = IVec3::new(args.width, args.height, args.depth);

    let save = cli.save.as_deref().or(Some(registry.world.save.as_str()));
    let meta = resolve_save_from(save, cli.instance.as_deref())?;
    // A small resident capacity: `chunk_generated` is only ever probed for
    // the handful of chunk columns a candidate's marker ring touches, never
    // a bulk survey, so there's nothing to size against a render distance
    // (contrast `ranvil_cli::chunk::region_span`).
    let cache = RefCell::new(RegionCache::new(meta, 8));
    let chunk_generated = |chunk: IVec2| -> bool {
        let mut cache = cache.borrow_mut();
        matches!(
            load_chunk_nbt(&mut cache, (chunk.x, chunk.y)),
            Ok(Some(nbt)) if nbt.get_string("Status").map(String::as_str) == Some("minecraft:full")
        )
    };

    let origin = allocate(&registry.world, &registry.slots, size, args.below, chunk_generated)?;

    let slot = ModelSlot { name: args.name.clone(), origin, size, out: None };
    let file = cli.models_dir.join(format!("{}.ron", args.name));

    let markers = if args.dry_run {
        MarkersStatus::DryRun
    } else {
        save_slot(&cli.models_dir, &slot)
            .map_err(|e| CliError::Data(format!("could not write {}: {e}", file.display())))?;
        MarkersStatus::Skipped
    };

    Ok(NewResult {
        name: slot.name.clone(),
        origin: slot.origin,
        size: slot.size,
        min: slot.min(),
        max: slot.max(),
        tp: tp_position(&slot, &registry.world),
        file,
        markers,
    })
}

impl Render for NewResult {
    fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "{}: origin ({},{},{}) size {}x{}x{}  box {}",
            self.name,
            self.origin.x,
            self.origin.y,
            self.origin.z,
            self.size.x,
            self.size.y,
            self.size.z,
            box_text(self.min, self.max),
        )];
        lines.push(match self.markers {
            MarkersStatus::Placed => {
                "  build inside the orange ring, no higher than the pillar tops".to_string()
            }
            MarkersStatus::Skipped => format!(
                "  registered at {} — markers not placed (model-exporter mark isn't implemented yet, ticket 134)",
                self.file.display()
            ),
            MarkersStatus::DryRun => "  dry run: nothing written".to_string(),
        });
        lines.push(format!("  {}", tp_text(self.tp)));
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "name": self.name,
            "origin": [self.origin.x, self.origin.y, self.origin.z],
            "size": [self.size.x, self.size.y, self.size.z],
            "min": [self.min.x, self.min.y, self.min.z],
            "max": [self.max.x, self.max.y, self.max.z],
            "tp": [self.tp.x, self.tp.y, self.tp.z],
            "file": self.file.display().to_string(),
            "markers": self.markers.as_str(),
        })
    }

    fn render_compact(&self) -> String {
        format!(
            "{}:{},{},{}:{}x{}x{}:{}:{}",
            self.name,
            self.origin.x,
            self.origin.y,
            self.origin.z,
            self.size.x,
            self.size.y,
            self.size.z,
            self.markers.as_str(),
            tp_text(self.tp),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION, REGION_WIDTH_IN_CHUNKS};
    use rnbt::{NbtField, NbtList, NbtValue};

    use super::*;
    use super::super::cli::Command;

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_model_exporter_new_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// A finished chunk: one all-stone section, `Status = minecraft:full` —
    /// the same shape `ranvil_cli::structure::tests::full_chunk` builds,
    /// copied rather than shared since it's private to that module.
    fn full_chunk() -> NbtField {
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
                NbtField::new_string("Status", "minecraft:full"),
            ],
        )
    }

    /// A save with chunk columns `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` (all in
    /// region `(0,0)`) marked `minecraft:full`, everything else absent. Paired
    /// with `world.ron`'s `area.min` of `(16, 16)` below: the first
    /// candidate's marker footprint (`(15,15)..(21,21)`, one block outside a
    /// `5x5x5` box at `(16,16)`) touches exactly those four chunk columns, so
    /// `allocate` finds ground to build on without needing every other
    /// candidate's chunks generated too.
    fn make_save_dir(dir: &Path) {
        let region_dir = dir.join("region");
        fs::create_dir_all(&region_dir).expect("create region dir");

        let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
        for (cx, cz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let index = cx + cz * REGION_WIDTH_IN_CHUNKS;
            payloads[index] = Some(ChunkPayload::Nbt(full_chunk()));
        }
        let path = region_dir.join("r.0.0.mca");
        Region::new(0, 0, &path).write(&payloads).expect("write fixture region");
    }

    fn write_world(dir: &Path, save_dir: &Path) {
        fs::write(
            dir.join("world.ron"),
            format!(
                r#"ModelWorld(
    save: "{}",
    ground_y: -61,
    area: (min: (x: 16, z: 16), max: (x: 511, z: 511)),
    gap: 3,
    grid: 8,
    marker: "minecraft:orange_terracotta",
    blueprints_dir: "assets/city/blueprints",
)"#,
                save_dir.display().to_string().replace('\\', "/")
            ),
        )
        .expect("write world.ron");
    }

    /// Builds the temp `--models-dir` and a generated-chunks save fixture
    /// together, returning `(models_dir, save_dir)` so both stay alive for
    /// the test.
    fn fixture(name: &str) -> (PathBuf, PathBuf) {
        let dir = temp_dir(name);
        let save_dir = temp_dir(&format!("{name}_save"));
        make_save_dir(&save_dir);
        write_world(&dir, &save_dir);
        (dir, save_dir)
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

    fn new_args(name: &str) -> NewArgs {
        NewArgs {
            name: name.to_string(),
            width: 5,
            height: 5,
            depth: 5,
            below: 1,
            no_markers: true,
            dry_run: false,
        }
    }

    #[test]
    fn new_writes_a_ron_that_load_registry_accepts_and_list_shows() {
        let (dir, _save_dir) = fixture("writes_ron");

        let result = new(&cli(&dir), &new_args("barn")).expect("should allocate");
        assert_eq!(result.origin, IVec3::new(16, -62, 16));
        assert_eq!(result.markers, MarkersStatus::Skipped);
        assert!(dir.join("barn.ron").is_file());

        let registry = load_registry(&dir).expect("should load");
        assert_eq!(registry.slots.len(), 1);
        assert_eq!(registry.slots[0].name, "barn");

        let listed = super::super::list::list(&cli(&dir), &super::super::cli::ListArgs {}).expect("should list");
        assert_eq!(listed.entries.len(), 1);
        assert_eq!(listed.entries[0].name, "barn");
    }

    #[test]
    fn new_of_an_already_registered_name_is_a_usage_error() {
        let (dir, _save_dir) = fixture("already_registered");

        new(&cli(&dir), &new_args("barn")).expect("first new should allocate");
        let err = new(&cli(&dir), &new_args("barn")).expect_err("second new should fail");
        assert!(matches!(err, CliError::Usage(_)));
    }

    #[test]
    fn dry_run_writes_nothing() {
        let (dir, _save_dir) = fixture("dry_run");

        let mut args = new_args("barn");
        args.dry_run = true;
        let result = new(&cli(&dir), &args).expect("should allocate");
        assert_eq!(result.markers, MarkersStatus::DryRun);
        assert!(!dir.join("barn.ron").is_file());

        let registry = load_registry(&dir).expect("should load");
        assert!(registry.slots.is_empty());
    }

    #[test]
    fn an_invalid_name_is_a_usage_error() {
        let (dir, _save_dir) = fixture("invalid_name");

        let err = new(&cli(&dir), &new_args("Not-Valid")).expect_err("should reject");
        assert!(matches!(err, CliError::Usage(_)));
    }
}
