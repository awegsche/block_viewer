//! `model-exporter list` / `show <name>` (ticket 132, `MODEL_EXPORTER_ROADMAP.md`
//! "Coordinates and markers") — the first two commands, both read-only over
//! the registry [`super::registry::load_registry`] already validated.
//! Neither opens the models world: `.nbt` status comes from the file on disk
//! ([`crate::blueprint::read_structure_file`]), and the `/tp` line and marker
//! ring geometry are computed from `world.ron` alone, so `list` works with
//! Minecraft running against the same save.

use std::path::PathBuf;

use bevy::math::{IVec2, IVec3};
use serde_json::{json, Value};

use crate::blueprint::read_structure_file;
use crate::ranvil_cli::error::CliError;
use crate::ranvil_cli::format::Render;

use super::cli::{Cli, ListArgs, ShowArgs};
use super::registry::{load_registry, ModelSlot, ModelWorld};

/// Where `new barn 14 9 11` (ticket 133) teleports after allocating a slot,
/// and what `list`/`show` print for an already-registered one: `x` at the
/// center of the footprint (`origin.x + size.x / 2` — matches the roadmap's
/// worked example, `origin (16,-62,0) size 12x10x12` → `/tp @s 22 -60 -3`),
/// `y` one above the ground, `z` three blocks north of the box so the
/// teleport lands just outside the ring's north edge, facing the slot.
pub fn tp_position(slot: &ModelSlot, world: &ModelWorld) -> IVec3 {
    IVec3::new(
        slot.origin.x + slot.size.x / 2,
        world.ground_y + 1,
        slot.origin.z - 3,
    )
}

/// A slot's `.nbt` file, as `list`/`show` report it: whether it exists, and
/// — when it does and reads cleanly — its actual size, so a mismatch against
/// the slot's own `size` (an export from before a `size` edit, say) is
/// visible without opening the file by hand.
#[derive(Debug)]
pub struct NbtStatus {
    pub present: bool,
    /// `Some` when the file exists and [`read_structure_file`] could parse
    /// it; `None` either because the file is missing, or because it exists
    /// but didn't parse (corrupt/foreign `.nbt`) — both render as distinct
    /// text, see [`NbtStatus::render_text`].
    pub size: Option<IVec3>,
}

impl NbtStatus {
    fn of(slot: &ModelSlot, world: &ModelWorld) -> NbtStatus {
        let path = slot.out_path(world);
        if !path.is_file() {
            return NbtStatus { present: false, size: None };
        }
        match read_structure_file(&path) {
            Ok(blueprint) => NbtStatus { present: true, size: Some(blueprint.size) },
            Err(_) => NbtStatus { present: true, size: None },
        }
    }

    /// `missing` / `present (12x10x12)` / `present (SIZE MISMATCH 12x8x12)`
    /// / `present (unreadable)` — the ticket's four cases.
    fn render_text(&self, expected: IVec3) -> String {
        if !self.present {
            return "missing".to_string();
        }
        match self.size {
            Some(size) if size == expected => format!("present ({}x{}x{})", size.x, size.y, size.z),
            Some(size) => format!("present (SIZE MISMATCH {}x{}x{})", size.x, size.y, size.z),
            None => "present (unreadable)".to_string(),
        }
    }

    fn render_json(&self) -> Value {
        json!({
            "present": self.present,
            "size": self.size.map(|s| json!([s.x, s.y, s.z])),
        })
    }
}

/// `pub(super)`: ticket 133's `new` reuses these for its own `text` output
/// rather than reformatting the box/tp line a second way.
pub(super) fn box_text(min: IVec3, max: IVec3) -> String {
    format!(
        "({},{},{})..({},{},{})",
        min.x, min.y, min.z, max.x, max.y, max.z
    )
}

pub(super) fn tp_text(tp: IVec3) -> String {
    format!("/tp @s {} {} {}", tp.x, tp.y, tp.z)
}

// -----------------------------------------------------------------------------------------------
// ---- list -------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct ListEntry {
    pub name: String,
    pub origin: IVec3,
    pub size: IVec3,
    pub min: IVec3,
    pub max: IVec3,
    pub out: PathBuf,
    pub nbt: NbtStatus,
    pub tp: IVec3,
}

#[derive(Debug)]
pub struct ListResult {
    pub dir: PathBuf,
    pub entries: Vec<ListEntry>,
}

/// Runs `list`: one [`ListEntry`] per registered slot, in the registry's own
/// (filename) order.
pub fn list(cli: &Cli, _args: &ListArgs) -> Result<ListResult, CliError> {
    let registry = load_registry(&cli.models_dir)?;
    let entries = registry
        .slots
        .iter()
        .map(|slot| ListEntry {
            name: slot.name.clone(),
            origin: slot.origin,
            size: slot.size,
            min: slot.min(),
            max: slot.max(),
            out: slot.out_path(&registry.world),
            nbt: NbtStatus::of(slot, &registry.world),
            tp: tp_position(slot, &registry.world),
        })
        .collect();

    Ok(ListResult { dir: registry.dir, entries })
}

impl Render for ListResult {
    fn render_text(&self) -> String {
        if self.entries.is_empty() {
            return format!("0 models registered under {}", self.dir.display());
        }

        let mut lines = vec![format!(
            "{} model(s) registered under {}:",
            self.entries.len(),
            self.dir.display()
        )];
        for entry in &self.entries {
            lines.push(format!(
                "  {}  origin ({},{},{})  size {}x{}x{}  box {}  nbt {}  {}",
                entry.name,
                entry.origin.x,
                entry.origin.y,
                entry.origin.z,
                entry.size.x,
                entry.size.y,
                entry.size.z,
                box_text(entry.min, entry.max),
                entry.nbt.render_text(entry.size),
                tp_text(entry.tp),
            ));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> Value {
        Value::Array(
            self.entries
                .iter()
                .map(|entry| {
                    json!({
                        "name": entry.name,
                        "origin": [entry.origin.x, entry.origin.y, entry.origin.z],
                        "size": [entry.size.x, entry.size.y, entry.size.z],
                        "min": [entry.min.x, entry.min.y, entry.min.z],
                        "max": [entry.max.x, entry.max.y, entry.max.z],
                        "out": entry.out.display().to_string(),
                        "nbt": entry.nbt.render_json(),
                        "tp": [entry.tp.x, entry.tp.y, entry.tp.z],
                    })
                })
                .collect(),
        )
    }

    fn render_compact(&self) -> String {
        if self.entries.is_empty() {
            return "0 models".to_string();
        }
        self.entries
            .iter()
            .map(|entry| {
                format!(
                    "{}:{},{},{}:{}x{}x{}:{}",
                    entry.name,
                    entry.origin.x,
                    entry.origin.y,
                    entry.origin.z,
                    entry.size.x,
                    entry.size.y,
                    entry.size.z,
                    entry.nbt.render_text(entry.size),
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// -----------------------------------------------------------------------------------------------
// ---- show -------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------

#[derive(Debug)]
pub struct ShowResult {
    pub name: String,
    pub origin: IVec3,
    pub size: IVec3,
    pub min: IVec3,
    pub max: IVec3,
    pub out: PathBuf,
    pub nbt: NbtStatus,
    /// The marker ring's `y` (134's geometry: `world.ground_y`) and its
    /// corners — one block outside the footprint on every side, the same
    /// `Footprint::expanded(1)` 131's overlap check already computes.
    pub ring_y: i32,
    pub ring_min: IVec2,
    pub ring_max: IVec2,
    pub tp: IVec3,
    /// The `ranvil-cli --save <world> get-area <min> <max>` line an agent
    /// runs to read this slot's blocks directly.
    pub get_area: String,
}

/// Runs `show <name>`: every field `list` reports for one slot, plus the
/// marker ring's geometry (so a human or agent can find the slot without the
/// markers 134 places) and the equivalent `ranvil-cli get-area` line.
/// Unknown `name` → [`CliError::Usage`].
pub fn show(cli: &Cli, args: &ShowArgs) -> Result<ShowResult, CliError> {
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
        })?;

    let min = slot.min();
    let max = slot.max();
    let ring = slot.footprint().expanded(1);

    Ok(ShowResult {
        name: slot.name.clone(),
        origin: slot.origin,
        size: slot.size,
        min,
        max,
        out: slot.out_path(&registry.world),
        nbt: NbtStatus::of(slot, &registry.world),
        ring_y: registry.world.ground_y,
        ring_min: ring.min,
        ring_max: ring.max,
        tp: tp_position(slot, &registry.world),
        get_area: format!(
            "ranvil-cli --save {} get-area {},{},{} {},{},{}",
            registry.world.save, min.x, min.y, min.z, max.x, max.y, max.z
        ),
    })
}

impl Render for ShowResult {
    fn render_text(&self) -> String {
        vec![
            self.name.clone(),
            format!("  origin: ({},{},{})", self.origin.x, self.origin.y, self.origin.z),
            format!("  size: {}x{}x{}", self.size.x, self.size.y, self.size.z),
            format!("  box: {}", box_text(self.min, self.max)),
            format!("  out: {}", self.out.display()),
            format!("  nbt: {}", self.nbt.render_text(self.size)),
            format!(
                "  marker ring: y={} corners ({},{})..({},{})",
                self.ring_y, self.ring_min.x, self.ring_min.y, self.ring_max.x, self.ring_max.y
            ),
            format!("  {}", tp_text(self.tp)),
            format!("  {}", self.get_area),
        ]
        .join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "name": self.name,
            "origin": [self.origin.x, self.origin.y, self.origin.z],
            "size": [self.size.x, self.size.y, self.size.z],
            "min": [self.min.x, self.min.y, self.min.z],
            "max": [self.max.x, self.max.y, self.max.z],
            "out": self.out.display().to_string(),
            "nbt": self.nbt.render_json(),
            "ring": {
                "y": self.ring_y,
                "min": [self.ring_min.x, self.ring_min.y],
                "max": [self.ring_max.x, self.ring_max.y],
            },
            "tp": [self.tp.x, self.tp.y, self.tp.z],
            "get_area": self.get_area,
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
            self.nbt.render_text(self.size),
            tp_text(self.tp),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use super::super::cli::{Cli, Command};

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "block_viewer_test_model_exporter_list_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    const WORLD_RON: &str = r#"
ModelWorld(
    save: "models",
    ground_y: -61,
    area: (min: (x: 0, z: 0), max: (x: 511, z: 511)),
    gap: 3,
    grid: 8,
    marker: "minecraft:orange_terracotta",
    blueprints_dir: "assets/city/blueprints",
)
"#;

    fn write_world(dir: &Path) {
        fs::write(dir.join("world.ron"), WORLD_RON).expect("write world.ron");
    }

    fn write_test_slot(dir: &Path) {
        fs::write(
            dir.join("test.ron"),
            "ModelSlot(name: \"test\", origin: (16, -62, 0), size: (12, 10, 12), out: None)",
        )
        .expect("write test.ron");
    }

    fn cli(dir: &Path) -> Cli {
        Cli {
            models_dir: dir.to_path_buf(),
            save: None,
            instance: None,
            format: crate::ranvil_cli::format::OutputFormat::Text,
            command: Command::List(ListArgs {}),
        }
    }

    #[test]
    fn list_over_an_empty_registry_reports_zero_models() {
        let dir = temp_dir("empty_list");
        write_world(&dir);

        let result = list(&cli(&dir), &ListArgs {}).expect("should load");
        assert!(result.entries.is_empty());
        assert_eq!(
            result.render_text(),
            format!("0 models registered under {}", dir.display())
        );
        assert_eq!(result.render_json(), Value::Array(vec![]));
    }

    #[test]
    fn list_reports_the_box_and_tp_line_for_a_registered_slot() {
        let dir = temp_dir("list_one_slot");
        write_world(&dir);
        write_test_slot(&dir);

        let result = list(&cli(&dir), &ListArgs {}).expect("should load");
        assert_eq!(result.entries.len(), 1);
        let entry = &result.entries[0];
        assert_eq!(entry.name, "test");
        assert_eq!(entry.min, IVec3::new(16, -62, 0));
        assert_eq!(entry.max, IVec3::new(27, -53, 11));
        // origin (16,-62,0) size 12x10x12 -> tp x = 16 + 12/2 = 22,
        // y = ground_y + 1 = -60, z = origin.z - 3 = -3, matching the
        // roadmap's worked example.
        assert_eq!(entry.tp, IVec3::new(22, -60, -3));
        assert!(!entry.nbt.present);
        assert!(result.render_text().contains("nbt missing"));
        assert!(result.render_text().contains("/tp @s 22 -60 -3"));
    }

    #[test]
    fn show_reports_the_ring_geometry_and_get_area_line() {
        let dir = temp_dir("show_one_slot");
        write_world(&dir);
        write_test_slot(&dir);

        let result = show(
            &cli(&dir),
            &ShowArgs { name: "test".to_string() },
        )
        .expect("should load");
        assert_eq!(result.min, IVec3::new(16, -62, 0));
        assert_eq!(result.max, IVec3::new(27, -53, 11));
        assert_eq!(result.ring_y, -61);
        assert_eq!(result.ring_min, IVec2::new(15, -1));
        assert_eq!(result.ring_max, IVec2::new(28, 12));
        assert_eq!(result.tp, IVec3::new(22, -60, -3));
        assert_eq!(
            result.get_area,
            "ranvil-cli --save models get-area 16,-62,0 27,-53,11"
        );
    }

    #[test]
    fn show_of_an_unknown_name_is_a_usage_error() {
        let dir = temp_dir("show_unknown");
        write_world(&dir);

        let err = show(&cli(&dir), &ShowArgs { name: "nope".to_string() }).expect_err("should fail");
        assert!(matches!(err, CliError::Usage(_)));
    }

    #[test]
    fn a_malformed_registry_is_reported_as_a_usage_error() {
        let dir = temp_dir("malformed");
        fs::write(dir.join("world.ron"), "not valid ron(").expect("write world.ron");

        let err = list(&cli(&dir), &ListArgs {}).expect_err("should fail");
        assert!(matches!(err, CliError::Usage(_)));
        assert!(err.to_string().contains("world.ron"));
    }
}
