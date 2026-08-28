//! `ranvil-cli saves` (ticket 087) — the first real command, deliberately
//! the one that doesn't need `--save` resolved at all (see [`saves`]'s doc
//! comment). Later tickets (088+) add `info`/`regions`/`lock` here, all
//! reading a single resolved save rather than listing many.

use std::path::{Path, PathBuf};

use mc_anvil::get_saves_from_instance;
use serde_json::json;

use super::cli::{Cli, SavesArgs};
use super::error::CliError;
use super::format::Render;

/// One save as `saves` reports it.
pub struct SaveEntry {
    pub name: String,
    pub path: PathBuf,
    /// Total on-disk footprint of the save directory (region files,
    /// `level.dat`, `playerdata`, ...) — a plain recursive walk, not a
    /// parse of region sector tables. Good enough to answer "how big is
    /// this world"; a real `regions.rs` (roadmap's `save.rs` table) can do
    /// better per-dimension accounting later.
    pub size_bytes: u64,
    /// `None` when the lock probe itself failed (e.g. a permissions
    /// error) — deliberately not folded into `false`, which would silently
    /// misreport an unlocked save.
    pub locked: Option<bool>,
}

pub struct SavesResult {
    pub instance: PathBuf,
    pub saves: Vec<SaveEntry>,
}

/// `dirs::config_dir()/.minecraft/saves` — matches `mc_anvil::get_saves`'s
/// own default. Duplicated here (rather than calling `get_saves()` itself)
/// because `saves` needs the resolved directory for display, not just the
/// save list it would filter down to.
fn default_instance_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_default()
        .join(".minecraft/saves")
}

/// Runs `saves`: lists every save under `--instance` (or the default
/// instance directory), each with its on-disk size and lock state.
///
/// Every other `ranvil-cli` command resolves `--save` against an instance
/// directory before doing anything else; `saves` is the one command that
/// *is* that resolution, which is why ticket 087 built it first — it
/// exercises the parser, the format contract and the error contract
/// without touching the save-resolution path 088+ needs.
pub fn saves(cli: &Cli, _args: &SavesArgs) -> Result<SavesResult, CliError> {
    let instance = cli.instance.clone().unwrap_or_else(default_instance_dir);

    let metas = get_saves_from_instance(&instance).map_err(|e| {
        CliError::Usage(format!(
            "could not read instance directory {}: {e}",
            instance.display()
        ))
    })?;

    let saves = metas
        .into_iter()
        .map(|meta| SaveEntry {
            size_bytes: dir_size(&meta.path),
            locked: meta.is_locked().ok(),
            name: meta.name,
            path: meta.path,
        })
        .collect();

    Ok(SavesResult { instance, saves })
}

/// Sums file sizes under `path`, recursing into subdirectories. Entries
/// that can't be read (a race with something deleting files, a permission
/// error on one subfolder) are skipped rather than failing the whole
/// listing — an approximate size beats no size for every other save in the
/// same instance directory.
fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };

    entries
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => dir_size(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

/// `1.5 MB`-style formatting for `render_text`/`render_compact` — this
/// ticket's only consumer of `size_bytes` outside `json`, which reports the
/// raw byte count instead.
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

fn lock_text(locked: Option<bool>) -> &'static str {
    match locked {
        Some(true) => "locked",
        Some(false) => "unlocked",
        None => "lock unknown",
    }
}

impl Render for SavesResult {
    fn render_text(&self) -> String {
        if self.saves.is_empty() {
            return format!("no saves found under {}", self.instance.display());
        }

        let mut lines = vec![format!(
            "{} save(s) under {}:",
            self.saves.len(),
            self.instance.display()
        )];
        for save in &self.saves {
            lines.push(format!(
                "  {}  {}  {}  ({})",
                save.name,
                human_size(save.size_bytes),
                lock_text(save.locked),
                save.path.display(),
            ));
        }
        lines.join("\n")
    }

    fn render_json(&self) -> serde_json::Value {
        json!({
            "instance": self.instance.display().to_string(),
            "saves": self.saves.iter().map(|save| json!({
                "name": save.name,
                "path": save.path.display().to_string(),
                "size_bytes": save.size_bytes,
                "locked": save.locked,
            })).collect::<Vec<_>>(),
        })
    }

    fn render_compact(&self) -> String {
        if self.saves.is_empty() {
            return "0 saves".to_string();
        }
        self.saves
            .iter()
            .map(|save| {
                let locked = match save.locked {
                    Some(true) => "L",
                    Some(false) => "-",
                    None => "?",
                };
                format!("{}:{}:{locked}", save.name, human_size(save.size_bytes))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}
