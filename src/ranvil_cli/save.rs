//! `ranvil-cli saves` (ticket 087) — the first real command, deliberately
//! the one that doesn't need `--save` resolved at all (see [`saves`]'s doc
//! comment). Ticket 088 adds [`resolve_save`], the `--save` resolution path
//! every later command needs, plus the first three commands that use it:
//! [`info`], [`regions`], [`lock`].

use std::path::{Path, PathBuf};

use mc_anvil::region::{Region, CHUNKS_PER_REGION};
use mc_anvil::{get_saves_from_instance, SaveMeta};
use serde_json::{json, Value};

use super::cli::{Cli, InfoArgs, LockArgs, RegionLayout, RegionsArgs, SavesArgs};
use super::error::CliError;
use super::format::{OutputFormat, Render};

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

// -------------------------------------------------------------------------------------------------
// ---- --save resolution (ticket 088) --------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

/// Resolves `--save` (a name under `--instance`, or a path straight to one
/// save) plus `--instance` (default: [`default_instance_dir`]) into a
/// concrete [`SaveMeta`] — the one substrate every `ranvil-cli` command
/// below `saves` needs before it can read anything.
///
/// Mirrors the app's own `try_load_save_from`/`pick_named_save` (ticket 064)
/// rather than growing a second implementation of "which save did they
/// mean": a `--save` value that is itself a readable directory is opened
/// directly via [`SaveMeta::from_path`]; otherwise it's looked up by name
/// (exact, then case-insensitive) among the saves [`get_saves_from_instance`]
/// finds under the instance directory. `--save` omitted picks the first save
/// found there, same as the app's own startup default.
///
/// Every failure here is [`CliError::Usage`] — an unresolvable `--save` or
/// `--instance` is a bad argument, not a read that failed against a save
/// that *was* found (the roadmap's exit-code split, and 088's own scope
/// note).
pub fn resolve_save(cli: &Cli) -> Result<SaveMeta, CliError> {
    resolve_save_from(cli.save.as_deref(), cli.instance.as_deref())
}

/// [`resolve_save`]'s body, taking `save`/`instance` directly rather than a
/// whole [`Cli`] — ticket 132's split, so `model_exporter` (whose own `--save`
/// falls back to `world.ron`'s `save` rather than `ranvil-cli`'s `Cli` shape)
/// shares this exact resolution logic instead of growing a second one.
/// `resolve_save` above is the one-line wrapper `ranvil-cli`'s commands keep
/// calling.
pub fn resolve_save_from(save: Option<&str>, instance: Option<&Path>) -> Result<SaveMeta, CliError> {
    let instance = instance.map(Path::to_path_buf).unwrap_or_else(default_instance_dir);

    if let Some(path) = save.map(Path::new).filter(|path| path.is_dir()) {
        return SaveMeta::from_path(path).map_err(|e| {
            CliError::Usage(format!(
                "{} is not a Minecraft save (no readable region directory): {e}",
                path.display()
            ))
        });
    }

    let saves = get_saves_from_instance(&instance).map_err(|e| {
        CliError::Usage(format!(
            "could not read instance directory {}: {e}",
            instance.display()
        ))
    })?;

    match save {
        Some(name) => crate::pick_named_save(&saves, name, &instance).map_err(CliError::Usage),
        None => saves.into_iter().next().ok_or_else(|| {
            CliError::Usage(format!("no Minecraft saves found under {}", instance.display()))
        }),
    }
}

// -------------------------------------------------------------------------------------------------
// ---- info -----------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

pub struct InfoResult {
    pub name: String,
    pub path: PathBuf,
    /// A `DataVersion` sampled from one populated chunk, or `None` when the
    /// save has no chunks to sample — see [`sample_data_version`]. A save is
    /// a patchwork of `DataVersion`s across chunks (ticket 069), so this is
    /// *a* version present, not *the* version.
    pub data_version: Option<i32>,
    pub region_count: usize,
    pub locked: Option<bool>,
    pub nether: bool,
    pub the_end: bool,
}

/// Runs `info`: the save-level summary ticket 088 scopes — name, path,
/// a sampled `DataVersion`, which dimensions are present, region count,
/// locked. Level-level fields (seed, spawn point, gamerules) stay out of
/// scope, since `SaveMeta` doesn't parse `level.dat`; [`Render::render_text`]
/// says so rather than silently omitting them.
pub fn info(cli: &Cli, _args: &InfoArgs) -> Result<InfoResult, CliError> {
    let meta = resolve_save(cli)?;

    Ok(InfoResult {
        data_version: sample_data_version(&meta),
        region_count: meta.regions.len(),
        locked: meta.is_locked().ok(),
        nether: dimension_present(&meta.path, "dimensions/minecraft/the_nether", "DIM-1"),
        the_end: dimension_present(&meta.path, "dimensions/minecraft/the_end", "DIM1"),
        name: meta.name,
        path: meta.path,
    })
}

/// Whether dimension directory `new_dir` (the current `dimensions/<ns>/name`
/// layout) or `legacy_dir` (the pre-1.16 `DIM-1`/`DIM1` layout) exists under
/// a save's root — the same two layouts [`mc_anvil::resolve_region_dir`]
/// already picks between for the overworld, checked here just for presence
/// rather than picking one to read from.
fn dimension_present(save_path: &Path, new_dir: &str, legacy_dir: &str) -> bool {
    save_path.join(new_dir).is_dir() || save_path.join(legacy_dir).is_dir()
}

/// Peeks one populated chunk's `DataVersion` tag — the cheapest way to
/// answer "what Minecraft version is this save" without a second
/// `level.dat` reader (`info`'s scope note) or reaching for the full
/// per-chunk survey ticket 089 builds. Scans regions in coordinate order and
/// stops at the first chunk slot that decodes, so a save with any chunks at
/// all resolves this by reading one region file, not every region.
fn sample_data_version(meta: &SaveMeta) -> Option<i32> {
    let mut regions = meta.regions.clone();
    regions.sort();

    for (rx, rz) in regions {
        let mut region = Region::new(rx, rz, meta.get_region_path(rx, rz));
        let Ok(data) = region.load() else { continue };

        for index in 0..CHUNKS_PER_REGION {
            let Ok(Some(bytes)) = region.get_chunk_nbt_data(&data, index) else { continue };
            let Ok(nbt) = rnbt::read_nbt(&mut std::io::Cursor::new(bytes)) else { continue };
            if let Some(version) = nbt.get_int("DataVersion") {
                return Some(version);
            }
        }
    }
    None
}

impl Render for InfoResult {
    fn render_text(&self) -> String {
        let mut dimensions = vec!["overworld"];
        if self.nether {
            dimensions.push("nether");
        }
        if self.the_end {
            dimensions.push("the_end");
        }

        vec![
            self.name.clone(),
            format!("  path: {}", self.path.display()),
            format!(
                "  data version: {}",
                self.data_version
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown (no chunks found to sample)".to_string())
            ),
            format!("  dimensions: {}", dimensions.join(", ")),
            format!("  regions: {}", self.region_count),
            format!("  locked: {}", lock_text(self.locked)),
            "  seed/spawn/gamerules: not read (level.dat parsing is out of scope for this command)"
                .to_string(),
        ]
        .join("\n")
    }

    fn render_json(&self) -> Value {
        json!({
            "name": self.name,
            "path": self.path.display().to_string(),
            "data_version": self.data_version,
            "dimensions": {
                "overworld": true,
                "nether": self.nether,
                "the_end": self.the_end,
            },
            "region_count": self.region_count,
            "locked": self.locked,
        })
    }

    fn render_compact(&self) -> String {
        let dv = self
            .data_version
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".to_string());
        format!(
            "{} dv={dv} regions={} locked={}",
            self.name,
            self.region_count,
            lock_text(self.locked),
        )
    }
}

// -------------------------------------------------------------------------------------------------
// ---- regions --------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

pub struct RegionEntry {
    pub rx: i32,
    pub rz: i32,
    pub size_bytes: u64,
    pub chunk_count: usize,
}

pub struct RegionsResult {
    pub save_name: String,
    pub layout: RegionLayout,
    pub regions: Vec<RegionEntry>,
    /// `SaveMeta::get_grid_view`'s ASCII map, rendered up front while
    /// `SaveMeta` is still around — only read when `layout` is
    /// [`RegionLayout::Grid`], but cheap either way (it's just the region
    /// coordinate list, no file I/O).
    grid: String,
}

/// Runs `regions`: the region files [`SaveMeta`] lists for this save, each
/// with its file size and populated-chunk count (ticket 088). `--layout
/// grid` combined with `--format json` is rejected up front — `grid` only
/// makes sense as `text`/`compact` output, and json's array-of-objects shape
/// doesn't have a "layout" to pick.
pub fn regions(cli: &Cli, args: &RegionsArgs) -> Result<RegionsResult, CliError> {
    if cli.format == OutputFormat::Json && args.layout == Some(RegionLayout::Grid) {
        return Err(CliError::Usage(
            "--layout grid only applies to --format text or --format compact, not json"
                .to_string(),
        ));
    }

    let meta = resolve_save(cli)?;
    let grid = meta.get_grid_view().to_string();

    let mut coords = meta.regions.clone();
    coords.sort();

    // A region listed by `SaveMeta` came straight from a directory scan of
    // real files, but the size/chunk-count read below can still race a file
    // being deleted or still being written — same "approximate beats none"
    // trade [`dir_size`] makes for `saves`, rather than failing the whole
    // listing over one region.
    let regions = coords
        .into_iter()
        .map(|(rx, rz)| {
            let path = meta.get_region_path(rx, rz);
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let chunk_count = Region::new(rx, rz, &path).chunk_count().unwrap_or(0);
            RegionEntry { rx, rz, size_bytes, chunk_count }
        })
        .collect();

    Ok(RegionsResult {
        save_name: meta.name,
        layout: args.layout.unwrap_or(RegionLayout::List),
        regions,
        grid,
    })
}

impl RegionsResult {
    fn render_list(&self) -> String {
        if self.regions.is_empty() {
            return format!("{}: no regions", self.save_name);
        }

        let mut lines = vec![format!(
            "{} ({} region(s)):",
            self.save_name,
            self.regions.len()
        )];
        for region in &self.regions {
            lines.push(format!(
                "  ({}, {})  {}  {} chunk(s)",
                region.rx,
                region.rz,
                human_size(region.size_bytes),
                region.chunk_count,
            ));
        }
        lines.join("\n")
    }
}

impl Render for RegionsResult {
    fn render_text(&self) -> String {
        match self.layout {
            RegionLayout::List => self.render_list(),
            RegionLayout::Grid => self.grid.clone(),
        }
    }

    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "regions": self.regions.iter().map(|region| json!({
                "x": region.rx,
                "z": region.rz,
                "size_bytes": region.size_bytes,
                "chunk_count": region.chunk_count,
            })).collect::<Vec<_>>(),
        })
    }

    fn render_compact(&self) -> String {
        match self.layout {
            RegionLayout::Grid => self.grid.clone(),
            RegionLayout::List => {
                if self.regions.is_empty() {
                    return format!("{}: 0 regions", self.save_name);
                }
                self.regions
                    .iter()
                    .map(|region| {
                        format!(
                            "{},{}:{}:{}",
                            region.rx,
                            region.rz,
                            human_size(region.size_bytes),
                            region.chunk_count,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// ---- lock -----------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------

pub struct LockResult {
    pub save_name: String,
    pub locked: bool,
}

/// Runs `lock`: a live probe of whether Minecraft (or another writer) is
/// holding this save's `session.lock` right now ([`SaveMeta::is_locked`]).
/// Unlike `saves`/`info` (which fold a lock-probe failure into `None` rather
/// than fail their whole listing), `lock`'s entire job *is* the probe, so a
/// failure here is the command's own [`CliError::Data`] result.
pub fn lock(cli: &Cli, _args: &LockArgs) -> Result<LockResult, CliError> {
    let meta = resolve_save(cli)?;
    let locked = meta
        .is_locked()
        .map_err(|e| CliError::Data(format!("could not check lock state for {}: {e}", meta.name)))?;

    Ok(LockResult { save_name: meta.name, locked })
}

impl Render for LockResult {
    fn render_text(&self) -> String {
        format!("{}: {}", self.save_name, if self.locked { "locked" } else { "unlocked" })
    }

    fn render_json(&self) -> Value {
        json!({ "save": self.save_name, "locked": self.locked })
    }

    fn render_compact(&self) -> String {
        format!("{}:{}", self.save_name, if self.locked { "L" } else { "-" })
    }
}
