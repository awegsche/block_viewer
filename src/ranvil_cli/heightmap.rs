//! `ranvil-cli heightmap` (ticket 090) — one chunk's full 16×16 heightmap
//! grid, for whichever of the four kinds `--kind` names. Output shaping only:
//! all the decode work is `mc_anvil::heightmap::read_heightmap` and
//! `HeightmapKind`'s existing variants (ticket 013), reached through
//! [`super::chunk::load_chunk_nbt`] the same way `chunk`/`chunks` (ticket 089)
//! load a chunk's NBT.
//!
//! # The "no value" cell
//!
//! A heightmap's stored value is one *above* the highest qualifying block
//! ([`mc_anvil::heightmap`]'s module docs), measured from the chunk's own
//! `min_y`. `min_y` itself is therefore never a real block's height — the
//! lowest an actual qualifying block can produce is `min_y + 1` — so a column
//! reading exactly `min_y` unambiguously means "nothing here qualifies for
//! this map" (a cave roof with no sky-exposed block above it, an unloaded
//! column edge, ...). `text`/`compact` print [`EMPTY_CELL`] for it instead of
//! a number that would look like a real height; `--format json` prints `null`
//! for the same reason — see the ticket's note on why a wrong-looking zero
//! (or, here, a wrong-looking `-64`) is worse than a stated absence.
//!
//! This is distinct from a chunk having **no heightmap of the requested kind
//! at all** (an unfinished chunk, or one whose heightmaps were stripped,
//! ticket 026/089) — that fails the whole command with [`CliError::Data`]
//! naming the kind and chunk, not a grid of placeholders.

use mc_anvil::heightmap::{chunk_min_y, column_index, read_heightmap, ColumnHeights};
use serde_json::{json, Value};

use crate::region_cache::RegionCache;

use super::chunk::load_chunk_nbt;
use super::cli::{Cli, HeightmapArgs, HeightmapKindArg};
use super::error::CliError;
use super::format::Render;
use super::save::resolve_save;

/// Columns per side of a chunk's heightmap grid — 16, the same width
/// `mc_anvil::heightmap::column_index` assumes but doesn't itself export
/// (its own `CHUNK_WIDTH_IN_BLOCKS` is crate-private there).
const GRID_SIDE: usize = 16;

/// Field width `text`/`compact`'s grid pads every cell to — wide enough for
/// any height 9 bits can store (`-64`..`447` with the default `min_y`) plus a
/// column of spacing.
const CELL_WIDTH: usize = 5;

/// What `text`/`compact` print for a column with no qualifying block — see
/// the module docs' note on why this isn't a number.
const EMPTY_CELL: &str = "-";

pub struct HeightmapResult {
    pub save_name: String,
    pub cx: i32,
    pub cz: i32,
    pub kind: HeightmapKindArg,
    /// The world Y the grid's values (and the `min_y` "no value" sentinel)
    /// are measured from — the chunk's own `yPos`-derived bottom, not
    /// necessarily [`mc_anvil::heightmap::DEFAULT_MIN_Y`].
    pub min_y: i32,
    pub heights: ColumnHeights,
}

/// Runs `heightmap`: decodes one chunk and returns the requested kind's full
/// 16×16 grid.
///
/// An ungenerated chunk, and a generated chunk with no heightmap of the
/// requested kind, are both [`CliError::Data`] — never a grid of zeroes (or
/// of `min_y`s), per the module docs.
pub fn heightmap(cli: &Cli, args: &HeightmapArgs) -> Result<HeightmapResult, CliError> {
    let meta = resolve_save(cli)?;
    let (cx, cz) = (args.pos.0.x, args.pos.0.y);
    let kind = args.kind;

    let mut cache = RegionCache::new(meta.clone(), 1);
    let nbt = load_chunk_nbt(&mut cache, (cx, cz)).map_err(|e| {
        CliError::Data(format!(
            "could not read the region for chunk ({cx}, {cz}): {e}"
        ))
    })?;
    let Some(nbt) = nbt else {
        return Err(CliError::Data(format!(
            "chunk ({cx}, {cz}) is not generated"
        )));
    };

    let heights = read_heightmap(&nbt, kind.to_kind()).map_err(|_| {
        let status = nbt
            .get_string("Status")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        CliError::Data(format!(
            "chunk ({cx}, {cz}) has no {} heightmap (status: {status})",
            kind.as_str()
        ))
    })?;

    Ok(HeightmapResult {
        save_name: meta.name,
        cx,
        cz,
        kind,
        min_y: chunk_min_y(&nbt),
        heights,
    })
}

/// The grid `render_json` emits: outer index `dz`, inner index `dx` —
/// matching [`column_index`]'s own `(dx, dz)` ordering
/// (`column_index(dx, dz) = dz * 16 + dx`, X fastest then Z) so a caller
/// cross-referencing `get`/`get-area` output doesn't have to guess the axis
/// order. Pinned here, once, in a function the unit tests below exercise
/// directly — see the ticket's note on why this is worth fixing rather than
/// leaving every later ticket to decide for itself (019).
fn json_rows(heights: &ColumnHeights, min_y: i32) -> Vec<Vec<Value>> {
    (0..GRID_SIDE)
        .map(|dz| {
            (0..GRID_SIDE)
                .map(|dx| {
                    let height = heights[column_index(dx, dz)];
                    if height == min_y {
                        Value::Null
                    } else {
                        json!(height)
                    }
                })
                .collect()
        })
        .collect()
}

/// The `text`/`compact` ASCII grid: one row per `dz`, one column per `dx`,
/// each cell right-padded to [`CELL_WIDTH`]. `headers` adds a `dx` column
/// header row and a `z<n>` row label — `text` passes `true`, `compact`
/// (which drops them) passes `false`.
fn render_grid(heights: &ColumnHeights, min_y: i32, headers: bool) -> String {
    let mut lines = Vec::with_capacity(GRID_SIDE + 1);

    if headers {
        let mut header = " ".repeat(CELL_WIDTH);
        for dx in 0..GRID_SIDE {
            header.push_str(&format!("{:>1$}", dx, CELL_WIDTH));
        }
        lines.push(header);
    }

    for dz in 0..GRID_SIDE {
        let mut line = if headers {
            format!("{:>1$}", format!("z{dz}"), CELL_WIDTH)
        } else {
            String::new()
        };
        for dx in 0..GRID_SIDE {
            let height = heights[column_index(dx, dz)];
            let cell = if height == min_y {
                EMPTY_CELL.to_string()
            } else {
                height.to_string()
            };
            line.push_str(&format!("{:>1$}", cell, CELL_WIDTH));
        }
        lines.push(line);
    }

    lines.join("\n")
}

impl Render for HeightmapResult {
    fn render_text(&self) -> String {
        format!(
            "heightmap {} for chunk ({}, {}) in {} (min_y {}, '{EMPTY_CELL}' = no qualifying block)\n{}",
            self.kind.as_str(),
            self.cx,
            self.cz,
            self.save_name,
            self.min_y,
            render_grid(&self.heights, self.min_y, true),
        )
    }

    fn render_json(&self) -> Value {
        json!({
            "save": self.save_name,
            "chunk": [self.cx, self.cz],
            "kind": self.kind.as_str(),
            "heights": json_rows(&self.heights, self.min_y),
        })
    }

    fn render_compact(&self) -> String {
        format!(
            "{},{} kind={} min_y={}\n{}",
            self.cx,
            self.cz,
            self.kind.as_str(),
            self.min_y,
            render_grid(&self.heights, self.min_y, false),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_anvil::heightmap::COLUMNS_PER_CHUNK;

    const MIN_Y: i32 = -64;

    fn heights_with_one_set(dx: usize, dz: usize, value: i32) -> ColumnHeights {
        let mut heights = [MIN_Y; COLUMNS_PER_CHUNK];
        heights[column_index(dx, dz)] = value;
        heights
    }

    /// The ticket's own required check: a known `(dx, dz)` lands at
    /// `heights[dz][dx]` in `--format json`'s output, matching
    /// `column_index`'s `(dx, dz)` ordering exactly.
    #[test]
    fn json_rows_places_a_known_dx_dz_at_dz_dx() {
        let heights = heights_with_one_set(3, 7, 100);
        let rows = json_rows(&heights, MIN_Y);

        assert_eq!(rows[7][3], json!(100));
        // Its transpose stays empty — this is a real axis check, not a
        // coincidence of a symmetric fixture.
        assert_eq!(rows[3][7], Value::Null);
    }

    #[test]
    fn json_rows_is_null_for_every_column_with_no_qualifying_block() {
        let heights = [MIN_Y; COLUMNS_PER_CHUNK];
        let rows = json_rows(&heights, MIN_Y);
        assert!(rows.iter().all(|row| row.iter().all(|cell| cell.is_null())));
    }

    #[test]
    fn json_rows_has_sixteen_rows_of_sixteen() {
        let heights = [MIN_Y; COLUMNS_PER_CHUNK];
        let rows = json_rows(&heights, MIN_Y);
        assert_eq!(rows.len(), GRID_SIDE);
        assert!(rows.iter().all(|row| row.len() == GRID_SIDE));
    }

    #[test]
    fn render_grid_prints_the_empty_placeholder_not_min_y() {
        let heights = [MIN_Y; COLUMNS_PER_CHUNK];
        let grid = render_grid(&heights, MIN_Y, false);
        assert!(grid.contains(EMPTY_CELL));
        assert!(!grid.contains(&MIN_Y.to_string()));
    }

    #[test]
    fn render_grid_headers_toggle_the_column_and_row_labels() {
        let heights = heights_with_one_set(0, 0, 64);
        let with_headers = render_grid(&heights, MIN_Y, true);
        let without_headers = render_grid(&heights, MIN_Y, false);

        assert!(with_headers.contains("z0"));
        assert!(!without_headers.contains("z0"));
        // Both still carry the one real value.
        assert!(with_headers.contains("64"));
        assert!(without_headers.contains("64"));
    }

    #[test]
    fn kind_arg_as_str_round_trips_the_clap_spelling() {
        assert_eq!(HeightmapKindArg::WorldSurface.as_str(), "world-surface");
        assert_eq!(HeightmapKindArg::MotionBlocking.as_str(), "motion-blocking");
        assert_eq!(
            HeightmapKindArg::MotionBlockingNoLeaves.as_str(),
            "motion-blocking-no-leaves"
        );
        assert_eq!(HeightmapKindArg::OceanFloor.as_str(), "ocean-floor");
    }
}
