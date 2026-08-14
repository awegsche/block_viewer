//! The structure writer (ticket 023): a [`Blueprint`] out to a gzipped
//! vanilla **structure** file — the format `/structure` blocks save and load,
//! so a blueprint captured here can be placed back into a world by the game
//! itself.
//!
//! Path in, [`io::Result`] out. The filename is ticket 024's problem; nothing
//! here knows about dialogs, and nothing here touches Bevy.
//!
//! ## The format
//!
//! Root is an *unnamed* `TAG_Compound`, gzipped:
//!
//! ```text
//! DataVersion : TAG_Int      -- Blueprint::data_version
//! size        : TAG_List of TAG_Int, 3 entries [x, y, z]
//! palette     : TAG_List of TAG_Compound
//!                 Name       : TAG_String   ("minecraft:oak_stairs")
//!                 Properties : TAG_Compound (omitted when there are none)
//! blocks      : TAG_List of TAG_Compound
//!                 state : TAG_Int                        (palette index)
//!                 pos   : TAG_List of TAG_Int, 3 entries (0-based, relative
//!                                                         to the min corner)
//! entities    : TAG_List (empty)
//! ```
//!
//! Decisions the format leaves open, and which way this writer went:
//!
//! - **Air is written.** The `blocks` list *can* omit blocks, and dropping
//!   `minecraft:air` would shrink a sparse export considerably — but a
//!   structure block only replaces the blocks it has entries for, so an
//!   air-less structure placed over existing terrain leaves that terrain
//!   standing inside it. Emitting air keeps "place this and get exactly what
//!   I selected". A deliberate trade, not an oversight.
//! - **`pos` is relative and 0-based**, so a structure is position
//!   independent. [`Blueprint::origin`] is deliberately *not* written; it only
//!   ever feeds the UI and ticket 024's default filename.
//! - **Property values are strings**, including the numeric-looking
//!   (`"level": "3"`) and boolean-looking (`"waterlogged": "false"`) ones.
//!   That's what vanilla writes and what [`super::extract`] already holds, so
//!   nothing is re-typed on the way out.
//! - **Property order is [`BlockState`]'s sorted order**, which makes writing
//!   the same blueprint twice byte-identical (see
//!   `writing_the_same_blueprint_twice_gives_identical_bytes` — gzip's own
//!   header is deterministic too, `flate2` stamps `mtime` 0).
//! - **An empty `Properties` compound is omitted** rather than written empty,
//!   as vanilla does; some parsers dislike the empty form.
//!
//! ## Why the `blocks` list is streamed rather than built
//!
//! Everything except `blocks` is assembled as an [`NbtField`] tree and handed
//! to [`write_nbt`], which is the cheap and obvious way to use `rnbt`. The
//! `blocks` list is not, because one block costs ~250 bytes as a tag tree (a
//! compound of two fields, each with its own heap-allocated name, plus a
//! three-element `Vec<i32>` for `pos`) against 2 bytes in
//! [`Blueprint::blocks`]. At [`super::MAX_BLOCKS`] — the size the extractor
//! will happily hand over — that tree is several GB, i.e. an out-of-memory
//! abort on a path a user can reach from the export button.
//!
//! So [`write_blocks`] emits the list header itself and writes one entry at a
//! time, and peak memory stays that of the blueprint. The cost is the handful
//! of raw wire-format bytes in [`write_list_header`] and the `TAG_END`s —
//! the only NBT encoding this repo does by hand. They're pinned down by the
//! round-trip tests, which read every written file back through `rnbt`'s own
//! reader.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;
use rnbt::{write_nbt, NbtError, NbtField, NbtList};

use super::{BlockState, Blueprint};

/// NBT tag ids, needed only for the pieces [`write_blocks`] writes by hand.
/// `rnbt` keeps its own copies private, and three constants is a smaller
/// price than widening its API.
const TAG_END: u8 = 0;
const TAG_LIST: u8 = 9;
const TAG_COMPOUND: u8 = 10;

/// The largest structure a **structure block** will load, per side.
///
/// Not a limit on the file format, and deliberately not enforced: a bigger
/// blueprint writes fine and other tools can read it. It's worth a line in
/// the console, though, so the user learns about it here rather than from a
/// structure block silently declining to load their export.
pub const STRUCTURE_BLOCK_MAX_SIZE: i32 = 48;

/// Writes `blueprint` to `path` as a gzipped vanilla structure file,
/// replacing whatever was there.
///
/// No caller until ticket 024 hands it a filename from the save dialog; the
/// tests write real files through it in the meantime.
#[allow(dead_code)]
pub fn write_structure_file(path: &Path, blueprint: &Blueprint) -> io::Result<()> {
    // `BufWriter` because the gzip encoder emits its output in small bursts,
    // and a `write` syscall per burst is a waste on a file that can run to
    // megabytes.
    let mut file = BufWriter::new(File::create(path)?);
    write_structure(&mut file, blueprint)?;
    file.flush()
}

/// Writes `blueprint` to any sink as a gzipped vanilla structure file.
///
/// Split from [`write_structure_file`] so the round-trip tests can write to a
/// `Vec<u8>` — and so ticket 024 could write somewhere other than a file
/// without this module learning about it.
pub fn write_structure<W: Write>(writer: &mut W, blueprint: &Blueprint) -> io::Result<()> {
    warn_if_a_structure_block_cannot_load_it(blueprint);

    // The inner `BufWriter` is what makes the streamed `blocks` list cheap:
    // without it every `state`, every `pos` and every `TAG_END` is its own
    // call into the compressor.
    let mut out = BufWriter::new(GzEncoder::new(writer, Compression::default()));
    write_root(&mut out, blueprint)?;
    out.into_inner()
        .map_err(io::IntoInnerError::into_error)?
        .finish()?
        .flush()
}

/// The root compound: an unnamed `TAG_Compound`, its fields, and the
/// terminating `TAG_End`.
fn write_root<W: Write>(w: &mut W, blueprint: &Blueprint) -> io::Result<()> {
    // Tag, then a zero-length name — the root of a vanilla structure file is
    // unnamed. `write_nbt` would write this same header from a `NbtField`
    // with an empty name, but the root has to be opened by hand anyway for
    // `blocks` to be streamed into it.
    w.write_all(&[TAG_COMPOUND, 0, 0])?;

    write_field(w, &NbtField::new_i32("DataVersion", blueprint.data_version))?;
    write_field(w, &size_field(blueprint))?;
    write_field(w, &palette_field(&blueprint.palette))?;
    write_blocks(w, blueprint)?;
    // Entities are out of scope (ticket 022's decision — they're a second NBT
    // path in the chunk root), but the tag is not optional: vanilla writes it
    // and readers expect it. An empty `TAG_List` is `TAG_End`-typed with
    // length 0, which is what `NbtList::End` encodes.
    write_field(w, &NbtField::new_list("entities", NbtList::End))?;

    w.write_all(&[TAG_END])
}

/// `size` — the selection's extent in blocks, Minecraft axes.
fn size_field(blueprint: &Blueprint) -> NbtField {
    let size = blueprint.size;
    NbtField::new_list("size", NbtList::Int(vec![size.x, size.y, size.z]))
}

/// `palette` — one compound per distinct block state, in the blueprint's own
/// order, so a `state` index means the same thing on both sides of the file.
fn palette_field(palette: &[BlockState]) -> NbtField {
    let entries: Vec<NbtField> = palette.iter().map(palette_entry).collect();
    NbtField::new_list("palette", NbtList::Compound(entries))
}

/// One palette entry: `Name`, plus `Properties` when there are any.
///
/// The entry's own name is empty because elements of a `TAG_List` carry no
/// name on the wire — `rnbt` drops it when writing a list of compounds.
fn palette_entry(state: &BlockState) -> NbtField {
    let mut fields = vec![NbtField::new_string("Name", state.name.clone())];
    if !state.properties.is_empty() {
        let properties: Vec<NbtField> = state
            .properties
            .iter()
            .map(|(key, value)| NbtField::new_string(key.clone(), value.clone()))
            .collect();
        fields.push(NbtField::new_compound("Properties", properties));
    }
    NbtField::new_compound("", fields)
}

/// `blocks` — one compound per block, written straight to the stream.
///
/// See the module docs for why this doesn't go through an [`NbtField`] tree
/// like everything else. The per-entry fields are built and dropped as we go,
/// so the allocations are churn rather than growth.
fn write_blocks<W: Write>(w: &mut W, blueprint: &Blueprint) -> io::Result<()> {
    debug_assert_eq!(
        blueprint.blocks.len(),
        blueprint.volume(),
        "the blocks array should hold exactly one index per block"
    );
    write_list_header(w, "blocks", TAG_COMPOUND, blueprint.blocks.len())?;

    let size = blueprint.size;
    // `blocks` is in `SelectionBounds::iter_blocks`' Y-outer / Z-middle /
    // X-inner order, so a position is the index decomposed rather than
    // anything that needs looking up. Driving the loop off `blocks` itself
    // (rather than a nested `for y/z/x`) is what guarantees the entry count
    // matches the header written above, whatever `size` claims.
    let layer = (size.z as usize).saturating_mul(size.x as usize);
    for (index, &state) in blueprint.blocks.iter().enumerate() {
        let (y, rest) = (index / layer, index % layer);
        let (z, x) = (rest / size.x as usize, rest % size.x as usize);

        write_field(w, &NbtField::new_i32("state", state as i32))?;
        write_field(
            w,
            &NbtField::new_list("pos", NbtList::Int(vec![x as i32, y as i32, z as i32])),
        )?;
        w.write_all(&[TAG_END])?;
    }
    Ok(())
}

/// Opens a named `TAG_List` of `element_tag`, `len` entries long, leaving the
/// entries themselves to the caller.
fn write_list_header<W: Write>(
    w: &mut W,
    name: &str,
    element_tag: u8,
    len: usize,
) -> io::Result<()> {
    w.write_all(&[TAG_LIST])?;
    w.write_all(&(name.len() as u16).to_be_bytes())?;
    w.write_all(name.as_bytes())?;
    w.write_all(&[element_tag])?;
    w.write_all(&(len as i32).to_be_bytes())
}

/// One named field of a compound, through `rnbt`.
///
/// [`write_nbt`]'s `NbtError` can only be the I/O one here — the other two
/// variants are read-side failures — so it collapses to [`io::Error`] rather
/// than growing an error type of its own.
fn write_field<W: Write>(w: &mut W, field: &NbtField) -> io::Result<()> {
    write_nbt(w, field).map_err(|err| match err {
        NbtError::IOError(err) => err,
        other => io::Error::new(io::ErrorKind::InvalidData, format!("{other:?}")),
    })
}

/// Says so, once per file, when the blueprint is too big for a structure
/// block to load. See [`STRUCTURE_BLOCK_MAX_SIZE`].
fn warn_if_a_structure_block_cannot_load_it(blueprint: &Blueprint) {
    let size = blueprint.size;
    if size.max_element() <= STRUCTURE_BLOCK_MAX_SIZE {
        return;
    }
    println!(
        "block_viewer: this structure is {}x{}x{} — larger than the \
         {max}x{max}x{max} a structure block will load. The file is written \
         anyway; loading it needs a tool that doesn't share that limit.",
        size.x,
        size.y,
        size.z,
        max = STRUCTURE_BLOCK_MAX_SIZE,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use bevy::math::IVec3;
    use flate2::read::GzDecoder;
    use rnbt::read_nbt;

    /// What the tests assert against: the written file, read back through
    /// `rnbt`'s own reader, in the same shape as the [`Blueprint`] that went
    /// in.
    ///
    /// `origin` and `failed_columns` are absent by design (a structure is
    /// position independent, and how the extraction went is not the file's
    /// business), so this is deliberately not a `Blueprint` — a round trip
    /// that had to invent two fields to compare would be hiding the fact that
    /// they aren't written.
    #[derive(Debug, PartialEq)]
    struct Structure {
        data_version: i32,
        size: IVec3,
        palette: Vec<BlockState>,
        /// `(state, pos)` per entry, in file order.
        blocks: Vec<(i32, IVec3)>,
        entities: usize,
    }

    fn write(blueprint: &Blueprint) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_structure(&mut bytes, blueprint).expect("writing to a Vec should not fail");
        bytes
    }

    /// Ungzips and parses what [`write`] produced. Panics loudly on anything
    /// missing — every field here is one this writer is supposed to emit.
    fn read_back(bytes: &[u8]) -> Structure {
        let mut decoder = GzDecoder::new(bytes);
        let root = read_nbt(&mut decoder).expect("should be readable NBT");
        assert_eq!(root.name, "", "the root compound is unnamed");

        let size = root
            .get_list("size")
            .and_then(|l| l.as_int_list().cloned())
            .expect("size should be a list of ints");
        assert_eq!(size.len(), 3, "size is [x, y, z]");

        let palette = root
            .get_list("palette")
            .and_then(|l| l.as_compound_list())
            .expect("palette should be a list of compounds")
            .iter()
            .map(|entry| BlockState {
                name: entry
                    .get_string("Name")
                    .expect("every palette entry has a Name")
                    .clone(),
                properties: entry
                    .get_compound("Properties")
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|f| {
                                (
                                    f.name.clone(),
                                    f.as_string().expect("properties are strings").clone(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect();

        let blocks = root
            .get_list("blocks")
            .and_then(|l| l.as_compound_list())
            .expect("blocks should be a list of compounds")
            .iter()
            .map(|entry| {
                let pos = entry
                    .get_list("pos")
                    .and_then(|l| l.as_int_list().cloned())
                    .expect("every block entry has a pos");
                assert_eq!(pos.len(), 3, "pos is [x, y, z]");
                (
                    entry.get_int("state").expect("every block entry has state"),
                    IVec3::new(pos[0], pos[1], pos[2]),
                )
            })
            .collect();

        Structure {
            data_version: root.get_int("DataVersion").expect("DataVersion"),
            size: IVec3::new(size[0], size[1], size[2]),
            palette,
            blocks,
            entities: root
                .get_list("entities")
                .map(|l| l.as_compound_list().map_or(0, Vec::len))
                .expect("entities should be present, even empty"),
        }
    }

    fn state(name: &str, properties: &[(&str, &str)]) -> BlockState {
        BlockState {
            name: name.to_string(),
            properties: properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// A 2x2x2 with air, a plain block and an oriented one — the smallest
    /// blueprint that exercises a multi-entry palette, properties, and the
    /// index decomposition on all three axes.
    fn blueprint() -> Blueprint {
        Blueprint {
            size: IVec3::splat(2),
            origin: IVec3::new(10, 64, -30),
            palette: vec![
                BlockState::air(),
                state("minecraft:stone", &[]),
                state(
                    "minecraft:oak_stairs",
                    &[("facing", "north"), ("half", "bottom")],
                ),
            ],
            //  y=0: (0,0,0)=stone (1,0,0)=air (0,0,1)=air (1,0,1)=stairs
            //  y=1: all air but (1,1,1)=stone
            blocks: vec![1, 0, 0, 2, 0, 0, 0, 1],
            data_version: 3953,
            failed_columns: 0,
        }
    }

    /// The ticket's round trip: every field survives the write.
    #[test]
    fn a_written_structure_reads_back_as_the_blueprint_that_went_in() {
        let blueprint = blueprint();
        let structure = read_back(&write(&blueprint));

        assert_eq!(structure.data_version, 3953);
        assert_eq!(structure.size, IVec3::splat(2));
        assert_eq!(structure.palette, blueprint.palette);
        assert_eq!(structure.blocks.len(), 8);
        assert_eq!(structure.entities, 0);

        // Y-outer / Z-middle / X-inner, so index 3 is (x=1, y=0, z=1) — the
        // stairs, palette index 2.
        assert_eq!(structure.blocks[3], (2, IVec3::new(1, 0, 1)));
        assert_eq!(structure.blocks[0], (1, IVec3::new(0, 0, 0)));
        assert_eq!(structure.blocks[7], (1, IVec3::new(1, 1, 1)));
    }

    /// The properties are the whole reason ticket 022 reads raw NBT rather
    /// than the decoded world; losing them here would waste that.
    #[test]
    fn palette_properties_survive_with_their_values_as_strings() {
        let structure = read_back(&write(&blueprint()));
        assert_eq!(
            structure.palette[2],
            state(
                "minecraft:oak_stairs",
                &[("facing", "north"), ("half", "bottom")]
            )
        );
    }

    /// Vanilla omits an empty `Properties`, and some parsers dislike the
    /// empty-compound form — so a state with no properties must not write
    /// the tag at all, rather than writing it empty.
    #[test]
    fn a_state_without_properties_omits_the_properties_tag() {
        let bytes = write(&blueprint());
        let mut decoder = GzDecoder::new(&bytes[..]);
        let root = read_nbt(&mut decoder).unwrap();
        let palette = root
            .get_list("palette")
            .unwrap()
            .as_compound_list()
            .unwrap();

        assert!(palette[0].get("Properties").is_none(), "air has none");
        assert!(palette[1].get("Properties").is_none(), "stone has none");
        assert!(palette[2].get("Properties").is_some(), "stairs do");
    }

    /// Air is emitted like any other block: a structure block only replaces
    /// what the file has entries for, so omitting air would leave existing
    /// terrain standing inside a placed structure.
    #[test]
    fn air_is_written_rather_than_omitted() {
        let structure = read_back(&write(&blueprint()));
        assert_eq!(
            structure.blocks.len(),
            8,
            "one entry per block, air included"
        );
        assert_eq!(structure.blocks[1], (0, IVec3::new(1, 0, 0)));
    }

    /// `pos` is relative to the blueprint's min corner, not world space —
    /// `origin` is not written at all.
    #[test]
    fn pos_is_relative_to_the_min_corner_whatever_the_origin() {
        let mut blueprint = blueprint();
        blueprint.origin = IVec3::new(-1234, -60, 4321);
        let structure = read_back(&write(&blueprint));

        assert_eq!(structure.blocks[0].1, IVec3::ZERO);
        assert_eq!(structure.blocks[7].1, IVec3::new(1, 1, 1));
    }

    /// The degenerate case: one block, a one-entry palette, and the index
    /// decomposition with every dimension 1.
    #[test]
    fn a_single_block_blueprint_round_trips() {
        let blueprint = Blueprint {
            size: IVec3::ONE,
            origin: IVec3::new(3, 70, 9),
            palette: vec![state("minecraft:torch", &[("facing", "east")])],
            blocks: vec![0],
            data_version: 4189,
            failed_columns: 0,
        };
        let structure = read_back(&write(&blueprint));

        assert_eq!(structure.size, IVec3::ONE);
        assert_eq!(structure.palette.len(), 1);
        assert_eq!(structure.blocks, vec![(0, IVec3::ZERO)]);
        assert_eq!(structure.data_version, 4189);
    }

    /// Byte-identical output for the same blueprint: it makes the files
    /// diffable, and it's what says the palette's sorted properties and
    /// gzip's own header (`flate2` stamps `mtime` 0) leave nothing to vary
    /// between runs.
    #[test]
    fn writing_the_same_blueprint_twice_gives_identical_bytes() {
        let blueprint = blueprint();
        assert_eq!(write(&blueprint), write(&blueprint));
    }

    /// The file has to actually be gzipped — a structure block reads nothing
    /// else — and that's invisible in a round trip that ungzips its own
    /// output.
    #[test]
    fn the_written_file_is_gzipped() {
        let bytes = write(&blueprint());
        assert_eq!(&bytes[..2], &[0x1f, 0x8b], "gzip magic");
    }

    /// A blueprint bigger than a structure block can load still writes — the
    /// limit is the game's, not the format's, and refusing would lose data
    /// the user asked for.
    #[test]
    fn a_blueprint_too_large_for_a_structure_block_is_still_written() {
        let side = STRUCTURE_BLOCK_MAX_SIZE + 1;
        let volume = (side * side * side) as usize;
        let blueprint = Blueprint {
            size: IVec3::splat(side),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air()],
            blocks: vec![0; volume],
            data_version: 3953,
            failed_columns: 0,
        };
        let structure = read_back(&write(&blueprint));

        assert_eq!(structure.blocks.len(), volume);
        assert_eq!(
            structure.blocks[volume - 1].1,
            IVec3::splat(side - 1),
            "the last entry is the far corner"
        );
    }

    /// The path-taking half, end to end: a real file on disk, read back.
    #[test]
    fn write_structure_file_writes_a_readable_file() {
        let path = std::env::temp_dir().join(format!(
            "block_viewer_test_structure_{}.nbt",
            std::process::id()
        ));
        let blueprint = blueprint();
        write_structure_file(&path, &blueprint).expect("should write");

        let bytes = std::fs::read(&path).expect("should read back");
        assert_eq!(read_back(&bytes), read_back(&write(&blueprint)));

        std::fs::remove_file(&path).ok();
    }

    /// An unwritable path is an `Err`, not a panic — ticket 008's rule that
    /// nothing a user can trigger takes the process down, and ticket 024
    /// turns this into a message in the panel.
    #[test]
    fn an_unwritable_path_is_an_error_rather_than_a_panic() {
        let path = std::env::temp_dir()
            .join("block_viewer_test_no_such_dir")
            .join("nested")
            .join("structure.nbt");
        assert!(write_structure_file(&path, &blueprint()).is_err());
    }

    /// End to end against the real save, and the only way to do this
    /// ticket's in-Minecraft check before ticket 024 wires the export button
    /// to a save dialog: extracts a 16x16x16 box out of the save, writes it,
    /// reads it back, and prints where it left the file so a human can drop
    /// it into `generated/minecraft/structures/` and load it with a
    /// structure block.
    ///
    /// It's also the only test whose palette is *real* — the synthetic ones
    /// above can only contain the properties they were handed. Run with
    /// `cargo test writes_a_real_box_from_the_save -- --nocapture`; the file
    /// is deliberately left behind.
    #[test]
    fn writes_a_real_box_from_the_save() {
        use std::sync::{Arc, Mutex};

        use mc_anvil::region::REGION_WIDTH_IN_CHUNKS;

        use crate::blueprint::{extract_blueprint, ExtractProgress};
        use crate::region_cache::RegionCache;
        use crate::selection::SelectionBounds;
        use crate::world::SECTION_SIZE;

        let saves = mc_anvil::get_saves().expect("could not read the Minecraft saves directory");
        let meta = saves
            .into_iter()
            .find(|s| !s.regions.is_empty())
            .expect("need a save with at least one region");
        let (rx, rz) = meta.regions[0];

        // The middle of a region a player has visited, the same bet
        // `extract`'s real-save tests make about finding generated terrain.
        let region_width = REGION_WIDTH_IN_CHUNKS as i32 * SECTION_SIZE as i32;
        let origin = IVec3::new(
            rx * region_width + region_width / 2,
            60,
            rz * region_width + region_width / 2,
        );
        let bounds = SelectionBounds::from_corners(origin, origin, origin + IVec3::splat(15));
        let blueprint = extract_blueprint(
            bounds,
            &Arc::new(Mutex::new(RegionCache::new(meta, 4))),
            &ExtractProgress::default(),
        )
        .expect("a 16x16x16 box should extract");

        let path = std::env::temp_dir().join(format!(
            "block_viewer_real_box_{}_{}_{}.nbt",
            origin.x, origin.y, origin.z
        ));
        write_structure_file(&path, &blueprint).expect("should write");

        let structure = read_back(&std::fs::read(&path).expect("should read back"));
        assert_eq!(structure.size, IVec3::splat(16));
        assert_eq!(structure.blocks.len(), 16 * 16 * 16);
        assert_eq!(structure.palette, blueprint.palette);
        assert_eq!(structure.data_version, blueprint.data_version);

        println!(
            "wrote {} — {} states:",
            path.display(),
            structure.palette.len()
        );
        for state in &structure.palette {
            println!("  {state}");
        }
    }

    /// Not a correctness test — the write half of the measurement ticket 022
    /// started (it timed extraction at ~175 ms for 2.1M blocks and left
    /// ticket 021's `VOLUME_WARN` alone because the write cost was unknown).
    /// Run with `cargo test measure_large_write -- --nocapture`.
    #[test]
    fn measure_large_write() {
        use std::time::Instant;

        // 128x128x128 = 2,097,152 blocks, the same volume 022 measured.
        let side = 128;
        let volume = (side * side * side) as usize;
        let blueprint = Blueprint {
            size: IVec3::splat(side),
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state("minecraft:stone", &[])],
            // Alternating, so gzip has something to do rather than deflating
            // a megabyte of zeroes into nothing.
            blocks: (0..volume).map(|i| (i % 2) as u16).collect(),
            data_version: 3953,
            failed_columns: 0,
        };

        let start = Instant::now();
        let bytes = write(&blueprint);
        println!(
            "wrote {volume} blocks in {:?} — {} bytes gzipped",
            start.elapsed(),
            bytes.len()
        );
    }
}
