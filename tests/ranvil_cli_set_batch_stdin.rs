//! Integration test for `ranvil-cli set-batch -` (ticket 096): the whole
//! point of accepting `-` is that a real process reads a real pipe, which no
//! in-process test (`ranvil_cli::edit`'s own, which call `set_batch`
//! directly) can exercise the same way. This is the one `ranvil-cli` test
//! that spawns the actual compiled binary rather than calling its Rust
//! functions, precisely because stdin is the thing under test.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use mc_anvil::region::{ChunkPayload, Region, CHUNKS_PER_REGION};
use rnbt::{NbtField, NbtList, NbtValue};

/// A 1.21 release — same fixture `DataVersion` `ranvil_cli::edit`'s own
/// tests use, so an edit built with no declared version (`set-batch` never
/// sets one) is compatible with it.
const FIXTURE_DATA_VERSION: i32 = 4438;

/// A finished chunk: one all-stone section at `Y = 0`, `Status =
/// minecraft:full` — the same shape `ranvil_cli::edit`'s own fixture builds,
/// copied rather than shared since that one is private to its module.
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
            NbtField::new_i32("xPos", 0),
            NbtField::new_i32("zPos", 0),
            NbtField::new_i32("yPos", -4),
            NbtField::new_i32("DataVersion", FIXTURE_DATA_VERSION),
            NbtField::new_string("Status", "minecraft:full"),
            NbtField { name: "isLightOn".to_string(), value: NbtValue::Byte(1) },
        ],
    )
}

/// A single-region, single-chunk fixture save in a temp directory, removed
/// on drop.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir()
            .join(format!("block_viewer-ranvil-cli-set-batch-stdin-{label}-{nanos}"));
        let region_dir = dir.join("region");
        std::fs::create_dir_all(&region_dir).expect("create temp dir");

        let mut payloads: Vec<Option<ChunkPayload>> = vec![None; CHUNKS_PER_REGION];
        payloads[0] = Some(ChunkPayload::Nbt(full_chunk()));
        let path = region_dir.join("r.0.0.mca");
        Region::new(0, 0, &path).write(&payloads).expect("write the fixture region");

        Self { dir }
    }

    fn save_path(&self) -> String {
        self.dir.to_string_lossy().to_string()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn ranvil_cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ranvil-cli"))
}

/// `set-batch -` piped a batch with a comment, a blank line, and a
/// correction (the same position written twice) writes exactly what the
/// ticket promises: every line parsed, last write wins per position, and
/// the result reads back through a second `ranvil-cli` invocation.
#[test]
fn set_batch_reads_a_batch_piped_through_stdin() {
    let fixture = Fixture::new("piped");
    let save = fixture.save_path();

    let batch = "# a comment, and a blank line follow\n\
\n\
1,5,1 minecraft:dirt\n\
2,5,1 minecraft:oak_stairs[facing=east,half=top]\n\
1,5,1 minecraft:gold_block\n";

    let mut child = ranvil_cli()
        .args(["--save", &save, "--format", "json", "set-batch", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ranvil-cli");
    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(batch.as_bytes())
        .expect("write batch to stdin");
    let output = child.wait_with_output().expect("wait for ranvil-cli");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(json["status"], serde_json::json!("applied"));
    // Three non-comment, non-blank lines parsed...
    assert_eq!(json["lines_applied"], serde_json::json!(3));
    // ...but (1, 5, 1) is written twice, so only 2 distinct blocks land.
    assert_eq!(json["blocks_written"], serde_json::json!(2));

    let get_output = ranvil_cli()
        .args(["--save", &save, "--format", "json", "get", "1,5,1"])
        .output()
        .expect("run get");
    assert!(get_output.status.success());
    let get_json: serde_json::Value = serde_json::from_slice(&get_output.stdout).expect("valid JSON");
    // Last write wins: the correction (gold_block), not the first line (dirt).
    assert_eq!(get_json["name"], serde_json::json!("minecraft:gold_block"));

    let stairs_output = ranvil_cli()
        .args(["--save", &save, "--format", "json", "get", "2,5,1"])
        .output()
        .expect("run get");
    let stairs_json: serde_json::Value =
        serde_json::from_slice(&stairs_output.stdout).expect("valid JSON");
    assert_eq!(stairs_json["name"], serde_json::json!("minecraft:oak_stairs"));
    assert_eq!(stairs_json["properties"]["facing"], serde_json::json!("east"));
}

/// A malformed line piped through stdin refuses the whole batch (exit 2,
/// `CliError::Usage`) with its line number, and writes nothing — the ticket's
/// "parse-then-apply" contract exercised through the same stdin path as the
/// success case above.
#[test]
fn set_batch_stdin_with_a_malformed_line_writes_nothing() {
    let fixture = Fixture::new("piped-malformed");
    let save = fixture.save_path();

    // Line 3 (1-indexed, comment on line 1 counted) is missing its blockstate.
    let batch = "# comment\n\
1,5,1 minecraft:dirt\n\
2,5,1\n";

    let mut child = ranvil_cli()
        .args(["--save", &save, "--format", "json", "set-batch", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ranvil-cli");
    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(batch.as_bytes())
        .expect("write batch to stdin");
    let output = child.wait_with_output().expect("wait for ranvil-cli");

    assert_eq!(output.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert!(
        json["error"]["message"].as_str().unwrap().contains("line 3"),
        "{json}"
    );

    let get_output = ranvil_cli()
        .args(["--save", &save, "--format", "json", "get", "1,5,1"])
        .output()
        .expect("run get");
    let get_json: serde_json::Value = serde_json::from_slice(&get_output.stdout).expect("valid JSON");
    assert_eq!(get_json["name"], serde_json::json!("minecraft:stone"), "nothing should have been written");
}
