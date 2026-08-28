# 087 - ranvil-cli: bin shim, arg parsing, output/error contract, `saves`

First ticket of `RANVIL_CLI_ROADMAP.md`. Nothing here is new save-format
logic — it's the scaffolding every later `ranvil-cli` ticket attaches to,
proven end to end with one real command.

## Problem

There is no headless entrypoint into this crate. `block_viewer` and
`citybuilder` both open a window; an agent can't drive either one. Every
piece of read/write logic `ranvil-cli` will wrap already exists
(`mc_anvil::get_saves_from_instance`, `SaveMeta`, ...) but nothing calls it
without a Bevy `App`.

## Scope

**Cargo.toml**: a third `[[bin]]`, and one new dependency.

```toml
[[bin]]
name = "ranvil-cli"
path = "src/bin/ranvil_cli.rs"
```

```toml
# Argument parsing for the ranvil-cli binary (ticket 087). The only new
# dependency this whole executable adds — everything else it calls into
# already exists in the lib.
clap = { version = "4", features = ["derive"] }
```

`serde_json` is *not* new — it's already in `Cargo.lock` transitively (bevy
pulls it), same situation `flate2`/`ron`'s Cargo.toml comments already
describe for this crate. Name it directly for `ranvil_cli::format`'s JSON
output.

**`src/bin/ranvil_cli.rs`** — three lines, same shape as the other two:

```rust
fn main() -> std::process::ExitCode {
    block_viewer::ranvil_cli::run()
}
```

**`src/lib.rs`** — `pub mod ranvil_cli;` alongside `city`/`viewer`.

**`src/ranvil_cli/mod.rs`** — `pub fn run() -> ExitCode`: parses `argv` with
`cli::Cli::parse()`, resolves the save (see below), dispatches on
`Cli::command`, catches a `CliError` and prints it per `Cli::format` before
mapping to an exit code. Submodules per the roadmap's crate layout
(`cli`, `format`, `error`, `coords`, `save`, with `chunk`/`block`/`edit`/
`structure` added by later tickets, each an empty stub module here so the
`Command` enum has somewhere to route to as it grows).

**`cli.rs`** — `#[derive(clap::Parser)] struct Cli { save: Option<String>,
instance: Option<PathBuf>, format: OutputFormat, #[command(subcommand)]
command: Command }` (`save`/`instance`/`format` are global args, available
before or after the subcommand). `enum Command` has exactly one variant this
ticket, `Saves(SavesArgs)`; every later ticket adds one.

**Save resolution** reuses ticket 064's classification, not a fresh parser:
`--save <name-or-path>` is checked against the filesystem the same way
`citybuilder`'s `resolve_selection_in` already does (a directory that lists
at least one save is an instance dir; anything else is the save itself, or a
path straight to one). Factor that function out of `city::mod` into
somewhere both bins can call it (`ranvil_cli::save` can call it directly, or
it moves to `mc_anvil`/a shared spot — whichever keeps `city::run()`'s
behaviour byte-identical). Not every `ranvil-cli` command needs a resolved
save (`saves` itself doesn't), so resolution happens lazily, on first use,
not in `run()` up front.

**`format.rs`** — `OutputFormat { Text, Json, Compact }` (`clap::ValueEnum`);
`trait Render { fn render_text(&self) -> String; fn render_json(&self) ->
serde_json::Value; fn render_compact(&self) -> String { self.render_text() }
}` — `render_compact`'s default falls back to `text`, so a command that has
no terser form (this ticket's `saves`, most single-value reads) doesn't have
to write one. `fn print(result: &impl Render, format: OutputFormat)`
dispatches to stdout.

**`error.rs`** — `enum CliError { Usage(String), Data(String) }` (roadmap's
exit-code split: 2 for a bad request, 1 for a well-formed one that failed).
`fn report(err: &CliError, format: OutputFormat) -> ExitCode` — `text`:
one line to stderr; `json`: `{"error": {"kind": "usage"|"data", "message":
...}}` to **stdout** (so a JSON-parsing caller never branches on exit status
to find the error shape — this is the contract every later ticket's error
paths follow).

**`coords.rs`** — `BlockPos(IVec3)`, `ChunkPos(IVec2)`, each `FromStr`
parsing `"x,y,z"` / `"x,z"` with no surrounding whitespace tolerance beyond
`trim`. Unit-tested here once rather than in every command ticket that needs
coordinates.

**`save.rs`** — the one real command: `ranvil-cli saves [--instance <dir>]`,
wrapping `mc_anvil::get_saves_from_instance` (default instance:
`dirs::config_dir()/.minecraft/saves`, matching `get_saves`). Per save:
name, path, on-disk size (sum of region file sizes — `regions.rs`'s job in
later tickets can do better; this ticket just needs *a* size, e.g. a
recursive dir walk), `SaveMeta::is_locked`. This command doesn't need
`--save` resolved at all, which is deliberately why it's the first one
built — it exercises the parser, the format contract and the error contract
without touching the save-resolution path 088+ needs.

## Out of scope

- Every other subcommand (088–103).
- Moving `resolve_selection_in` all the way into `ranvil` — stays in
  `block_viewer` for now, shared by both bins' `run()`.

## Done when

- `cargo build --bin ranvil-cli` produces a working executable.
- `ranvil-cli saves`, `ranvil-cli saves --format json` and
  `ranvil-cli saves --format compact` all run against a real
  `.minecraft/saves` directory and produce sensible, differently-shaped
  output.
- An unparseable argument (e.g. a subcommand needing a coordinate, once one
  exists — for this ticket, an unknown `--instance` path) exits 2 in `text`
  mode and prints the `{"error": ...}` envelope in `json` mode.
- `cargo check` and `cargo test` pass; `coords::tests` cover both `BlockPos`
  and `ChunkPos` parsing, including rejecting extra/missing components.
