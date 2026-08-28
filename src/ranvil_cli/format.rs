//! The `--format text|json|compact` contract every `ranvil-cli` command
//! follows (ticket 087). One [`Render`] impl per command's result type, one
//! [`fn@print`] call site in [`super::run`] — no command hand-rolls its own
//! `println!`s, so a new subcommand can't quietly grow a fourth idea of what
//! `--format json` means.

use clap::ValueEnum;
use serde_json::Value;

/// Selects which [`Render`] method a command's result is printed through.
/// `Text` is the default — human-facing prose read at a terminal; `Json`
/// and `Compact` are the agent-facing contract (see the roadmap's "Output
/// formats" section).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Compact,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            OutputFormat::Text => "text",
            OutputFormat::Json => "json",
            OutputFormat::Compact => "compact",
        })
    }
}

/// One command result, rendered three ways. `render_compact`'s default
/// falls back to `render_text`, so a command with no terser form (this
/// ticket's `saves`, most single-value reads) doesn't have to write one —
/// see the roadmap's per-command note on `compact`.
pub trait Render {
    /// Multi-line, labeled, meant for a human terminal.
    fn render_text(&self) -> String;
    /// One JSON value — stable field names, safe to `jq`/parse.
    fn render_json(&self) -> Value;
    /// One line, terse, optimised for token cost in an agent's own context
    /// rather than for parsing.
    fn render_compact(&self) -> String {
        self.render_text()
    }
}

/// Dispatches a rendered result to stdout per `format`. The one call site
/// [`super::run`]'s dispatch uses for every successful command — errors go
/// through [`super::error::report`] instead, since they follow a different
/// (stdout-for-json, stderr-for-text) rule.
pub fn print(result: &impl Render, format: OutputFormat) {
    match format {
        OutputFormat::Text => println!("{}", result.render_text()),
        OutputFormat::Json => {
            let json = result.render_json();
            match serde_json::to_string_pretty(&json) {
                Ok(text) => println!("{text}"),
                // Only reachable if a `Render` impl produces a `Value` that
                // isn't valid UTF-8 (maps with non-string keys, NaN floats,
                // ...) — none of this ticket's data can, so this is a
                // last-resort message rather than a silent empty line.
                Err(e) => println!("{{\"error\": \"failed to serialize result: {e}\"}}"),
            }
        }
        OutputFormat::Compact => println!("{}", result.render_compact()),
    }
}
