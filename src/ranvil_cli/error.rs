//! The error/exit-code contract every `ranvil-cli` command follows (ticket
//! 087, roadmap's "Output formats" section): `2` for a bad request caught
//! before anything is touched, `1` for a well-formed request that failed
//! for a data reason. `json` mode prints the error to **stdout** as
//! `{"error": {"kind": ..., "message": ...}}` rather than stderr, so a
//! caller parsing JSON never has to branch on exit status to find out what
//! shape the output is.

use std::process::ExitCode;

use serde_json::json;

use super::format::OutputFormat;

/// A command's failure, already sorted into the roadmap's two buckets.
/// Every later `ranvil-cli` ticket's error paths return one of these rather
/// than growing their own — see the module doc for why the split matters.
#[derive(Debug)]
pub enum CliError {
    /// Bad arguments, an unparseable coordinate/block state, a save or
    /// instance directory that doesn't exist — caught before anything is
    /// touched. Exit code `2`.
    Usage(String),
    /// A well-formed request that failed for a data reason: chunk not
    /// found, region locked, a `DataVersion` mismatch. Exit code `1`.
    Data(String),
}

impl CliError {
    fn kind(&self) -> &'static str {
        match self {
            CliError::Usage(_) => "usage",
            CliError::Data(_) => "data",
        }
    }

    fn message(&self) -> &str {
        match self {
            CliError::Usage(message) | CliError::Data(message) => message,
        }
    }

    fn exit_code(&self) -> ExitCode {
        match self {
            CliError::Usage(_) => ExitCode::from(2),
            CliError::Data(_) => ExitCode::from(1),
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for CliError {}

/// Prints `err` per `format` and returns the exit code [`super::run`] hands
/// back to the shell — the one call site every command's error path
/// eventually reaches.
pub fn report(err: &CliError, format: OutputFormat) -> ExitCode {
    match format {
        OutputFormat::Json => {
            let envelope = json!({ "error": { "kind": err.kind(), "message": err.message() } });
            // `to_string_pretty` only fails on a `Value` that isn't valid
            // UTF-8, which a plain string message never produces.
            println!("{}", serde_json::to_string_pretty(&envelope).unwrap());
        }
        OutputFormat::Text | OutputFormat::Compact => {
            eprintln!("ranvil-cli: {}", err.message());
        }
    }

    err.exit_code()
}
