//! Warn-once ledgers for missing resources (ticket 081).
//!
//! Several places in this crate fall back to a default when a resource
//! isn't there — no texture for a block name, no colormap entry for a
//! biome, a chunk section with no biome data — and print a line about it so
//! the gaps stay enumerable. Each of them deduped against a `HashSet`
//! created inside the function that builds a per-chunk lookup table
//! ([`super::atlas::build_block_uv_table`] and its two tint siblings,
//! [`super::decode::decode_chunk`]). Those tables are rebuilt **per
//! background chunk task** — block and biome ids are interned lazily as
//! chunks stream in, so there is no fixed set to resolve once at startup —
//! which made "warn once" mean *once per chunk*: a few hundred streamed
//! chunks reprinted the same handful of lines a few hundred times each and
//! buried every other console message.
//!
//! A [`WarnLedger`] is that same set, just declared as a `static` so it
//! outlives the task that reads it. Each distinct missing name then prints
//! exactly once for the life of the process: the diagnostic survives, the
//! repetition doesn't.
//!
//! It stays a plain value rather than a hidden global — [`WarnLedger::new`]
//! is a `const fn` and callers take one by reference, so a test can hold its
//! own ledger and assert on it without touching what the app uses.

use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The keys already warned about for one warning category.
///
/// Shared across the background chunk tasks (hence the `Mutex`) and cheap
/// to declare as a `static` (hence the `OnceLock` — `HashSet::new` isn't a
/// `const fn`). The lock is only taken around the set lookup itself, never
/// across a whole table build, so parallel chunk tasks don't serialise on
/// it.
pub(crate) struct WarnLedger {
    seen: OnceLock<Mutex<HashSet<String>>>,
}

impl WarnLedger {
    pub(crate) const fn new() -> Self {
        Self {
            seen: OnceLock::new(),
        }
    }

    /// Records `key` and reports whether it had *not* been seen before —
    /// i.e. whether the caller should print its warning:
    ///
    /// ```ignore
    /// if MISSING_TEXTURE.first_time(name) {
    ///     println!("block_viewer: no texture mapping for '{name}'");
    /// }
    /// ```
    ///
    /// Keyed by the missing thing (a block name, a biome name, a chunk
    /// coordinate) rather than by call site, so distinct gaps each get their
    /// own line and only the repeats are dropped.
    pub(crate) fn first_time(&self, key: &str) -> bool {
        let mut seen = self.lock();
        // Checked before the `insert` so the already-warned path — by far
        // the common one, once streaming is under way — doesn't allocate a
        // `String` just to throw it away.
        if seen.contains(key) {
            return false;
        }
        seen.insert(key.to_string());
        true
    }

    fn lock(&self) -> MutexGuard<'_, HashSet<String>> {
        self.seen
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            // A poisoned ledger is not worth a second panic. Whatever task
            // panicked while holding this lock left a set of strings behind,
            // and a set of strings has no invariant to break — the worst
            // case is a line printed twice.
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// How many distinct keys have been warned about.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_key_only_warns_once() {
        let ledger = WarnLedger::new();
        assert!(ledger.first_time("minecraft:some_block"));
        assert!(!ledger.first_time("minecraft:some_block"));
        assert!(!ledger.first_time("minecraft:some_block"));
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn distinct_keys_each_warn() {
        let ledger = WarnLedger::new();
        assert!(ledger.first_time("a"));
        assert!(ledger.first_time("b"));
        assert_eq!(ledger.len(), 2);
    }

    /// The point of taking a ledger by reference rather than reaching for a
    /// global inside the warning site: two ledgers don't see each other.
    #[test]
    fn a_fresh_ledger_starts_empty() {
        let first = WarnLedger::new();
        assert!(first.first_time("shared_key"));

        let second = WarnLedger::new();
        assert_eq!(second.len(), 0);
        assert!(second.first_time("shared_key"));
    }
}
