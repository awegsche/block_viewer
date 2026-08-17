//! A tiny cross-plugin gate (ticket 049, roadmap E5): at most one
//! [`crate::edit::session::WriteSession`] open at a time, across
//! `city::commit` (E4) and `city::demolish` (E5).
//!
//! Each module already serializes *itself* — `CommitState`/`DemolishState`'s
//! own single-slot `pending` refuses a second click while its own write is
//! in flight. What neither alone can prevent is *each other*. A placement on
//! one tile and a demolition on an unrelated one could both pass their own
//! checks in the same frame and each open a `WriteSession`. That's not safe
//! the way it sounds like it should be: `SessionLock::acquire` takes a
//! mandatory lock on a freshly opened file handle
//! (`ranvil`'s `session.lock`), not on the *process* — a second concurrent
//! `acquire` from this same app conflicts with the first exactly the way it
//! would conflict with Minecraft, and comes back
//! [`WriteError::WorldIsOpen`](crate::edit::session::WriteError::WorldIsOpen).
//! That's the wrong diagnosis for a race between our own two write paths,
//! and worse, it fails the *loser* outright rather than making it wait —
//! there is no retry built into that error.
//!
//! [`WriteGate::try_acquire`] is a shared "is anything writing right now"
//! flag both `try_commit_placement` and `try_demolish` check (and set)
//! immediately before spawning their own task; [`WriteGate::release`] is
//! called from each module's own poll system once its task resolves,
//! success or failure alike. Never held across a frame boundary except while
//! a task is actually in flight, and never checked without a matching
//! release on the same code path — the invariant that keeps this from
//! deadlocking is "every `pending = Some(..)` was preceded by a successful
//! `try_acquire`, and every place that takes `pending` back out calls
//! `release`."

use bevy::prelude::Resource;

/// Whether some write session, in either `city::commit` or `city::demolish`,
/// is currently in flight. See the module docs.
#[derive(Resource, Default)]
pub(super) struct WriteGate(bool);

impl WriteGate {
    /// Claims the gate and returns `true` if nothing else currently holds
    /// it; leaves it untouched and returns `false` otherwise. Bevy systems
    /// run to completion without interleaving, so a check-then-set here
    /// inside one system's body can't race a second system doing the same —
    /// the two can't be mid-check at the same instant.
    pub(super) fn try_acquire(&mut self) -> bool {
        if self.0 {
            false
        } else {
            self.0 = true;
            true
        }
    }

    /// Releases the gate. Idempotent — releasing an already-free gate is a
    /// harmless no-op, which is what a test that never actually dispatched
    /// through `try_acquire` (both modules' `poll_*` tests set `pending`
    /// directly) hits every time.
    pub(super) fn release(&mut self) {
        self.0 = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_acquire_fails_while_the_first_is_held() {
        let mut gate = WriteGate::default();
        assert!(gate.try_acquire());
        assert!(!gate.try_acquire(), "already held");
    }

    #[test]
    fn releasing_frees_the_gate_for_a_new_acquire() {
        let mut gate = WriteGate::default();
        assert!(gate.try_acquire());
        gate.release();
        assert!(gate.try_acquire(), "the gate should be free again");
    }

    #[test]
    fn releasing_a_free_gate_is_a_harmless_no_op() {
        let mut gate = WriteGate::default();
        gate.release();
        assert!(gate.try_acquire());
    }
}
