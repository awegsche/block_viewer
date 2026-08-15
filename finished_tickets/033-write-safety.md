# 033 - Write safety: lock, backup, atomic, dry run (roadmap W6)

## Status
Done — implemented and tested. See the Resolution.

## Depends on

| needs | state |
|---|---|
| the chunk edit model (W4) | done — ticket 031 |
| boundary routing and region batching (W5) | done — ticket 032 |
| `ChunkRegion::save` (`mc_anvil` 009) | done — nothing has called it outside tests yet |
| `Region::write`'s temp-file + fsync + rename (`mc_anvil` 009) | done — this ticket's "atomic" is *inherited*, not written here |
| `SaveMeta::is_locked` / `SessionLock::acquire` (`mc_anvil` 016) | done |
| `RegionCache::dirty_regions` / `discard` / `is_resident` (032) | done |

## What this ticket is

031 and 032 both stop one step short of the disk, deliberately: `apply` and
`apply_routed` mutate regions in memory and leave them dirty, because a
function that both edits and writes can't be tested without a disk, and
because the rollback in 032 *is* "throw the in-memory region away", which only
works while nothing has been saved.

This ticket is the step they stopped short of — the roadmap's "sound way to
save a modified world":

- **Refuse to write to a world Minecraft has open.** `mc_anvil` reports it;
  deciding to refuse is this layer's call.
- **Back up before the first write of a session**, per region file.
- **Write atomically.** Already true upstream (`Region::write` writes a temp
  file in the same directory, flushes, `sync_all`s and renames over the
  original). What's left here is not to undermine it, and to say where the
  guarantee ends: it is per file, and a building spanning four region files
  has no cross-file atomicity that any filesystem can give us.
- **Dry run**: which regions and chunks would change, how many blocks, and
  *which files* would be touched — without writing anything.

It also implements the contract 032 stated and could not enforce: **one
transaction at a time, saved before the next.** `EditRefusal::RegionHasUnsavedChanges`
is what a caller sees when it skips this layer.

## Design

### 1. `WriteSession` — the thing that holds the lock

```rust
pub struct WriteSession { /* lock, backup dir, backups taken */ }

impl WriteSession {
    pub fn open(save: &SaveMeta) -> Result<Self, WriteError>;
    pub fn open_with(save: &SaveMeta, safety: WriteSafety) -> Result<Self, WriteError>;

    pub fn plan<S: RegionSource>(&self, edit, source: &mut S, policy) -> Result<WritePlan, EditRefusal>;
    pub fn commit<S: RegionSource>(&mut self, edit, source: &mut S, policy) -> Result<WriteSummary, WriteError>;
    pub fn flush<S: RegionSource>(&mut self, source: &mut S) -> Result<WriteSummary, WriteError>;
}
```

**Hold the lock, don't probe it.** `SaveMeta::is_locked` is racy by
construction — the game can open the world between the check and the write —
and `mc_anvil` 016 shipped `SessionLock::acquire` precisely so a caller
doesn't have to live with that. Holding it also stops Minecraft opening the
world *mid-edit*, which is the half a probe can never cover. `Ok(None)` from
`acquire` becomes `WriteError::WorldIsOpen`.

The session is therefore a *write* session, not the app's lifetime: opened
when the user starts editing, dropped (releasing the lock) when they stop. A
read-only viewer that never edits never takes the lock and never keeps
Minecraft out of its own world.

The lock is a **claim, not a re-checked condition**: `commit` does not re-ask
whether the world is open, because the guard it is holding is the answer.
That's also why the Windows/Unix asymmetry in `is_locked` (our own lock reads
as locked on Windows, unlocked on Unix) never surfaces here — nothing in this
module calls it.

### 2. Backups: per region, once per session, before anything is written

`<save>/block_viewer_backups/<session timestamp>/r.<x>.<z>.mca`.

- **Per session, not per write.** The point of a backup is the state *before
  we touched it*; backing up again on the second edit of a region would
  overwrite the only copy of that state with our own output.
- **A new directory per session**, named by UTC timestamp, for the same
  reason one level up: yesterday's backup must survive today's session.
- **Outside the region directory.** A stray file in `region/` is a file
  Minecraft's own directory scan sees; `mc_anvil`'s save discovery skips
  anything that isn't `r.<x>.<z>.mca`, but the game is not ours to bet on.
- **No backup, no write.** A backup failure aborts the whole transaction and
  rolls it back (032's `discard`), rather than proceeding unprotected.
  `WriteSafety::back_up = false` opts out for a caller with its own copy.
- Every backup for a transaction is taken **before the first save**, so the
  "we're half way through and the backup failed" window doesn't exist.

### 3. Ordering inside `commit`, and where atomicity stops

1. `apply_routed` — every region planned, then every region mutated in memory,
   or none (032).
2. Back up every touched region file that this session hasn't backed up yet.
3. Save each region: `ChunkRegion::save` → `Region::write` → temp, fsync,
   rename.

A failure in 1 or 2 rolls back and nothing on disk has changed. A failure in
3 cannot roll back, and pretending otherwise would be the dishonest option:
`WriteError::WriteFailed` names the regions that *were* written, the one that
failed, and the backup directory to restore from. The regions that failed stay
dirty — `ChunkRegion::save` clears the dirty set on success only — so a retry
still writes everything, and the next transaction over them is refused by
032's guard until the user has resolved it.

This is the honest statement of the guarantee: **atomic per region file,
all-or-nothing in memory, and reported-and-recoverable across files.** No
filesystem offers a four-file transaction, and a journal that replayed one
would be a bigger machine than the thing it protects.

### 4. `flush`, and `RegionSource::dirty_regions`

`commit` saves the regions its own transaction touched. `flush` saves
everything dirty in the source — the entry point for a caller that batched
with `EditPolicy::allow_dirty_regions`, and the thing "save before quitting"
will call.

That needs the source to be able to say what's dirty, so `RegionSource` grows
a third method. Both implementations are one line (`RegionCache` already has
`dirty_regions`), and having it on the trait keeps `commit` and `flush`
sharing one save path instead of two.

`flush` sorts what it gets: `RegionCache::dirty_regions` iterates a `HashMap`,
and a save order that varies run to run is the kind of thing that makes a test
flake once a month (032 hit exactly this).

### 5. The dry run

`WritePlan` is 032's `EditReport` plus what only the session knows: the file
each region lives in, where its backup would go, and whether it already has
one this session. `Display` renders it as the text a UI or a CLI shows before
committing.

It writes nothing and creates nothing — not even the backup directory, which
is created on first use rather than at `open`, so a session that only ever
plans leaves no trace on disk.

## Explicitly not in this ticket

- **Restoring from a backup.** The files are plain `.mca`s and copying one
  back is a file manager operation; an in-app restore is worth having once
  something can go wrong in a way the user didn't cause.
- **Pruning old backup directories.** Every session leaves one. Deleting the
  user's only copy of their world on a heuristic is not a feature to add
  casually, and disk is cheap.
- **Re-meshing what was written** — W7.
- **Any UI or any caller** — W8, as it was for 031 and 032. This ticket ends
  with the module and its tests.

## Tests

Against the `SaveFixture` temp-directory save 032 built (so: no real world,
and the four regions that meet at the origin):

- A committed edit reaches the *file*, and the region comes back clean.
- The first write of a session copies the pre-edit region file into the
  backup directory; a second write to the same region does **not** overwrite
  that copy — the backup still holds the original blocks.
- A refused edit creates no backup directory and writes nothing.
- A backup that can't be taken (a plain file where the backup directory would
  go) refuses the transaction, rolls it back, and leaves the file on disk
  untouched.
- `flush` saves everything an `allow_dirty_regions` batch left behind, backs
  each file up once, and leaves nothing dirty.
- The dry run names the files and backups it would write and touches nothing:
  no dirty regions, no backup directory, byte-identical region files.
- A world whose `session.lock` somebody else holds is refused
  (`#[cfg(windows)]`: a POSIX record lock never conflicts with its own owner,
  so the Unix version of this test would need a child process — `mc_anvil`
  016's test suite has one and this layer's rule is the same rule).
- The backup directory name is the session's UTC timestamp, against a couple
  of known epochs.

## Done when

- An edit committed through a `WriteSession` is on disk, backed up, and the
  world was provably not open in Minecraft while it happened.
- A dry run reports the same regions, chunks and block count as the commit
  would, plus the files, and leaves the save byte-identical.
- Nothing in the viewer or the citybuilder behaves differently yet — W8 is the
  first caller.

## Resolution

`src/edit/session.rs` (~490 lines with the docs), a third method on
`RegionSource`, and 9 new tests in `src/edit/tests.rs`. `cargo test`: 231 pass.
No new clippy warnings.

### Deviations from the design above

**1. `commit` decides the rollback, not the write path.** The obvious shape was
for the backup/save routine to roll back on any failure it saw, but that
routine is shared with `flush` — and `flush` saving a batch it didn't create
must **not** discard on failure, because the edits it would throw away aren't
its own. So `write_regions` rolls nothing back and `commit` does it, for
`Backup` and `Refused` only.

`WriteFailed` is excluded even when it failed on the very first region and
nothing reached disk. Nothing is lost in that case — the edit is still in
memory and still dirty — and discarding would turn a retryable disk error into
lost work. The cost is that the next transaction over those regions is refused
until the user resolves it, which is the correct thing to happen.

**2. `route::refusal_for` split out of `fetch`.** The save path goes back to
the source for each region and meets the same two `RegionUnavailable` cases, so
the mapping is shared rather than written twice.

**3. A hand-rolled UTC timestamp.** The backup directory is named
`YYYY-MM-DDTHH-MM-SSZ`, which needs a civil date from a Unix epoch and there is
no date crate in this project. Howard Hinnant's `civil_from_days` is twelve
lines and has a test with the two leap-year cases in it (2024, and the century
that is one anyway). Epoch seconds would have been free, and would have been no
answer to "which of these backups is yesterday's" — which is the only question
this string exists to answer.

### Confirmed while building it

- **The backup ordering matters more than it looks.** Backing up each region
  immediately before saving it would mean a backup failure on region 3 with
  regions 1 and 2 already replaced: no rollback possible, and the user holding
  backups for exactly the two files that no longer need them. Copying every
  file first makes "the backup failed" and "the save failed" two states instead
  of one blurred one.
- **`Display for WriteError` is load-bearing here in a way it wasn't for
  `EditRefusal`.** Every one of these is something the user has to *do*
  something about — close Minecraft, free up disk, restore from a directory
  whose path they need to be told. The `WriteFailed` message names the count,
  the region and the directory for that reason.
- **The lock test is Windows-only, honestly.** A POSIX record lock never
  conflicts with its own owner, so staging the lock in-process on Linux would
  produce a test that passes without testing anything. `mc_anvil` 016 solved
  that with a child process for the lock's own tests; what's under test here is
  only that this layer refuses when the lock is refused, so the Windows version
  is enough — and `mc_anvil` ticket 025 is where the "does it conflict with the
  actual game" check lives. Noted in `todo.md`, because that check is now
  load-bearing for this repo.

### Next

W7: live re-mesh — an edit marks its chunks dirty (`EditReport::chunks`) and
005-f's existing re-mesh queue picks them up, neighbours included. Then W8, the
paint/fill command in the viewer, which is the first caller of any of
031–033 and the gate the rest of the roadmap waits on.
