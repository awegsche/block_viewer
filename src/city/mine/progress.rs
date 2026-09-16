//! The progress cursor (ticket 114, `MINES_DESIGN.md`'s "Order of work"): a
//! small struct, not a list of slices. [`MineProgress::next_slice`] reads
//! where the cursor currently is; [`MineProgress::advance`] applies the
//! outcome of digging that slice and walks the cursor forward to the next
//! one, closing arms/galleries/rows and transitioning `Sinking -> Mining ->
//! Sinking -> ... -> MinedOut` exactly as the design's diagram draws it.
//!
//! [`MineProgress::advance`] is the only place the cursor moves; everything
//! else (`settle`, `advance_row_arm`) is a private helper it calls. That
//! keeps [`MineProgress::next_slice`] a total, read-only function: whenever
//! the cursor is sitting in [`Phase::Mining`], its `arm` is open and its
//! `step` names real, unfinished work — `advance` never leaves it pointing
//! at a closed arm or a finished gallery pair.

use serde::{Deserialize, Serialize};

use crate::city::definition::Mine;

use super::layout::{Arm, GallerySide, LevelFrame, MineFrame, Slice};

/// One mining level's cursor — which row, which arm, and whether that
/// arm/row pair is extending its secondary or advancing its galleries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LevelCursor {
    pub level: u32,
    pub row: u32,
    pub arm: Arm,
    pub step: RowStep,
    /// How far each arm's secondary has been dug so far, indexed by
    /// [`Arm::index`].
    pub arm_reach: [i32; 2],
    /// Whether each arm has stopped for good (refusal or bedrock) — no rows
    /// beyond that point are ever opened on it, indexed by [`Arm::index`].
    pub arm_closed: [bool; 2],
}

impl LevelCursor {
    fn start(level: u32) -> Self {
        LevelCursor {
            level,
            row: 0,
            arm: Arm::North,
            step: RowStep::Secondary,
            arm_reach: [0, 0],
            arm_closed: [false, false],
        }
    }
}

/// What a [`LevelCursor`] is doing for its current `(row, arm)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowStep {
    /// Extending `arm`'s secondary shaft toward this row's mouths.
    Secondary,
    /// Advancing the row's two galleries alternately; `next` is which one
    /// [`MineProgress::next_slice`] should name.
    Galleries { faces: [Face; 2], next: GallerySide },
}

/// One gallery's progress, indexed by [`GallerySide::index`] inside
/// `RowStep::Galleries::faces`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Face {
    pub distance: i32,
    pub void_run: u32,
    /// No more slices are ever produced for this face once set — either it
    /// was refused/hit bedrock, ran `max_void_run` consecutive empty
    /// slices, or reached `gallery_length`.
    pub closed: bool,
}

/// Where a mine's progress cursor currently is —
/// `MINES_DESIGN.md`'s `Sinking -> Mining -> ... -> MinedOut` diagram.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Sinking the primary shaft toward `target` (a level floor, or the
    /// first level's floor from a fresh mine).
    Sinking { target: i32 },
    Mining(LevelCursor),
    /// Terminal: no level below `min_level_y` is left to sink to.
    MinedOut,
}

/// The outcome of digging one [`Slice`] — what [`MineProgress::advance`]
/// applies to move the cursor forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceOutcome {
    /// Something was removed; resets a gallery face's `void_run`.
    Dug { cost: u32 },
    /// Nothing solid to excavate; bumps a gallery face's `void_run`.
    Void,
    /// The slice would touch bedrock (`never_dig`) — closes the face (or,
    /// for a `Sink`, changes nothing; 116 retries next tick).
    Bedrock,
    /// `EditRefusal` — closes the face (or, for a `Sink`, changes nothing;
    /// 116 logs, refunds and retries next tick, same as a gatherer dig).
    Refused,
}

/// A mine's progress cursor — [`MineFrame::level`](super::layout::MineFrame::level)
/// paired with [`Phase`]. Ticket 116 persists this (`mines.ron`); a
/// [`super::super::state::BuildingId`] stays outside it, the same
/// `SavedProducer` way `city::production`'s buffers are keyed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MineProgress {
    /// The shaft's current bottom — a mining level's floor while
    /// [`Phase::Mining`], an intermediate flight end while [`Phase::Sinking`].
    pub bottom: i32,
    pub phase: Phase,
}

impl MineProgress {
    /// A fresh mine: `bottom = floor_y`, sinking toward level 0's floor.
    pub fn new(frame: &MineFrame) -> Self {
        MineProgress { bottom: frame.floor_y, phase: Phase::Sinking { target: frame.level_floor(0) } }
    }

    /// The current mining level's [`LevelFrame`], or `None` while sinking
    /// or mined out.
    pub fn level<'a>(&self, frame: &'a MineFrame) -> Option<LevelFrame<'a>> {
        match &self.phase {
            Phase::Mining(cursor) => Some(frame.level(cursor.level)),
            _ => None,
        }
    }

    /// The next slice to dig, or `None` iff [`Phase::MinedOut`]. See the
    /// module docs: this trusts the invariants [`advance`](Self::advance)
    /// maintains rather than re-deriving them, which is also why — unlike
    /// the ticket's sketch — it needs neither `mine` nor a [`MineFrame`]:
    /// everything it reads is already resolved onto the cursor by
    /// [`settle`](Self::settle), the only place either is consulted.
    pub fn next_slice(&self) -> Option<Slice> {
        match &self.phase {
            Phase::Sinking { .. } => Some(Slice::Sink),
            Phase::Mining(cursor) => match &cursor.step {
                RowStep::Secondary => {
                    Some(Slice::Secondary { arm: cursor.arm, distance: cursor.arm_reach[cursor.arm.index()] })
                }
                RowStep::Galleries { faces, next } => Some(Slice::Gallery {
                    arm: cursor.arm,
                    row: cursor.row,
                    side: *next,
                    distance: faces[next.index()].distance,
                }),
            },
            Phase::MinedOut => None,
        }
    }

    /// Applies `outcome` for the slice [`next_slice`](Self::next_slice)
    /// just named, then walks the cursor forward to the next real slice —
    /// see the module docs.
    pub fn advance(&mut self, mine: &Mine, frame: &MineFrame, slice: Slice, outcome: SliceOutcome) {
        match slice {
            Slice::Sink => match outcome {
                SliceOutcome::Dug { .. } | SliceOutcome::Void => {
                    self.bottom -= frame.level_spacing();
                }
                SliceOutcome::Bedrock | SliceOutcome::Refused => {
                    // A shaft job is a real error — 116 logs, refunds and
                    // retries next tick; nothing here changes.
                }
            },
            Slice::Secondary { arm, .. } => {
                if let Phase::Mining(cursor) = &mut self.phase {
                    let idx = arm.index();
                    match outcome {
                        SliceOutcome::Dug { .. } | SliceOutcome::Void => cursor.arm_reach[idx] += 1,
                        SliceOutcome::Bedrock | SliceOutcome::Refused => cursor.arm_closed[idx] = true,
                    }
                }
            }
            Slice::Gallery { side, .. } => {
                if let Phase::Mining(cursor) = &mut self.phase
                    && let RowStep::Galleries { faces, next } = &mut cursor.step
                {
                    let idx = side.index();
                    match outcome {
                        SliceOutcome::Dug { .. } => {
                            faces[idx].distance += 1;
                            faces[idx].void_run = 0;
                        }
                        SliceOutcome::Void => {
                            faces[idx].distance += 1;
                            faces[idx].void_run += 1;
                            if faces[idx].void_run >= mine.max_void_run {
                                faces[idx].closed = true;
                            }
                        }
                        SliceOutcome::Bedrock | SliceOutcome::Refused => {
                            faces[idx].closed = true;
                        }
                    }
                    if faces[idx].distance >= mine.gallery_length as i32 {
                        faces[idx].closed = true;
                    }
                    *next = side.opposite();
                }
            }
        }
        self.settle(mine, frame);
    }

    /// Rolls the cursor forward through every transition that doesn't
    /// require digging anything: `Secondary` done for this row/arm ->
    /// `Galleries`; both galleries closed, or the arm itself closed ->
    /// the next `(row, arm)` per `MINES_DESIGN.md`'s "row 0 north, row 0
    /// south, row 1 north, …"; every row done or every arm closed ->
    /// `Sinking`; a sink that reached its target -> `Mining` the next
    /// level, or `MinedOut` if that level would be below `min_level_y`.
    fn settle(&mut self, mine: &Mine, frame: &MineFrame) {
        loop {
            match &self.phase {
                Phase::Sinking { target } => {
                    if self.bottom != *target {
                        return;
                    }
                    match frame.level_at_floor(self.bottom) {
                        Some(level) => {
                            self.phase = Phase::Mining(LevelCursor::start(level));
                        }
                        None => return, // shouldn't happen: targets are always level floors
                    }
                }
                Phase::Mining(_) => {
                    if !self.settle_mining(mine, frame) {
                        return;
                    }
                }
                Phase::MinedOut => return,
            }
        }
    }

    /// One step of [`settle`](Self::settle)'s `Mining` handling. Returns
    /// `true` if the phase changed and `settle` should loop again, `false`
    /// once the cursor is sitting on real, unfinished work.
    fn settle_mining(&mut self, mine: &Mine, frame: &MineFrame) -> bool {
        let Phase::Mining(cursor) = &mut self.phase else { unreachable!() };
        let level = frame.level(cursor.level);

        if cursor.arm_closed[cursor.arm.index()] {
            return self.advance_arm_or_sink(mine, frame, &level);
        }

        match &mut cursor.step {
            RowStep::Secondary => {
                let reach = cursor.arm_reach[cursor.arm.index()];
                if reach >= level.row_reach(cursor.row) {
                    cursor.step = RowStep::Galleries { faces: [Face::default(), Face::default()], next: GallerySide::East };
                    true
                } else {
                    false
                }
            }
            RowStep::Galleries { faces, next } => {
                if faces.iter().all(|f| f.closed) {
                    return self.advance_arm_or_sink(mine, frame, &level);
                }
                if faces[next.index()].closed {
                    *next = next.opposite();
                }
                false
            }
        }
    }

    /// Moves the cursor to the next `(row, arm)` in the design's sequence,
    /// skipping any arm that's closed. Returns `true` (and leaves the
    /// phase as `Mining`) if there's a row left to work; otherwise starts
    /// sinking to the next level (or mines out) and returns `true` so
    /// `settle` re-evaluates the new phase.
    fn advance_arm_or_sink(&mut self, mine: &Mine, frame: &MineFrame, level: &LevelFrame) -> bool {
        let Phase::Mining(cursor) = &mut self.phase else { unreachable!() };
        let total_rows = level.rows_per_arm(mine.level_reach as i32);
        loop {
            match cursor.arm {
                Arm::North => cursor.arm = Arm::South,
                Arm::South => {
                    cursor.arm = Arm::North;
                    cursor.row += 1;
                }
            }
            if cursor.arm_closed[0] && cursor.arm_closed[1] {
                break;
            }
            if cursor.row >= total_rows {
                break;
            }
            if cursor.arm_closed[cursor.arm.index()] {
                continue;
            }
            cursor.step = RowStep::Secondary;
            return true;
        }

        let current_level = cursor.level;
        let target = frame.level_floor(current_level + 1);
        self.phase = if target < mine.min_level_y { Phase::MinedOut } else { Phase::Sinking { target } };
        true
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::IVec2;

    use crate::city::definition::ShaftAt;

    use super::super::layout::MineFrame;
    use super::*;

    fn mine_with(level_reach: u32, gallery_length: u32, min_level_y: i32, max_void_run: u32) -> Mine {
        Mine {
            shaft: ShaftAt { x: 5, z: 5 },
            shaft_size: 6,
            first_level_depth: 12,
            min_level_y,
            level_reach,
            gallery_length,
            torch_spacing: 8,
            max_void_run,
            blocks_per_minute: 60.0,
            buffer_stacks: 512,
            haul_at_stacks: Some(64),
            valuables: Vec::new(),
        }
    }

    fn frame(first_level_depth: i32) -> MineFrame {
        MineFrame { shaft_min: IVec2::new(0, 0), shaft_size: 6, floor_y: 100, first_level_depth }
    }

    #[test]
    fn a_fresh_mine_sinks_three_flights_then_starts_row_0() {
        let f = frame(12); // 12 / (6 - 2) == 3 flights
        let mine = mine_with(100, 200, 16, 6);
        let mut progress = MineProgress::new(&f);
        assert_eq!(progress.next_slice(), Some(Slice::Sink));

        for _ in 0..3 {
            let slice = progress.next_slice().unwrap();
            assert_eq!(slice, Slice::Sink);
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 10 });
        }

        assert_eq!(progress.next_slice(), Some(Slice::Secondary { arm: Arm::North, distance: 0 }));
        assert!(matches!(progress.phase, Phase::Mining(_)));
        assert_eq!(progress.bottom, f.level_floor(0));
    }

    fn dig_to_secondary(progress: &mut MineProgress, mine: &Mine, f: &MineFrame) {
        while !matches!(progress.next_slice(), Some(Slice::Secondary { .. })) {
            let slice = progress.next_slice().unwrap();
            progress.advance(mine, f, slice, SliceOutcome::Dug { cost: 1 });
        }
    }

    #[test]
    fn first_row_is_four_secondary_slices_then_alternating_galleries() {
        let f = frame(12);
        let mine = mine_with(100, 200, 16, 6);
        let mut progress = MineProgress::new(&f);
        dig_to_secondary(&mut progress, &mine, &f);

        for d in 0..4 {
            assert_eq!(progress.next_slice(), Some(Slice::Secondary { arm: Arm::North, distance: d }));
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }

        let expected = [
            Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance: 0 },
            Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::West, distance: 0 },
            Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance: 1 },
        ];
        for want in expected {
            assert_eq!(progress.next_slice(), Some(want));
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }
    }

    #[test]
    fn void_run_closes_a_face_and_the_other_keeps_going() {
        let f = frame(12);
        let mine = mine_with(100, 200, 16, 3); // max_void_run: 3
        let mut progress = MineProgress::new(&f);
        dig_to_secondary(&mut progress, &mine, &f);
        for _ in 0..4 {
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }

        // Close the East face with 3 consecutive Void outcomes.
        for _ in 0..3 {
            let slice = Slice::Gallery { arm: Arm::North, row: 0, side: GallerySide::East, distance: progress_east_distance(&progress) };
            assert_eq!(progress.next_slice(), Some(slice));
            progress.advance(&mine, &f, slice, SliceOutcome::Void);
            // Advance the alternating West slice too, with real progress,
            // so it doesn't also close.
            let slice = progress.next_slice().unwrap();
            if matches!(slice, Slice::Gallery { side: GallerySide::West, .. }) {
                progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
            }
        }

        // East is closed now; every further slice at this row is West.
        for _ in 0..3 {
            let slice = progress.next_slice().unwrap();
            assert!(matches!(slice, Slice::Gallery { side: GallerySide::West, .. }), "{slice:?}");
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }
    }

    fn progress_east_distance(progress: &MineProgress) -> i32 {
        match &progress.phase {
            Phase::Mining(cursor) => match &cursor.step {
                RowStep::Galleries { faces, .. } => faces[GallerySide::East.index()].distance,
                _ => panic!("not in galleries"),
            },
            _ => panic!("not mining"),
        }
    }

    #[test]
    fn refused_secondary_closes_the_arm_for_every_later_row() {
        let f = frame(12);
        let mine = mine_with(100, 200, 16, 6);
        let mut progress = MineProgress::new(&f);
        dig_to_secondary(&mut progress, &mine, &f);

        let slice = progress.next_slice().unwrap();
        assert_eq!(slice, Slice::Secondary { arm: Arm::North, distance: 0 });
        progress.advance(&mine, &f, slice, SliceOutcome::Refused);

        // North is closed; every remaining slice at this level is on South.
        for _ in 0..50 {
            match progress.next_slice() {
                Some(Slice::Secondary { arm, .. }) | Some(Slice::Gallery { arm, .. }) => {
                    assert_eq!(arm, Arm::South);
                }
                Some(Slice::Sink) | None => break,
            }
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }
    }

    #[test]
    fn gallery_length_reached_closes_the_face() {
        let f = frame(12);
        let mine = mine_with(100, 5, 16, 6); // gallery_length: 5
        let mut progress = MineProgress::new(&f);
        dig_to_secondary(&mut progress, &mine, &f);
        for _ in 0..4 {
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }

        // Dig alternately (East, West, East, West, ...) until East reaches
        // its full length and closes — West is one step behind and still
        // open, so the row hasn't moved on yet.
        loop {
            let Phase::Mining(cursor) = &progress.phase else { panic!("not mining") };
            let RowStep::Galleries { faces, .. } = &cursor.step else { panic!("not in galleries") };
            if faces[GallerySide::East.index()].closed {
                break;
            }
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }
        let Phase::Mining(cursor) = &progress.phase else { panic!("not mining") };
        let RowStep::Galleries { faces, .. } = &cursor.step else { panic!("not in galleries") };
        assert!(faces[GallerySide::East.index()].closed);
        assert_eq!(faces[GallerySide::East.index()].distance, 5);
    }

    #[test]
    fn a_finished_level_starts_sinking_to_the_next_and_mines_out_below_min_level_y() {
        // A tiny mine: 1 row per arm, 1-block galleries, so it finishes fast.
        let f = frame(4); // first level right after one flight
        let mine = mine_with(4, 1, f.level_floor(0) - 4, 6); // min_level_y == level 1's floor: exactly one more level allowed
        let mut progress = MineProgress::new(&f);

        // Drain every slice until MinedOut or a hard cap (guards against an
        // infinite loop if the state machine is wrong).
        for _ in 0..2000 {
            let Some(slice) = progress.next_slice() else { break };
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }
        assert!(matches!(progress.phase, Phase::MinedOut));
        assert_eq!(progress.next_slice(), None);
    }

    #[test]
    fn round_trips_through_ron() {
        let f = frame(12);
        let mine = mine_with(100, 200, 16, 6);
        let mut progress = MineProgress::new(&f);
        for _ in 0..10 {
            let slice = progress.next_slice().unwrap();
            progress.advance(&mine, &f, slice, SliceOutcome::Dug { cost: 1 });
        }
        let text = ron::to_string(&progress).expect("serialize");
        let back: MineProgress = ron::from_str(&text).expect("deserialize");
        assert_eq!(progress, back);
    }
}
