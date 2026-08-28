//! Coordinate parsing shared by every `ranvil-cli` subcommand that takes a
//! position on the command line (ticket 087). Unit-tested once here rather
//! than in every command ticket (091+) that needs it.
//!
//! Both forms parse `,`-separated integers with `str::trim` tolerance on
//! each component (so `" 1 , 2 , 3 "` is fine) but no other leniency — an
//! extra or missing component, or a non-integer component, is a
//! [`CliError::Usage`](super::error::CliError::Usage)-shaped parse failure
//! for the caller to wrap.

use std::str::FromStr;

use bevy::math::{IVec2, IVec3};

/// A block position, `"x,y,z"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockPos(pub IVec3);

/// A chunk (or region) position, `"x,z"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPos(pub IVec2);

/// An `(x, z)` **block**-coordinate pair — `column`'s target column (ticket
/// 093). Kept distinct from [`ChunkPos`] (same shape, different unit) so a
/// caller reading either type's name never has to guess which coordinate
/// space it parses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnPos(pub IVec2);

impl FromStr for BlockPos {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match components(s).as_slice() {
            [x, y, z] => Ok(BlockPos(IVec3::new(
                parse(x, "x")?,
                parse(y, "y")?,
                parse(z, "z")?,
            ))),
            _ => Err(format!("expected a block position \"x,y,z\", got \"{s}\"")),
        }
    }
}

impl FromStr for ChunkPos {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match components(s).as_slice() {
            [x, z] => Ok(ChunkPos(IVec2::new(parse(x, "x")?, parse(z, "z")?))),
            _ => Err(format!("expected a chunk position \"x,z\", got \"{s}\"")),
        }
    }
}

impl FromStr for ColumnPos {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match components(s).as_slice() {
            [x, z] => Ok(ColumnPos(IVec2::new(parse(x, "x")?, parse(z, "z")?))),
            _ => Err(format!("expected a column position \"x,z\", got \"{s}\"")),
        }
    }
}

/// Splits on `,` and trims each piece — the one place both `FromStr` impls
/// share, so a fourth format never sneaks in via only one of them.
fn components(s: &str) -> Vec<&str> {
    s.split(',').map(str::trim).collect()
}

fn parse(s: &str, axis: &str) -> Result<i32, String> {
    s.parse::<i32>()
        .map_err(|_| format!("invalid {axis} coordinate \"{s}\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_pos_parses_three_components() {
        assert_eq!("1,2,3".parse(), Ok(BlockPos(IVec3::new(1, 2, 3))));
    }

    #[test]
    fn block_pos_parses_negative_components() {
        assert_eq!(
            "-10,64,-200".parse(),
            Ok(BlockPos(IVec3::new(-10, 64, -200)))
        );
    }

    #[test]
    fn block_pos_tolerates_surrounding_whitespace() {
        assert_eq!("  1 , 2 ,3  ".parse(), Ok(BlockPos(IVec3::new(1, 2, 3))));
    }

    #[test]
    fn block_pos_rejects_missing_component() {
        assert!("1,2".parse::<BlockPos>().is_err());
    }

    #[test]
    fn block_pos_rejects_extra_component() {
        assert!("1,2,3,4".parse::<BlockPos>().is_err());
    }

    #[test]
    fn block_pos_rejects_non_integer_component() {
        assert!("1,a,3".parse::<BlockPos>().is_err());
    }

    #[test]
    fn block_pos_rejects_internal_whitespace_split() {
        // "1 2,3,4" trims to "1 2" for the x component, which isn't a bare
        // integer — this isn't the "extra component" case, it's a bad one.
        assert!("1 2,3,4".parse::<BlockPos>().is_err());
    }

    #[test]
    fn chunk_pos_parses_two_components() {
        assert_eq!("4,-5".parse(), Ok(ChunkPos(IVec2::new(4, -5))));
    }

    #[test]
    fn chunk_pos_tolerates_surrounding_whitespace() {
        assert_eq!(" 4 , -5 ".parse(), Ok(ChunkPos(IVec2::new(4, -5))));
    }

    #[test]
    fn chunk_pos_rejects_missing_component() {
        assert!("4".parse::<ChunkPos>().is_err());
    }

    #[test]
    fn chunk_pos_rejects_extra_component() {
        assert!("4,-5,6".parse::<ChunkPos>().is_err());
    }

    #[test]
    fn chunk_pos_rejects_non_integer_component() {
        assert!("4,z".parse::<ChunkPos>().is_err());
    }

    #[test]
    fn column_pos_parses_two_components() {
        assert_eq!("4,-5".parse(), Ok(ColumnPos(IVec2::new(4, -5))));
    }

    #[test]
    fn column_pos_rejects_missing_component() {
        assert!("4".parse::<ColumnPos>().is_err());
    }
}
