//! What a launch can refuse to do.

use core::fmt;

use cubecl::prelude::{LaunchError, StorageType};

/// A launch that could not be made.
#[derive(Debug)]
pub enum MathError {
    /// The element type has no kernels here. Only `f64` and `f32` do.
    UnsupportedDtype(StorageType),
    /// The function is not implemented for this element type.
    ///
    /// The single-precision transcendentals are the ones this catches: `f32`
    /// has the whole IEEE-exact family and nothing else yet.
    UnsupportedOp {
        /// The operation asked for.
        op: &'static str,
        /// The element type it was asked for in.
        dtype: StorageType,
    },
    /// Two arguments that have to agree on length do not.
    LengthMismatch {
        /// What the first argument holds.
        left: usize,
        /// What the second holds.
        right: usize,
    },
    /// The backend refused the launch.
    Launch(LaunchError),
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDtype(dtype) => {
                write!(f, "cube-math has no kernels for {dtype}; only f64 and f32")
            }
            Self::UnsupportedOp { op, dtype } => {
                write!(f, "cube-math has no `{op}` for {dtype} yet")
            }
            Self::LengthMismatch { left, right } => {
                write!(f, "arguments must have the same length, got {left} and {right}")
            }
            Self::Launch(e) => write!(f, "launch failed: {e:?}"),
        }
    }
}

impl core::error::Error for MathError {}

impl From<LaunchError> for MathError {
    fn from(e: LaunchError) -> Self {
        Self::Launch(e)
    }
}
