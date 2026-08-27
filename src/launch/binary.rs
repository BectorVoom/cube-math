//! Two arguments in, one or two out.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::error::MathError;
use crate::launch::unary::arg;
use crate::launch::{F32, F64, geometry, same_len};
use crate::{double as d, single as s};

/// The two-argument functions.
#[derive_cube_comptime]
pub enum Binary {
    /// `x^y`.
    Pow,
    /// `sqrt(x^2 + y^2)`, without the intermediate overflow.
    Hypot,
    /// The angle of `(x, y)` from the positive `x` axis, in radians.
    Atan2,
    /// Bessel function of the first kind, order `x` — an integer in a float
    /// lane — evaluated at `y`.
    Jn,
    /// Bessel function of the second kind, order `x`, evaluated at `y`.
    Yn,
    /// The magnitude of `x` with the sign of `y`.
    CopySign,
    /// `x - y` if `x > y`, and `+0` otherwise.
    Fdim,
    /// The larger, ignoring NaN.
    Fmax,
    /// The smaller, ignoring NaN.
    Fmin,
    /// `x * 2^y`.
    Ldexp,
    /// `x * 2^y`, under its other name.
    Scalbn,
    /// `x` reduced modulo `y`, with the sign of `x`.
    Fmod,
    /// The IEEE-754 remainder.
    Remainder,
    /// The next representable value after `x` towards `y`.
    NextAfter,
}

impl Binary {
    /// The name, for an error message.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pow => "pow",
            Self::Hypot => "hypot",
            Self::Atan2 => "atan2",
            Self::Jn => "jn",
            Self::Yn => "yn",
            Self::CopySign => "copysign",
            Self::Fdim => "fdim",
            Self::Fmax => "fmax",
            Self::Fmin => "fmin",
            Self::Ldexp => "ldexp",
            Self::Scalbn => "scalbn",
            Self::Fmod => "fmod",
            Self::Remainder => "remainder",
            Self::NextAfter => "nextafter",
        }
    }

    /// Whether single precision has this one. See [`crate::launch::Unary::has_f32`].
    pub const fn has_f32(self) -> bool {
        true
    }
}

/// The two-argument functions that return two values.
#[derive_cube_comptime]
pub enum BinaryPair {
    /// `x` modulo `y`, and the low bits of the quotient.
    Remquo,
}

impl BinaryPair {
    /// The name, for an error message.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Remquo => "remquo",
        }
    }
}

#[cube(launch_unchecked)]
fn kernel_f64(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>, #[comptime] op: Binary, #[comptime] cfg: MathConfig) {
    if ABSOLUTE_POS < out.len() {
        let x = a[ABSOLUTE_POS];
        let y = b[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = match op {
            Binary::Pow => d::pow::pow(x, y, cfg),
            Binary::Hypot => d::hypot::hypot(x, y, cfg),
            Binary::Atan2 => d::invtrig::atan2(x, y, cfg),
            Binary::Jn => d::bessel::jn(x, y, cfg),
            Binary::Yn => d::bessel::yn(x, y, cfg),
            Binary::CopySign => d::exact::copysign(x, y),
            Binary::Fdim => d::exact::fdim(x, y, cfg),
            Binary::Fmax => d::exact::fmax(x, y, cfg),
            Binary::Fmin => d::exact::fmin(x, y, cfg),
            Binary::Ldexp => d::exact::ldexp(x, y, cfg),
            Binary::Scalbn => d::exact::scalbn(x, y, cfg),
            Binary::Fmod => d::exact::fmod(x, y, cfg),
            Binary::Remainder => d::exact::remainder(x, y, cfg),
            Binary::NextAfter => d::exact::nextafter(x, y, cfg),
        };
    }
}

#[cube(launch_unchecked)]
fn kernel_f32(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>, #[comptime] op: Binary, #[comptime] cfg: MathConfig) {
    if ABSOLUTE_POS < out.len() {
        let x = a[ABSOLUTE_POS];
        let y = b[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = match op {
            Binary::CopySign => s::exact::copysign(x, y),
            Binary::Fdim => s::exact::fdim(x, y, cfg),
            Binary::Fmax => s::exact::fmax(x, y, cfg),
            Binary::Fmin => s::exact::fmin(x, y, cfg),
            Binary::Ldexp => s::exact::ldexp(x, y, cfg),
            Binary::Scalbn => s::exact::scalbn(x, y, cfg),
            Binary::Fmod => s::exact::fmod(x, y, cfg),
            Binary::Remainder => s::exact::remainder(x, y, cfg),
            Binary::NextAfter => s::exact::nextafter(x, y, cfg),
            Binary::Pow => s::wide::pow(x, y, cfg),
            Binary::Hypot => s::wide::hypot(x, y, cfg),
            Binary::Atan2 => s::wide::atan2(x, y, cfg),
            Binary::Jn => s::wide::jn(x, y, cfg),
            Binary::Yn => s::wide::yn(x, y, cfg),
        };
    }
}

#[cube(launch_unchecked)]
fn kernel_pair_f64(a: &Array<f64>, b: &Array<f64>, o1: &mut Array<f64>, o2: &mut Array<f64>, #[comptime] op: BinaryPair, #[comptime] cfg: MathConfig) {
    if ABSOLUTE_POS < o1.len() {
        let (p, q) = match op {
            BinaryPair::Remquo => d::exact::remquo(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg),
        };
        o1[ABSOLUTE_POS] = p;
        o2[ABSOLUTE_POS] = q;
    }
}

#[cube(launch_unchecked)]
fn kernel_pair_f32(a: &Array<f32>, b: &Array<f32>, o1: &mut Array<f32>, o2: &mut Array<f32>, #[comptime] op: BinaryPair, #[comptime] cfg: MathConfig) {
    if ABSOLUTE_POS < o1.len() {
        let (p, q) = match op {
            BinaryPair::Remquo => s::exact::remquo(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg),
        };
        o1[ABSOLUTE_POS] = p;
        o2[ABSOLUTE_POS] = q;
    }
}

/// Evaluate `op` over `a` and `b`, writing into `output`.
pub fn binary<R: Runtime>(
    client: &ComputeClient<R>,
    op: Binary,
    a: TensorBinding<R>,
    b: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
    config: MathConfig,
) -> Result<(), MathError> {
    let n = a.size();
    same_len(n, b.size())?;
    same_len(n, output.size())?;
    let (count, dim) = geometry(n);
    let (ah, bh, oh) = (a.handle, b.handle, output.handle);
    match dtype {
        s if s == F64 => unsafe {
            kernel_f64::launch_unchecked::<R>(
                client, count, dim, arg(ah, n), arg(bh, n), arg(oh, n), op, config,
            )
        },
        s if s == F32 => {
            if !op.has_f32() {
                return Err(MathError::UnsupportedOp { op: op.name(), dtype });
            }
            unsafe {
                kernel_f32::launch_unchecked::<R>(
                    client, count, dim, arg(ah, n), arg(bh, n), arg(oh, n), op, config,
                )
            }
        }
        other => return Err(MathError::UnsupportedDtype(other)),
    }
    Ok(())
}

/// Evaluate a two-output `op` over `a` and `b`.
#[allow(clippy::too_many_arguments)] // two inputs, two outputs, a dtype and a config
pub fn binary_pair<R: Runtime>(
    client: &ComputeClient<R>,
    op: BinaryPair,
    a: TensorBinding<R>,
    b: TensorBinding<R>,
    out1: TensorBinding<R>,
    out2: TensorBinding<R>,
    dtype: StorageType,
    config: MathConfig,
) -> Result<(), MathError> {
    let n = a.size();
    same_len(n, b.size())?;
    same_len(n, out1.size())?;
    same_len(n, out2.size())?;
    let (count, dim) = geometry(n);
    let (ah, bh, o1, o2) = (a.handle, b.handle, out1.handle, out2.handle);
    match dtype {
        s if s == F64 => unsafe {
            kernel_pair_f64::launch_unchecked::<R>(
                client, count, dim, arg(ah, n), arg(bh, n), arg(o1, n), arg(o2, n), op, config,
            )
        },
        s if s == F32 => unsafe {
            kernel_pair_f32::launch_unchecked::<R>(
                client, count, dim, arg(ah, n), arg(bh, n), arg(o1, n), arg(o2, n), op, config,
            )
        },
        other => return Err(MathError::UnsupportedDtype(other)),
    }
    Ok(())
}
