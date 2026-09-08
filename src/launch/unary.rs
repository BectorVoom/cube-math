//! One argument in, one or two out.

use cubecl::prelude::*;
use cubecl::server::Binding;

use crate::config::MathConfig;
use crate::error::MathError;
use crate::launch::{CpuShape, F32, F64, launch_1d, same_len};
use crate::{double as d, single as s};

/// The one-argument functions.
///
/// A comptime value, so the `match` in the kernel below is resolved during
/// expansion: asking for [`Unary::Exp`] compiles a kernel that contains `exp`
/// and nothing else, and the choice is part of the kernel's identity rather
/// than a branch in it.
#[derive_cube_comptime]
pub enum Unary {
    /// `e^x`.
    Exp,
    /// `2^x`.
    Exp2,
    /// `10^x`.
    Exp10,
    /// `e^x - 1`.
    Expm1,
    /// Natural logarithm.
    Ln,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    /// `ln(1 + x)`.
    Log1p,
    /// Cube root.
    Cbrt,
    /// Arc sine, in radians.
    Asin,
    /// Arc cosine, in radians.
    Acos,
    /// Arc tangent, in radians.
    Atan,
    /// Sine. The argument is in radians.
    Sin,
    /// Cosine. The argument is in radians.
    Cos,
    /// Tangent. The argument is in radians.
    Tan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic cosine.
    Acosh,
    /// Inverse hyperbolic sine.
    Asinh,
    /// Inverse hyperbolic tangent.
    Atanh,
    /// The error function.
    Erf,
    /// The complementary error function, `1 - erf(x)` without the cancellation.
    Erfc,
    /// Bessel function of the first kind, order 0.
    J0,
    /// Bessel function of the first kind, order 1.
    J1,
    /// Bessel function of the second kind, order 0.
    Y0,
    /// Bessel function of the second kind, order 1.
    Y1,
    /// `ln|Gamma(x)|`.
    LGamma,
    /// The Gamma function.
    TGamma,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Largest integer not greater than `x`.
    Floor,
    /// Smallest integer not less than `x`.
    Ceil,
    /// `x` truncated towards zero.
    Trunc,
    /// Nearest integer, ties away from zero.
    Round,
    /// Nearest integer, ties to even.
    Rint,
    /// The binary exponent, as an integer in a float lane.
    Ilogb,
}

impl Unary {
    /// The name, for an error message.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Exp10 => "exp10",
            Self::Expm1 => "expm1",
            Self::Ln => "ln",
            Self::Log2 => "log2",
            Self::Log10 => "log10",
            Self::Log1p => "log1p",
            Self::Cbrt => "cbrt",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Acosh => "acosh",
            Self::Asinh => "asinh",
            Self::Atanh => "atanh",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::J0 => "j0",
            Self::J1 => "j1",
            Self::Y0 => "y0",
            Self::Y1 => "y1",
            Self::LGamma => "lgamma",
            Self::TGamma => "tgamma",
            Self::Sqrt => "sqrt",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Round => "round",
            Self::Rint => "rint",
            Self::Ilogb => "ilogb",
        }
    }

    /// Whether single precision has this one.
    ///
    /// Everything does now. What differs is *how*: the five that are genuine
    /// schedule ports (`exp`, `exp2`, `exp10`, `ln`, `log2`) are bit-exact
    /// unconditionally, and the rest are computed in double precision and
    /// rounded once — correctly rounded, with the measured caveat
    /// [`crate::single::wide`] states.
    ///
    /// Kept as a method rather than deleted because the launch side still has
    /// to answer the question, and because the answer was not always `true`.
    pub const fn has_f32(self) -> bool {
        true
    }

    /// What the CPU runtime makes of this one. See [`crate::launch::CpuShape`]
    /// for the mechanism and the measurements; the split below is the measured
    /// one, taken at 262144 `f64` elements on sixteen logical cores.
    ///
    /// Note where it does *not* fall. `erf` and `atan` cost about the same and
    /// sit on opposite sides; so do `cbrt` and `sqrt`. What separates them is
    /// branches, not work.
    pub(crate) const fn cpu_shape(self) -> CpuShape {
        match self {
            // Table, polynomial, or pure bit manipulation.
            Self::Exp
            | Self::Exp2
            | Self::Exp10
            | Self::Expm1
            | Self::Ln
            | Self::Log2
            | Self::Log10
            | Self::Asin
            | Self::Acos
            | Self::Atan
            | Self::Cosh
            | Self::Atanh
            | Self::Sqrt
            | Self::Abs
            | Self::Floor
            | Self::Ceil
            | Self::Trunc
            | Self::Round
            | Self::Rint
            | Self::Ilogb => CpuShape::Vectorised,
            // Cased argument reduction, a classification tree, or a loop.
            // `sin`, `cos` and `tan` are here for the last of those: their
            // Payne-Hanek reduction is unbounded, and it is what any argument
            // large enough to need it runs into. On a narrow domain they would
            // rather be above — but that costs 15% when the guess is wrong,
            // where guessing the other way costs 2.2x (267 against 610 Melem/s
            // over -700..700).
            Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Log1p
            | Self::Cbrt
            | Self::Sinh
            | Self::Tanh
            | Self::Acosh
            | Self::Asinh
            | Self::Erf
            | Self::Erfc
            | Self::J0
            | Self::J1
            | Self::Y0
            | Self::Y1
            | Self::LGamma
            | Self::TGamma => CpuShape::Threaded,
        }
    }
}

/// The one-argument functions that return two values.
#[derive_cube_comptime]
pub enum UnaryPair {
    /// Split into a significand in `[0.5, 1)` and a power of two.
    Frexp,
    /// Split into fractional and integral parts.
    Modf,
    /// Sine and cosine of the same argument.
    SinCos,
    /// `ln|Gamma(x)|` together with the sign of `Gamma(x)`.
    LGammaR,
}

impl UnaryPair {
    /// The name, for an error message.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Frexp => "frexp",
            Self::Modf => "modf",
            Self::SinCos => "sincos",
            Self::LGammaR => "lgamma_r",
        }
    }

    /// Whether single precision has this one. See [`Unary::has_f32`].
    pub const fn has_f32(self) -> bool {
        true
    }

    /// What the CPU runtime makes of this one. See [`Unary::cpu_shape`].
    pub(crate) const fn cpu_shape(self) -> CpuShape {
        match self {
            Self::Frexp | Self::Modf => CpuShape::Vectorised,
            Self::SinCos | Self::LGammaR => CpuShape::Threaded,
        }
    }
}

#[cube(launch_unchecked)]
fn kernel_f64(
    input: &Array<f64>,
    output: &mut Array<f64>,
    #[comptime] op: Unary,
    #[comptime] cfg: MathConfig,
) {
    if ABSOLUTE_POS < input.len() {
        let x = input[ABSOLUTE_POS];
        output[ABSOLUTE_POS] = match op {
            Unary::Exp => d::exp::exp(x, cfg),
            Unary::Exp2 => d::exp2::exp2(x, cfg),
            Unary::Exp10 => d::exp10::exp10(x, cfg),
            Unary::Expm1 => d::expm1::expm1(x, cfg),
            Unary::Ln => d::ln::ln(x, cfg),
            Unary::Log2 => d::logx::log2(x, cfg),
            Unary::Log10 => d::logx::log10(x, cfg),
            Unary::Log1p => d::log1p::log1p(x, cfg),
            Unary::Cbrt => d::cbrt::cbrt(x, cfg),
            Unary::Asin => d::invtrig::asin(x, cfg),
            Unary::Acos => d::invtrig::acos(x, cfg),
            Unary::Atan => d::invtrig::atan(x, cfg),
            Unary::Sin => d::trig::sin(x, cfg),
            Unary::Cos => d::trig::cos(x, cfg),
            Unary::Tan => d::trig::tan(x, cfg),
            Unary::Sinh => d::hyper::sinh(x, cfg),
            Unary::Cosh => d::hyper::cosh(x, cfg),
            Unary::Tanh => d::hyper::tanh(x, cfg),
            Unary::Acosh => d::acosh::acosh(x, cfg),
            Unary::Asinh => d::asinh::asinh(x, cfg),
            Unary::Atanh => d::hyper::atanh(x, cfg),
            Unary::Erf => d::erf::erf(x, cfg),
            Unary::Erfc => d::erfc::erfc(x, cfg),
            Unary::J0 => d::bessel::j0(x, cfg),
            Unary::J1 => d::bessel::j1(x, cfg),
            Unary::Y0 => d::bessel::y0(x, cfg),
            Unary::Y1 => d::bessel::y1(x, cfg),
            Unary::LGamma => d::gamma::lgamma(x, cfg),
            Unary::TGamma => d::gamma::tgamma(x, cfg),
            Unary::Sqrt => d::exact::sqrt(x, cfg),
            Unary::Abs => d::exact::abs(x, cfg),
            Unary::Floor => d::exact::floor(x, cfg),
            Unary::Ceil => d::exact::ceil(x, cfg),
            Unary::Trunc => d::exact::trunc(x, cfg),
            Unary::Round => d::exact::round(x, cfg),
            Unary::Rint => d::exact::rint(x, cfg),
            Unary::Ilogb => d::exact::ilogb(x, cfg),
        };
    }
}

#[cube(launch_unchecked)]
fn kernel_f32(
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] op: Unary,
    #[comptime] cfg: MathConfig,
) {
    if ABSOLUTE_POS < input.len() {
        let x = input[ABSOLUTE_POS];
        output[ABSOLUTE_POS] = match op {
            Unary::Exp => s::exp::exp(x, cfg),
            Unary::Exp2 => s::exp::exp2(x, cfg),
            Unary::Exp10 => s::exp::exp10(x, cfg),
            Unary::Ln => s::logx::ln(x, cfg),
            Unary::Log2 => s::logx::log2(x, cfg),
            Unary::Expm1 => s::wide::expm1(x, cfg),
            Unary::Log10 => s::wide::log10(x, cfg),
            Unary::Log1p => s::wide::log1p(x, cfg),
            Unary::Cbrt => s::wide::cbrt(x, cfg),
            Unary::Asin => s::wide::asin(x, cfg),
            Unary::Acos => s::wide::acos(x, cfg),
            Unary::Atan => s::wide::atan(x, cfg),
            Unary::Sin => s::trig::sin(x, cfg),
            Unary::Cos => s::trig::cos(x, cfg),
            Unary::Tan => s::wide::tan(x, cfg),
            Unary::Sinh => s::wide::sinh(x, cfg),
            Unary::Cosh => s::wide::cosh(x, cfg),
            Unary::Tanh => s::wide::tanh(x, cfg),
            Unary::Acosh => s::wide::acosh(x, cfg),
            Unary::Asinh => s::wide::asinh(x, cfg),
            Unary::Atanh => s::wide::atanh(x, cfg),
            Unary::Erf => s::wide::erf(x, cfg),
            Unary::Erfc => s::wide::erfc(x, cfg),
            Unary::J0 => s::bessel::j0(x, cfg),
            Unary::J1 => s::bessel::j1(x, cfg),
            Unary::Y0 => s::bessel::y0(x, cfg),
            Unary::Y1 => s::bessel::y1(x, cfg),
            Unary::LGamma => s::wide::lgamma(x, cfg),
            Unary::TGamma => s::wide::tgamma(x, cfg),
            Unary::Sqrt => s::exact::sqrt(x, cfg),
            Unary::Abs => s::exact::abs(x, cfg),
            Unary::Floor => s::exact::floor(x, cfg),
            Unary::Ceil => s::exact::ceil(x, cfg),
            Unary::Trunc => s::exact::trunc(x, cfg),
            Unary::Round => s::exact::round(x, cfg),
            Unary::Rint => s::exact::rint(x, cfg),
            Unary::Ilogb => s::exact::ilogb(x, cfg),
        };
    }
}

#[cube(launch_unchecked)]
fn kernel_pair_f64(
    input: &Array<f64>,
    o1: &mut Array<f64>,
    o2: &mut Array<f64>,
    #[comptime] op: UnaryPair,
    #[comptime] cfg: MathConfig,
) {
    if ABSOLUTE_POS < input.len() {
        let x = input[ABSOLUTE_POS];
        let (a, b) = match op {
            UnaryPair::Frexp => d::exact::frexp(x, cfg),
            UnaryPair::Modf => d::exact::modf(x, cfg),
            UnaryPair::SinCos => d::trig::sincos(x, cfg),
            UnaryPair::LGammaR => d::gamma::lgamma_r(x, cfg),
        };
        o1[ABSOLUTE_POS] = a;
        o2[ABSOLUTE_POS] = b;
    }
}

#[cube(launch_unchecked)]
fn kernel_pair_f32(
    input: &Array<f32>,
    o1: &mut Array<f32>,
    o2: &mut Array<f32>,
    #[comptime] op: UnaryPair,
    #[comptime] cfg: MathConfig,
) {
    if ABSOLUTE_POS < input.len() {
        let x = input[ABSOLUTE_POS];
        let (a, b) = match op {
            UnaryPair::Frexp => s::exact::frexp(x, cfg),
            UnaryPair::Modf => s::exact::modf(x, cfg),
            UnaryPair::SinCos => s::trig::sincos(x, cfg),
            UnaryPair::LGammaR => s::wide::lgamma_r(x, cfg),
        };
        o1[ABSOLUTE_POS] = a;
        o2[ABSOLUTE_POS] = b;
    }
}

/// Evaluate `op` over `input`, writing into `output`.
///
/// `input` and `output` are read and written as flat element sequences, so any
/// contiguous layout works and the shapes only have to agree on total size.
pub fn unary<R: Runtime>(
    client: &ComputeClient<R>,
    op: Unary,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
    config: MathConfig,
) -> Result<(), MathError> {
    let n = input.size();
    same_len(n, output.size())?;
    let (count, dim) = launch_1d(client, n, op.cpu_shape());
    let (a, b) = (input.handle, output.handle);
    match dtype {
        s if s == F64 => unsafe {
            kernel_f64::launch_unchecked::<R>(client, count, dim, arg(a, n), arg(b, n), op, config)
        },
        s if s == F32 => {
            if !op.has_f32() {
                return Err(MathError::UnsupportedOp {
                    op: op.name(),
                    dtype,
                });
            }
            unsafe {
                kernel_f32::launch_unchecked::<R>(
                    client,
                    count,
                    dim,
                    arg(a, n),
                    arg(b, n),
                    op,
                    config,
                )
            }
        }
        other => return Err(MathError::UnsupportedDtype(other)),
    }
    Ok(())
}

/// Evaluate a two-output `op` over `input`.
pub fn unary_pair<R: Runtime>(
    client: &ComputeClient<R>,
    op: UnaryPair,
    input: TensorBinding<R>,
    out1: TensorBinding<R>,
    out2: TensorBinding<R>,
    dtype: StorageType,
    config: MathConfig,
) -> Result<(), MathError> {
    let n = input.size();
    same_len(n, out1.size())?;
    same_len(n, out2.size())?;
    let (count, dim) = launch_1d(client, n, op.cpu_shape());
    let (a, b, c) = (input.handle, out1.handle, out2.handle);
    match dtype {
        s if s == F64 => unsafe {
            kernel_pair_f64::launch_unchecked::<R>(
                client,
                count,
                dim,
                arg(a, n),
                arg(b, n),
                arg(c, n),
                op,
                config,
            )
        },
        s if s == F32 => {
            if !op.has_f32() {
                return Err(MathError::UnsupportedOp {
                    op: op.name(),
                    dtype,
                });
            }
            unsafe {
                kernel_pair_f32::launch_unchecked::<R>(
                    client,
                    count,
                    dim,
                    arg(a, n),
                    arg(b, n),
                    arg(c, n),
                    op,
                    config,
                )
            }
        }
        other => return Err(MathError::UnsupportedDtype(other)),
    }
    Ok(())
}

/// A binding as a flat array argument.
pub(crate) fn arg<R: Runtime>(binding: Binding, n: usize) -> ArrayArg<R> {
    // SAFETY: `n` is the element count the caller's own tensor reported.
    unsafe { ArrayArg::from_raw_parts_binding(binding, n) }
}
