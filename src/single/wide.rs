//! The single-precision functions that are computed in double precision and
//! rounded once.
//!
//! # Why this module exists, and what it does *not* claim
//!
//! Everywhere else in this crate, [`crate::Accuracy::BitExact`] is an
//! unconditional claim: the kernel replays the platform's operation schedule,
//! so it returns the platform's bits on every input, forever. **This module is
//! the one exception, and it is worth being precise about.**
//!
//! The platform's `float` routines for these functions are *correctly
//! rounded*: they return the representable `f32` nearest the true value. That
//! is a much easier target than a schedule, because it is a property of the
//! answer — evaluate the function in `f64`, where the error is some `2^-29` of
//! a single-precision ulp, and round once, and you land on the same `f32`.
//!
//! Almost always. Double rounding fails when the `f64` result lands within its
//! own error of an `f32` rounding boundary. `rmath` swept all `2^32` inputs of
//! every function here against the platform and found exactly three
//! disagreements: one for `log10f` and two for `sinhf`. Rates around one in
//! four billion — which no sampled test would ever catch, and which is why the
//! number is quoted rather than assumed.
//!
//! `rmath` responds by delegating: on a CPU, `BitExact` calls the platform's
//! own `float` routine lane by lane, which is both exact and (because the
//! platform's `float` routines are cheaper than its `double` ones) *faster*
//! than widening. Neither move is available here. A device has no `libm` to
//! delegate to, and its `f32` units are not a call away from its `f64` ones.
//!
//! So this module widens, and says so. For these functions, and only these,
//! `BitExact` means *correctly rounded*, with a measured deviation from glibc
//! of three inputs in `2^32` across the whole set. The five single-precision
//! functions that are genuine schedule ports — [`super::exp`]'s three and
//! [`super::logx`]'s two — carry the unconditional claim, because the platform
//! does *not* compute those correctly rounded and reproducing them meant
//! reproducing the schedule.
//!
//! # The other consequence
//!
//! These need `f64` to work on the device. [`crate::probe::fidelity`] reports
//! the two precisions separately for exactly this reason, and a backend with
//! usable `f32` and unusable `f64` can run [`super::exact`] and nothing else
//! here.

use cubecl::prelude::*;

use crate::bits::opaque32;
use crate::config::MathConfig;
use crate::double as d;

/// Widen, run the double-precision kernel, round once.
///
/// Spelled out per function rather than generated, because the argument
/// widening is not always the whole story — see [`lgamma_r()`], whose sign has
/// to be decided on the `f32` argument.
macro_rules! widen1 {
    ($(#[$doc:meta])* $name:ident => $($k:ident)::+) => {
        $(#[$doc])*
        #[cube]
        pub fn $name(x: f32, #[comptime] cfg: MathConfig) -> f32 {
            f32::cast_from($($k)::+(f64::cast_from(opaque32(x)), cfg))
        }
    };
}

/// [`widen1`] for a two-argument function.
macro_rules! widen2 {
    ($(#[$doc:meta])* $name:ident => $($k:ident)::+) => {
        $(#[$doc])*
        #[cube]
        pub fn $name(x: f32, y: f32, #[comptime] cfg: MathConfig) -> f32 {
            f32::cast_from($($k)::+(
                f64::cast_from(opaque32(x)),
                f64::cast_from(opaque32(y)),
                cfg,
            ))
        }
    };
}

widen1! { /// Sine. The argument is in radians.
sin => d::trig::sin }
widen1! { /// Cosine. The argument is in radians.
cos => d::trig::cos }
widen1! { /// Tangent. The argument is in radians.
tan => d::trig::tan }
widen1! { /// Arc sine, in radians.
asin => d::invtrig::asin }
widen1! { /// Arc cosine, in radians.
acos => d::invtrig::acos }
widen1! { /// Arc tangent, in radians.
atan => d::invtrig::atan }
widen1! { /// Hyperbolic sine.
sinh => d::hyper::sinh }
widen1! { /// Hyperbolic cosine.
cosh => d::hyper::cosh }
widen1! { /// Hyperbolic tangent.
tanh => d::hyper::tanh }
widen1! { /// Base-10 logarithm.
log10 => d::logx::log10 }
widen1! { /// `ln(1 + x)`.
log1p => d::log1p::log1p }
widen1! { /// `e^x - 1`.
expm1 => d::expm1::expm1 }
widen1! { /// Cube root.
cbrt => d::cbrt::cbrt }
widen1! { /// The error function.
erf => d::erf::erf }
widen1! { /// The complementary error function.
erfc => d::erfc::erfc }
widen1! { /// `ln|Gamma(x)|`.
lgamma => d::gamma::lgamma }
widen1! { /// The Gamma function.
tgamma => d::gamma::tgamma }

widen2! { /// `sqrt(x^2 + y^2)`, without the intermediate overflow.
hypot => d::hypot::hypot }

/// Inverse hyperbolic tangent.
///
/// `0.5 log1p(2x / (1 - x))`, evaluated in `f32` — which is Rust's own
/// `f32::atanh`, and therefore the reference `rmath` pins itself to. Widening
/// and calling the double-precision `atanh` gives a *different* function here:
/// the quotient and the halving happen at a different precision, and the two
/// disagree on about one input in eight.
#[cube]
pub fn atanh(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    0.5 * log1p(2.0 * x / (1.0 - x), cfg)
}

/// Sine and cosine of the same argument.
#[cube]
pub fn sincos(x: f32, #[comptime] cfg: MathConfig) -> (f32, f32) {
    let (s, c) = d::trig::sincos(f64::cast_from(opaque32(x)), cfg);
    (f32::cast_from(s), f32::cast_from(c))
}

/// The sign of `Gamma(x)` for a `f32` argument, as `+1.0` or `-1.0`.
///
/// Exact for every input, and computed on the `f32` directly rather than on
/// the widened value — because the one place the rule is a convention rather
/// than a fact, glibc draws the line at a *different* magnitude in each
/// precision. In `f64` every `|x|` past `0x1.006df1bfac84ep+1015` reports `+1`
/// whatever its sign; in `f32` the threshold is infinity itself, so every
/// finite negative pole reports `-1`. Widening first and asking the `f64`
/// question would answer the wrong one.
#[cube]
pub fn gamma_sign(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let t = u32::reinterpret(x);
    let negative = t >> 31u32 != 0u32;
    let fx = crate::single::exact::floor(x, cfg);

    let mut out = 1.0 + 0.0 * f32::abs(x);
    if t << 1u32 >= 0xff00_0000u32 {
        // Both infinities, and NaN.
        out = 1.0;
    } else if fx == x {
        // An integer: a pole for `x <= 0`, where the reported sign is just the
        // sign bit, and `+1` for every positive integer.
        out = select(x <= 0.0 && negative, -1.0, 1.0);
    } else if f32::abs(x) < 0.5 {
        // `Gamma` has no zero crossing inside `(-1/2, 1/2)`.
        out = select(negative, -1.0, 1.0);
    } else if negative {
        // `Gamma` alternates on the negative half-line; `floor(x)` names the
        // interval and its parity names the sign.
        out = select(i32::cast_from(fx) & 1i32 == 0i32, 1.0, -1.0);
    }
    out
}

/// `(ln|Gamma(x)|, sign(Gamma(x)))`.
#[cube]
pub fn lgamma_r(x: f32, #[comptime] cfg: MathConfig) -> (f32, f32) {
    (lgamma(x, cfg), gamma_sign(x, cfg))
}
