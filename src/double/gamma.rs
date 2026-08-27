//! The Gamma functions: `lgamma`, `lgamma_r` and `tgamma`.
//!
//! # Why the accuracy axis is a no-op here
//!
//! Everywhere else in this crate, [`crate::Accuracy::BitExact`] means "the
//! platform's own bits". Rust has no `f64::tgamma` or `f64::lgamma`, so for
//! Gamma there is no call the caller was already making for that to be a
//! claim *about*. `rmath` reaches the same conclusion for the same reason and
//! ships one implementation documented by its measured error; this is a port
//! of that one, so both policies run the same code, as they do for
//! [`super::cbrt`].
//!
//! # The algorithm
//!
//! Everything is reduced to `[1, 2]`, where a single minimax polynomial does
//! the work, using the recurrence `Gamma(z + 1) = z Gamma(z)` — as a sum of
//! logarithms for `lgamma`, as a running product for `tgamma`. Large arguments
//! take Stirling's asymptotic series instead, and negative ones go through
//! Euler's reflection formula first.
//!
//! Reducing to `[1, 2]` rather than up to the Stirling cutoff is what makes
//! `lgamma` usable near `x = 1` and `x = 2`. `lgamma` has zeros there, and the
//! Stirling route reaches them as a difference of two numbers near 12 — so it
//! returns something around `1e-15` where the answer is around `1e-6`, which
//! is no correct digits at all. The polynomial is fitted with those zeros
//! factored out, so it reproduces them exactly.
//!
//! # What the port changed
//!
//! `rmath` runs each recurrence a fixed number of iterations over the whole
//! vector, blending in the lanes that still need reducing, because that is
//! what a CPU vector register can do. Here each thread stops when it is done.
//! The result is the same — one `select`ed-away step is `z - 0` and
//! `acc + 0` — and the cost is the longest thread in the warp rather than the
//! longest lane in the machine.
//!
//! The one real difference is the logarithm underneath. `rmath` reaches for
//! its own table-free `ln`; this reaches for [`super::ln::fast()`], which is a
//! different series with a different fused-multiply-add schedule. Both are
//! inside two ulp, and `lgamma` sums up to seven of them, so the two crates
//! agree to a few ulp rather than bit for bit. That is the honest bound for a
//! function neither crate claims bit-exactness for.
//!
//! Measured against `rmath` over the equivalence sweep: `lgamma` within 4 ulp
//! away from its two zeros on the negative half-line, and within `9e-16`
//! absolutely at them — where the answer is reached by subtracting two
//! quantities near `1.5`, so the last ulp of either is already thousands of
//! ulp of the answer and a relative bound says nothing. `tgamma` comes out
//! *bit-identical*, because it never touches the table-free logarithm: its
//! recurrence is a product, and its Stirling branch goes through `pow`'s
//! double-double logarithm, which is this crate's port of `rmath`'s own. The
//! sign `lgamma_r` reports is exact.

use cubecl::prelude::*;

use crate::bits::{inf64, is_nan64, opaque64};
use crate::config::MathConfig;
use crate::double::exact::{copysign, rint, trunc};
use crate::double::ln;
use crate::double::pow::{pow_exp, pow_log};
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::gamma_poly_tab;
use crate::tables::double::poly as p;

/// `ln(pi)`.
const LN_PI: f64 = f64::from_bits(0x3ff250d048e7a1bd);
/// `pi`.
const PI: f64 = std::f64::consts::PI;

/// Offsets into [`gamma_poly_tab()`], in declaration order.
const SIN: u32 = 0;
/// The cosine polynomial, 8 coefficients.
const COS: u32 = 8;
/// Stirling's series, 7 coefficients.
const STIRLING: u32 = 16;
/// `lgamma` on `[1, 2]`, 25 coefficients.
const LGAMMA: u32 = 23;
/// `Gamma` on `[1, 2]`, 25 coefficients.
const GAMMA: u32 = 48;

/// Recurrence steps `lgamma` takes to reach `[1, 2]` from below its cutoff.
///
/// The cutoff is eight and each step subtracts one, so seven steps reach
/// `(0, 2]` from anything under it.
const LG_STEPS: u32 = 7;
/// Recurrence steps `tgamma` takes.
///
/// More than `lgamma` uses, because for `tgamma` a step is one multiply rather
/// than one logarithm — so extending the exactly-reduced range is nearly free,
/// and every argument it covers avoids the exponential entirely.
const TG_STEPS: u32 = 16;
/// Above this, `tgamma` has to go through `exp(lgamma(x))`.
const TG_DIRECT_LIMIT: f64 = 18.0;
/// Above this, `tgamma` overflows.
const TGAMMA_OVERFLOW: f64 = 171.624376956302725;

/// The cutoff above which glibc's `lgamma_r` reports `+1` whatever the sign of
/// `x`: `bits(x) << 1 >= HUGE` means `|x| >= 0x1.006df1bfac84ep+1015`, or a
/// NaN or an infinity.
///
/// Every `|x| >= 2^52` is an integer and therefore a pole, so the sign
/// reported there is a convention rather than a fact about `Gamma`. glibc
/// changes that convention at this one threshold — below it a negative pole
/// reports `-1`, at or above it `+1` — and matching the platform means
/// matching the quirk.
pub const HUGE_F64: u64 = 0xfeae_a9b2_4f16_a34c;

/// Horner evaluation of `c[base] + c[base+1] s + ... + c[base+n-1] s^(n-1)`.
///
/// Horner rather than Estrin, following `rmath`: these polynomials run inside
/// a recurrence that already fills the pipeline, so the shorter code is worth
/// more than the longer dependency chain costs.
#[cube]
pub fn horner(s: f64, base: u32, n: u32, #[comptime] fk: FmaKind) -> f64 {
    let tab = gamma_poly_tab();
    let mut acc = f64::reinterpret(tab[usize::cast_from(base + n - 1u32)]);
    let mut j = n - 1u32;
    while j > 0u32 {
        j = j - 1u32;
        acc = fma64(acc, s, f64::reinterpret(tab[usize::cast_from(base + j)]), fk);
    }
    acc
}

/// True for a zero, a subnormal, an infinity or a NaN.
#[cube]
pub fn not_normal(x: f64) -> bool {
    let e = (u64::reinterpret(x) >> 52u64) & 0x7ffu64;
    e == 0u64 || e == 0x7ffu64
}

/// `ln(Gamma(z))` for `z` at or above the Stirling cutoff.
#[cube]
pub fn stirling(z: f64, #[comptime] fk: FmaKind) -> f64 {
    let inv = 1.0 / z;
    let series = inv * horner(inv * inv, STIRLING, 7u32, fk);
    let lnz = ln::fast(z, false, fk);
    fma64(z - 0.5, lnz, p::HALF_LN_2PI - z, fk) + series
}

/// [`stirling()`] to more than double precision, as an unevaluated `hi + lo`.
///
/// [`stirling()`] rounds `ln(z)` to a single `f64` before ever multiplying it
/// by `z`, so what it returns already carries several ulp of error before
/// `tgamma` even exponentiates it. This carries the logarithm in double-double
/// throughout, through [`super::pow::pow_log()`] — already exercised by `pow`'s
/// own bit-exactness tests, rather than a second unvalidated implementation —
/// so that `hi + lo` compresses back to the correctly rounded `f64` nearest
/// the true `ln(Gamma(z))` rather than merely a close one.
///
/// The series tail stays single-precision: over this function's domain
/// (`z >= 18`) it is already three or more orders of magnitude below the
/// leading terms, so its own rounding is far under the budget those need.
#[cube]
pub fn stirling_dd(z: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let inv = 1.0 / z;
    let series = inv * horner(inv * inv, STIRLING, 7u32, fk);

    // `z - 1/2`, exact: a Fast2Sum with `|z| >= 1/2` at every call site.
    let zm = z - 0.5;
    let zml = (z - zm) - 0.5;

    let (lnhi, lnlo) = pow_log(u64::reinterpret(z));

    // `(z - 1/2) ln z`: a 2Product on the leading pair plus both cross terms,
    // so the product carries the precision the logarithm does.
    let ph = zm * lnhi;
    let pl = fma64(zm, lnhi, -ph, fk) + fma64(zm, lnlo, zml * lnhi, fk);

    // `ln(2 pi)/2 - z`, via Fast2Sum with `-z` as the larger operand — this
    // function's domain has `z >= 18`, well past `ln(2 pi)/2`.
    let s = p::HALF_LN_2PI - z;
    let e = p::HALF_LN_2PI - (s + z);

    // `|ph| >= |s|` throughout the domain (`z ln z` outgrows `z` past `e`), so
    // this Fast2Sum's precondition holds.
    let hi = ph + s;
    let r = (ph - hi) + s;
    (hi, (r + pl + e) + series)
}

/// `sin(pi x)`, accurate for every `x`.
///
/// Reducing modulo one first is what makes this work where a plain
/// `sin(PI * x)` would not: `PI * x` rounds away the very digits the
/// reflection formula needs when `|x|` is large, and near a negative integer
/// those digits *are* the answer.
#[cube]
pub fn sin_pi(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let fk = comptime!(cfg.fma());
    let n = rint(x, cfg);
    let r = x - n; // `|r| <= 1/2`
    let a = f64::abs(r);

    // `|sin(pi r)|` from the sine polynomial for `|r| <= 1/4`, and from the
    // cosine one at `pi(1/2 - |r|)` for the rest — both keep the argument
    // inside the `|t| <= pi/4` the polynomials were fitted on.
    let t1 = PI * a;
    let t2 = PI * (0.5 - a);
    let mut mag = horner(t2 * t2, COS, 8u32, fk);
    if a <= 0.25 {
        mag = t1 * horner(t1 * t1, SIN, 8u32, fk);
    }

    // `sin(pi(n + r)) = (-1)^n sin(pi r)`.
    let signed = copysign(mag, r);
    select(n == rint(n * 0.5, cfg) * 2.0, signed, -signed)
}

// ---------------------------------------------------------------------------
// lgamma
// ---------------------------------------------------------------------------

/// `lgamma(y)` for `y > 0`.
#[cube]
pub fn lgamma_positive(y: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut out = stirling(y, fk);
    if y < p::LGAMMA_CUTOFF {
        // Walk down to `(0, 2]`, accumulating the logarithms stepped over.
        let mut z = y;
        let mut acc = 0.0 * y;
        let mut i = 0u32.runtime();
        while i < LG_STEPS {
            if z > 2.0 {
                z = z - 1.0;
                acc = acc + ln::fast(z, false, fk);
            }
            i = i + 1u32;
        }
        // One step up for `(0, 1)`, which the loop above cannot reach.
        if z < 1.0 {
            acc = acc - ln::fast(z, false, fk);
            z = z + 1.0;
        }
        // `lgamma` on `[1, 2]`, with its zeros factored out so they come back
        // exactly.
        let t = z - 1.0;
        out = t * (t - 1.0) * horner(t, LGAMMA, 25u32, fk) + acc;
    }
    out
}

/// `ln|Gamma(x)|`.
#[cube]
pub fn lgamma(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());

    // The inputs the main path must not be trusted with. Both routes below
    // reach for an unchecked `ln`, whose domain is the positive normals — the
    // recurrence takes `ln(z)` directly for `0 < z < 1`, and the reflection
    // takes `ln|sin(pi x)|`. A subnormal, a zero and a negative integer (where
    // the sine vanishes) all leave that domain, as do the infinities and NaN.
    // They have closed forms, so they are answered here rather than defended
    // against inside the loops.
    let mut out = 0.0 + x;
    if is_nan64(x) {
        out = x + x;
    } else if not_normal(x) {
        // Both infinities: `lgamma` is even enough at the ends that C requires
        // `+inf` for either. A zero of either sign is a pole. A subnormal has
        // `Gamma(x) = 1/x - gamma + O(x)`, and at that magnitude the
        // correction is more than 250 orders of magnitude below the leading
        // term, so `ln|Gamma(x)|` is `-ln|x|` rounded once.
        out = inf64();
        if x != 0.0 && f64::abs(x) != inf64() {
            out = -ln::bit_exact(f64::abs(x));
        }
    } else if x < 0.0 && x == trunc(x, cfg) {
        out = inf64(); // a pole
    } else if x < 0.0 {
        // `lgamma(x) = ln(pi) - ln|sin(pi x)| - lgamma(1 - x)`. Reflecting
        // first means everything below only ever sees a positive argument.
        out = LN_PI - ln::fast(f64::abs(sin_pi(x, cfg)), false, fk) - lgamma_positive(1.0 - x, fk);
    } else {
        out = lgamma_positive(x, fk);
    }
    out
}

/// The sign of `Gamma(x)`, as `+1.0` or `-1.0`. Exact for every input.
///
/// Not derived from [`lgamma()`]: `Gamma` alternates sign on the negative
/// half-line, so the sign follows from the parity of `floor(x)` and costs one
/// rounding and one integer test.
///
/// The conventions at the awkward points are glibc's, and they are not all
/// what the parity rule alone would give: `+0` reports `+1` and `-0` reports
/// `-1`; every negative integer — a pole, where the value is `+inf` — reports
/// `-1`, including the odd ones parity would call `+1`; `-inf` reports `+1`
/// though its sign bit is set; and NaN reports `+1`.
#[cube]
pub fn gamma_sign(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let t = u64::reinterpret(x);
    let negative = t >> 63u64 != 0u64;
    let fx = crate::double::exact::floor(x, cfg);

    let mut out = 1.0 + 0.0 * f64::abs(x);
    if t << 1u64 >= HUGE_F64 {
        // Huge magnitudes, both infinities, and NaN.
        out = 1.0;
    } else if fx == x {
        // An integer: a pole for `x <= 0`, where the reported sign is just the
        // sign bit, and `+1` for every positive integer.
        out = select(x <= 0.0 && negative, -1.0, 1.0);
    } else if f64::abs(x) < 0.5 {
        // `Gamma` has no zero crossing inside `(-1/2, 1/2)`, so the sign bit
        // decides on its own.
        out = select(negative, -1.0, 1.0);
    } else if negative {
        // `Gamma` alternates on the negative half-line: negative on `(-1, 0)`,
        // positive on `(-2, -1)`, and so on. `floor(x)` names the interval and
        // its parity names the sign.
        out = select(i64::cast_from(fx) & 1i64 == 0i64, 1.0, -1.0);
    }
    out
}

/// `(ln|Gamma(x)|, sign(Gamma(x)))`.
#[cube]
pub fn lgamma_r(x0: f64, #[comptime] cfg: MathConfig) -> (f64, f64) {
    let x = opaque64(x0);
    (lgamma(x, cfg), gamma_sign(x, cfg))
}

// ---------------------------------------------------------------------------
// tgamma
// ---------------------------------------------------------------------------

/// `Gamma(y)` for `y > 0`.
#[cube]
pub fn tgamma_positive(y: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut out = 0.0 + y;
    if y < TG_DIRECT_LIMIT {
        // The recurrence, which for `tgamma` is a multiply per step and so
        // cheap enough to run further than `lgamma`'s.
        let mut z = y;
        let mut prod = 1.0 + 0.0 * y;
        let mut i = 0u32.runtime();
        while i < TG_STEPS {
            if z > 2.0 {
                z = z - 1.0;
                prod = prod * z;
            }
            i = i + 1u32;
        }
        if z < 1.0 {
            prod = prod / z;
            z = z + 1.0;
        }
        out = horner(z - 1.0, GAMMA, 25u32, fk) * prod;
    } else {
        // Stirling in double-double, handed straight to the multi-precision
        // exponential without rounding through `ln(Gamma(y))` first.
        let (hi, lo) = stirling_dd(y, fk);
        out = pow_exp(hi, lo, 0u64);
        if y > TGAMMA_OVERFLOW {
            out = inf64();
        }
    }
    out
}

/// `Gamma(x)`.
#[cube]
pub fn tgamma(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let mut out = x + x;
    if !is_nan64(x) {
        // A NaN is the one input the arithmetic below does not carry through:
        // `stirling_dd` reads its bit pattern as an exponent and a significand
        // and the exponential comes back with a number.
        if x < 0.0 {
            // `Gamma(x) = pi / (sin(pi x) Gamma(1 - x))`. When `Gamma(1 - x)`
            // overflows the quotient is zero, which is the correct limit, so
            // the large-`|x|` case needs no test of its own.
            out = PI / (sin_pi(x, cfg) * tgamma_positive(1.0 - x, fk));
        } else {
            out = tgamma_positive(x, fk);
        }
    }
    out
}
