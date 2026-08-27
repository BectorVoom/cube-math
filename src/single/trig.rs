//! `sinf`, `cosf` and `sincosf`.
//!
//! A port of glibc's `__sinf` and `__cosf` (`sysdeps/ieee754/flt-32/s_sinf.c`
//! and `s_cosf.c` — ARM's optimized-routines). Worst-case error 0.5607 ulp,
//! which is upstream's own figure and the reason these had to be *ported*
//! rather than widened: the platform's `sinf` is not correctly rounded, so
//! evaluating the double-precision `sin` and rounding once lands somewhere
//! else on roughly one input in a hundred. Measured, not assumed — that is
//! what [`crate::single::wide`]'s route was doing before this file existed.
//!
//! # Three reductions
//!
//! * `|x| < pi/4` — none at all; the polynomial takes the argument directly.
//! * `|x| < 120` — one multiply-subtract against a `pi/2` that is accurate to
//!   55 bits, which is enough because the worst cancellation happens at
//!   `6 pi/4`.
//! * everything else — a 32x96-to-128-bit integer multiply against `4/pi`
//!   held to 192 bits, which gives an exact fixed-point residue. This is the
//!   single-precision counterpart of [`crate::double::branred`], and it is
//!   very much cheaper: `f32` has 24 significand bits, so 192 bits of `4/pi`
//!   cover the whole exponent range and the reduction is integer arithmetic
//!   with no floating-point error to track at all.
//!
//! Both policy axes are accepted and have no effect.

use cubecl::prelude::*;

use crate::bits::opaque32;
use crate::config::MathConfig;
use crate::tables::single::bessel as bt;
use crate::tables::single::trig as t;

/// The top 12 bits of a `f32`, sign cleared.
#[cube]
pub fn abstop12(x: f32) -> u32 {
    (u32::reinterpret(x) >> 20u32) & 0x7ffu32
}

/// `abstop12(pi/4)`.
const TOP_PIO4: u32 = 0x3f4;
/// `abstop12(2^-12)`, below which `sin x` is `x` and `cos x` is `1`.
const TOP_TINY: u32 = 0x398;
/// `abstop12(120)`, the fast reduction's ceiling.
const TOP_BIG: u32 = 0x42f;
/// `abstop12(inf)`.
const TOP_INF: u32 = 0x7f8;

/// The sine quadrant signs, `{1, -1, -1, 1}`, as an expression rather than a
/// table: a four-entry gather is more machinery than the parity it encodes.
#[cube]
pub fn quadrant_sign(n: u32) -> f64 {
    select((n & 3u32) == 1u32 || (n & 3u32) == 2u32, -1.0, 1.0)
}

/// The fast reduction: `x` modulo `pi/2` in `[-pi/4, pi/4]`, plus the
/// quadrant.
///
/// `hpi_inv` is pre-scaled by `2^24`, so the quadrant falls out of bits 24..31
/// of a truncating conversion — which is what avoids the inaccuracy a plain
/// truncation of a negative value would introduce.
#[cube]
pub fn reduce_fast(x: f64) -> (f64, u32) {
    let r = x * t::HPI_INV;
    let n = (i32::cast_from(r) + 0x0080_0000i32) >> 24i32;
    (x - f64::cast_from(n) * t::HPI, u32::reinterpret(n))
}

/// The large reduction: `x` modulo `pi/2`, by an exact fixed-point multiply
/// against `4/pi`.
///
/// `xi` is the argument's bit pattern and must have magnitude at least `2`;
/// the sign bit is ignored and the caller folds it back in through the
/// quadrant. The residue can have at most 29 leading zeros after the binary
/// point, so the `f64` it comes back as is good to 33 bits — comfortably more
/// than the 24 the answer needs.
#[cube]
pub fn reduce_large(xi0: u32) -> (f64, u32) {
    let tab = Array::<u32>::from_data(comptime!(bt::INV_PIO4.to_vec()));
    let base = usize::cast_from((xi0 >> 26u32) & 15u32);
    let shift = (xi0 >> 23u32) & 7u32;

    let xi = ((xi0 & 0x00ff_ffffu32) | 0x0080_0000u32) << shift;

    // The first product is deliberately a *32-bit* multiply, as upstream's own
    // integer promotion makes it: the bits it drops are above the window this
    // reduction keeps.
    let res0a = u64::cast_from(xi * tab[base]);
    let res1 = u64::cast_from(xi) * u64::cast_from(tab[base + 4]);
    let res2 = u64::cast_from(xi) * u64::cast_from(tab[base + 8]);
    let res0 = ((res2 >> 32u64) | (res0a << 32u64)) + res1;

    let n = (res0 + (1u64 << 61u64)) >> 62u64;
    let rem = res0 - (n << 62u64);
    (f64::cast_from(i64::reinterpret(rem)) * t::PI63, u32::cast_from(n))
}

/// The shared polynomial. `cos_branch` picks the cosine coefficients over the
/// sine ones; `neg` applies the negation upstream gets from its second table
/// entry, which only ever touches the cosine side.
#[cube]
pub fn sinf_poly(x: f64, x2: f64, cos_branch: bool, neg: bool) -> f32 {
    let mut out = 0.0 + x;
    if cos_branch {
        let x4 = x2 * x2;
        let c2 = t::C3 + x2 * t::C4;
        let c1 = t::C0 + x2 * t::C1;
        let x6 = x4 * x2;
        let c = c1 + x4 * t::C2;
        out = c + x6 * c2;
        if neg {
            // Upstream reaches this by evaluating a table row whose cosine
            // coefficients are already negated. Negation is exact, so the
            // whole polynomial negates with them and this is the same value.
            out = -out;
        }
    } else {
        let x3 = x * x2;
        let s1 = t::S2 + x2 * t::S3;
        let x7 = x3 * x2;
        let s = x + x3 * t::S1;
        out = s + x7 * s1;
    }
    f32::cast_from(out)
}

/// `sin(x)` and `cos(x)`, sharing everything but the last branch.
///
/// Returned together because the two functions differ only in which parity
/// they hand the polynomial — `cos` asks for `n ^ 1` where `sin` asks for `n`
/// — and every step before that is identical. `sincos` therefore costs one
/// reduction rather than two, and `sin` and `cos` each drop half the tail.
#[cube]
pub fn both(y: f32) -> (f32, f32) {
    let x = f64::cast_from(y);
    let a = abstop12(y);

    let mut sn = y;
    let mut cs = 1.0;
    if a < TOP_PIO4 {
        let x2 = x * x;
        sn = sinf_poly(x, x2, false, false);
        cs = sinf_poly(x, x2, true, false);
        if a < TOP_TINY {
            sn = y;
            cs = 1.0;
        }
    } else if a < TOP_BIG {
        let (r, n) = reduce_fast(x);
        let s = quadrant_sign(n);
        let xs = r * s;
        let neg = (n & 2u32) != 0u32;
        sn = sinf_poly(xs, xs * xs, (n & 1u32) != 0u32, neg);
        cs = sinf_poly(xs, xs * xs, ((n ^ 1u32) & 1u32) != 0u32, neg);
    } else if a < TOP_INF {
        let xi = u32::reinterpret(y);
        let sign = xi >> 31u32;
        let (r, n) = reduce_large(xi);
        // The original sign rides in the quadrant rather than in the argument:
        // `reduce_large` ignores it.
        let s = quadrant_sign(n + sign);
        let xs = r * s;
        let neg = ((n + sign) & 2u32) != 0u32;
        sn = sinf_poly(xs, xs * xs, (n & 1u32) != 0u32, neg);
        cs = sinf_poly(xs, xs * xs, ((n ^ 1u32) & 1u32) != 0u32, neg);
    } else {
        // An infinity or a NaN. `__math_invalidf`'s own value, spelled as the
        // runtime division that produces this hardware's default NaN.
        sn = (y - y) / (y - y);
        cs = sn;
    }
    (sn, cs)
}

/// `sin(x)`. The argument is in radians.
#[cube]
pub fn sin(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    let (s, _c) = both(opaque32(x));
    comptime!(cfg);
    s
}

/// `cos(x)`. The argument is in radians.
#[cube]
pub fn cos(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    let (_s, c) = both(opaque32(x));
    comptime!(cfg);
    c
}

/// `(sin(x), cos(x))`.
///
/// Bit-for-bit the pair `sin` and `cos` return separately: glibc's own
/// `sincosf_poly` and `sinf_poly` evaluate the same expressions in the same
/// order, differing only in which of the two they keep.
#[cube]
pub fn sincos(x: f32, #[comptime] cfg: MathConfig) -> (f32, f32) {
    comptime!(cfg);
    both(opaque32(x))
}
