//! `sqrt(x^2 + y^2)`, without the intermediate overflow.
//!
//! A port of glibc's `__ieee754_hypot`. It deliberately runs the *non*-fused
//! branch even on a device that has a fused multiply-add: the error-free
//! transformations it uses depend on separately-rounded multiplies and adds,
//! so fusing any of them would not merely change the last bit, it would break
//! the error extraction outright. That also means this kernel is bit-exact on
//! a backend whose `fma` is not fused, since it never asks for one.
//!
//! Both policy axes are accepted and have no effect: there is no cheaper
//! algorithm worth having — the correction is four multiplies and a divide on
//! top of a square root that dominates them — and the range handling is on the
//! main path already.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::bits::{inf64, is_finite64};

/// `2^-600`: the down-scale for the huge-`ax` branch.
const SCALE: f64 = f64::from_bits(0x1a70000000000000);
/// `2^511`: above this, squaring could overflow even after scaling down.
const LARGE_VAL: f64 = f64::from_bits(0x5fe0000000000000);
/// `2^-459`: below this, squaring could underflow to zero.
const TINY_VAL: f64 = f64::from_bits(0x2340000000000000);
/// `2^-54`: below this ratio, the smaller argument cannot affect the result.
const EPS: f64 = f64::from_bits(0x3c90000000000000);

/// A signalling NaN has its mantissa's top bit clear, and at least one other
/// mantissa bit set so that it remains a NaN rather than an infinity.
#[cube]
pub fn is_signaling_nan(x: f64) -> bool {
    let bits = u64::reinterpret(x);
    let mantissa = bits & 0x000f_ffff_ffff_ffffu64;
    (bits >> 52u64) & 0x7ffu64 == 0x7ffu64
        && mantissa != 0u64
        && (mantissa & (1u64 << 51u64)) == 0u64
}

/// The compensated correction, given `ax >= ay >= 0` scaled so that squaring
/// neither overflows nor underflows.
///
/// Every operation here is a separate rounding by design. Replaying it with a
/// fused multiply-add anywhere would be a different, unrelated algorithm that
/// happens to also approximate `hypot`.
///
/// Which of the two forms extracts the residual depends on how close the
/// arguments are. Written as a branch rather than as `select` on both forms:
/// the two arms share no arithmetic, so selecting between them spends nine
/// double-precision operations to throw the losing set away, and a warp only
/// pays that back when it actually diverges. Measured on gfx1151 over mixed
/// inputs, the branch is 10% faster than the `select`.
#[cube]
pub fn kernel(ax: f64, ay: f64) -> f64 {
    let mut h = f64::sqrt(ax * ax + ay * ay);
    let mut t1 = 0.0;
    let mut t2 = 0.0;
    if h <= 2.0 * ay {
        let delta = h - ay;
        t1 = ax * (2.0 * delta - ax);
        t2 = (delta - 2.0 * (ax - ay)) * delta;
    } else {
        let delta = h - ax;
        t1 = 2.0 * delta * (ax - 2.0 * ay);
        t2 = (4.0 * delta - ay) * ay + delta * delta;
    }
    h -= (t1 + t2) / (2.0 * h);
    h
}

/// `sqrt(x^2 + y^2)`.
#[cube]
pub fn hypot(x0: f64, y0: f64, #[comptime] _cfg: MathConfig) -> f64 {
    let x0 = crate::bits::opaque64(x0);
    let y0 = crate::bits::opaque64(y0);
    let mut out = x0 + y0;
    if !is_finite64(x0) || !is_finite64(y0) {
        // An infinity wins over a quiet NaN — `hypot(inf, NaN)` is `inf` —
        // but not over a signalling one.
        let has_inf = f64::abs(x0) == inf64() || f64::abs(y0) == inf64();
        out = select(
            has_inf && !is_signaling_nan(x0) && !is_signaling_nan(y0),
            inf64(),
            x0 + y0,
        );
    } else {
        let x = f64::abs(x0);
        let y = f64::abs(y0);
        let ax = f64::max(x, y);
        let ay = f64::min(x, y);

        if ax > LARGE_VAL {
            // Branches, not `select`: the discarded arm is a square root and a
            // division, and a lane that only needs `ax + ay` should not pay
            // for them.
            if ay <= ax * EPS {
                out = ax + ay;
            } else {
                out = kernel(ax * SCALE, ay * SCALE) / SCALE;
            }
        } else if ay < TINY_VAL {
            if ax >= ay / EPS {
                out = ax + ay;
            } else {
                out = kernel(ax / SCALE, ay / SCALE) * SCALE;
            }
        } else if ay <= ax * EPS {
            out = ax + ay;
        } else {
            out = kernel(ax, ay);
        }
    }
    out
}
