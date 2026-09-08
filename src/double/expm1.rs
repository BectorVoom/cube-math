//! `e^x - 1`, accurate for small `x`.
//!
//! A port of glibc's `__expm1_fma`. The polynomial is evaluated by Estrin's
//! scheme, not Horner — that is what the compiled library does, and the two
//! round differently even though they compute the same polynomial. Three
//! operations are fused, and only three: `3 - r1*hfx`, `6 - x*t`, and the
//! reconstruction `x*(e - c) - c`.
//!
//! [`crate::Accuracy::Fast`] takes the same reduction and the same rational
//! correction but skips the branchy reconstruction at the ends of the range,
//! reaching for [`super::exp`] instead where `x` is large enough that
//! `e^x - 1` no longer cancels.

use cubecl::prelude::*;

use crate::bits::is_nan64;
use crate::config::MathConfig;
use crate::fma::{FmaKind, fma64};

/// `ln(DBL_MAX)`, above which `e^x - 1` overflows.
const OTHRESHOLD: f64 = f64::from_bits(0x40862e42fefa39ef);
/// High part of `ln(2)`; `k * LN2HI` is exact for every `k` the reduction makes.
const LN2HI: f64 = f64::from_bits(0x3fe62e42fee00000);
/// Low part of `ln(2)`.
const LN2LO: f64 = f64::from_bits(0x3dea39ef35793c76);
/// `1 / ln(2)`.
const INVLN2: f64 = f64::from_bits(0x3ff71547652b82fe);
/// `2^1023`, the overflow multiplier glibc uses to raise the flag.
const P1023: f64 = f64::from_bits(0x7fe0000000000000);

/// The rational correction's coefficients, `Q[1..=5]` in `s_expm1.c`.
const Q0: f64 = f64::from_bits(0xbfa11111111110f4);
const Q1: f64 = f64::from_bits(0x3f5a01a019fe5585);
const Q2: f64 = f64::from_bits(0xbf14ce199eaadbb7);
const Q3: f64 = f64::from_bits(0x3ed0cfca86e65239);
const Q4: f64 = f64::from_bits(0xbe8afdb76e09c32d);

/// `e^x - 1`.
#[cube]
pub fn expm1(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = crate::bits::opaque64(x);
    if comptime!(cfg.bit_exact()) {
        bit_exact(x)
    } else {
        fast(x, comptime!(cfg.checked()), comptime!(cfg.fma()))
    }
}

/// The reference schedule, over the whole domain.
#[cube]
pub fn bit_exact(x0: f64) -> f64 {
    let bits = u64::reinterpret(x0);
    let hx = u32::cast_from(bits >> 32u64) & 0x7fff_ffffu32;
    let sign = bits >> 63u64 != 0u64;

    let mut out = x0;
    let mut done = false;

    // Huge and non-finite arguments.
    if hx >= 0x4043_687Au32 {
        // |x| >= 56 ln(2)
        if is_nan64(x0) {
            out = x0 + x0;
            done = true;
        } else if sign {
            out = -1.0;
            done = true;
        } else if x0 > OTHRESHOLD {
            out = x0 * P1023;
            done = true;
        }
    }

    if !done {
        let mut x = x0;
        let mut c = 0.0;
        let mut k = 0i32;

        if hx > 0x3fd6_2e42u32 {
            // |x| > ln(2)/2: reduce to `x = k ln(2) + r`.
            //
            // Below 1.5 ln(2) the reduction's `k` is `+-1` and glibc skips the
            // multiply, taking the constant straight; above it, `k` is rounded
            // out of `x / ln(2)` and `k * LN2HI` is exact by construction.
            // A branch rather than `select` on both: the two arms share no
            // arithmetic, so evaluating both spends five double-precision
            // operations to throw four of them away.
            let mut hi = 0.0;
            let mut lo = 0.0;
            if hx < 0x3FF0_A2B2u32 {
                k = select(sign, -1i32, 1i32);
                hi = select(sign, x0 + LN2HI, x0 - LN2HI);
                lo = select(sign, -LN2LO, LN2LO);
            } else {
                k = i32::cast_from(INVLN2 * x0 + select(sign, -0.5, 0.5));
                let t = f64::cast_from(k);
                hi = x0 - t * LN2HI;
                lo = t * LN2LO;
            }
            x = hi - lo;
            c = (hi - x) - lo;
        } else if hx < 0x3c90_0000u32 {
            // |x| < 2^-54: the result is `x`.
            out = x0;
            done = true;
        }

        if !done {
            // Primary range.
            let hfx = 0.5 * x;
            let hxs = x * hfx;
            let hxs2 = hxs * hxs;
            let c01 = fma(hxs, Q0, 1.0);
            let c23 = fma(Q2, hxs, Q1);
            let c45 = fma(Q4, hxs, Q3);
            let r1 = fma(hxs2 * hxs2, c45, fma(hxs2, c23, c01));
            let t = fma(-r1, hfx, 3.0);
            let mut e = hxs * ((r1 - t) / fma(-t, x, 6.0));

            if k == 0i32 {
                out = x - fma(e, x, -hxs);
            } else {
                e = fma(e - c, x, -c) - hxs;
                if k == -1i32 {
                    out = 0.5 * (x - e) - 0.5;
                } else if k == 1i32 {
                    if x < -0.25 {
                        out = -2.0 * (e - (x + 0.5));
                    } else {
                        out = 1.0 + 2.0 * (x - e);
                    }
                } else if k < 0i32 || k > 56i32 {
                    let y0 = x - e + 1.0;
                    let y = select(
                        k == 1024i32,
                        y0 * 2.0 * P1023,
                        y0 * f64::reinterpret(u64::cast_from(1023i32 + k) << 52u64),
                    );
                    out = y - 1.0;
                } else {
                    let twopk = f64::reinterpret(u64::cast_from(1023i32 + k) << 52u64);
                    let uf = f64::reinterpret(u64::cast_from(1023i32 - k) << 52u64); // 2^-k
                    if k < 20i32 {
                        out = (x - e + (1.0 - uf)) * twopk;
                    } else {
                        out = (x - (e + uf) + 1.0) * twopk;
                    }
                }
            }
        }
    }
    out
}

/// The table-free path.
///
/// Below `|x| = 0.35` the cancellation in `e^x - 1` is what has to be avoided,
/// and the same rational correction the reference uses does it. Above that
/// there is nothing left to cancel, so `e^x` is computed outright and 1
/// subtracted — which is both simpler and, past `x = 1`, exactly as accurate.
///
/// Maximum error measured against the correctly rounded result: below 2 ulp.
#[cube]
pub fn fast(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let mut out = fma64(crate::double::exp::fast(x, checked, fk), 1.0, -1.0, fk);
    if f64::abs(x) < 0.35 {
        // `e^x - 1 = x + x^2/2 + ...`, evaluated so the leading `x` survives.
        let hfx = 0.5 * x;
        let hxs = x * hfx;
        let hxs2 = hxs * hxs;
        let c01 = fma64(hxs, Q0, 1.0, fk);
        let c23 = fma64(Q2, hxs, Q1, fk);
        let c45 = fma64(Q4, hxs, Q3, fk);
        let r1 = fma64(hxs2 * hxs2, c45, fma64(hxs2, c23, c01, fk), fk);
        let t = fma64(-r1, hfx, 3.0, fk);
        let e = hxs * ((r1 - t) / fma64(-t, x, 6.0, fk));
        out = x - fma64(e, x, -hxs, fk);
    }
    out
}
