//! `asinh(x)`, correctly rounded.
//!
//! A port of glibc's `__asinh` (`sysdeps/ieee754/dbl-64/s_asinh.c` —
//! CORE-MATH's `cr_asinh`). Like [`super::erf`], this one's target is a
//! *property of the answer* rather than a schedule, so it is bit-exact to
//! every correctly-rounded `asinh` and not only to this platform's.
//!
//! # Why this was the last function in the crate to land
//!
//! `rmath` ports the fast path and no further. Its `BitExact` kernel computes
//! the double-double `asinh` with a proven error bound, checks whether that
//! bound settles the rounding, and hands the inputs where it does not — about
//! one in `2^17` — to the platform's own `asinh`, one lane at a time. That is
//! the right trade on a CPU and it is not available here: a device has no
//! `libm` behind it. So the whole two-tier structure had to come across, which
//! meant the accurate path as well:
//!
//! 1. **The fast path** — three bands, all in double-double, each with an
//!    error bound and a certificate (`lb == ub`).
//! 2. **[`refine()`]** — a second logarithm to 159 bits, over a four-stage
//!    `2^-k` ladder with its logarithms tabulated to triple-double.
//! 3. **A hard-case table** — 35 arguments even that cannot resolve, reached
//!    by a test on how close the refined value landed to a rounding boundary.
//!
//! Both policy axes are accepted and have no effect: correct rounding is the
//! whole point, and there is no cheaper way to it.

use cubecl::prelude::*;

use crate::bits::opaque64;
use crate::config::MathConfig;
use crate::double::dd::{fast_two_sum, mul_dd_acc, mul_ddd};
use crate::double::exact::copysign;
use crate::double::logtab as lt;
use crate::double::logtab::LOG2E;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::asinh_tab;

/// Offsets into [`asinh_tab()`].
const ZERO_CH: u32 = 0;
/// The near-zero series' plain tail, 5 slots.
const ZERO_CL: u32 = ZERO_CH + 24;
/// The fast path's four band coefficient sets, 14 slots.
const SMALL: u32 = ZERO_CL + 5;
/// The hard-case table, 35 rows of 3.
const DB: u32 = SMALL + 14;
/// Rows in the hard-case table.
const DB_ROWS: u32 = 35;

/// `2^-60`. The multiply by it is what raises underflow where the
/// platform does; nothing here observes the flag.
const TWO_M60: f64 = f64::from_bits(0x3c30000000000000);

/// `0x1.79p-53`, the near-zero band's relative error bound.
const EPS_ZERO: f64 = f64::from_bits(0x3ca7900000000000);

/// One `f64` out of a `u64` table.
#[cube]
pub fn at(tab: &Array<u64>, i: u32) -> f64 {
    f64::reinterpret(tab[usize::cast_from(i)])
}

/// `asinh(x)` for `|x| < 1/4`, to double-double, when the fast path's bound
/// could not settle the rounding.
///
/// Upstream's `as_asinh_zero`.
#[cube]
pub fn near_zero(x: f64, x2h: f64, x2l: f64, #[comptime] fk: FmaKind) -> f64 {
    let tab = asinh_tab();
    let y2a = x2h
        * (at(&tab, ZERO_CL)
            + x2h
                * (at(&tab, ZERO_CL + 1u32)
                    + x2h
                        * (at(&tab, ZERO_CL + 2u32)
                            + x2h * (at(&tab, ZERO_CL + 3u32) + x2h * at(&tab, ZERO_CL + 4u32)))));

    // `polydd` over twelve double-double coefficients, high index down.
    let (mut ch, cl0) = fast_two_sum(at(&tab, ZERO_CH + 22u32), y2a);
    let mut cl = cl0 + at(&tab, ZERO_CH + 23u32);
    let mut i = 11u32.runtime();
    while i > 0u32 {
        i = i - 1u32;
        let (mh, ml) = mul_dd_acc(x2h, x2l, ch, cl, fk);
        let c0 = at(&tab, ZERO_CH + i * 2u32);
        let th = mh + c0;
        let tl = (c0 - th) + mh;
        ch = th;
        cl = ml + tl + at(&tab, ZERO_CH + i * 2u32 + 1u32);
    }

    let (m1h, m1l) = mul_dd_acc(ch, cl, x2h, x2l, fk);
    let (y1a, y2b) = mul_ddd(m1h, m1l, x, fk);
    let (y0, y1b) = fast_two_sum(x, y1a);
    let (y1, y2) = fast_two_sum(y1b, y2b);

    // If the low half is a power of two the sum straddles a rounding boundary;
    // nudge it to the side the residual points to.
    let t = u64::reinterpret(y1);
    let mut y1f = y1;
    if t & (0xffff_ffff_ffff_ffffu64 >> 12u64) == 0u64 {
        let w = u64::reinterpret(y2);
        y1f = f64::reinterpret(select((w ^ t) >> 63u64 != 0u64, t - 1u64, t + 1u64));
    }
    y0 + y1f
}

/// The hard cases, as a trailing overwrite.
///
/// Upstream bisects; this scans, for the reason [`super::erf::erf_accurate()`]
/// gives — it is the slow path of the slow path, and a straight line with no
/// early exit is what a kernel wants.
#[cube]
pub fn database(x: f64, f: f64) -> f64 {
    let tab = asinh_tab();
    let ax = f64::abs(x);
    let mut out = f;
    for k in 0..DB_ROWS {
        let row = DB + k * 3u32;
        if at(&tab, row) == ax {
            let sgn = copysign(1.0, x);
            out = sgn * at(&tab, row + 1u32) + sgn * at(&tab, row + 2u32);
        }
    }
    out
}

/// The accurate path: a second logarithm, to 159 bits.
///
/// Upstream's `as_asinh_refine`. `a` is `log2` of the fast path's own answer,
/// which is what selects the ladder entry — the refinement is a *correction*
/// to a value it already has, not a fresh evaluation.
#[cube]
pub fn refine(x: f64, zh: f64, zl: f64, a: f64, #[comptime] fk: FmaKind) -> f64 {
    let (v0a, v1a, v2a) = lt::refine_core(zh, zl, a, false, fk);
    let s = copysign(2.0, x);
    let v0 = v0a * s;
    let v1b = v1a * s;
    let v2 = v2a * s;

    let t0b = u64::reinterpret(v1b);
    let mut v1 = v1b;
    if t0b & (0xffff_ffff_ffff_ffffu64 >> 12u64) == 0u64 {
        let w = u64::reinterpret(v2);
        v1 = f64::reinterpret(select((w ^ t0b) >> 63u64 != 0u64, t0b - 1u64, t0b + 1u64));
    }
    let t = u64::reinterpret(v1);
    let t0 = u64::reinterpret(v0);
    let er = (t + 41u64) & (0xffff_ffff_ffff_ffffu64 >> 12u64);
    let de = ((t0 >> 52u64) & 0x7ffu64) - ((t >> 52u64) & 0x7ffu64);
    let res = v0 + v1;
    let mut out = res;
    if de > 99u64 || er < 80u64 {
        out = database(x, res);
    }
    out
}

/// `asinh(x)`.
#[cube]
pub fn asinh(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let ax = f64::abs(x);
    let u = u64::reinterpret(ax);
    let tab = asinh_tab();

    let mut out = x + x;
    if u < 0x3fbb_0000_0000_0000u64 {
        // `|x| < 0x1.bp-4`.
        if u < 0x3e57_137449123ef7u64 {
            // `|x| < 0x1.7137449123ef7p-26`, where `asinh(x)` rounds to `x`.
            // The multiply is what raises underflow when it should; nothing
            // here observes the flag.
            out = fma64(TWO_M60, -x, x, fk);
            if u == 0u64 {
                out = x;
            }
        } else {
            let x2h = x * x;
            let x2l = fma64(x, x, -x2h, fk);
            let x3h = x2h * x;

            let mut sl = x3h * at(&tab, SMALL);
            if u >= 0x3f93_0000_0000_0000u64 {
                // `0x1.3p-6 <= |x| < 0x1.bp-4`: a degree-15 minimax whose
                // relative error is below `2^-63`, in Estrin form.
                let b = SMALL + 7u32;
                let c1 = at(&tab, b + 1u32) + x2h * at(&tab, b + 2u32);
                let c3 = at(&tab, b + 3u32) + x2h * at(&tab, b + 4u32);
                let c5 = at(&tab, b + 5u32) + x2h * at(&tab, b + 6u32);
                let x4 = x2h * x2h;
                sl = x3h * (at(&tab, b) + x2h * (c1 + x4 * (c3 + x4 * c5)));
            } else if u >= 0x3f30_0000_0000_0000u64 {
                let b = SMALL + 3u32;
                sl = x3h
                    * (at(&tab, b)
                        + x2h
                            * (at(&tab, b + 1u32)
                                + x2h * (at(&tab, b + 2u32) + x2h * at(&tab, b + 3u32))));
            } else if u >= 0x3e5a_0000_0000_0000u64 {
                let b = SMALL + 1u32;
                sl = x3h * (at(&tab, b) + x2h * at(&tab, b + 1u32));
            }

            let eps = EPS_ZERO * x3h;
            let lb = x + (sl - eps);
            let ub = x + (sl + eps);
            out = lb;
            if lb != ub {
                out = near_zero(x, x2h, x2l, fk);
            }
        }
    } else if u >= 0x7ff0_0000_0000_0000u64 {
        out = x + x; // an infinity or a NaN
    } else {
        // `x + sqrt(x^2 + 1)` as a double-double, three ways by magnitude.
        let mut x2h = 0.0 * x;
        let mut x2l = 0.0 * x;
        let mut ah = ax;
        let mut al = 0.0 * x;
        let mut off = 0x3ffu32;
        if u < 0x4190_0000_0000_0000u64 {
            // `|x| < 2^26`.
            x2h = x * x;
            x2l = fma64(x, x, -x2h, fk);
            // `fast_two_sum` needs its larger operand first, and which of
            // `1` and `x^2` that is changes at `|x| = 1`.
            let mut th0 = 0.0 * x;
            let mut tl0 = 0.0 * x;
            if u < 0x3ff0_0000_0000_0000u64 {
                let (a, b) = fast_two_sum(1.0, x2h);
                th0 = a;
                tl0 = b;
            } else {
                let (a, b) = fast_two_sum(x2h, 1.0);
                th0 = a;
                tl0 = b;
            }
            let tl = tl0 + x2l;
            let sh = f64::sqrt(th0);
            let rs = 0.5 / th0;
            let sl = (tl - fma64(sh, sh, -th0, fk)) * (rs * sh);
            let (ah0, tl1) = fast_two_sum(sh, ax);
            ah = ah0;
            al = sl + tl1;
        } else if u < 0x4330_0000_0000_0000u64 {
            // `|x| < 2^52`: `sqrt(x^2 + 1)` is `|x|` to within `1/(2|x|)`.
            ah = 2.0 * ax;
            al = 0.5 / ax;
        } else {
            // `|x| >= 2^52`: it is `|x|` exactly, and the doubling moves into
            // the exponent bias instead.
            off = 0x3feu32;
            ah = ax;
            al = 0.0 * x;
        }

        // `asinh(x) = ln(|x| + sqrt(x^2 + 1))`, with the logarithm's two
        // halves kept apart so that the error bound below has something to
        // bound. The association is `asinh`'s own — `acosh` sums the same
        // terms in a different order, and both are transcribed.
        let (ed, dx, i1, i2) = lt::log_index(ah, off, fk);
        let f = lt::log_poly(dx);
        let lh0 = lt::L2H * ed + (lt::l1_hi(i1) + lt::l2_hi(i2));
        let ll0 = lt::L2L * ed + lt::l1_lo(i1) + lt::l2_lo(i2) + al / ah + f + dx;
        let s = copysign(1.0, x);
        let lh = lh0 * s;
        let ll = ll0 * s;
        let eps = 1.63e-19;
        let lb = lh + (ll - eps);
        let ub = lh + (ll + eps);
        out = lb;
        if lb != ub {
            if ax < 0.25 {
                out = near_zero(x, x2h, x2l, fk);
            } else {
                out = refine(x, ah, al, LOG2E * f64::abs(lb), fk);
            }
        }
    }
    out
}
