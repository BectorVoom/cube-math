//! `acosh(x)`, correctly rounded.
//!
//! A port of glibc's `__ieee754_acosh` (`sysdeps/ieee754/dbl-64/e_acosh.c` —
//! CORE-MATH's `cr_acosh`), and the twin of [`super::asinh`] in every
//! structural respect: a fast double-double path with a certificate, the
//! shared accurate logarithm in [`super::logtab`] below it, and a table of the
//! ten arguments neither can resolve. See that module for why the whole
//! two-tier structure had to be ported rather than the fast half alone.
//!
//! # Where it differs from `asinh`
//!
//! Three places, all of them because `acosh(x) ~ ln(2x)` rather than
//! `ln(2x) + 1/(4x^2)`:
//!
//! * past `0x1.bfp+6` there is no square root at all. The correction to
//!   `ln(2x)` is a short polynomial in `1/x^2`, and it gets shorter as `x`
//!   grows — four terms, then three, then two, then nothing.
//! * the factor of two goes into the *exponent bias* the logarithm is asked
//!   for rather than into its argument, which is what `off = 0x3fe` is.
//! * near `x = 1` the answer is a square root of a small number, so the branch
//!   is written in `z = x - 1` throughout: `x^2 - 1` would have destroyed the
//!   information before the square root ever saw it.
//!
//! Both policy axes are accepted and have no effect.

use cubecl::prelude::*;

use crate::bits::opaque64;
use crate::config::MathConfig;
use crate::double::dd::{fast_two_sum, mul_dd_acc2, mul_ddd};
use crate::double::logtab as lt;
use crate::double::logtab::LOG2E;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::acosh_tab;

/// Offsets into [`acosh_tab()`].
const ONE_CH: u32 = 0;
/// The near-one series' plain tail, 6 slots.
const ONE_CL: u32 = ONE_CH + 20;
/// The fast path's near-one band, 9 slots.
const BAND0: u32 = ONE_CL + 6;
/// The three asymptotic bands' coefficients, 4 + 3 + 2 slots.
const ASYMPT: u32 = BAND0 + 9;
/// The hard-case table, 10 rows of 3.
const DB: u32 = ASYMPT + 9;
/// Rows in the hard-case table.
const DB_ROWS: u32 = 10;

/// `0x1.fcp-51`, the near-one band's relative error bound.
const EPS_ONE: f64 = f64::from_bits(0x3cbfe00000000000);

/// `2^-104`, the absolute floor under that bound.
const EPS_ONE_ABS: f64 = f64::from_bits(0x3970000000000000);

/// One `f64` out of a `u64` table.
#[cube]
pub fn at(tab: &Array<u64>, i: u32) -> f64 {
    f64::reinterpret(tab[usize::cast_from(i)])
}

/// `acosh(1 + z)` to double-double, when the near-one band's bound could not
/// settle the rounding.
///
/// Upstream's `as_acosh_one`. `sh + sl` is `sqrt(2z)`, already computed by the
/// caller — the series below is the *correction* to it.
#[cube]
pub fn near_one(z: f64, sh: f64, sl: f64, #[comptime] fk: FmaKind) -> f64 {
    let tab = acosh_tab();
    let y2a = z
        * (at(&tab, ONE_CL)
            + z * (at(&tab, ONE_CL + 1u32)
                + z * (at(&tab, ONE_CL + 2u32)
                    + z * (at(&tab, ONE_CL + 3u32)
                        + z * (at(&tab, ONE_CL + 4u32) + z * at(&tab, ONE_CL + 5u32))))));

    // `polydd3` over ten double-double coefficients, high index down.
    let (mut ch, cl0) = fast_two_sum(at(&tab, ONE_CH + 18u32), y2a);
    let mut cl = cl0 + at(&tab, ONE_CH + 19u32);
    let mut i = 9u32.runtime();
    while i > 0u32 {
        i = i - 1u32;
        let (mh, ml) = mul_dd_acc2(z, 0.0, ch, cl, fk);
        let (th, tl) = fast_two_sum(at(&tab, ONE_CH + i * 2u32), mh);
        ch = th;
        cl = ml + tl + at(&tab, ONE_CH + i * 2u32 + 1u32);
    }

    let (y1a, y2b) = mul_ddd(ch, cl, z, fk);
    let (y0a, y1b) = fast_two_sum(1.0, y1a);
    let (y0, y1) = mul_dd_acc2(y0a, y1b + y2b, sh, sl, fk);
    y0 + y1
}

/// The hard cases, as a trailing overwrite. See
/// [`super::asinh::database()`] on why this scans rather than bisects.
#[cube]
pub fn database(x: f64, f: f64) -> f64 {
    let tab = acosh_tab();
    let mut out = f;
    for k in 0..DB_ROWS {
        let row = DB + k * 3u32;
        if at(&tab, row) == x {
            out = at(&tab, row + 1u32) + at(&tab, row + 2u32);
        }
    }
    out
}

/// The accurate path.
#[cube]
pub fn refine(x: f64, a: f64, #[comptime] fk: FmaKind) -> f64 {
    let ix = u64::reinterpret(x);
    let mut zh = x;
    let mut zl = 0.0 * x;
    if ix < 0x4190_0000_0000_0000u64 {
        let x2h = x * x;
        let x2l = fma64(x, x, -x2h, fk);
        let (wh, wl) = fast_two_sum(x2h - 1.0, x2l);
        let sh = f64::sqrt(wh);
        let ish = 0.5 / wh;
        let sl = (ish * sh) * (wl - fma64(sh, sh, -wh, fk));
        let (zh0, zl0) = fast_two_sum(x, sh);
        let (zh1, zl1) = fast_two_sum(zh0, zl0 + sl);
        zh = zh1;
        zl = zl1;
    } else if ix < 0x4330_0000_0000_0000u64 {
        zh = 2.0 * x;
        zl = -0.5 / x;
    }

    let (v0a, v1a, v2) = lt::refine_core(zh, zl, a, true, fk);
    let v0 = v0a * 2.0;
    let v1b = v1a * 2.0;
    let v2s = v2 * 2.0;

    let t0b = u64::reinterpret(v1b);
    let mut v1 = v1b;
    if t0b & (0xffff_ffff_ffff_ffffu64 >> 12u64) == 0u64 {
        let w = u64::reinterpret(v2s);
        v1 = f64::reinterpret(select((w ^ t0b) >> 63u64 != 0u64, t0b - 1u64, t0b + 1u64));
    }
    let t = u64::reinterpret(v1);
    let t0 = u64::reinterpret(v0);
    let er = (t + 7u64) & (0xffff_ffff_ffff_ffffu64 >> 12u64);
    let de = ((t0 >> 52u64) & 0x7ffu64) - ((t >> 52u64) & 0x7ffu64);
    let res = v0 + v1;
    let mut out = res;
    if de > 102u64 || er < 15u64 {
        out = database(x, res);
    }
    out
}

/// `acosh(x)`.
#[cube]
pub fn acosh(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let ix = u64::reinterpret(x);
    let tab = acosh_tab();

    let mut out = x + x;
    if ix >= 0x7ff0_0000_0000_0000u64 {
        // Negative, or a NaN, or `-inf`.
        out = (x - x) / (x - x);
        if ix == 0x7ff0_0000_0000_0000u64 || (ix << 1u64) > (0x7ffu64 << 53u64) {
            out = x + x; // `+inf`, or a NaN propagating its payload
        }
    } else if ix <= 0x3ff0_0000_0000_0000u64 {
        out = (x - x) / (x - x);
        if ix == 0x3ff0_0000_0000_0000u64 {
            out = 0.0;
        }
    } else if ix < 0x3ff1_e83e425aee63u64 {
        // Near one, written in `z = x - 1` throughout.
        let z = x - 1.0;
        let iz = -0.25 / z;
        let zt = 2.0 * z;
        let sh = f64::sqrt(zt);
        let sl = fma64(sh, sh, -zt, fk) * (sh * iz);
        let z2 = z * z;
        let z4 = z2 * z2;
        let inner = ((at(&tab, BAND0 + 1u32) + z * at(&tab, BAND0 + 2u32))
            + z2 * (at(&tab, BAND0 + 3u32) + z * at(&tab, BAND0 + 4u32)))
            + z4 * ((at(&tab, BAND0 + 5u32) + z * at(&tab, BAND0 + 6u32))
                + z2 * (at(&tab, BAND0 + 7u32) + z * at(&tab, BAND0 + 8u32)));
        let ds = fma64(sh * z, at(&tab, BAND0) + z * inner, sl, fk);
        let eps = ds * EPS_ONE - EPS_ONE_ABS * sh;
        let lb = sh + (ds - eps);
        let ub = sh + (ds + eps);
        out = lb;
        if lb != ub {
            out = near_one(z, sh, sl, fk);
        }
    } else {
        // `acosh(x) = ln(x + sqrt(x^2 - 1))`, with the factor of two that
        // `ln(2x)` carries folded into the exponent bias rather than the
        // argument.
        let mut g = 0.0 * x;
        let mut off = 0x3feu32;
        let mut tbits = ix;
        if ix < 0x405b_f000_0000_0000u64 {
            off = 0x3ffu32;
            let x2h = x * x;
            let wh = x2h - 1.0;
            let wl = fma64(x, x, -x2h, fk);
            let sh = f64::sqrt(wh);
            let ish = 0.5 / wh;
            let sl = (wl - fma64(sh, sh, -wh, fk)) * (sh * ish);
            let (th, tl0) = fast_two_sum(x, sh);
            tbits = u64::reinterpret(th);
            g = (tl0 + sl) / th;
        } else if ix < 0x4087_1000_0000_0000u64 {
            let z = 1.0 / (x * x);
            g = at(&tab, ASYMPT)
                + z * (at(&tab, ASYMPT + 1u32)
                    + z * (at(&tab, ASYMPT + 2u32) + z * at(&tab, ASYMPT + 3u32)));
        } else if ix < 0x40e0_1000_0000_0000u64 {
            let z = 1.0 / (x * x);
            g = at(&tab, ASYMPT + 4u32)
                + z * (at(&tab, ASYMPT + 5u32) + z * at(&tab, ASYMPT + 6u32));
        } else if ix < 0x41ea_0000_0000_0000u64 {
            let z = 1.0 / (x * x);
            g = at(&tab, ASYMPT + 7u32) + z * at(&tab, ASYMPT + 8u32);
        }

        let (ed, dx, i1, i2) = lt::log_index(f64::reinterpret(tbits), off, fk);
        let f = lt::log_poly(dx);
        // The association is `acosh`'s own; `asinh` sums the same terms in a
        // different order and both are transcribed rather than reconciled.
        let lh = (lt::l1_hi(i1) + lt::l2_hi(i2)) + lt::L2H * ed;
        let ll = ((dx + lt::L2L * ed) + g) + (lt::l1_lo(i1) + lt::l2_lo(i2)) + f;
        let eps = 2.8e-19;
        let lb = lh + (ll - eps);
        let ub = lh + (ll + eps);
        out = lb;
        if lb != ub {
            out = refine(x, LOG2E * lb, fk);
        }
    }
    out
}
