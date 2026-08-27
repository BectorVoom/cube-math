//! `atan2f`.
//!
//! A port of glibc's `__atan2f` (`sysdeps/ieee754/flt-32/e_atan2f.c` —
//! CORE-MATH's `cr_atan2f`).
//!
//! # Why this one is a port and `hypotf` is not
//!
//! `cr_atan2f` is a *correctly rounded* routine, so the widening route
//! [`super::wide`] uses should reach it — and it does, on 129,598 of the
//! 129,600 argument pairs the equivalence sweep tries. The two it misses both
//! produce **subnormal** results, and they miss because upstream's own tiny-`y/x`
//! shortcut returns `(float)(y/x)` with a double-rounding guard that tests the
//! low 28 bits of the `double`. That test is the right one for a normal
//! result; for a subnormal one the `f32` rounding boundary sits elsewhere, and
//! the guard does not fire. So glibc's answer there is one ulp off the
//! correctly rounded value — checked against a 200-bit evaluation, not
//! assumed.
//!
//! Bit-exactness is a claim about the platform, not about the mathematics.
//! Reproducing it therefore means reproducing that, which means porting the
//! routine rather than computing the right answer a cheaper way.
//!
//! Both policy axes are accepted and have no effect.

use cubecl::prelude::*;

use crate::bits::opaque32;
use crate::config::MathConfig;
use crate::single::exact::copysign;
use crate::fma::{FmaKind, fma64};
use crate::tables::single::atan2 as t;

/// One `f64` out of a `u64` constant table.
#[cube]
pub fn at(tab: &Array<u64>, i: u32) -> f64 {
    f64::reinterpret(tab[usize::cast_from(i)])
}

/// The quadrant offset, `off[i]`: `{0, pi/2, pi, pi/2}` with the sign of `y`.
#[cube]
pub fn off(i: u32) -> f64 {
    let k = i & 3u32;
    let v = select(k == 0u32, 0.0, select(k == 2u32, t::PI, t::PI2));
    select(i >= 4u32, -v, v)
}

/// `offl[i]`, the low half of [`off()`].
#[cube]
pub fn offl(i: u32) -> f64 {
    let k = i & 3u32;
    let v = select(k == 0u32, 0.0, select(k == 2u32, 2.0 * t::PI2L, t::PI2L));
    select(i >= 4u32, -v, v)
}

/// `x * (ch + cl)` as a double-double, dropping the `xl cl` term.
///
/// Upstream's `muldd`. Not [`crate::double::dd::d_mul`]: the two accumulate
/// their cross terms in a different order, and the accurate path's whole
/// purpose is to be reproduced step for step.
#[cube]
pub fn muldd(xh: f64, xl: f64, ch: f64, cl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = fma64(ch, xh, -ahhh, fk) + alhh + ahlh;
    let hi = ahhh + ahhl;
    (hi, (ahhh - hi) + ahhl)
}

/// The accurate path's degree-31 double-double polynomial in `z^2`.
#[cube]
pub fn polydd(xh: f64, xl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let tab = Array::<u64>::from_data(comptime!(t::C.to_vec()));
    let mut ch = at(&tab, 31u32 * 2u32);
    let mut cl = at(&tab, 31u32 * 2u32 + 1u32);
    let mut i = 31u32.runtime();
    while i > 0u32 {
        i = i - 1u32;
        let (mh, ml) = muldd(xh, xl, ch, cl, fk);
        let c0 = at(&tab, i * 2u32);
        let th = mh + c0;
        let tl = (c0 - th) + mh;
        ch = th;
        cl = ml + tl + at(&tab, i * 2u32 + 1u32);
    }
    (ch, cl)
}

/// The tiny-`y/x` path: `atan(z) = z - z^3/3` with `z = y/x`.
///
/// The boundary nudge is upstream's, and so is its blind spot — see the module
/// documentation. `t & 0xfffffff == 0` is the right test for a `double` about
/// to be rounded to a *normal* `f32`; where the result is subnormal the
/// boundary sits elsewhere and the nudge does not fire.
#[cube]
pub fn tiny(y: f32, x: f32, #[comptime] fk: FmaKind) -> f32 {
    let dy = f64::cast_from(y);
    let dx = f64::cast_from(x);
    let z = dy / dx;
    let e0 = fma64(-z, dx, dy, fk);
    // `z x + e = y`, so `y/x = z + e/x`.
    let zz = z * z;
    let cz = t::MTHIRD * z;
    let e = e0 / dx + cz * zz;
    let tb = u64::reinterpret(z);
    let mut tn = tb;
    if tb & 0x0fff_ffffu64 == 0u64 {
        // Same sign: nudge up; opposite: nudge down. Either way the value
        // moves off the boundary that a second rounding would decide wrongly.
        tn = select(z * e > 0.0, tb + 1u64, tb - 1u64);
    }
    f32::cast_from(f64::reinterpret(tn))
}

/// `atan2(y, x)`.
#[cube]
pub fn atan2(y0: f32, x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let y = opaque32(y0);
    let x = opaque32(x0);
    let fk = comptime!(cfg.fma());
    let ux = u32::reinterpret(x);
    let uy = u32::reinterpret(y);
    let ax = ux & 0x7fff_ffffu32;
    let ay = uy & 0x7fff_ffffu32;

    let mut out = 0.0 + y;
    let mut done = false;

    if ay >= 0x7f80_0000u32 || ax >= 0x7f80_0000u32 {
        // `x + y` rather than either alone, so that a signalling NaN in the
        // *other* argument still raises invalid.
        if ay > 0x7f80_0000u32 || ax > 0x7f80_0000u32 {
            out = x + y;
            done = true;
        } else if ay == 0x7f80_0000u32 && ax == 0x7f80_0000u32 {
            // The sign is `y`'s throughout; applied by `copysign` rather than
            // by a multiply, because a literal zero times a runtime sign is
            // something the backends fold to `+0`.
            out = copysign(f32::cast_from(select(ux >> 31u32 != 0u32, t::TQPI, t::QPI)), y);
            done = true;
        } else if ax == 0x7f80_0000u32 {
            out = copysign(f32::cast_from(select(ux >> 31u32 != 0u32, t::PI, 0.0)), y);
            done = true;
        } else {
            out = copysign(f32::cast_from(t::PI2), y);
            done = true;
        }
    }

    if !done && ay == 0u32 {
        if ax == 0u32 {
            let i = (uy >> 31u32) * 4u32 + (ux >> 31u32) * 2u32;
            out = f32::cast_from(select(ux >> 31u32 != 0u32, off(i) + offl(i), off(i)));
            done = true;
        } else if ux >> 31u32 == 0u32 {
            out = copysign(0.0, y);
            done = true;
        }
    }

    if !done {
        let gt = u32::cast_from(ay > ax);
        let i = (uy >> 31u32) * 4u32 + (ux >> 31u32) * 2u32 + gt;

        let zx = f64::cast_from(x);
        let zy = f64::cast_from(y);
        // `z = x/y` when `|y| > |x|`, and `z = y/x` otherwise.
        let z = select(gt == 1u32, zx / zy, zy / zx);

        // `z^2` cannot underflow — for `|y| = 2^-149` and `|x|` at `FLT_MAX`,
        // `|z| > 2^-277` — but `z^4` and `z^8` might, which would raise a
        // spurious underflow this crate does not observe anyway.
        let mut r = 1.0 + 0.0 * z;
        let d = i32::reinterpret(ax) - i32::reinterpret(ay);
        if d < (27i32 << 23i32) && d > -(27i32 << 23i32) {
            let cnt = Array::<u64>::from_data(comptime!(t::CN.to_vec()));
            let cdt = Array::<u64>::from_data(comptime!(t::CD.to_vec()));
            let z2 = z * z;
            let z4 = z2 * z2;
            let z8 = z4 * z4;

            let mut cn0 = at(&cnt, 0u32) + z2 * at(&cnt, 1u32);
            let cn2 = at(&cnt, 2u32) + z2 * at(&cnt, 3u32);
            let mut cn4 = at(&cnt, 4u32) + z2 * at(&cnt, 5u32);
            cn0 = cn0 + z4 * cn2;
            cn4 = cn4 + z4 * at(&cnt, 6u32);
            cn0 = cn0 + z8 * cn4;

            let mut cd0 = at(&cdt, 0u32) + z2 * at(&cdt, 1u32);
            let cd2 = at(&cdt, 2u32) + z2 * at(&cdt, 3u32);
            let mut cd4 = at(&cdt, 4u32) + z2 * at(&cdt, 5u32);
            cd0 = cd0 + z4 * cd2;
            cd4 = cd4 + z4 * at(&cdt, 6u32);
            cd0 = cd0 + z8 * cd4;

            r = cn0 / cd0;
        }
        let zs = select(gt == 1u32, -z, z);
        let mut rr = zs * r + off(i);

        if (u64::reinterpret(rr) + 8u64) & 0x0fff_ffffu64 <= 16u64 {
            // The main path landed within a few ulp of an `f32` rounding
            // boundary, so it cannot settle the rounding on its own.
            if ay < ax && (ax - ay) >> 23u32 >= 25u32 {
                out = tiny(y, x, fk);
                done = true;
            } else {
                let mut zh = zy / zx;
                let mut zl = fma64(zh, -zx, zy, fk) / zx;
                if gt == 1u32 {
                    zh = zx / zy;
                    zl = fma64(zh, -zy, zx, fk) / zy;
                }
                let (z2h, z2l) = muldd(zh, zl, zh, zl, fk);
                let (ph0, pl0) = polydd(z2h, z2l, fk);
                let zhs = select(gt == 1u32, -zh, zh);
                let zls = select(gt == 1u32, -zl, zl);
                let (ph, pl) = muldd(zhs, zls, ph0, pl0, fk);

                let sh = ph + off(i);
                let sl = ((off(i) - sh) + ph) + pl + offl(i);
                let rf = f32::cast_from(sh);
                let th = f64::cast_from(rf);
                let dh = sh - th;
                let mut tm = dh + sl;
                if th + th * f64::from_bits(0x3c30000000000000)
                    == th - th * f64::from_bits(0x3c30000000000000)
                {
                    // `th` is a power of two, where the neighbouring `f32`
                    // spacings differ and the correction has to be scaled to
                    // the side it falls on.
                    let tth = (u64::reinterpret(th) & (0x7ffu64 << 52u64)) - (24u64 << 52u64);
                    tm = select(f64::abs(tm) > f64::reinterpret(tth), tm * 1.25, tm * 0.75);
                }
                rr = th + tm;
            }
        }
        if !done {
            out = f32::cast_from(rr);
        }
    }
    out
}
