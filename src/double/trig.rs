//! `sin`, `cos`, `sincos` and `tan`.
//!
//! A port of glibc's `__sin`, `__cos`, `__sincos` and `__tan`
//! (`sysdeps/ieee754/dbl-64/s_sin.c`, `s_sincos.c`, `s_tan.c` — the IBM
//! Accurate Mathematical Library), by way of `rmath`'s reference module, whose
//! fused-multiply-add placement was read out of a disassembly of the compiled
//! `_fma` entry points rather than inferred from the C source's grouping.
//!
//! Three entry points, not one shared by projection. `__sin`'s and
//! `__sincos`'s mid-band (`0.855469 <= |x| < 2.426265`) compute the
//! complementary angle *differently*: `__sin` calls `do_cos(y, hp1)` with the
//! low part passed straight through, while `__sincos` first forms a
//! compensated sum `a = y + hp1`, `da = (y - a) + hp1` and calls
//! `do_cos(a, da)`. Those are not the same computation — [`do_cos()`]'s own
//! reduction rounds through `|x| - (u - BIG) + dx` using whichever of `y` or
//! `a` it was handed, and `a != y` in general — so each of [`sin()`],
//! [`cos()`] and [`sincos()`] reproduces its own function's control flow
//! rather than a shared helper's.
//!
//! # The huge-argument band
//!
//! `rmath` stops at `105414350` (`1e8` for `tan`) and hands the rest to the
//! platform's own `sin`: on a CPU that is a `patch_lanes` repair, correct by
//! construction. A kernel has no platform to hand anything to, so this crate
//! ports [`super::branred`] as well and the band is a real band here. See that
//! module for what it costs and how its fusion was settled.
//!
//! # Both policies run the same code
//!
//! [`crate::Accuracy::Fast`] is accepted and has no effect, for the reason
//! [`super::pow`] and [`super::invtrig`] give. The reduction *is* the cost of
//! a trigonometric function on a large argument, and it is the one part no
//! approximation can skip without changing which quadrant the answer is in.

use cubecl::prelude::*;

use crate::bits::opaque64;
use crate::config::MathConfig;
use crate::double::branred::branred;
use crate::double::exact::copysign;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::{sincos_tab, tan_xfg_tab};
use crate::tables::double::trig as t;

// ---------------------------------------------------------------------------
// sin / cos / sincos
// ---------------------------------------------------------------------------

/// `TAYLOR_SIN`: `sin(x + dx)` by the degree-11 series, for `|x| < 0.126`.
///
/// Every step fuses except the leading `dx * 0.5` and the final `x + t`, which
/// are a separate multiply and add on the disassembly.
#[cube]
pub fn taylor_sin(xx: f64, x: f64, dx: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut poly = fma64(xx, t::S5, t::S4, fk);
    poly = fma64(xx, poly, t::S3, fk);
    poly = fma64(xx, poly, t::S2, fk);
    poly = fma64(xx, poly, t::S1, fk);
    let inner = fma64(poly, x, -(dx * 0.5), fk);
    x + fma64(xx, inner, dx, fk)
}

/// `cs2 + xx (cs4 + xx cs6)`, the cosine part's inner polynomial.
#[cube]
pub fn cos_inner(xx: f64, #[comptime] fk: FmaKind) -> f64 {
    fma64(xx, fma64(xx, t::CS6, t::CS4, fk), t::CS2, fk)
}

/// `sn3 + xx sn5`, the sine part's inner polynomial.
#[cube]
pub fn sin_inner(xx: f64, #[comptime] fk: FmaKind) -> f64 {
    fma64(xx, t::SN5, t::SN3, fk)
}

/// `cos(x + dx)`, with `|x + dx|` already inside the table's domain.
///
/// `cor` is three separate fused multiply-adds, not the plain arithmetic the C
/// source's grouping alone would suggest.
#[cube]
pub fn do_cos(x: f64, dx0: f64, #[comptime] fk: FmaKind) -> f64 {
    let dx = select(x < 0.0, -dx0, dx0);
    // `BIG + |x|` rounds `|x|` to the table's resolution, and the low bits of
    // the sum *are* the row index: everything finer than `2^-7` has been
    // rounded away, so what is left in the low word is `round(128 |x|)`.
    let u = t::BIG + f64::abs(x);
    let xr = f64::abs(x) - (u - t::BIG) + dx;

    let xx = xr * xr;
    let s = fma64(xr * xx, sin_inner(xx, fk), xr, fk);
    let c = xx * cos_inner(xx, fk);

    let tab = sincos_tab();
    let row = usize::cast_from(u32::cast_from(u64::reinterpret(u) & 0xffff_ffffu64) * 4u32);
    let sn = f64::reinterpret(tab[row]);
    let ssn = f64::reinterpret(tab[row + 1]);
    let cs = f64::reinterpret(tab[row + 2]);
    let ccs = f64::reinterpret(tab[row + 3]);

    let step1 = fma64(s, -ssn, ccs, fk);
    let step2 = fma64(c, -cs, step1, fk);
    cs + fma64(s, -sn, step2, fk)
}

/// `sin(x + dx)`.
#[cube]
pub fn do_sin(x: f64, dx0: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut out = x;
    if f64::abs(x) < 0.126 {
        out = taylor_sin(x * x, x, dx0, fk);
    } else {
        let dx = select(x <= 0.0, -dx0, dx0);
        let u = t::BIG + f64::abs(x);
        let xr = f64::abs(x) - (u - t::BIG);

        let xx = xr * xr;
        let s = xr + fma64(xr * xx, sin_inner(xx, fk), dx, fk);
        // `xx * cos_inner` rounds separately and *then* `xr * dx` fuses into
        // the add — the opposite pairing from what a left-to-right reading of
        // the C would give, and confirmed by trace rather than assumed.
        let c = fma64(xr, dx, xx * cos_inner(xx, fk), fk);

        let tab = sincos_tab();
        let row = usize::cast_from(u32::cast_from(u64::reinterpret(u) & 0xffff_ffffu64) * 4u32);
        let sn = f64::reinterpret(tab[row]);
        let ssn = f64::reinterpret(tab[row + 1]);
        let cs = f64::reinterpret(tab[row + 2]);
        let ccs = f64::reinterpret(tab[row + 3]);

        let step1 = fma64(s, ccs, ssn, fk);
        let step2 = fma64(c, -sn, step1, fk);
        out = copysign(sn + fma64(s, cs, step2, fk), x);
    }
    out
}

/// `reduce_sincos`: `(a, da, n)` with `x = n pi/2 + a + da` and
/// `|a + da| <= pi/4`, for `|x| < 105414350`.
///
/// The two places where the same product (`xn * PP3`, `xn * PP4`) is re-issued
/// inside a second fused multiply-add rather than reused are deliberate: that
/// is what the compiled glibc does, because reusing the value would mean
/// keeping an unfused intermediate around.
#[cube]
pub fn reduce_sincos(x: f64, #[comptime] fk: FmaKind) -> (f64, f64, u32) {
    let tv = fma64(x, t::HPINV, t::TOINT, fk);
    let xn = tv - t::TOINT;
    let n = u32::cast_from(u64::reinterpret(tv) & 0xffff_ffffu64) & 3u32;

    let mut y = fma64(xn, -t::MP1, x, fk);
    y = fma64(xn, -t::MP2, y, fk);

    let t2 = fma64(xn, -t::PP3, y, fk);
    let db = fma64(xn, -t::PP3, y - t2, fk);

    let b = fma64(xn, -t::PP4, t2, fk);
    let da = fma64(xn, -t::PP4, t2 - b, fk) + db;

    (b, da, n)
}

/// `do_sincos`: `sin` or `cos` of `a + da` in quadrant `n`, as one value.
#[cube]
pub fn do_sincos_one(a: f64, da: f64, n: u32, #[comptime] fk: FmaKind) -> f64 {
    let mut retval = do_sin(a, da, fk);
    if n & 1u32 != 0u32 {
        retval = do_cos(a, da, fk);
    }
    select(n & 2u32 != 0u32, -retval, retval)
}

/// `sin(x)`.
#[cube]
pub fn sin(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let k = u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7fff_ffffu32;

    let mut out = x;
    if k < 0x3e50_0000u32 {
        out = x;
    } else if k < 0x3feb_6000u32 {
        out = do_sin(x, 0.0, fk);
    } else if k < 0x4003_68fdu32 {
        out = copysign(do_cos(t::HP0 - f64::abs(x), t::HP1, fk), x);
    } else if k < 0x4199_21fbu32 {
        let (a, da, n) = reduce_sincos(x, fk);
        out = do_sincos_one(a, da, n, fk);
    } else if k < 0x7ff0_0000u32 {
        let (a, da, n) = branred(x);
        out = do_sincos_one(a, da, n, fk);
    } else {
        // An infinity or a NaN. `x / x` is glibc's own idiom and a real
        // runtime division: the NaN it produces carries this hardware's own
        // sign and payload, which a constant would not.
        out = x / x;
    }
    out
}

/// `cos(x)`.
#[cube]
pub fn cos(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let k = u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7fff_ffffu32;

    let mut out = x;
    if k < 0x3e40_0000u32 {
        out = 1.0;
    } else if k < 0x3feb_6000u32 {
        out = do_cos(x, 0.0, fk);
    } else if k < 0x4003_68fdu32 {
        let y = t::HP0 - f64::abs(x);
        let a = y + t::HP1;
        out = do_sin(a, (y - a) + t::HP1, fk);
    } else if k < 0x4199_21fbu32 {
        let (a, da, n) = reduce_sincos(x, fk);
        out = do_sincos_one(a, da, n + 1u32, fk);
    } else if k < 0x7ff0_0000u32 {
        let (a, da, n) = branred(x);
        out = do_sincos_one(a, da, n + 1u32, fk);
    } else {
        out = x / x;
    }
    out
}

/// `(sin(x), cos(x))`.
#[cube]
pub fn sincos(x0: f64, #[comptime] cfg: MathConfig) -> (f64, f64) {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let k = u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7fff_ffffu32;

    let mut sx = x;
    let mut cx = x;
    if k < 0x4003_68fdu32 {
        if k < 0x3e40_0000u32 {
            sx = x;
            cx = 1.0;
        } else if k < 0x3feb_6000u32 {
            sx = do_sin(x, 0.0, fk);
            cx = do_cos(x, 0.0, fk);
        } else {
            // `__sincos` forms the compensated pair before calling `do_cos`;
            // `__sin` does not. See the module doc — they are different
            // computations, not two spellings of one.
            let y = t::HP0 - f64::abs(x);
            let a = y + t::HP1;
            let da = (y - a) + t::HP1;
            sx = copysign(do_cos(a, da, fk), x);
            cx = do_sin(a, da, fk);
        }
    } else if k < 0x7ff0_0000u32 {
        let mut a = 0.0 + x;
        let mut da = 0.0 + x;
        let mut n = 0u32;
        if k < 0x4199_21fbu32 {
            let (a0, da0, n0) = reduce_sincos(x, fk);
            a = a0;
            da = da0;
            n = n0;
        } else {
            let (a0, da0, n0) = branred(x);
            a = a0;
            da = da0;
            n = n0;
        }
        n = n & 3u32;
        if n == 1u32 || n == 2u32 {
            a = -a;
            da = -da;
        }
        let s = do_sin(a, da, fk);
        let xx = do_cos(a, da, fk);
        let c = select(n & 2u32 != 0u32, -xx, xx);
        // `__sincos` swaps the *output pointers* for an odd quadrant, so the
        // sine slot takes the cosine's value and the negation still lands on
        // the cosine computation.
        sx = s;
        cx = c;
        if n & 1u32 != 0u32 {
            sx = c;
            cx = s;
        }
    } else {
        sx = x / x;
        cx = x / x;
    }
    (sx, cx)
}

// ---------------------------------------------------------------------------
// tan
// ---------------------------------------------------------------------------

/// `utan.tbl` row `round_toward_zero(256 w - 15.5)`, as `(x, f, g)`.
#[cube]
pub fn xfg_row(w: f64, #[comptime] fk: FmaKind) -> (f64, f64, f64) {
    let tab = tan_xfg_tab();
    let i = usize::cast_from(u32::cast_from(fma64(w, 256.0, t::MFFTNHF, fk)) * 3u32);
    (
        f64::reinterpret(tab[i]),
        f64::reinterpret(tab[i + 1]),
        f64::reinterpret(tab[i + 2]),
    )
}

/// `pz`, the odd correction polynomial the table bands interpolate with.
#[cube]
pub fn tan_pz(z: f64, #[comptime] fk: FmaKind) -> f64 {
    let z2 = z * z;
    fma64(z * z2, fma64(z2, t::E1, t::E0, fk), z, fk)
}

/// `EADD`: `a + b` as an unevaluated pair, ordering the operands by magnitude.
#[cube]
pub fn eadd(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let mut c = (b - s) + a;
    if f64::abs(a) > f64::abs(b) {
        c = (a - s) + b;
    }
    (s, c)
}

/// The tail shared by every reduced band: `tan` (or `-cot`) of `a + da`, with
/// quadrant parity `n`.
///
/// The polynomial bands evaluate the *signed* `a` and `da` — the series is
/// odd, and `-cot` is its own odd evaluation, so the sign rides on the values.
/// Only the table bands multiply by `sy`.
#[cube]
pub fn tan_tail(a: f64, da: f64, n: u32, #[comptime] fk: FmaKind) -> f64 {
    let ya = select(a < 0.0, -a, a);
    let yya = select(a < 0.0, -da, da);
    let sy = select(a < 0.0, -1.0, 1.0);

    let mut out = a;
    if ya <= t::GY2 {
        let a2 = a * a;
        let mut t2 = fma64(a2, t::D11, t::D9, fk);
        t2 = fma64(a2, t2, t::D7, fk);
        t2 = fma64(a2, t2, t::D5, fk);
        t2 = fma64(a2, t2, t::D3, fk);
        t2 = fma64(a * a2, t2, da, fk);
        let y = a + t2;
        if n & 1u32 != 0u32 {
            // `-cot`, through `DIV2`'s compensated reciprocal. The `+ 0.0` is
            // the dividend's own low part `1.0 + 0.0`, kept as the compiled
            // code writes it.
            let (b, db) = eadd(a, t2);
            let c = 1.0 / b;
            let u = c * b;
            let uu = fma64(c, b, -u, fk);
            let t3 = ((1.0 - u) - uu) + 0.0;
            let cc = fma64(-db, c, t3, fk) / b;
            let z = c + cc;
            out = -(z + ((c - z) + cc));
        } else {
            out = y;
        }
    } else {
        let (xfg, fi, gi) = xfg_row(ya, fk);
        let pz = tan_pz((ya - xfg) + yya, fk);
        if n & 1u32 != 0u32 {
            out = -sy * (gi - pz * (fi + gi) / (fi + pz));
        } else {
            out = sy * (fi + pz * (gi + fi) / (gi - pz));
        }
    }
    out
}

/// `tan(x)`.
#[cube]
pub fn tan(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let k = u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7ff0_0000u32;
    let w = f64::abs(x);

    let mut out = x;
    if k == 0x7ff0_0000u32 {
        // An infinity or a NaN. `x - x` is glibc's own spelling, and a real
        // subtraction: the NaN it produces carries `x`'s own payload.
        out = x - x;
    } else if w <= t::G1 {
        // (I) `|x| < 1.259e-8`. The underflow-forcing `w * w` glibc does here
        // only sets a flag, which nothing in this crate observes.
        out = x;
    } else if w <= t::G2 {
        // (II) `|x| < 0.0608`: the direct Taylor series.
        let x2 = x * x;
        let mut t2 = fma64(x2, t::D11, t::D9, fk);
        t2 = fma64(x2, t2, t::D7, fk);
        t2 = fma64(x2, t2, t::D5, fk);
        t2 = fma64(x2, t2, t::D3, fk);
        out = fma64(x * x2, t2, x, fk);
    } else if w <= t::G3 {
        // (III) `|x| < 0.787`: the table, on the *signed* argument's sign —
        // the disassembly tests `x`, not `w`.
        let (xfg, fi, gi) = xfg_row(w, fk);
        let pz = tan_pz(w - xfg, fk);
        out = select(x < 0.0, -1.0, 1.0) * (fi + pz * (gi + fi) / (gi - pz));
    } else if w <= t::G4 {
        // (IV) `|x| < 25`: reduction by the three-part `mp` split.
        let tv = fma64(x, t::HPINV, t::TOINT, fk);
        let xn = tv - t::TOINT;
        let mut t1 = fma64(xn, -t::MP1, x, fk);
        t1 = fma64(xn, -t::MP2, t1, fk);
        let a = fma64(xn, -t::MP3, t1, fk);
        let da = fma64(xn, -t::MP3, t1 - a, fk);
        out = tan_tail(a, da, u32::cast_from(u64::reinterpret(tv) & 1u64), fk);
    } else if w <= t::G5 {
        // (V) `|x| <= 1e8`: reduction by the four-part `pp` split, which
        // carries the reduced angle far enough for `tan`'s 0.62-ulp bound.
        let tv = fma64(x, t::HPINV, t::TOINT, fk);
        let xn = tv - t::TOINT;
        let mut t1 = fma64(xn, -t::MP1, x, fk);
        t1 = fma64(xn, -t::MP2, t1, fk);
        let a = fma64(xn, -t::PP3, t1, fk);
        let da0 = fma64(xn, -t::PP3, t1 - a, fk);
        let b = fma64(xn, -t::PP4, a, fk);
        let db = fma64(xn, -t::PP4, a - b, fk);
        let (sum, cc) = eadd(b, db + da0);
        out = tan_tail(sum, cc, u32::cast_from(u64::reinterpret(tv) & 1u64), fk);
    } else {
        // (VI) `|x| > 1e8`: the Payne-Hanek reduction.
        let (a, da, n) = branred(x);
        let (t1, t2) = eadd(a, da);
        out = tan_tail(t1, t2, n & 1u32, fk);
    }
    out
}
