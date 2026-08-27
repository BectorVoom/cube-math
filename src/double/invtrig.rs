//! `atan`, `atan2`, `asin` and `acos`.
//!
//! A port of glibc's `__atan`, `__ieee754_atan2`, `__ieee754_asin` and
//! `__ieee754_acos` (`sysdeps/ieee754/dbl-64/s_atan.c`, `e_atan2.c`,
//! `e_asin.c` — the IBM Accurate Mathematical Library), by way of `rmath`'s
//! reference module, which read the fused multiply-add placement out of a
//! disassembly of the compiled `_fma` entry points rather than guessing it
//! from the C source. Two patterns recur in every band of all four functions:
//!
//! * a band's final "polynomial times something, plus a leading term" step is
//!   one fused multiply-add, even where the C spells it as a separate multiply
//!   and add;
//! * the reciprocal bands use `dla.h`'s `EMULV`, which under `__FP_FAST_FMA`
//!   collapses to the ordinary 2Product — [`super::dd::a_mul()`], reused here
//!   rather than reinvented.
//!
//! # Both policies run the same code
//!
//! [`crate::Accuracy::Fast`] is accepted and has no effect, for the reason
//! [`super::pow`] and [`super::hypot`] give: these *are* table-driven
//! algorithms whose whole cost is one gather and one short polynomial, and
//! there is no cheaper shape that is still worth having. `atan`'s Taylor band
//! is six fused multiply-adds; its table band is a seven-slot row and five
//! more. An approximation would save the gather and nothing else.
//!
//! # What the port changed
//!
//! `e_asin.c` selects a table row with six different formulas over the top 32
//! bits of `|x|`, into rows of five different widths. All six are the single
//! expression `floor(256 |x|) - 32` in disguise — the table is really indexed
//! by the top eight bits of the argument — so this kernel computes the row
//! once, from a float multiply and a truncation, and reads it out of the
//! uniform-stride repacking in [`crate::tables::double::asincos_packed`]. The
//! coefficient slots a row's own degree does not use are `+0.0` there, and
//! `fma(xx, 0, c) == c` exactly, so one unrolled degree-9 fold reproduces
//! every band's shorter chain bit for bit.

use cubecl::prelude::*;

use crate::bits::{is_nan64, nan64, opaque64};
use crate::config::MathConfig;
use crate::double::dd::{a_mul, two_sum};
use crate::double::exact::copysign;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::{asncs_tab, atan_cij_tab, inroot_tab, powtwo_tab};
use crate::tables::double::asincos_data as ac;
use crate::tables::double::asincos_packed as pk;
use crate::tables::double::atan2_data as at2;
use crate::tables::double::atan_data as at;

/// `2^52`: added to round a small positive `f64` to the nearest integer, then
/// subtracted back off.
///
/// Exact throughout — the multiply that feeds it (`256 * w`, a power of two)
/// never rounds either, so the trick is insensitive to a backend fusing that
/// multiply into the add.
const TWO52: f64 = 4503599627370496.0;

/// `atan2`'s far-apart-exponents threshold, `57` in the exponent field's own
/// units: past it the ratio is zero or infinite to within a rounding.
const EP: i32 = 59768832;

/// Slots in one packed `asncs` row. See [`pk::STRIDE`].
const STRIDE: u32 = pk::STRIDE as u32;

// ---------------------------------------------------------------------------
// atan
// ---------------------------------------------------------------------------

/// The `cij` row index, `round(256 w) - 16`.
///
/// The rounding is the classic add-and-subtract: `256 w` is exact (a power of
/// two), `TWO52 + it` rounds to the nearest integer, and subtracting `TWO52`
/// back leaves that integer as a `f64`.
#[cube]
pub fn table_index(w: f64) -> u32 {
    u32::cast_from((TWO52 + 256.0 * w) - TWO52) - 16u32
}

/// `atan(|x|)` for `A <= u < B`: the direct Taylor series in `v = x*x`.
#[cube]
pub fn atan_taylor(x: f64, v: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut yy = fma64(v, at::D13, at::D11, fk);
    yy = fma64(v, yy, at::D9, fk);
    yy = fma64(v, yy, at::D7, fk);
    yy = fma64(v, yy, at::D5, fk);
    yy = fma64(v, yy, at::D3, fk);
    fma64(x * v, yy, x, fk)
}

/// `atan(u)` for `B <= u < C`: the direct table band, `z = u - x0`.
#[cube]
pub fn atan_table(u: f64, #[comptime] fk: FmaKind) -> f64 {
    let tab = atan_cij_tab();
    let row = usize::cast_from(table_index(u) * 7u32);
    let z = u - f64::reinterpret(tab[row]);
    let mut yy = fma64(z, f64::reinterpret(tab[row + 6]), f64::reinterpret(tab[row + 5]), fk);
    yy = fma64(z, yy, f64::reinterpret(tab[row + 4]), fk);
    yy = fma64(z, yy, f64::reinterpret(tab[row + 3]), fk);
    yy = fma64(z, yy, f64::reinterpret(tab[row + 2]), fk);
    fma64(z, yy, f64::reinterpret(tab[row + 1]), fk)
}

/// `atan(u)` for `C <= u < D`: the reciprocal fold plus the table, `w = 1/u`.
#[cube]
pub fn atan_recip_table(u: f64, #[comptime] fk: FmaKind) -> f64 {
    let w = 1.0 / u;
    let (t1p, t2p) = a_mul(w, u, fk);
    let s = (1.0 - t1p) - t2p;
    let tab = atan_cij_tab();
    let row = usize::cast_from(table_index(w) * 7u32);
    let z = fma64(s, w, w - f64::reinterpret(tab[row]), fk);
    let mut yy = fma64(z, f64::reinterpret(tab[row + 6]), f64::reinterpret(tab[row + 5]), fk);
    yy = fma64(z, yy, f64::reinterpret(tab[row + 4]), fk);
    yy = fma64(z, yy, f64::reinterpret(tab[row + 3]), fk);
    yy = fma64(z, yy, f64::reinterpret(tab[row + 2]), fk);
    yy = fma64(-z, yy, at::HPI1, fk);
    (at::HPI - f64::reinterpret(tab[row + 1])) + yy
}

/// `atan(u)` for `D <= u < E`: the reciprocal fold plus the Taylor series.
///
/// `dla.h`'s `ESUB(HPI, w, ..)` has no fused-multiply-add-specialised form,
/// but this band only ever reaches it with `w = 1/u <= 1/16 < HPI`, so its
/// magnitude-ordered branch always takes the same arm — the compiled glibc
/// proves that range and folds the branch away, and so does this.
#[cube]
pub fn atan_recip_taylor(u: f64, #[comptime] fk: FmaKind) -> f64 {
    let w = 1.0 / u;
    let v = w * w;
    let (t1p, t2p) = a_mul(w, u, fk);
    let mut yy = fma64(v, at::D13, at::D11, fk);
    yy = fma64(v, yy, at::D9, fk);
    yy = fma64(v, yy, at::D7, fk);
    yy = fma64(v, yy, at::D5, fk);
    yy = fma64(v, yy, at::D3, fk);
    let t3 = at::HPI - w;
    let cor = (at::HPI - t3) - w;
    let s = (1.0 - t1p) - t2p;
    let inner = fma64(-s, w, at::HPI1 + cor, fk);
    t3 + fma64(-(w * v), yy, inner, fk)
}

/// `atan(x)`.
///
/// No `math_check_force_underflow` equivalent: that glibc call only raises the
/// underflow exception, and nothing in this crate observes an exception flag.
#[cube]
pub fn atan(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let u = f64::abs(x);
    let mut out = x + x;
    if !is_nan64(x) {
        if u < at::A {
            out = x;
        } else if u < at::B {
            out = atan_taylor(x, x * x, fk);
        } else if u < at::C {
            out = copysign(atan_table(u, fk), x);
        } else if u < at::D {
            out = copysign(atan_recip_table(u, fk), x);
        } else if u < at::E {
            out = copysign(atan_recip_taylor(u, fk), x);
        } else {
            out = copysign(at::HPI, x);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// atan2
// ---------------------------------------------------------------------------
//
// `__ieee754_atan2` does not call into `atan`'s own code — its disassembly has
// no call into `__atan_fma`. It reimplements the same two-shapes-per-band
// structure inline, once per quadrant, each with its own additive constant
// (`0`, `pi/2`, `pi`) and one extra compensated term (`du`, the division's own
// rounding residual) that `atan`'s bands do not carry, because here the
// argument is a ratio `y/x` rather than `x` itself.

/// The degree-13 Horner chain every `atan2` Taylor band shares.
#[cube]
pub fn atan2_taylor_poly(v: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut yy = fma64(v, at::D13, at::D11, fk);
    yy = fma64(v, yy, at::D9, fk);
    yy = fma64(v, yy, at::D7, fk);
    yy = fma64(v, yy, at::D5, fk);
    fma64(v, yy, at::D3, fk)
}

/// Case (i): `x > 0`, `ay < ax`, `u < 1/16`.
#[cube]
pub fn atan2_i_taylor(u: f64, du: f64, #[comptime] fk: FmaKind) -> f64 {
    let v = u * u;
    let poly = atan2_taylor_poly(v, fk);
    u + fma64(u * v, poly, du, fk)
}

/// Case (i): `x > 0`, `ay < ax`, `u >= 1/16`.
///
/// The `cij` row reads as `[x0, t1, t2, c3..c6]` here — slot 2 is its own
/// leading coefficient rather than the polynomial's, unlike `atan`'s own
/// `[x0, t1, c2..c6]` reading of the same bytes.
#[cube]
pub fn atan2_i_table(u: f64, du: f64, #[comptime] fk: FmaKind) -> f64 {
    let tab = atan_cij_tab();
    let row = usize::cast_from(table_index(u) * 7u32);
    let t2 = f64::reinterpret(tab[row + 2]);
    let t3 = u - f64::reinterpret(tab[row]);
    let (v, dv) = two_sum(t3, du);
    let mut poly = fma64(v, f64::reinterpret(tab[row + 6]), f64::reinterpret(tab[row + 5]), fk);
    poly = fma64(v, poly, f64::reinterpret(tab[row + 4]), fk);
    poly = fma64(v, poly, f64::reinterpret(tab[row + 3]), fk);
    let inner = fma64(v * v, poly, dv * t2, fk);
    f64::reinterpret(tab[row + 1]) + fma64(v, t2, inner, fk)
}

/// The table band shared by cases (ii), (iii) and (iv).
///
/// `atan_recip_table`'s shape exactly, with a caller-supplied base (`pi/2` or
/// `pi`), a caller-supplied sign, and a `du`-compensated `v` in place of the
/// plain subtraction.
#[cube]
pub fn atan2_table_shared(
    u: f64,
    du: f64,
    base: f64,
    base1: f64,
    #[comptime] add: bool,
    #[comptime] fk: FmaKind,
) -> f64 {
    let tab = atan_cij_tab();
    let row = usize::cast_from(table_index(u) * 7u32);
    let v = (u - f64::reinterpret(tab[row])) + du;
    let mut poly = fma64(v, f64::reinterpret(tab[row + 6]), f64::reinterpret(tab[row + 5]), fk);
    poly = fma64(v, poly, f64::reinterpret(tab[row + 4]), fk);
    poly = fma64(v, poly, f64::reinterpret(tab[row + 3]), fk);
    poly = fma64(v, poly, f64::reinterpret(tab[row + 2]), fk);
    let t1c = f64::reinterpret(tab[row + 1]);
    let mut out = (base - t1c) + fma64(-v, poly, base1, fk);
    if comptime!(add) {
        out = (base + t1c) + fma64(v, poly, base1, fk);
    }
    out
}

/// Case (ii): `x > 0`, `ay >= ax`, `u < 1/16`. `pi/2 - atan(u)`.
#[cube]
pub fn atan2_ii_taylor(u: f64, du: f64, #[comptime] fk: FmaKind) -> f64 {
    let v = u * u;
    let zz = (u * v) * atan2_taylor_poly(v, fk);
    let t2 = at::HPI - u;
    // `ESUB`, its branch folded: `u < 1/16 < HPI` always.
    let cor = (at::HPI - t2) - u;
    t2 + (((at::HPI1 + cor) - du) - zz)
}

/// Case (iii): `x < 0`, `ax < ay`, `u < 1/16`. `pi/2 + atan(u)`.
#[cube]
pub fn atan2_iii_taylor(u: f64, du: f64, #[comptime] fk: FmaKind) -> f64 {
    let v = u * u;
    let zz = (u * v) * atan2_taylor_poly(v, fk);
    let t2 = at::HPI + u;
    // `EADD`, its branch folded: `u < 1/16 < HPI` always.
    let cor = (at::HPI - t2) + u;
    t2 + (((at::HPI1 + cor) + du) + zz)
}

/// Case (iv): `x < 0`, `ax >= ay`, `u < 1/16`. `pi - atan(u)`.
#[cube]
pub fn atan2_iv_taylor(u: f64, du: f64, #[comptime] fk: FmaKind) -> f64 {
    let v = u * u;
    let zz = (u * v) * atan2_taylor_poly(v, fk);
    let t2 = at2::OPI - u;
    // `ESUB`, its branch folded: `u <= 1 < OPI` always.
    let cor = (at2::OPI - t2) - u;
    t2 + (((at2::OPI1 + cor) - du) - zz)
}

/// `atan2(y, x)`.
///
/// Every special case is `copysign` of a constant onto `y`, which is what the
/// reference's own eleven-way `is_sign_positive` cascade computes: the
/// constants come in exact `+-` pairs, and the sign is always `y`'s.
#[cube]
pub fn atan2(y0: f64, x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let y = opaque64(y0);
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());

    let mut out = x + y;
    if is_nan64(x) {
        out = x + y;
    } else if is_nan64(y) {
        out = y + y;
    } else if y == 0.0 {
        // The sign of `x` decides between `+-0` and `+-pi`; the sign of `y`
        // decides which of the pair.
        out = copysign(0.0, y);
        if u64::reinterpret(x) >> 63u64 != 0u64 {
            out = copysign(at2::OPI, y);
        }
    } else if x == 0.0 {
        out = copysign(at::HPI, y);
    } else if f64::abs(x) == crate::bits::inf64() {
        if u64::reinterpret(x) >> 63u64 == 0u64 {
            out = copysign(0.0, y);
            if f64::abs(y) == crate::bits::inf64() {
                out = copysign(at2::QPI, y);
            }
        } else {
            out = copysign(at2::OPI, y);
            if f64::abs(y) == crate::bits::inf64() {
                out = copysign(at2::TQPI, y);
            }
        }
    } else if f64::abs(y) == crate::bits::inf64() {
        out = copysign(at::HPI, y);
    } else {
        let mut ax = f64::abs(x);
        let mut ay = f64::abs(y);
        let yhi = u32::cast_from(u64::reinterpret(y) >> 32u64) & 0x7ff0_0000u32;
        let xhi = u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7ff0_0000u32;
        let de = i32::reinterpret(yhi) - i32::reinterpret(xhi);

        if de >= EP {
            out = copysign(at::HPI, y);
        } else if de <= -EP {
            out = copysign(at2::OPI, y);
            if x > 0.0 {
                out = copysign(ay / ax, y);
            }
        } else {
            if ax < at2::TWOM500 || ay < at2::TWOM500 {
                ax = ax * at2::TWO500;
                ay = ay * at2::TWO500;
            }
            if ax > at2::TWO500 || ay > at2::TWO500 {
                ax = ax * at2::TWOM500;
                ay = ay * at2::TWOM500;
            }

            // The ratio, plus the residual its own division threw away.
            let mut u = ay / ax;
            let mut du = 0.0;
            if ay < ax {
                let (v, vv) = a_mul(ax, u, fk);
                du = ((ay - v) - vv) / ax;
            } else {
                u = ax / ay;
                let (v, vv) = a_mul(ay, u, fk);
                du = ((ax - v) - vv) / ay;
            }

            let mut z = 0.0;
            if x > 0.0 {
                if ay < ax {
                    if u < at::B {
                        z = atan2_i_taylor(u, du, fk);
                    } else {
                        z = atan2_i_table(u, du, fk);
                    }
                } else if u < at::B {
                    z = atan2_ii_taylor(u, du, fk);
                } else {
                    z = atan2_table_shared(u, du, at::HPI, at::HPI1, false, fk);
                }
            } else if ax < ay {
                if u < at::B {
                    z = atan2_iii_taylor(u, du, fk);
                } else {
                    z = atan2_table_shared(u, du, at::HPI, at::HPI1, true, fk);
                }
            } else if u < at::B {
                z = atan2_iv_taylor(u, du, fk);
            } else {
                z = atan2_table_shared(u, du, at2::OPI, at2::OPI1, false, fk);
            }
            out = copysign(z, y);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// asin / acos
// ---------------------------------------------------------------------------

/// The band edges, as the `f64` values the reference's top-32-bit comparisons
/// are really testing for.
///
/// The top 32 bits of a non-negative `f64` are a monotone function of its
/// value, so `k < K` and `|x| < f64::from_bits(K << 32)` select the same
/// inputs — and the float form costs one compare against a constant instead of
/// a shift out of the bit pattern. Every bound is exactly representable (each
/// *is* a `f64` with a zero low word), so nothing moves at the boundary.
const ASIN_TINY: f64 = f64::from_bits(0x3e50_0000_0000_0000);
/// `acos`'s own lower edge, `1.5 * 2^-55`.
const ACOS_TINY: f64 = f64::from_bits(0x3c88_0000_0000_0000);
/// Below this the Taylor band runs; above it, the table.
const TAYLOR_HI: f64 = 0.125;
/// Above this the near-one band runs.
const TABLE_HI: f64 = 0.96875;

/// One packed row's reduced argument and slot offset.
///
/// Returns `(xx, base)` with `xx = |x| - x0` for the band centre
/// `x0 = (floor(256 |x|) + 0.5) / 256`, which is upstream's own slot 0
/// exactly. Every step is exact: `256 u` is a power-of-two scaling, `x0` is a
/// multiple of `2^-9` well inside 53 bits, and `u - x0` is exact by Sterbenz.
/// Exact at every step is what makes this reproduce the reference's `|x| -
/// row[0]` bit for bit rather than merely closely.
#[cube]
pub fn asncs_key(u: f64) -> (f64, u32) {
    let m = u32::cast_from(256.0 * u);
    let x0 = (f64::cast_from(m) + 0.5) * 0.00390625;
    (u - x0, (m - 32u32) * STRIDE)
}

/// One band's polynomial: `p = xx^2 * horner(xx, c) + outer`, then
/// `t = xx * t1 + p`, each one fused multiply-add.
///
/// The fold runs all nine coefficient slots whatever the row's own degree is.
/// The slots past that degree are `+0.0` in the table, and `fma(xx, 0, c) == c`
/// exactly, so `val` is identically zero until the fold reaches the row's real
/// leading coefficient and the chain is bit for bit the reference's shorter
/// one.
///
/// Returns `(t + p, final)` rather than their sum: `asin`'s bands finish with
/// the unfused `final + t`, and `acos`'s apply their own `pi/2` combination on
/// top instead.
#[cube]
pub fn asncs_poly(xx: f64, base: u32, #[comptime] fk: FmaKind) -> (f64, f64) {
    let tab = asncs_tab();
    let row = usize::cast_from(base);
    let mut val = f64::reinterpret(tab[row + 9]);
    val = fma64(xx, val, f64::reinterpret(tab[row + 8]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 7]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 6]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 5]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 4]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 3]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 2]), fk);
    val = fma64(xx, val, f64::reinterpret(tab[row + 1]), fk);
    let p = fma64(xx * xx, val, f64::reinterpret(tab[row + 10]), fk);
    (fma64(xx, f64::reinterpret(tab[row]), p, fk), f64::reinterpret(tab[row + 11]))
}

/// `1/sqrt(z)`'s seed from a bit-pattern lookup, refined by a degree-3
/// Newton-style polynomial in `r = 1 - t^2 z`.
///
/// `r`'s own `t*t*z` is two roundings, not three: the disassembly fuses the
/// final `-(t*t)*z` into the `1 -`, but leaves `t*t` a separate, earlier
/// rounding.
#[cube]
pub fn near_one_root(z: f64, #[comptime] fk: FmaKind) -> f64 {
    let k = u32::cast_from(u64::reinterpret(z) >> 32u64);
    let inroot = inroot_tab();
    let powtwo = powtwo_tab();
    let seed = f64::reinterpret(inroot[usize::cast_from((k & 0x001f_ffffu32) >> 14u32)])
        * f64::reinterpret(powtwo[usize::cast_from(511u32 - (k >> 21u32))]);
    let tt = seed * seed;
    let r = fma64(-tt, z, 1.0, fk);
    let mut poly = fma64(r, ac::RT3, ac::RT2, fk);
    poly = fma64(r, poly, ac::RT1, fk);
    poly = fma64(r, poly, ac::RT0, fk);
    seed * poly
}

/// The near-one band's shared front half: `z`, `c`, `inner` and `p`.
///
/// `asin` and `acos` compute these identically and then diverge in how they
/// split `c` and combine the tail.
#[cube]
pub fn near1_common(u: f64, #[comptime] fk: FmaKind) -> (f64, f64, f64, f64) {
    let z = 0.5 * (1.0 - u);
    let t = near_one_root(z, fk);
    let c = t * z;
    let inner = fma64(t * 0.5, -c, 1.5, fk);
    let mut p = fma64(z, ac::F6, ac::F5, fk);
    p = fma64(z, p, ac::F4, fk);
    p = fma64(z, p, ac::F3, fk);
    p = fma64(z, p, ac::F2, fk);
    p = fma64(z, p, ac::F1, fk);
    (z, c, inner, p * z)
}

/// `asin`'s near-one band, on the magnitude; the caller applies the sign.
#[cube]
pub fn asin_near1(u: f64, #[comptime] fk: FmaKind) -> f64 {
    let (z, c, inner, p) = near1_common(u, fk);
    let y = (c + ac::T24) - ac::T24;
    let t_plus_y = fma64(inner, c, y, fk);
    let cc = fma64(y, -y, z, fk) / t_plus_y;
    let hp1_minus_2cc = fma64(cc, -2.0, ac::HP1, fk);
    let res1 = fma64(y, -2.0, ac::HP0, fk);
    let y_plus_cc_x2 = (y + cc) + (y + cc);
    res1 + fma64(y_plus_cc_x2, -p, hp1_minus_2cc, fk)
}

/// `acos`'s near-one band, sign-aware.
///
/// Both arms use the `2^27` Dekker split — `e_asin.c` never uses `asin`'s own
/// `2^24` constant anywhere in `acos` — and both end in the unfused
/// `res + res`.
#[cube]
pub fn acos_near1(x: f64, #[comptime] fk: FmaKind) -> f64 {
    let (z, c, inner, p) = near1_common(f64::abs(x), fk);
    let y = fma64(ac::T27, c, c, fk) - ac::T27 * c;
    let t_plus_y = fma64(inner, c, y, fk);
    let cc = fma64(y, -y, z, fk) / t_plus_y;
    let mut res = y + (cc + p * (y + cc));
    if x < 0.0 {
        res = (ac::HP0 - y) + ((ac::HP1 - cc) - (y + cc) * p);
    }
    res + res
}

/// `asin(|x|)` for `|x| < 0.125`: `x + (x^2 x) * poly(x^2)`, the same fused
/// shape `atan`'s direct-Taylor band uses.
#[cube]
pub fn asin_taylor(x: f64, x2: f64, #[comptime] fk: FmaKind) -> f64 {
    let mut t = fma64(x2, ac::F6, ac::F5, fk);
    t = fma64(x2, t, ac::F4, fk);
    t = fma64(x2, t, ac::F3, fk);
    t = fma64(x2, t, ac::F2, fk);
    t = fma64(x2, t, ac::F1, fk);
    fma64(x2 * x, t, x, fk)
}

/// `asin(x)`.
#[cube]
pub fn asin(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let u = f64::abs(x);
    // A NaN's exponent field never matches a band below, so the early test is
    // load-bearing rather than decorative: it keeps the payload-preserving
    // `x + x` separate from the *canonical* NaN the `|x| > 1` domain error
    // returns.
    let mut out = x + x;
    if !is_nan64(x) {
        if u < ASIN_TINY {
            out = x;
        } else if u < TAYLOR_HI {
            out = asin_taylor(x, x * x, fk);
        } else if u < TABLE_HI {
            let (xx, base) = asncs_key(u);
            let (t, fin) = asncs_poly(xx, base, fk);
            out = copysign(fin + t, x);
        } else if u < 1.0 {
            out = copysign(asin_near1(u, fk), x);
        } else if u == 1.0 {
            out = copysign(ac::HP0, x);
        } else {
            // `__ieee754_asin`'s own `(x-x)/(x-x)` is dead code on this
            // platform: the exported wrapper checks the domain itself and
            // returns this canonical NaN before ever calling the `_fma` entry
            // point.
            out = nan64();
        }
    }
    out
}

/// `acos(x)`.
#[cube]
pub fn acos(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let u = f64::abs(x);
    // See `asin`'s identical early test.
    let mut out = x + x;
    if !is_nan64(x) {
        if u < ACOS_TINY {
            out = ac::HP0;
        } else if u < TAYLOR_HI {
            let x2 = x * x;
            let mut t = fma64(x2, ac::F6, ac::F5, fk);
            t = fma64(x2, t, ac::F4, fk);
            t = fma64(x2, t, ac::F3, fk);
            t = fma64(x2, t, ac::F2, fk);
            t = fma64(x2, t, ac::F1, fk);
            let r = ac::HP0 - x;
            let inner = ((ac::HP0 - r) - x) + ac::HP1;
            out = r + fma64(x2 * x, -t, inner, fk);
        } else if u < TABLE_HI {
            let (xx, base) = asncs_key(u);
            let (t, fin) = asncs_poly(xx, base, fk);
            // `e_asin.c`'s `acos` bands never add the row's `final` themselves;
            // they apply their own `pi/2 -+ ..` combination to it instead.
            out = (ac::HP0 + fin) + (ac::HP1 + t);
            if x > 0.0 {
                out = (ac::HP0 - fin) + (ac::HP1 - t);
            }
        } else if u < 1.0 {
            out = acos_near1(x, fk);
        } else if u == 1.0 {
            out = 2.0 * ac::HP0;
            if x > 0.0 {
                out = 0.0;
            }
        } else {
            // See `asin`: dead code on this platform.
            out = nan64();
        }
    }
    out
}
