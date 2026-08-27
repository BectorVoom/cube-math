//! Double-double arithmetic, in kernel form.
//!
//! A double-double is an unevaluated sum `h + l` with `|l| <= ulp(h)/2`, which
//! carries about 106 significand bits. The functions that need it need it for
//! one of two reasons: either the reference algorithm itself is written in
//! double-double (`erf` and `erfc` are *correctly rounded*, and deciding the
//! last bit of a `double` means knowing the true value to rather more than a
//! `double`'s worth of digits), or a single step of it is — `atan2`'s division
//! residual, `asin`'s reciprocal square root.
//!
//! Each operation is exact, or exact but for a stated dropped term, and each
//! has a precondition that is part of its contract rather than a nicety:
//! [`fast_two_sum()`] is wrong without `|a| >= |b|`, and is used anyway
//! wherever that ordering is known, because it costs half what [`two_sum()`]
//! costs.
//!
//! These are the same sequences `rmath`'s own `dd` module spells, and
//! deliberately so. Every IEEE-754 operation rounds identically on every
//! conforming device, so replaying the schedule one thread per element returns
//! the scalar bits exactly — which is the whole reason a bit-exact kernel is
//! possible at all.
//!
//! The multiplications take a [`FmaKind`], because it is the *fused* multiply
//! that makes them exact: on a backend whose `fma` rounds twice, the residual
//! `fma(a, b, -hi)` is not the product's error but a rounded version of it,
//! and every digit past the first `double` is then wrong. See [`crate::fma`].

use cubecl::prelude::*;

use crate::fma::{FmaKind, fma64};

/// `a + b` as a double-double, **assuming `|a| >= |b|`**. Exact.
#[cube]
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let e = hi - a;
    (hi, b - e)
}

/// `a + b` as a double-double, for any `a` and `b`. Exact, at twice the cost.
#[cube]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let aa = hi - b;
    let bb = hi - aa;
    (hi, (a - aa) + (b - bb))
}

/// `a * b` as a double-double. Exact, and it is the fused multiply-add that
/// makes it so.
#[cube]
pub fn a_mul(a: f64, b: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let hi = a * b;
    (hi, fma64(a, b, -hi, fk))
}

/// `a * (bh + bl)`, dropping the rounding error of `a * bl`.
#[cube]
pub fn s_mul(a: f64, bh: f64, bl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (hi, lo) = a_mul(a, bh, fk);
    (hi, fma64(a, bl, lo, fk))
}

/// `(ah + al) * (bh + bl)`, dropping the `al * bl` term.
#[cube]
pub fn d_mul(ah: f64, al: f64, bh: f64, bl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (hi, lo0) = a_mul(ah, bh, fk);
    let lo1 = fma64(ah, bl, lo0, fk);
    (hi, fma64(al, bh, lo1, fk))
}

/// `a + (bh + bl)`, **assuming `|a| >= |bh|`**.
#[cube]
pub fn fast_sum(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = fast_two_sum(a, bh);
    (hi, lo + bl)
}

// ---------------------------------------------------------------------------
// The CORE-MATH shapes
// ---------------------------------------------------------------------------
//
// `asinh`, `acosh` and `atan2f` are ports of CORE-MATH routines, whose
// double-double layer (glibc's `ddcoremath.h`) is *not* the one above: it
// accumulates cross terms in a different order and normalises at different
// points. Algebraically the two agree; bit for bit they do not, and these are
// ports. So the shapes those routines use live here under their upstream
// names, beside rather than instead of the ones the rest of the crate uses.

/// `(xh + xl) + (ch + cl)`, without assuming an ordering. Upstream's `adddd`.
#[cube]
pub fn add_dd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let s = xh + ch;
    let d = s - xh;
    (s, ((ch - d) + (xh + (d - s))) + (xl + cl))
}

/// `(xh + xl) (ch + cl)`, dropping the `xl cl` term. Upstream's `muldd_acc`
/// — Joldeş, Muller and Popescu's DWTimesDW1, whose relative error is bounded
/// by `5 u^2`.
#[cube]
pub fn mul_dd_acc(xh: f64, xl: f64, ch: f64, cl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = fma64(ch, xh, -ahhh, fk) + alhh + ahlh;
    let hi = ahhh + ahhl;
    (hi, (ahhh - hi) + ahhl)
}

/// [`mul_dd_acc()`] closing with a true [`fast_two_sum()`] instead of its own
/// two lines. Upstream's `muldd_acc2`.
///
/// The difference is real and upstream documents it: the trailing pair
/// `ch = ahhh + ahhl; l = (ahhh - ch) + ahhl` emulates a *variant* of
/// `fasttwosum` with the two subtractions the other way round, and the two
/// disagree when the `|x| >= |y|` precondition does not hold.
#[cube]
pub fn mul_dd_acc2(xh: f64, xl: f64, ch: f64, cl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let ahlh = ch * xl;
    let alhh = cl * xh;
    let ahhh = ch * xh;
    let ahhl = fma64(ch, xh, -ahhh, fk) + alhh + ahlh;
    fast_two_sum(ahhh, ahhl)
}

/// `(xh + xl) c` for a single `c`. Upstream's `mulddd`.
#[cube]
pub fn mul_ddd(xh: f64, xl: f64, c: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let ahlh = c * xl;
    let ahhh = c * xh;
    let ahhl = fma64(c, xh, -ahhh, fk) + ahlh;
    let hi = ahhh + ahhl;
    (hi, (ahhh - hi) + ahhl)
}
