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
