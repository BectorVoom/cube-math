//! `erfc(x)`, correctly rounded.
//!
//! Like [`super::erf`], this is CORE-MATH's routine, so matching it is not a
//! claim about *this* platform's `libm`: any correctly-rounded `erfc` returns
//! the same bits.
//!
//! # Why `erfc` is a separate algorithm and not `1 - erf`
//!
//! Because that subtraction is catastrophic. `erfc(6)` is about `2e-17`, so
//! `1 - erf(6)` computes it as the difference of two numbers that agree to
//! sixteen digits and returns noise; by `x = 27` the true value is `2^-1074`
//! and the subtraction returns exactly zero. So the routine splits:
//!
//! * `x < 0` and `x <= 0x1.713786d9c7c09p+1` — where `erfc` is still `O(1)`
//!   and the cancellation is harmless — reuse [`super::erf::erf_fast()`] and
//!   one `fast_two_sum` against one.
//! * above that, `erfc(x) = exp(-x^2) p(1/x)` with a Chebyshev fit `p` per
//!   interval of `1/x`, and `exp(-x^2)` computed to 74 bits by a
//!   double-double exponential of its own — because a relative error of
//!   `2^-53` in `exp(-x^2)` is a relative error of `2^-53` in the answer, and
//!   the answer has to be correctly rounded.
//!
//! Both branches produce a double-double with an *absolute* error bound, and
//! the driver rounds it when the bound settles the rounding and calls the
//! accurate path when it does not.
//!
//! # The exponential is this file's own
//!
//! [`exp_1()`] and [`exp_accurate()`] are not [`super::exp`] with more digits;
//! they are different functions with different contracts. `exp_1` returns a
//! double-double good to `2^-74`, which `super::exp` — correct to the last bit
//! of a single `double` and no further — cannot supply at any price.
//! `exp_accurate` goes further still and returns the exponent *separately*,
//! because its argument reaches `-742` and the result would underflow long
//! before it could be scaled.
//!
//! Both policy axes are accepted and have no effect. See [`super::erf`].

use cubecl::prelude::*;

use crate::bits::opaque64;
use crate::config::MathConfig;
use crate::double::dd::{a_mul, d_mul, fast_sum, fast_two_sum, s_mul, two_sum};
use crate::double::erf::{erf_accurate, erf_fast};
use crate::double::exact::ldexp;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::{
    erfc_e2_tab, erfc_exc_acc2_tab, erfc_exc_acc_tab, erfc_exc_tab, erfc_q1_tab, erfc_t1_tab,
    erfc_t2_tab, erfc_t_tab, erfc_tacc_tab,
};

/// `2^12 / ln(2)`, [`exp_1()`]'s reduction scale.
const INVLOG2: f64 = f64::from_bits(0x40b71547652b82fe);
/// `ln(2)/2^12`, high part.
const LOG2H: f64 = f64::from_bits(0x3f262e42fefa39ef);
/// `ln(2)/2^12`, low part.
const LOG2L: f64 = f64::from_bits(0x3bbabc9e3b39803f);
/// `2^52`, the add-and-subtract round-to-nearest constant.
const TWO52: f64 = 4503599627370496.0;

/// Above this the asymptotic branch takes over.
pub const THRESHOLD1: f64 = f64::from_bits(0x400713786d9c7c09);
/// At or above this, `1 - erf(x)` is exact by Sterbenz, so the `fast_two_sum`
/// contributes no error of its own.
const STERBENZ: f64 = f64::from_bits(0x3fde861fbb24c00a);
/// The extra absolute error the `x < 0` reflection costs.
const EPS_NEG: f64 = f64::from_bits(0x3994000000000000);
/// The same for `0 <= x < STERBENZ`.
const EPS_POS: f64 = f64::from_bits(0x3974000000000000);

/// Above this `erfc(x) < 2^-970` and the fast path's low word would underflow,
/// so it defers to the accurate path outright.
const ASYMPT_MAX: f64 = f64::from_bits(0x4039db1bb14e15ca);
/// The asymptotic branch's relative error bound.
const ERR_ASYMPT: f64 = f64::from_bits(0x3bbd900000000000);
/// Below this, `ERR_ASYMPT * h` would itself underflow.
const UFLOW_GUARD: f64 = f64::from_bits(0x044151b9a3fdd5c9);
/// `2^-1022`, the overestimate used instead of [`UFLOW_GUARD`]'s product.
const MIN_NORMAL: f64 = f64::from_bits(0x0010000000000000);
/// Where the accurate path switches to the asymptotic formula — higher than
/// the fast path's threshold, because the accurate polynomial keeps its digits
/// further out.
const ACC_SPLIT: f64 = f64::from_bits(0x3ffb59ffb450828c);
/// Half an ulp of one, used to step just off `1` and `2`.
const HALF_ULP1: f64 = f64::from_bits(0x3c90000000000000);

/// At or below this `erfc` rounds to two.
pub const NEG_LIMIT: u64 = 0xc017744f8f74e94b;
/// At or above this `erfc(x) < 2^-1075`.
pub const POS_LIMIT: u64 = 0x403b39dc41e48bfd;
/// Below this magnitude `erfc` rounds to one.
pub const UNIT_LIMIT: f64 = f64::from_bits(0x3c8c5bf891b4ef6a);

/// Rows in each hard-case table.
const EXC_ROWS: u32 = 22;
/// Rows in the accurate path's negative-argument hard-case table.
const EXC_ACC_ROWS: u32 = 17;
/// Rows in the accurate path's positive-argument hard-case table.
const EXC_ACC2_ROWS: u32 = 29;

/// `x` rounded to the nearest integer, ties to even.
///
/// The add-and-subtract trick rather than a `round` intrinsic, for the reason
/// [`crate::double::exact::rint()`] gives: the backends disagree about ties and
/// the addition does not. Every caller here has `|x| < 2^22`, so the overflow
/// guard that function carries is not needed.
#[cube]
pub fn rint_small(x: f64) -> f64 {
    let a = f64::abs(x);
    crate::double::exact::copysign((a + TWO52) - TWO52, x)
}

// ---------------------------------------------------------------------------
// The double-double exponential
// ---------------------------------------------------------------------------

/// `exp(z)` for `|z| < 2^-12.88`, as a double-double.
#[cube]
pub fn q_1(zh: f64, zl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let q1 = erfc_q1_tab();
    let z = zh + zl;
    let mut q = fma64(f64::reinterpret(q1[4]), zh, f64::reinterpret(q1[3]), fk);
    q = fma64(q, z, f64::reinterpret(q1[2]), fk);
    let (hi0, lo0) = fast_two_sum(f64::reinterpret(q1[1]), q * z);
    let (hi1, lo1) = d_mul(zh, zl, hi0, lo0, fk);
    fast_sum(f64::reinterpret(q1[0]), hi1, lo1)
}

/// `exp(xh + xl)` to a relative accuracy of `2^-74`, as a double-double.
///
/// Two 64-entry tables rather than one: `2^(K/4096)` splits as
/// `2^(K>>12) 2^(i2/64) 2^(i1/4096)`, so 4096 table entries become 128.
#[cube]
pub fn exp_1(xh: f64, xl: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let k = rint_small(xh * INVLOG2);
    let (kh, kl) = s_mul(k, LOG2H, LOG2L, fk);
    let (yh, yl0) = fast_two_sum(xh - kh, xl);
    let yl = yl0 - kl;

    // `k` fits an `i32`: `|xh| < 676` here, so `|k| < 4 * 10^6`.
    let ki = i32::cast_from(k);
    let kb = u32::reinterpret(ki);
    let m = (ki >> 12i32) + 1023i32;
    // The masks make the arithmetic and logical shifts agree, so the unsigned
    // form below is the signed one the C source writes.
    let i2 = usize::cast_from((kb >> 6u32) & 0x3fu32);
    let i1 = usize::cast_from(kb & 0x3fu32);

    let t1 = erfc_t1_tab();
    let t2 = erfc_t2_tab();
    let (hi0, lo0) = d_mul(
        f64::reinterpret(t2[i1 * 2]),
        f64::reinterpret(t2[i1 * 2 + 1]),
        f64::reinterpret(t1[i2 * 2]),
        f64::reinterpret(t1[i2 * 2 + 1]),
        fk,
    );
    let (qh, ql) = q_1(yh, yl, fk);
    let (hi, lo) = d_mul(hi0, lo0, qh, ql, fk);
    let df = f64::reinterpret(u64::cast_from(m) << 52u64);
    (hi * df, lo * df)
}

/// One step of [`exp_accurate()`]'s double-double Horner fold.
#[cube]
pub fn exp_acc_step(h: f64, l: f64, yh: f64, yl: f64, c: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (th, tl0) = a_mul(h, yh, fk);
    let tl1 = fma64(h, yl, tl0, fk);
    let tl = fma64(l, yh, tl1, fk);
    let (nh, nl) = fast_two_sum(c, th);
    (nh, nl + tl)
}

/// [`exp_acc_step()`] with a double-double coefficient `chi + clo`.
#[cube]
pub fn exp_acc_step2(
    h: f64,
    l: f64,
    yh: f64,
    yl: f64,
    chi: f64,
    clo: f64,
    #[comptime] fk: FmaKind,
) -> (f64, f64) {
    let (th, tl0) = a_mul(h, yh, fk);
    let tl1 = fma64(h, yl, tl0, fk);
    let tl = fma64(l, yh, tl1, fk);
    let (nh, nl) = fast_two_sum(chi, th);
    (nh, nl + (tl + clo))
}

/// `exp(xh + xl)` as `2^e (h + l)` to about 104 bits.
///
/// The exponent travels alongside the significand rather than inside it: this
/// argument reaches `-742`, and the result would underflow long before it
/// could be scaled.
#[cube]
pub fn exp_accurate(xh: f64, xl: f64, #[comptime] fk: FmaKind) -> (f64, f64, i32) {
    /// `1/ln(2)`.
    const INVLOG2ACC: f64 = f64::from_bits(0x3ff71547652b82fe);
    /// `ln(2)`, high part.
    const LOG2HACC: f64 = f64::from_bits(0x3fe62e42fefa39ef);
    /// `ln(2) - LOG2HACC` to 38 bits, so `k LOG2LACC` is exact for the 11-bit
    /// `k` this reduction produces.
    const LOG2LACC: f64 = f64::from_bits(0x3c7abc9e3b398000);
    /// What is left of `ln(2)` after the two above.
    const LOG2TINY: f64 = f64::from_bits(0x398f97b57a079a19);

    let e2 = erfc_e2_tab();
    let kf = rint_small(xh * INVLOG2ACC);
    let k = i32::cast_from(kf);
    // Exact by Sterbenz: `|xh| >= 2.92` forces `|k| >= 4`, which puts
    // `xh / (k ln 2)` inside `[1 - 1/(2|k|), 1 + 1/(2|k|)]`.
    let yh0 = fma64(-kf, LOG2HACC, xh, fk);
    let (th0, tl0) = two_sum(-kf * LOG2LACC, xl);
    let (yh, yl0) = fast_two_sum(yh0, th0);
    let yl = fma64(-kf, LOG2TINY, yl0 + tl0, fk);

    // Degrees 19 down to 16 ignore `yl`: its contribution there is below
    // `2^-104` and would be rounded away.
    let mut h0 = f64::reinterpret(e2[27]);
    h0 = fma64(h0, yh, f64::reinterpret(e2[26]), fk);
    h0 = fma64(h0, yh, f64::reinterpret(e2[25]), fk);
    h0 = fma64(h0, yh, f64::reinterpret(e2[24]), fk);

    let (th, tl1) = a_mul(h0, yh, fk);
    let tl = fma64(h0, yl, tl1, fk);
    let (h1, l1a) = fast_two_sum(f64::reinterpret(e2[23]), th);
    let l1 = l1a + tl;

    let (h2, l2) = exp_acc_step(h1, l1, yh, yl, f64::reinterpret(e2[22]), fk);
    let (h3, l3) = exp_acc_step(h2, l2, yh, yl, f64::reinterpret(e2[21]), fk);
    let (h4, l4) = exp_acc_step(h3, l3, yh, yl, f64::reinterpret(e2[20]), fk);
    let (h5, l5) = exp_acc_step(h4, l4, yh, yl, f64::reinterpret(e2[19]), fk);
    let (h6, l6) = exp_acc_step(h5, l5, yh, yl, f64::reinterpret(e2[18]), fk);
    let (h7, l7) = exp_acc_step(h6, l6, yh, yl, f64::reinterpret(e2[17]), fk);
    let (h8, l8) = exp_acc_step(h7, l7, yh, yl, f64::reinterpret(e2[16]), fk);

    let (h9, l9) =
        exp_acc_step2(h8, l8, yh, yl, f64::reinterpret(e2[14]), f64::reinterpret(e2[15]), fk);
    let (h10, l10) =
        exp_acc_step2(h9, l9, yh, yl, f64::reinterpret(e2[12]), f64::reinterpret(e2[13]), fk);
    let (h11, l11) =
        exp_acc_step2(h10, l10, yh, yl, f64::reinterpret(e2[10]), f64::reinterpret(e2[11]), fk);
    let (h12, l12) =
        exp_acc_step2(h11, l11, yh, yl, f64::reinterpret(e2[8]), f64::reinterpret(e2[9]), fk);
    let (h13, l13) =
        exp_acc_step2(h12, l12, yh, yl, f64::reinterpret(e2[6]), f64::reinterpret(e2[7]), fk);
    let (h14, l14) =
        exp_acc_step2(h13, l13, yh, yl, f64::reinterpret(e2[4]), f64::reinterpret(e2[5]), fk);
    let (h15, l15) =
        exp_acc_step2(h14, l14, yh, yl, f64::reinterpret(e2[2]), f64::reinterpret(e2[3]), fk);
    let (h16, l16) =
        exp_acc_step2(h15, l15, yh, yl, f64::reinterpret(e2[0]), f64::reinterpret(e2[1]), fk);

    (h16, l16, k)
}

// ---------------------------------------------------------------------------
// The fast path
// ---------------------------------------------------------------------------

/// One step of the asymptotic polynomial's double-double Horner fold.
#[cube]
pub fn asym_step(zh: f64, zl: f64, uh: f64, ul: f64, c: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (h, l) = d_mul(zh, zl, uh, ul, fk);
    let (nh, nl) = fast_two_sum(c, h);
    (nh, nl + l)
}

/// Which Chebyshev fit of `erfc(x) exp(x^2) x` covers a given `1/x`.
///
/// The thresholds increase, so the row is the *count* of thresholds below
/// `yh` — written as a sum rather than a search loop, which is both shorter
/// and branch-free.
#[cube]
pub fn asympt_row(yh: f64) -> u32 {
    let mut i = 0u32;
    i = i + u32::cast_from(yh > f64::from_bits(0x3fbd500000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fc59da6ca291ba6));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fcbc00000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fd0c00000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fd3800000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fd6300000000000));
    i
}

/// `erfc(x)` as `(h, l, err)` for `x > 0x1.713786d9c7c09p+1`, by the
/// asymptotic formula. `err` is an *absolute* bound.
#[cube]
pub fn erfc_asympt_fast(x: f64, #[comptime] fk: FmaKind) -> (f64, f64, f64) {
    let (xh, xl) = a_mul(x, x, fk);
    let (eh, el) = exp_1(-xh, -xl, fk);

    // `1/x` as a double-double: one divide, then one Newton step
    // `y -> y + y(1 - x y)`, which the fused multiply-add makes exact enough
    // for 103 bits.
    let yh = 1.0 / x;
    let yl = yh * fma64(-x, yh, 1.0, fk);

    let tab = erfc_t_tab();
    let p = usize::cast_from(asympt_row(yh) * 13u32);

    let (uh, ul0) = a_mul(yh, yh, fk);
    let ul = fma64(2.0 * yh, yl, ul0, fk);

    let mut zh0 = f64::reinterpret(tab[p + 12]);
    zh0 = fma64(zh0, uh, f64::reinterpret(tab[p + 11]), fk);
    zh0 = fma64(zh0, uh, f64::reinterpret(tab[p + 10]), fk);
    let (sh, sl) = s_mul(zh0, uh, ul, fk);
    let (z9h, z9l0) = fast_two_sum(f64::reinterpret(tab[p + 9]), sh);
    let z9l = z9l0 + sl;

    let (z8h, z8l) = asym_step(z9h, z9l, uh, ul, f64::reinterpret(tab[p + 8]), fk);
    let (z7h, z7l) = asym_step(z8h, z8l, uh, ul, f64::reinterpret(tab[p + 7]), fk);
    let (z6h, z6l) = asym_step(z7h, z7l, uh, ul, f64::reinterpret(tab[p + 6]), fk);
    let (z5h, z5l) = asym_step(z6h, z6l, uh, ul, f64::reinterpret(tab[p + 5]), fk);
    let (z4h, z4l) = asym_step(z5h, z5l, uh, ul, f64::reinterpret(tab[p + 4]), fk);
    let (z3h, z3l) = asym_step(z4h, z4l, uh, ul, f64::reinterpret(tab[p + 3]), fk);
    let (z2h, z2l) = asym_step(z3h, z3l, uh, ul, f64::reinterpret(tab[p + 2]), fk);

    let (fh, fl) = d_mul(z2h, z2l, uh, ul, fk);
    let (z0h, z0l0) = fast_two_sum(f64::reinterpret(tab[p]), fh);
    let z0l = z0l0 + (fl + f64::reinterpret(tab[p + 1]));

    let (vh, vl) = d_mul(z0h, z0l, yh, yl, fk);
    let (h, l) = d_mul(vh, vl, eh, el, fk);

    // `MIN_NORMAL` is an overestimate, but unlike the product it cannot itself
    // underflow.
    let err = select(h >= UFLOW_GUARD, ERR_ASYMPT * h, MIN_NORMAL);

    let mut rh = h;
    let mut rl = l;
    let mut re = err;
    if x >= ASYMPT_MAX {
        // Below `2^-970` the low word would underflow; hand the whole thing to
        // the accurate path by returning a bound no rounding test passes.
        rh = 0.0;
        rl = 0.0;
        re = 1.0;
    }
    (rh, rl, re)
}

/// `erfc(x)` as `(h, l, err)` with `err` an absolute bound, for
/// `-0x1.7744f8f74e94bp+2 < x < 0x1.b39dc41e48bfdp+4`.
#[cube]
pub fn erfc_fast(x: f64, #[comptime] fk: FmaKind) -> (f64, f64, f64) {
    let mut rh = 0.0 + x;
    let mut rl = 0.0 * x;
    let mut re = 0.0 * x;
    if x < 0.0 {
        // `erfc(x) = 1 + erf(-x)`, and `h <= 2` keeps the sum's own error at
        // `2^-104` — small enough that the stated constant absorbs it.
        let (h0, l0, err0) = erf_fast(-x, fk);
        let (h1, tt) = fast_two_sum(1.0, h0);
        rh = h1;
        rl = tt + l0;
        re = err0 * h0 + EPS_NEG;
    } else if x <= THRESHOLD1 {
        let (h0, l0, err0) = erf_fast(x, fk);
        let (h1, tt) = fast_two_sum(1.0, -h0);
        rh = h1;
        rl = tt - l0;
        // Above `STERBENZ`, `1 - h` is exact and `tt` is zero, so the only
        // error left is the one `erf_fast` reported.
        re = err0 * h0 + EPS_POS;
        if x >= STERBENZ {
            re = err0 * h0;
        }
    } else {
        let (h0, l0, err0) = erfc_asympt_fast(x, fk);
        rh = h0;
        rl = l0;
        re = err0;
    }
    (rh, rl, re)
}

// ---------------------------------------------------------------------------
// The accurate path
// ---------------------------------------------------------------------------

/// Which of the ten accurate fits covers a given `1/x`. See [`asympt_row()`].
#[cube]
pub fn asympt_acc_row(yh: f64) -> u32 {
    let mut i = 0u32;
    i = i + u32::cast_from(yh > f64::from_bits(0x3fb4500000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fbe000000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fc3f00000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fc9500000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fcf500000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fd3100000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fd7100000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fdbc00000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fe0b00000000000));
    i = i + u32::cast_from(yh > f64::from_bits(0x3fe3000000000000));
    i
}

/// One step of the accurate asymptotic polynomial's fold.
#[cube]
pub fn acc_step(zh: f64, zl: f64, uh: f64, ul: f64, c: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (h, l0) = a_mul(zh, uh, fk);
    let l1 = fma64(zh, ul, l0, fk);
    let l = fma64(zl, uh, l1, fk);
    let (nh, nl) = two_sum(c, h);
    (nh, nl + l)
}

/// [`acc_step()`] with a double-double coefficient `chi + clo`.
#[cube]
pub fn acc_step2(
    zh: f64,
    zl: f64,
    uh: f64,
    ul: f64,
    chi: f64,
    clo: f64,
    #[comptime] fk: FmaKind,
) -> (f64, f64) {
    let (h, l0) = a_mul(zh, uh, fk);
    let l1 = fma64(zh, ul, l0, fk);
    let l = fma64(zl, uh, l1, fk);
    let (nh, nl) = two_sum(chi, h);
    (nh, nl + (l + clo))
}

/// The accurate asymptotic branch, for `1.70 < x < 27.3`.
#[cube]
pub fn erfc_asympt_accurate(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let fk = comptime!(cfg.fma());
    let (xh, xl) = a_mul(x, x, fk);
    let (eh, el, e) = exp_accurate(-xh, -xl, fk);

    let yh = 1.0 / x;
    let yl = yh * fma64(-x, yh, 1.0, fk);

    let i = asympt_acc_row(yh);
    let tab = erfc_tacc_tab();
    let base = usize::cast_from(i * 30u32);

    let (uh, ul0) = a_mul(yh, yh, fk);
    let ul = fma64(2.0 * yh, yl, ul0, fk);

    // Degree `29 + 2i`, leading coefficient at slot `20 + i`. The trip count
    // depends on the band, so this one stays a loop where the rest unroll.
    let mut zh = f64::reinterpret(tab[base + usize::cast_from(20u32 + i)]);
    let mut zl = zh - zh;
    let mut idx = 19u32 + i;
    while idx >= 12u32 {
        let (nh, nl) = acc_step(
            zh,
            zl,
            uh,
            ul,
            f64::reinterpret(tab[base + usize::cast_from(idx)]),
            fk,
        );
        zh = nh;
        zl = nl;
        idx = idx - 1u32;
    }

    let (a10h, a10l) =
        acc_step2(zh, zl, uh, ul, f64::reinterpret(tab[base + 10]), f64::reinterpret(tab[base + 11]), fk);
    let (a8h, a8l) =
        acc_step2(a10h, a10l, uh, ul, f64::reinterpret(tab[base + 8]), f64::reinterpret(tab[base + 9]), fk);
    let (a6h, a6l) =
        acc_step2(a8h, a8l, uh, ul, f64::reinterpret(tab[base + 6]), f64::reinterpret(tab[base + 7]), fk);
    let (a4h, a4l) =
        acc_step2(a6h, a6l, uh, ul, f64::reinterpret(tab[base + 4]), f64::reinterpret(tab[base + 5]), fk);
    let (a2h, a2l) =
        acc_step2(a4h, a4l, uh, ul, f64::reinterpret(tab[base + 2]), f64::reinterpret(tab[base + 3]), fk);
    let (a0h, a0l) =
        acc_step2(a2h, a2l, uh, ul, f64::reinterpret(tab[base]), f64::reinterpret(tab[base + 1]), fk);

    let (vh, vl0) = a_mul(a0h, yh, fk);
    let vl1 = fma64(a0h, yl, vl0, fk);
    let vl = fma64(a0l, yh, vl1, fk);

    // Normalising before the last product keeps the number of hard cases down.
    let (nh, nl) = fast_two_sum(vh, vl);
    let (ph, pl0) = a_mul(nh, eh, fk);
    let pl1 = fma64(nh, el, pl0, fk);
    let pl = fma64(nl, eh, pl1, fk);

    let ef = f64::cast_from(e);
    let mut res = ldexp(ph + pl, ef, cfg);
    if res < MIN_NORMAL {
        // In the subnormal range the scaling above rounds twice. Recover the
        // discarded part and add it back at the right exponent.
        let corr = ph - ldexp(res, -ef, cfg) + pl;
        res = res + ldexp(corr, ef, cfg);
    }

    // The one argument whose result is both subnormal and a hard case.
    if u64::reinterpret(x) == 0x403a8f7bfbd15495u64 {
        res = fma64(f64::from_bits(1), -0.25, f64::from_bits(0x000667bd620fd95b), fk);
    }
    // The general hard cases, as a trailing overwrite. See
    // [`super::erf::erf_accurate_tiny()`] on why this is not an early exit.
    let exc = erfc_exc_tab();
    for k in 0..EXC_ROWS {
        let row = usize::cast_from(k * 3u32);
        if f64::reinterpret(exc[row]) == x {
            res = f64::reinterpret(exc[row + 1]) + f64::reinterpret(exc[row + 2]);
        }
    }
    res
}

/// `erfc(x)` for the inputs the rounding test could not settle.
#[cube]
pub fn erfc_accurate(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let fk = comptime!(cfg.fma());
    let mut out = 0.0 + x;
    if x < 0.0 {
        let (h0, l0) = erf_accurate(-x, fk);
        let (h1, tt) = fast_two_sum(1.0, h0);
        out = h1 + (tt + l0);
        let exc = erfc_exc_acc_tab();
        for k in 0..EXC_ACC_ROWS {
            let row = usize::cast_from(k * 3u32);
            if f64::reinterpret(exc[row]) == x {
                out = f64::reinterpret(exc[row + 1]) + f64::reinterpret(exc[row + 2]);
            }
        }
    } else if x <= ACC_SPLIT {
        let (h0, l0) = erf_accurate(x, fk);
        let (h1, tt) = fast_two_sum(1.0, -h0);
        out = h1 + (tt - l0);
        let exc = erfc_exc_acc2_tab();
        for k in 0..EXC_ACC2_ROWS {
            let row = usize::cast_from(k * 3u32);
            if f64::reinterpret(exc[row]) == x {
                out = f64::reinterpret(exc[row + 1]) + f64::reinterpret(exc[row + 2]);
            }
        }
    } else {
        out = erfc_asympt_accurate(x, cfg);
    }
    out
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// `erfc(x)`.
#[cube]
pub fn erfc(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let tb = u64::reinterpret(x);
    let at = tb & 0x7fff_ffff_ffff_ffffu64;

    let mut out = 0.0 + x;
    let mut done = at >= 0x7ff0_0000_0000_0000u64;
    if done {
        // An infinity is `2` or `0` by sign; a NaN propagates its payload.
        out = x + x;
        if at == 0x7ff0_0000_0000_0000u64 {
            out = select(tb >= 0x8000_0000_0000_0000u64, 2.0, 0.0);
        }
    } else if tb >= 0x8000_0000_0000_0000u64 {
        if tb >= NEG_LIMIT {
            out = 2.0 - HALF_ULP1;
            done = true;
        } else if -UNIT_LIMIT - UNIT_LIMIT <= x {
            out = fma64(-x, HALF_ULP1, 1.0, fk);
            done = true;
        }
    } else if at >= POS_LIMIT {
        // Below `2^-1075`: zero, or the smallest subnormal under a directed
        // rounding mode.
        out = f64::from_bits(1) * 0.25;
        done = true;
    } else if x <= UNIT_LIMIT {
        out = fma64(-x, HALF_ULP1, 1.0, fk);
        done = true;
    }

    if !done {
        let (h, l, err) = erfc_fast(x, fk);
        let left = h + (l - err);
        let right = h + (l + err);
        out = left;
        if left != right {
            out = erfc_accurate(x, cfg);
        }
    }
    out
}
