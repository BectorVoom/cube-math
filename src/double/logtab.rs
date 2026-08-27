//! The table-driven logarithm `asinh` and `acosh` share.
//!
//! Neither function computes a logarithm the way [`super::ln`] does. Both need
//! `ln` of a *double-double* argument, to more precision than a single `f64`
//! holds, and both need a second, sharper evaluation for the inputs the first
//! cannot round. That is a different routine with a different contract, and it
//! belongs to the two functions that use it rather than to `ln`.
//!
//! Two tiers, matching CORE-MATH's:
//!
//! * [`log_index()`] and [`log_poly()`] — the fast tier. One 1024-way table
//!   reached through a correction step, so that the reduced argument is within
//!   `2^-11` of one, and a degree-5 polynomial on top. Good to about `2^-68`.
//! * [`refine_core()`] — the accurate tier. A four-stage `2^-k` ladder whose
//!   logarithms are tabulated to *triple*-double, and a degree-3 series in the
//!   remaining `2^-16`. Good to about `2^-159`.
//!
//! Both tiers are shared verbatim between the two functions; what differs is
//! how each assembles the pieces, and that stays with each function.

use cubecl::prelude::*;

use crate::double::dd::{add_dd, fast_two_sum, mul_dd_acc, mul_dd_acc2};
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::{asincosh_refine_tab, asincosh_tab};

/// `ln(2)`, high part, as the fast tier splits it.
pub const L2H: f64 = f64::from_bits(0x3fe62e42fefa3800);
/// `ln(2)`, low part.
pub const L2L: f64 = f64::from_bits(0x3d2ef35793c76730);
/// `log2(e)`, which turns the fast tier's answer into a ladder index.
pub const LOG2E: f64 = f64::from_bits(0x3ff71547652b82fe);
/// `ln(2)/2`, high part, as the accurate tier splits it.
pub const L20: f64 = f64::from_bits(0x3fd62e42fefa3800);
/// `ln(2)/2`, middle part.
pub const L21: f64 = f64::from_bits(0x3d1ef35793c76800);
/// `ln(2)/2`, low part.
pub const L22: f64 = f64::from_bits(0xba49ff0342542fc3);
/// The offset that turns `log2(z) - e` into a ladder index in `[0, 2^16)`.
pub const LADDER_BIAS: f64 = f64::from_bits(0x3ff0000800000000);

/// Offsets into [`asincosh_tab()`].
const B: u32 = 0;
/// `1/c` for the first stage, 33 slots.
const R1: u32 = 64;
/// `1/c` for the second stage, 33 slots.
const R2: u32 = R1 + 33;
/// `log c` for the first stage, 33 rows of 2.
const L1: u32 = R2 + 33;
/// `log c` for the second stage, 33 rows of 2.
const L2: u32 = L1 + 66;
/// The fast tier's polynomial, 5 slots.
const CP: u32 = L2 + 66;

/// Offsets into [`asincosh_refine_tab()`].
const T1: u32 = 0;
/// The second ladder stage, 16 slots.
const T2: u32 = 17;
/// The third, 16 slots.
const T3: u32 = T2 + 16;
/// The fourth, 16 slots.
const T4: u32 = T3 + 16;
/// The ladder's logarithms, four stages of 17 rows of 3.
const LL: u32 = T4 + 16;
/// The accurate tier's `log1p` series, 3 double-double pairs.
const RCH: u32 = LL + 204;
/// Its plain tail, 3 slots.
const RCL: u32 = RCH + 6;

/// One `f64` out of a `u64` table.
#[cube]
pub fn at(tab: &Array<u64>, i: u32) -> f64 {
    f64::reinterpret(tab[usize::cast_from(i)])
}

/// The fast tier's index work: `(ed, dx, i1, i2)`.
///
/// `off` is the exponent bias the caller wants removed — `0x3ff` normally, and
/// `0x3fe` where the caller has folded a factor of two into it rather than
/// doubling its argument.
///
/// The two-step index is what makes the table 66 entries instead of 1024:
/// `B` holds a linear correction per 32nd of the significand, and applying it
/// before the shift lands `j` on a 1024-way grid that the two 33-entry tables
/// then reconstruct as a product.
#[cube]
pub fn log_index(ah: f64, off: u32, #[comptime] fk: FmaKind) -> (f64, f64, u32, u32) {
    let tab = asincosh_tab();
    let t0 = u64::reinterpret(ah);
    let ex = u32::cast_from(t0 >> 52u64);
    let ed = f64::cast_from(i32::reinterpret(ex) - i32::reinterpret(off));
    let t = t0 & (0xffff_ffff_ffff_ffffu64 >> 12u64);

    let i = u32::cast_from(t >> 47u64);
    let d = i64::reinterpret(t & (0xffff_ffff_ffff_ffffu64 >> 17u64));
    let c0 = tab[usize::cast_from(B + i * 2u32)];
    let c1 = i64::reinterpret(tab[usize::cast_from(B + i * 2u32 + 1u32)]);
    let j = u32::cast_from((t + (c0 << 33u64) + u64::reinterpret(c1 * (d >> 16i64))) >> 42u64);

    let tf = f64::reinterpret(t | (0x3ffu64 << 52u64));
    let i1 = u32::min(j >> 5u32, 32u32);
    let i2 = j & 0x1fu32;
    let r = at(&tab, R1 + i1) * at(&tab, R2 + i2);
    (ed, fma64(r, tf, -1.0, fk), i1, i2)
}

/// The fast tier's degree-5 polynomial in the reduced argument.
#[cube]
pub fn log_poly(dx: f64) -> f64 {
    let tab = asincosh_tab();
    let dx2 = dx * dx;
    dx2 * ((at(&tab, CP) + dx * at(&tab, CP + 1u32))
        + dx2 * ((at(&tab, CP + 2u32) + dx * at(&tab, CP + 3u32)) + dx2 * at(&tab, CP + 4u32)))
}

/// `log c`'s high half for stage `i1` of the first table.
#[cube]
pub fn l1_hi(i1: u32) -> f64 {
    at(&asincosh_tab(), L1 + i1 * 2u32 + 1u32)
}
/// `log c`'s low half. See [`l1_hi()`].
#[cube]
pub fn l1_lo(i1: u32) -> f64 {
    at(&asincosh_tab(), L1 + i1 * 2u32)
}
/// `log c`'s high half for stage `i2` of the second table.
#[cube]
pub fn l2_hi(i2: u32) -> f64 {
    at(&asincosh_tab(), L2 + i2 * 2u32 + 1u32)
}
/// `log c`'s low half. See [`l2_hi()`].
#[cube]
pub fn l2_lo(i2: u32) -> f64 {
    at(&asincosh_tab(), L2 + i2 * 2u32)
}

/// The accurate tier: `ln(zh + zl)/2` as a triple-double `(v0, v1, v2)`.
///
/// `a` is `log2` of the fast tier's own answer, which is what selects the
/// ladder entry — this refines a value the caller already has rather than
/// evaluating afresh, and that is why it can be so much sharper for so little
/// more work.
///
/// `acosh` selects the variant `cr_acosh` uses: the same arithmetic with
/// [`mul_dd_acc2()`] in place of [`mul_dd_acc()`] and two normalising steps at the
/// end instead of four. The difference is real — see [`mul_dd_acc2()`] — and
/// both are transcribed rather than reconciled.
#[cube]
pub fn refine_core(
    zh: f64,
    zl: f64,
    a: f64,
    #[comptime] acosh: bool,
    #[comptime] fk: FmaKind,
) -> (f64, f64, f64) {
    let rt = asincosh_refine_tab();
    let t0 = u64::reinterpret(zh);
    let ex = i32::reinterpret(u32::cast_from(t0 >> 52u64));
    let e = ex - 1023i32 + select(zl == 0.0, 1i32, 0i32);
    let t = (t0 & (0xffff_ffff_ffff_ffffu64 >> 12u64)) | (0x3ffu64 << 52u64);
    let ed = f64::cast_from(e);

    let v = u64::reinterpret(a - ed + LADDER_BIAS);
    let i = (v - (0x3ffu64 << 52u64)) >> 36u64;
    let i1 = u32::cast_from((i >> 12u64) & 0x1fu64);
    let i2 = u32::cast_from((i >> 8u64) & 0xfu64);
    let i3 = u32::cast_from((i >> 4u64) & 0xfu64);
    let i4 = u32::cast_from(i & 0xfu64);

    let el2 = L22 * ed;
    let el1 = L21 * ed;
    let el0 = L20 * ed;

    let r0 = LL + i1 * 3u32;
    let r1 = LL + (17u32 + i2) * 3u32;
    let r2 = LL + (34u32 + i3) * 3u32;
    let r3 = LL + (51u32 + i4) * 3u32;
    let ll0 = (at(&rt, r0) + at(&rt, r1) + (at(&rt, r2) + at(&rt, r3))) + el0;
    let ll1 = at(&rt, r0 + 1u32) + at(&rt, r1 + 1u32) + (at(&rt, r2 + 1u32) + at(&rt, r3 + 1u32));
    let ll2 = at(&rt, r0 + 2u32) + at(&rt, r1 + 2u32) + (at(&rt, r2 + 2u32) + at(&rt, r3 + 2u32));

    let t12 = at(&rt, T1 + i1) * at(&rt, T2 + i2);
    let t34 = at(&rt, T3 + i3) * at(&rt, T4 + i4);
    let th = t12 * t34;
    let tl = fma64(t12, t34, -th, fk);
    let tf = f64::reinterpret(t);
    let dh = th * tf;
    let dl = fma64(th, tf, -dh, fk);
    let sh0 = tl * tf;
    let sl0 = fma64(tl, tf, -sh0, fk);
    let (xh0, xl0) = fast_two_sum(dh - 1.0, dl);
    // The low half of the argument enters here rather than being carried
    // through the reduction: its exponent is shifted by the same `e` the high
    // half's was, which is one integer subtraction on the bit pattern.
    let mut xl1 = xl0;
    if zl != 0.0 {
        let tz = u64::reinterpret(zl) - (u64::reinterpret(i64::cast_from(e)) << 52u64);
        xl1 = xl0 + th * f64::reinterpret(tz);
    }
    let (xh, xl) = add_dd(xh0, xl1, sh0, sl0);

    let sla = xh * (at(&rt, RCL) + xh * (at(&rt, RCL + 1u32) + xh * at(&rt, RCL + 2u32)));
    // `polydd` over three double-double coefficients.
    let (mut ch, cl0) = fast_two_sum(at(&rt, RCH + 4u32), sla);
    let mut cl = cl0 + at(&rt, RCH + 5u32);
    let mut k = 2u32.runtime();
    while k > 0u32 {
        k = k - 1u32;
        let c0 = at(&rt, RCH + k * 2u32);
        if comptime!(acosh) {
            let (mh, ml) = mul_dd_acc2(xh, xl, ch, cl, fk);
            let (th2, tl2) = fast_two_sum(c0, mh);
            ch = th2;
            cl = ml + tl2 + at(&rt, RCH + k * 2u32 + 1u32);
        } else {
            let (mh, ml) = mul_dd_acc(xh, xl, ch, cl, fk);
            let th2 = mh + c0;
            let tl2 = (c0 - th2) + mh;
            ch = th2;
            cl = ml + tl2 + at(&rt, RCH + k * 2u32 + 1u32);
        }
    }

    let (mut sh, mut sl) = mul_dd_acc(xh, xl, ch, cl, fk);
    if comptime!(acosh) {
        let (a2, b2) = mul_dd_acc2(xh, xl, ch, cl, fk);
        sh = a2;
        sl = b2;
    }
    let (sh1, sl1) = add_dd(sh, sl, el1, el2);
    let (sh2, sl2) = add_dd(sh1, sl1, ll1, ll2);

    let (v0a, v2a) = fast_two_sum(ll0, sh2);
    let (v1a, v2b) = fast_two_sum(v2a, sl2);
    let mut v0 = v0a;
    let mut v1 = v1a;
    let mut v2 = v2b;
    if comptime!(!acosh) {
        let (v0b, v1b) = fast_two_sum(v0a, v1a);
        let (v1c, v2c) = fast_two_sum(v1b, v2b);
        v0 = v0b;
        v1 = v1c;
        v2 = v2c;
    }
    (v0, v1, v2)
}
