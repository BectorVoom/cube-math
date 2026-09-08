//! Natural logarithm.
//!
//! * [`crate::Accuracy::BitExact`] — glibc's `__ieee754_log_fma` schedule: a
//!   128-entry `[1/c, log c]` table, a degree-5 correction, and a separate
//!   near-one path that carries a Veltkamp split so `r - r^2/2` is evaluated
//!   in double-double where the table path would lose everything to
//!   cancellation.
//! * [`crate::Accuracy::Fast`] — no table, no near-one special case: reduce to
//!   a mantissa in `[sqrt(2)/2, sqrt(2))`, then a degree-15 odd series in
//!   `s = (m - 1) / (m + 1)`, which converges fast enough over that interval
//!   to stay inside an ulp without one.

use cubecl::prelude::*;

use crate::bits::neg_inf64;
use crate::config::MathConfig;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::log_tab;
use crate::tables::double::log as t;

/// Low end of the near-one window, `bits(1.0 - 0x1p-4)`.
const NEAR_LO: u64 = 0x3fee000000000000;
/// High end, `bits(1.0 + 0x1.09p-4)`.
const NEAR_HI: u64 = 0x3ff1090000000000;
/// The table-centring offset, `bits(0x1.6p-1)`.
const OFF: u64 = 0x3fe6000000000000;
/// `2^52`, the scale that normalises a subnormal.
const P52: f64 = 4503599627370496.0;

/// `ln(x)`.
#[cube]
pub fn ln(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = crate::bits::opaque64(x);
    if comptime!(cfg.bit_exact()) {
        bit_exact(x)
    } else {
        fast(x, comptime!(cfg.checked()), comptime!(cfg.fma()))
    }
}

/// The reference schedule, over the whole domain.
#[cube]
pub fn bit_exact(x: f64) -> f64 {
    let ix = u64::reinterpret(x);
    let top = u32::cast_from(ix >> 48u64);
    let mut out = x;

    if ix - NEAR_LO < NEAR_HI - NEAR_LO {
        // The near-one window, where the table path loses too much to
        // cancellation. `ln(1)` is exactly zero and the series would not give
        // a signed zero, so it is called out.
        out = select(ix == 0x3ff0_0000_0000_0000u64, 0.0, near_one(x));
    } else if top - 0x0010u32 >= 0x7ff0u32 - 0x0010u32 {
        // Zero, subnormal, negative, infinite or NaN.
        if ix * 2u64 == 0u64 {
            out = neg_inf64(); // log(+-0)
        } else if ix == 0x7ff0_0000_0000_0000u64 {
            out = x; // log(+inf)
        } else if (top & 0x8000u32) != 0u32 || (top & 0x7ff0u32) == 0x7ff0u32 {
            // glibc's `__math_invalid(x)`. Not the positive quiet NaN:
            // `0/0` on x86 yields the negative one, and the sign of a NaN is
            // part of a bit-exactness contract.
            out = (x - x) / (x - x);
        } else {
            // Subnormal: scale into the normal range and correct the exponent.
            let iz = u64::reinterpret(x * P52) - (52u64 << 52u64);
            out = main(iz);
        }
    } else {
        out = main(ix);
    }
    out
}

/// The table path, taking already-normalised bits.
#[cube]
pub fn main(ix: u64) -> f64 {
    let tmp = ix - OFF;
    let i = usize::cast_from((tmp >> 45u64) & 127u64);
    // An *arithmetic* shift: `tmp`'s top bits carry the exponent, sign and all.
    let k = i64::reinterpret(tmp) >> 52i64;
    let iz = ix - (tmp & (0xfffu64 << 52u64));

    let tab = log_tab();
    let base = 2usize * i;
    let invc = f64::reinterpret(tab[base]);
    let logc = f64::reinterpret(tab[base + 1]);
    let z = f64::reinterpret(iz);
    let kd = f64::cast_from(k);

    let w = fma(kd, t::LN2HI, logc);
    let r = fma(z, invc, -1.0);
    let q12 = fma(r, t::A2, t::A1);
    let hi = r + w;
    let r2 = r * r;
    let tt = (w - hi) + r;
    let lo = fma(kd, t::LN2LO, tt);
    let r3 = r * r2;
    let q34 = fma(r, t::A4, t::A3);
    let s1 = fma(r2, t::A0, lo);
    let q = fma(r2, q34, q12);
    fma(r3, q, s1) + hi
}

/// The `0.9375 <= x < 1 + 0x1.09p-4` path.
///
/// Carries a Veltkamp split so that `r - r^2/2` is evaluated in
/// double-double: near one, the leading terms cancel almost completely, and
/// evaluating them at working precision would leave nothing behind.
#[cube]
pub fn near_one(x: f64) -> f64 {
    let r = x - 1.0;
    let p12 = fma(r, t::B2, t::B1);
    let p45 = fma(r, t::B5, t::B4);
    let r2 = r * r;
    let p78 = fma(r, t::B8, t::B7);
    let p123 = fma(r2, t::B3, p12);
    let p456 = fma(r2, t::B6, p45);
    let r3 = r * r2;
    let p789 = fma(r2, t::B9, p78);
    let p78910 = fma(r3, t::B10, p789);
    let pin = fma(p78910, r3, p456);
    let poly = fma(pin, r3, p123);

    let split = 134217728.0; // 0x1p27
    let rhi_t = fma(r, split, r);
    let rhi = fma(-r, split, rhi_t);
    let rlo = r - rhi;
    let s = rhi * rhi;
    let hi = fma(s, t::B0, r);
    let tt = r - hi;
    let rpr = r + rhi;
    let lo0 = fma(s, t::B0, tt);
    let lo = fma(t::B0 * rlo, rpr, lo0);
    fma(poly, r3, lo) + hi
}

/// `2 / (2k + 1)` for `k` in `0..=10`: the odd series for `ln((1+s)/(1-s))`.
///
/// Eleven terms, not eight. The reduction leaves `|s| <= 0.1716`, where
/// truncating after the eighth leaves some 80 ulp — the sweep found it at
/// `x = ln(2)`, where `s` is near the top of its range and the result is small
/// enough that the absolute error dominates. The twelfth term is below a fifth
/// of an ulp.
const S1: f64 = 2.0;
const S3: f64 = 2.0 / 3.0;
const S5: f64 = 2.0 / 5.0;
const S7: f64 = 2.0 / 7.0;
const S9: f64 = 2.0 / 9.0;
const S11: f64 = 2.0 / 11.0;
const S13: f64 = 2.0 / 13.0;
const S15: f64 = 2.0 / 15.0;
const S17: f64 = 2.0 / 17.0;
const S19: f64 = 2.0 / 19.0;
const S21: f64 = 2.0 / 21.0;

/// `ln(2)`, split so that `k * LN2HI` is exact.
const LN2HI: f64 = f64::from_bits(0x3fe62e42fee00000);
/// `ln(2)`, low part.
const LN2LO: f64 = f64::from_bits(0x3dea39ef35793c76);

/// The table-free path.
///
/// Reduce `x` to `m * 2^k` with `m` in `[sqrt(2)/2, sqrt(2))`, so that
/// `s = (m - 1) / (m + 1)` has `|s| <= 0.1716`. The odd series in `s`
/// converges in that interval fast enough that a degree-15 truncation is below
/// a tenth of an ulp, which is what buys the table away.
///
/// Maximum error measured against the correctly rounded result: 2 ulp,
/// reached where the reduction cancels — around `x = ln(2)`, where `k` is
/// `-1` and `k ln(2)` and the series are the same size and opposite in sign.
/// Away from that band it is below 1 ulp.
#[cube]
pub fn fast(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let (kd, poly) = fold(x, fk);
    let mut out = fma64(kd, LN2LO, fma64(kd, LN2HI, poly, fk), fk);

    if comptime!(checked) {
        // Zero, negative, subnormal, infinite and NaN all take the reference
        // path: the reduction assumes a normal positive `x`, and these are
        // exactly the inputs `rmath` repairs rather than special-cases per
        // policy. See `exp`'s `fast`.
        let top = u32::cast_from(u64::reinterpret(x) >> 48u64);
        if top - 0x0010u32 >= 0x7ff0u32 - 0x0010u32 {
            out = bit_exact(x);
        }
    }
    out
}

/// The table-free reduction, shared with [`super::logx`].
///
/// Splits a positive normal `x` into `(k, ln(m))` with `x = m 2^k` and `m` in
/// `[sqrt(2)/2, sqrt(2))`, so that the caller can scale to whatever base it
/// wants. Returning the two pieces rather than their sum is the point:
/// `log2` and `log10` want `k` multiplied by a different constant, and folding
/// it into a natural logarithm first and dividing afterwards would inherit the
/// error of a large `ln` where the exponent term should be exact.
#[cube]
pub fn fold(x: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    // A subnormal has no exponent of its own; scale it into the normal range
    // and take the shift back off `k`. Done here rather than by editing the
    // exponent field, the way the table paths do it, because the field can go
    // negative and this reduction reads it as an unsigned quantity.
    let raw = u64::reinterpret(x);
    let sub = (raw >> 52u64) & 0x7ffu64 == 0u64;
    let bits = select(sub, u64::reinterpret(x * P52), raw);
    let mut k = i32::cast_from((bits >> 52u64) & 0x7ffu64) - 1023i32 - select(sub, 52i32, 0i32);
    // The mantissa, with the exponent replaced by zero: `m` in `[1, 2)`.
    let mut m = f64::reinterpret((bits & 0x000f_ffff_ffff_ffffu64) | 0x3ff0_0000_0000_0000u64);
    // Recentre onto `[sqrt(2)/2, sqrt(2))`, where the series is shortest.
    if m > std::f64::consts::SQRT_2 {
        m = m * 0.5;
        k += 1i32;
    }

    let s = (m - 1.0) / (m + 1.0);
    let t = s * s;
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;

    let c13 = fma64(t, S3, S1, fk);
    let c57 = fma64(t, S7, S5, fk);
    let c911 = fma64(t, S11, S9, fk);
    let c1315 = fma64(t, S15, S13, fk);
    let c1719 = fma64(t, S19, S17, fk);
    let lo = fma64(t2, c57, c13, fk);
    let mid = fma64(t2, c1315, c911, fk);
    let hi = fma64(t2, S21, c1719, fk);
    let both = fma64(t4, mid, lo, fk);
    let poly = s * fma64(t8, hi, both, fk);

    (f64::cast_from(k), poly)
}
