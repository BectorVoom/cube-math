//! `2^x`.
//!
//! * [`crate::Accuracy::BitExact`] — glibc's `__ieee754_exp2` schedule. Note
//!   that it uses **no** fused multiply-adds: glibc ships no `_fma` variant of
//!   `exp2`, so what runs is the baseline SSE2 build, with two roundings where
//!   [`super::exp`] has one. Replacing any of the arithmetic below with a
//!   fused multiply-add would be *more accurate* and would break
//!   bit-exactness — which also means this kernel is bit-exact on a backend
//!   whose `fma` is not fused, as long as it does not contract.
//! * [`crate::Accuracy::Fast`] — the table-free path, reduced to
//!   `|r| <= 1/2` and corrected by a degree-11 series.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::bits::inf64;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::exp_tab;
use crate::tables::double::exp as t;

/// `2^-1022`, the down-scale the `k < 0` fix-up arm undoes.
const P_M1022: f64 = f64::from_bits(0x0010000000000000);
/// `bits(-1075.0)`, the underflow threshold, compared as a bit pattern.
const NEG_1075_BITS: u64 = (-1075.0f64).to_bits();
/// `bits(928.0)`, above which the scale factor alone can leave the range.
const BITS_928: u64 = 928.0f64.to_bits();

/// `2^x`.
#[cube]
pub fn exp2(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = crate::bits::opaque64(x);
    if comptime!(cfg.bit_exact()) {
        bit_exact(x)
    } else {
        fast(x, comptime!(cfg.checked()), comptime!(cfg.fma()))
    }
}

/// The reference schedule, over the whole domain.
///
/// The classification cascade is glibc's, spelled with plain comparisons
/// rather than its unsigned-underflow trick. The one non-obvious test is the
/// last: `bits << 1` drops the sign, so comparing shifted bit patterns is a
/// magnitude comparison against 928 — the point above which the scale factor
/// alone can leave the exponent range while the result is still finite.
#[cube]
pub fn bit_exact(x: f64) -> f64 {
    let bits = u64::reinterpret(x);
    let abstop = u32::cast_from(bits >> 52u64) & 0x7ffu32;
    // The initial value is the `|x| < 2^-54` answer, where `2^x` rounds to
    // `1 + x`. It is also the answer for `+inf` and for a NaN.
    let mut out = 1.0 + x;

    if abstop < 0x3c9u32 {
        // Already correct.
    } else if abstop >= 0x7ffu32 {
        out = select(bits == 0xfff0_0000_0000_0000u64, 0.0, 1.0 + x);
    } else if abstop >= 0x409u32 && bits >> 63u64 == 0u64 {
        out = inf64(); // genuine overflow
    } else if abstop >= 0x409u32 && bits >= NEG_1075_BITS {
        out = 0.0; // genuine underflow
    } else {
        let (tmp, sbits, ki) = core(x);
        if (bits << 1u64) > (BITS_928 << 1u64) {
            out = specialcase(tmp, sbits, ki);
        } else {
            let scale = f64::reinterpret(sbits);
            out = scale + scale * tmp;
        }
    }
    out
}

/// The shared main path: `(tmp, sbits, ki)`.
///
/// Every operation here is a separate multiply and add on purpose. See the
/// module documentation.
#[cube]
pub fn core(x: f64) -> (f64, u64, u64) {
    let kd_s = x + t::EXP2_SHIFT;
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - t::EXP2_SHIFT;
    let r = x - kd;

    let tab = exp_tab();
    let idx = usize::cast_from((ki & 127u64) * 2u64);
    let tail = f64::reinterpret(tab[idx]);
    let sbits = tab[idx + 1] + (ki << 45u64);

    let r2 = r * r;
    let tmp = tail
        + r * t::EXP2_C1
        + r2 * (t::EXP2_C2 + r * t::EXP2_C3)
        + r2 * r2 * (t::EXP2_C4 + r * t::EXP2_C5);
    (tmp, sbits, ki)
}

/// glibc `e_exp2.c: specialcase`.
///
/// Differs from [`super::exp`]'s: the overflow arm backs the exponent off by
/// one and doubles, rather than by 1009.
#[cube]
pub fn specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
    let mut out = tmp;
    if ki & 0x8000_0000u64 == 0u64 {
        let scale = f64::reinterpret(sbits - (1u64 << 52u64));
        out = 2.0 * (scale + scale * tmp);
    } else {
        let scale = f64::reinterpret(sbits + (1022u64 << 52u64));
        let st = scale * tmp;
        let mut y = scale + st;
        if y < 1.0 {
            let lo = scale - y + st;
            let hi = 1.0 + y;
            let lo2 = 1.0 - hi + y + lo;
            y = (hi + lo2) - 1.0;
        }
        out = P_M1022 * y;
    }
    out
}

/// `ln(2)^k / k!` for `k` in `1..=13`, so the series is `2^r - 1`.
///
/// Thirteen terms, not eleven. The reduction leaves `|r| <= 1/2`, where
/// truncating after the eleventh leaves a relative error around `2e-14` — some
/// 85 ulp, which the sweep found immediately. The fourteenth term is below
/// `0.1` ulp, so this is where the series stops being what limits the path.
///
/// Evaluated without its leading 1 for the same reason [`super::exp`]'s is:
/// the final combine with the scale is then one fused multiply-add rather than
/// a rounded `1 + poly` and a rounded multiply.
const G1: f64 = std::f64::consts::LN_2;
const G2: f64 = 0.2402265069591007;
const G3: f64 = 0.055504108664821576;
const G4: f64 = 0.009618129107628477;
const G5: f64 = 0.0013333558146428441;
const G6: f64 = 0.00015403530393381606;
const G7: f64 = 1.5252733804059838e-05;
const G8: f64 = 1.3215486790144305e-06;
const G9: f64 = 1.0178086009239696e-07;
const G10: f64 = 7.054911620801121e-09;
const G11: f64 = 4.44553827187081e-10;
const G12: f64 = 2.5678435993488196e-11;
const G13: f64 = 1.3691488853904124e-12;

/// The table-free path.
///
/// Maximum error measured against the correctly rounded result: below 1 ulp
/// over `|x| < 1000`.
#[cube]
pub fn fast(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let kd_s = x + t::SHIFT;
    let kd = kd_s - t::SHIFT;
    let r = x - kd;

    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c12 = fma64(r, G2, G1, fk);
    let c34 = fma64(r, G4, G3, fk);
    let c56 = fma64(r, G6, G5, fk);
    let c78 = fma64(r, G8, G7, fk);
    let c910 = fma64(r, G10, G9, fk);

    let lo = fma64(r2, c34, c12, fk);
    let hi = fma64(r2, c78, c56, fk);
    let c1112 = fma64(r, G12, G11, fk);
    let top = fma64(r2, c1112, c910, fk);
    let hi2 = fma64(r4, G13, top, fk);
    let mid = fma64(r4, hi, lo, fk);
    let poly = r * fma64(r8, hi2, mid, fk);

    let ki = u64::reinterpret(kd_s);
    let k = (ki & 0x000f_ffff_ffff_ffffu64) - (1u64 << 51u64);
    let scale = f64::reinterpret((k + 1023u64) << 52u64);
    let mut out = fma64(scale, poly, scale, fk);

    if comptime!(checked) {
        // Outside `|x| < 512` the exponent arithmetic wraps and a subnormal
        // result would be built from an already-overflowed exponent field, so
        // those inputs take the reference path. See `exp`'s `fast` for why
        // that is the right place for them.
        let abstop = u32::cast_from(u64::reinterpret(x) >> 52u64) & 0x7ffu32;
        if abstop >= 0x408u32 || abstop < 0x3c9u32 {
            out = bit_exact(x);
        }
    }
    out
}
