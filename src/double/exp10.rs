//! `10^x`.
//!
//! glibc 2.39 replaced the old `exp(x * log(10))` composition with a routine
//! of its own: the same 128-entry table `exp` and `exp2` share, reduced
//! against `log10(2)/128` and corrected by a degree-4 polynomial. Like
//! [`super::exp2`] and unlike [`super::exp`] it ships with **no** fused
//! variant, so the schedule below is separate multiplies and adds throughout —
//! using a fused multiply-add anywhere in it would be more accurate and would
//! break bit-exactness.
//!
//! [`crate::Accuracy::Fast`] routes through [`super::exp2`]'s table-free path:
//! `10^x` is `2^(x log2 10)`, and carrying the product in two pieces keeps the
//! reduction exact enough that the result stays inside an ulp.

use cubecl::prelude::*;

use crate::bits::inf64;
use crate::config::MathConfig;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::exp_tab;
use crate::tables::double::exp as t;
use crate::tables::double::exp10 as x10;

/// `10^x`.
#[cube]
pub fn exp10(x: f64, #[comptime] cfg: MathConfig) -> f64 {
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
    let abstop = u32::cast_from(ix >> 52u64) & 0x7ffu32;
    // The `|x| < 2^-57` answer, where `10^x` rounds to `1 + x`; also the
    // answer for `+inf` and for a NaN.
    let mut out = x + 1.0;

    if abstop < x10::SMALL_TOP {
        // Already correct.
    } else if abstop == 0x7ffu32 {
        out = select(ix == 0xfff0_0000_0000_0000u64, 0.0, x + 1.0);
    } else if x >= x10::OFLOW_BOUND {
        out = inf64();
    } else if x < x10::UFLOW_BOUND {
        out = 0.0;
    } else {
        let (tmp, sbits, ki) = core(x);
        if abstop >= x10::SMALL_TOP + x10::THRESH {
            // Large but representable: the scale factor alone can leave the
            // exponent range. glibc's `e_exp10.c: special_case` is
            // `e_exp2.c: specialcase` with the same two arms and the same
            // constants, so it is that code here rather than a second
            // transcription of it.
            out = crate::double::exp2::specialcase(tmp, sbits, ki);
        } else {
            let scale = f64::reinterpret(sbits);
            out = scale * tmp + scale;
        }
    }
    out
}

/// The shared main path.
///
/// `r` is rounded twice on purpose — once after the high part of the reduction
/// and again after the low part — because that is what the compiled library
/// does.
#[cube]
pub fn core(x: f64) -> (f64, u64, u64) {
    let z = x10::INVLOG10_2N * x;
    let kd_s = z + t::SHIFT;
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - t::SHIFT;

    let r0 = x10::NEGLOG10_2HIN * kd + x;
    let r = x10::NEGLOG10_2LON * kd + r0;

    let tab = exp_tab();
    let idx = usize::cast_from((ki & 127u64) * 2u64);
    let tail = f64::reinterpret(tab[idx]);
    let sbits = tab[idx + 1] + (ki << 45u64);

    let r2 = r * r;
    let p = x10::C0 + r * x10::C1;
    let y0 = x10::C2 + r * x10::C3;
    let y1 = y0 + r2 * x10::C4;
    let y2 = p + r2 * y1;
    (tail + y2 * r, sbits, ki)
}

/// `ln(10)^k / k!` for `k` in `1..=13`, so the series is `10^r - 1`.
///
/// Evaluated without its leading 1, for the same reason [`super::exp`]'s is:
/// the final combine with the scale is then one fused multiply-add. Thirteen
/// terms cover `|r| <= log10(2)/2`, where the fourteenth is below a tenth of
/// an ulp.
const H1: f64 = std::f64::consts::LN_10;
const H2: f64 = 2.6509490552391997;
const H3: f64 = 2.034678592293477;
const H4: f64 = 1.1712551489122673;
const H5: f64 = 0.5393829291955817;
const H6: f64 = 0.2069958486968682;
const H7: f64 = 0.06808936507443711;
const H8: f64 = 0.01959769462647854;
const H9: f64 = 0.0050139288337754445;
const H10: f64 = 0.0011544997789984359;
const H11: f64 = 0.00024166672554424716;
const H12: f64 = 4.637151664257225e-5;
const H13: f64 = 8.213412535439399e-6;

/// The table-free path.
///
/// Its own Cody-Waite reduction rather than a composition through
/// [`super::exp2`]: `10^x` is `2^(x log2 10)`, but that product has to be
/// carried to more than working precision or the reduction throws away the
/// bits that decide the answer, and there is no way to hand a two-piece
/// argument to a function that takes one `f64`. Reducing against `log10(2)`
/// instead keeps `kd * LOG10_2HI` exact — `kd` is a small integer and the
/// constant has few significant bits — so the whole reduction is two fused
/// multiply-adds and no lost precision.
///
/// Maximum error measured against the correctly rounded result: below 1.5 ulp
/// over the whole finite range.
#[cube]
pub fn fast(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let kd_s = fma64(x, x10::LOG2_10, t::SHIFT, fk);
    let kd = kd_s - t::SHIFT;
    let r = fma64(-kd, x10::LOG10_2LO, fma64(-kd, x10::LOG10_2HI, x, fk), fk);

    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c12 = fma64(r, H2, H1, fk);
    let c34 = fma64(r, H4, H3, fk);
    let c56 = fma64(r, H6, H5, fk);
    let c78 = fma64(r, H8, H7, fk);
    let c910 = fma64(r, H10, H9, fk);
    let c1112 = fma64(r, H12, H11, fk);

    let lo = fma64(r2, c34, c12, fk);
    let mid = fma64(r2, c78, c56, fk);
    let top = fma64(r2, c1112, c910, fk);
    let hi = fma64(r4, H13, top, fk);
    let mid2 = fma64(r4, mid, lo, fk);
    let poly = r * fma64(r8, hi, mid2, fk);

    let ki = u64::reinterpret(kd_s);
    let k = (ki & 0x000f_ffff_ffff_ffffu64) - (1u64 << 51u64);
    let scale = f64::reinterpret((k + 1023u64) << 52u64);
    let mut out = fma64(scale, poly, scale, fk);

    if comptime!(checked) {
        // The scale is assembled straight into the exponent field, which only
        // works while the result is normal; outside that the reference path
        // takes over. See `exp`'s `fast` for why the hard cases live there.
        let abstop = u32::cast_from(u64::reinterpret(x) >> 52u64) & 0x7ffu32;
        if abstop >= 0x405u32 || abstop < x10::SMALL_TOP {
            out = bit_exact(x);
        }
    }
    out
}
