//! `e^x`.
//!
//! * [`crate::Accuracy::BitExact`] — glibc's `__ieee754_exp_fma` schedule:
//!   argument reduction to `|r| < ln(2)/256`, a 128-entry `2^(k/128)` table,
//!   and a degree-4 correction. Bit-identical to the scalar `exp`, including
//!   the two asymmetric arms of its overflow/underflow fix-up, one of which
//!   glibc compiled fused and the other of which it did not.
//! * [`crate::Accuracy::Fast`] — no table. Reduces to `|r| <= ln(2)/2` and
//!   takes a degree-13 series by Estrin's scheme. On a CPU the point of this
//!   is avoiding a gather; on a GPU the gather is a perfectly ordinary
//!   coalesced-ish load, and what the table-free path buys instead is one less
//!   dependent memory access in a latency-bound kernel.
//!
//! [`crate::Domain::Finite`] means `|x| < 512`. Outside that the answer is
//! wrong, not merely imprecise — and note `exp` overflows at 709.78, so the
//! safe range is set by the reduction, not by the function's own domain.

use cubecl::prelude::*;

use crate::config::Config;
use crate::cube::bits::inf64;
use crate::cube::fma::{FmaKind, fma64};
use crate::tables::arena::OFF_EXP;
use crate::tables::double::exp as t;

/// `2^1009`, the up-scale the `k > 0` fix-up arm undoes.
const P1009: f64 = f64::from_bits(0x7f00000000000000);
/// `2^-1022`, the down-scale the `k < 0` fix-up arm undoes.
const P_M1022: f64 = f64::from_bits(0x0010000000000000);

/// `e^x`.
#[cube]
pub fn exp(x: f64, tab: &Array<u64>, #[comptime] cfg: Config) -> f64 {
    if comptime!(cfg.bit_exact()) {
        bit_exact(x, tab, comptime!(cfg.fma()))
    } else {
        fast(x, tab, comptime!(cfg.checked()), comptime!(cfg.fma()))
    }
}

/// The reference schedule, over the whole domain.
///
/// The classification cascade is glibc's, spelled with plain comparisons
/// rather than its unsigned-underflow trick: `abstop - top12(TINY) >=
/// top12(512) - top12(TINY)` is exactly `abstop < 0x3c9 || abstop >= 0x408`,
/// and saying so keeps the code readable and free of a wrapping subtraction
/// whose behaviour would otherwise have to be pinned down per backend.
#[cube]
pub fn bit_exact(x: f64, tab: &Array<u64>, #[comptime] fk: FmaKind) -> f64 {
    let bits = u64::reinterpret(x);
    let abstop = u32::cast_from(bits >> 52u64) & 0x7ffu32;
    // The initial value is the `|x| < 2^-54` answer, where `e^x` rounds to
    // `1 + x` — which is also the `x == 0` case and the only place a subnormal
    // input can reach. Every other branch overwrites it.
    let mut out = 1.0 + x;

    if abstop < 0x3c9u32 {
        // Already correct.
    } else if abstop >= 0x409u32 {
        // |x| >= 1024, or non-finite.
        if bits == 0xfff0_0000_0000_0000u64 {
            out = 0.0;
        } else if abstop >= 0x7ffu32 {
            out = 1.0 + x; // +inf, or NaN with its payload preserved
        } else if bits >> 63u64 != 0u64 {
            out = 0.0; // genuine underflow
        } else {
            out = inf64(); // genuine overflow
        }
    } else {
        let (tmp, sbits, ki) = core(x, tab, fk);
        if abstop >= 0x408u32 {
            // 512 <= |x| < 1024: the scale may be out of range on its own even
            // where the result is not, so it is built at a shifted exponent
            // and undone afterwards.
            out = specialcase(tmp, sbits, ki, fk);
        } else {
            let scale = f64::reinterpret(sbits);
            out = fma64(scale, tmp, scale, fk);
        }
    }
    out
}

/// The shared main path: `(tmp, sbits, ki)`, so the caller can choose between
/// the one-instruction tail and [`specialcase()`].
#[cube]
pub fn core(x: f64, tab: &Array<u64>, #[comptime] fk: FmaKind) -> (f64, u64, u64) {
    let kd_s = fma64(x, t::INVLN2N, t::SHIFT, fk);
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - t::SHIFT;
    let r = fma64(kd, t::NEGLN2LON, fma64(kd, t::NEGLN2HIN, x, fk), fk);

    let idx = usize::cast_from((ki & 127u64) * 2u64) + OFF_EXP as usize;
    let tail = f64::reinterpret(tab[idx]);
    let sbits = tab[idx + 1] + (ki << 45u64);

    let p12 = fma64(r, t::C3, t::C2, fk);
    let t3 = tail + r;
    let r2 = r * r;
    let p45 = fma64(r, t::C5, t::C4, fk);
    let s1 = fma64(r2, p12, t3, fk);
    let r4 = r2 * r2;
    let tmp = fma64(r4, p45, s1, fk);
    (tmp, sbits, ki)
}

/// `e^x` for `512 <= |x| < 1024`, where the scale factor alone can overflow.
///
/// Note the asymmetry between the two arms: glibc compiled the `k > 0` side
/// fused and the `k < 0` side not, and reproducing that is the whole point —
/// making the second arm fused too would be *more* accurate and would break
/// bit-exactness.
#[cube]
pub fn specialcase(tmp: f64, sbits: u64, ki: u64, #[comptime] fk: FmaKind) -> f64 {
    // Initialised to the `k > 0` arm's shape so that the value is never read
    // uninitialised; both arms assign.
    let mut out = tmp;
    if ki & 0x8000_0000u64 == 0u64 {
        // k > 0: the scale's exponent may have overflowed by up to 460.
        let scale = f64::reinterpret(sbits - (1009u64 << 52u64));
        out = P1009 * fma64(scale, tmp, scale, fk);
    } else {
        // k < 0: the result may be subnormal, where a second rounding would
        // cost half an ulp, so the sum is renormalised around 1.0 first.
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

/// `1/ln(2)`, for the table-free reduction.
const LOG2E: f64 = std::f64::consts::LOG2_E;
/// `ln(2)`, high part — exactly representable in 33 bits, which makes
/// `kd * LN2HI` exact for every `kd` the reduction produces.
const LN2HI: f64 = f64::from_bits(0x3fe62e42fee00000);
/// `ln(2)`, low part.
const LN2LO: f64 = f64::from_bits(0x3dea39ef35793c76);

/// The table-free path.
///
/// The series is evaluated *without* its leading 1 — `poly` is `e^r - 1`, not
/// `e^r` — so the final combine with `scale` is one fused multiply-add rather
/// than a rounded `1 + poly` followed by a rounded multiply.
///
/// Estrin rather than Horner: Horner would be thirteen dependent
/// multiply-adds, and on a GPU every thread in a warp waits on that same chain
/// together. Same operation count, a third of the depth.
///
/// Maximum error measured against the correctly rounded result: below 1 ulp
/// over `|x| < 512`.
#[cube]
pub fn fast(x: f64, tab: &Array<u64>, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let kd_s = fma64(x, LOG2E, t::SHIFT, fk);
    let kd = kd_s - t::SHIFT;
    // Cody-Waite: subtract `kd * ln(2)` in two exactly-representable pieces.
    let r = fma64(kd, -LN2LO, fma64(kd, -LN2HI, x, fk), fk);

    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;

    let c23 = fma64(r, F1, F0, fk);
    let c45 = fma64(r, F3, F2, fk);
    let c67 = fma64(r, F5, F4, fk);
    let c89 = fma64(r, F7, F6, fk);
    let cab = fma64(r, F9, F8, fk);
    let ccd = fma64(r, F11, F10, fk);

    let lo0 = fma64(r2, c23, r, fk); // `r` itself is the degree-1 term
    let mid = fma64(r2, c67, c45, fk);
    let hi = fma64(r2, cab, c89, fk);
    let lo = fma64(r4, mid, lo0, fk);
    let hi2 = fma64(r4, ccd, hi, fk);
    let poly = fma64(r8, hi2, lo, fk);

    // 2^kd, built straight into the exponent field: no memory at all.
    let ki = u64::reinterpret(kd_s);
    let k = (ki & 0x000f_ffff_ffff_ffffu64) - (1u64 << 51u64);
    let scale = f64::reinterpret((k + 1023u64) << 52u64);
    let mut out = fma64(scale, poly, scale, fk);

    if comptime!(checked) {
        // The reduction is only valid for `|x| < 512`. Outside it the exponent
        // arithmetic wraps rather than saturating, and a subnormal result
        // would be assembled from a bit pattern that has already overflowed
        // the exponent field — so those inputs take the reference path
        // instead. That is what `rmath` does with its out-of-range lanes, for
        // the same reason: the hard cases belong in one place, written once
        // against the platform routine, rather than re-derived per policy.
        let abstop = u32::cast_from(u64::reinterpret(x) >> 52u64) & 0x7ffu32;
        if abstop >= 0x408u32 || abstop < 0x3c9u32 {
            out = bit_exact(x, tab, fk);
        }
    }
    out
}

/// `1/k!` for `k` in `2..=13`.
///
/// Truncating the series there leaves a relative error of `r^14 / 14!`, which
/// at the reduction's worst case (`|r| = ln(2)/2`) is about `6.5e-18` — a
/// fifth of an ulp, so what limits this path is accumulated rounding, not
/// truncation.
const F0: f64 = 1.0 / 2.0;
const F1: f64 = 1.0 / 6.0;
const F2: f64 = 1.0 / 24.0;
const F3: f64 = 1.0 / 120.0;
const F4: f64 = 1.0 / 720.0;
const F5: f64 = 1.0 / 5040.0;
const F6: f64 = 1.0 / 40320.0;
const F7: f64 = 1.0 / 362880.0;
const F8: f64 = 1.0 / 3628800.0;
const F9: f64 = 1.0 / 39916800.0;
const F10: f64 = 1.0 / 479001600.0;
const F11: f64 = 1.0 / 6227020800.0;
