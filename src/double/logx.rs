//! `log2` and `log10`.
//!
//! Two logarithms that are not one logarithm scaled. `log2` has its own
//! 64-subinterval table and its own near-one path, because a scaled `ln` would
//! inherit `ln`'s error where `log2` of a power of two has to be exact.
//! `log10`, on the other hand, genuinely *is* a wrapper in glibc — the
//! disassembly of `__log10_finite` shows it reducing the argument, calling
//! straight into `__ieee754_log_fma`, and combining with three unfused
//! operations — so that is what it does here, reusing [`super::ln`]'s own
//! table walk rather than re-deriving one.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::bits::neg_inf64;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::log2_tab;
use crate::tables::double::log2 as t;

/// The table-centring offset, `bits(0x1.6p-1)`.
const OFF: u64 = 0x3fe6000000000000;
/// Low end of the near-one window, `1.0 - 0x1.5b51p-5`.
const NEAR_LO: f64 = f64::from_bits(0x3feea4af00000000);
/// High end, `1.0 + 0x1.6ab2p-5`.
const NEAR_HI: f64 = f64::from_bits(0x3ff0b55900000000);
/// `2^52`, the scale that normalises a subnormal.
const P52: f64 = 4503599627370496.0;

/// True when `x` is not a positive normal — the inputs the main paths below
/// are not written for.
#[cube]
pub fn not_positive_normal(x: f64) -> bool {
    let top = u32::cast_from(u64::reinterpret(x) >> 48u64);
    top - 0x0010u32 >= 0x7ff0u32 - 0x0010u32
}

/// True when `x` is one of the degenerate inputs — zero, negative, infinite
/// or NaN — as opposed to merely subnormal.
#[cube]
pub fn degenerate(x: f64) -> bool {
    let ix = u64::reinterpret(x);
    ix * 2u64 == 0u64 || (ix >> 63u64) != 0u64 || (ix >> 52u64) & 0x7ffu64 == 0x7ffu64
}

/// The answer for a degenerate `x`. The same for every logarithm.
#[cube]
pub fn edge(x: f64) -> f64 {
    let ix = u64::reinterpret(x);
    let mut out = x;
    if ix * 2u64 == 0u64 {
        out = neg_inf64();
    } else if ix == 0x7ff0_0000_0000_0000u64 {
        out = x; // +inf
    } else {
        // Negative, or a NaN. glibc's `__math_invalid(x)`, spelled as it
        // spells it: not the positive quiet NaN, because `0/0` on x86 yields
        // the negative one and the sign of a NaN is part of the contract.
        out = (x - x) / (x - x);
    }
    out
}

/// The bit pattern of a positive input, with a subnormal renormalised.
///
/// glibc does not correct a subnormal by taking the logarithm of the rescaled
/// value and subtracting: it edits the *exponent field* before the table path
/// ever runs, so that the correction is folded into `kd` and there is only one
/// rounding at the end. Subtracting afterwards costs a last bit, which the
/// sweep found on exactly one input.
#[cube]
pub fn normalised_bits(x: f64) -> u64 {
    let ix = u64::reinterpret(x);
    let sub = (ix >> 52u64) & 0x7ffu64 == 0u64;
    select(sub, u64::reinterpret(x * P52) - (52u64 << 52u64), ix)
}

/// Base-2 logarithm.
///
/// A port of glibc's `__ieee754_log2_fma`: a 64-subinterval `[1/c, log2 c]`
/// table, a degree-6 correction, and a separate near-one polynomial where the
/// main path would lose too much to cancellation.
#[cube]
pub fn log2(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = crate::bits::opaque64(x);
    let mut out = x;
    if degenerate(x) {
        out = edge(x);
    } else if comptime!(cfg.bit_exact()) {
        if x >= NEAR_LO && x < NEAR_HI {
            out = log2_near_one(x);
        } else {
            out = log2_main(normalised_bits(x));
        }
    } else {
        out = log2_fast(x, comptime!(cfg.fma()));
    }
    out
}

/// The table path, taking already-normalised bits.
#[cube]
pub fn log2_main(ix: u64) -> f64 {
    let tmp = ix - OFF;
    let idx = usize::cast_from((tmp >> 46u64) & 63u64);
    let z = f64::reinterpret(ix - (tmp & (0xfffu64 << 52u64)));
    let kd = f64::cast_from(i64::reinterpret(tmp) >> 52i64);

    let tab = log2_tab();
    let base = 2usize * idx;
    let invc = f64::reinterpret(tab[base]);
    let logc = f64::reinterpret(tab[base + 1]);

    // `r = z/c - 1`, then `r / ln 2` carried in double-double as `t1 + t2`.
    let r = fma(z, invc, -1.0);
    let t1 = r * t::INVLN2HI;
    let t2 = fma(r, t::INVLN2LO, fma(r, t::INVLN2HI, -t1));

    let t3 = kd + logc;
    let hi = t3 + t1;
    let lo = t3 - hi + t1 + t2;

    let r2 = r * r;
    let r4 = r2 * r2;
    let a01 = fma(r, t::A1, t::A0);
    let a23 = fma(r, t::A3, t::A2);
    let a45 = fma(r, t::A5, t::A4);
    let poly = fma(r4, a45, fma(r2, a23, a01));
    hi + fma(r2, poly, lo)
}

/// The near-one path, where `log2(x)` is small and the table path's `hi + lo`
/// would cancel away its own accuracy.
#[cube]
pub fn log2_near_one(x: f64) -> f64 {
    let r = x - 1.0;
    let hi0 = r * t::INVLN2HI;
    let r2 = r * r;
    let r4 = r2 * r2;

    let lo0 = fma(r, t::INVLN2LO, fma(r, t::INVLN2HI, -hi0));

    let b01 = fma(r, t::B1, t::B0);
    let y = fma(r2, b01, hi0);
    let lo = lo0 + fma(r2, b01, hi0 - y);

    let b23 = fma(r, t::B3, t::B2);
    let b45 = fma(r, t::B5, t::B4);
    let b23_45 = fma(r2, b45, b23);

    let b67 = fma(r, t::B7, t::B6);
    let b89 = fma(r, t::B9, t::B8);
    let b67_89 = fma(r2, b89, b67);

    let tail = fma(r4, b67_89, b23_45);
    y + fma(r4, tail, lo)
}

/// The table-free path: [`super::ln`]'s significand fold, scaled to base 2.
///
/// Measured error: below 2 ulp over the positive normals.
#[cube]
pub fn log2_fast(x: f64, #[comptime] fk: FmaKind) -> f64 {
    let (e, poly) = crate::double::ln::fold(x, fk);
    fma64(poly, LOG2_E, e, fk)
}

/// `1 / ln(10)`, bit-identical to `__log10_finite`'s `ivln10`.
const INV_LN10: f64 = f64::from_bits(0x3fdbcb7b1526e50e);
/// High part of `log10(2)`, `__log10_finite`'s `log10_2hi`.
const LOG10_2HI: f64 = f64::from_bits(0x3fd34413509f6000);
/// Low part of `log10(2)`, `__log10_finite`'s `log10_2lo`.
const LOG10_2LO: f64 = f64::from_bits(0x3d59fef311f12b36);
/// `log10(2)`, in one piece — only the approximate path wants it.
const LOG10_2: f64 = f64::from_bits(0x3fd34413509f79ff);
/// `log2(e)`.
const LOG2_E: f64 = std::f64::consts::LOG2_E;

/// Base-10 logarithm.
///
/// `BitExact` is a port of glibc's `__log10_finite`, which the disassembly
/// shows to be a thin wrapper rather than a table algorithm of its own: it
/// extracts the unbiased exponent `k` and a rounding-parity bit `i`, forces
/// `x`'s exponent field to `0x3ff - i` so the reduced argument sits within one
/// exponent step of 1, calls straight into `__ieee754_log_fma`, and combines
/// with three *unfused* operations. There is no `vfmadd` anywhere in it, so
/// the final combine below is genuinely three separate roundings.
#[cube]
pub fn log10(x: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = crate::bits::opaque64(x);
    let mut out = x;
    if degenerate(x) {
        out = edge(x);
    } else if comptime!(cfg.bit_exact()) {
        out = log10_main(normalised_bits(x));
    } else {
        let (e, poly) = crate::double::ln::fold(x, comptime!(cfg.fma()));
        // `e log10(2) + ln(m) / ln(10)` rather than `ln(x) / ln(10)`: the
        // exponent term is then exact to within one rounding instead of
        // inheriting the error of a large `ln`.
        out = fma64(poly, INV_LN10, e * LOG10_2, comptime!(cfg.fma()));
    }
    out
}

/// The reduce-and-delegate path, for already-normalised bits.
#[cube]
pub fn log10_main(b: u64) -> f64 {
        // An *arithmetic* shift: `normalised_bits` can leave the exponent
        // field negative for a renormalised subnormal, and reading it as an
        // unsigned field would turn that into a huge positive exponent.
        let k = (i64::reinterpret(b) >> 52i64) - 1023i64;
        // `i` is glibc's own rounding-parity bit: 1 when `k` is negative.
        let i = i64::reinterpret(u64::reinterpret(k) >> 63u64);
        let y = f64::cast_from(k + i);
        let exp_field = (1023u64 - u64::reinterpret(i)) << 52u64;
        let reduced = f64::reinterpret((b & 0x000f_ffff_ffff_ffffu64) | exp_field);

        // The *whole* of `ln`, near-one path included — that is what
        // `__log10_finite` calls. Going straight to the table walk would make
        // `log10(1)` a tiny nonzero instead of an exact zero.
        let lr = crate::double::ln::bit_exact(reduced);
        // Deliberately not fused: the disassembly has three separate
        // multiply/add pairs here, not a fusion opportunity.
        (lr * INV_LN10 + y * LOG10_2LO) + y * LOG10_2HI
}
