//! `powf`.
//!
//! A port of glibc's `__powf` (`sysdeps/ieee754/flt-32/e_powf.c` — ARM's
//! optimized-routines). Worst-case error 0.82 ulp, which is upstream's own
//! figure and the reason this had to be ported rather than widened: `powf` is
//! not correctly rounded, so evaluating the double-precision `pow` and
//! rounding once lands elsewhere on a handful of inputs — four in the
//! equivalence sweep's 130,000 pairs, before this file existed.
//!
//! # The shape
//!
//! `x^y = 2^(y log2 x)`, with a 16-row logarithm table and a degree-5
//! polynomial for `log2`, and the 32-row `exp2f` table for the exponential —
//! the same table [`super::exp`] uses, indexed the same way, with the result's
//! sign folded into the table index rather than applied afterwards. That is
//! what `sign_bias` is: `1 << 16`, which the `<< 47` that places the exponent
//! carries all the way into the sign bit.
//!
//! `y log2 x` cannot overflow, because `y` is single precision and `log2 x`
//! is bounded by 128; that one fact is what lets the whole routine be two
//! table lookups and no scaling.
//!
//! Both policy axes are accepted and have no effect.

use cubecl::prelude::*;

use crate::bits::{inf32, opaque32};
use crate::config::MathConfig;
use crate::tables::consts::{expf_tab, powf_tab};
use crate::tables::single::exp as et;
use crate::tables::single::pow as pt;

/// `1 << (EXP2F_TABLE_BITS + 11)`: added to the exponential's table index, it
/// reaches the sign bit through the same shift that places the exponent.
const SIGN_BIAS: u64 = 1u64 << 16u64;
/// `2^23`, the subnormal rescale.
const P23: f32 = f32::from_bits(0x4b00_0000);
/// `FLT_MAX`.
const FLT_MAX: f32 = f32::from_bits(0x7f7f_ffff);
/// The smallest positive subnormal `f32`, which is what `powf` returns when
/// `y log2 x` lands in `(-150, -149)`.
const MIN_SUB: f32 = f32::from_bits(1);

/// True for a zero, an infinity or a NaN bit pattern.
#[cube]
pub fn zeroinfnan(ix: u32) -> bool {
    2u32 * ix - 1u32 >= 2u32 * 0x7f80_0000u32 - 1u32
}

/// `0` if `y` is not an integer, `1` if it is odd, `2` if it is even.
///
/// Decided from the exponent and the trailing significand bits rather than by
/// comparing against `trunc`, so it stays exact at every magnitude.
#[cube]
pub fn checkint(iy: u32) -> u32 {
    let e = (iy >> 23u32) & 0xffu32;
    let mut out = 2u32;
    if e < 0x7fu32 {
        out = 0u32; // `|y| < 1`
    } else if e <= 0x7fu32 + 23u32 {
        let shift = 0x7fu32 + 23u32 - e;
        if iy & ((1u32 << shift) - 1u32) != 0u32 {
            out = 0u32; // fractional bits set
        } else {
            out = select(iy & (1u32 << shift) != 0u32, 1u32, 2u32);
        }
    }
    out
}

/// True for a signalling NaN.
#[cube]
pub fn is_signaling(x: f32) -> bool {
    let b = u32::reinterpret(x);
    b & 0x7f80_0000u32 == 0x7f80_0000u32 && b & 0x007f_ffffu32 != 0u32 && b & 0x0040_0000u32 == 0u32
}

/// `log2(x)` from already-normalised bits, in double precision.
#[cube]
pub fn log2_inline(ix: u32) -> f64 {
    // `x = 2^k z` with `z` in `[OFF, 2 OFF]` and exact; the range is split
    // into sixteen subintervals with `c` near each one's centre.
    let tmp = ix - pt::OFF;
    let i = usize::cast_from((tmp >> 19u32) % 16u32);
    let top = tmp & 0xff80_0000u32;
    let iz = ix - top;
    let k = i32::reinterpret(top) >> 23i32; // arithmetic

    let tab = powf_tab();
    let invc = f64::reinterpret(tab[2 * i]);
    let logc = f64::reinterpret(tab[2 * i + 1]);
    let z = f64::cast_from(f32::reinterpret(iz));

    // `log2(x) = log1p(z/c - 1)/ln2 + log2(c) + k`.
    let r = z * invc - 1.0;
    let y0 = logc + f64::cast_from(k);

    // Pipelined rather than Horner: four independent products before the
    // chain closes, which is what upstream's own grouping is for.
    let r2 = r * r;
    let ya = pt::A0 * r + pt::A1;
    let p = pt::A2 * r + pt::A3;
    let r4 = r2 * r2;
    let q0 = pt::A4 * r + y0;
    let q = p * r2 + q0;
    ya * r4 + q
}

/// `2^xd` with the result's sign carried in `sign_bias`, in double precision.
#[cube]
pub fn exp2_inline(xd: f64, sign_bias: u64) -> f64 {
    // `x = k/N + r` with `r` in `[-1/(2N), 1/(2N)]`.
    let kd_s = xd + et::SHIFT_SCALED;
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - et::SHIFT_SCALED;
    let r = xd - kd;

    // `2^(k/N)`, with the exponent *and the sign* folded in by one integer
    // add: `sign_bias` sits at bit 16, and the shift that places the exponent
    // carries it to bit 63.
    let tab = expf_tab();
    let s = f64::reinterpret(tab[usize::cast_from(ki % 32u64)] + ((ki + sign_bias) << 47u64));

    let z = et::C0 * r + et::C1;
    let r2 = r * r;
    let y = et::C2 * r + 1.0;
    (z * r2 + y) * s
}

/// `x^y`.
#[cube]
pub fn pow(x0: f32, y0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let y = opaque32(y0);
    let ix0 = u32::reinterpret(x);
    let iy = u32::reinterpret(y);

    let mut sign_bias = 0u64;
    let mut ix = ix0;
    let mut out = 0.0 + x;
    let mut done = false;

    if ix0 - 0x0080_0000u32 >= 0x7f80_0000u32 - 0x0080_0000u32 || zeroinfnan(iy) {
        // Either `x` is below `2^-126`, infinite or NaN, or `y` is zero,
        // infinite or NaN.
        if zeroinfnan(iy) {
            done = true;
            if 2u32 * iy == 0u32 {
                out = select(is_signaling(x), x + y, 1.0);
            } else if ix0 == 0x3f80_0000u32 {
                out = select(is_signaling(y), x + y, 1.0);
            } else if 2u32 * ix0 > 2u32 * 0x7f80_0000u32 || 2u32 * iy > 2u32 * 0x7f80_0000u32 {
                out = x + y;
            } else if 2u32 * ix0 == 2u32 * 0x3f80_0000u32 {
                out = 1.0;
            } else if (2u32 * ix0 < 2u32 * 0x3f80_0000u32) == (iy & 0x8000_0000u32 == 0u32) {
                out = 0.0; // `|x| < 1 && y == inf`, or `|x| > 1 && y == -inf`
            } else {
                out = y * y;
            }
        } else if zeroinfnan(ix0) {
            done = true;
            let mut x2 = x * x;
            if ix0 & 0x8000_0000u32 != 0u32 && checkint(iy) == 1u32 {
                x2 = -x2;
            }
            out = select(iy & 0x8000_0000u32 != 0u32, 1.0 / x2, x2);
        } else {
            // `x` and `y` are both non-zero and finite.
            if ix0 & 0x8000_0000u32 != 0u32 {
                let yint = checkint(iy);
                if yint == 0u32 {
                    // A negative base to a non-integer power.
                    out = (x - x) / (x - x);
                    done = true;
                }
                if yint == 1u32 {
                    sign_bias = SIGN_BIAS;
                }
                ix = ix0 & 0x7fff_ffffu32;
            }
            if !done && ix < 0x0080_0000u32 {
                // Normalise a subnormal `x` so its exponent goes negative.
                ix = (u32::reinterpret(x * P23) & 0x7fff_ffffu32) - (23u32 << 23u32);
            }
        }
    }

    if !done {
        // `y log2(x)` cannot overflow: `y` is single precision.
        let ylogx = f64::cast_from(y) * log2_inline(ix);
        out = f32::cast_from(exp2_inline(ylogx, sign_bias));

        if (u64::reinterpret(ylogx) >> 47u64) & 0xffffu64 >= pt::GUARD_TOP {
            let neg = sign_bias != 0u64;
            if ylogx <= -150.0 {
                out = select(neg, -0.0, 0.0);
            } else if ylogx < -149.0 {
                // The answer is between `2^-150` and `2^-149`, so it rounds to
                // the smallest subnormal — which the main path cannot produce,
                // because the table's scale factor has already underflowed.
                out = select(neg, -MIN_SUB, MIN_SUB);
            } else if ylogx > pt::OFLOW_HI {
                out = select(neg, -inf32(), inf32());
            } else if ylogx > pt::OFLOW_LO {
                // Overflow only under a rounding mode that rounds away from
                // zero. There is no way to leave round-to-nearest from here,
                // and in that mode upstream's own test folds to `FLT_MAX`.
                out = select(neg, -FLT_MAX, FLT_MAX);
            }
        }
    }
    comptime!(cfg);
    out
}
