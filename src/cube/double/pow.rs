//! `x^y`.
//!
//! A port of glibc's `__pow_fma`, which is not `exp(y * log(x))` — that would
//! lose most of the answer for a large `|y|`, because an error of one ulp in
//! `log(x)` becomes an error of `y` ulp in the result. Instead the logarithm
//! is computed to *more* than double precision as an unevaluated pair
//! `hi + lo`, the product with `y` is kept as a pair too, and the exponential
//! takes both halves.
//!
//! That extra precision is what the whole structure is for, and it is why
//! `pow` has its own table (`[1/c, log c, log-c tail]`, 128 rows of three)
//! rather than sharing [`super::ln`]'s.
//!
//! # Why there is no separate approximate path
//!
//! [`crate::Accuracy::Fast`] is accepted and has no effect here. That is not
//! an omission, it is what the error analysis says. The logarithm's error is
//! multiplied by `y`, so a table-free logarithm accurate to a fifth of an ulp
//! — which is as good as a degree-21 series gets — leaves `x^y` around 200 ulp
//! out at `|y| = 360`, and the sweep says exactly that. Reaching the accuracy
//! `pow` needs means the double-double table logarithm, and once you have paid
//! for that there is nothing left for an approximation to save: the
//! exponential's table read is one load against a chain of a dozen
//! multiply-adds.
//!
//! So `pow` has one implementation, and it is the bit-exact one.

use cubecl::prelude::*;

use crate::config::Config;
use crate::cube::bits::inf64;
use crate::tables::arena::{OFF_EXP, OFF_POW};
use crate::tables::double::exp as et;
use crate::tables::double::pow as pt;

/// The table-centring offset, `bits(0x1.6955p-1)`.
const OFF: u64 = 0x3fe6955500000000;
/// Added to the exponent field to negate the result: `0x800 << 7`, which lands
/// on the sign bit once the table's `ki` is shifted left by 45.
const SIGN_BIAS: u64 = 0x800 << 7;
/// `2^52`, the scale that normalises a subnormal base.
const P52: f64 = f64::from_bits(0x4330000000000000);
/// `2^1009`.
const P1009: f64 = f64::from_bits(0x7f00000000000000);
/// `2^-1022`.
const P_M1022: f64 = f64::from_bits(0x0010000000000000);
/// `bits(1.0)`.
const ONE: u64 = 0x3ff0000000000000;
/// `bits(inf) * 2`.
const INF2: u64 = 0xffe0000000000000;

/// The top 12 bits of a `f64`: sign and biased exponent.
#[cube]
pub fn top12(x: f64) -> u32 {
    u32::cast_from(u64::reinterpret(x) >> 52u64)
}

/// True when the bit pattern is a zero, an infinity or a NaN.
#[cube]
pub fn zeroinfnan(i: u64) -> bool {
    i * 2u64 - 1u64 >= INF2 - 1u64
}

/// Whether `y` is an integer, and if so whether it is odd.
///
/// `0` for a non-integer, `1` for an odd integer, `2` for an even one — the
/// same three-way answer `checkint` gives in the C source, because `pow` needs
/// all three: a non-integer exponent on a negative base is a domain error, an
/// odd one flips the sign, and an even one does not.
#[cube]
pub fn checkint(iy: u64) -> u32 {
    let e = i32::cast_from((iy >> 52u64) & 0x7ffu64);
    let mut out = 0u32;
    if e < 0x3ffi32 {
        out = 0u32; // |y| < 1
    } else if e > 0x3ffi32 + 52i32 {
        out = 2u32; // too large to have a fractional part; always even
    } else {
        let shift = u64::cast_from(0x3ffi32 + 52i32 - e);
        if iy & ((1u64 << shift) - 1u64) != 0u64 {
            out = 0u32; // fractional bits set
        } else {
            out = select(iy & (1u64 << shift) != 0u64, 1u32, 2u32);
        }
    }
    out
}

/// True when `x` is a signalling NaN.
#[cube]
pub fn is_signaling(x: f64) -> bool {
    let b = u64::reinterpret(x);
    b & 0x7ff0_0000_0000_0000u64 == 0x7ff0_0000_0000_0000u64
        && b & 0x000f_ffff_ffff_ffffu64 != 0u64
        && b & 0x0008_0000_0000_0000u64 == 0u64
}

/// `x^y`.
#[cube]
pub fn pow(x: f64, y: f64, tab: &Array<u64>, #[comptime] cfg: Config) -> f64 {
    let mut ix = u64::reinterpret(x);
    let iy = u64::reinterpret(y);
    let mut topx = top12(x);
    let topy = top12(y);
    let mut sign_bias = 0u64;
    let mut out = 0.0;
    let mut done = false;

    // `x` is not a positive normal, or `|y|` is outside `[2^-65, 2^63)`.
    if topx - 1u32 >= 0x7ffu32 - 1u32 || (topy & 0x7ffu32) - 0x3beu32 >= 0x43eu32 - 0x3beu32 {
        if zeroinfnan(iy) {
            if iy * 2u64 == 0u64 {
                out = select(is_signaling(x), x + y, 1.0);
            } else if ix == ONE {
                out = select(is_signaling(y), x + y, 1.0);
            } else if ix * 2u64 > INF2 || iy * 2u64 > INF2 {
                out = x + y; // a NaN in either argument
            } else if ix * 2u64 == ONE * 2u64 {
                out = 1.0; // (-1)^inf
            } else if (ix * 2u64 < ONE * 2u64) == (iy >> 63u64 == 0u64) {
                out = 0.0; // |x| < 1 with y = +inf, or |x| > 1 with y = -inf
            } else {
                out = y * y;
            }
            done = true;
        } else if zeroinfnan(ix) {
            let x2raw = x * x;
            let x2 = select(ix >> 63u64 != 0u64 && checkint(iy) == 1u32, -x2raw, x2raw);
            // The division is what produces a signed infinity for a zero base
            // with a negative exponent.
            out = select(iy >> 63u64 != 0u64, 1.0 / x2, x2);
            done = true;
        } else {
            if ix >> 63u64 != 0u64 {
                // Finite `x < 0`: only an integer exponent is defined.
                let kind = checkint(iy);
                if kind == 0u32 {
                    out = (x - x) / (x - x); // glibc's `__math_invalid`
                    done = true;
                } else {
                    sign_bias = select(kind == 1u32, SIGN_BIAS, 0u64);
                    ix = ix & 0x7fff_ffff_ffff_ffffu64;
                    topx = topx & 0x7ffu32;
                }
            }
            if !done && (topy & 0x7ffu32) - 0x3beu32 >= 0x43eu32 - 0x3beu32 {
                // `sign_bias` is zero here: such a `y` cannot be odd.
                if ix == ONE {
                    out = 1.0;
                } else if (topy & 0x7ffu32) < 0x3beu32 {
                    // |y| < 2^-65: `x^y` is `1 + y log(x)` to within a rounding.
                    out = select(ix > ONE, 1.0 + y, 1.0 - y);
                } else {
                    out = select((ix > ONE) == (topy < 0x800u32), inf64(), 0.0);
                }
                done = true;
            }
            if !done && topx == 0u32 {
                // Subnormal `x`: normalise so the exponent becomes negative.
                ix = (u64::reinterpret(x * P52) & 0x7fff_ffff_ffff_ffffu64) - (52u64 << 52u64);
            }
        }
    }

    if !done {
        let _ = comptime!(cfg);
        let (hi, lo) = pow_log(ix, tab);
        // `y * (hi + lo)` as an unevaluated pair, both products fused.
        let ehi = y * hi;
        let elo = fma(y, lo, fma(y, hi, -ehi));
        out = pow_exp(ehi, elo, sign_bias, tab);
    }
    out
}

/// `log(x)` to better than double precision, as an unevaluated pair.
#[cube]
pub fn pow_log(ix: u64, tab: &Array<u64>) -> (f64, f64) {
    let tmp = ix - OFF;
    let i = usize::cast_from((tmp >> 45u64) & 127u64);
    let k = i64::reinterpret(tmp) >> 52i64;
    let iz = ix - (tmp & (0xfffu64 << 52u64));
    let z = f64::reinterpret(iz);
    let kd = f64::cast_from(k);

    let base = OFF_POW as usize + 3usize * i;
    let invc = f64::reinterpret(tab[base]);
    let logc = f64::reinterpret(tab[base + 1]);
    let logctail = f64::reinterpret(tab[base + 2]);

    // `|z/c - 1| < 1/N`, and the fused form makes `r` exactly representable.
    let r = fma(z, invc, -1.0);

    // `k ln2 + log(c) + r`, with the low half kept.
    let t1 = kd * pt::LN2HI + logc;
    let t2 = t1 + r;
    let lo1 = kd * pt::LN2LO + logctail;
    let lo2 = t1 - t2 + r;

    let ar = pt::A0 * r; // A0 = -0.5
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let hi = t2 + ar2;
    let lo3 = fma(ar, r, -ar2);
    let lo4 = t2 - hi + ar2;
    let p = ar3 * (pt::A1 + r * pt::A2 + ar2 * (pt::A3 + r * pt::A4 + ar2 * (pt::A5 + r * pt::A6)));
    let lo = lo1 + lo2 + lo3 + lo4 + p;
    let yv = hi + lo;
    (yv, hi - yv + lo)
}

/// `e^(x + xtail)`, with a sign the exponent's parity may have set.
#[cube]
pub fn pow_exp(x: f64, xtail: f64, sign_bias: u64, tab: &Array<u64>) -> f64 {
    let abstop = top12(x) & 0x7ffu32;
    let mut out = 0.0;
    let mut done = false;

    if abstop < 0x3c9u32 {
        // |x| < 2^-54: the result is 1, signed.
        let one = 1.0 + x;
        out = select(sign_bias != 0u64, -one, one);
        done = true;
    } else if abstop >= 0x409u32 {
        let neg = u64::reinterpret(x) >> 63u64 != 0u64;
        let zero = select(sign_bias != 0u64, -0.0, 0.0);
        let big = select(sign_bias != 0u64, -inf64(), inf64());
        out = select(neg, zero, big);
        done = true;
    }

    if !done {
        // `INVLN2N * x` is fused straight into the rounding add, so it never
        // exists as a rounded double — the same placement the C source hides.
        let kd_s = fma(x, et::INVLN2N, et::SHIFT);
        let ki = u64::reinterpret(kd_s);
        let kd = kd_s - et::SHIFT;
        let r = fma(kd, et::NEGLN2LON, fma(kd, et::NEGLN2HIN, x)) + xtail;

        let idx = usize::cast_from(2u64 * (ki % 128u64)) + OFF_EXP as usize;
        let top = (ki + sign_bias) << 45u64;
        let tail = f64::reinterpret(tab[idx]);
        let sbits = tab[idx + 1] + top;

        // The polynomial in the association `__pow_fma` compiles to: two
        // independent coefficient pairs, then two fused accumulations.
        let r2 = r * r;
        let p23 = fma(et::C3, r, et::C2);
        let s = tail + r;
        let p45 = fma(r, et::C5, et::C4);
        let t = fma(p23, r2, s);
        let tmp = fma(p45, r2 * r2, t);

        if abstop >= 0x408u32 {
            out = specialcase(tmp, sbits, ki);
        } else {
            let scale = f64::reinterpret(sbits);
            out = fma(scale, tmp, scale);
        }
    }
    out
}

/// The overflow/underflow fix-up.
///
/// Note the asymmetry between the arms — the first is fused and the second is
/// not. That is the same asymmetry [`super::exp`]'s handler has, and for the
/// same reason: it is what the compiler chose, and it changes the last bit.
#[cube]
pub fn specialcase(tmp: f64, sbits0: u64, ki: u64) -> f64 {
    let mut out = 0.0;
    if ki & 0x8000_0000u64 == 0u64 {
        // k > 0: the scale's exponent may have overflowed by up to 460.
        let scale = f64::reinterpret(sbits0 - (1009u64 << 52u64));
        out = P1009 * fma(scale, tmp, scale);
    } else {
        // k < 0: the result may be subnormal, where a second rounding would
        // cost half an ulp, so the sum is renormalised around +-1 first.
        let sbits = sbits0 + (1022u64 << 52u64);
        let scale = f64::reinterpret(sbits);
        let mut y = scale + scale * tmp;
        if f64::abs(y) < 1.0 {
            let one = select(y < 0.0, -1.0, 1.0);
            let lo0 = scale - y + scale * tmp;
            let hi = one + y;
            let lo = one - hi + y + lo0;
            y = (hi + lo) - one;
            // Fix the sign of a zero the renormalisation produced.
            y = select(y == 0.0, f64::reinterpret(sbits & 0x8000_0000_0000_0000u64), y);
        }
        out = P_M1022 * y;
    }
    out
}
