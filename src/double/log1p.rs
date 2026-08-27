//! `ln(1 + x)`, accurate for small `x`.
//!
//! A port of glibc's `__log1p_fma`. Branchy near zero, where the naive
//! `ln(1.0 + x)` would round away the low bits of a small `x` before the
//! logarithm ever saw them: the reduction recovers exactly what got lost —
//! `c = (u - 1) - x`, exact by Sterbenz wherever it matters — and folds it
//! back in as a first-order correction, `ln(u) - c/u`.
//!
//! [`crate::Accuracy::Fast`] keeps that correction, which is the whole point
//! of the function, and drops the platform's exponent bookkeeping in favour of
//! [`super::ln`]'s own reduction.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::bits::neg_inf64;
use crate::fma::{FmaKind, fma64};

/// High part of `ln(2)`; `n * LN2_HI` is exact for every `|n| < 2000`.
const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000);
/// Low part of `ln(2)`. See [`LN2_HI`].
const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76);

/// `Lp[1..=7]`: the odd-series minimax coefficients for `R(z)` on
/// `s in [0, 0.1716]`, with `s = f / (2 + f)`.
const LP0: f64 = f64::from_bits(0x3FE5555555555593);
const LP1: f64 = f64::from_bits(0x3FD999999997FA04);
const LP2: f64 = f64::from_bits(0x3FD2492494229359);
const LP3: f64 = f64::from_bits(0x3FCC71C51D8E78AF);
const LP4: f64 = f64::from_bits(0x3FC7466496CB03DE);
const LP5: f64 = f64::from_bits(0x3FC39A09D078C69F);
const LP6: f64 = f64::from_bits(0x3FC2F112DF3E5244);

/// `ln(1 + x)`.
#[cube]
pub fn log1p(x: f64, #[comptime] cfg: MathConfig) -> f64 {
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
    let bits = u64::reinterpret(x);
    let hxu = u32::cast_from(bits >> 32u64);
    let hx = i32::reinterpret(hxu);
    let ax = hxu & 0x7fff_ffffu32;

    // A NaN comes back with its sign and payload untouched but the quiet bit
    // forced on. Not `x + x`: on this hardware an identical-operand NaN
    // addition collapses to the canonical quiet NaN and loses the payload,
    // and `s_log1p.c` does not take its domain-error branch for a NaN at all.
    let mut out = f64::reinterpret(bits | (1u64 << 51u64));
    let mut done = ax > 0x7ff0_0000u32 || (ax == 0x7ff0_0000u32 && bits << 32u64 != 0u64);

    if !done {
        let mut k = 1i32;
        let mut c = 0.0;
        let mut hu = 0u32;
        let mut f = 0.0;

        if hx < 0x3FDA_827Au32 as i32 {
            // x < 0.41422
            if ax >= 0x3ff0_0000u32 {
                // x <= -1.0
                // `x < -1` is a genuine runtime `0/0`, which resolves to the
                // hardware's default NaN — sign bit *set*, unlike the positive
                // quiet NaN a constant would give. The sign of a NaN is part
                // of a bit-exactness contract.
                //
                // `x == -1` is `-inf`, which the C source raises by dividing
                // `-2^54` by a literal zero. Spelled as a constant here
                // instead: the division folds at compile time either way, and
                // a folded infinity is a literal the C++ backends cannot
                // print. Nothing in this crate observes the exception flag.
                out = select(x == -1.0, neg_inf64(), (x - x) / (x - x));
                done = true;
            } else if ax < 0x3e20_0000u32 {
                // |x| < 2^-29
                if ax < 0x3c90_0000u32 {
                    out = x; // |x| < 2^-54
                } else {
                    // Fused on the disassembly: `x - (x*x)*0.5` as one
                    // rounding, not a product formed first.
                    out = fma(x * x, -0.5, x);
                }
                done = true;
            } else if hx > 0i32 || hx <= 0xbfd2_bec3u32 as i32 {
                // -0.2929 < x < 0.41422: direct, no reduction needed.
                k = 0i32;
                hu = 1u32;
                f = x;
            }
        } else if ax >= 0x7ff0_0000u32 {
            out = x + x; // +inf
            done = true;
        }

        if !done && !(hx < 0x3FDA_827Au32 as i32 && (hx > 0i32 || hx <= 0xbfd2_bec3u32 as i32)) {
            // Reduce `1 + x` to `2^k (1 + f)`.
            // Past 2^52 the `+1` is a no-op, and glibc skips both it and
            // the correction it would need.
            let huge = hx >= 0x4340_0000u32 as i32;
            let mut u = x;
            if !huge {
                u = 1.0 + x;
            }
            hu = u32::cast_from(u64::reinterpret(u) >> 32u64);
            k = (i32::reinterpret(hu) >> 20i32) - 1023i32;
            // Branches rather than `select` on both forms: the outer arm is a
            // division, and the inner two share no arithmetic, so evaluating
            // both spends a divide and two subtractions to discard three of
            // the four results.
            if !huge {
                let mut num = x - (u - 1.0);
                if k > 0i32 {
                    num = 1.0 - (u - x);
                }
                c = num / u;
            }
            hu = hu & 0x000f_ffffu32;
            let low32 = u64::reinterpret(u) & 0xffff_ffffu64;
            // `0x6a09e` is the top of `sqrt(2)`'s significand: below it the
            // reduced value already sits in `[1, sqrt 2)`, above it one more
            // halving puts it there.
            let big = hu >= 0x6a09eu32;
            let field = select(big, 0x3fe0_0000u32, 0x3ff0_0000u32);
            let u_norm = f64::reinterpret((u64::cast_from(hu | field) << 32u64) | low32);
            k += select(big, 1i32, 0i32);
            hu = select(big, (0x0010_0000u32 - hu) >> 2u32, hu);
            f = u_norm - 1.0;
        }

        if !done {
            out = tail(f, hu, k, c);
        }
    }
    out
}

/// The tail shared by every reduction path.
///
/// The `Lp[]` evaluation is a genuine chain of fused multiply-adds on the
/// disassembly, not the flat sum the C source's grouping suggests:
/// algebraically the same polynomial, not the same roundings.
#[cube]
pub fn tail(f: f64, hu: u32, k: i32, c: f64) -> f64 {
    let kf = f64::cast_from(k);
    let hfsq = 0.5 * f * f;
    let mut out = 0.0;

    if hu == 0u32 {
        // |f| < 2^-20. `1 - (2/3) f` is one fused operation here, not two
        // roundings.
        if f == 0.0 {
            if k == 0i32 {
                out = 0.0;
            } else {
                out = fma(kf, LN2_HI, c + kf * LN2_LO);
            }
        } else {
            let r = fma(f, -0.66666666666666666, 1.0) * hfsq;
            if k == 0i32 {
                out = f - r;
            } else {
                out = kf * LN2_HI - ((r - (kf * LN2_LO + c)) - f);
            }
        }
    } else {
        let s = f / (2.0 + f);
        let z = s * s;
        let r2 = fma(z, LP2, LP1);
        let z2 = z * z;
        let r2z2 = r2 * z2;
        let z4 = z2 * z2;
        let r3 = fma(z, LP4, LP3);
        let z6 = z4 * z2;
        let r4 = fma(z, LP6, LP5);
        let acc0 = fma(z, LP0, r2z2);
        let acc1 = fma(z4, r3, acc0);
        let r = fma(z6, r4, acc1);
        if k == 0i32 {
            out = f - (hfsq - s * (hfsq + r));
        } else {
            // `kf*LN2_HI - (...)` is one fused operation, and so is
            // `kf*LN2_LO + c`. Both matter: this is the branch that dominates
            // large-`|x|` inputs.
            out = fma(kf, LN2_HI, -((hfsq - (s * (hfsq + r) + fma(kf, LN2_LO, c))) - f));
        }
    }
    out
}

/// The table-free path.
///
/// Keeps the reference's correction — recover what `1 + x` rounded away and
/// fold it back as `ln(u) - c/u` — because that is what makes `log1p` worth
/// having, and takes [`super::ln`]'s own reduction for the logarithm itself.
///
/// Maximum error measured against the correctly rounded result: 3 ulp, which
/// is [`super::ln`]'s own 2 plus the correction's rounding. Below `2^-29`,
/// where the series takes over, it is exact to within half an ulp.
#[cube]
pub fn fast(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let u = 1.0 + x;
    // The part of `x` that `1 + x` rounded away, exactly. Sterbenz makes both
    // subtractions exact in the range where either matters. The correction is
    // only meaningful where `u` is a positive finite number; at `x = -1` and
    // at infinity it is `0/0`, and the logarithm alone is the answer.
    // The split is on `u >= 2`, not `u >= 1`: below two, `u - 1` is exact by
    // Sterbenz and `x - (u - 1)` recovers the lost part; at or above it, `x`
    // is the large operand and `1 - (u - x)` is the exact form instead.
    let c0 = select(u >= 2.0, 1.0 - (u - x), x - (u - 1.0)) / u;
    let c = select(u > 0.0 && crate::bits::is_finite64(u), c0, 0.0);
    let mut out = crate::double::ln::fast(u, checked, fk) + c;

    // Below 2^-29 the correction *is* the answer; `1 + x` has thrown away
    // everything else.
    if f64::abs(x) < 1.862645149230957e-9 {
        out = fma64(x * x, -0.5, x, fk);
    }
    out
}
