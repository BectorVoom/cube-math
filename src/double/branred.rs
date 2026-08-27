//! `__branred`: the Payne-Hanek argument reduction the trigonometric family
//! needs past `105414350`.
//!
//! A port of glibc's `branred.c` (the IBM Accurate Mathematical Library),
//! reducing `x` to `x = n pi/2 + a + aa` with `|a + aa| < pi/4` and returning
//! `n mod 4` alongside.
//!
//! # Why this had to be ported and the rest of the family did not
//!
//! `rmath` leaves this band to the platform: its vector kernels compute the
//! whole vector by the table algorithm and then repair the handful of lanes
//! that landed here by calling the scalar `libm` on them one at a time. A
//! kernel has no platform to fall back to — there is no `sin` on the device
//! but the one this crate compiles — so `sin` on a GPU is not ported at all
//! unless `__branred` is ported with it. That is the whole reason this file
//! exists, and the reason the trigonometric family was the last of the
//! double-precision set to land rather than the first.
//!
//! # Accuracy
//!
//! The reduction carries `x * 2/pi` to about 128 bits by multiplying the
//! argument's two 26-bit halves against `2/pi` held as 24-bit digits, keeping
//! six digits either side of the binade the argument selects. Six digits is
//! what makes the result exact enough that the fractional part is still
//! correct to a `double`-and-a-half after the leading integer part has
//! cancelled — which for an argument near `2^1000` means knowing `2/pi` to a
//! thousand bits and using six of its digits' worth around the right place.
//!
//! # On fused multiply-adds
//!
//! Unlike every other kernel here, this one's fusion placement was **not**
//! read out of a disassembly — `rmath` never ported this function, so there
//! was no prior reading to inherit. It is written unfused, exactly as the C
//! source is spelled, and the equivalence sweep is what settles the question:
//! the inputs in this band are compared against the platform's own `sin`,
//! `cos` and `tan`, so a wrong fusion is a test failure and not a silent
//! difference. It passes unfused, which is the answer.

use cubecl::prelude::*;

use crate::tables::consts::toverp_tab;
use crate::tables::double::trig as t;

/// `branred.h`'s own `mp2` — the second part of its `pi/2` split.
///
/// **Not** [`t::MP2`]. `usncs.h` and `branred.h` each declare a `mp2`, they
/// have the same name and the same role, and they are *different numbers*:
/// `0xbe4dde973c000000` there against `0xbe4dde9740000000` here. `rmath`'s
/// table generator reads `usncs.h`'s, because `rmath` never ported this
/// function and never needed the other one; nothing in the generated tables
/// carries it, so it is written here. `hp0`, `hp1` and `mp1` *are*
/// bit-identical between the two headers and are taken from the table.
///
/// The difference is 4 parts in 2^27 of one residue term and it is not
/// cosmetic: using `usncs.h`'s constant leaves the reduction about a third of
/// an ulp out, which turns into a wrong last bit in `sin` for roughly one
/// argument in four across this band. The equivalence sweep catches it — that
/// is how it was found.
const MP2: f64 = f64::from_bits(0xbe4dde9740000000);

/// One 26-bit half of the scaled argument, folded against `2/pi`.
///
/// Returns `(b, bb, sum)`: the fractional part as an unevaluated double-double
/// and the integer part that came off it.
#[cube]
pub fn half(xh: f64) -> (f64, f64, f64) {
    let toverp = toverp_tab();

    // Which six `2/pi` digits this binade needs. `k` is clamped rather than
    // wrapped: below `2^-450`-ish the leading digits are the right ones.
    let kexp = u32::cast_from(u64::reinterpret(xh) >> 52u64) & 2047u32;
    let mut k = 0u32;
    if kexp >= 450u32 {
        k = (kexp - 450u32) / 24u32;
    }
    // `2^576` walked down by `2^24` per digit, so that digit `k + i` lands at
    // the weight the multiply needs. Spelled as a subtraction from the
    // exponent field, which is what the C source's `mynumber` union does.
    let mut gor = f64::reinterpret(u64::reinterpret(t::T576) - (u64::cast_from(k * 24u32) << 52u64));
    let i = usize::cast_from(k);

    let mut r0 = xh * f64::reinterpret(toverp[i]) * gor;
    gor = gor * t::TM24;
    let mut r1 = xh * f64::reinterpret(toverp[i + 1]) * gor;
    gor = gor * t::TM24;
    let mut r2 = xh * f64::reinterpret(toverp[i + 2]) * gor;
    gor = gor * t::TM24;
    let r3 = xh * f64::reinterpret(toverp[i + 3]) * gor;
    gor = gor * t::TM24;
    let r4 = xh * f64::reinterpret(toverp[i + 4]) * gor;
    gor = gor * t::TM24;
    let r5 = xh * f64::reinterpret(toverp[i + 5]) * gor;

    // Peel the integer part off the three leading digits — only they can
    // carry one — and accumulate it in `sum`.
    let s0 = (r0 + t::BRANRED_BIG) - t::BRANRED_BIG;
    r0 = r0 - s0;
    let s1 = (r1 + t::BRANRED_BIG) - t::BRANRED_BIG;
    r1 = r1 - s1;
    let s2 = (r2 + t::BRANRED_BIG) - t::BRANRED_BIG;
    r2 = r2 - s2;
    let mut sum = (s0 + s1) + s2;

    // Summed smallest first, which is what makes `bb` the residual rather
    // than noise.
    let mut tt = 0.0 + r5;
    tt = tt + r4;
    tt = tt + r3;
    tt = tt + r2;
    tt = tt + r1;
    tt = tt + r0;
    let mut bb = (((((r0 - tt) + r1) + r2) + r3) + r4) + r5;

    let s3 = (tt + t::BRANRED_BIG) - t::BRANRED_BIG;
    sum = sum + s3;
    tt = tt - s3;
    let b = tt + bb;
    bb = (tt - b) + bb;
    // `BIG1`'s ulp is 4, so this leaves `|sum| <= 2`: the quadrant count is
    // all that is wanted, and the multiples of four that fall out of it are
    // exactly the whole turns.
    let s4 = (sum + t::BRANRED_BIG1) - t::BRANRED_BIG1;
    sum = sum - s4;
    (b, bb, sum)
}

/// `x = n pi/2 + a + aa`, returning `(a, aa, n mod 4)`.
#[cube]
pub fn branred(x0: f64) -> (f64, f64, u32) {
    // Scaling by `2^-600` first is what keeps every product below in range;
    // it is exact, and the `2^576` the digits are weighted by puts it back.
    let x = x0 * t::TM600;
    let tsplit = x * t::SPLIT;
    let x1 = tsplit - (tsplit - x);
    let x2 = x - x1;

    let (b1, bb1, sum1) = half(x1);
    let (b2, bb2, sum2) = half(x2);

    let mut sum = sum1 + sum2;
    let mut b = b1 + b2;
    let mut bb = (b2 - b) + b1;
    if f64::abs(b1) > f64::abs(b2) {
        bb = (b1 - b) + b2;
    }
    // Fold the fraction into `(-1/2, 1/2]` and move the carry into `sum`.
    if b > 0.5 {
        b = b - 1.0;
        sum = sum + 1.0;
    } else if b < -0.5 {
        b = b + 1.0;
        sum = sum - 1.0;
    }

    let s = b + (bb + bb1 + bb2);
    let tt = ((b - s) + bb) + (bb1 + bb2);

    // Multiply the fraction back by `pi/2`, in three-part form so that the
    // product is good to more than a `double`.
    let bs = s * t::SPLIT;
    let t1 = bs - (bs - s);
    let t2 = s - t1;
    let bh = s * t::HP0;
    let bl = (((t1 * t::MP1 - bh) + t1 * MP2) + t2 * t::MP1)
        + (t2 * MP2 + s * t::HP1 + tt * t::HP0);
    let a = bh + bl;
    let aa = (bh - a) + bl;

    (a, aa, u32::reinterpret(i32::cast_from(sum) & 3i32))
}
