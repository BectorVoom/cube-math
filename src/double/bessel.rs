//! The Bessel functions: `j0`, `j1`, `y0`, `y1`, `jn`, `yn`.
//!
//! Ports of what glibc actually runs, which for this family is still Sun's
//! fdlibm — `rmath` established that by finding fdlibm's own constants in the
//! installed `libm.so`, where `erf` and `lgamma` no longer have theirs. The
//! schedule is plain `f64` arithmetic with **no fused multiply-adds** anywhere:
//! `mulsd`, `addsd`, `divsd` throughout, and no `vfmadd` in `__j0_finite`. So
//! nothing here takes a [`crate::FmaKind`], and these six are bit-exact even
//! on a backend whose `fma` rounds twice.
//!
//! # The two regimes
//!
//! Below `|x| = 2` each function is a rational approximation in `x^2`, with
//! `y0` and `y1` carrying an explicit logarithm because they are singular at
//! the origin.
//!
//! At or above 2 all four use the asymptotic form
//! `j0(x) = sqrt(2/(pi x)) (P(x) cos(x0) - Q(x) sin(x0))` with `x0 = x - pi/4`,
//! and the interesting part is how `cos(x0)` and `sin(x0)` are obtained.
//! Writing them as `(cos x +- sin x)/sqrt 2` would lose every significant digit
//! wherever the two nearly cancel, so fdlibm computes the *larger* of the two
//! that way and recovers the smaller from
//! `sin x +- cos x = -cos(2x) / (sin x -+ cos x)`, which does not cancel at
//! all. That is why these routines evaluate `cos(x + x)` as well as `sin x`
//! and `cos x`.
//!
//! # What the trigonometry costs here, and no longer costs
//!
//! In `rmath` this family is "bit-exact by delegation" for its trigonometric
//! part: `sin`, `cos` and `ln` are the platform's, because that crate does not
//! reproduce them under every policy. Here they are [`super::trig`] and
//! [`super::ln`], compiled into the same kernel — so the whole function is
//! ported, on a device with no `libm` to delegate to.
//!
//! # Where the work actually goes
//!
//! `jn` and `yn` recur, and how far depends on the arguments: `jn`'s backward
//! branch runs a continued fraction until a convergent exceeds `1e9` and then
//! recurs back down. Two neighbouring threads with different orders diverge for
//! as long as the longer one takes. That is inherent — the trip count is set by
//! the data — and it is the one place in this crate where a warp's cost is
//! decided by its worst element rather than its average.
//!
//! Both policy axes are accepted and have no effect: fdlibm's rational fits
//! *are* the cheap algorithm, and the range handling is on the main path.

use cubecl::prelude::*;

use crate::bits::{is_nan64, neg_inf64, opaque64};
use crate::config::MathConfig;
use crate::double::ln::bit_exact as ln;
use crate::double::trig::{cos, sin};
use crate::tables::consts::{bessel_asympt_tab, bessel_small_tab};
use crate::tables::double::bessel as t;

/// Offsets into [`bessel_small_tab()`], in declaration order.
const J0_R: u32 = 0;
/// `j0`'s denominator, 5 slots.
const J0_S: u32 = J0_R + 6;
/// `y0`'s numerator, 7 slots.
const Y0_U: u32 = J0_S + 5;
/// `y0`'s denominator, 4 slots.
const Y0_V: u32 = Y0_U + 7;
/// `j1`'s numerator, 4 slots.
const J1_R: u32 = Y0_V + 4;
/// `j1`'s denominator, 6 slots.
const J1_S: u32 = J1_R + 4;
/// `y1`'s numerator, 5 slots.
const Y1_U: u32 = J1_S + 6;
/// `y1`'s denominator, 5 slots.
const Y1_V: u32 = Y1_U + 5;

/// Slots per interval in [`bessel_asympt_tab()`].
const AW: u32 = 6;
/// Intervals per table.
const AN: u32 = 4;

/// `ln(DBL_MAX)`, spelled as fdlibm spells it. Rounding it to its shortest
/// round-tripping form would move `jn`'s renormalisation boundary.
const LN_MAX: f64 = 7.09782712893383973096e+02;

/// The high word of `|x|`.
#[cube]
pub fn hi(x: f64) -> u32 {
    u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7fff_ffffu32
}

/// Which of the four rational fits covers `|x|`. fdlibm's `if` ladder, spelled
/// as a count.
#[cube]
pub fn interval(ix: u32) -> u32 {
    let mut i = 3u32;
    if ix >= 0x4020_0000u32 {
        i = 0u32; // [8, inf)
    } else if ix >= 0x4012_2e8bu32 {
        i = 1u32; // [4.5454, 8)
    } else if ix >= 0x4006_db6du32 {
        i = 2u32; // [2.8571, 4.547)
    }
    i
}

/// `1 + R(s)/S(s)` with `s = 1/x^2`, the shape both amplitude factors share.
///
/// `pr` and `ps` are slot offsets into [`bessel_asympt_tab()`].
#[cube]
pub fn rational_p(x: f64, pr: u32, ps: u32) -> f64 {
    let tab = bessel_asympt_tab();
    let p = usize::cast_from(pr);
    let q = usize::cast_from(ps);
    let z = 1.0 / (x * x);
    let r1 = f64::reinterpret(tab[p]) + z * f64::reinterpret(tab[p + 1]);
    let z2 = z * z;
    let r2 = f64::reinterpret(tab[p + 2]) + z * f64::reinterpret(tab[p + 3]);
    let z4 = z2 * z2;
    let r3 = f64::reinterpret(tab[p + 4]) + z * f64::reinterpret(tab[p + 5]);
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = 1.0 + z * f64::reinterpret(tab[q]);
    let s2 = f64::reinterpret(tab[q + 1]) + z * f64::reinterpret(tab[q + 2]);
    let s3 = f64::reinterpret(tab[q + 3]) + z * f64::reinterpret(tab[q + 4]);
    let s = s1 + z2 * s2 + z4 * s3;
    1.0 + r / s
}

/// `R(s)/S(s)` with `s = 1/x^2`, the shape both phase factors share. One term
/// longer in the denominator than [`rational_p()`].
#[cube]
pub fn rational_q(x: f64, pr: u32, ps: u32) -> f64 {
    let tab = bessel_asympt_tab();
    let p = usize::cast_from(pr);
    let q = usize::cast_from(ps);
    let z = 1.0 / (x * x);
    let r1 = f64::reinterpret(tab[p]) + z * f64::reinterpret(tab[p + 1]);
    let z2 = z * z;
    let r2 = f64::reinterpret(tab[p + 2]) + z * f64::reinterpret(tab[p + 3]);
    let z4 = z2 * z2;
    let r3 = f64::reinterpret(tab[p + 4]) + z * f64::reinterpret(tab[p + 5]);
    let z6 = z4 * z2;
    let r = r1 + z2 * r2 + z4 * r3;
    let s1 = 1.0 + z * f64::reinterpret(tab[q]);
    let s2 = f64::reinterpret(tab[q + 1]) + z * f64::reinterpret(tab[q + 2]);
    let s3 = f64::reinterpret(tab[q + 3]) + z * f64::reinterpret(tab[q + 4]);
    let s = s1 + z2 * s2 + z4 * s3 + z6 * f64::reinterpret(tab[q + 5]);
    r / s
}

/// `pzero`: the amplitude factor of the order-0 asymptotic form.
#[cube]
pub fn pzero(x: f64) -> f64 {
    let ix = hi(x);
    // Initialised from a runtime expression, which a `#[cube]` local that a
    // branch reassigns has to be; `x` is finite at every call site, so this is
    // the `1.0` the C source writes.
    let mut out = 1.0 + 0.0 * x;
    if ix < 0x41b0_0000u32 {
        let i = interval(ix);
        out = rational_p(x, i * AW, (AN + i) * AW);
    }
    out
}

/// `qzero`: the phase factor of the order-0 asymptotic form.
#[cube]
pub fn qzero(x: f64) -> f64 {
    let ix = hi(x);
    let mut out = -0.125 / x;
    if ix < 0x41b0_0000u32 {
        let i = interval(ix);
        out = (-0.125 + rational_q(x, (2u32 * AN + i) * AW, (3u32 * AN + i) * AW)) / x;
    }
    out
}

/// `pone`: the amplitude factor of the order-1 asymptotic form.
#[cube]
pub fn pone(x: f64) -> f64 {
    let ix = hi(x);
    let mut out = 1.0 + 0.0 * x;
    if ix < 0x41b0_0000u32 {
        let i = interval(ix);
        out = rational_p(x, (4u32 * AN + i) * AW, (5u32 * AN + i) * AW);
    }
    out
}

/// `qone`: the phase factor of the order-1 asymptotic form.
#[cube]
pub fn qone(x: f64) -> f64 {
    let ix = hi(x);
    let mut out = 0.375 / x;
    if ix < 0x41b0_0000u32 {
        let i = interval(ix);
        out = (0.375 + rational_q(x, (6u32 * AN + i) * AW, (7u32 * AN + i) * AW)) / x;
    }
    out
}

/// `(sin x - cos x, sin x + cos x)`, each computed the way that does not
/// cancel.
///
/// `ss` is `sqrt(2) sin(x - pi/4)` and `cc` is `sqrt(2) cos(x - pi/4)`;
/// whichever is small is recovered from `-cos(2x)` divided by the other, since
/// `(sin x - cos x)(sin x + cos x) = -cos(2x)` exactly.
#[cube]
pub fn ss_cc(x: f64, ix: u32, #[comptime] cfg: MathConfig) -> (f64, f64) {
    let s = sin(x, cfg);
    let c = cos(x, cfg);
    let mut ss = s - c;
    let mut cc = s + c;
    if ix < 0x7fe0_0000u32 {
        // Guarded so that `x + x` cannot overflow to an infinity.
        let z = -cos(x + x, cfg);
        if s * c < 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
    }
    (ss, cc)
}

/// The order-1 counterpart of [`ss_cc()`]: the same construction with the
/// roles and signs exchanged, because `x0` is `x - 3pi/4` rather than
/// `x - pi/4`.
#[cube]
pub fn ss_cc_one(x: f64, ix: u32, #[comptime] cfg: MathConfig) -> (f64, f64) {
    let s = sin(x, cfg);
    let c = cos(x, cfg);
    let mut ss = -s - c;
    let mut cc = s - c;
    if ix < 0x7fe0_0000u32 {
        let z = cos(x + x, cfg);
        if s * c > 0.0 {
            cc = z / ss;
        } else {
            ss = z / cc;
        }
    }
    (ss, cc)
}

/// The Bessel function of the first kind, order 0.
#[cube]
pub fn j0(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x0 = opaque64(x0);
    let ix = hi(x0);
    let x = f64::abs(x0);
    let mut out = 1.0 / (x0 * x0);
    if ix < 0x7ff0_0000u32 {
        if ix >= 0x4000_0000u32 {
            let (ss, cc) = ss_cc(x, ix, cfg);
            if ix > 0x4800_0000u32 {
                // Beyond `2^129` the rational fits are indistinguishable from
                // `1` and `0`, so they are dropped entirely. A branch rather
                // than a blend: the discarded arm is two rational fits, and a
                // thread that only needs the leading term should not pay for
                // them.
                out = (t::INVSQRTPI * cc) / f64::sqrt(x);
            } else {
                out = t::INVSQRTPI * (pzero(x) * cc - qzero(x) * ss) / f64::sqrt(x);
            }
        } else {
            let z = x * x;
            let tab = bessel_small_tab();
            let r = usize::cast_from(J0_R);
            let s = usize::cast_from(J0_S);

            let r1 = z * f64::reinterpret(tab[r + 2]);
            let z2 = z * z;
            let r2 = f64::reinterpret(tab[r + 3]) + z * f64::reinterpret(tab[r + 4]);
            let z4 = z2 * z2;
            let rn = r1 + z2 * r2 + z4 * f64::reinterpret(tab[r + 5]);

            let s1 = 1.0 + z * f64::reinterpret(tab[s + 1]);
            let s2 = f64::reinterpret(tab[s + 2]) + z * f64::reinterpret(tab[s + 3]);
            let sd = s1 + z2 * s2 + z4 * f64::reinterpret(tab[s + 4]);

            let u = 0.5 * x;
            out = (1.0 + u) * (1.0 - u) + z * (rn / sd);
            if ix < 0x3ff0_0000u32 {
                out = 1.0 + z * (-0.25 + (rn / sd));
            }
            // `HUGE + x > 1.0` is fdlibm's inexact-raising guard and is always
            // true, so the branches it wraps are unconditional here.
            if ix < 0x3f20_0000u32 {
                out = 1.0 - 0.25 * x * x; // `|x| < 2^-13`
                if ix < 0x3e40_0000u32 {
                    out = 1.0; // `|x| < 2^-27`
                }
            }
        }
    }
    out
}

/// The Bessel function of the second kind, order 0.
#[cube]
pub fn y0(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let ix = hi(x);
    let bits = u64::reinterpret(x);
    let mut out = 1.0 / (x + x * x);
    if ix < 0x7ff0_0000u32 {
        if bits & 0x7fff_ffff_ffff_ffffu64 == 0u64 {
            out = neg_inf64();
        } else if bits >> 63u64 != 0u64 {
            // A NaN for a negative argument. fdlibm spells it `0.0/(0.0*x)`;
            // written here as the equal-operand form, because a literal zero
            // times a value is something the backends fold away and this is
            // not — it is a genuine runtime division of zero by zero, which
            // resolves to the hardware's own default NaN, sign bit and all.
            out = (x - x) / (x - x);
        } else if ix >= 0x4000_0000u32 {
            let (ss, cc) = ss_cc(x, ix, cfg);
            if ix > 0x4800_0000u32 {
                out = (t::INVSQRTPI * ss) / f64::sqrt(x);
            } else {
                out = t::INVSQRTPI * (pzero(x) * ss + qzero(x) * cc) / f64::sqrt(x);
            }
        } else if ix <= 0x3e40_0000u32 {
            // `x < 2^-27`: `U/V` is `U[0]` and `j0(x)` is `1`.
            let tab = bessel_small_tab();
            out = f64::reinterpret(tab[usize::cast_from(Y0_U)])
                + t::TPI * ln(x, comptime!(cfg.fma()));
        } else {
            let z = x * x;
            let tab = bessel_small_tab();
            let u = usize::cast_from(Y0_U);
            let v = usize::cast_from(Y0_V);

            let u1 = f64::reinterpret(tab[u]) + z * f64::reinterpret(tab[u + 1]);
            let z2 = z * z;
            let u2 = f64::reinterpret(tab[u + 2]) + z * f64::reinterpret(tab[u + 3]);
            let z4 = z2 * z2;
            let u3 = f64::reinterpret(tab[u + 4]) + z * f64::reinterpret(tab[u + 5]);
            let z6 = z4 * z2;
            let un = u1 + z2 * u2 + z4 * u3 + z6 * f64::reinterpret(tab[u + 6]);

            let v1 = 1.0 + z * f64::reinterpret(tab[v]);
            let v2 = f64::reinterpret(tab[v + 1]) + z * f64::reinterpret(tab[v + 2]);
            let vd = v1 + z2 * v2 + z4 * f64::reinterpret(tab[v + 3]);

            out = un / vd + t::TPI * (j0(x, cfg) * ln(x, comptime!(cfg.fma())));
        }
    }
    out
}

/// The Bessel function of the first kind, order 1.
#[cube]
pub fn j1(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let ix = hi(x);
    let y = f64::abs(x);
    let mut out = 1.0 / x;
    if ix < 0x7ff0_0000u32 {
        if ix >= 0x4000_0000u32 {
            let (ss, cc) = ss_cc_one(y, ix, cfg);
            let mut z = 0.0 + y;
            if ix > 0x4800_0000u32 {
                z = (t::INVSQRTPI * cc) / f64::sqrt(y);
            } else {
                z = t::INVSQRTPI * (pone(y) * cc - qone(y) * ss) / f64::sqrt(y);
            }
            out = select(u64::reinterpret(x) >> 63u64 != 0u64, -z, z);
        } else {
            let z = x * x;
            let tab = bessel_small_tab();
            let r = usize::cast_from(J1_R);
            let s = usize::cast_from(J1_S);

            let r1 = z * f64::reinterpret(tab[r]);
            let z2 = z * z;
            let r2 = f64::reinterpret(tab[r + 1]) + z * f64::reinterpret(tab[r + 2]);
            let z4 = z2 * z2;
            let rn = (r1 + z2 * r2 + z4 * f64::reinterpret(tab[r + 3])) * x;

            let s1 = 1.0 + z * f64::reinterpret(tab[s + 1]);
            let s2 = f64::reinterpret(tab[s + 2]) + z * f64::reinterpret(tab[s + 3]);
            let s3 = f64::reinterpret(tab[s + 4]) + z * f64::reinterpret(tab[s + 5]);
            let sd = s1 + z2 * s2 + z4 * s3;

            out = x * 0.5 + rn / sd;
            if ix < 0x3e40_0000u32 {
                // `|x| < 2^-27`: `j1(x)` is `x/2`, and the multiply is what
                // raises underflow when it should.
                out = 0.5 * x;
            }
        }
    }
    out
}

/// The Bessel function of the second kind, order 1.
#[cube]
pub fn y1(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let ix = hi(x);
    let bits = u64::reinterpret(x);
    let mut out = 1.0 / (x + x * x);
    if ix < 0x7ff0_0000u32 {
        if bits & 0x7fff_ffff_ffff_ffffu64 == 0u64 {
            out = neg_inf64();
        } else if bits >> 63u64 != 0u64 {
            // See `y0`.
            out = (x - x) / (x - x);
        } else if ix >= 0x4000_0000u32 {
            let (ss, cc) = ss_cc_one(x, ix, cfg);
            if ix > 0x4800_0000u32 {
                out = (t::INVSQRTPI * ss) / f64::sqrt(x);
            } else {
                out = t::INVSQRTPI * (pone(x) * ss + qone(x) * cc) / f64::sqrt(x);
            }
        } else if ix <= 0x3c90_0000u32 {
            // `x < 2^-54`, where `y1(x)` is `-2/(pi x)` and nothing else
            // survives.
            out = -t::TPI / x;
        } else {
            let z = x * x;
            let tab = bessel_small_tab();
            let u = usize::cast_from(Y1_U);
            let v = usize::cast_from(Y1_V);

            let u1 = f64::reinterpret(tab[u]) + z * f64::reinterpret(tab[u + 1]);
            let z2 = z * z;
            let u2 = f64::reinterpret(tab[u + 2]) + z * f64::reinterpret(tab[u + 3]);
            let z4 = z2 * z2;
            let un = u1 + z2 * u2 + z4 * f64::reinterpret(tab[u + 4]);

            let v1 = 1.0 + z * f64::reinterpret(tab[v]);
            let v2 = f64::reinterpret(tab[v + 1]) + z * f64::reinterpret(tab[v + 2]);
            let v3 = f64::reinterpret(tab[v + 3]) + z * f64::reinterpret(tab[v + 4]);
            let vd = v1 + z2 * v2 + z4 * v3;

            out = x * (un / vd) + t::TPI * (j1(x, cfg) * ln(x, comptime!(cfg.fma())) - 1.0 / x);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// jn and yn
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind, order `n`.
///
/// Three regimes, and which one runs depends on `n` against `x` rather than on
/// `x` alone:
///
/// * `n <= x` — forward recurrence from `j0` and `j1`, which is stable in that
///   direction.
/// * `n > x`, `x` not tiny — *backward* recurrence, because forward recurrence
///   is violently unstable there. The starting ratio comes from a continued
///   fraction whose length is decided at run time by iterating until a
///   convergent exceeds `1e9`, and the whole thing is renormalised at the end
///   against whichever of `j0(x)` and `j1(x)` is further from zero.
/// * `x < 2^-29` — the leading term of the Taylor series, `(x/2)^n / n!`.
#[cube]
pub fn jn(nf: f64, x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let xin = opaque64(x0);
    let n0 = i32::cast_from(nf);
    // `J(-n, x) = J(n, -x)`, so the negative orders fold away immediately.
    let n = select(n0 < 0i32, -n0, n0);
    let xs = select(n0 < 0i32, -xin, xin);
    let negative = u64::reinterpret(xs) >> 63u64 != 0u64;
    // Even `n` is even in `x`; odd `n` is odd.
    let sgn = (n & 1i32) != 0i32 && negative;
    let x = f64::abs(xs);
    let ix = hi(xs);

    let mut out = xin + xin; // a NaN argument propagates
    if !is_nan64(xin) {
        if n == 0i32 {
            out = j0(x, cfg);
        } else if n == 1i32 {
            out = j1(select(negative, -x, x), cfg);
        } else if x == 0.0 || ix >= 0x7ff0_0000u32 {
            out = select(sgn, -0.0, 0.0);
        } else {
            let mut b = 0.0 + x;
            if f64::cast_from(n) <= x {
                if ix >= 0x52d0_0000u32 {
                    // `x > 2^302`, where `x` dwarfs `n^2` and the asymptotic
                    // form is exact to the last bit.
                    let s = sin(x, cfg);
                    let c = cos(x, cfg);
                    let m = n & 3i32;
                    let mut temp = c - s;
                    if m == 0i32 {
                        temp = c + s;
                    } else if m == 1i32 {
                        temp = -c + s;
                    } else if m == 2i32 {
                        temp = -c - s;
                    }
                    b = t::INVSQRTPI * temp / f64::sqrt(x);
                } else {
                    let mut a = j0(x, cfg);
                    b = j1(x, cfg);
                    let mut i = 1i32.runtime();
                    while i < n {
                        let temp = b;
                        // The division rather than a multiply by `1/x` is
                        // fdlibm's, and it is what keeps the recurrence from
                        // underflowing.
                        b = b * (f64::cast_from(i + i) / x) - a;
                        a = temp;
                        i = i + 1i32;
                    }
                }
            } else if ix < 0x3e10_0000u32 {
                // `x < 2^-29`: `J(n,x) = (x/2)^n / n!`, and above `n = 33`
                // that underflows to zero.
                b = 0.0 * x;
                if n <= 33i32 {
                    let temp = x * 0.5;
                    b = temp;
                    let mut a = 1.0 + 0.0 * x;
                    let mut i = 2i32.runtime();
                    while i <= n {
                        a = a * f64::cast_from(i);
                        b = b * temp;
                        i = i + 1i32;
                    }
                    b = b / a;
                }
            } else {
                // Backward recurrence. The continued fraction for
                // `J(n,x)/J(n-1,x)` runs until a convergent exceeds `1e9`,
                // which is fdlibm's stopping rule for double precision.
                let w = f64::cast_from(n + n) / x;
                let h = 2.0 / x;
                let mut q0 = w;
                let mut z = w + h;
                let mut q1 = w * z - 1.0;
                let mut k = 1i32.runtime();
                while q1 < 1.0e9 {
                    k = k + 1i32;
                    z = z + h;
                    let tmp = z * q1 - q0;
                    q0 = q1;
                    q1 = tmp;
                }

                let m = n + n;
                let mut tt = 0.0 * x;
                let mut i = 2i32 * (n + k);
                while i >= m {
                    tt = 1.0 / (f64::cast_from(i) / x - tt);
                    i = i - 2i32;
                }
                let mut a = tt;
                b = 1.0 + 0.0 * x;

                // If `n ln(2n/x)` exceeds `ln(DBL_MAX)` the recurrence
                // overflows on the way down, so the second loop renormalises
                // whenever it gets large. Both loops exist because the test is
                // made once, not per iteration.
                let v = 2.0 / x;
                let tmp =
                    f64::cast_from(n) * ln(f64::abs(v * f64::cast_from(n)), comptime!(cfg.fma()));
                let mut di = f64::cast_from((n - 1i32) + (n - 1i32));
                let mut j = 1i32.runtime();
                if tmp < LN_MAX {
                    while j < n {
                        let temp = b;
                        b = b * di;
                        b = b / x - a;
                        a = temp;
                        di = di - 2.0;
                        j = j + 1i32;
                    }
                } else {
                    while j < n {
                        let temp = b;
                        b = b * di;
                        b = b / x - a;
                        a = temp;
                        di = di - 2.0;
                        if b > 1e100 {
                            a = a / b;
                            tt = tt / b;
                            b = 1.0;
                        }
                        j = j + 1i32;
                    }
                }
                // Normalise against `j0` or `j1` — whichever is further from
                // zero, since their zeros never coincide. The result goes to a
                // fresh variable: the `j0` arm divides by the recurrence's own
                // `b`, so assigning into `b` first would divide by the other
                // arm's answer instead.
                let zz = j0(x, cfg);
                let ww = j1(x, cfg);
                let mut res = tt * ww / a;
                if f64::abs(zz) >= f64::abs(ww) {
                    res = tt * zz / b;
                }
                b = res;
            }
            out = select(sgn, -b, b);
        }
    }
    out
}

/// The Bessel function of the second kind, order `n`.
///
/// Forward recurrence from `y0` and `y1` throughout, which for `y` is the
/// *stable* direction — the opposite of `j`. It stops early if the recurrence
/// reaches `-inf`, which it does for moderate `n` at small `x`.
#[cube]
pub fn yn(nf: f64, x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let n0 = i32::cast_from(nf);
    // `Y(-n, x) = (-1)^n Y(n, x)`.
    let n = select(n0 < 0i32, -n0, n0);
    let sign = select(n0 < 0i32, 1i32 - ((n & 1i32) << 1u32), 1i32);
    let ix = hi(x);
    let bits = u64::reinterpret(x);

    let mut out = x + x; // a NaN argument propagates
    if !is_nan64(x) {
        if n == 0i32 {
            out = y0(x, cfg);
        } else if bits & 0x7fff_ffff_ffff_ffffu64 == 0u64 {
            out = select(sign > 0i32, neg_inf64(), -neg_inf64());
        } else if bits >> 63u64 != 0u64 {
            // See `y0`.
            out = (x - x) / (x - x);
        } else if n == 1i32 {
            out = f64::cast_from(sign) * y1(x, cfg);
        } else if ix == 0x7ff0_0000u32 {
            out = 0.0;
        } else {
            let mut b = 0.0 + x;
            if ix >= 0x52d0_0000u32 {
                let s = sin(x, cfg);
                let c = cos(x, cfg);
                let m = n & 3i32;
                let mut temp = s + c;
                if m == 0i32 {
                    temp = s - c;
                } else if m == 1i32 {
                    temp = -s - c;
                } else if m == 2i32 {
                    temp = -s + c;
                }
                b = t::INVSQRTPI * temp / f64::sqrt(x);
            } else {
                let mut a = y0(x, cfg);
                b = y1(x, cfg);
                let mut high = u32::cast_from(u64::reinterpret(b) >> 32u64);
                let mut i = 1i32.runtime();
                while i < n && high != 0xfff0_0000u32 {
                    let temp = b;
                    b = (f64::cast_from(i + i) / x) * b - a;
                    high = u32::cast_from(u64::reinterpret(b) >> 32u64);
                    a = temp;
                    i = i + 1i32;
                }
            }
            out = select(sign > 0i32, b, -b);
        }
    }
    out
}
