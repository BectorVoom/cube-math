//! The Bessel functions in single precision: `j0f`, `j1f`, `y0f`, `y1f`,
//! `jnf`, `ynf`.
//!
//! Same family as [`crate::double::bessel`], **different algorithm**. glibc's
//! `float` versions are fdlibm's rational fits plus a repair the `double` ones
//! do not have, and the repair is the interesting part.
//!
//! # Why the rational fit alone is not enough
//!
//! `j0` is computed as `sqrt(2/(pi x)) (P cos(x0) - Q sin(x0))`. Near a zero
//! of `j0` the bracket is the difference of two nearly equal numbers, so it
//! loses digits — not a few, but most of them, because the zeros of `j0` are
//! not representable and the cancellation is essentially total. glibc's answer
//! is to fit a cubic to each of the first 64 zeros and use it whenever the
//! bracket comes out small, and beyond the 64th zero to fall back on an
//! asymptotic form with a Payne-Hanek argument reduction.
//!
//! So each of the four order-0 and order-1 routines has three paths — the
//! rational fit, the near-a-zero polynomial, and the asymptotic form — and
//! which one runs is decided *after* the rational fit, by the size of the
//! bracket.
//!
//! On a CPU vector unit that data-dependent branch is fatal, and it is why
//! this is the one family `rmath`'s single-precision kernels do not
//! vectorise at all. Here it is a branch like any other: threads that land
//! near a zero diverge for the length of a cubic, which is four operations.
//! The family that could not be vectorised is the family a kernel handles
//! without comment.
//!
//! These are not correctly rounded — glibc states a bound of 9 ulp — so
//! matching them means reproducing the schedule, as everywhere else.
//!
//! Both policy axes are accepted and have no effect.

use cubecl::prelude::*;

use crate::bits::{inf32, is_nan32, nan32, neg_inf32, opaque32};
use crate::config::MathConfig;
use crate::single::exact::round;
use crate::single::logx::ln;
use crate::single::trig::{cos, sin};
use crate::tables::consts::{besself_asympt_tab, besself_small_tab, besself_zeros_tab};
use crate::tables::single::bessel as t;

/// `pi` in `f32`, glibc's `M_PIf`.
const PI_F32: f32 = f32::from_bits(0x40490fdb);
/// `sqrt(2/pi)` rounded to `f32`, the asymptotic branches' scale.
const SQRT_2_OVER_PI: f32 = f32::from_bits(0x3f4c422a);
/// How many zeros the near-a-zero tables cover.
const SMALL_SIZE: f32 = 64.0;
/// `2 pi * 2^-64`, the scale that turns the fixed-point remainder into radians.
const PI63: f64 = f64::from_bits(0x3c1921fb54442d18);
/// `pi/2` to `double`.
const PI_OVER_2: f64 = f64::from_bits(0x3ff921fb54442d18);
/// `ln(FLT_MAX)`, as fdlibm spells it. See the double-precision port for why
/// the digits are not trimmed to the shortest round-tripping form.
const LN_MAX: f32 = 8.8721679688e+01;

/// Offsets into [`besself_small_tab()`], in declaration order.
const J0_R: u32 = 0;
/// `j0`'s denominator, 4 slots.
const J0_S: u32 = J0_R + 4;
/// `y0`'s numerator, 7 slots.
const Y0_U: u32 = J0_S + 4;
/// `y0`'s denominator, 4 slots.
const Y0_V: u32 = Y0_U + 7;
/// `j1`'s numerator, 4 slots.
const J1_R: u32 = Y0_V + 4;
/// `j1`'s denominator, 5 slots.
const J1_S: u32 = J1_R + 4;
/// `y1`'s numerator, 5 slots.
const Y1_U: u32 = J1_S + 5;
/// `y1`'s denominator, 5 slots.
const Y1_V: u32 = Y1_U + 5;

/// Slots per interval in [`besself_asympt_tab()`].
const AW: u32 = 6;
/// Intervals per table.
const AN: u32 = 4;
/// Slots per row of [`besself_zeros_tab()`].
const ZW: u32 = 7;
/// Rows per family.
const ZN: u32 = 64;

/// `|x|`'s bit pattern.
#[cube]
pub fn ix(x: f32) -> u32 {
    u32::reinterpret(x) & 0x7fff_ffffu32
}

/// One `f32` out of a `u32` table.
#[cube]
pub fn at(tab: &Array<u32>, i: u32) -> f32 {
    f32::reinterpret(tab[usize::cast_from(i)])
}

/// Which of the four rational fits covers `|x|`.
#[cube]
pub fn interval(i: u32) -> u32 {
    let mut k = 3u32;
    if i >= 0x4100_0000u32 {
        k = 0u32;
    } else if i >= 0x40f7_1c58u32 {
        k = 1u32;
    } else if i >= 0x4036_db68u32 {
        k = 2u32;
    }
    k
}

// ---------------------------------------------------------------------------
// The Payne-Hanek reduction the asymptotic branches need
// ---------------------------------------------------------------------------

/// `|x|` modulo `pi/2`, exactly: `(h, n)` with `|x| = h + n pi/2` and
/// `|h| <= pi/4`.
///
/// The same 32x96-to-128-bit multiply against a 192-bit `4/pi` that
/// [`crate::single::trig`] uses, and for the same reason: subtracting a
/// `double` approximation of `pi/2` cannot work, because for `x` near `2^127`
/// the reduced argument depends on bits of `4/pi` far below anything a
/// `double` holds.
#[cube]
pub fn reduce_large(xi0: u32) -> (f64, i32) {
    let tab = Array::<u32>::from_data(comptime!(t::INV_PIO4.to_vec()));
    let base = usize::cast_from((xi0 >> 26u32) & 15u32);
    let shift = (xi0 >> 23u32) & 7u32;
    let xi = ((xi0 & 0x00ff_ffffu32) | 0x0080_0000u32) << shift;

    let res0a = u64::cast_from(xi) * u64::cast_from(tab[base]);
    let res1 = u64::cast_from(xi) * u64::cast_from(tab[base + 4]);
    let res2 = u64::cast_from(xi) * u64::cast_from(tab[base + 8]);
    let res0 = ((res2 >> 32u64) | (res0a << 32u64)) + res1;

    let n = (res0 + (1u64 << 61u64)) >> 62u64;
    let rem = res0 - (n << 62u64);
    (f64::cast_from(i64::reinterpret(rem)) * PI63, i32::cast_from(n))
}

/// `(h, n)` with `x - pi/4 - alpha = h + n pi/2` modulo `2 pi`.
///
/// `alpha` is the asymptotic phase correction, folded into the reduction
/// rather than added afterwards so that it does not reintroduce the
/// cancellation the reduction just removed.
#[cube]
pub fn reduce_aux(x: f32, alpha: f64) -> (f64, i32) {
    let (h0, n0) = reduce_large(u32::reinterpret(x));
    let mut h = select(x < 0.0, -h0, h0);
    let mut n = select(x < 0.0, -n0, n0);
    if h >= 0.0 {
        h = h - PI_OVER_2 / 2.0;
    } else {
        h = h + PI_OVER_2 / 2.0;
        n = n - 1i32;
    }
    h = h - alpha;
    if h > PI_OVER_2 {
        h = h - PI_OVER_2;
        n = n + 1i32;
    } else if h < -PI_OVER_2 {
        h = h + PI_OVER_2;
        n = n - 1i32;
    }
    (h, n)
}

/// `t cos(xr + n pi/2)`, the tail every asymptotic `j` branch ends with.
#[cube]
pub fn quadrant_cos(tv: f32, xr: f32, n: i32, #[comptime] cfg: MathConfig) -> f32 {
    let m = n & 3i32;
    let mut out = tv * sin(xr, cfg);
    if m == 0i32 {
        out = tv * cos(xr, cfg);
    } else if m == 2i32 {
        out = -tv * cos(xr, cfg);
    } else if m == 1i32 {
        out = -tv * sin(xr, cfg);
    }
    out
}

/// `t sin(xr + n pi/2)`, the tail every asymptotic `y` branch ends with.
#[cube]
pub fn quadrant_sin(tv: f32, xr: f32, n: i32, #[comptime] cfg: MathConfig) -> f32 {
    let m = n & 3i32;
    let mut out = -tv * cos(xr, cfg);
    if m == 0i32 {
        out = tv * sin(xr, cfg);
    } else if m == 2i32 {
        out = -tv * sin(xr, cfg);
    } else if m == 1i32 {
        out = tv * cos(xr, cfg);
    }
    out
}

/// `(beta0, alpha0)`: the order-0 asymptotic amplitude and phase, from
/// Harrison's expansion — `beta0 = 1 - 1/(16x^2) + 53/(512x^4)` and
/// `alpha0 = 1/(8x) - 25/(384x^3)`.
#[cube]
pub fn ab0(x: f32) -> (f64, f64) {
    let y = 1.0 / f64::cast_from(x);
    let y2 = y * y;
    (
        1.0 + y2 * (-0.0625 + f64::from_bits(0x3fba800000000000) * y2),
        y * (0.125 - f64::from_bits(0x3fb0aaaaa0000000) * y2),
    )
}

/// `(beta1, alpha1)`: the order-1 amplitude and phase —
/// `beta1 = 1 + 3/(16x^2) - 99/(512x^4)` and
/// `alpha1 = -3/(8x) + 21/(128x^3) - 1899/(5120x^5)`.
#[cube]
pub fn ab1(x: f32) -> (f64, f64) {
    let y = 1.0 / f64::cast_from(x);
    let y2 = y * y;
    (
        1.0 + y2 * (0.1875 - f64::from_bits(0x3fc8c00000000000) * y2),
        y * (-0.375 + y2 * (0.1640625 - f64::from_bits(0x3fd7bccccccccccd) * y2)),
    )
}

/// `j0(x)` beyond the near-a-zero tables.
#[cube]
pub fn j0_asympt(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    let (beta, alpha) = ab0(x);
    let (h, n) = reduce_aux(x, alpha);
    let tv = SQRT_2_OVER_PI / f32::sqrt(x) * f32::cast_from(beta);
    let mut out = quadrant_cos(tv, f32::cast_from(h), n, cfg);
    // Two arguments the expansion misses by more than the 9-ulp claim allows,
    // tabulated by glibc rather than fitted around.
    if u32::reinterpret(x) == 0x4ba332e9u32 {
        out = f32::from_bits(0x27250206);
    }
    if u32::reinterpret(x) == 0x4354d7efu32 {
        out = f32::from_bits(0x33747039);
    }
    out
}

/// `y0(x)` beyond the near-a-zero tables.
#[cube]
pub fn y0_asympt(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    let (beta, alpha) = ab0(x);
    let (h, n) = reduce_aux(x, alpha);
    let tv = SQRT_2_OVER_PI / f32::sqrt(x) * f32::cast_from(beta);
    let mut out = quadrant_sin(tv, f32::cast_from(h), n, cfg);
    if u32::reinterpret(x) == 0x435fd6cbu32 {
        out = f32::from_bits(0xb0fe657a);
    }
    if u32::reinterpret(x) == 0x48171521u32 {
        out = f32::from_bits(0x2bd244ba);
    }
    out
}

/// `j1(x)` beyond the near-a-zero tables.
///
/// Odd, so the sign rides on the scale rather than being applied afterwards.
#[cube]
pub fn j1_asympt(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = f32::abs(x0);
    let cst = select(x0 < 0.0, -SQRT_2_OVER_PI, SQRT_2_OVER_PI);
    let (beta, alpha) = ab1(x);
    let (h, n) = reduce_aux(x, alpha);
    let tv = cst / f32::sqrt(x) * f32::cast_from(beta);
    quadrant_cos(tv, f32::cast_from(h), n - 1i32, cfg)
}

/// `y1(x)` beyond the near-a-zero tables.
#[cube]
pub fn y1_asympt(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    let (beta, alpha) = ab1(x);
    let (h, n) = reduce_aux(x, alpha);
    let tv = SQRT_2_OVER_PI / f32::sqrt(x) * f32::cast_from(beta);
    quadrant_sin(tv, f32::cast_from(h), n - 1i32, cfg)
}

// ---------------------------------------------------------------------------
// The near-a-zero repairs
// ---------------------------------------------------------------------------

/// Which of the 64 tabulated zeros `x` is nearest, and whether it is one of
/// them at all.
///
/// Clamped at zero rather than left to whatever the rounding produced: glibc
/// reads the table with the raw index on the reasoning that the caller only
/// reaches here when `x` really is near a zero, and clamping makes that
/// reasoning unnecessary. The interval test the caller applies rejects the
/// argument anyway.
#[cube]
pub fn zero_index(x: f32, first: f32, #[comptime] cfg: MathConfig) -> (u32, bool) {
    let i = round((x - first) / PI_F32, cfg);
    (u32::cast_from(f32::max(i, 0.0)), i < SMALL_SIZE)
}

/// The shared cubic: `c0 + y (c1 + y (c2 + y c3))` about the tabulated zero.
///
/// Returns `(value, applies)` — `applies` is false when `x` falls outside the
/// row's own interval, where the rational fit's answer stands after all.
#[cube]
pub fn near_root(x: f32, family: u32, index: u32, extra: f32) -> (f32, bool) {
    let tab = besself_zeros_tab();
    let r = (family * ZN + index) * ZW;
    let lo = at(&tab, r);
    let hi = at(&tab, r + 2u32);
    let y = x - at(&tab, r + 1u32);
    let p6 = at(&tab, r + 6u32) + extra;
    let v = at(&tab, r + 3u32) + y * (at(&tab, r + 4u32) + y * (at(&tab, r + 5u32) + y * p6));
    (v, lo <= x && x <= hi)
}

// ---------------------------------------------------------------------------
// The rational fits for the asymptotic amplitude and phase
// ---------------------------------------------------------------------------

/// `1 + R(s)/S(s)` with `s = 1/x^2`.
#[cube]
pub fn rational_p(x: f32, pr: u32, ps: u32) -> f32 {
    let tab = besself_asympt_tab();
    let z = 1.0 / (x * x);
    let r = at(&tab, pr)
        + z * (at(&tab, pr + 1u32)
            + z * (at(&tab, pr + 2u32)
                + z * (at(&tab, pr + 3u32) + z * (at(&tab, pr + 4u32) + z * at(&tab, pr + 5u32)))));
    let s = 1.0
        + z * (at(&tab, ps)
            + z * (at(&tab, ps + 1u32)
                + z * (at(&tab, ps + 2u32) + z * (at(&tab, ps + 3u32) + z * at(&tab, ps + 4u32)))));
    1.0 + r / s
}

/// `R(s)/S(s)` with `s = 1/x^2`. One term longer in the denominator.
#[cube]
pub fn rational_q(x: f32, pr: u32, ps: u32) -> f32 {
    let tab = besself_asympt_tab();
    let z = 1.0 / (x * x);
    let r = at(&tab, pr)
        + z * (at(&tab, pr + 1u32)
            + z * (at(&tab, pr + 2u32)
                + z * (at(&tab, pr + 3u32) + z * (at(&tab, pr + 4u32) + z * at(&tab, pr + 5u32)))));
    let s = 1.0
        + z * (at(&tab, ps)
            + z * (at(&tab, ps + 1u32)
                + z * (at(&tab, ps + 2u32)
                    + z * (at(&tab, ps + 3u32)
                        + z * (at(&tab, ps + 4u32) + z * at(&tab, ps + 5u32))))));
    r / s
}

/// `pzero`: the amplitude factor of the order-0 asymptotic form.
#[cube]
pub fn pzero(x: f32) -> f32 {
    let i = interval(ix(x));
    rational_p(x, i * AW, (AN + i) * AW)
}

/// `qzero`: the phase factor of the order-0 asymptotic form.
#[cube]
pub fn qzero(x: f32) -> f32 {
    let i = interval(ix(x));
    (-0.125 + rational_q(x, (2u32 * AN + i) * AW, (3u32 * AN + i) * AW)) / x
}

/// `pone`: the amplitude factor of the order-1 asymptotic form.
#[cube]
pub fn pone(x: f32) -> f32 {
    let i = interval(ix(x));
    rational_p(x, (4u32 * AN + i) * AW, (5u32 * AN + i) * AW)
}

/// `qone`: the phase factor of the order-1 asymptotic form.
#[cube]
pub fn qone(x: f32) -> f32 {
    let i = interval(ix(x));
    (0.375 + rational_q(x, (6u32 * AN + i) * AW, (7u32 * AN + i) * AW)) / x
}

// ---------------------------------------------------------------------------
// The drivers
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind, order 0.
#[cube]
pub fn j0(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let xin = opaque32(x0);
    let i = ix(xin);
    let x = f32::abs(xin);
    let mut out = 1.0 / (xin * xin);
    if i < 0x7f80_0000u32 {
        if i >= 0x4000_0000u32 {
            if i >= 0x7f00_0000u32 {
                // `x >= 2^127`, where `x + x` would overflow.
                out = j0_asympt(x, cfg);
            } else {
                let s = sin(x, cfg);
                let c = cos(x, cfg);
                let mut ss = s - c;
                let mut cc = s + c;
                let z0 = -cos(x + x, cfg);
                if s * c < 0.0 {
                    cc = z0 / ss;
                } else {
                    ss = z0 / cc;
                }
                if i <= 0x5c00_0000u32 {
                    cc = pzero(x) * cc - qzero(x) * ss;
                }
                let z = (t::INVSQRTPI * cc) / f32::sqrt(x);
                out = z;
                // A small bracket means the subtraction cancelled, and the
                // result has lost more digits than the 9-ulp claim allows.
                if f32::abs(cc) <= f32::from_bits(0x3dabcd93) {
                    let (index, tabulated) = zero_index(x, f32::from_bits(0x4019e8a9), cfg);
                    out = j0_asympt(x, cfg);
                    if tabulated {
                        let (v, applies) = near_root(x, 0u32, index, 0.0);
                        out = select(applies, v, z);
                    }
                }
            }
        } else {
            let z = x * x;
            let tab = besself_small_tab();
            let rn = z
                * (at(&tab, J0_R)
                    + z * (at(&tab, J0_R + 1u32)
                        + z * (at(&tab, J0_R + 2u32) + z * at(&tab, J0_R + 3u32))));
            let sd = 1.0
                + z * (at(&tab, J0_S)
                    + z * (at(&tab, J0_S + 1u32)
                        + z * (at(&tab, J0_S + 2u32) + z * at(&tab, J0_S + 3u32))));
            let u = 0.5 * x;
            out = (1.0 + u) * (1.0 - u) + z * (rn / sd);
            if i < 0x3f80_0000u32 {
                out = 1.0 + z * (-0.25 + (rn / sd));
            }
            if i < 0x3900_0000u32 {
                out = 1.0 - 0.25 * x * x; // `|x| < 2^-13`
                if i < 0x3200_0000u32 {
                    out = 1.0; // `|x| < 2^-27`
                }
            }
        }
    }
    out
}

/// The Bessel function of the second kind, order 0.
#[cube]
pub fn y0(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let hx = u32::reinterpret(x);
    let i = ix(x);
    let mut out = 1.0 / (x + x * x);
    if i >= 0x7f80_0000u32 {
        if hx == 0xff80_0000u32 {
            out = nan32(); // `y0(-inf)`
        }
    } else if i == 0u32 {
        out = neg_inf32();
    } else if hx >> 31u32 != 0u32 {
        out = nan32();
    } else if i >= 0x4000_0000u32 || (i >= 0x3f53_40edu32 && i <= 0x3f77_b5e5u32) {
        // The second window is around `y0`'s first zero, where the rational
        // fit below is useless even though `|x| < 2`.
        if i >= 0x7f00_0000u32 {
            out = y0_asympt(x, cfg);
        } else {
            let s = sin(x, cfg);
            let c = cos(x, cfg);
            let mut ss = s - c;
            let mut cc = s + c;
            let z0 = -cos(x + x, cfg);
            if s * c < 0.0 {
                cc = z0 / ss;
            } else {
                ss = z0 / cc;
            }
            if i <= 0x5c00_0000u32 {
                ss = pzero(x) * ss + qzero(x) * cc;
            }
            let z = (t::INVSQRTPI * ss) / f32::sqrt(x);
            out = z;
            if f32::abs(ss) <= f32::from_bits(0x3ddf2c2d) {
                let (index, tabulated) = zero_index(x, f32::from_bits(0x3f64c166), cfg);
                out = y0_asympt(x, cfg);
                if tabulated {
                    // The first zero needs two extra degrees, which glibc
                    // hard-codes rather than widening the whole table for one
                    // row.
                    let tabz = besself_zeros_tab();
                    let y = x - at(&tabz, (ZN + index) * ZW + 1u32);
                    let extra = select(
                        index > 0u32,
                        0.0,
                        y * (f32::from_bits(0xbe691b24) + y * f32::from_bits(0x3e5cd51e)),
                    );
                    let (v, applies) = near_root(x, 1u32, index, extra);
                    out = select(applies, v, z);
                }
            }
        }
    } else if i <= 0x3980_0000u32 {
        let tab = besself_small_tab();
        out = at(&tab, Y0_U) + t::TPI * ln(x, cfg);
    } else {
        let z = x * x;
        let tab = besself_small_tab();
        let un = at(&tab, Y0_U)
            + z * (at(&tab, Y0_U + 1u32)
                + z * (at(&tab, Y0_U + 2u32)
                    + z * (at(&tab, Y0_U + 3u32)
                        + z * (at(&tab, Y0_U + 4u32)
                            + z * (at(&tab, Y0_U + 5u32) + z * at(&tab, Y0_U + 6u32))))));
        let vd = 1.0
            + z * (at(&tab, Y0_V)
                + z * (at(&tab, Y0_V + 1u32)
                    + z * (at(&tab, Y0_V + 2u32) + z * at(&tab, Y0_V + 3u32))));
        out = un / vd + t::TPI * (j0(x, cfg) * ln(x, cfg));
    }
    out
}

/// The Bessel function of the first kind, order 1.
#[cube]
pub fn j1(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let hx = u32::reinterpret(x);
    let i = ix(x);
    let y = f32::abs(x);
    let mut out = 1.0 / x;
    if i < 0x7f80_0000u32 {
        if i >= 0x4000_0000u32 {
            if i >= 0x7f00_0000u32 {
                out = j1_asympt(x, cfg);
            } else {
                let s = sin(y, cfg);
                let c = cos(y, cfg);
                let mut ss = -s - c;
                let mut cc = s - c;
                let z0 = cos(y + y, cfg);
                if s * c > 0.0 {
                    cc = z0 / ss;
                } else {
                    ss = z0 / cc;
                }
                if i <= 0x5c00_0000u32 {
                    cc = pone(y) * cc - qone(y) * ss;
                }
                let zz = (t::INVSQRTPI * cc) / f32::sqrt(y);
                let z = select(hx >> 31u32 != 0u32, -zz, zz);
                out = z;
                if f32::abs(cc) <= f32::from_bits(0x3ddbcb1c) {
                    let sign = select(x < 0.0, -1.0, 1.0);
                    let (index, tabulated) = zero_index(y, f32::from_bits(0x40753aac), cfg);
                    out = sign * j1_asympt(y, cfg);
                    if tabulated {
                        let tabz = besself_zeros_tab();
                        let yy = y - at(&tabz, (2u32 * ZN + index) * ZW + 1u32);
                        let extra =
                            select(index > 0u32, 0.0, yy * f32::from_bits(0xbb9f28d5));
                        let (v, applies) = near_root(y, 2u32, index, extra);
                        out = select(applies, sign * v, z);
                    }
                }
            }
        } else if i < 0x3200_0000u32 {
            out = 0.5 * x; // `|x| < 2^-27`
        } else {
            let z = x * x;
            let tab = besself_small_tab();
            let rn = z
                * (at(&tab, J1_R)
                    + z * (at(&tab, J1_R + 1u32)
                        + z * (at(&tab, J1_R + 2u32) + z * at(&tab, J1_R + 3u32))));
            let sd = 1.0
                + z * (at(&tab, J1_S)
                    + z * (at(&tab, J1_S + 1u32)
                        + z * (at(&tab, J1_S + 2u32)
                            + z * (at(&tab, J1_S + 3u32) + z * at(&tab, J1_S + 4u32)))));
            out = x * 0.5 + rn * x / sd;
        }
    }
    out
}

/// The Bessel function of the second kind, order 1.
#[cube]
pub fn y1(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let hx = u32::reinterpret(x);
    let i = ix(x);
    let mut out = 1.0 / (x + x * x);
    if i >= 0x7f80_0000u32 {
        if hx == 0xff80_0000u32 {
            out = nan32();
        }
    } else if i == 0u32 {
        out = neg_inf32();
    } else if hx >> 31u32 != 0u32 {
        out = nan32();
    } else if i >= 0x3fe0_dfbcu32 {
        // The crossover is below 2 here: `y1`'s first zero is at 2.197, and
        // the rational fit has already lost its digits by 1.757.
        if i >= 0x7f00_0000u32 {
            out = y1_asympt(x, cfg);
        } else {
            let s = sin(x, cfg);
            let c = cos(x, cfg);
            let mut ss = -s - c;
            let mut cc = s - c;
            let z0 = cos(x + x, cfg);
            if s * c > 0.0 {
                cc = z0 / ss;
            } else {
                ss = z0 / cc;
            }
            if i <= 0x5c00_0000u32 {
                ss = pone(x) * ss + qone(x) * cc;
            }
            let z = (t::INVSQRTPI * ss) / f32::sqrt(x);
            out = z;
            if f32::abs(ss) <= f32::from_bits(0x3e9f00a6) {
                let (index, tabulated) = zero_index(x, f32::from_bits(0x400c9df7), cfg);
                out = y1_asympt(x, cfg);
                if tabulated {
                    let tabz = besself_zeros_tab();
                    let y = x - at(&tabz, (3u32 * ZN + index) * ZW + 1u32);
                    let mut extra = 0.0 * y;
                    if index == 0u32 {
                        extra = y * (f32::from_bits(0xbb940218) + y * f32::from_bits(0x3c143a0c));
                    } else if index == 1u32 {
                        extra = y * f32::from_bits(0xbb7ff6b8);
                    }
                    let (v, applies) = near_root(x, 3u32, index, extra);
                    out = select(applies, v, z);
                }
            }
        }
    } else if i <= 0x3300_0000u32 {
        out = -t::TPI / x; // `x < 2^-25`
    } else {
        let z = x * x;
        let tab = besself_small_tab();
        let un = at(&tab, Y1_U)
            + z * (at(&tab, Y1_U + 1u32)
                + z * (at(&tab, Y1_U + 2u32)
                    + z * (at(&tab, Y1_U + 3u32) + z * at(&tab, Y1_U + 4u32))));
        let vd = 1.0
            + z * (at(&tab, Y1_V)
                + z * (at(&tab, Y1_V + 1u32)
                    + z * (at(&tab, Y1_V + 2u32)
                        + z * (at(&tab, Y1_V + 3u32) + z * at(&tab, Y1_V + 4u32)))));
        out = x * (un / vd) + t::TPI * (j1(x, cfg) * ln(x, cfg) - 1.0 / x);
    }
    out
}

// ---------------------------------------------------------------------------
// jnf and ynf
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind, order `n`.
///
/// The same three regimes as [`crate::double::bessel::jn`], with one
/// difference worth noting: the forward recurrence is evaluated in `double`
/// and rounded back to `float` each step. That is glibc's, and it is what
/// stops the recurrence underflowing to zero long before the answer does.
#[cube]
pub fn jn(nf: f32, x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let xin = opaque32(x0);
    let n0 = i32::cast_from(nf);
    let n = select(n0 < 0i32, -n0, n0);
    let xs = select(n0 < 0i32, -xin, xin);
    let negative = u32::reinterpret(xs) >> 31u32 != 0u32;
    let sgn = (n & 1i32) != 0i32 && negative;
    let x = f32::abs(xs);
    let i0 = ix(xs);

    let mut out = xin + xin;
    if !is_nan32(xin) {
        if n == 0i32 {
            out = j0(x, cfg);
        } else if n == 1i32 {
            out = j1(select(negative, -x, x), cfg);
        } else if i0 == 0u32 || i0 >= 0x7f80_0000u32 {
            out = select(sgn, -0.0, 0.0);
        } else {
            let mut b = 0.0 + x;
            if f32::cast_from(n) <= x {
                let mut a = j0(x, cfg);
                b = j1(x, cfg);
                let mut i = 1i32.runtime();
                while i < n {
                    let temp = b;
                    b = f32::cast_from(
                        f64::cast_from(b) * (f64::cast_from(i + i) / f64::cast_from(x))
                            - f64::cast_from(a),
                    );
                    a = temp;
                    i = i + 1i32;
                }
            } else if i0 < 0x3080_0000u32 {
                // `x < 2^-29`: `J(n,x) = (x/2)^n / n!`.
                b = 0.0 * x;
                if n <= 33i32 {
                    let temp = x * 0.5;
                    b = temp;
                    let mut a = 1.0 + 0.0 * x;
                    let mut i = 2i32.runtime();
                    while i <= n {
                        a = a * f32::cast_from(i);
                        b = b * temp;
                        i = i + 1i32;
                    }
                    b = b / a;
                }
            } else {
                let w = f32::cast_from(n + n) / x;
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
                    tt = 1.0 / (f32::cast_from(i) / x - tt);
                    i = i - 2i32;
                }
                let mut a = tt;
                b = 1.0 + 0.0 * x;

                let v = 2.0 / x;
                let tmp = f32::cast_from(n) * ln(f32::abs(v * f32::cast_from(n)), cfg);
                let mut di = f32::cast_from((n - 1i32) + (n - 1i32));
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
                        if b > 1e10 {
                            a = a / b;
                            tt = tt / b;
                            b = 1.0;
                        }
                        j = j + 1i32;
                    }
                }
                let zz = j0(x, cfg);
                let ww = j1(x, cfg);
                let mut res = tt * ww / a;
                if f32::abs(zz) >= f32::abs(ww) {
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
#[cube]
pub fn yn(nf: f32, x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let hx = u32::reinterpret(x);
    let i0 = ix(x);
    let n0 = i32::cast_from(nf);
    let n = select(n0 < 0i32, -n0, n0);
    let sign = select(n0 < 0i32, 1i32 - ((n & 1i32) << 1u32), 1i32);

    let mut out = 1.0 / (x + x * x);
    if i0 >= 0x7f80_0000u32 {
        if hx == 0xff80_0000u32 {
            out = nan32();
        }
    } else if n == 0i32 {
        out = y0(x, cfg);
    } else if i0 == 0u32 {
        out = select(sign == 1i32, neg_inf32(), inf32());
    } else if hx >> 31u32 != 0u32 {
        out = nan32();
    } else if n == 1i32 {
        out = f32::cast_from(sign) * y1(x, cfg);
    } else if i0 == 0x7f80_0000u32 {
        out = 0.0;
    } else {
        let mut a = y0(x, cfg);
        let mut b = y1(x, cfg);
        let mut ib = u32::reinterpret(b);
        let mut i = 1i32.runtime();
        while i < n && ib != 0xff80_0000u32 {
            let temp = b;
            // Note the operand order: `ynf` multiplies the ratio *by* `b`,
            // while `jnf` multiplies `b` by the ratio. In `double` the two
            // round the same way, but this is transcribed rather than tidied.
            b = f32::cast_from(
                (f64::cast_from(i + i) / f64::cast_from(x)) * f64::cast_from(b)
                    - f64::cast_from(a),
            );
            ib = u32::reinterpret(b);
            a = temp;
            i = i + 1i32;
        }
        out = select(sign > 0i32, b, -b);
    }
    out
}
