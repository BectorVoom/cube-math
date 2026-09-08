//! `sinh`, `cosh`, `tanh` and `atanh`.
//!
//! The other two inverse hyperbolics are not here: `asinh` and `acosh` are
//! correctly-rounded CORE-MATH routines with a whole accurate tier of their
//! own, and they get modules to themselves — [`super::asinh`] and
//! [`super::acosh`].
//!
//! These came almost free, and that is the interesting thing about them: the
//! platform computes all four as compositions on `exp`, `expm1` and `log1p`,
//! exactly, with no fused-multiply-add subtleties of their own. Once those
//! three were ported the four followed, and their [`crate::Accuracy::BitExact`]
//! paths call this crate's own kernels rather than anything of the platform's.
//!
//! What each one actually reproduces:
//!
//! * `sinh`, `cosh`, `tanh` — glibc's `__ieee754_sinh`, `__ieee754_cosh` and
//!   `__tanh` (the fdlibm lineage), band for band, including the two-halves
//!   split near the overflow threshold that keeps neither half from
//!   overflowing on its own.
//! * `atanh` — `0.5 log1p(2x / (1 - x))`, which is **Rust's** `f64::atanh`
//!   rather than the C library's. The two disagree on roughly one input in
//!   ten, and this is the one `rmath` pins itself to, so it is the one that
//!   agreeing with `rmath` means agreeing with.
//!
//! [`crate::Accuracy::Fast`] takes the identities that stay accurate where the
//! obvious ones cancel — `sinh` as `t (t + 2) / (2 (t + 1))` in `t = expm1|x|`
//! rather than `(e^x - e^-x)/2`, `tanh` through `expm1(-2|x|)` — over this
//! crate's table-free exponentials. `cosh` is the exception: both its terms
//! are positive, so there is nothing to cancel and the naive form is the right
//! one.

use cubecl::prelude::*;

use crate::bits::{inf64, opaque64};
use crate::config::MathConfig;
use crate::double::exact::copysign;
use crate::double::{exp, expm1, log1p};
use crate::fma::FmaKind;

/// `2^-55`: below this `tanh` is the identity, and `cosh` is exactly
/// `1 + expm1|x|` with no correction term left to add.
///
/// Not fdlibm's `2^-28` for `tanh`. Between the two thresholds the platform
/// takes the general path, and the division there rounds the result one ulp
/// away from `x` — so the wider shortcut is not what runs.
const TINY: f64 = f64::from_bits(0x3c80000000000000);
/// `2^-54`, `atanh`'s own identity threshold.
const ATANH_TINY: f64 = f64::from_bits(0x3c90000000000000);
/// `ln(DBL_MAX)`, as the platform tests it — on the high word alone.
///
/// This is where the table-free forms stop making a claim, and it is set by
/// the *intermediate* rather than the result: past it `e^|x|` overflows on its
/// own, and `t + t/(t+1)` becomes `inf/inf`, while `sinh` and `cosh` are still
/// a hundred million times short of overflowing. The platform's answer is to
/// halve the argument and multiply the result back in two steps; the reference
/// schedule already carries that arm, so under [`crate::Domain::FullRange`] it
/// takes those inputs back over rather than this file growing a second copy.
pub const OVERFLOW: f64 = f64::from_bits(0x40862e4200000000);

// ---------------------------------------------------------------------------
// sinh / cosh / tanh
// ---------------------------------------------------------------------------

/// `cosh(x)`.
#[cube]
pub fn cosh(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let mut out = bit_exact_cosh(x, comptime!(cfg.fma()));
    if comptime!(!cfg.bit_exact()) {
        out = fast_cosh(x, comptime!(cfg.checked()), comptime!(cfg.fma()));
    }
    out
}

/// `cosh`'s reference schedule, over the whole domain.
#[cube]
pub fn bit_exact_cosh(x: f64, #[comptime] fk: FmaKind) -> f64 {
    let a = f64::abs(x);
    let mut out = x * x;
    {
        let ix = u32::cast_from(u64::reinterpret(x) >> 32u64) & 0x7fff_ffffu32;
        if ix >= 0x7ff0_0000u32 {
            // An infinity, or a NaN to quieten.
            out = x * x;
        } else if ix < 0x3fd6_2e43u32 {
            // `|x| < 0.5 ln2`: `1 + t^2 / (2 (1 + t))`, which does not cancel.
            let t = expm1::bit_exact(a);
            let w = 1.0 + t;
            out = 1.0 + (t * t) / (w + w);
            if ix < 0x3c80_0000u32 {
                out = w;
            }
        } else if ix < 0x4036_0000u32 {
            // `|x| < 22`.
            let t = exp::bit_exact(a, fk);
            out = 0.5 * t + 0.5 / t;
        } else if ix < 0x4086_2e42u32 {
            // Below `ln(DBL_MAX)`: the reciprocal term has underflowed away.
            out = 0.5 * exp::bit_exact(a, fk);
        } else if ix <= 0x4086_33ceu32 {
            // Up to the overflow threshold, in two halves so that neither one
            // overflows on its own.
            let w = exp::bit_exact(0.5 * a, fk);
            out = (0.5 * w) * w;
        } else {
            // `DBL_MAX * DBL_MAX` in the C source. Written as the infinity it
            // folds to, because a folded infinity is a literal the C++
            // backends cannot print.
            out = inf64();
        }
    }
    out
}

/// `cosh`'s table-free form. Both terms are positive, so there is nothing to
/// cancel and the naive form is the right one — unlike [`sinh()`]'s.
///
/// Measured error: below 3 ulp over `|x| < ln(DBL_MAX)`. See [`OVERFLOW`] for
/// what happens past that and why the cutoff is where it is.
#[cube]
pub fn fast_cosh(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let a = f64::abs(x);
    let t = exp::fast(a, checked, fk);
    let mut out = 0.5 * (t + 1.0 / t);
    if comptime!(checked) {
        if !(a < OVERFLOW) {
            out = bit_exact_cosh(x, fk);
        }
    }
    out
}

/// `sinh(x)`.
#[cube]
pub fn sinh(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let mut out = bit_exact_sinh(x, comptime!(cfg.fma()));
    if comptime!(!cfg.bit_exact()) {
        out = fast_sinh(x, comptime!(cfg.checked()), comptime!(cfg.fma()));
    }
    out
}

/// `sinh`'s reference schedule, over the whole domain.
#[cube]
pub fn bit_exact_sinh(x: f64, #[comptime] fk: FmaKind) -> f64 {
    let a = f64::abs(x);
    let mut out = x + x;
    {
        let jx = u32::cast_from(u64::reinterpret(x) >> 32u64);
        let ix = jx & 0x7fff_ffffu32;
        let h = select(jx >> 31u32 != 0u32, -0.5, 0.5);
        if ix >= 0x7ff0_0000u32 {
            out = x + x;
        } else if ix < 0x4036_0000u32 {
            // `|x| < 22`.
            let t = expm1::bit_exact(a);
            out = h * (t + t / (t + 1.0));
            if ix < 0x3ff0_0000u32 {
                out = h * (2.0 * t - t * t / (t + 1.0));
            }
            if ix < 0x3e30_0000u32 {
                out = x; // `|x| < 2^-28`
            }
        } else if ix < 0x4086_2e42u32 {
            out = h * exp::bit_exact(a, fk);
        } else if ix <= 0x4086_33ceu32 {
            let w = exp::bit_exact(0.5 * a, fk);
            out = (h * w) * w;
        } else {
            out = copysign(inf64(), x);
        }
    }
    out
}

/// `sinh`'s table-free form.
///
/// `t (t + 2) / (2 (t + 1))` with `t = expm1|x|`, rather than
/// `(e^x - e^-x) / 2`. The two agree mathematically; the difference is that
/// for small `x` the subtraction form cancels to nothing while this one
/// reduces to `t`, which is already the answer.
///
/// Measured error: below 3 ulp over `|x| < ln(DBL_MAX)`. See [`OVERFLOW`].
#[cube]
pub fn fast_sinh(x: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let a = f64::abs(x);
    let t = expm1::fast(a, checked, fk);
    let mut out = copysign(0.5 * (t + t / (t + 1.0)), x);
    if comptime!(checked) {
        if !(a < OVERFLOW) {
            out = bit_exact_sinh(x, fk);
        }
    }
    out
}

/// `tanh(x)`.
#[cube]
pub fn tanh(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let a = f64::abs(x);
    let mut out = x;
    if comptime!(cfg.bit_exact()) {
        let jx = u32::cast_from(u64::reinterpret(x) >> 32u64);
        let ix = jx & 0x7fff_ffffu32;
        if ix >= 0x7ff0_0000u32 {
            // `1/x + 1` at `+inf` is `1`, and at `-inf` `1/x - 1` is `-1`; a
            // NaN falls through either arm unchanged.
            out = 1.0 / x + 1.0;
            if jx >> 31u32 != 0u32 {
                out = 1.0 / x - 1.0;
            }
        } else {
            // Initialised from a runtime value because a `#[cube]` local that
            // several branches assign cannot start life as a literal.
            let mut z = a;
            if ix < 0x4036_0000u32 {
                if ix >= 0x3ff0_0000u32 {
                    let t = expm1::bit_exact(2.0 * a);
                    z = 1.0 - 2.0 / (t + 2.0);
                } else {
                    let t = expm1::bit_exact(-2.0 * a);
                    z = -t / (t + 2.0);
                }
            } else {
                // `|x| >= 22`: the platform's `one - tiny`, which *is* `1.0`.
                // The subtraction only raises inexact, and nothing here
                // observes an exception flag.
                z = 1.0;
            }
            out = copysign(z, x);
            if ix < 0x3c80_0000u32 {
                out = x;
            }
        }
    } else {
        out = copysign(
            fast_tanh(a, comptime!(cfg.checked()), comptime!(cfg.fma())),
            x,
        );
    }
    out
}

/// `tanh`'s table-free form: `-u / (u + 2)` with `u = expm1(-2|x|)`.
///
/// Saturates on its own — as `|x|` grows `u` tends to `-1` and the quotient to
/// `1` — so nothing has to be special-cased at the top of the range.
///
/// Measured error: below 4 ulp over the whole real line — `expm1`'s own 2,
/// doubled by the quotient `-u / (u + 2)` as `u` approaches `-2`.
#[cube]
pub fn fast_tanh(a: f64, #[comptime] checked: bool, #[comptime] fk: FmaKind) -> f64 {
    let u = expm1::fast(-(a + a), checked, fk);
    let mut out = -u / (u + 2.0);
    if a < TINY {
        out = a;
    }
    out
}

// ---------------------------------------------------------------------------
// atanh
// ---------------------------------------------------------------------------

/// `atanh(x)`.
///
/// `0.5 log1p(2x / (1 - x))` — Rust's `f64::atanh`, not the C library's. See
/// the module documentation for why that is the reference here.
#[cube]
pub fn atanh(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let mut out = x;
    if comptime!(cfg.bit_exact()) {
        out = 0.5 * log1p::bit_exact(2.0 * x / (1.0 - x));
        if f64::abs(x) < ATANH_TINY {
            out = x;
        }
    } else {
        // The *signed* identity, not `|x|` folded back with `copysign`. The
        // two are not the same function in floating point: at `x = -1 + ulp`
        // the signed form's argument is `-1` and the answer `-inf`, which is
        // what the reference gives, while the folded form's is enormous and
        // positive. Everything either form saves is smaller than that gap.
        out = 0.5
            * log1p::fast(
                2.0 * x / (1.0 - x),
                comptime!(cfg.checked()),
                comptime!(cfg.fma()),
            );
        if f64::abs(x) < ATANH_TINY {
            out = x;
        }
    }
    out
}
