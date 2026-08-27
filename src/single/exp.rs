//! `expf`, `exp2f` and `exp10f`.
//!
//! Ports of what the platform runs for `float`, which on this target is ARM's
//! optimized-routines — the same source glibc compiles into `__expf` and
//! friends.
//!
//! # Why these are `f64` inside
//!
//! Every one of them evaluates in *double* precision over a small table and
//! rounds once at the end. That is not an implementation detail to be
//! optimised away, it is the schedule: reproducing it is what makes these
//! bit-exact, and computing them in `f32` instead would give a different
//! function that happens to be close.
//!
//! It also means a device that cannot do `f64` cannot run these, which is why
//! [`crate::probe::fidelity`] reports the two precisions separately.
//!
//! # Both policies run the same code
//!
//! [`crate::Accuracy::Fast`] is accepted and has no effect. The whole routine
//! is a 32-entry table lookup and a degree-3 polynomial; there is nothing left
//! for an approximation to remove.

use cubecl::prelude::*;

use crate::bits::{inf32, opaque32};
use crate::config::MathConfig;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::expf_tab;
use crate::tables::single::exp as t;
use crate::tables::single::exp10 as x10;

/// `log(2^128)`, above which `expf` overflows.
const EXP_OFLOW: f32 = f32::from_bits(0x42b17217);

/// `-log(2^-150)`, below whose negation `expf` underflows to zero.
const EXP_UFLOW: f32 = f32::from_bits(0x42cff1b4);

/// The sign and exponent bits, as ARM's `top12`.
#[cube]
pub fn top12(x: f32) -> u32 {
    u32::reinterpret(x) >> 20u32
}

/// `2^(k/32)` with the exponent folded in by one integer add.
///
/// The pre-subtracted table entries are what let the exponent ride along in an
/// integer addition rather than a second multiply.
#[cube]
pub fn scale32(ki: u64) -> f64 {
    let tab = expf_tab();
    f64::reinterpret(tab[usize::cast_from(ki % 32u64)] + (ki << 47u64))
}

/// The shared main path of `expf`, in double precision.
#[cube]
pub fn exp_core(xd: f64, #[comptime] fk: FmaKind) -> f64 {
    // `x N / ln2 = k + r` with `|r| <= 1/2` and integer `k`.
    //
    // Both uses of `INVLN2_SCALED * xd` are fused, so the product is never
    // rounded to a `double` of its own. Computing it once into a variable and
    // reusing it is the obvious transcription of the C source and is wrong on
    // two inputs out of `2^32` — precisely the kind of difference only an
    // exhaustive check finds, and `rmath` found it.
    let kd_s = fma64(t::INVLN2_SCALED, xd, t::SHIFT, fk);
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - t::SHIFT;
    let r = fma64(t::INVLN2_SCALED, xd, -kd, fk);

    let s = scale32(ki);
    let z = fma64(r, t::CS0, t::CS1, fk);
    let r2 = r * r;
    let y0 = fma64(r, t::CS2, 1.0, fk);
    fma64(r2, z, y0, fk) * s
}

/// `e^x`.
#[cube]
pub fn exp(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let fk = comptime!(cfg.fma());
    let abstop = top12(x) & 0x7ffu32;

    let mut out = f32::cast_from(exp_core(f64::cast_from(x), fk));
    if abstop >= 0x42bu32 {
        // `|x| >= 88`, or a NaN. `0x42b` is `top12(88.0f32)`.
        if u32::reinterpret(x) == 0xff80_0000u32 {
            out = 0.0;
        } else if abstop >= 0x7f8u32 {
            out = x + x; // `+inf`, or a NaN propagating its payload
        } else if x > EXP_OFLOW {
            out = inf32(); // `x > log(2^128)`
        } else if x < -EXP_UFLOW {
            out = 0.0; // `x < log(2^-150)`
        }
        // Large but representable: the main path's answer stands.
    }
    out
}

/// The shared main path of `exp2f`, in double precision.
#[cube]
pub fn exp2_core(xd: f64, #[comptime] fk: FmaKind) -> f64 {
    let kd_s = xd + t::SHIFT_SCALED;
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - t::SHIFT_SCALED; // `k/N`, for integer `k`
    let r = xd - kd;

    let s = scale32(ki);
    let z = fma64(r, t::C0, t::C1, fk);
    let r2 = r * r;
    let y0 = fma64(r, t::C2, 1.0, fk);
    fma64(r2, z, y0, fk) * s
}

/// `2^x`.
#[cube]
pub fn exp2(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let fk = comptime!(cfg.fma());
    let abstop = top12(x) & 0x7ffu32;

    let mut out = f32::cast_from(exp2_core(f64::cast_from(x), fk));
    if abstop >= 0x430u32 {
        // `|x| >= 128`, or a NaN. `0x430` is `top12(128.0f32)`.
        if u32::reinterpret(x) == 0xff80_0000u32 {
            out = 0.0;
        } else if abstop >= 0x7f8u32 {
            out = x + x;
        } else if x > 0.0 {
            out = inf32();
        } else if x <= -150.0 {
            out = 0.0;
        }
    }
    out
}

/// The shared main path of `exp10f`, in double precision.
///
/// Unfused throughout, unlike [`exp_core()`]: glibc ships no fused variant of
/// `exp10f`, and the compiled routine is `mulsd`/`addsd` the whole way.
#[cube]
pub fn exp10_core(xd: f64) -> f64 {
    let z0 = x10::INVLN10N * xd;
    let kd_s = z0 + t::SHIFT;
    let ki = u64::reinterpret(kd_s);
    let kd = kd_s - t::SHIFT;
    let r = z0 - kd;

    let s = scale32(ki);
    let z = t::CS0 * r + t::CS1;
    let r2 = r * r;
    let y = t::CS2 * r + 1.0;
    (z * r2 + y) * s
}

/// `10^x`.
///
/// `exp2f` with one constant changed: the same 32-entry table, the same three
/// scaled coefficients, and a reduction scale of `N log2(10)` rather than `N`.
#[cube]
pub fn exp10(x0: f32, #[comptime] cfg: MathConfig) -> f32 {
    let x = opaque32(x0);
    let abstop = (u32::reinterpret(x) >> 19u32) & 0xfffu32;

    let mut out = f32::cast_from(exp10_core(f64::cast_from(x)));
    if abstop >= x10::BIG_TOP13 {
        // `|x| >= 38`, or a NaN.
        if u32::reinterpret(x) == 0xff80_0000u32 {
            out = 0.0;
        } else if abstop >= x10::INF_TOP13 {
            out = x + x;
        } else if x > x10::OFLOW_BOUND {
            out = inf32();
        } else if x < x10::UFLOW_BOUND {
            out = 0.0;
        }
    }
    comptime!(cfg);
    out
}
