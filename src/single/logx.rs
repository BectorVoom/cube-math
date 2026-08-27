//! `logf` and `log2f`.
//!
//! ARM's optimized-routines, as glibc compiles them: a 16-row table, a
//! degree-3 polynomial, and — like [`super::exp`] — double-precision
//! arithmetic throughout with one rounding at the end. See that module for why
//! the `f64` is the schedule rather than an implementation detail.
//!
//! Both policy axes are accepted and have no effect.

use cubecl::prelude::*;

use crate::bits::{neg_inf32, opaque32};
use crate::config::MathConfig;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::{log2f_tab, logf_tab};
use crate::tables::single::log as lt;
use crate::tables::single::log2 as l2t;

/// The table-centring offset, `bits(0x1.66p-1)`.
const LOG_OFF: u32 = 0x3f33_0000;
/// `2^23`, the subnormal rescale.
const P23: f32 = f32::from_bits(0x4b00_0000);

/// The main path of `logf`, taking already-normalised bits.
#[cube]
pub fn ln_core(ix: u32, #[comptime] fk: FmaKind) -> f64 {
    // `x = 2^k z`, with `z` in `[OFF, 2 OFF)` and exact.
    let tmp = ix - LOG_OFF;
    let i = usize::cast_from((tmp >> 19u32) % 16u32);
    let k = i32::reinterpret(tmp) >> 23i32;
    let iz = ix - (tmp & 0xff80_0000u32);

    let tab = logf_tab();
    let invc = f64::reinterpret(tab[2 * i]);
    let logc = f64::reinterpret(tab[2 * i + 1]);
    let z = f64::cast_from(f32::reinterpret(iz));

    // `log(x) = log1p(z/c - 1) + log(c) + k ln2`.
    let r = fma64(z, invc, -1.0, fk);
    let y0 = fma64(f64::cast_from(k), lt::LN2, logc, fk);

    let r2 = r * r;
    let y1 = fma64(r, lt::A1, lt::A2, fk);
    let y = fma64(r2, lt::A0, y1, fk);
    fma64(y, r2, y0 + r, fk)
}

/// The main path of `log2f`, taking already-normalised bits.
#[cube]
pub fn log2_core(ix: u32, #[comptime] fk: FmaKind) -> f64 {
    let tmp = ix - LOG_OFF;
    let i = usize::cast_from((tmp >> 19u32) % 16u32);
    let top = tmp & 0xff80_0000u32;
    let iz = ix - top;
    let k = i32::reinterpret(top) >> 23i32;

    let tab = log2f_tab();
    let invc = f64::reinterpret(tab[2 * i]);
    let logc = f64::reinterpret(tab[2 * i + 1]);
    let z = f64::cast_from(f32::reinterpret(iz));

    // `log2(x) = log2(z/c) + log2(c) + k`.
    let r = fma64(z, invc, -1.0, fk);
    let y0 = logc + f64::cast_from(k);

    let r2 = r * r;
    let y1 = fma64(r, l2t::A1, l2t::A2, fk);
    let p = fma64(r, l2t::A3, y0, fk);
    let y = fma64(r2, l2t::A0, y1, fk);
    fma64(y, r2, p, fk)
}

/// The shared entry: normalise, dispatch, and answer the special cases.
///
/// `base2` selects [`log2_core()`] over [`ln_core()`]. The two routines have
/// byte-for-byte the same special-case ladder, so it lives here once.
#[cube]
pub fn logx(x0: f32, #[comptime] base2: bool, #[comptime] fk: FmaKind) -> f32 {
    let x = opaque32(x0);
    let raw = u32::reinterpret(x);

    // The normalised bits the core takes. A subnormal is scaled into the
    // normal range and the shift taken back off the exponent field.
    let mut ix = raw;
    if raw - 0x0080_0000u32 >= 0x7f80_0000u32 - 0x0080_0000u32 {
        ix = u32::reinterpret(x * P23) - (23u32 << 23u32);
    }

    let mut out = f32::cast_from(ln_core(ix, fk));
    if comptime!(base2) {
        out = f32::cast_from(log2_core(ix, fk));
    }

    if raw - 0x0080_0000u32 >= 0x7f80_0000u32 - 0x0080_0000u32 {
        // Below `2^-126`, or an infinity, or a NaN — the subnormals having
        // already been handled above.
        if raw * 2u32 == 0u32 {
            out = neg_inf32();
        } else if raw == 0x7f80_0000u32 {
            out = x;
        } else if (raw & 0x8000_0000u32) != 0u32 || raw * 2u32 >= 0xff00_0000u32 {
            // Not a NaN constant: `0/0` yields this hardware's own default
            // NaN, whose sign bit is part of a bit-exactness contract. See
            // `crate::double::log1p`.
            out = (x - x) / (x - x);
        }
    }
    // `x == 1` must be `+0` even under a directed rounding mode.
    if raw == 0x3f80_0000u32 {
        out = 0.0;
    }
    out
}

/// `ln(x)`.
#[cube]
pub fn ln(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    logx(x, false, comptime!(cfg.fma()))
}

/// `log2(x)`.
#[cube]
pub fn log2(x: f32, #[comptime] cfg: MathConfig) -> f32 {
    logx(x, true, comptime!(cfg.fma()))
}
