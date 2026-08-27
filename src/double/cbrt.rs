//! `x^(1/3)`.
//!
//! **The reference here is Rust's, not the C library's**, and for this one
//! function they differ. Rust's `std` does not forward `cbrt` to the platform
//! `libm`; it uses its own port of the correctly-rounded CORE-MATH routine.
//! glibc's `cbrt` is a much cruder `frexp`-plus-one-Halley-step algorithm,
//! accurate to about an ulp, and the two disagree on roughly half of a random
//! sweep — by one ulp, but that is the whole point.
//!
//! Matching Rust is the right choice for a Rust crate: the premise of
//! [`crate::Accuracy::BitExact`] is that substituting this crate for the call
//! you were already making cannot change a result, and the call you were
//! already making is `f64::cbrt`. If you need a C program's `cbrt`, this is
//! not the function for it.
//!
//! Because CORE-MATH's routine is correctly rounded, `BitExact` here is a
//! claim about mathematics rather than about a platform: any correctly-rounded
//! `cbrt` anywhere agrees with it.
//!
//! Both policy axes are accepted and have no effect. The Newton iteration is
//! already the cheap algorithm; what would be dropped is the second refinement
//! step, and that is what makes the answer correctly rounded.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::tables::consts::f64_const;

/// `2^(i/3)` for `i` in `0..3`.
const ESCALE0: f64 = 1.0;
const ESCALE1: f64 = f64::from_bits(0x3ff428a2f98d728b);
const ESCALE2: f64 = f64::from_bits(0x3ff965fea53d6e3d);

/// Degree-3 seed for `z^(1/3)` on `[1, 2]`; maximum error below 9.2e-5.
const CB0: f64 = f64::from_bits(0x3fe1b0babccfef9c);
const CB1: f64 = f64::from_bits(0x3fe2c9a3e94d1da5);
const CB2: f64 = f64::from_bits(0xbfc4dc30b1a1ddba);
const CB3: f64 = f64::from_bits(0x3f97a8d3e4ec9b07);

/// `1/3`, to working precision.
const U0: f64 = f64::from_bits(0x3fd5555555555555);
/// `2/9`.
const U1: f64 = f64::from_bits(0x3fcc71c71c71c71c);

/// `2^-53`, the offset that centres the rounding-boundary test.
const OFF_NEAREST: f64 = f64::from_bits(0x3ca0000000000000);
const P_M52: f64 = f64::from_bits(0x3cb0000000000000);
const P_M75: f64 = f64::from_bits(0x3b40000000000000);
const P_M98: f64 = f64::from_bits(0x39d0000000000000);
const P_M60: f64 = f64::from_bits(0x3c30000000000000);

/// `+-2^-k` selectors, indexed by `(it << 1) | sign`.
#[cube]
pub fn rsc(idx: u32) -> f64 {
    let mag = select(idx >> 1u32 == 0u32, 1.0, select(idx >> 1u32 == 1u32, 0.5, 0.25));
    select(idx & 1u32 == 1u32, -mag, mag)
}

/// `2^(i/3)`, indexed.
#[cube]
pub fn escale(it: u32) -> f64 {
    select(it == 0u32, ESCALE0, select(it == 1u32, ESCALE1, ESCALE2))
}

/// `x^(1/3)`.
#[cube]
pub fn cbrt(x: f64, #[comptime] _cfg: MathConfig) -> f64 {
    let x = crate::bits::opaque64(x);
    let hx = u64::reinterpret(x);
    let sign = hx >> 63u64;
    let ix = hx & 0x7fff_ffff_ffff_ffffu64;
    let e0 = u32::cast_from((hx >> 52u64) & 0x7ffu64);

    let mut out = x + x;
    let mut done = e0 == 0x7ffu32 || ix == 0u64; // 0, inf, NaN

    if !done {
        // A subnormal has to have its leading bit shifted up into place.
        let sub = e0 == 0u32;
        let nz = u64::cast_from(u64::leading_zeros(ix) - 11u32);
        let mant = select(
            sub,
            (hx & 0x000f_ffff_ffff_ffffu64) << nz & 0x000f_ffff_ffff_ffffu64,
            hx & 0x000f_ffff_ffff_ffffu64,
        );
        let e = select(sub, e0 - (u32::cast_from(nz) - 1u32), e0) + 3072u32;

        let cvt1 = mant | (0x3ffu64 << 52u64);
        let et = e / 3u32;
        let it = e % 3u32;

        // `2^(3k+it) <= x < 2^(3k+it+1)`, with `zz` in `[1, 8)`.
        let zz = f64::reinterpret((cvt1 + (u64::cast_from(it) << 52u64)) | (sign << 63u64));
        let cvt2 = u64::reinterpret(escale(it)) | (sign << 63u64);
        let z = f64::reinterpret(cvt1);

        let r = 1.0 / z;
        let rr = r * rsc((it << 1u32) | u32::cast_from(sign));
        let z2 = z * z;
        let c0 = CB0 + z * CB1;
        let c2 = CB2 + z * CB3;
        let mut y = c0 + z2 * c2;
        let mut y2 = y * y;

        // Cubic Newton on `f(y) = 1 - z/y^3`.
        let mut h = y2 * (y * r) - 1.0;
        y -= (h * y) * (U0 - U1 * h);
        y *= f64::reinterpret(cvt2);

        // One more Newton step, this time carrying `y^2` and `y^3` in
        // double-double so the residual survives the cancellation in
        // `y3 - zz`.
        y2 = y * y;
        let mut y2l = fma(y, y, -y2);
        let mut y3 = y2 * y;
        let mut y3l = fma(y, y2, -y3) + y * y2l;
        h = ((y3 - zz) + y3l) * rr;
        let mut dy = h * (y * U0);
        let mut y1 = y - dy;
        dy = (y - y1) - dy;

        let mut ady = f64::abs(dy);
        let mut ady0 = f64::abs(ady - OFF_NEAREST);
        let mut ady1 = f64::abs(ady - (P_M52 + OFF_NEAREST));

        // Too close to a rounding boundary to call: refine once more.
        if ady0 < P_M75 || ady1 < P_M75 {
            y2 = y1 * y1;
            y2l = fma(y1, y1, -y2);
            y3 = y2 * y1;
            y3l = fma(y1, y2, -y3) + y1 * y2l;
            h = ((y3 - zz) + y3l) * rr;
            dy = h * (y1 * U0);
            y = y1 - dy;
            dy = (y1 - y) - dy;
            y1 = y;
            ady = f64::abs(dy);
            ady0 = f64::abs(ady - OFF_NEAREST);
            ady1 = f64::abs(ady - (P_M52 + OFF_NEAREST));

            // Still undecidable: two inputs are hard enough to be tabulated.
            if ady0 < P_M98 || ady1 < P_M98 {
                let azz = f64::abs(zz);
                if azz == f64_const(0x4009b78223aa307cu64) {
                    y1 = copysign(f64_const(0x3ff79d15d0e8d59cu64), zz);
                }
                if azz == f64_const(0x401a202bfc89ddffu64) {
                    y1 = copysign(f64_const(0x3ffde87aa837820fu64), zz);
                }
            }
        }

        let mut cvt3 =
            u64::reinterpret(y1) + (u64::cast_from(i32::cast_from(et) - 342i32 - 1023i32) << 52u64);
        let m0 = cvt3 << 30u64;
        let m1 = u64::reinterpret(i64::reinterpret(m0) >> 63i64);

        // The result sits near the middle of a rounding interval; snap it if
        // the remaining error says it should be.
        if (m0 ^ m1) <= (1u64 << 30u64) {
            let cvt4 = (u64::reinterpret(y1) + (164u64 << 15u64)) & 0xffff_ffff_ffff_0000u64;
            if f64::abs((f64::reinterpret(cvt4) - y1) - dy) < P_M60 || f64::abs(zz) == 1.0 {
                cvt3 = (cvt3 + (1u64 << 15u64)) & 0xffff_ffff_ffff_0000u64;
            }
        }
        out = f64::reinterpret(cvt3);
        done = true;
    }
    let _ = done;
    out
}

/// The magnitude of `x` with the sign of `y`.
///
/// See `crate::double::exact::copysign` for why this is `abs` and a negation
/// rather than bit manipulation.
#[cube]
pub fn copysign(x: f64, y: f64) -> f64 {
    let a = f64::abs(x);
    select(u64::reinterpret(y) & 0x8000_0000_0000_0000u64 != 0u64, -a, a)
}
