//! Fused multiply-add, and what to do when the backend does not have one.
//!
//! # Why this file exists
//!
//! Every bit-exact kernel here transcribes a schedule that glibc compiled with
//! `vfmadd`, so reproducing it needs `a * b + c` computed with **one**
//! rounding. Two roundings is a different number, and once one polynomial term
//! is off by an ulp the whole claim is gone.
//!
//! CubeCL exposes [`cubecl::prelude::fma`], and on the CUDA, HIP and WGSL
//! backends it lowers to the target's real `fma` — one rounding, as required.
//! The SPIR-V backend does not, and that is the backend `wgpu` uses on Vulkan,
//! which is the only route to `f64` on a non-NVIDIA GPU. As of `cubecl-spirv`
//! 0.10 (and 0.11-pre, which restates it as a `#[cube] fn fma(a,b,c) { a*b+c }`)
//! it lowers `Fma` to a separate `OpFMul` and `OpFAdd` — two roundings. That
//! is defensible on its own terms: GLSL.std.450's `Fma` is only *required* to
//! be fused when it carries the `NoContraction` decoration, which cubecl does
//! not emit. But it means "does this device have a fused multiply-add" is a
//! runtime property here, not a compile-time one.
//!
//! [`crate::probe::Fidelity`] measures it on the live device, with a case
//! whose fused and unfused answers differ, and the answer becomes the
//! [`FmaKind`] comptime parameter of every kernel:
//!
//! * [`FmaKind::Hardware`] — use the intrinsic. Costs nothing.
//! * [`FmaKind::Software`] — use [`fma_f64`] / [`fma_f32`] below, which are
//!   correctly rounded. Around 25 operations instead of one, and the only way
//!   to stay bit-exact on such a device.
//!
//! Nothing forces the software path on you: [`crate::policy::Accuracy::Fast`]
//! makes no exactness claim, so `Fast` kernels always take the intrinsic.

use cubecl::prelude::*;

use crate::cube::bits::is_finite64 as is_finite;

/// Which multiply-add a kernel should use.
///
/// A comptime parameter, so a kernel built for [`FmaKind::Hardware`] contains
/// the intrinsic and nothing else — there is no runtime test anywhere in the
/// generated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FmaKind {
    /// The backend's `fma` is a true single-rounding fused multiply-add.
    #[default]
    Hardware,
    /// The backend's `fma` rounds twice; emulate the fused one exactly.
    Software,
}

impl FmaKind {
    /// A short stable tag, used to name the generated kernel.
    pub const fn tag(self) -> &'static str {
        match self {
            FmaKind::Hardware => "hwfma",
            FmaKind::Software => "swfma",
        }
    }
}

/// `a * b + c` for `f64`, with one rounding.
#[cube]
pub fn fma64(a: f64, b: f64, c: f64, #[comptime] kind: FmaKind) -> f64 {
    if comptime!(matches!(kind, FmaKind::Hardware)) {
        fma(a, b, c)
    } else {
        fma_f64(a, b, c)
    }
}

/// `a * b + c` for `f32`, with one rounding.
#[cube]
pub fn fma32(a: f32, b: f32, c: f32, #[comptime] kind: FmaKind) -> f32 {
    if comptime!(matches!(kind, FmaKind::Hardware)) {
        fma(a, b, c)
    } else {
        fma_f32(a, b, c)
    }
}

// ---------------------------------------------------------------------------
// Exact building blocks.
// ---------------------------------------------------------------------------

/// Dekker's split: `x = hi + lo` exactly, with 26 significant bits each.
///
/// Requires `|x| < 2^996` so the scaling cannot overflow. Every caller here
/// has already scaled its operand into `[1, 2)`.
#[cube]
pub fn split(x: f64) -> (f64, f64) {
    let c = x * 134217729.0; // 2^27 + 1
    let hi = c - (c - x);
    (hi, x - hi)
}

/// `(p, e)` with `p = a * b` rounded and `a * b = p + e` exactly.
///
/// Requires `a` and `b` in `[1, 2)` in magnitude, which is what makes the
/// split safe and `p` normal — so `e` is representable.
#[cube]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = (((ah * bh - p) + ah * bl) + al * bh) + al * bl;
    (p, e)
}

/// `(s, e)` with `s = a + b` rounded and `a + b = s + e` exactly.
///
/// Knuth's two-sum: no ordering assumption on the operands, exact whenever the
/// sum does not overflow.
#[cube]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// The exact value `h + l` rounded **to odd** — to the neighbour whose
/// significand's last bit is 1, or to `h` itself when the pair is exact.
///
/// This is the trick that makes the emulation correct rather than merely
/// accurate. Rounding an intermediate to nearest and then rounding again to
/// nearest can land on the wrong side of a tie; rounding the intermediate to
/// *odd* provably cannot, because an odd significand is never a tie point for
/// the next rounding. (Boldo and Melquiond, *Emulation of FMA and
/// correctly-rounded sums with round to odd*, 2008.)
///
/// `(h, l)` must come from [`two_sum`], so `l` is `h`'s exact residual.
#[cube]
pub fn round_odd(h: f64, l: f64) -> f64 {
    let mut out = h;
    let bits = u64::reinterpret(h);
    if l != 0.0 && bits & 1u64 == 0u64 {
        // `h` is even and the true value is strictly between `h` and its
        // neighbour in the direction of `l`; that neighbour is the odd one.
        let up = l > 0.0;
        let positive = bits >> 63u64 == 0u64;
        // Away from zero when the direction agrees with the sign, towards it
        // otherwise. Adding to the magnitude field does both, because IEEE
        // orders same-sign floats the way their bit patterns order.
        if up == positive {
            out = f64::reinterpret(bits + 1u64);
        } else {
            out = f64::reinterpret(bits - 1u64);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The emulation.
// ---------------------------------------------------------------------------

/// The correctly-rounded `a * b + c`, computed without a hardware fused
/// multiply-add.
///
/// # How
///
/// `a` and `b` are first scaled to `[1, 2)` in magnitude, which costs nothing
/// (they are exponent edits) and buys the two things the exact algorithm needs:
/// Dekker's split cannot overflow, and the product is a normal number, so its
/// rounding error is representable. `c` is scaled by the matching amount.
///
/// Then, writing `k` for the total scale removed:
///
/// ```text
/// (uh, ul) = two_product(a', b')     a' b'      = uh + ul   exactly
/// (th, tl) = two_sum(c', uh)         c' + uh    = th + tl   exactly
/// (vh, vl) = two_sum(ul, tl)         ul + tl    = vh + vl   exactly
/// ```
///
/// so `a' b' + c' = th + vh + vl` exactly, as three doubles. Collapsing that
/// to one double is where a naive emulation goes wrong, and where
/// [`round_odd`] comes in: rounding `vh + vl` to odd and then `th + that` to
/// odd leaves a value that the final scale-back rounds correctly, ties
/// included — and the scale-back is the *only* rounding, so a subnormal result
/// is correct too instead of being rounded twice.
///
/// # Range
///
/// Exact on every input. The scaling handles subnormal and huge operands; the
/// two regimes where `c'` cannot be represented at all (one term utterly
/// dominating the other) are decided directly, because there the answer is
/// determined by the dominant term alone.
#[cube]
pub fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    let mut out = a * b + c;
    // When either factor is zero or not finite, `a * b + c` is already the
    // IEEE answer: the exact product is then a special value that propagates
    // the same way through both forms, or an exact zero, and neither leaves
    // the second rounding anything to discard.
    if is_finite(a) && is_finite(b) && a != 0.0 && b != 0.0 {
        if !is_finite(c) {
            // A finite product plus an infinity is that infinity — but `a * b`
            // may itself overflow to an infinity of the opposite sign, and
            // `inf + -inf` is NaN. So take `c` directly. (A NaN `c` falls out
            // of this too: `c` is the answer either way.)
            out = c;
        }
    }
    if is_finite(a) && is_finite(b) && is_finite(c) && a != 0.0 && b != 0.0 {
        let (am, ka) = to_unit(a);
        let (bm, kb) = to_unit(b);
        let k = ka + kb;

        // `c` scaled into the product's frame. `cexp` is where it would land.
        let cexp = exponent_of(c);
        if cexp - k > 600i32 {
            // `c` outranks the product by more than 2^598. The product is far
            // below half an ulp of `c`, so the sum rounds to `c`.
            out = c;
        } else {
            // Below -600 the product outranks `c` by more than 2^598 and `c`
            // would scale away to nothing; a signed minimum subnormal keeps
            // the round-to-odd direction, which is the only thing that
            // survives at that distance.
            let tiny = select(c == 0.0, c, copysign_f64(f64::from_bits(1), c));
            let cs = select(cexp - k < -600i32, tiny, scalbn(c, -k));

            let (uh, ul) = two_product(am, bm);
            let (th, tl) = two_sum(cs, uh);
            let (vh, vl) = two_sum(ul, tl);
            // `a' b' + c' = th + vh + vl` exactly. Rounding `vh + vl` to odd
            // and then `th + that` to nearest gives the correctly rounded sum
            // of the three — that is the round-to-odd theorem.
            let z = round_odd(vh, vl);
            let (rh, rl) = two_sum(th, z);

            // `rh` is the answer in the scaled frame, and scaling back by a
            // power of two is exact for a normal result — so there, `rh` is
            // the whole story.
            //
            // A *subnormal* result is not. `scalbn` would have to round, and
            // `rh` has already been rounded once, which is one rounding too
            // many. Round-to-odd does not rescue this one either: it needs the
            // intermediate to carry at least two more bits than the target,
            // and a result at the top of the subnormal range has only 52 bits
            // to `rh`'s 53. So the subnormal case is rounded directly onto the
            // target grid instead, with `rl` breaking the ties that are not
            // really ties.
            if exponent_of(rh) + k < -1022i32 {
                out = round_to_subnormal_grid(rh, rl, k);
            } else {
                out = scalbn(rh, k);
            }
        }
    }
    out
}

/// The correctly-rounded `a * b + c` for `f32`.
///
/// Far simpler than [`fma_f64`], because `f64` is wide enough to hold the
/// whole exact product: two 24-bit significands make 48, and `f64` carries 53.
/// So `prod` below is exact, `sum` is `a*b + c` with a single `f64` rounding,
/// and `err` is that rounding's exact residual. Narrowing `sum` to `f32` is
/// then a double rounding, which differs from the single one only when `sum`
/// sits exactly on an `f32` tie while `err` is nonzero — and [`round_odd`]
/// fixes exactly that, by moving `sum` off the tie in the direction the
/// discarded residual points.
#[cube]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let pa = f64::cast_from(a);
    let pb = f64::cast_from(b);
    let pc = f64::cast_from(c);
    let prod = pa * pb; // exact
    let (sum, err) = two_sum(prod, pc);
    f32::cast_from(round_odd(sum, err))
}

/// Round `(rh + rl) * 2^k` onto the subnormal grid, correctly.
///
/// The grid is `2^-1074`, so the trick is to measure `rh` *in grid units*:
/// `w = rh * 2^(1074 + k)` is exact (the result being subnormal bounds `|w|`
/// below `2^52`), and the problem becomes rounding a `f64` to a nearby
/// integer, which is exact arithmetic all the way down.
///
/// `rl` matters in exactly one place. When `w` lands on a half-integer the
/// grid rounding is a tie, and ties-to-even would pick a neighbour without
/// knowing that the true value is not actually on the midpoint — `rl` says
/// which side it is on, and a nonzero `rl` therefore settles the tie in its own
/// direction rather than by parity.
#[cube]
pub fn round_to_subnormal_grid(rh: f64, rl: f64, k: i32) -> f64 {
    // `rh` in units of the target grid.
    let w = scalbn(rh, 1074i32 + k);
    let f = f64::floor(w);
    let d = w - f; // in [0, 1), exact
    let up = f + 1.0;
    // Ties to even, except where `rl` overrules — see above.
    let tie_even = select(is_odd_integer(f), up, f);
    let tie = select(rl > 0.0, up, select(rl < 0.0, f, tie_even));
    let q = select(d > 0.5, up, select(d < 0.5, f, tie));
    // A result that underflows all the way to zero still carries the sign of
    // the exact value, which the grid arithmetic has thrown away — `floor` of
    // a tiny negative is `-1`, and rounding it back up lands on `+0`.
    copysign_f64(scalbn(q, -1074i32), select(rh != 0.0, rh, rl))
}

/// True when `f` is an odd integer. `f` must be an integral `f64` with
/// `|f| <= 2^52`, which is what [`round_to_subnormal_grid`] hands it.
#[cube]
pub fn is_odd_integer(f: f64) -> bool {
    f - 2.0 * f64::floor(f * 0.5) != 0.0
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

/// `|x|` with the sign of `y`.
#[cube]
pub fn copysign_f64(x: f64, y: f64) -> f64 {
    let m = 0x8000_0000_0000_0000u64;
    f64::reinterpret((u64::reinterpret(x) & !m) | (u64::reinterpret(y) & m))
}

/// The exponent `e` such that `|x|` is in `[2^e, 2^(e+1))`, with subnormals
/// resolved. Zero reports a very negative exponent, which is what the callers
/// want: a zero `c` is dominated by any nonzero product.
#[cube]
pub fn exponent_of(x: f64) -> i32 {
    let bits = u64::reinterpret(x) & 0x7fff_ffff_ffff_ffffu64;
    let e = i32::cast_from(bits >> 52u64);
    // Subnormals carry no exponent of their own, so renormalise by scaling up
    // a known power of two and reading the exponent that produces. Computed
    // unconditionally: it is one multiply, and a branch-free kernel is worth
    // more than a saved multiply on hardware where a divergent warp runs both
    // sides anyway.
    let up = u64::reinterpret(f64::reinterpret(bits) * 18014398509481984.0); // 2^54
    let sub = i32::cast_from(up >> 52u64) - 1077i32;
    select(bits == 0u64, -5000i32, select(e == 0i32, sub, e - 1023i32))
}

/// Split `x` into `(m, k)` with `x = m * 2^k` and `|m|` in `[1, 2)`.
///
/// Exact — it is an exponent edit — and it is what lets [`two_product`] use
/// Dekker's split without an overflow guard.
#[cube]
pub fn to_unit(x: f64) -> (f64, i32) {
    let k = exponent_of(x);
    let bits = u64::reinterpret(x);
    let sign = bits & 0x8000_0000_0000_0000u64;
    // Rebuild with a zero exponent. Subnormals have to be normalised first,
    // and scaling by 2^54 does that without disturbing the significand.
    let mag0 = bits & 0x7fff_ffff_ffff_ffffu64;
    let magn =
        u64::reinterpret(f64::reinterpret(mag0) * 18014398509481984.0) & 0x7fff_ffff_ffff_ffffu64;
    let mag = select(mag0 >> 52u64 == 0u64, magn, mag0);
    let m = f64::reinterpret(sign | ((mag & 0x000f_ffff_ffff_ffffu64) | (1023u64 << 52u64)));
    (m, k)
}

/// `x * 2^n`, correctly rounded, for any `n`.
///
/// Three multiplications by powers of two rather than one: a single factor
/// `2^n` is not representable for large `|n|`, and splitting the scale keeps
/// every intermediate in range. Each step is exact unless the result is
/// subnormal, where the last one rounds — once.
#[cube]
pub fn scalbn(x: f64, n: i32) -> f64 {
    // Step in chunks of 1022, the largest exponent a normal power of two has.
    // Three steps span the whole `f64` range and then some, and unrolled they
    // cost three multiplies with no loop and no divergence.
    let s1 = clamp(n, -1022i32, 1022i32);
    let s2 = clamp(n - s1, -1022i32, 1022i32);
    let s3 = clamp(n - s1 - s2, -1022i32, 1022i32);
    ((x * pow2(s1)) * pow2(s2)) * pow2(s3)
}

/// `2^n` for `n` in `[-1022, 1023]`, built straight into the exponent field.
#[cube]
pub fn pow2(n: i32) -> f64 {
    f64::reinterpret(u64::cast_from(n + 1023i32) << 52u64)
}
