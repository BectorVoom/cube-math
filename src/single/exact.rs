//! The functions IEEE-754 pins down exactly, in single precision.
//!
//! Rounding to an integer, sign and exponent manipulation, `fmod`,
//! `remainder`, `sqrt`. What these have in common is that there is no
//! approximation anywhere in them: the mathematically correct result is always
//! representable, or — for `sqrt` — IEEE-754 requires it to be correctly
//! rounded. So there is no reference *algorithm* to reproduce. Any
//! implementation that is correct is automatically bit-identical to the
//! platform's, on every input, on every device, forever.
//!
//! That is a much stronger footing than the rest of the crate stands on, and
//! it is why:
//!
//! * both policy axes are genuine no-ops here — there is no cheaper
//!   approximation worth having and no special case for `FullRange` to repair,
//!   because the special cases are on the main path at no cost;
//! * these need no fused multiply-add, so they are exact even on a backend
//!   [`crate::probe`] rejects for everything else;
//! * bit-exactness here is not a claim about this platform's `libm`.
//!
//! The bodies live in the crate's `exact_impl` template, which
//! generates both precisions from one source.

crate::exact_impl::exact_for! {
    f32, u32,
    mant: 23u32,
    mant_i: 23i32,
    sign: 0x8000_0000u32,
    exp_mask: 0xffu32,
    exp_field: 0x7f80_0000u32,
    one: 1u32,
    bias: 127i32,
    max_biased_exp: 254u32,
    half_minus_ulp: 0.49999997f32,
    two_pow_mant: 8388608.0f32,
    min_positive: 1.1754943508222875e-38f32,
    inf: crate::bits::inf32,
    opaque: crate::bits::opaque32,
}
