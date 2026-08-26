//! Reinterpreting floats as integers, inside a kernel.
//!
//! Every table-driven `libm` routine is really integer code with some
//! floating-point in the middle: the exponent field selects a branch, the
//! mantissa selects a table row, and the result is assembled by adding to an
//! exponent. These are the four primitives that makes that possible in a
//! `#[cube]` function, plus the `top12` helper the reference algorithms spell
//! out by hand.
//!
//! All four are exact bit reinterpretations, not conversions.

use cubecl::prelude::*;

/// The bit pattern of a `f64`.
#[cube]
pub fn f64_bits(x: f64) -> u64 {
    u64::reinterpret(x)
}

/// The `f64` with this bit pattern.
#[cube]
pub fn f64_of_bits(b: u64) -> f64 {
    f64::reinterpret(b)
}

/// The bit pattern of a `f32`.
#[cube]
pub fn f32_bits(x: f32) -> u32 {
    u32::reinterpret(x)
}

/// The `f32` with this bit pattern.
#[cube]
pub fn f32_of_bits(b: u32) -> f32 {
    f32::reinterpret(b)
}

/// The top 12 bits of a `f64`: sign and biased exponent.
///
/// The reference routines test the exponent field constantly — `top12(x) &
/// 0x7ff` is the magnitude's exponent, and comparing it against `top12` of a
/// constant is how they classify an input without a floating-point compare
/// (which would misbehave on NaN).
#[cube]
pub fn top12(x: f64) -> u32 {
    u32::cast_from(u64::reinterpret(x) >> 52u32)
}

/// The top 9 bits of a `f32`: sign and biased exponent.
#[cube]
pub fn top9(x: f32) -> u32 {
    u32::reinterpret(x) >> 23u32
}
