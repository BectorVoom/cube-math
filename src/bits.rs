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

use crate::tables::consts::{f32_const, f64_const};

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

/// Force `x` into a runtime variable.
///
/// A no-op at runtime — the backend folds the copy away — and load-bearing at
/// compile time. The C++ backends compile a reinterpretation to
/// `reinterpret_cast<T const&>(x)`, which needs an lvalue; a constant is not
/// one, so `ln(2.0)` with a literal argument fails to compile where `ln(x)`
/// with a buffer read succeeds. Since every one of these functions starts by
/// reinterpreting its argument, every public entry point launders its
/// arguments through this first, and calling them with constants works.
///
/// `.runtime()` is not enough: the optimiser folds that straight back into a
/// constant. A cell survives it.
#[cube]
pub fn opaque64(x: f64) -> f64 {
    RuntimeCell::<f64>::new(x).read()
}

/// [`opaque64()`] in single precision.
#[cube]
pub fn opaque32(x: f32) -> f32 {
    RuntimeCell::<f32>::new(x).read()
}

/// True when `x` is NaN.
///
/// Read off the bits, not written as `x != x`. The IEEE definition is the
/// clearer spelling, but it depends on `!=` being an *unordered* compare, and
/// not every backend lowers it that way — CubeCL's CPU runtime emits an
/// ordered compare, which answers `false` for a NaN and quietly turns every
/// NaN test into a no-op. A magnitude above the infinity pattern is a NaN on
/// every conforming device, with no comparison involved at all.
#[cube]
pub fn is_nan64(x: f64) -> bool {
    (u64::reinterpret(x) & 0x7fff_ffff_ffff_ffffu64) > 0x7ff0_0000_0000_0000u64
}

/// True when `x` is NaN, single precision. See [`is_nan64()`].
#[cube]
pub fn is_nan32(x: f32) -> bool {
    (u32::reinterpret(x) & 0x7fff_ffffu32) > 0x7f80_0000u32
}

/// True unless `x` is an infinity or a NaN.
#[cube]
pub fn is_finite64(x: f64) -> bool {
    (u64::reinterpret(x) & 0x7fff_ffff_ffff_ffffu64) < 0x7ff0_0000_0000_0000u64
}

/// True unless `x` is an infinity or a NaN, single precision.
#[cube]
pub fn is_finite32(x: f32) -> bool {
    (u32::reinterpret(x) & 0x7fff_ffffu32) < 0x7f80_0000u32
}

/// Positive infinity, built from its bit pattern.
///
/// Two things are going on in one line. Not `f64::INFINITY`, because A Rust constant reaches the backend as a literal, and
/// WGSL has no spelling for an infinite one — `f64(inf)` is not valid source,
/// a Rust constant reaches the backend as a literal and WGSL has no spelling
/// for an infinite one — `f64(inf)` is not valid source, so a kernel
/// mentioning `f64::INFINITY` fails to compile, and `wgpu` reports that by
/// leaving the output buffer untouched rather than by returning an error.
///
/// And `.runtime()`, because the C++ backend emits a reinterpretation as
/// `reinterpret_cast<double const&>(x)`, which needs an lvalue: handed a
/// literal it produces `reinterpret_cast` from an rvalue, and `hipcc` rejects
/// it. Forcing the bit pattern into a variable first costs nothing — the
/// compiler folds it straight back — and compiles everywhere.
#[cube]
pub fn inf64() -> f64 {
    f64_const(0x7ff0_0000_0000_0000u64)
}

/// Negative infinity. See [`inf64()`].
#[cube]
pub fn neg_inf64() -> f64 {
    f64_const(0xfff0_0000_0000_0000u64)
}

/// A quiet NaN. See [`inf64()`].
#[cube]
pub fn nan64() -> f64 {
    f64_const(0x7ff8_0000_0000_0000u64)
}

/// Positive infinity, single precision. See [`inf64()`].
#[cube]
pub fn inf32() -> f32 {
    f32_const(0x7f80_0000u32)
}

/// Negative infinity, single precision. See [`inf64()`].
#[cube]
pub fn neg_inf32() -> f32 {
    f32_const(0xff80_0000u32)
}

/// A quiet NaN, single precision. See [`inf64()`].
#[cube]
pub fn nan32() -> f32 {
    f32_const(0x7fc0_0000u32)
}
