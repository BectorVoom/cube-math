//! The tables, as constant arrays the kernels carry with them.
//!
//! # Why not a device buffer
//!
//! An earlier version of this crate uploaded every table into one `u64` buffer
//! and passed it to each kernel. That works, but it makes the device functions
//! unusable as a *library*: a caller who wants `exp(x)` in the middle of their
//! own kernel would have to thread a buffer they did not ask for through their
//! own launch signature, and keep it alive alongside their own data.
//!
//! `Array::from_data` takes a comptime `Vec` and emits a constant array into
//! the kernel instead. The table becomes part of the compiled code — read out
//! of constant memory, indexed dynamically, with nothing to bind and nothing
//! to upload. Measured on gfx1151, a dynamically-indexed 256-entry constant
//! table gathers at 5.4 Gelem/s, so nothing was paid for the convenience.
//!
//! # The other reason
//!
//! Constant arrays are also how a `f64` bit pattern reaches the C++ backends
//! at all. `cubecl-cpp` compiles a reinterpretation to
//! `reinterpret_cast<double const&>(x)`, which needs an lvalue; handed a
//! literal it emits a cast from an rvalue and `hipcc` rejects the kernel. A
//! constant-array element is an lvalue. That is what [`f64_const()`] is for, and
//! why the infinities in [`crate::bits`] go through it.

use cubecl::prelude::*;

use super::double as d;
use super::single as s;

/// `exp` / `exp2` / `exp10` / `pow`: `2^(k/128)` as `[tail, scale]` pairs,
/// already bit patterns.
#[cube]
pub fn exp_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(d::exp::TAB.to_vec()))
}

/// `ln`: `[1/c, log c]` for 128 subintervals.
#[cube]
pub fn log_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::log::TAB)))
}

/// `log2`: `[1/c, log2 c]` for 64 subintervals.
#[cube]
pub fn log2_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::log2::TAB)))
}

/// `pow`: `[1/c, log c, log-c tail]` for 128 subintervals.
#[cube]
pub fn pow_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::pow::TAB)))
}

/// `expf` / `exp2f`: `2^(k/32)`, as the bit patterns of `double`s.
#[cube]
pub fn expf_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(s::exp::TAB.to_vec()))
}

/// A `f64` constant that survives the trip to a C++ backend.
///
/// See the module documentation: a scalar constant reaches `cubecl-cpp` as an
/// rvalue, and a reinterpretation of an rvalue is not valid C++. Routing the
/// bit pattern through a one-element constant array makes it an lvalue. The
/// compiler folds the array away again, so this costs nothing at runtime — it
/// only changes what the generated source looks like.
#[cube]
pub fn f64_const(#[comptime] bits: u64) -> f64 {
    let one = Array::<u64>::from_data(comptime!(vec![bits]));
    f64::reinterpret(one[0])
}

/// [`f64_const()`] in single precision.
#[cube]
pub fn f32_const(#[comptime] bits: u32) -> f32 {
    let one = Array::<u32>::from_data(comptime!(vec![bits]));
    f32::reinterpret(one[0])
}

/// The bit patterns of a `f64` table, for [`Array::from_data`].
fn bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}
