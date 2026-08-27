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

/// `atan` / `atan2`: `[x0, t1, c2, c3, c4, c5, c6]` for 241 subintervals,
/// indexed by `round(256 w) - 16` where `w` is the argument or its reciprocal.
#[cube]
pub fn atan_cij_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(d::atan_data::CIJ.to_vec()))
}

/// `asin` / `acos`: the band table, repacked to a uniform stride.
///
/// Upstream's rows are `4 + degree` wide with `degree` running 5..9, reached
/// through six different index formulas; [`crate::tables::double::asincos_packed`]
/// re-derives all six as the single expression `floor(256 |x|) - 32` and pads
/// every row to [`crate::tables::double::asincos_packed::STRIDE`] slots. Only
/// the real rows are emitted here — a kernel branches, so a lane outside the
/// band never reaches the load, and the padding that a vector gather needed to
/// stay in bounds is dead weight in a kernel.
#[cube]
pub fn asncs_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(
        d::asincos_packed::PACKED.0[..d::asincos_packed::REAL_ROWS * d::asincos_packed::STRIDE]
            .to_vec()
    ))
}

/// `asin` / `acos`: `1/sqrt(z)`'s seed, indexed by the top 7 mantissa bits.
#[cube]
pub fn inroot_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::asincos_data::INROOT)))
}

/// `asin` / `acos`: the matching power-of-two scale for [`inroot_tab()`].
#[cube]
pub fn powtwo_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::asincos_data::POWTWO)))
}

/// `sin` / `cos` / `sincos`: `sincostab`'s `[sn, ssn, cs, ccs]` rows, indexed
/// by the low bits of `BIG + |x|`.
#[cube]
pub fn sincos_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(d::trig::TAB.to_vec()))
}

/// `tan`: `utan.tbl`'s `[x, f, g]` rows, indexed by `256 |x| - 15.5`.
#[cube]
pub fn tan_xfg_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(d::trig::XFG.to_vec()))
}

/// `__branred`: `2/pi` in base `2^24`, as 75 exactly-representable digits.
#[cube]
pub fn toverp_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::trig::TOVERP)))
}

/// `erf`: the near-zero polynomial's coefficients.
#[cube]
pub fn erf_c0_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::erf::C0)))
}

/// `erf`: the fast path's 94 bands of 13 coefficients, row-major.
#[cube]
pub fn erf_c_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erf::C)))
}

/// `erf`: the accurate path's 47 bands of 27 coefficients, row-major.
#[cube]
pub fn erf_c2_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erf::C2)))
}

/// `erf`: the accurate path's near-zero polynomial.
#[cube]
pub fn erf_p_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::erf::P)))
}

/// `erf`: the arguments even the accurate path cannot resolve, as
/// `[z, h, l]` triples.
#[cube]
pub fn erf_exc_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erf::EXCEPTIONS)))
}

/// `erf`: the same, for `|z| < 1/8`. Sorted, so the kernel bisects it.
#[cube]
pub fn erf_exc_tiny_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erf::EXCEPTIONS_TINY)))
}

/// `erfc`: the fast path's six bands of 13 coefficients.
#[cube]
pub fn erfc_t_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::T)))
}

/// `erfc`: the asymptotic path's polynomial.
#[cube]
pub fn erfc_e2_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::erfc::E2)))
}

/// `erfc`: the accurate path's ten bands of 30 coefficients.
#[cube]
pub fn erfc_tacc_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::TACC)))
}

/// `erfc`: the fast path's hard cases, as `[z, h, l]` triples.
#[cube]
pub fn erfc_exc_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::EXCEPTIONS)))
}

/// `erfc`: the accurate path's hard cases.
#[cube]
pub fn erfc_exc_acc_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::EXCEPTIONS_ACCURATE)))
}

/// `erfc`: the second accurate path's hard cases.
#[cube]
pub fn erfc_exc_acc2_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::EXCEPTIONS_ACCURATE_2)))
}

/// `erfc`: `e^-x` reduction table, high 6 bits.
#[cube]
pub fn erfc_t1_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::T1)))
}

/// `erfc`: `e^-x` reduction table, low 6 bits.
#[cube]
pub fn erfc_t2_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits2(&d::erfc::T2)))
}

/// `erfc`: the `e^-x` core polynomial.
#[cube]
pub fn erfc_q1_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!(bits(&d::erfc::Q1)))
}

/// The Bessel family's small-argument rational fits, as one flat table.
///
/// Nine arrays of different lengths, concatenated: `j0`'s numerator and
/// denominator, `y0`'s, `j1`'s, `y1`'s. See [`crate::double::bessel`] for the
/// offsets, which are named constants there rather than magic numbers here.
#[cube]
pub fn bessel_small_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!({
        let mut v: Vec<u64> = Vec::new();
        v.extend(bits(&d::bessel::J0_R));
        v.extend(bits(&d::bessel::J0_S));
        v.extend(bits(&d::bessel::Y0_U));
        v.extend(bits(&d::bessel::Y0_V));
        v.extend(bits(&d::bessel::J1_R));
        v.extend(bits(&d::bessel::J1_S));
        v.extend(bits(&d::bessel::Y1_U));
        v.extend(bits(&d::bessel::Y1_V));
        v
    }))
}

/// The Bessel family's asymptotic amplitude and phase fits, as one flat table.
///
/// Eight tables of four intervals each, every interval padded out to six slots
/// so that a row is `(table * 4 + interval) * 6 + slot`. The three tables that
/// only use five slots leave the sixth unread; a uniform stride is what turns
/// four `if` ladders into one index.
#[cube]
pub fn bessel_asympt_tab() -> Array<u64> {
    Array::<u64>::from_data(comptime!({
        let mut v: Vec<u64> = Vec::new();
        for row in d::bessel::P0R.iter() {
            v.extend(bits(row));
        }
        for row in d::bessel::P0S.iter() {
            v.extend(bits(row));
            v.push(0);
        }
        for row in d::bessel::Q0R.iter() {
            v.extend(bits(row));
        }
        for row in d::bessel::Q0S.iter() {
            v.extend(bits(row));
        }
        for row in d::bessel::P1R.iter() {
            v.extend(bits(row));
        }
        for row in d::bessel::P1S.iter() {
            v.extend(bits(row));
            v.push(0);
        }
        for row in d::bessel::Q1R.iter() {
            v.extend(bits(row));
        }
        for row in d::bessel::Q1S.iter() {
            v.extend(bits(row));
        }
        v
    }))
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

/// The bit patterns of a table of fixed-width rows, flattened row-major.
///
/// A kernel indexes a constant array, not a slice of slices, so the row shape
/// becomes arithmetic on the index: row `i` slot `j` is `i * N + j`.
fn bits2<const N: usize>(xs: &[[f64; N]]) -> Vec<u64> {
    xs.iter().flatten().map(|x| x.to_bits()).collect()
}
