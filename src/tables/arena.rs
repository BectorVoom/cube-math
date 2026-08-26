//! Every table the kernels read, concatenated into one `u64` device buffer.
//!
//! # Why one buffer
//!
//! A GPU kernel cannot index a Rust `const` array — the data has to live in
//! device memory, bound to the kernel. Giving each function its own buffer
//! would mean a per-function binding, a per-function upload and a
//! per-function launch signature; the arena gives every kernel the same
//! signature (`tab: &Array<u64>`) and one upload for the whole crate. The
//! offsets below are compile-time constants, so `tab[OFF_EXP + i]` compiles to
//! the same addressing a dedicated buffer would have produced.
//!
//! Everything is stored as raw `u64` bit patterns — `f64` through
//! [`f64::to_bits`], `f32` zero-extended from [`f32::to_bits`]. One element
//! type for the whole arena, and it is what the kernels want anyway: the
//! bit-exact paths read most of these tables as bit patterns and do integer
//! arithmetic on them before ever making a float.
//!
//! # Adding a table
//!
//! Add an `OFF_`/`LEN_` pair below, chaining the offset onto the previous
//! entry, and push the same data in the same order in [`build`]. [`check`]
//! verifies the two agree and runs on every upload.

use super::double as d;
use super::single as s;

// --- double precision -------------------------------------------------------

/// `exp` / `exp2`: `2^(k/128)` as `[tail, scale]` pairs, already bit patterns.
pub const OFF_EXP: u32 = 0;
/// Length of [`OFF_EXP`].
pub const LEN_EXP: u32 = 256;

/// `ln`: `[invc, logc]` for 128 subintervals.
pub const OFF_LOG: u32 = OFF_EXP + LEN_EXP;
/// Length of [`OFF_LOG`].
pub const LEN_LOG: u32 = 256;

/// `log2`: `[invc, logc]` for 64 subintervals.
pub const OFF_LOG2: u32 = OFF_LOG + LEN_LOG;
/// Length of [`OFF_LOG2`].
pub const LEN_LOG2: u32 = 128;

/// `pow`: `[invc, logc, logctail]` for 128 subintervals.
pub const OFF_POW: u32 = OFF_LOG2 + LEN_LOG2;
/// Length of [`OFF_POW`].
pub const LEN_POW: u32 = 384;

/// `exp10`: the `10^x` polynomial and reduction constants that are indexed.
pub const OFF_EXP10: u32 = OFF_POW + LEN_POW;
/// Length of [`OFF_EXP10`].
pub const LEN_EXP10: u32 = 0;

// --- single precision -------------------------------------------------------

/// `expf` / `exp2f`: `2^(k/32)`, 32 entries, bit patterns of `double`s.
pub const OFF_EXPF: u32 = OFF_EXP10 + LEN_EXP10;
/// Length of [`OFF_EXPF`].
pub const LEN_EXPF: u32 = 32;

/// `logf`: `[invc, logc]` for 16 subintervals, as `double`s.
pub const OFF_LOGF: u32 = OFF_EXPF + LEN_EXPF;
/// Length of [`OFF_LOGF`].
pub const LEN_LOGF: u32 = 32;

/// `log2f`: `[invc, logc]` for 16 subintervals, as `double`s.
pub const OFF_LOG2F: u32 = OFF_LOGF + LEN_LOGF;
/// Length of [`OFF_LOG2F`].
pub const LEN_LOG2F: u32 = 32;

/// Total length of the arena, in `u64` slots.
pub const ARENA_LEN: usize = (OFF_LOG2F + LEN_LOG2F) as usize;

/// Push `f64`s as bit patterns.
fn f64s(v: &mut Vec<u64>, xs: &[f64]) {
    v.extend(xs.iter().map(|x| x.to_bits()));
}

/// Build the arena. Called once per client; see [`crate::Tables`].
pub fn build() -> Vec<u64> {
    let mut v: Vec<u64> = Vec::with_capacity(ARENA_LEN);
    v.extend_from_slice(&d::exp::TAB);
    f64s(&mut v, &d::log::TAB);
    f64s(&mut v, &d::log2::TAB);
    f64s(&mut v, &d::pow::TAB);
    v.extend_from_slice(&s::exp::TAB);
    f64s(&mut v, &s::log::TAB);
    f64s(&mut v, &s::log2::TAB);
    debug_assert_eq!(v.len(), ARENA_LEN);
    v
}

/// Assert that [`build`] agrees with the declared offsets.
///
/// Cheap enough to run on every upload — a handful of comparisons against a
/// table already in cache — and a mismatch here would otherwise show up as
/// silently wrong numbers rather than as a crash.
pub fn check(v: &[u64]) {
    assert_eq!(v.len(), ARENA_LEN, "arena length disagrees with the offsets");
    let at = |off: u32| v[off as usize];
    assert_eq!(at(OFF_EXP), d::exp::TAB[0]);
    assert_eq!(at(OFF_LOG), d::log::TAB[0].to_bits());
    assert_eq!(at(OFF_LOG2), d::log2::TAB[0].to_bits());
    assert_eq!(at(OFF_POW), d::pow::TAB[0].to_bits());
    assert_eq!(at(OFF_EXPF), s::exp::TAB[0]);
    assert_eq!(at(OFF_LOGF), s::log::TAB[0].to_bits());
    assert_eq!(at(OFF_LOG2F), s::log2::TAB[0].to_bits());
}

#[cfg(test)]
mod tests {
    #[test]
    fn arena_layout_is_consistent() {
        super::check(&super::build());
    }
}
