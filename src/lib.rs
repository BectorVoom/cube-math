//! A CubeCL `libm` for `f64` and `f32` — bit-exact to the platform `libm`
//! where you ask for it, and callable from inside your own kernels.
//!
//! This is a port of [`rmath`](https://github.com/BectorVoom/rmath), a SIMD
//! `libm` whose defining property is that its default policy is not "about one
//! ulp" but *bit-identical to the platform's scalar `libm`*. The port keeps
//! that property and changes what runs it: instead of a CPU vector register,
//! the parallelism is a CubeCL launch.
//!
//! # The device functions are the product
//!
//! [`double`] and [`single`] hold `#[cube]` functions. Call them from your own
//! kernel — that is what the crate is for, and it is faster than launching a
//! pass of its own over memory:
//!
//! ```ignore
//! use cube_math::{double as m, MathConfig};
//!
//! #[cube]
//! fn my_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] cfg: MathConfig) {
//!     out[ABSOLUTE_POS] = m::exp::exp(x[ABSOLUTE_POS], cfg);
//! }
//! ```
//!
//! Nothing has to be bound or uploaded for that: the tables are constant
//! arrays compiled into the kernel, so a math function is an ordinary function
//! and not a resource. See [`tables::consts`].
//!
//! [`launch`] is the other half — free functions in the shape the rest of the
//! CubeCL ecosystem uses, for when a whole elementwise pass *is* what you
//! want:
//!
//! ```no_run
//! # use cube_math::prelude::*;
//! # use cubecl::prelude::*;
//! # use cubecl::std::tensor::TensorHandle;
//! # fn go<R: Runtime>(client: &ComputeClient<R>, x: TensorHandle<R>) -> Result<(), MathError> {
//! let out = TensorHandle::<R>::empty(client, x.shape().to_vec(), cube_math::launch::F64);
//! cube_math::launch::unary(
//!     client,
//!     Unary::Exp,
//!     x.binding(),
//!     out.binding(),
//!     cube_math::launch::F64,
//!     MathConfig::exact_for(fidelity(client).f64),
//! )
//! # }
//! ```
//!
//! # What changed in the port, and why it got simpler
//!
//! `rmath` computes a whole vector on the main path and then *repairs* the
//! lanes that needed a different branch, because the lanes of a CPU vector
//! register cannot diverge. A GPU is SIMT: lanes are threads, and a branch is a
//! branch. So the kernels here are written in the shape of `rmath`'s scalar
//! reference module — the platform algorithm, branches and all — and the
//! parallelism comes from the launch geometry.
//!
//! The whole `patch_lanes` / `map_lanes` apparatus disappears with it, and so
//! does `rmath`'s "delegating" category: the functions whose bit-exact path had
//! to run one lane at a time on a CPU are ordinary parallel kernels here.
//!
//! # Where a narrow version of it came back
//!
//! Six functions — [`double::exp`], [`double::exp2`], [`double::exp10`],
//! [`double::ln`], and [`double::logx`]'s `log2` and `log10` — also carry a
//! `_vec` entry point that evaluates `Vector<f64, N>` in one call, and those
//! *are* `rmath`'s shape: the main path on the whole vector, the elements
//! belonging to another branch repaired afterwards one at a time.
//!
//! It came back because the reason it went away does not cover every caller.
//! The launch geometry is the right parallelism for a pass over an array; it is
//! not available to a kernel that already holds N points in a vector for
//! reasons of its own — a grid collocation, an unrolled stencil — and whose
//! alternative is N extracts, N scalar calls and N inserts. Each `_vec`
//! function is bit-identical to its scalar twin on every element, so nothing is
//! traded for it; `tests/vector.rs` is the proof and the README has the
//! measurements. On the CPU runtime the vector form runs about three times as
//! fast at width 8; on a GPU it is close to flat, which is what SIMT should
//! predict.
//!
//! # The two questions, unchanged
//!
//! [`Accuracy`] and [`Domain`] mean what they mean in `rmath`, and are
//! [`Policy`]'s two fields. They are *comptime* values, so a kernel is expanded
//! once per policy and contains only the path you asked for. Two policies are
//! two kernels with two [`cubecl::prelude::KernelId`]s, which is what keeps
//! them out of each other's compilation cache.
//!
//! # What is ported
//!
//! Everything, in **both** precisions.
//!
//! Everything IEEE-754 pins down exactly: `floor`, `ceil`, `trunc`, `round`,
//! `rint`, `sqrt`, `abs`, `ilogb`, `copysign`, `fdim`, `fmax`, `fmin`,
//! `ldexp`, `scalbn`, `fmod`, `remainder`, `nextafter`, `frexp`, `modf` and
//! `remquo`. The exponentials `exp`, `exp2`, `exp10`, `expm1`; the logarithms
//! `ln`, `log2`, `log10`, `log1p`; `pow`, `cbrt`, `hypot`; the trigonometric
//! family `sin`, `cos`, `tan`, `sincos` and its inverse `asin`, `acos`,
//! `atan`, `atan2`; the hyperbolics `sinh`, `cosh`, `tanh` and their inverses
//! `asinh`, `acosh`, `atanh`; `erf` and `erfc`; `lgamma`, `lgamma_r` and
//! `tgamma`; and the Bessel functions `j0`, `j1`, `y0`, `y1`, `jn`, `yn`.
//!
//! Three things the port had to supply that `rmath` leaves to the platform,
//! because on a CPU there is a platform to leave them to and a kernel has
//! none: [`double::branred`], the Payne-Hanek reduction the trigonometric
//! family needs past `105414350`; the accurate half of [`double::asinh`] and
//! [`double::acosh`]; and schedule ports of `sinf`, `cosf`, `powf`, `atan2f`
//! and the single-precision Bessel family, which `rmath` calls the platform
//! for one lane at a time.
//!
//! [`single::wide`] is the one place `BitExact` is not an unconditional claim,
//! and that module says so in as many words.
//!
//! The README says what adding a function involves.
//!
//! # Bit-exactness on a device
//!
//! Every IEEE-754 operation rounds identically on every conforming device, so
//! replaying a reference schedule on a GPU gives the same bits as replaying it
//! on a CPU — provided the backend does not rewrite the arithmetic. Five things
//! can, none of them is guaranteed by CubeCL, and every one of them fails on
//! some backend in wide use. [`probe::fidelity`] measures them on the live
//! device, and finishes with a functional canary: a real kernel evaluated on
//! inputs whose correctly-rounded answers are mathematical constants.
//!
//! See [`fma`] for the one failure that is recoverable, and the README for
//! what the probes found on real hardware.

#![forbid(unsafe_op_in_unsafe_fn)]
// Four lints that this crate's subject matter contradicts, rather than four
// pieces of sloppiness:
//
// * `eq_op` — `(x - x) / (x - x)` is how glibc spells "raise invalid and
//   return this machine's own NaN". Substituting a NaN constant gives the
//   *positive* quiet NaN where a real division gives the negative one, and the
//   sign of a NaN is part of a bit-exactness contract.
// * `excessive_precision` — the constants here are exact bit patterns, quoted
//   at the precision that makes a product exact. Rounding one to its shortest
//   round-tripping form destroys the property it was chosen for.
// * `int_plus_one` — the reference algorithms classify inputs with wrapping
//   unsigned arithmetic (`top - 1 >= 0x7ff - 1` is "not a positive normal").
//   The "simpler" form clippy suggests is a different predicate on the
//   wrapping case, which is the case that matters.
// * `assign_op_pattern`, `manual_range_contains`, `collapsible_if` — `#[cube]`
//   expands the body it is given, and the compound and range forms do not
//   survive that expansion.
// * `neg_cmp_op_on_partial_ord` — `!(a < limit)` is not `a >= limit` here, and
//   the difference is the whole point: a NaN answers `false` to both, so the
//   negated form routes it to the reference schedule and the positive form
//   would leave it on an approximation that has nothing to say about it.
#![allow(
    clippy::eq_op,
    clippy::excessive_precision,
    clippy::int_plus_one,
    clippy::assign_op_pattern,
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord
)]
// A `#[cube]` local that several branches assign has to be *initialised* from
// a runtime value first: the macro turns a literal initialiser into a comptime
// constant, and assigning a runtime value to it afterwards is a type error. So
// the kernels open with `let mut out = <some real expression>` and then
// overwrite it on every path, which the compiler reads as a dead store. It is
// not one; it is the only spelling the macro accepts.
#![allow(unused_assignments)]

pub mod bits;
pub mod config;
pub mod double;
pub mod error;
pub(crate) mod exact_impl;
pub mod fma;
pub mod launch;
pub mod policy;
pub mod probe;
pub mod single;
pub mod tables;

pub use config::MathConfig;
pub use error::MathError;
pub use fma::FmaKind;
pub use policy::{Accuracy, Domain, Policy};
pub use probe::{Fidelity, Precision, fidelity};

/// Everything the launch side needs.
pub mod prelude {
    pub use crate::config::MathConfig;
    pub use crate::error::MathError;
    pub use crate::fma::FmaKind;
    pub use crate::launch::{Binary, BinaryPair, Unary, UnaryPair};
    pub use crate::policy::{Accuracy, Domain, Policy};
    pub use crate::probe::{Fidelity, Precision, fidelity};
}
