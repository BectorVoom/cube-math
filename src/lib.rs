//! A CubeCL `libm` for `f64` and `f32` — one kernel source, CPU and GPU.
//!
//! This is a port of [`rmath`](https://github.com/BectorVoom/rmath), a SIMD
//! `libm` whose defining property is that its default policy is not "about one
//! ulp" but *bit-identical to the platform's scalar `libm`*. The port keeps
//! that property and changes what runs it: instead of a CPU vector register,
//! the parallelism is a CubeCL launch, so the same functions run on Vulkan,
//! CUDA, HIP, Metal, WebGPU and the CPU runtime.
//!
//! # What changed in the port, and why it is simpler
//!
//! `rmath` computes a whole vector on the main path and then *repairs* the
//! lanes that needed a different branch, because the lanes of a CPU vector
//! register cannot diverge. A GPU is SIMT: lanes are threads, and a branch is a
//! branch. So the kernels here are written in the shape of `rmath`'s scalar
//! reference module — the platform algorithm, branches and all — and the
//! parallelism comes from the launch geometry.
//!
//! The result is that the whole `patch_lanes` / `map_lanes` apparatus
//! disappears, and with it `rmath`'s "delegating" category: the functions whose
//! bit-exact path had to run one lane at a time on a CPU — `log1p` and `hypot`
//! among those ported so far — are ordinary parallel kernels here.
//!
//! # What is ported
//!
//! Everything IEEE-754 pins down exactly, in **both** precisions: [`Floor`],
//! [`Ceil`], [`Trunc`], [`Round`], [`Rint`], [`Sqrt`], [`Abs`], [`Ilogb`],
//! [`CopySign`], [`Fdim`], [`Fmax`], [`Fmin`], [`Ldexp`], [`Scalbn`],
//! [`Fmod`], [`Remainder`], [`NextAfter`], [`Frexp`], [`Modf`] and [`Remquo`].
//!
//! In double precision, the exponentials [`Exp`], [`Exp2`], [`Exp10`] and
//! [`Expm1`]; the logarithms [`Ln`], [`Log2`], [`Log10`] and [`Log1p`]; and
//! [`Pow`], [`Cbrt`] and [`Hypot`].
//!
//! Not yet ported: the trigonometric, inverse-trigonometric, hyperbolic, error,
//! gamma and Bessel families, and the single-precision transcendentals. The
//! README says what adding one involves.
//!
//! [`Floor`]: crate::function::Floor
//! [`Ceil`]: crate::function::Ceil
//! [`Trunc`]: crate::function::Trunc
//! [`Round`]: crate::function::Round
//! [`Rint`]: crate::function::Rint
//! [`Sqrt`]: crate::function::Sqrt
//! [`Abs`]: crate::function::Abs
//! [`Ilogb`]: crate::function::Ilogb
//! [`CopySign`]: crate::function::CopySign
//! [`Fdim`]: crate::function::Fdim
//! [`Fmax`]: crate::function::Fmax
//! [`Fmin`]: crate::function::Fmin
//! [`Ldexp`]: crate::function::Ldexp
//! [`Scalbn`]: crate::function::Scalbn
//! [`Fmod`]: crate::function::Fmod
//! [`Remainder`]: crate::function::Remainder
//! [`NextAfter`]: crate::function::NextAfter
//! [`Frexp`]: crate::function::Frexp
//! [`Modf`]: crate::function::Modf
//! [`Remquo`]: crate::function::Remquo
//! [`Exp`]: crate::function::Exp
//! [`Exp2`]: crate::function::Exp2
//! [`Exp10`]: crate::function::Exp10
//! [`Expm1`]: crate::function::Expm1
//! [`Ln`]: crate::function::Ln
//! [`Log2`]: crate::function::Log2
//! [`Log10`]: crate::function::Log10
//! [`Log1p`]: crate::function::Log1p
//! [`Pow`]: crate::function::Pow
//! [`Cbrt`]: crate::function::Cbrt
//! [`Hypot`]: crate::function::Hypot
//!
//! # The two questions, unchanged
//!
//! [`Accuracy`] and [`Domain`] mean exactly what they mean in `rmath`, and are
//! [`Policy`]'s two fields. They are *comptime* values: a kernel is expanded
//! once per policy, so the generated shader contains only the path you asked
//! for.
//!
//! # Bit-exactness on a device
//!
//! Every IEEE-754 operation rounds identically on every conforming device, so
//! replaying the reference schedule on a GPU gives the same bits as replaying
//! it on a CPU — provided the backend does not rewrite the arithmetic. Four
//! things can, none of them is guaranteed by CubeCL, and every one of them
//! fails on some backend in wide use. [`probe`] measures them on the live
//! device and [`Ctx`] finishes with a functional canary, so the answer is a
//! measurement rather than an assumption. See [`cube::fma`] for the one that
//! is recoverable, and the README for what the probes found on real hardware.

#![forbid(unsafe_op_in_unsafe_fn)]
// Four lints that this crate's subject matter contradicts, rather than four
// pieces of sloppiness:
//
// * `eq_op` — `(x - x) / (x - x)` is how glibc spells "raise invalid and
//   return this machine's own NaN". Substituting a NaN constant gives the
//   *positive* quiet NaN where a real division gives the negative one, and the
//   sign of a NaN is part of a bit-exactness contract. `x != x` likewise.
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
#![allow(
    clippy::eq_op,
    clippy::excessive_precision,
    clippy::int_plus_one,
    clippy::assign_op_pattern,
    clippy::manual_range_contains,
    clippy::collapsible_if
)]

pub mod config;
pub mod cube;
pub mod function;
pub mod host;
pub mod policy;
pub mod probe;
pub mod tables;

pub use config::Config;
pub use cube::fma::FmaKind;
pub use host::{Ctx, MathFn};
pub use policy::{Accuracy, Domain, Policy};
pub use probe::Fidelity;

/// Everything you need to call a function.
pub mod prelude {
    pub use crate::function::*;
    pub use crate::host::{Ctx, MathFn};
    pub use crate::policy::{Accuracy, Domain, Policy};
    pub use crate::probe::Fidelity;
}
