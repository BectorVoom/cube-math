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
//! bit-exact path had to run one lane at a time on a CPU (`sin`, `cos`, `tan`,
//! `asin`, `atan2`, `log1p`, `hypot`, the inverse hyperbolics, and all of
//! single-precision Bessel) are ordinary parallel kernels here.
//!
//! # The two questions, unchanged
//!
//! [`Accuracy`] and [`Domain`] mean exactly what they mean in `rmath`, and are
//! [`Policy`]'s two fields. They are *comptime* values: a kernel is expanded
//! once per policy, so the generated shader contains only the path you asked
//! for.
//!
//! # Bit-exactness on a GPU
//!
//! Every IEEE-754 operation rounds identically on every conforming device, so
//! replaying the reference schedule on a GPU gives the same bits as replaying
//! it on a CPU — provided the backend does not rewrite the arithmetic.
//! [`probe`] measures the three things that could, on the live device, before
//! you rely on the claim. See [`cube::fma`] for the one that actually varies
//! in practice.

#![forbid(unsafe_op_in_unsafe_fn)]

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
