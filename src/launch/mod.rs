//! Launching the kernels over a tensor.
//!
//! The shape here follows the rest of the CubeCL ecosystem rather than
//! inventing one: free functions that take a [`ComputeClient`], operands as
//! [`TensorBinding`]s, the element type as a runtime [`StorageType`], and the
//! configuration as a value — the same signature `cubek`'s reduce, random and
//! matmul entry points have. There is no context object to build and no
//! handle to keep alive, because the kernels carry their tables with them (see
//! [`crate::tables::consts`]).
//!
//! ```no_run
//! # use cube_math::prelude::*;
//! # use cubecl::prelude::*;
//! # use cubecl::std::tensor::TensorHandle;
//! # fn go<R: Runtime>(client: &ComputeClient<R>, input: TensorHandle<R>) -> Result<(), MathError> {
//! let output = TensorHandle::<R>::empty(client, input.shape().to_vec(), cube_math::launch::F64);
//! cube_math::launch::unary(
//!     client,
//!     Unary::Exp,
//!     input.binding(),
//!     output.binding(),
//!     cube_math::launch::F64,
//!     MathConfig::exact_for(fidelity(client).f64),
//! )
//! # }
//! ```
//!
//! # If you are writing your own kernel
//!
//! Do not reach for this module. The functions in [`crate::double`] and
//! [`crate::single`] are `#[cube]` functions you can call directly from inside
//! your kernel, which is both faster — no extra pass over memory — and the
//! reason the crate is arranged this way:
//!
//! ```ignore
//! use cube_math::double as m;
//!
//! #[cube]
//! fn my_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] cfg: MathConfig) {
//!     out[ABSOLUTE_POS] = m::exp::exp(x[ABSOLUTE_POS], cfg);
//! }
//! ```

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;

mod binary;
mod unary;

pub use binary::{Binary, BinaryPair, binary, binary_pair};
pub use unary::{Unary, UnaryPair, unary, unary_pair};

/// The `f64` storage type, spelled once so that callers and the dispatch below
/// cannot disagree about it.
pub const F64: StorageType = StorageType::Scalar(ElemType::Float(FloatKind::F64));
/// The `f32` storage type. See [`F64`].
pub const F32: StorageType = StorageType::Scalar(ElemType::Float(FloatKind::F32));

/// One thread per element, in cubes of 256.
///
/// Elementwise transcendental work is ALU-bound rather than bandwidth-bound,
/// so there is nothing to gain from giving each thread several elements, and a
/// flat mapping keeps the bounds check to one comparison.
pub(crate) fn geometry(n: usize) -> (CubeCount, CubeDim) {
    let dim = 256u32;
    let cubes = n.div_ceil(dim as usize) as u32;
    (CubeCount::Static(cubes.max(1), 1, 1), CubeDim::new_1d(dim))
}

/// Check that two operands agree on length.
pub(crate) fn same_len(left: usize, right: usize) -> Result<(), crate::MathError> {
    if left == right {
        Ok(())
    } else {
        Err(crate::MathError::LengthMismatch { left, right })
    }
}
