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

/// How much arithmetic a launch should carry before a second CPU thread is
/// asked for.
///
/// A unit on the CPU runtime is an operating-system thread out of a worker
/// pool, and waking one costs on the order of a microsecond — a few thousand
/// cycles, which is tens of thousands of scalar operations. Below that much
/// work the thread does not pay for itself.
const WORK_PER_CPU_UNIT: usize = 32 * 1024;

/// What one element of a [`CpuShape::Threaded`] function costs, in scalar
/// operations. Only the ratio to [`WORK_PER_CPU_UNIT`] matters: together they
/// say that a few hundred elements are worth a second thread.
const THREADED_WORK_PER_ELEM: usize = 256;

/// How far past the core count a [`CpuShape::Threaded`] launch is allowed to
/// go.
///
/// Deliberately oversubscribed. Asking for twice the cores measured at or
/// above the core count itself for nearly every branchy function in the set,
/// and well above it for some — `erfc` 1153 against 438 Melem/s, `erf` 1068
/// against 452, `atan2` 847 against 487 — which says the cost being amortised
/// is the runtime's own dispatch and not the arithmetic.
const THREADED_CORE_FACTOR: usize = 2;

/// A ceiling on CPU units per cube, against a machine that reports an
/// implausible core count.
const CPU_CUBE_DIM_MAX: u32 = 64;

/// Units per cube for a [`CpuShape::Vectorised`] body.
///
/// Two, and not the core count, because the loop the CPU runtime wraps around
/// the kernel is what is doing the work here — see [`CpuShape`]. Two rather
/// than one because it measures faster than one everywhere it was tried
/// (`abs` 2681 against 2130 Melem/s, `exp` 1183 against 687, `ln` 906 against
/// 585), which is worth having and is not worth a theory.
const CPU_VECTORISED_UNITS: u32 = 2;

/// What a kernel body does to the loop the CPU runtime wraps around it, which
/// is the only thing that decides how many units to ask for there.
///
/// The CPU runtime dispatches one OS thread per unit in the cube dimension,
/// and each thread runs the kernel over the whole cube count. That loop is
/// ordinary compiled code, so LLVM vectorises it when the body is
/// straight-line — and splitting it across more threads breaks that up. The
/// two halves of this crate fall on opposite sides of that, by a wide margin
/// and consistently:
///
/// | function | few units | many units |
/// |---|---|---|
/// | `abs` | 2916 Me/s | 500 Me/s |
/// | `floor` | 2670 | 503 |
/// | `exp` | 1184 | 522 |
/// | `erfc` | 693 | 1171 |
/// | `tgamma` | 175 | 495 |
/// | `j0` | 76 | 221 |
/// | `fmod` | 69 | 130 |
///
/// (`f64`, 262144 elements, 16 logical cores, `cubecl-cpu` 0.10, the
/// geometries interleaved within one process so that drift lands on both.)
///
/// So this is not "how expensive is the function" — `exp` and `erf` cost
/// about the same and want opposite geometries. It is whether the body has
/// branches in it. On a device with planes the distinction does not arise and
/// this is ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuShape {
    /// Straight-line: a table lookup, a polynomial, bit manipulation. The
    /// vectorised cube loop is worth more than the threads that would replace
    /// it.
    Vectorised,
    /// Branchy: argument reduction with cases, a classification tree, a loop.
    /// There is no vectorisation to lose, and threads scale.
    Threaded,
}

/// Whether the runtime has hardware planes — warps or wavefronts — as opposed
/// to being the CPU runtime, where a unit is an OS thread.
///
/// The two want opposite things from a kernel, and this is the question to ask
/// before writing one that stages through shared memory or synchronises: on
/// the CPU runtime shared memory is ordinary DRAM and `sync_cube` is a
/// thread-pool barrier, so both cost without paying anything back.
pub fn has_planes<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().hardware.plane_size_max > 1
}

/// Launch geometry for a flat elementwise pass over `lanes` units.
///
/// On a device with planes the workgroup is sized from the hardware's own
/// plane configuration, which is what keeps the wavefronts full: a hardcoded
/// 256 is a reasonable guess for a discrete NVIDIA card and a bad one for
/// everything else, including a 64-wide AMD wavefront and the CPU runtime.
///
/// On the CPU runtime `shape` decides it instead — see [`CpuShape`], which is
/// where the reasoning and the measurements are.
///
/// One unit per element, in both cases. Elementwise transcendental work is
/// ALU-bound rather than bandwidth-bound, so there is nothing to gain from
/// giving each unit several elements or from reading them as vectors: the
/// loads are already hidden behind the hundred operations that follow them.
pub fn launch_1d<R: Runtime>(
    client: &ComputeClient<R>,
    lanes: usize,
    shape: CpuShape,
) -> (CubeCount, CubeDim) {
    let hardware = &client.properties().hardware;
    let cube_dim = if hardware.plane_size_max > 1 {
        // `CubeDim::new` picks the size — a whole number of planes, capped by
        // `max_units_per_cube` — and hands it back as a 2D block of
        // `plane_size * planes`. Take the size and flatten the shape: for a
        // flat elementwise pass the second axis has nothing to describe.
        //
        // On a Radeon 860M (32-wide planes, 1024 units per cube) this comes to
        // 256 units, so it lands on the same number the hardcoded constant
        // did, and measures the same to within a few percent — the gain here
        // is that a 64-wide wavefront or a smaller cube limit now gets a
        // different answer instead of the same one. Flattening is worth 21% on
        // `atan` against `CubeDim::new`'s own (32, 8) and a wash elsewhere.
        CubeDim::new_1d(CubeDim::new(client, lanes).num_elems())
    } else {
        match shape {
            CpuShape::Vectorised => CubeDim::new_1d(CPU_VECTORISED_UNITS),
            CpuShape::Threaded => {
                let cores = (hardware.num_cpu_cores.unwrap_or(1).max(1) as usize)
                    .saturating_mul(THREADED_CORE_FACTOR);
                let total = lanes.saturating_mul(THREADED_WORK_PER_ELEM);
                let units = (total / WORK_PER_CPU_UNIT).clamp(1, cores.min(lanes.max(1)));
                CubeDim::new_1d((units as u32).min(CPU_CUBE_DIM_MAX))
            }
        }
    };
    (
        cubecl::calculate_cube_count_elemwise(client, lanes, cube_dim),
        cube_dim,
    )
}

/// Check that two operands agree on length.
pub(crate) fn same_len(left: usize, right: usize) -> Result<(), crate::MathError> {
    if left == right {
        Ok(())
    } else {
        Err(crate::MathError::LengthMismatch { left, right })
    }
}
