//! What the device actually does with floating point.
//!
//! Bit-exactness is a claim about a *device*, not about this source. Three
//! properties have to hold for the kernels' claim to survive, none is
//! guaranteed by CubeCL, and each of them fails on some backend in wide use.
//! So rather than assume, [`Fidelity::measure`] runs kernels whose answers
//! differ depending on each one, and reports what it found.
//!
//! ```no_run
//! # use cube_math::probe::Fidelity;
//! # use cubecl::prelude::*;
//! # fn f<R: Runtime>(client: &ComputeClient<R>) {
//! let fidelity = Fidelity::measure(client);
//! assert!(fidelity.f64.bit_exact_capable(), "{fidelity:#?}");
//! # }
//! ```
//!
//! [`crate::Ctx`] does this once when it is built and uses the answer to pick
//! [`FmaKind`] per precision, so the default configuration is correct
//! everywhere and pays for emulation only where emulation is needed.
//!
//! # What is measured, and what it costs to fail
//!
//! | property | if false | recoverable? |
//! |---|---|---|
//! | `usable` | the precision does not run here at all | no |
//! | `fused_fma` | `fma` rounds twice | yes — [`FmaKind::Software`] |
//! | `separate_mul_add` | `a * b + c` is contracted | no |
//! | `subnormals` | subnormals are flushed to zero | no |
//!
//! The two irrecoverable ones are not pessimism. A contracted `a * b + c`
//! breaks every schedule glibc compiled *without* `-mfma` — `exp2` is one —
//! and there is no way to express "do not fuse this" through CubeCL. Flushed
//! subnormals change results near the bottom of the range in a way no
//! rearrangement recovers.

use cubecl::prelude::*;

use crate::cube::fma::FmaKind;

/// What one precision does on this device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Precision {
    /// This precision runs here at all.
    ///
    /// False for `f64` on the WGSL path without `SHADER_F64`, and on WebGPU
    /// generally.
    pub usable: bool,
    /// `fma(a, b, c)` is a true fused multiply-add — **one** rounding.
    ///
    /// False on the SPIR-V backend, which lowers it to a separate multiply and
    /// add. Recoverable: see [`FmaKind::Software`].
    pub fused_fma: bool,
    /// `a * b + c` is *not* contracted into a fused multiply-add.
    pub separate_mul_add: bool,
    /// Subnormals survive arithmetic instead of being flushed to zero.
    pub subnormals: bool,
}

impl Precision {
    /// Whether [`crate::Accuracy::BitExact`] can be honoured for this
    /// precision, given the right [`FmaKind`].
    pub fn bit_exact_capable(self) -> bool {
        self.usable && self.separate_mul_add && self.subnormals
    }

    /// Which multiply-add the kernels should be built with here.
    pub fn fma_kind(self) -> FmaKind {
        if self.fused_fma { FmaKind::Hardware } else { FmaKind::Software }
    }

    /// A one-line summary, for a test log or a failure message.
    pub fn summary(self) -> String {
        if !self.usable {
            return "unusable".into();
        }
        let mut s = String::new();
        s.push_str(if self.fused_fma { "fused-fma" } else { "SPLIT-FMA" });
        s.push_str(if self.separate_mul_add { ", no-contract" } else { ", CONTRACTS" });
        s.push_str(if self.subnormals { ", subnormals" } else { ", FLUSHES-SUBNORMALS" });
        s
    }
}

/// What a device does with the operations bit-exactness depends on, per
/// precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Fidelity {
    /// Double precision.
    pub f64: Precision,
    /// Single precision.
    pub f32: Precision,
}

impl Fidelity {
    /// Whether `f64` runs here at all. A shorthand for `self.f64.usable`.
    pub fn has_f64(self) -> bool {
        self.f64.usable
    }

    /// Run the probe kernels on `client` and report what they found.
    pub fn measure<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self { f64: measure_f64(client), f32: measure_f32(client) }
    }
}

/// `[a*b+c, fma(a,b,c), tiny*4, sentinel]`.
#[cube(launch_unchecked)]
fn probe_k64(inp: &Array<f64>, out: &mut Array<f64>) {
    if ABSOLUTE_POS < 1usize {
        let a = inp[0];
        let b = inp[1];
        let c = inp[2];
        out[0] = a * b + c;
        out[1] = fma(a, b, c);
        out[2] = inp[3] * 4.0f64;
        out[3] = 1.0f64;
    }
}

/// The single-precision counterpart.
#[cube(launch_unchecked)]
fn probe_k32(inp: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS < 1usize {
        let a = inp[0];
        let b = inp[1];
        let c = inp[2];
        out[0] = a * b + c;
        out[1] = fma(a, b, c);
        out[2] = inp[3] * 4.0f32;
        out[3] = 1.0f32;
    }
}

/// `a` and `b` chosen so that their exact product is `1 + 2^-53 - 2^-105`,
/// which rounds to exactly 1 — so `a * b - 1` is zero unfused and
/// `2^-53 - 2^-105` fused. One input pair separates both properties.
fn measure_f64<R: Runtime>(client: &ComputeClient<R>) -> Precision {
    let a = 1.0 + f64::EPSILON;
    let b = 1.0 - f64::EPSILON / 2.0;
    let input = vec![a, b, -1.0, f64::from_bits(1)];
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(input));
    let out_h = client.empty(4 * size_of::<f64>());
    unsafe {
        probe_k64::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(in_h, 4),
            ArrayArg::from_raw_parts(out_h.clone(), 4),
        );
    }
    let Ok(bytes) = client.read_one(out_h) else {
        return Precision::default();
    };
    let o = f64::from_bytes(&bytes);
    // A backend that cannot compile the kernel does not always report an
    // error — `wgpu` on the WGSL path simply leaves the buffer untouched — so
    // the only reliable test is whether the kernel wrote its sentinel.
    if o[3] != 1.0 {
        return Precision::default();
    }
    Precision {
        usable: true,
        fused_fma: o[1].to_bits() == a.mul_add(b, -1.0).to_bits(),
        separate_mul_add: o[0] == 0.0,
        subnormals: o[2].to_bits() == f64::from_bits(4).to_bits(),
    }
}

/// The same test one precision down: `a * b` is `1 + 2^-24 - 2^-47`, which
/// rounds to 1.
fn measure_f32<R: Runtime>(client: &ComputeClient<R>) -> Precision {
    let a = 1.0f32 + f32::EPSILON;
    let b = 1.0f32 - f32::EPSILON / 2.0;
    let input = vec![a, b, -1.0f32, f32::from_bits(1)];
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(input));
    let out_h = client.empty(4 * size_of::<f32>());
    unsafe {
        probe_k32::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(in_h, 4),
            ArrayArg::from_raw_parts(out_h.clone(), 4),
        );
    }
    let Ok(bytes) = client.read_one(out_h) else {
        return Precision::default();
    };
    let o = f32::from_bytes(&bytes);
    if o[3] != 1.0 {
        return Precision::default();
    }
    Precision {
        usable: true,
        fused_fma: o[1].to_bits() == a.mul_add(b, -1.0).to_bits(),
        separate_mul_add: o[0] == 0.0,
        subnormals: o[2].to_bits() == f32::from_bits(4).to_bits(),
    }
}
