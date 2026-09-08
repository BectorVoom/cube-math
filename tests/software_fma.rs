//! The bit-exact kernels are still bit-exact when the fused multiply-add is
//! emulated.
//!
//! [`FmaKind::Software`] exists for the backends whose `fma` rounds twice —
//! SPIR-V, and so `wgpu` on Vulkan, the only route to `f64` on a non-NVIDIA
//! GPU. Neither backend this suite can reach is one of those: the CPU runtime
//! and ROCm both fuse, so `fidelity()` reports `Hardware` on both and the
//! whole software path goes unexercised by `tests/equivalence.rs`, which takes
//! its `FmaKind` from the device.
//!
//! So this file forces it. Every case below builds the kernel with
//! `FmaKind::Software` on a device that has a real fused multiply-add, and
//! requires the same bits `rmath` produces. That is a meaningful check on such
//! a device precisely because [`cube_math::fma::fma_f64()`] is *correctly
//! rounded*: a correct emulation must land on the hardware instruction's
//! answer, so agreement is evidence and disagreement is a bug in the
//! emulation or in the schedule that calls it.
//!
//! What it caught: `ln`, `log2` and `log10` used the raw `fma` intrinsic
//! rather than `fma64(.., fk)`, so their bit-exact paths ignored `FmaKind`
//! entirely and would have rounded twice on a backend that does not fuse.
//! `tests/equivalence.rs` could not see it — on a fused device the two spell
//! the same number — and this file cannot see it either, for the same reason.
//! What this file does is hold the *repaired* code to the contract: the
//! software path now really runs, and it really reproduces glibc's schedule.
//! The static half of the check is that no `fma(` remains in those modules.

mod harness;

use cube_math::launch::{F64, Unary};
use cube_math::prelude::*;
use cube_math::{FmaKind, MathConfig};
use cubecl::prelude::*;
use harness::{check, eval1, sweep_f64};
use rmath::prelude::*;

fn suite<R: Runtime>(backend: &'static str, client: &ComputeClient<R>) {
    let fid = fidelity(client);
    if !fid.f64.usable {
        eprintln!("[{backend}] f64 does not run on this backend; skipping");
        return;
    }
    if !fid.f64.bit_exact_capable() {
        eprintln!(
            "[{backend}] not bit-exact capable ({}); skipping",
            fid.f64.summary(),
        );
        return;
    }
    if !matches!(fid.f64.fma_kind(), FmaKind::Hardware) {
        // The device emulates already, so `equivalence.rs` is this test.
        eprintln!("[{backend}] device has no fused multiply-add; already covered");
        return;
    }

    let cfg = MathConfig::new(Policy::EXACT, FmaKind::Software);

    macro_rules! emulated {
        ($name:literal, $op:expr, $rm:ident, $sweep:expr) => {
            let sw = $sweep;
            check(
                backend,
                concat!($name, " (software fma)"),
                |v| eval1(client, $op, v, F64, cfg),
                |x| rmath::$rm::new().eval(x),
                &sw,
            );
            eprintln!(
                "[{backend}] {} with an emulated fma: bit-identical to rmath over {} inputs",
                $name,
                sw.len(),
            );
        };
    }

    // The three this file was written for.
    emulated!("ln", Unary::Ln, Ln, sweep_f64(1e300));
    emulated!("log2", Unary::Log2, Log2, sweep_f64(1e300));
    emulated!("log10", Unary::Log10, Log10, sweep_f64(1e300));
    // Controls: these threaded `FmaKind` from the start, so they are what a
    // passing run of the three above is being compared against.
    emulated!("exp", Unary::Exp, Exp, sweep_f64(709.9));
    emulated!("exp2", Unary::Exp2, Exp2, sweep_f64(1024.0));
    emulated!("exp10", Unary::Exp10, Exp10, sweep_f64(310.0));
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    suite("cpu", &client);
}

#[cfg(feature = "hip")]
#[test]
fn hip_runtime() {
    let client = cubecl::hip::HipRuntime::client(&Default::default());
    suite("hip", &client);
}
