//! The claim, checked: every kernel agrees with `rmath` bit for bit.
//!
//! `rmath` is the crate this one is a port of, and its own suite pins it to
//! the platform `libm` over millions of inputs — so agreeing with `rmath` is
//! agreeing with glibc, established transitively without this suite having to
//! re-derive it.
//!
//! Everything runs on a real CubeCL runtime rather than by calling the kernels
//! as ordinary Rust functions. That is the point, not a limitation to work
//! around: what could be wrong is what the *backend* does with the arithmetic,
//! and only a compiled kernel exercises that. The same cases run on every
//! runtime the build enabled.

mod harness;

use cube_math::prelude::*;
use cubecl::prelude::Runtime;
use harness::{check_f64, sweep_f64};
use rmath::prelude::*;

/// Every double-precision case, against one device.
fn suite_f64<R: Runtime>(backend: &'static str, ctx: &Ctx<R>) {
    if !ctx.fidelity.f64.usable {
        eprintln!("[{backend}] no usable f64; skipping the double-precision suite");
        return;
    }
    assert!(
        ctx.fidelity.f64.bit_exact_capable(),
        "[{backend}] this device rewrites f64 arithmetic the kernels depend on: {}",
        ctx.fidelity.f64.summary(),
    );

    let exp_sweep = sweep_f64(709.9);
    check_f64(
        backend,
        "exp",
        |xs| cube_math::function::Exp::new().eval_f64(ctx, xs),
        |x| rmath::Exp::new().eval(x),
        &exp_sweep,
    );
}

/// Run everything on one runtime.
fn run<R: Runtime>(backend: &'static str, device: &R::Device) {
    let ctx = Ctx::<R>::new(device);
    eprintln!("[{backend}] {:#?}", ctx.fidelity);
    suite_f64(backend, &ctx);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime() {
    run::<cubecl::cpu::CpuRuntime>("cpu", &Default::default());
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_runtime() {
    run::<cubecl::wgpu::WgpuRuntime>("wgpu", &Default::default());
}
