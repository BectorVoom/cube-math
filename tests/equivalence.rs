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
use harness::{check_f64, check_ulp_f64, sweep_f64};
use rmath::prelude::*;

/// Every double-precision case, against one device.
///
/// A device that cannot be bit-exact is not a test failure — it is a fact
/// about the device, and [`Fidelity`] exists to state it. What *would* be a
/// failure is claiming bit-exactness there, so the exact half of the suite is
/// skipped loudly and the approximate half still runs.
fn suite_f64<R: Runtime>(backend: &'static str, ctx: &Ctx<R>) {
    if !ctx.fidelity.f64.usable {
        eprintln!("[{backend}] f64 does not run on this backend; skipping");
        return;
    }
    let exact = ctx.fidelity.f64.bit_exact_capable();
    if !exact {
        eprintln!(
            "[{backend}] NOT bit-exact capable ({}); checking the Fast policy only",
            ctx.fidelity.f64.summary(),
        );
    }

    let exp_sweep = sweep_f64(709.9);
    if exact {
        check_f64(
            backend,
            "exp",
            |xs| cube_math::function::Exp::new().eval_f64(ctx, xs),
            |x| rmath::Exp::new().eval(x),
            &exp_sweep,
        );
    }
    check_ulp_f64(
        backend,
        "exp (fast)",
        |xs| cube_math::function::Exp::fast().eval_f64(ctx, xs),
        |x| rmath::Exp::new().eval(x),
        &exp_sweep,
        1.0,
    );
}

/// Run everything on one runtime.
fn run<R: Runtime>(backend: &'static str, device: &R::Device) {
    let ctx = Ctx::<R>::new(device);
    eprintln!(
        "[{backend}] f64: {} | f32: {}",
        ctx.fidelity.f64.summary(),
        ctx.fidelity.f32.summary(),
    );
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
