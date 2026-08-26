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
use harness::{check, check2, check2_pair, check_pair, check_ulp, sweep2_f32, sweep2_f64, sweep_f32, sweep_f64};
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
        check(
            backend,
            "exp",
            |xs| cube_math::function::Exp::new().eval_f64(ctx, xs),
            |x| rmath::Exp::new().eval(x),
            &exp_sweep,
        );
    }
    exact_family_f64(backend, ctx);

    // The ported transcendentals. Each is checked against `rmath`'s bit-exact
    // object, which its own suite pins to the platform `libm`.
    macro_rules! ported {
        ($name:literal, $cube:ident, $rm:ident, $limit:expr, $sweep:expr) => {
            let sw = $sweep;
            if exact {
                check(
                    backend,
                    $name,
                    |v| cube_math::function::$cube::new().eval_f64(ctx, v),
                    |x| rmath::$rm::new().eval(x),
                    &sw,
                );
            }
            check_ulp(
                backend,
                concat!($name, " (fast)"),
                |v| cube_math::function::$cube::fast().eval_f64(ctx, v),
                |x| rmath::$rm::new().eval(x),
                &sw,
                $limit,
            );
        };
    }
    ported!("exp2", Exp2, Exp2, 1.0, sweep_f64(1024.0));
    ported!("exp10", Exp10, Exp10, 1.5, sweep_f64(310.0));
    ported!("ln", Ln, Ln, 2.0, sweep_f64(1e300));
    ported!("log2", Log2, Log2, 2.0, sweep_f64(1e300));
    ported!("log10", Log10, Log10, 2.0, sweep_f64(1e300));
    ported!("expm1", Expm1, Expm1, 2.0, sweep_f64(710.0));
    ported!("log1p", Log1p, Log1p, 3.0, sweep_f64(1e300));

    check_ulp(
        backend,
        "exp (fast)",
        |xs| cube_math::function::Exp::fast().eval_f64(ctx, xs),
        |x| rmath::Exp::new().eval(x),
        &exp_sweep,
        1.0,
    );
}

/// The functions IEEE-754 pins down exactly.
///
/// These need no fused multiply-add and make no claim about a particular
/// `libm`, so they run on every backend that can do `f64` at all — including
/// the ones the bit-exact transcendentals have to skip.
macro_rules! exact_family {
    ($fname:ident, $ty:ty, $eval:ident, $sweep:expr, $sweep2:expr) => {
fn $fname<R: Runtime>(backend: &'static str, ctx: &Ctx<R>) {
    use cube_math::function as f;

    let xs: Vec<$ty> = $sweep;
    macro_rules! unary {
        ($name:literal, $cube:ident, $rm:ident) => {
            check(
                backend,
                concat!($name, " ", stringify!($ty)),
                |v| f::$cube::new().$eval(ctx, v),
                |x| rmath::$rm::new().eval(x),
                &xs,
            );
        };
    }
    unary!("floor", Floor, Floor);
    unary!("ceil", Ceil, Ceil);
    unary!("trunc", Trunc, Trunc);
    unary!("round", Round, Round);
    unary!("rint", Rint, Rint);
    unary!("sqrt", Sqrt, Sqrt);
    unary!("abs", Abs, Abs);
    unary!("ilogb", Ilogb, Ilogb);

    let (a, b): (Vec<$ty>, Vec<$ty>) = $sweep2;
    macro_rules! binary {
        ($name:literal, $cube:ident, $rm:ident) => {
            check2(
                backend,
                concat!($name, " ", stringify!($ty)),
                |p, q| f::$cube::new().$eval(ctx, p, q),
                |x, y| rmath::$rm::new().eval(x, y),
                &a,
                &b,
            );
        };
    }
    binary!("copysign", CopySign, CopySign);
    binary!("fdim", Fdim, Fdim);
    binary!("fmax", Fmax, Fmax);
    binary!("fmin", Fmin, Fmin);
    binary!("fmod", Fmod, Fmod);
    binary!("remainder", Remainder, Remainder);
    binary!("nextafter", NextAfter, NextAfter);
    binary!("ldexp", Ldexp, Ldexp);
    binary!("scalbn", Scalbn, Scalbn);

    check_pair(
        backend,
        concat!("frexp ", stringify!($ty)),
        |v| f::Frexp::new().$eval(ctx, v),
        |x| rmath::Frexp::new().eval(x),
        &xs,
    );
    check_pair(
        backend,
        concat!("modf ", stringify!($ty)),
        |v| f::Modf::new().$eval(ctx, v),
        |x| rmath::Modf::new().eval(x),
        &xs,
    );
    check2_pair(
        backend,
        concat!("remquo ", stringify!($ty)),
        |p, q| f::Remquo::new().$eval(ctx, p, q),
        |x, y| rmath::Remquo::new().eval(x, y),
        &a,
        &b,
    );
}
    };
}

exact_family!(exact_family_f64, f64, eval_f64, sweep_f64(1000.0), sweep2_f64());
exact_family!(exact_family_f32, f32, eval_f32, sweep_f32(1000.0), sweep2_f32());

/// Every single-precision case, against one device.
fn suite_f32<R: Runtime>(backend: &'static str, ctx: &Ctx<R>) {
    if !ctx.fidelity.f32.usable {
        eprintln!("[{backend}] f32 does not run on this backend; skipping");
        return;
    }
    if !ctx.fidelity.f32.bit_exact_capable() {
        eprintln!(
            "[{backend}] f32 NOT bit-exact capable ({}); skipping the exact suite",
            ctx.fidelity.f32.summary(),
        );
        return;
    }
    exact_family_f32(backend, ctx);
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
    suite_f32(backend, &ctx);
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
