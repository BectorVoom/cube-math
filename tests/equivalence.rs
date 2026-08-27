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

use cube_math::launch::{F32, F64};
use cube_math::prelude::*;
use cubecl::prelude::*;
use harness::{
    check, check2, check2_pair, check_pair, check_ulp, eval1, eval1_pair, eval2, eval2_pair,
    sweep2_f32, sweep2_f64, sweep_f32, sweep_f64,
};
use rmath::prelude::*;

/// Every double-precision case, against one device.
///
/// A device that cannot be bit-exact is not a test failure — it is a fact
/// about the device, and [`Fidelity`] exists to state it. What *would* be a
/// failure is claiming bit-exactness there, so the exact half of the suite is
/// skipped loudly and the approximate half still runs.
fn suite_f64<R: Runtime>(backend: &'static str, client: &ComputeClient<R>, fid: Fidelity) {
    if !fid.f64.usable {
        eprintln!("[{backend}] f64 does not run on this backend; skipping");
        return;
    }
    let exact = fid.f64.bit_exact_capable();
    if !exact {
        eprintln!(
            "[{backend}] NOT bit-exact capable ({}); checking the Fast policy only",
            fid.f64.summary(),
        );
    }
    let cfg_exact = MathConfig::new(Policy::EXACT, fid.f64.fma_kind());
    let cfg_fast = MathConfig::new(Policy::FAST, fid.f64.fma_kind());

    exact_family_f64(backend, client, cfg_exact);

    // `pow` and `hypot`, on a cross product: what breaks a two-argument
    // function is the relationship between the arguments, not either alone.
    let (pa, pb) = sweep2_f64();
    check2(
        backend,
        "hypot",
        |p, q| eval2(client, Binary::Hypot, p, q, F64, cfg_exact),
        |x, y| rmath::Hypot::new().eval(x, y),
        &pa,
        &pb,
    );
    if exact {
        check2(
            backend,
            "pow",
            |p, q| eval2(client, Binary::Pow, p, q, F64, cfg_exact),
            |x, y| rmath::Pow::new().eval(x, y),
            &pa,
            &pb,
        );
        // `Fast` is the same code for `pow` — see the kernel's module docs —
        // so it is held to the same bit-exact standard, not to a bound.
        check2(
            backend,
            "pow (fast)",
            |p, q| eval2(client, Binary::Pow, p, q, F64, cfg_fast),
            |x, y| rmath::Pow::new().eval(x, y),
            &pa,
            &pb,
        );
    }

    check(
        backend,
        "cbrt",
        |v| eval1(client, Unary::Cbrt, v, F64, cfg_exact),
        |x| rmath::Cbrt::new().eval(x),
        &sweep_f64(1e300),
    );

    // The ported transcendentals, each against its `rmath` object.
    macro_rules! ported {
        ($name:literal, $op:expr, $rm:ident, $limit:expr, $sweep:expr) => {
            let sw = $sweep;
            if exact {
                check(
                    backend,
                    $name,
                    |v| eval1(client, $op, v, F64, cfg_exact),
                    |x| rmath::$rm::new().eval(x),
                    &sw,
                );
            }
            check_ulp(
                backend,
                concat!($name, " (fast)"),
                |v| eval1(client, $op, v, F64, cfg_fast),
                |x| rmath::$rm::new().eval(x),
                &sw,
                $limit,
            );
        };
    }
    ported!("exp", Unary::Exp, Exp, 1.0, sweep_f64(709.9));
    ported!("exp2", Unary::Exp2, Exp2, 1.0, sweep_f64(1024.0));
    ported!("exp10", Unary::Exp10, Exp10, 1.5, sweep_f64(310.0));
    ported!("expm1", Unary::Expm1, Expm1, 2.0, sweep_f64(710.0));
    ported!("ln", Unary::Ln, Ln, 2.0, sweep_f64(1e300));
    ported!("log2", Unary::Log2, Log2, 2.0, sweep_f64(1e300));
    ported!("log10", Unary::Log10, Log10, 2.0, sweep_f64(1e300));
    ported!("log1p", Unary::Log1p, Log1p, 3.0, sweep_f64(1e300));
}

/// The functions IEEE-754 pins down exactly.
///
/// These need no fused multiply-add and make no claim about a particular
/// `libm`, so they run on every backend that can do the precision at all —
/// including ones the bit-exact transcendentals have to skip.
macro_rules! exact_family {
    ($fname:ident, $ty:ty, $dtype:expr, $sweep:expr, $sweep2:expr) => {
        fn $fname<R: Runtime>(backend: &'static str, client: &ComputeClient<R>, cfg: MathConfig) {
            let xs: Vec<$ty> = $sweep;
            macro_rules! unary {
                ($name:literal, $op:expr, $rm:ident) => {
                    check(
                        backend,
                        concat!($name, " ", stringify!($ty)),
                        |v| eval1(client, $op, v, $dtype, cfg),
                        |x| rmath::$rm::new().eval(x),
                        &xs,
                    );
                };
            }
            unary!("floor", Unary::Floor, Floor);
            unary!("ceil", Unary::Ceil, Ceil);
            unary!("trunc", Unary::Trunc, Trunc);
            unary!("round", Unary::Round, Round);
            unary!("rint", Unary::Rint, Rint);
            unary!("sqrt", Unary::Sqrt, Sqrt);
            unary!("abs", Unary::Abs, Abs);
            unary!("ilogb", Unary::Ilogb, Ilogb);

            let (a, b): (Vec<$ty>, Vec<$ty>) = $sweep2;
            macro_rules! binary {
                ($name:literal, $op:expr, $rm:ident) => {
                    check2(
                        backend,
                        concat!($name, " ", stringify!($ty)),
                        |p, q| eval2(client, $op, p, q, $dtype, cfg),
                        |x, y| rmath::$rm::new().eval(x, y),
                        &a,
                        &b,
                    );
                };
            }
            binary!("copysign", Binary::CopySign, CopySign);
            binary!("fdim", Binary::Fdim, Fdim);
            binary!("fmax", Binary::Fmax, Fmax);
            binary!("fmin", Binary::Fmin, Fmin);
            binary!("fmod", Binary::Fmod, Fmod);
            binary!("remainder", Binary::Remainder, Remainder);
            binary!("nextafter", Binary::NextAfter, NextAfter);
            binary!("ldexp", Binary::Ldexp, Ldexp);
            binary!("scalbn", Binary::Scalbn, Scalbn);

            check_pair(
                backend,
                concat!("frexp ", stringify!($ty)),
                |v| eval1_pair(client, UnaryPair::Frexp, v, $dtype, cfg),
                |x| rmath::Frexp::new().eval(x),
                &xs,
            );
            check_pair(
                backend,
                concat!("modf ", stringify!($ty)),
                |v| eval1_pair(client, UnaryPair::Modf, v, $dtype, cfg),
                |x| rmath::Modf::new().eval(x),
                &xs,
            );
            check2_pair(
                backend,
                concat!("remquo ", stringify!($ty)),
                |p, q| eval2_pair(client, BinaryPair::Remquo, p, q, $dtype, cfg),
                |x, y| rmath::Remquo::new().eval(x, y),
                &a,
                &b,
            );
        }
    };
}

exact_family!(exact_family_f64, f64, F64, sweep_f64(1000.0), sweep2_f64());
exact_family!(exact_family_f32, f32, F32, sweep_f32(1000.0), sweep2_f32());

/// Every single-precision case, against one device.
fn suite_f32<R: Runtime>(backend: &'static str, client: &ComputeClient<R>, fid: Fidelity) {
    if !fid.f32.usable {
        eprintln!("[{backend}] f32 does not run on this backend; skipping");
        return;
    }
    if !fid.f32.bit_exact_capable() {
        eprintln!(
            "[{backend}] f32 NOT bit-exact capable ({}); skipping the exact suite",
            fid.f32.summary(),
        );
        return;
    }
    exact_family_f32(backend, client, MathConfig::new(Policy::EXACT, fid.f32.fma_kind()));
}

/// Run everything on one runtime.
fn run<R: Runtime>(backend: &'static str, device: &R::Device) {
    let client = R::client(device);
    let fid = fidelity(&client);
    eprintln!("[{backend}] f64: {} | f32: {}", fid.f64.summary(), fid.f32.summary());
    suite_f64(backend, &client, fid);
    suite_f32(backend, &client, fid);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime() {
    run::<cubecl::cpu::CpuRuntime>("cpu", &Default::default());
}

#[cfg(feature = "hip")]
#[test]
fn hip_runtime() {
    run::<cubecl::hip::HipRuntime>("hip", &Default::default());
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_runtime() {
    run::<cubecl::wgpu::WgpuRuntime>("wgpu", &Default::default());
}
