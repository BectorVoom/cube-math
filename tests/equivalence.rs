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
    check, check_mixed, check_pair, check_ulp, check2, check2_pair, eval1, eval1_pair, eval2,
    eval2_pair, sweep_f32, sweep_f64, sweep_order_f32, sweep_order_f64, sweep2_f32, sweep2_f64,
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

    // The families with one algorithm and no cheaper approximation worth
    // having — see each kernel's module docs. `Fast` runs the same code, so it
    // is held to the same bit-exact standard rather than to an ulp bound.
    macro_rules! single_algo {
        ($name:literal, $op:expr, $rm:ident, $sweep:expr) => {
            if exact {
                let sw = $sweep;
                check(
                    backend,
                    $name,
                    |v| eval1(client, $op, v, F64, cfg_exact),
                    |x| rmath::$rm::new().eval(x),
                    &sw,
                );
                check(
                    backend,
                    concat!($name, " (fast)"),
                    |v| eval1(client, $op, v, F64, cfg_fast),
                    |x| rmath::$rm::new().eval(x),
                    &sw,
                );
            }
        };
    }
    single_algo!("asin", Unary::Asin, Asin, sweep_f64(1.25));
    single_algo!("acos", Unary::Acos, Acos, sweep_f64(1.25));
    single_algo!("atan", Unary::Atan, Atan, sweep_f64(1e300));
    single_algo!("sin", Unary::Sin, Sin, sweep_f64(1e300));
    single_algo!("cos", Unary::Cos, Cos, sweep_f64(1e300));
    single_algo!("tan", Unary::Tan, Tan, sweep_f64(1e300));
    single_algo!("asinh", Unary::Asinh, Asinh, sweep_f64(1e300));
    single_algo!("acosh", Unary::Acosh, Acosh, sweep_f64(1e300));
    single_algo!("erf", Unary::Erf, Erf, sweep_f64(6.0));
    single_algo!("erfc", Unary::Erfc, Erfc, sweep_f64(30.0));
    single_algo!("j0", Unary::J0, J0, sweep_f64(50.0));
    single_algo!("j1", Unary::J1, J1, sweep_f64(50.0));
    single_algo!("y0", Unary::Y0, Y0, sweep_f64(50.0));
    single_algo!("y1", Unary::Y1, Y1, sweep_f64(50.0));

    if exact {
        // `jn` and `yn` take the order in the first lane, so the sweep is a
        // cross product of orders against arguments rather than the
        // magnitude-driven one the others use: what selects the recurrence is
        // `n` *against* `x`, and each of the three branches has to be reached.
        let (na, xb) = sweep_order_f64();
        check2(
            backend,
            "jn",
            |p, q| eval2(client, Binary::Jn, p, q, F64, cfg_exact),
            |n, x| rmath::Jn::new().eval(n, x),
            &na,
            &xb,
        );
        check2(
            backend,
            "yn",
            |p, q| eval2(client, Binary::Yn, p, q, F64, cfg_exact),
            |n, x| rmath::Yn::new().eval(n, x),
            &na,
            &xb,
        );
    }

    if exact {
        // The trigonometric sweep on the band each reduction owns, not just on
        // the whole range: `1e300` lands almost every input in `branred`, and
        // the three cheaper reductions would go all but untested.
        for limit in [0.5, 2.0, 25.0, 1e7, 1e9] {
            let sw = sweep_f64(limit);
            for (name, op, rm) in [
                ("sin", Unary::Sin, 0u8),
                ("cos", Unary::Cos, 1u8),
                ("tan", Unary::Tan, 2u8),
            ] {
                check(
                    backend,
                    &format!("{name} (|x| < {limit:e})"),
                    |v| eval1(client, op, v, F64, cfg_exact),
                    |x| match rm {
                        0 => rmath::Sin::new().eval(x),
                        1 => rmath::Cos::new().eval(x),
                        _ => rmath::Tan::new().eval(x),
                    },
                    &sw,
                );
            }
        }
        check_pair(
            backend,
            "sincos",
            |v| eval1_pair(client, UnaryPair::SinCos, v, F64, cfg_exact),
            |x| rmath::SinCos::new().eval(x),
            &sweep_f64(1e300),
        );
    }

    if exact {
        check2(
            backend,
            "atan2",
            |p, q| eval2(client, Binary::Atan2, p, q, F64, cfg_exact),
            |x, y| rmath::Atan2::new().eval(x, y),
            &pa,
            &pb,
        );
    }

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
    ported!("sinh", Unary::Sinh, Sinh, 3.0, sweep_f64(710.0));
    ported!("cosh", Unary::Cosh, Cosh, 3.0, sweep_f64(710.0));
    ported!("tanh", Unary::Tanh, Tanh, 4.0, sweep_f64(30.0));
    ported!("atanh", Unary::Atanh, Atanh, 4.0, sweep_f64(1.25));
    // Gamma makes no bit-exactness claim in either crate — see the kernel's
    // module docs — so it is held to an ulp bound against `rmath` under both
    // policies rather than to the bits.
    check_mixed(
        backend,
        "lgamma",
        |v| eval1(client, Unary::LGamma, v, F64, cfg_exact),
        |x| rmath::LGamma::new().eval(x),
        &sweep_f64(200.0),
        8.0,
        1e-15,
    );
    // `tgamma` never touches the table-free logarithm — its recurrence is a
    // product and its Stirling branch goes through `pow`'s double-double
    // logarithm — so it comes out bit-identical to `rmath` despite neither
    // crate claiming bit-exactness for the family. Held to that.
    check(
        backend,
        "tgamma",
        |v| eval1(client, Unary::TGamma, v, F64, cfg_exact),
        |x| rmath::TGamma::new().eval(x),
        &sweep_f64(175.0),
    );
    // `lgamma_r`'s *sign* is exact — it comes from the parity of `floor(x)`,
    // not from the value — so it is checked bit for bit. The value it returns
    // alongside is `lgamma`'s, already covered above.
    {
        let xs = sweep_f64(200.0);
        let (_, signs) = eval1_pair(client, UnaryPair::LGammaR, &xs, F64, cfg_exact);
        check(
            backend,
            "lgamma_r sign",
            |_| signs.clone(),
            |x| rmath::LGammaR::new().eval(x).1,
            &xs,
        );
    }
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
    let cfg = MathConfig::new(Policy::EXACT, fid.f32.fma_kind());
    exact_family_f32(backend, client, cfg);

    // The single-precision transcendentals that are genuine ports: ARM's
    // optimized-routines, evaluated in `f64` over a small table and rounded
    // once, which is the schedule rather than an implementation detail. They
    // need the *double* precision to be usable, so they are skipped where it
    // is not.
    if !fid.f64.usable || !fid.f64.bit_exact_capable() {
        eprintln!("[{backend}] f32 transcendentals need a bit-exact f64; skipping");
        return;
    }
    macro_rules! ported32 {
        ($name:literal, $op:expr, $rm:ident, $sweep:expr) => {
            check(
                backend,
                concat!($name, " f32"),
                |v| eval1(client, $op, v, F32, cfg),
                |x| rmath::$rm::new().eval(x),
                &$sweep,
            );
        };
    }
    ported32!("exp", Unary::Exp, Exp, sweep_f32(90.0));
    ported32!("exp2", Unary::Exp2, Exp2, sweep_f32(130.0));
    ported32!("exp10", Unary::Exp10, Exp10, sweep_f32(40.0));
    ported32!("ln", Unary::Ln, Ln, sweep_f32(1e30));
    ported32!("log2", Unary::Log2, Log2, sweep_f32(1e30));

    // Everything else. Some of these are schedule ports of their own
    // (`sinf`, `cosf`, `powf`, `atan2f`, the Bessel family); the rest are
    // computed in double precision and rounded once, which reaches the
    // platform's answer because the platform computes *those* correctly
    // rounded. Either way the sweep holds them to the bits — see
    // `cube_math::single::wide` for the three inputs in 2^32 across the whole
    // widened set that are known to differ, and why a sampled sweep is not
    // expected to land on them.
    macro_rules! wide32 {
        ($name:literal, $op:expr, $rm:ident, $sweep:expr) => {
            check(
                backend,
                concat!($name, " f32"),
                |v| eval1(client, $op, v, F32, cfg),
                |x| rmath::$rm::new().eval(x),
                &$sweep,
            );
        };
    }
    wide32!("log10", Unary::Log10, Log10, sweep_f32(1e30));
    wide32!("log1p", Unary::Log1p, Log1p, sweep_f32(1e30));
    wide32!("expm1", Unary::Expm1, Expm1, sweep_f32(90.0));
    wide32!("cbrt", Unary::Cbrt, Cbrt, sweep_f32(1e30));
    wide32!("sin", Unary::Sin, Sin, sweep_f32(1e6));
    wide32!("cos", Unary::Cos, Cos, sweep_f32(1e6));
    wide32!("tan", Unary::Tan, Tan, sweep_f32(1e6));
    wide32!("asin", Unary::Asin, Asin, sweep_f32(1.25));
    wide32!("acos", Unary::Acos, Acos, sweep_f32(1.25));
    wide32!("atan", Unary::Atan, Atan, sweep_f32(1e30));
    wide32!("sinh", Unary::Sinh, Sinh, sweep_f32(90.0));
    wide32!("cosh", Unary::Cosh, Cosh, sweep_f32(90.0));
    wide32!("tanh", Unary::Tanh, Tanh, sweep_f32(30.0));
    wide32!("asinh", Unary::Asinh, Asinh, sweep_f32(1e30));
    wide32!("acosh", Unary::Acosh, Acosh, sweep_f32(1e30));
    wide32!("atanh", Unary::Atanh, Atanh, sweep_f32(1.25));
    wide32!("erf", Unary::Erf, Erf, sweep_f32(6.0));
    wide32!("erfc", Unary::Erfc, Erfc, sweep_f32(30.0));
    wide32!("j0", Unary::J0, J0, sweep_f32(50.0));
    wide32!("j1", Unary::J1, J1, sweep_f32(50.0));
    wide32!("y0", Unary::Y0, Y0, sweep_f32(50.0));
    wide32!("y1", Unary::Y1, Y1, sweep_f32(50.0));

    let (wa, wb) = sweep2_f32();
    macro_rules! wide32_2 {
        ($name:literal, $op:expr, $rm:ident) => {
            check2(
                backend,
                concat!($name, " f32"),
                |p, q| eval2(client, $op, p, q, F32, cfg),
                |x, y| rmath::$rm::new().eval(x, y),
                &wa,
                &wb,
            );
        };
    }
    wide32_2!("pow", Binary::Pow, Pow);
    wide32_2!("hypot", Binary::Hypot, Hypot);
    wide32_2!("atan2", Binary::Atan2, Atan2);

    // `jnf` and `ynf`, on the same order-against-argument cross product the
    // double-precision pair get.
    let (na, xb) = sweep_order_f32();
    check2(
        backend,
        "jn f32",
        |p, q| eval2(client, Binary::Jn, p, q, F32, cfg),
        |n, x| rmath::Jn::new().eval(n, x),
        &na,
        &xb,
    );
    check2(
        backend,
        "yn f32",
        |p, q| eval2(client, Binary::Yn, p, q, F32, cfg),
        |n, x| rmath::Yn::new().eval(n, x),
        &na,
        &xb,
    );

    check_pair(
        backend,
        "sincos f32",
        |v| eval1_pair(client, UnaryPair::SinCos, v, F32, cfg),
        |x| rmath::SinCos::new().eval(x),
        &sweep_f32(1e6),
    );
    {
        let xs = sweep_f32(200.0);
        let (_, signs) = eval1_pair(client, UnaryPair::LGammaR, &xs, F32, cfg);
        check(
            backend,
            "lgamma_r sign f32",
            |_| signs.clone(),
            |x| rmath::LGammaR::new().eval(x).1,
            &xs,
        );
    }
    wide32!("lgamma", Unary::LGamma, LGamma, sweep_f32(200.0));
    wide32!("tgamma", Unary::TGamma, TGamma, sweep_f32(35.0));
}

/// Run everything on one runtime.
fn run<R: Runtime>(backend: &'static str, device: &R::Device) {
    let client = R::client(device);
    let fid = fidelity(&client);
    eprintln!(
        "[{backend}] f64: {} | f32: {}",
        fid.f64.summary(),
        fid.f32.summary()
    );
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
