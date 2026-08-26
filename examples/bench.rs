//! Throughput, against `rmath` on the CPU and against the platform `libm`.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release \
//!     --features "cpu,wgpu-spirv" --example bench
//! ```
//!
//! `target-cpu=native` matters for the two CPU baselines and not at all for
//! this crate: `rmath` and `libm` are compiled ahead of time and will not use
//! AVX-512 unless told to, while a CubeCL kernel is compiled for the machine
//! it is about to run on either way. Without the flag `rmath` measures about a
//! tenth of its real speed, and the comparison is meaningless.
//!
//! # What is being timed
//!
//! Kernel throughput on data that is already on the device — the `_into`
//! form, not the `Vec` one. That is the honest number for a kernel and the
//! dishonest one for a workload: a single `exp` over a host `Vec` is dominated
//! by the two transfers, and no amount of arithmetic speed changes that. The
//! GPU is worth reaching for when the data is already there, or when enough
//! work happens per element to pay for the trip.

use std::time::Instant;

use cube_math::prelude::*;
use cubecl::prelude::Runtime;
use rmath::prelude::*;

const N: usize = 1 << 20;
const REPS: usize = 20;

/// Inputs spread over the whole useful range of each function.
fn data(lo: f64, hi: f64) -> Vec<f64> {
    (0..N).map(|i| lo + (hi - lo) * (i as f64) / (N as f64)).collect()
}

fn timed(label: &str, n: usize, mut run: impl FnMut()) -> f64 {
    run();
    let t = Instant::now();
    for _ in 0..REPS {
        run();
    }
    let el = t.elapsed().as_secs_f64() / REPS as f64;
    println!("{label:<28} {:>8.3} ms  {:>9.1} Melem/s", el * 1e3, n as f64 / el / 1e6);
    n as f64 / el / 1e6
}

macro_rules! bench_one {
    ($ctx:expr, $xs:expr, $name:literal, $cube:ident, $rm:ident) => {{
        let ctx = $ctx;
        let xs: &[f64] = $xs;
        let inp = ctx.upload(xs);
        let out = ctx.alloc::<f64>(xs.len());
        let f = cube_math::function::$cube::new();
        timed(concat!($name, " [device]"), xs.len(), || {
            unsafe { f.eval_f64_into(ctx, &inp, &out, xs.len()) };
            let _ = ctx.client.read_one(out.clone());
        });
    }};
}

fn cpu_baselines(xs: &[f64]) {
    let mut sink = vec![0.0f64; xs.len()];
    println!("--- CPU baselines (scalar libm, and rmath's SIMD) ---");
    timed("libm exp", xs.len(), || {
        for (o, x) in sink.iter_mut().zip(xs) {
            *o = x.exp();
        }
    });
    timed("rmath exp (bit-exact)", xs.len(), || {
        rmath::Exp::new().eval_slice(xs, &mut sink);
    });
    timed("libm ln", xs.len(), || {
        for (o, x) in sink.iter_mut().zip(xs) {
            *o = x.abs().ln();
        }
    });
    timed("rmath ln (bit-exact)", xs.len(), || {
        rmath::Ln::new().eval_slice(xs, &mut sink);
    });
    timed("libm pow", xs.len(), || {
        for (o, x) in sink.iter_mut().zip(xs) {
            *o = x.abs().powf(2.5);
        }
    });
}

fn bench<R: Runtime>(name: &str) {
    let ctx = Ctx::<R>::new(&Default::default());
    println!(
        "\n--- {name}: f64 {} ---",
        ctx.fidelity.f64.summary(),
    );
    if !ctx.fidelity.f64.usable {
        println!("(no usable f64 on this backend)");
        return;
    }
    if !ctx.fidelity.f64.bit_exact_capable() {
        println!("(not bit-exact capable; the numbers below are the Fast policy's)");
    }

    let wide = data(-700.0, 700.0);
    let positive = data(1e-8, 1e8);

    bench_one!(&ctx, &wide, "exp", Exp, Exp);
    bench_one!(&ctx, &wide, "exp2", Exp2, Exp2);
    bench_one!(&ctx, &data(-300.0, 300.0), "exp10", Exp10, Exp10);
    bench_one!(&ctx, &positive, "ln", Ln, Ln);
    bench_one!(&ctx, &positive, "log2", Log2, Log2);
    bench_one!(&ctx, &positive, "log10", Log10, Log10);
    bench_one!(&ctx, &data(-1.0, 1.0), "expm1", Expm1, Expm1);
    bench_one!(&ctx, &data(-0.9, 10.0), "log1p", Log1p, Log1p);
    bench_one!(&ctx, &positive, "cbrt", Cbrt, Cbrt);
    bench_one!(&ctx, &positive, "sqrt", Sqrt, Sqrt);
    bench_one!(&ctx, &wide, "floor", Floor, Floor);
    bench_one!(&ctx, &wide, "rint", Rint, Rint);

    // The two-argument ones.
    let a = data(1e-6, 1e6);
    let b = data(-8.0, 8.0);
    let ah = ctx.upload(&a);
    let bh = ctx.upload(&b);
    let out = ctx.alloc::<f64>(N);
    let p = cube_math::function::Pow::new();
    timed("pow [device]", N, || {
        unsafe { p.eval_f64_into(&ctx, &ah, &bh, &out, N) };
        let _ = ctx.client.read_one(out.clone());
    });
    let h = cube_math::function::Hypot::new();
    timed("hypot [device]", N, || {
        unsafe { h.eval_f64_into(&ctx, &ah, &bh, &out, N) };
        let _ = ctx.client.read_one(out.clone());
    });
    let m = cube_math::function::Fmod::new();
    timed("fmod [device]", N, || {
        unsafe { m.eval_f64_into(&ctx, &ah, &bh, &out, N) };
        let _ = ctx.client.read_one(out.clone());
    });

    // What a round trip through host memory costs, for contrast.
    let f = cube_math::function::Exp::new();
    timed("exp [via host Vec]", N, || {
        let _ = f.eval_f64(&ctx, &wide);
    });
}

fn main() {
    cpu_baselines(&data(-700.0, 700.0));
    #[cfg(feature = "cpu")]
    bench::<cubecl::cpu::CpuRuntime>("CubeCL CPU runtime");
    #[cfg(feature = "wgpu")]
    bench::<cubecl::wgpu::WgpuRuntime>("wgpu");
}
