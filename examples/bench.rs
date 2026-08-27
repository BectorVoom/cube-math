//! Throughput, against `rmath` on the CPU and against the platform `libm`.
//!
//! ```text
//! export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
//! export HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off
//! RUSTFLAGS="-C target-cpu=native" cargo run --release \
//!     --features "cpu,hip" --example bench
//! ```
//!
//! `target-cpu=native` matters for the two CPU baselines and not at all for
//! this crate: `rmath` and `libm` are compiled ahead of time and will not use
//! the wide instructions unless told to, while a CubeCL kernel is compiled for
//! the machine it is about to run on either way. Without the flag `rmath`
//! measures about a tenth of its real speed and the comparison is meaningless.
//!
//! `HIPRTC_COMPILE_OPTIONS_APPEND` matters for correctness, not speed —
//! see [`cube_math::probe`].
//!
//! # What is being timed
//!
//! Kernel throughput on data that is already on the device. That is the honest
//! number for a kernel and the dishonest one for a workload: a single `exp`
//! over a host `Vec` is dominated by the two transfers, and no amount of
//! arithmetic speed changes that. The last row measures exactly that, for
//! contrast.

use std::time::Instant;

use cube_math::launch::{Binary, F64, Unary, binary, unary};
use cube_math::prelude::*;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use rmath::prelude::*;

const N: usize = 1 << 20;
const REPS: usize = 20;

fn data(lo: f64, hi: f64) -> Vec<f64> {
    (0..N).map(|i| lo + (hi - lo) * (i as f64) / (N as f64)).collect()
}

fn timed(label: &str, n: usize, mut run: impl FnMut()) {
    run();
    let t = Instant::now();
    for _ in 0..REPS {
        run();
    }
    let el = t.elapsed().as_secs_f64() / REPS as f64;
    println!("{label:<26} {:>8.3} ms  {:>9.1} Melem/s", el * 1e3, n as f64 / el / 1e6);
}

/// Wait for the queue to drain, without copying anything back.
///
/// Reading the output buffer would work too, and would measure the wrong
/// thing: eight megabytes over the host link every iteration swamps the
/// arithmetic and turns a throughput benchmark into a transfer benchmark.
fn wait<R: Runtime>(client: &ComputeClient<R>) {
    cubecl::future::block_on(client.sync()).unwrap();
}

fn cpu_baselines(xs: &[f64]) {
    let mut sink = vec![0.0f64; xs.len()];
    println!("--- CPU baselines: scalar libm, and rmath's SIMD ---");
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

fn upload<R: Runtime>(client: &ComputeClient<R>, xs: &[f64]) -> TensorHandle<R> {
    let h = client.create(cubecl::bytes::Bytes::from_elems(xs.to_vec()));
    TensorHandle::new_contiguous(vec![xs.len()], h, F64)
}

fn bench<R: Runtime>(name: &str) {
    let client = R::client(&Default::default());
    let fid = fidelity(&client);
    println!("\n--- {name}: f64 {} ---", fid.f64.summary());
    if !fid.f64.usable {
        println!("(no usable f64 on this backend)");
        return;
    }
    if !fid.f64.bit_exact_capable() {
        println!("(NOT bit-exact capable; these are the Fast policy's numbers)");
    }
    let cfg = MathConfig::new(
        if fid.f64.bit_exact_capable() { Policy::EXACT } else { Policy::FAST },
        fid.f64.fma_kind(),
    );

    let wide = data(-700.0, 700.0);
    let positive = data(1e-8, 1e8);

    let one = |label: &str, op: Unary, xs: &[f64]| {
        let input = upload(&client, xs);
        let output = TensorHandle::<R>::empty(&client, vec![xs.len()], F64);
        timed(label, xs.len(), || {
            unary(&client, op, input.clone().binding(), output.clone().binding(), F64, cfg).unwrap();
            wait(&client);
        });
    };
    one("exp", Unary::Exp, &wide);
    one("exp2", Unary::Exp2, &wide);
    one("exp10", Unary::Exp10, &data(-300.0, 300.0));
    one("ln", Unary::Ln, &positive);
    one("log2", Unary::Log2, &positive);
    one("log10", Unary::Log10, &positive);
    one("expm1", Unary::Expm1, &data(-1.0, 1.0));
    one("log1p", Unary::Log1p, &data(-0.9, 10.0));
    one("cbrt", Unary::Cbrt, &positive);
    one("sin", Unary::Sin, &data(-100.0, 100.0));
    one("cos", Unary::Cos, &data(-100.0, 100.0));
    one("tan", Unary::Tan, &data(-100.0, 100.0));
    one("sin (branred)", Unary::Sin, &data(1e9, 1e18));
    one("asin", Unary::Asin, &data(-1.0, 1.0));
    one("atan", Unary::Atan, &wide);
    one("sinh", Unary::Sinh, &data(-700.0, 700.0));
    one("tanh", Unary::Tanh, &data(-20.0, 20.0));
    one("asinh", Unary::Asinh, &wide);
    one("acosh", Unary::Acosh, &data(1.0, 1e18));
    one("erf", Unary::Erf, &data(-6.0, 6.0));
    one("erfc", Unary::Erfc, &data(-6.0, 27.0));
    one("lgamma", Unary::LGamma, &data(-50.0, 170.0));
    one("tgamma", Unary::TGamma, &data(-20.0, 170.0));
    one("j0", Unary::J0, &data(0.0, 100.0));
    one("y1", Unary::Y1, &data(0.0, 100.0));
    one("sqrt", Unary::Sqrt, &positive);
    one("rint", Unary::Rint, &wide);

    let a = upload(&client, &data(1e-6, 1e6));
    let b = upload(&client, &data(-8.0, 8.0));
    let output = TensorHandle::<R>::empty(&client, vec![N], F64);
    let two = |label: &str, op: Binary| {
        timed(label, N, || {
            binary(
                &client,
                op,
                a.clone().binding(),
                b.clone().binding(),
                output.clone().binding(),
                F64,
                cfg,
            )
            .unwrap();
            wait(&client);
        });
    };
    two("pow", Binary::Pow);
    two("hypot", Binary::Hypot);
    two("atan2", Binary::Atan2);
    two("fmod", Binary::Fmod);

    // What a round trip through host memory costs, for contrast. This one
    // *does* include both transfers, on purpose.
    timed("exp [upload+read]", N, || {
        let input = upload(&client, &wide);
        let output = TensorHandle::<R>::empty(&client, vec![N], F64);
        let h = output.handle.clone();
        unary(&client, Unary::Exp, input.binding(), output.binding(), F64, cfg).unwrap();
        let _ = client.read_one(h);
    });
}

fn main() {
    cpu_baselines(&data(-700.0, 700.0));
    #[cfg(feature = "cpu")]
    bench::<cubecl::cpu::CpuRuntime>("CubeCL CPU runtime");
    #[cfg(feature = "hip")]
    bench::<cubecl::hip::HipRuntime>("ROCm / HIP");
    #[cfg(feature = "wgpu")]
    bench::<cubecl::wgpu::WgpuRuntime>("wgpu");
}
