//! Throughput, against `rmath` on the CPU and against the platform `libm`.

use std::time::Instant;

use cube_math::prelude::*;
use cubecl::prelude::Runtime;
use rmath::prelude::*;

const N: usize = 1 << 20;

fn data() -> Vec<f64> {
    (0..N).map(|i| -700.0 + 1400.0 * (i as f64) / (N as f64)).collect()
}

fn bench<R: Runtime>(name: &str, xs: &[f64]) {
    let ctx = Ctx::<R>::new(&Default::default());
    if !ctx.fidelity.f64.usable {
        println!("{name}: no f64");
        return;
    }
    let inp = ctx.upload(xs);
    let out = ctx.alloc::<f64>(xs.len());

    for (label, f) in [
        ("exact", cube_math::function::Exp::new()),
        ("fast", cube_math::function::Exp::fast()),
    ] {
        // Warm up: the first launch compiles the kernel.
        unsafe { f.eval_f64_into(&ctx, &inp, &out, xs.len()) };
        let _ = ctx.client.read_one(out.clone());

        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            unsafe { f.eval_f64_into(&ctx, &inp, &out, xs.len()) };
        }
        let _ = ctx.client.read_one(out.clone());
        let el = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{name:>6} exp {label:<6} {:>8.3} ms  {:>8.1} Melem/s",
            el * 1e3,
            xs.len() as f64 / el / 1e6
        );
    }
}

fn main() {
    let xs = data();
    let mut sink = vec![0.0f64; xs.len()];

    for (label, run) in [
        ("libm", &(|xs: &[f64], out: &mut [f64]| {
            for (o, x) in out.iter_mut().zip(xs) {
                *o = x.exp();
            }
        }) as &dyn Fn(&[f64], &mut [f64])),
        ("rmath-exact", &|xs: &[f64], out: &mut [f64]| {
            rmath::Exp::new().eval_slice(xs, out);
        }),
        ("rmath-fast", &|xs: &[f64], out: &mut [f64]| {
            rmath::Exp::builder().accuracy(rmath::Fast).build().eval_slice(xs, out);
        }),
    ] {
        run(&xs, &mut sink);
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            run(&xs, &mut sink);
        }
        let el = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "{label:>12} {:>8.3} ms  {:>8.1} Melem/s",
            el * 1e3,
            xs.len() as f64 / el / 1e6
        );
    }

    #[cfg(feature = "cpu")]
    bench::<cubecl::cpu::CpuRuntime>("cpu", &xs);
    #[cfg(feature = "wgpu")]
    bench::<cubecl::wgpu::WgpuRuntime>("wgpu", &xs);
}
