//! Scalar against vector, on the same data, for every `*_vec` entry point.
//!
//! ```text
//! export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
//! export HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off
//! cargo run --release --features "cpu,hip" --example bench_vec
//! ```
//!
//! # What is being timed, and what the answer means
//!
//! One kernel over an array already on the device, launched once per element
//! for the scalar row and once per N elements for the vector rows — so the
//! rows differ in the width of the arithmetic *and* in the number of threads,
//! which is the whole point. A vector entry point exists to let one thread
//! carry N points; if that does not beat N threads carrying one point each,
//! it has no reason to exist on that device.
//!
//! The two answers this reports are different answers, and both are real:
//!
//! * On the **CPU runtime** the win is large and grows with width, because a
//!   `Vector<f64, 8>` is what the machine's vector registers were built for
//!   and the scalar kernel leaves them empty.
//! * On a **GPU** it is close to flat. A GPU is already SIMT — the scalar
//!   kernel's threads *are* the lanes — so widening buys no parallelism that
//!   was not there, costs registers, and can cost occupancy. Width 2 is often
//!   a small win on a memory-bound kernel and wider is often a small loss.
//!
//! Neither of those is a defect. The vector entry points are for a caller who
//! already holds a `Vector<f64, N>` — a grid collocation, an unrolled stencil
//! — and whose alternative is extracting each element, calling the scalar
//! routine and inserting the result. This benchmark measures the floor of what
//! that caller saves, not the ceiling.

use std::time::Instant;

use cube_math::prelude::*;
use cube_math::{MathConfig, double as m};
use cubecl::prelude::*;

const N: usize = 1 << 22;
const REPS: usize = 20;

/// Threads per cube. The same for both rows, so the comparison is between the
/// kernels and not between two launch geometries.
const BLOCK: usize = 256;

fn timed(label: &str, n: usize, mut run: impl FnMut()) -> f64 {
    run();
    let t = Instant::now();
    for _ in 0..REPS {
        run();
    }
    let el = t.elapsed().as_secs_f64() / REPS as f64;
    println!(
        "  {label:<22} {:>8.3} ms  {:>9.1} Melem/s",
        el * 1e3,
        n as f64 / el / 1e6
    );
    el
}

/// The scalar and vector kernels for one function, and the row that times
/// them.
///
/// Generated rather than written six times, for the reason `tests/vector.rs`
/// gives: the plumbing is identical and only the pair of `#[cube]` functions
/// varies.
macro_rules! bench_case {
    ($name:ident, $module:ident, $scalar:ident, $vector:ident) => {
        mod $name {
            use super::*;

            #[cube(launch_unchecked)]
            fn scalar_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] cfg: MathConfig) {
                if ABSOLUTE_POS < x.len() {
                    out[ABSOLUTE_POS] = m::$module::$scalar(x[ABSOLUTE_POS], cfg);
                }
            }

            #[cube(launch_unchecked)]
            fn vector_kernel<N: Size>(
                x: &Array<Vector<f64, N>>,
                out: &mut Array<Vector<f64, N>>,
                #[comptime] cfg: MathConfig,
            ) {
                if ABSOLUTE_POS < x.len() {
                    out[ABSOLUTE_POS] = m::$module::$vector::<N>(x[ABSOLUTE_POS], cfg);
                }
            }

            pub fn row<R: Runtime>(client: &ComputeClient<R>, xs: &[f64], cfg: MathConfig) {
                let n = xs.len();
                let input = client.create_from_slice(f64::as_bytes(xs));
                let output = client.empty(n * 8);
                println!("{}:", stringify!($scalar));
                let base = timed("scalar", n, || {
                    unsafe {
                        scalar_kernel::launch_unchecked::<R>(
                            client,
                            CubeCount::Static(n.div_ceil(BLOCK) as u32, 1, 1),
                            CubeDim::new_1d(BLOCK as u32),
                            ArrayArg::from_raw_parts(input.clone(), n),
                            ArrayArg::from_raw_parts(output.clone(), n),
                            cfg,
                        );
                    }
                    cubecl::future::block_on(client.sync()).unwrap();
                });
                for line in [2usize, 4, 8] {
                    let lanes = n / line;
                    let el = timed(&format!("vector width {line}"), n, || {
                        unsafe {
                            vector_kernel::launch_unchecked::<R>(
                                client,
                                CubeCount::Static(lanes.div_ceil(BLOCK) as u32, 1, 1),
                                CubeDim::new_1d(BLOCK as u32),
                                line,
                                ArrayArg::from_raw_parts(input.clone(), n),
                                ArrayArg::from_raw_parts(output.clone(), n),
                                cfg,
                            );
                        }
                        cubecl::future::block_on(client.sync()).unwrap();
                    });
                    println!("  {:<22} {:>8.2}x", format!("^ speedup"), base / el);
                }
            }
        }
    };
}

bench_case!(v_exp, exp, exp, exp_vec);
bench_case!(v_exp2, exp2, exp2, exp2_vec);
bench_case!(v_exp10, exp10, exp10, exp10_vec);
bench_case!(v_ln, ln, ln, ln_vec);
bench_case!(v_log2, logx, log2, log2_vec);
bench_case!(v_log10, logx, log10, log10_vec);

fn data(lo: f64, hi: f64) -> Vec<f64> {
    (0..N)
        .map(|i| lo + (hi - lo) * (i as f64) / (N as f64))
        .collect()
}

fn bench<R: Runtime>(name: &str) {
    let client = R::client(&Default::default());
    let fid = fidelity(&client);
    println!("\n=== {name}: f64 {} ===", fid.f64.summary());
    if !fid.f64.usable {
        println!("(no usable f64 on this backend)");
        return;
    }
    if !fid.f64.bit_exact_capable() {
        println!("(NOT bit-exact capable; these are the Fast policy's numbers)");
    }
    let cfg = MathConfig::new(
        if fid.f64.bit_exact_capable() {
            Policy::EXACT
        } else {
            Policy::FAST
        },
        fid.f64.fma_kind(),
    );

    // Each function on data that keeps it on its main path, so the rows
    // measure the vectorised schedule and not the per-element repair.
    v_exp::row(&client, &data(-700.0, 700.0), cfg);
    v_exp2::row(&client, &data(-900.0, 900.0), cfg);
    v_exp10::row(&client, &data(-250.0, 250.0), cfg);
    v_ln::row(&client, &data(1e-8, 1e8), cfg);
    v_log2::row(&client, &data(1e-8, 1e8), cfg);
    v_log10::row(&client, &data(1e-8, 1e8), cfg);
}

fn main() {
    #[cfg(feature = "cpu")]
    bench::<cubecl::cpu::CpuRuntime>("cpu");
    #[cfg(feature = "hip")]
    bench::<cubecl::hip::HipRuntime>("hip");
    #[cfg(feature = "cuda")]
    bench::<cubecl::cuda::CudaRuntime>("cuda");
}
