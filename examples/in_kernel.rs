//! Calling `cube-math` from inside your own kernel.
//!
//! ```text
//! export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
//! export HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off
//! cargo run --release --features "cpu,hip" --example in_kernel
//! ```
//!
//! This is what the crate is arranged for. The Gaussian log-density below
//! needs a logarithm and an exponential in the middle of an expression; doing
//! that with elementwise launches would mean four passes over memory and three
//! temporaries, and the arithmetic is not what would cost you.
//!
//! Note what the kernel signature does *not* contain: any table, buffer or
//! context belonging to `cube-math`. The math functions are functions.

use cube_math::launch::CpuShape;
use cube_math::{MathConfig, double as m, fidelity};
use cubecl::prelude::*;

/// `ln N(x | mu, sigma)`, summed with a soft weight — one pass, no temporaries.
///
/// `#[comptime] cfg` is the only thing threaded through, and it is the same
/// comptime configuration every `cube-math` function takes: what accuracy you
/// want, and what this device's multiply-add actually does.
#[cube(launch_unchecked)]
fn gaussian_logpdf(
    x: &Array<f64>,
    out: &mut Array<f64>,
    mu: f64,
    sigma: f64,
    #[comptime] cfg: MathConfig,
) {
    if ABSOLUTE_POS < x.len() {
        let z = (x[ABSOLUTE_POS] - mu) / sigma;
        // `ln(sigma)` and `-z^2/2`, then a weight that needs the exponential
        // back again — the sort of expression that is a single kernel here and
        // four launches otherwise.
        let log_norm = m::ln::ln(sigma, cfg) + 0.5 * m::ln::ln(TWO_PI, cfg);
        let logpdf = -0.5 * z * z - log_norm;
        let weight = m::exp::exp(-0.25 * z * z, cfg);
        out[ABSOLUTE_POS] = logpdf * weight;
    }
}

const TWO_PI: f64 = std::f64::consts::TAU;

fn run<R: Runtime>(name: &str) {
    let client = R::client(&Default::default());
    let fid = fidelity(&client);
    if !fid.f64.usable {
        println!("{name}: no usable f64");
        return;
    }
    let cfg = MathConfig::new(cube_math::Policy::EXACT, fid.f64.fma_kind());

    let n = 1 << 16;
    let (mu, sigma) = (0.5f64, 2.0f64);
    let xs: Vec<f64> = (0..n).map(|i| -8.0 + 16.0 * i as f64 / n as f64).collect();

    // `create_from_slice` reads the host slice straight into the device
    // buffer. `Bytes::from_elems` wants an owned `Vec` instead, so a slice has
    // to be cloned onto the heap to be handed over and the copy dropped again
    // the moment the transfer is done.
    let inp = client.create_from_slice(f64::as_bytes(&xs));
    let out = client.empty(n * size_of::<f64>());
    // `ln` and `exp` are both straight-line here, so on the CPU runtime this
    // body is one LLVM will vectorise and the geometry should leave it alone —
    // see `CpuShape`. On a GPU the parameter is ignored and the workgroup
    // comes from the hardware's plane size.
    let (count, dim) = cube_math::launch::launch_1d(&client, n, CpuShape::Vectorised);
    unsafe {
        gaussian_logpdf::launch_unchecked::<R>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts(inp, n),
            ArrayArg::from_raw_parts(out.clone(), n),
            mu,
            sigma,
            cfg,
        );
    }
    let bytes = client.read_one(out).unwrap();
    let got = f64::from_bytes(&bytes);

    // The same expression on the host, through `rmath` — which is bit-exact to
    // the platform `libm`. Every operation between the math calls is plain
    // IEEE arithmetic, so if the math agrees the whole expression agrees, bit
    // for bit, and that is what this checks.
    use rmath::prelude::*;
    let mut worst = 0usize;
    for (i, x) in xs.iter().enumerate() {
        let z = (x - mu) / sigma;
        let log_norm = rmath::Ln::new().eval(sigma) + 0.5 * rmath::Ln::new().eval(TWO_PI);
        let logpdf = -0.5 * z * z - log_norm;
        let weight = rmath::Exp::new().eval(-0.25 * z * z);
        if got[i].to_bits() != (logpdf * weight).to_bits() {
            worst += 1;
        }
    }
    if worst == 0 {
        println!("{name}: {n} values, bit-identical to the host expression");
    } else {
        println!("{name}: {worst} of {n} differ ({})", fid.f64.summary());
    }
}

fn main() {
    #[cfg(feature = "cpu")]
    run::<cubecl::cpu::CpuRuntime>("cpu ");
    #[cfg(feature = "hip")]
    run::<cubecl::hip::HipRuntime>("hip ");
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime>("wgpu");
}
