//! `exp_vec` is `exp`, element for element.
//!
//! The vector entry point's contract is bit-identity with the scalar
//! routine at every width, on every policy, over the whole domain — so the
//! comparison is the scalar kernel's output against the vector kernel's, at
//! `to_bits()`, over the same sweep the equivalence suite uses (which already
//! pins the scalar routine to `rmath`, hence to glibc).

mod harness;

use cube_math::prelude::*;
use cube_math::{MathConfig, double as m};
use cubecl::prelude::*;
use harness::sweep_f64;

#[cube(launch_unchecked)]
fn scalar_exp_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] cfg: MathConfig) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = m::exp::exp(x[ABSOLUTE_POS], cfg);
    }
}

#[cube(launch_unchecked)]
fn vector_exp_kernel<N: Size>(
    x: &Array<Vector<f64, N>>,
    out: &mut Array<Vector<f64, N>>,
    #[comptime] cfg: MathConfig,
) {
    if ABSOLUTE_POS < x.len() {
        out[ABSOLUTE_POS] = m::exp::exp_vec::<N>(x[ABSOLUTE_POS], cfg);
    }
}

fn run_scalar<R: Runtime>(client: &ComputeClient<R>, xs: &[f64], cfg: MathConfig) -> Vec<f64> {
    let input = client.create_from_slice(f64::as_bytes(xs));
    let output = client.empty(xs.len() * 8);
    let n = xs.len();
    unsafe {
        scalar_exp_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output.clone(), n),
            cfg,
        );
    }
    f64::from_bytes(&client.read_one(output).expect("read"))[..n].to_vec()
}

fn run_vector<R: Runtime>(
    client: &ComputeClient<R>,
    xs: &[f64],
    line: usize,
    cfg: MathConfig,
) -> Vec<f64> {
    assert!(xs.len().is_multiple_of(line));
    let input = client.create_from_slice(f64::as_bytes(xs));
    let output = client.empty(xs.len() * 8);
    let n = xs.len();
    let lanes = n / line;
    unsafe {
        vector_exp_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(lanes.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            line,
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output.clone(), n),
            cfg,
        );
    }
    f64::from_bytes(&client.read_one(output).expect("read"))[..n].to_vec()
}

fn suite<R: Runtime>(backend: &str, client: &ComputeClient<R>) {
    let fid = fidelity(client);
    if !fid.f64.usable {
        eprintln!("[{backend}] f64 does not run on this backend; skipping");
        return;
    }
    let mut xs = sweep_f64(709.9);
    // The branches the vector path repairs per element.
    xs.extend_from_slice(&[
        0.0,
        -0.0,
        1e-300,
        -1e-300,
        f64::from_bits(1),
        512.0,
        -512.0,
        700.0,
        -700.0,
        -745.13,
        -746.0,
        709.78,
        710.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MAX,
        f64::MIN,
    ]);
    while !xs.len().is_multiple_of(8) {
        xs.push(0.5);
    }
    let cfgs = [
        ("exact", MathConfig::new(Policy::EXACT, fid.f64.fma_kind())),
        ("fast", MathConfig::new(Policy::FAST, fid.f64.fma_kind())),
        (
            "exact_finite",
            MathConfig::new(Policy::EXACT_FINITE, fid.f64.fma_kind()),
        ),
    ];
    for (name, cfg) in cfgs {
        let scalar = run_scalar(client, &xs, cfg);
        for line in [1usize, 2, 4, 8] {
            let vector = run_vector(client, &xs, line, cfg);
            let mut bad = 0usize;
            for (i, (&s, &v)) in scalar.iter().zip(&vector).enumerate() {
                if s.to_bits() != v.to_bits() {
                    if bad < 5 {
                        eprintln!(
                            "[{backend}] {name} width {line}: x = {:e} scalar {:e} ({:#018x}) vector {:e} ({:#018x})",
                            xs[i],
                            s,
                            s.to_bits(),
                            v,
                            v.to_bits()
                        );
                    }
                    bad += 1;
                }
            }
            assert_eq!(
                bad,
                0,
                "[{backend}] exp_vec {name} at width {line}: {bad} of {} elements differ from exp",
                xs.len()
            );
            eprintln!(
                "[{backend}] exp_vec {name} width {line}: bit-identical to exp over {} inputs",
                xs.len()
            );
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime() {
    let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    suite("cpu", &client);
}

#[cfg(feature = "hip")]
#[test]
fn hip_runtime() {
    let client = cubecl::hip::HipRuntime::client(&Default::default());
    suite("hip", &client);
}
