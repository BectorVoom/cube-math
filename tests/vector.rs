//! Every `*_vec` entry point is its scalar twin, element for element.
//!
//! The vector entry points' contract is bit-identity with the scalar routine
//! at every width, on every policy, over the whole domain — so the comparison
//! is the scalar kernel's output against the vector kernel's, at `to_bits()`,
//! over the same sweep the equivalence suite uses (which already pins the
//! scalar routines to `rmath`, hence to glibc).
//!
//! What makes this the right test rather than a re-run of the equivalence
//! suite: a vector routine computes its main path on *every* element,
//! including the ones whose answer is thrown away, and then overwrites the
//! ones that belong to another branch. Getting that repair predicate wrong is
//! the failure mode, and it shows up only as a disagreement with the scalar
//! routine on inputs near a branch boundary. So the sweeps below carry every
//! boundary each function has.

mod harness;

use cube_math::prelude::*;
use cube_math::{MathConfig, double as m};
use cubecl::prelude::*;
use harness::sweep_f64;

/// One function's pair of kernels and the comparison that drives them.
///
/// Generated rather than written six times: the plumbing is identical and the
/// only thing that varies is which pair of `#[cube]` functions the two kernels
/// call, which is not something a generic can abstract over.
macro_rules! vector_case {
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

            fn run_scalar<R: Runtime>(
                client: &ComputeClient<R>,
                xs: &[f64],
                cfg: MathConfig,
            ) -> Vec<f64> {
                let n = xs.len();
                let input = client.create_from_slice(f64::as_bytes(xs));
                let output = client.empty(n * 8);
                unsafe {
                    scalar_kernel::launch_unchecked::<R>(
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
                let n = xs.len();
                let lanes = n / line;
                let input = client.create_from_slice(f64::as_bytes(xs));
                let output = client.empty(n * 8);
                unsafe {
                    vector_kernel::launch_unchecked::<R>(
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

            /// Hold the vector entry point to the scalar one, bit for bit, at
            /// every width and on every policy.
            pub fn check<R: Runtime>(backend: &str, client: &ComputeClient<R>, xs: &[f64]) {
                let fid = fidelity(client);
                let cfgs = [
                    ("exact", Policy::EXACT),
                    ("fast", Policy::FAST),
                    ("exact_finite", Policy::EXACT_FINITE),
                    ("fast_finite", Policy::FAST_FINITE),
                ];
                for (policy_name, policy) in cfgs {
                    let cfg = MathConfig::new(policy, fid.f64.fma_kind());
                    let scalar = run_scalar(client, xs, cfg);
                    for line in [1usize, 2, 4, 8] {
                        let vector = run_vector(client, xs, line, cfg);
                        let mut bad = Vec::new();
                        let mut count = 0usize;
                        for (i, (&s, &v)) in scalar.iter().zip(&vector).enumerate() {
                            // NaN payloads are compared here, unlike in the
                            // equivalence suite: both sides are this crate's
                            // own code on the same device, so any difference
                            // is a difference the vector path introduced.
                            if s.to_bits() != v.to_bits() {
                                count += 1;
                                if bad.len() < 8 {
                                    bad.push(format!(
                                        "  x = {:e} ({:#018x})  scalar {:e} ({:#018x})  vector {:e} ({:#018x})",
                                        xs[i], xs[i].to_bits(),
                                        s, s.to_bits(),
                                        v, v.to_bits(),
                                    ));
                                }
                            }
                        }
                        assert!(
                            bad.is_empty(),
                            "[{backend}] {}_vec {policy_name} at width {line}: {count} of {} inputs differ from {}\n{}",
                            stringify!($scalar),
                            xs.len(),
                            stringify!($scalar),
                            bad.join("\n"),
                        );
                    }
                    eprintln!(
                        "[{backend}] {}_vec {policy_name}: bit-identical to {} at widths 1/2/4/8 over {} inputs",
                        stringify!($scalar),
                        stringify!($scalar),
                        xs.len(),
                    );
                }
            }
        }
    };
}

vector_case!(v_exp, exp, exp, exp_vec);
vector_case!(v_exp2, exp2, exp2, exp2_vec);
vector_case!(v_exp10, exp10, exp10, exp10_vec);
vector_case!(v_ln, ln, ln, ln_vec);
vector_case!(v_log2, logx, log2, log2_vec);
vector_case!(v_log10, logx, log10, log10_vec);

/// Round a sweep up to a multiple of 8, so every width divides it.
///
/// Padded with `0.75` — an ordinary main-path input for every function here,
/// so the padding can never be what makes a case pass.
fn padded(mut xs: Vec<f64>, extra: &[f64]) -> Vec<f64> {
    xs.extend_from_slice(extra);
    while !xs.len().is_multiple_of(8) {
        xs.push(0.75);
    }
    xs
}

/// A dense walk of `[lo, hi]`, for pinning down a branch boundary.
fn walk(lo: f64, hi: f64, steps: usize) -> Vec<f64> {
    (0..=steps)
        .map(|i| lo + (hi - lo) * (i as f64) / (steps as f64))
        .collect()
}

/// The values that sit on a logarithm's branch boundaries.
///
/// Both near-one windows, walked densely, plus the inputs that separate a
/// positive normal from everything else. A logarithm's vector path repairs
/// exactly these, so an off-by-one in the window test shows up here and
/// nowhere else.
fn log_edges() -> Vec<f64> {
    let mut v = walk(0.9, 1.12, 4000);
    v.extend_from_slice(&[
        // `ln`'s window, `[bits(0x3fee...), bits(0x3ff109...))`.
        f64::from_bits(0x3fee000000000000),
        f64::from_bits(0x3fedffffffffffff),
        f64::from_bits(0x3ff1090000000000),
        f64::from_bits(0x3ff108ffffffffff),
        // `log2`'s, which is narrower.
        f64::from_bits(0x3feea4af00000000),
        f64::from_bits(0x3feea4aeffffffff),
        f64::from_bits(0x3ff0b55900000000),
        f64::from_bits(0x3ff0b558ffffffff),
        // Exactly one, where every logarithm owes a signed zero.
        1.0,
        // Powers of the base, where `log2` and `log10` owe an exact integer.
        2.0,
        4.0,
        8.0,
        1024.0,
        0.5,
        0.25,
        10.0,
        100.0,
        1e22,
        // Not a positive normal.
        0.0,
        -0.0,
        -1.0,
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MAX,
        f64::MIN,
    ]);
    v
}

fn suite<R: Runtime>(backend: &str, client: &ComputeClient<R>) {
    let fid = fidelity(client);
    if !fid.f64.usable {
        eprintln!("[{backend}] f64 does not run on this backend; skipping");
        return;
    }

    // `exp`: the tiny answer, the two overflow arms, `specialcase`'s window.
    v_exp::check(
        backend,
        client,
        &padded(
            sweep_f64(709.9),
            &[
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
            ],
        ),
    );

    // `exp2`: 928 is where the one-multiply tail stops and `specialcase`
    // starts; 1024 and -1075 are the range answers on either side.
    v_exp2::check(
        backend,
        client,
        &padded(
            sweep_f64(1080.0),
            &[
                927.0,
                928.0,
                -928.0,
                928.5,
                -928.5,
                1023.0,
                1024.0,
                -1024.0,
                1074.0,
                -1074.0,
                -1075.0,
                -1076.0,
                1023.9999,
                -1075.5,
                f64::from_bits(1),
                1e-300,
                -1e-300,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                -f64::NAN,
            ],
        ),
    );

    // `exp10`: 256 ends the main path; 308.25 and -350 are the range answers.
    v_exp10::check(
        backend,
        client,
        &padded(
            sweep_f64(360.0),
            &[
                255.0,
                256.0,
                -256.0,
                255.9999,
                -255.9999,
                300.0,
                -300.0,
                308.0,
                308.25471555991675,
                308.2547155599168,
                -323.0,
                -324.0,
                -350.0,
                -351.0,
                1e-300,
                -1e-300,
                f64::from_bits(1),
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                -f64::NAN,
            ],
        ),
    );

    let edges = log_edges();
    v_ln::check(backend, client, &padded(sweep_f64(1000.0), &edges));
    v_log2::check(backend, client, &padded(sweep_f64(1000.0), &edges));
    v_log10::check(backend, client, &padded(sweep_f64(1000.0), &edges));
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
