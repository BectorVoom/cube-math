//! The software fused multiply-add, checked against the hardware one.
//!
//! [`cube_math::cube::fma::fma_f64`] exists because the SPIR-V backend lowers
//! `fma` to a separate multiply and add, and the bit-exact kernels cannot do
//! without a fused one. It is only worth having if it is *exactly* right, so
//! this compares it against `f64::mul_add` — the hardware instruction on the
//! host — over the cases that break naive emulations: ties, cancellation,
//! subnormal results, overflow, and operands at the ends of the range.
//!
//! It runs on a CubeCL runtime rather than by calling the function directly,
//! because a `#[cube]` function is not callable as ordinary Rust — its bit
//! reinterpretations have no native implementation.

mod harness;

use cube_math::fma::{fma_f32, fma_f64};
use cubecl::prelude::*;
use harness::{Rng, same_f32, same_f64};

#[cube(launch_unchecked)]
fn k64(a: &Array<f64>, b: &Array<f64>, c: &Array<f64>, out: &mut Array<f64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = fma_f64(a[ABSOLUTE_POS], b[ABSOLUTE_POS], c[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
fn k32(a: &Array<f32>, b: &Array<f32>, c: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = fma_f32(a[ABSOLUTE_POS], b[ABSOLUTE_POS], c[ABSOLUTE_POS]);
    }
}

fn run64<R: Runtime>(client: &ComputeClient<R>, a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    let n = a.len();
    let ah = client.create(cubecl::bytes::Bytes::from_elems(a.to_vec()));
    let bh = client.create(cubecl::bytes::Bytes::from_elems(b.to_vec()));
    let ch = client.create(cubecl::bytes::Bytes::from_elems(c.to_vec()));
    let oh = client.empty(n * 8);
    unsafe {
        k64::launch_unchecked::<R>(
            client,
            CubeCount::Static(n.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(ah, n),
            ArrayArg::from_raw_parts(bh, n),
            ArrayArg::from_raw_parts(ch, n),
            ArrayArg::from_raw_parts(oh.clone(), n),
        );
    }
    let bytes = client.read_one(oh).unwrap();
    f64::from_bytes(&bytes)[..n].to_vec()
}

fn run32<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    let n = a.len();
    let ah = client.create(cubecl::bytes::Bytes::from_elems(a.to_vec()));
    let bh = client.create(cubecl::bytes::Bytes::from_elems(b.to_vec()));
    let ch = client.create(cubecl::bytes::Bytes::from_elems(c.to_vec()));
    let oh = client.empty(n * 4);
    unsafe {
        k32::launch_unchecked::<R>(
            client,
            CubeCount::Static(n.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(ah, n),
            ArrayArg::from_raw_parts(bh, n),
            ArrayArg::from_raw_parts(ch, n),
            ArrayArg::from_raw_parts(oh.clone(), n),
        );
    }
    let bytes = client.read_one(oh).unwrap();
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// The triples the emulation has to survive.
fn cases() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |x: f64, y: f64, z: f64| {
        a.push(x);
        b.push(y);
        c.push(z);
    };

    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(2),
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::EPSILON,
        1.0 + f64::EPSILON,
        1.0 - f64::EPSILON / 2.0,
        (2.0f64).powi(-537),
        (2.0f64).powi(537),
        (2.0f64).powi(-1000),
        (2.0f64).powi(1000),
        (2.0f64).powi(-1074),
        std::f64::consts::PI,
    ];
    for &x in &specials {
        for &y in &specials {
            for &z in &specials {
                push(x, y, z);
            }
        }
    }

    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    // Whole-range random bit patterns.
    for _ in 0..300_000 {
        push(
            f64::from_bits(rng.next()),
            f64::from_bits(rng.next()),
            f64::from_bits(rng.next()),
        );
    }
    // Ordinary magnitudes: where a polynomial evaluation actually lives.
    let ord = |r: &mut Rng| {
        let m = (r.next() >> 12) as f64 / (1u64 << 52) as f64 + 1.0;
        let e = (r.next() % 41) as i32 - 20;
        let sign = if r.next() & 1 == 0 { 1.0 } else { -1.0 };
        sign * m * (2.0f64).powi(e)
    };
    for _ in 0..300_000 {
        let (x, y, z) = (ord(&mut rng), ord(&mut rng), ord(&mut rng));
        push(x, y, z);
    }
    // Near-total cancellation: the case that exists only because the
    // multiply-add is fused, and is exactly zero for an unfused one.
    for _ in 0..200_000 {
        let x = 1.0 + (rng.next() >> 12) as f64 / (1u64 << 52) as f64;
        let y = 1.0 + (rng.next() >> 12) as f64 / (1u64 << 52) as f64;
        push(x, y, -(x * y));
        push(x, y, -(x * y) * 1.5);
        push(-x, y, x * y);
    }
    // Products that land in the subnormal range, where rounding twice costs
    // half an ulp and the scale-back has to be the only rounding.
    for _ in 0..200_000 {
        let x = f64::from_bits((rng.next() & 0x000f_ffff_ffff_ffff) | (0x1ffu64 << 52));
        let y = f64::from_bits((rng.next() & 0x000f_ffff_ffff_ffff) | (0x1ffu64 << 52));
        let z = f64::from_bits(rng.next() & 0x001f_ffff_ffff_ffff);
        push(x, y, z);
        push(x, y, -z);
    }
    (a, b, c)
}

fn check<R: Runtime>(backend: &str) {
    let client = R::client(&Default::default());

    // The emulation is built from error-free transformations — two-sum and
    // Dekker's split — and those are identities only where the backend does
    // not reassociate. On a backend that does, `fma_f64` is not merely slow,
    // it is wrong, and `Fidelity` already says so by refusing to call the
    // precision bit-exact-capable. Testing it there would be testing something
    // the crate does not claim.
    let fidelity = cube_math::fidelity(&client);
    if !fidelity.f64.usable || !fidelity.f64.stable_arithmetic {
        eprintln!(
            "[{backend}] skipping: the software multiply-add needs arithmetic this \
             backend rewrites ({})",
            fidelity.f64.summary(),
        );
        return;
    }
    let (a, b, c) = cases();
    let got = run64(&client, &a, &b, &c);
    let mut bad = Vec::new();
    let mut count = 0usize;
    for i in 0..a.len() {
        let want = a[i].mul_add(b[i], c[i]);
        if !same_f64(got[i], want) {
            count += 1;
            if bad.len() < 10 {
                bad.push(format!(
                    "  fma({:e}, {:e}, {:e}) = {:#018x}, want {:#018x}",
                    a[i],
                    b[i],
                    c[i],
                    got[i].to_bits(),
                    want.to_bits(),
                ));
            }
        }
    }
    assert!(bad.is_empty(), "[{backend}] {count} of {} triples differ\n{}", a.len(), bad.join("\n"));

    // Single precision, on random bit patterns and the specials.
    let mut rng = Rng(0x0f0f_0f0f_f0f0_f0f0);
    let mut a32 = Vec::new();
    let mut b32 = Vec::new();
    let mut c32 = Vec::new();
    for _ in 0..500_000 {
        a32.push(f32::from_bits(rng.next() as u32));
        b32.push(f32::from_bits(rng.next() as u32));
        c32.push(f32::from_bits(rng.next() as u32));
    }
    let sp32 = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0 + f32::EPSILON,
    ];
    for &x in &sp32 {
        for &y in &sp32 {
            for &z in &sp32 {
                a32.push(x);
                b32.push(y);
                c32.push(z);
            }
        }
    }
    let got32 = run32(&client, &a32, &b32, &c32);
    let mut bad32 = Vec::new();
    for i in 0..a32.len() {
        let want = a32[i].mul_add(b32[i], c32[i]);
        if !same_f32(got32[i], want) && bad32.len() < 10 {
            bad32.push(format!(
                "  fma_f32({:e}, {:e}, {:e}) = {:#010x}, want {:#010x}",
                a32[i],
                b32[i],
                c32[i],
                got32[i].to_bits(),
                want.to_bits(),
            ));
        }
    }
    assert!(bad32.is_empty(), "[{backend}] single precision\n{}", bad32.join("\n"));
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime() {
    check::<cubecl::cpu::CpuRuntime>("cpu");
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_runtime() {
    check::<cubecl::wgpu::WgpuRuntime>("wgpu");
}

#[cfg(feature = "hip")]
#[test]
fn hip_runtime() {
    check::<cubecl::hip::HipRuntime>("hip");
}
