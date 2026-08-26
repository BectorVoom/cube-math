//! Shared machinery for the equivalence suite.
//!
//! Runs each case on every runtime the build enabled, so a single `cargo test`
//! covers the CPU runtime and the GPU, and a backend that quietly rewrites the
//! arithmetic shows up as a failure rather than as a footnote.

#![allow(dead_code)]

use cube_math::prelude::*;
use cubecl::prelude::Runtime;

/// A reproducible xorshift, so a failing sweep can be replayed.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The inputs every double-precision unary sweep is checked on.
///
/// Not a uniform sample: uniform samples miss exactly the values that break
/// these algorithms. This is every special value, every power of two across
/// the whole exponent range and its two neighbours, a dense walk of the
/// interesting interval, and random bit patterns — which between them land on
/// subnormals, branch boundaries, and the ties that separate a correct
/// rounding from a nearly correct one.
pub fn sweep_f64(limit: f64) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(1 << 17);

    v.extend_from_slice(&[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::EPSILON,
        std::f64::consts::PI,
        std::f64::consts::E,
        std::f64::consts::LN_2,
    ]);

    // Powers of two across the range, and their neighbours.
    for e in -1080i32..=1024 {
        let x = (2.0f64).powi(e);
        if x == 0.0 || !x.is_finite() {
            continue;
        }
        for y in [x, next_up(x), next_down(x)] {
            v.push(y);
            v.push(-y);
        }
    }

    // A dense walk of the interval the function actually lives on.
    let steps = 20_000;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        v.push(-limit + 2.0 * limit * t);
    }

    // Random bit patterns, and random values inside the interval.
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    for _ in 0..40_000 {
        v.push(f64::from_bits(rng.next()));
        let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        v.push(-limit + 2.0 * limit * u);
    }

    v
}

/// The single-precision counterpart of [`sweep_f64`].
pub fn sweep_f32(limit: f32) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(1 << 17);
    v.extend_from_slice(&[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::EPSILON,
        std::f32::consts::PI,
    ]);
    for e in -155i32..=128 {
        let x = (2.0f32).powi(e);
        if x == 0.0 || !x.is_finite() {
            continue;
        }
        v.push(x);
        v.push(-x);
    }
    let steps = 20_000;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        v.push(-limit + 2.0 * limit * t);
    }
    let mut rng = Rng(0x13198a2e_03707344);
    for _ in 0..40_000 {
        v.push(f32::from_bits(rng.next() as u32));
    }
    v
}

/// The next representable value above `x`.
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    if x > 0.0 { f64::from_bits(x.to_bits() + 1) } else { f64::from_bits(x.to_bits() - 1) }
}

/// The next representable value below `x`.
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// True when two results are the same number, counting all NaNs as equal.
///
/// NaN payloads are deliberately *not* compared. glibc does not promise one,
/// the reference implementations propagate whichever the arithmetic produced,
/// and a device is free to canonicalise — so requiring payload equality would
/// be testing something no one guarantees.
pub fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// [`same_f64`] for single precision.
pub fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Compare a device kernel against a scalar reference over `xs`.
///
/// Reports up to a handful of differing inputs with both bit patterns, rather
/// than the first one and a count: when a schedule is wrong it is usually
/// wrong on a whole band of inputs, and seeing the band is what identifies
/// which branch is at fault.
pub fn check_f64(
    backend: &str,
    name: &str,
    device: impl Fn(&[f64]) -> Vec<f64>,
    reference: impl Fn(f64) -> f64,
    xs: &[f64],
) {
    let got = device(xs);
    assert_eq!(got.len(), xs.len(), "[{backend}] {name}: wrong output length");
    let mut bad: Vec<String> = Vec::new();
    let mut count = 0usize;
    for (x, &g) in xs.iter().zip(got.iter()) {
        let want = reference(*x);
        if !same_f64(g, want) {
            count += 1;
            if bad.len() < 8 {
                bad.push(format!(
                    "  x = {x:24.17e} ({:#018x})  got {g:24.17e} ({:#018x})  want {want:24.17e} ({:#018x})",
                    x.to_bits(),
                    g.to_bits(),
                    want.to_bits(),
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "[{backend}] {name}: {count} of {} inputs differ from the reference\n{}",
        xs.len(),
        bad.join("\n"),
    );
}

/// [`check_f64`] for single precision.
pub fn check_f32(
    backend: &str,
    name: &str,
    device: impl Fn(&[f32]) -> Vec<f32>,
    reference: impl Fn(f32) -> f32,
    xs: &[f32],
) {
    let got = device(xs);
    assert_eq!(got.len(), xs.len(), "[{backend}] {name}: wrong output length");
    let mut bad: Vec<String> = Vec::new();
    let mut count = 0usize;
    for (x, &g) in xs.iter().zip(got.iter()) {
        let want = reference(*x);
        if !same_f32(g, want) {
            count += 1;
            if bad.len() < 8 {
                bad.push(format!(
                    "  x = {x:17.9e} ({:#010x})  got {g:17.9e} ({:#010x})  want {want:17.9e} ({:#010x})",
                    x.to_bits(),
                    g.to_bits(),
                    want.to_bits(),
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "[{backend}] {name}: {count} of {} inputs differ from the reference\n{}",
        xs.len(),
        bad.join("\n"),
    );
}

/// The largest error, in ulp, between a device kernel and a reference.
///
/// What the `Fast` policy is held to: it makes no bit-exactness claim, so the
/// suite checks the bound each kernel documents instead.
pub fn max_ulp_f64(
    device: impl Fn(&[f64]) -> Vec<f64>,
    reference: impl Fn(f64) -> f64,
    xs: &[f64],
) -> (f64, f64) {
    let got = device(xs);
    let mut worst = 0.0f64;
    let mut at = f64::NAN;
    for (x, &g) in xs.iter().zip(got.iter()) {
        let want = reference(*x);
        if want.is_nan() || g.is_nan() || !want.is_finite() {
            continue;
        }
        let ulp = ulp_diff(g, want);
        if ulp > worst {
            worst = ulp;
            at = *x;
        }
    }
    (worst, at)
}

/// The distance between two finite doubles, in units of the last place of the
/// larger.
pub fn ulp_diff(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    let scale = a.abs().max(b.abs());
    let e = scale.log2().floor() as i32;
    let ulp = (2.0f64).powi(e - 52);
    if ulp == 0.0 { 0.0 } else { (a - b).abs() / ulp }
}

/// Check a kernel against a reference to within `limit` ulp.
///
/// What the `Fast` policy is held to: it makes no bit-exactness claim, so the
/// suite checks the bound each kernel documents. Non-finite results still have
/// to agree exactly — an approximation is allowed to be a little off, not to
/// invent an infinity.
pub fn check_ulp_f64(
    backend: &str,
    name: &str,
    device: impl Fn(&[f64]) -> Vec<f64>,
    reference: impl Fn(f64) -> f64,
    xs: &[f64],
    limit: f64,
) {
    let got = device(xs);
    assert_eq!(got.len(), xs.len(), "[{backend}] {name}: wrong output length");
    let mut worst = 0.0f64;
    let mut at = f64::NAN;
    let mut bad = Vec::new();
    for (x, &g) in xs.iter().zip(got.iter()) {
        let want = reference(*x);
        if !want.is_finite() || want == 0.0 {
            if !same_f64(g, want) && bad.len() < 8 {
                bad.push(format!("  x = {x:e}: got {g:e}, want {want:e} (special)"));
            }
            continue;
        }
        let u = ulp_diff(g, want);
        if u > worst {
            worst = u;
            at = *x;
        }
    }
    assert!(bad.is_empty(), "[{backend}] {name}: specials differ\n{}", bad.join("\n"));
    assert!(
        worst <= limit,
        "[{backend}] {name}: {worst:.3} ulp at x = {at:e}, budget {limit}",
    );
    eprintln!("[{backend}] {name}: {worst:.3} ulp (budget {limit})");
}
