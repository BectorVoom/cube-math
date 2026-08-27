//! Shared machinery for the equivalence suite.
//!
//! Runs each case on every runtime the build enabled, so a single `cargo test`
//! covers the CPU runtime and the GPU, and a backend that quietly rewrites the
//! arithmetic shows up as a failure rather than as a footnote.

#![allow(dead_code)]

use cube_math::prelude::*;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;

/// Put a slice on the device as a flat contiguous tensor.
pub fn upload<R: Runtime, E: CubeElement>(
    client: &ComputeClient<R>,
    xs: &[E],
    dtype: cubecl::prelude::StorageType,
) -> TensorHandle<R> {
    let handle = client.create(cubecl::bytes::Bytes::from_elems(xs.to_vec()));
    TensorHandle::new_contiguous(vec![xs.len()], handle, dtype)
}

/// Reserve a flat contiguous tensor of `n` elements.
pub fn empty<R: Runtime>(
    client: &ComputeClient<R>,
    n: usize,
    dtype: cubecl::prelude::StorageType,
) -> TensorHandle<R> {
    TensorHandle::empty(client, vec![n], dtype)
}

/// Read a tensor back.
pub fn download<R: Runtime, E: CubeElement + Clone>(
    client: &ComputeClient<R>,
    t: TensorHandle<R>,
    n: usize,
) -> Vec<E> {
    let bytes = client.read_one(t.handle).expect("device read failed");
    E::from_bytes(&bytes)[..n].to_vec()
}

/// Evaluate a one-argument op over a slice, the long way round.
pub fn eval1<R: Runtime, E: CubeElement + Clone>(
    client: &ComputeClient<R>,
    op: Unary,
    xs: &[E],
    dtype: cubecl::prelude::StorageType,
    cfg: MathConfig,
) -> Vec<E> {
    let input = upload(client, xs, dtype);
    let output = empty(client, xs.len(), dtype);
    let out = output.handle.clone();
    cube_math::launch::unary(client, op, input.binding(), output.binding(), dtype, cfg)
        .expect("launch failed");
    download(client, TensorHandle::new_contiguous(vec![xs.len()], out, dtype), xs.len())
}

/// Evaluate a two-argument op over two slices.
pub fn eval2<R: Runtime, E: CubeElement + Clone>(
    client: &ComputeClient<R>,
    op: Binary,
    a: &[E],
    b: &[E],
    dtype: cubecl::prelude::StorageType,
    cfg: MathConfig,
) -> Vec<E> {
    let ah = upload(client, a, dtype);
    let bh = upload(client, b, dtype);
    let output = empty(client, a.len(), dtype);
    let out = output.handle.clone();
    cube_math::launch::binary(client, op, ah.binding(), bh.binding(), output.binding(), dtype, cfg)
        .expect("launch failed");
    download(client, TensorHandle::new_contiguous(vec![a.len()], out, dtype), a.len())
}

/// Evaluate a one-argument, two-output op over a slice.
pub fn eval1_pair<R: Runtime, E: CubeElement + Clone>(
    client: &ComputeClient<R>,
    op: UnaryPair,
    xs: &[E],
    dtype: cubecl::prelude::StorageType,
    cfg: MathConfig,
) -> (Vec<E>, Vec<E>) {
    let input = upload(client, xs, dtype);
    let o1 = empty(client, xs.len(), dtype);
    let o2 = empty(client, xs.len(), dtype);
    let (h1, h2) = (o1.handle.clone(), o2.handle.clone());
    cube_math::launch::unary_pair(
        client, op, input.binding(), o1.binding(), o2.binding(), dtype, cfg,
    )
    .expect("launch failed");
    (
        download(client, TensorHandle::new_contiguous(vec![xs.len()], h1, dtype), xs.len()),
        download(client, TensorHandle::new_contiguous(vec![xs.len()], h2, dtype), xs.len()),
    )
}

/// Evaluate a two-argument, two-output op over two slices.
pub fn eval2_pair<R: Runtime, E: CubeElement + Clone>(
    client: &ComputeClient<R>,
    op: BinaryPair,
    a: &[E],
    b: &[E],
    dtype: cubecl::prelude::StorageType,
    cfg: MathConfig,
) -> (Vec<E>, Vec<E>) {
    let ah = upload(client, a, dtype);
    let bh = upload(client, b, dtype);
    let o1 = empty(client, a.len(), dtype);
    let o2 = empty(client, a.len(), dtype);
    let (h1, h2) = (o1.handle.clone(), o2.handle.clone());
    cube_math::launch::binary_pair(
        client, op, ah.binding(), bh.binding(), o1.binding(), o2.binding(), dtype, cfg,
    )
    .expect("launch failed");
    (
        download(client, TensorHandle::new_contiguous(vec![a.len()], h1, dtype), a.len()),
        download(client, TensorHandle::new_contiguous(vec![a.len()], h2, dtype), a.len()),
    )
}

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

/// The two element types, behind one interface, so every comparison below is
/// written once.
///
/// The suite checks `f32` and `f64` with the same cases and the same failure
/// reports; without this the harness would be the same six functions twice,
/// which is exactly the kind of duplication that lets one copy drift.
pub trait Elem: Copy + PartialEq + std::fmt::LowerExp {
    /// A hex rendering of the bit pattern, for failure messages.
    fn hex(self) -> String;
    /// True when the value is NaN.
    fn nan(self) -> bool;
    /// True when the value is finite.
    fn finite(self) -> bool;
    /// Distance from `other` in units of the last place of the larger.
    fn ulps_from(self, other: Self) -> f64;
}

impl Elem for f64 {
    fn hex(self) -> String {
        format!("{:#018x}", self.to_bits())
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn ulps_from(self, other: Self) -> f64 {
        ulp_diff(self, other)
    }
}

impl Elem for f32 {
    fn hex(self) -> String {
        format!("{:#010x}", self.to_bits())
    }
    fn nan(self) -> bool {
        self.is_nan()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn ulps_from(self, other: Self) -> f64 {
        ulp_diff(self as f64, other as f64) / (1u64 << 29) as f64
    }
}

/// True when two results are the same number, counting all NaNs as equal.
///
/// NaN payloads are deliberately *not* compared. glibc does not promise one,
/// the reference implementations propagate whichever the arithmetic produced,
/// and a device is free to canonicalise — so requiring payload equality would
/// be testing something nobody guarantees.
pub fn same<E: Elem>(a: E, b: E) -> bool {
    (a.nan() && b.nan()) || a.hex() == b.hex()
}

/// Report at most eight differing inputs, then fail.
///
/// Eight rather than one: when a schedule is wrong it is usually wrong on a
/// whole band of inputs, and seeing the band is what identifies which branch
/// is at fault.
fn verdict(backend: &str, name: &str, total: usize, count: usize, bad: Vec<String>) {
    assert!(
        bad.is_empty(),
        "[{backend}] {name}: {count} of {total} inputs differ from the reference\n{}",
        bad.join("\n"),
    );
}

/// Compare a one-argument kernel against a scalar reference.
pub fn check<E: Elem>(
    backend: &str,
    name: &str,
    device: impl Fn(&[E]) -> Vec<E>,
    reference: impl Fn(E) -> E,
    xs: &[E],
) {
    let got = device(xs);
    assert_eq!(got.len(), xs.len(), "[{backend}] {name}: wrong output length");
    let (mut bad, mut count) = (Vec::new(), 0usize);
    for (x, &g) in xs.iter().zip(got.iter()) {
        let want = reference(*x);
        if !same(g, want) {
            count += 1;
            if bad.len() < 8 {
                bad.push(format!(
                    "  x = {x:e} ({})  got {g:e} ({})  want {want:e} ({})",
                    x.hex(),
                    g.hex(),
                    want.hex(),
                ));
            }
        }
    }
    verdict(backend, name, xs.len(), count, bad);
}

/// Compare a two-argument kernel against a scalar reference.
pub fn check2<E: Elem>(
    backend: &str,
    name: &str,
    device: impl Fn(&[E], &[E]) -> Vec<E>,
    reference: impl Fn(E, E) -> E,
    a: &[E],
    b: &[E],
) {
    let got = device(a, b);
    let (mut bad, mut count) = (Vec::new(), 0usize);
    for i in 0..a.len() {
        let want = reference(a[i], b[i]);
        if !same(got[i], want) {
            count += 1;
            if bad.len() < 8 {
                bad.push(format!(
                    "  x = {:e} ({}), y = {:e} ({})  got {}  want {}",
                    a[i],
                    a[i].hex(),
                    b[i],
                    b[i].hex(),
                    got[i].hex(),
                    want.hex(),
                ));
            }
        }
    }
    verdict(backend, name, a.len(), count, bad);
}

/// Compare a pair-returning kernel against a scalar reference.
pub fn check_pair<E: Elem>(
    backend: &str,
    name: &str,
    device: impl Fn(&[E]) -> (Vec<E>, Vec<E>),
    reference: impl Fn(E) -> (E, E),
    xs: &[E],
) {
    let (g1, g2) = device(xs);
    let (mut bad, mut count) = (Vec::new(), 0usize);
    for (i, x) in xs.iter().enumerate() {
        let (w1, w2) = reference(*x);
        if !same(g1[i], w1) || !same(g2[i], w2) {
            count += 1;
            if bad.len() < 8 {
                bad.push(format!(
                    "  x = {x:e} ({})  got ({}, {})  want ({}, {})",
                    x.hex(),
                    g1[i].hex(),
                    g2[i].hex(),
                    w1.hex(),
                    w2.hex(),
                ));
            }
        }
    }
    verdict(backend, name, xs.len(), count, bad);
}

/// Compare a two-argument, pair-returning kernel against a scalar reference.
pub fn check2_pair<E: Elem>(
    backend: &str,
    name: &str,
    device: impl Fn(&[E], &[E]) -> (Vec<E>, Vec<E>),
    reference: impl Fn(E, E) -> (E, E),
    a: &[E],
    b: &[E],
) {
    let (g1, g2) = device(a, b);
    let (mut bad, mut count) = (Vec::new(), 0usize);
    for i in 0..a.len() {
        let (w1, w2) = reference(a[i], b[i]);
        if !same(g1[i], w1) || !same(g2[i], w2) {
            count += 1;
            if bad.len() < 8 {
                bad.push(format!(
                    "  x = {:e}, y = {:e}  got ({}, {})  want ({}, {})",
                    a[i],
                    b[i],
                    g1[i].hex(),
                    g2[i].hex(),
                    w1.hex(),
                    w2.hex(),
                ));
            }
        }
    }
    verdict(backend, name, a.len(), count, bad);
}

/// Check a kernel against a reference to within `limit` ulp.
///
/// What the `Fast` policy is held to: it makes no bit-exactness claim, so the
/// suite checks the bound each kernel documents. Non-finite results still have
/// to agree exactly — an approximation is allowed to be a little off, not to
/// invent an infinity.
pub fn check_ulp<E: Elem>(
    backend: &str,
    name: &str,
    device: impl Fn(&[E]) -> Vec<E>,
    reference: impl Fn(E) -> E,
    xs: &[E],
    limit: f64,
) {
    let got = device(xs);
    let mut worst = 0.0f64;
    let mut at: Option<E> = None;
    let mut bad = Vec::new();
    for (x, &g) in xs.iter().zip(got.iter()) {
        let want = reference(*x);
        // An approximation is allowed to be a little off; it is not allowed to
        // invent an infinity, lose one, or return a NaN. Those are checked
        // exactly, in both directions — an earlier version of this compared
        // only when `want` was non-finite, and an infinity where a subnormal
        // belonged sailed through as a NaN ulp count that never exceeded the
        // budget.
        if !want.finite() || !g.finite() {
            if !same(g, want) && bad.len() < 8 {
                bad.push(format!(
                    "  x = {x:e} ({}): got {g:e} ({}), want {want:e} ({})",
                    x.hex(),
                    g.hex(),
                    want.hex(),
                ));
            }
            continue;
        }
        let u = g.ulps_from(want);
        if u.is_nan() && bad.len() < 8 {
            bad.push(format!("  x = {x:e}: got {g:e}, want {want:e} (not comparable)"));
        }
        if u > worst {
            worst = u;
            at = Some(*x);
        }
    }
    assert!(bad.is_empty(), "[{backend}] {name}: specials differ\n{}", bad.join("\n"));
    assert!(
        worst <= limit,
        "[{backend}] {name}: {worst:.3} ulp at x = {}, budget {limit}",
        at.map(|v| format!("{v:e}")).unwrap_or_default(),
    );
    eprintln!("[{backend}] {name}: {worst:.3} ulp (budget {limit})");
}

/// A pair of argument lists covering the interesting two-argument cases.
///
/// The cross product of a modest set rather than two long parallel sweeps:
/// what breaks a two-argument function is the *relationship* between the
/// arguments — equal magnitudes, exponents far apart, one subnormal and the
/// other not — and only a cross product visits those.
pub fn sweep2_f64() -> (Vec<f64>, Vec<f64>) {
    cross(&sweep2_base_f64())
}

/// The single-precision counterpart of [`sweep2_f64`].
pub fn sweep2_f32() -> (Vec<f32>, Vec<f32>) {
    cross(&sweep2_base_f64().iter().map(|&x| x as f32).collect::<Vec<_>>())
}

fn sweep2_base_f64() -> Vec<f64> {
    let mut base: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        3.0,
        -3.0,
        7.0,
        0.1,
        -0.1,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(3),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        std::f64::consts::PI,
        std::f64::consts::E,
    ];
    for e in [-140i32, -130, -126, -125, -60, -23, -1, 0, 1, 23, 60, 126, 127] {
        let v = (2.0f64).powi(e);
        base.push(v);
        base.push(-v);
        base.push(v * 1.0000001);
    }
    let mut rng = Rng(0x452821e6_38d01377);
    for _ in 0..300 {
        // Random, but inside the single-precision range so the same base
        // serves both sweeps.
        let m = 1.0 + (rng.next() >> 12) as f64 / (1u64 << 52) as f64;
        let e = (rng.next() % 200) as i32 - 100;
        let s = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
        base.push(s * m * (2.0f64).powi(e));
    }
    base
}

fn cross<E: Copy>(base: &[E]) -> (Vec<E>, Vec<E>) {
    let (mut a, mut b) = (Vec::with_capacity(base.len() * base.len()), Vec::new());
    for &x in base {
        for &y in base {
            a.push(x);
            b.push(y);
        }
    }
    (a, b)
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
