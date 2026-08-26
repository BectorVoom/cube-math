//! The software fused multiply-add, checked against the hardware one.
//!
//! `cube_math::cube::fma::fma_f64` exists because some backends do not have a
//! fused multiply-add and the bit-exact kernels cannot do without one. It is
//! only worth having if it is *exactly* right, so this compares it against
//! `f64::mul_add` — which on this host is the hardware instruction — over the
//! cases that break naive emulations: ties, cancellation, subnormal results,
//! overflow, and operands at the ends of the range.

use cube_math::cube::fma::{fma_f32, fma_f64};

/// A tiny xorshift, so the sweep is reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(self.next())
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.next() as u32)
    }
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn check(a: f64, b: f64, c: f64) {
    let want = a.mul_add(b, c);
    let got = fma_f64(a, b, c);
    assert!(
        same(got, want),
        "fma_f64({a:e}, {b:e}, {c:e}) = {got:e} ({:#018x}), want {want:e} ({:#018x})",
        got.to_bits(),
        want.to_bits(),
    );
}

#[test]
fn random_bit_patterns() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..2_000_000 {
        check(rng.f64(), rng.f64(), rng.f64());
    }
}

#[test]
fn ordinary_magnitudes() {
    // Where the kernels actually live: everything within a few octaves of 1,
    // which is the regime a polynomial evaluation stays in.
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..2_000_000 {
        let s = |r: &mut Rng| {
            let m = (r.next() >> 12) as f64 / (1u64 << 52) as f64 + 1.0;
            let e = (r.next() % 41) as i32 - 20;
            let sign = if r.next() & 1 == 0 { 1.0 } else { -1.0 };
            sign * m * (2.0f64).powi(e)
        };
        check(s(&mut rng), s(&mut rng), s(&mut rng));
    }
}

#[test]
fn near_cancellation() {
    // `fma(a, b, -a*b)` is the residual of the product: the case that exists
    // only because the multiply-add is fused, and zero for an unfused one.
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    for _ in 0..1_000_000 {
        let a = 1.0 + (rng.next() >> 12) as f64 / (1u64 << 52) as f64;
        let b = 1.0 + (rng.next() >> 12) as f64 / (1u64 << 52) as f64;
        check(a, b, -(a * b));
        check(a, b, -(a * b) * 1.5);
        check(-a, b, a * b);
    }
}

#[test]
fn ties_and_boundaries() {
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
    for &a in &specials {
        for &b in &specials {
            for &c in &specials {
                check(a, b, c);
            }
        }
    }
}

#[test]
fn subnormal_results() {
    // Products that land in the subnormal range, where rounding twice costs
    // half an ulp and the scale-back has to be the only rounding.
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for _ in 0..500_000 {
        let a = f64::from_bits((rng.next() & 0x000f_ffff_ffff_ffff) | (0x1ffu64 << 52));
        let b = f64::from_bits((rng.next() & 0x000f_ffff_ffff_ffff) | (0x1ffu64 << 52));
        let c = f64::from_bits(rng.next() & 0x001f_ffff_ffff_ffff);
        check(a, b, c);
        check(a, b, -c);
    }
}

#[test]
fn single_precision() {
    let mut rng = Rng(0x0f0f_0f0f_f0f0_f0f0);
    for _ in 0..2_000_000 {
        let (a, b, c) = (rng.f32(), rng.f32(), rng.f32());
        let want = a.mul_add(b, c);
        let got = fma_f32(a, b, c);
        assert!(
            same32(got, want),
            "fma_f32({a:e}, {b:e}, {c:e}) = {got:e}, want {want:e}",
        );
    }
    for a in [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::from_bits(1), f32::MAX, f32::INFINITY, f32::NAN] {
        for b in [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::from_bits(1), f32::MAX, f32::INFINITY, f32::NAN] {
            for c in [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::from_bits(1), f32::MAX, f32::INFINITY, f32::NAN] {
                let want = a.mul_add(b, c);
                let got = fma_f32(a, b, c);
                assert!(same32(got, want), "fma_f32({a:e}, {b:e}, {c:e}) = {got:e}, want {want:e}");
            }
        }
    }
}
