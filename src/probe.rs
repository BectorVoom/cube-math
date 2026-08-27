//! What the device actually does with floating point.
//!
//! Bit-exactness is a claim about a *device*, not about this source. Three
//! properties have to hold for the kernels' claim to survive, none is
//! guaranteed by CubeCL, and each of them fails on some backend in wide use.
//! So rather than assume, [`Fidelity::measure`] runs kernels whose answers
//! differ depending on each one, and reports what it found.
//!
//! ```no_run
//! # use cube_math::probe::Fidelity;
//! # use cubecl::prelude::*;
//! # fn f<R: Runtime>(client: &ComputeClient<R>) {
//! let fidelity = Fidelity::measure(client);
//! assert!(fidelity.f64.bit_exact_capable(), "{fidelity:#?}");
//! # }
//! ```
//!
//! [`fidelity`] does this once, and [`crate::MathConfig::exact_for`] uses the answer to pick
//! [`FmaKind`] per precision, so the default configuration is correct
//! everywhere and pays for emulation only where emulation is needed.
//!
//! # What is measured, and what it costs to fail
//!
//! | property | if false | recoverable? |
//! |---|---|---|
//! | `usable` | the precision does not run here at all | no |
//! | `fused_fma` | `fma` rounds twice | yes — [`FmaKind::Software`] |
//! | `separate_mul_add` | `a * b + c` is contracted | no |
//! | `subnormals` | subnormals are flushed to zero | no |
//!
//! The two irrecoverable ones are not pessimism. A contracted `a * b + c`
//! breaks every schedule glibc compiled *without* `-mfma` — `exp2` is one —
//! and there is no way to express "do not fuse this" through CubeCL. Flushed
//! subnormals change results near the bottom of the range in a way no
//! rearrangement recovers.

use cubecl::prelude::*;

use crate::config::MathConfig;
use crate::fma::FmaKind;
use crate::policy::Policy;

/// What one precision does on this device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Precision {
    /// This precision runs here at all.
    ///
    /// False for `f64` on the WGSL path without `SHADER_F64`, and on WebGPU
    /// generally.
    pub usable: bool,
    /// `fma(a, b, c)` is a true fused multiply-add — **one** rounding.
    ///
    /// False on the SPIR-V backend, which lowers it to a separate multiply and
    /// add. Recoverable: see [`FmaKind::Software`].
    pub fused_fma: bool,
    /// `a * b + c` is *not* contracted into a fused multiply-add.
    pub separate_mul_add: bool,
    /// Subnormals survive arithmetic instead of being flushed to zero.
    pub subnormals: bool,
    /// `(a + b) - a` is not folded to `b`.
    ///
    /// An algebraic identity over the reals, and false over floating point:
    /// with `a = 1` and `b = 2^-53` the sum rounds back to `1`, so the
    /// difference is `0`, not `b`. A backend that rewrites it has enabled
    /// reassociation, and every exact-arithmetic algorithm — two-sum, Dekker
    /// splitting, and so the whole software multiply-add — is unusable there.
    ///
    /// This is not hypothetical: the AMD Vulkan driver on the machine this was
    /// developed on does exactly this rewrite. Combined with the SPIR-V
    /// backend's unfused `fma`, it puts bit-exact double precision out of
    /// reach on that path — the emulation that would rescue the missing `fma`
    /// is itself built from the identities the driver rewrites.
    pub stable_arithmetic: bool,
    /// A real kernel got a known answer right.
    ///
    /// The mechanical probes above each test one property, and a backend can
    /// pass all of them and still be wrong — `wgpu`'s WGSL path advertises
    /// `f64`, passes every probe here, and then evaluates `exp` to something
    /// that is not `exp`. So the last word is a whole kernel run on inputs
    /// whose correctly-rounded answers are mathematical constants, checked
    /// bit for bit. Set by [`Fidelity::measure`], after the mechanical probes.
    pub verified: bool,
}

impl Precision {
    /// Whether [`crate::Accuracy::BitExact`] can be honoured for this
    /// precision, given the right [`FmaKind`].
    ///
    /// The multiply-add is the one recoverable failure — but only where the
    /// emulation that recovers it can itself be trusted, which is what
    /// `stable_arithmetic` decides.
    pub fn bit_exact_capable(self) -> bool {
        self.usable
            && self.verified
            && self.separate_mul_add
            && self.subnormals
            && (self.fused_fma || self.stable_arithmetic)
    }

    /// Which multiply-add the kernels should be built with here.
    pub const fn fma_kind(self) -> FmaKind {
        if self.fused_fma { FmaKind::Hardware } else { FmaKind::Software }
    }

    /// A one-line summary, for a test log or a failure message.
    pub fn summary(self) -> String {
        if !self.usable {
            return "unusable".into();
        }
        let mut s = String::new();
        s.push_str(if self.fused_fma { "fused-fma" } else { "SPLIT-FMA" });
        s.push_str(if self.separate_mul_add { ", no-contract" } else { ", CONTRACTS" });
        s.push_str(if self.subnormals { ", subnormals" } else { ", FLUSHES-SUBNORMALS" });
        s.push_str(if self.stable_arithmetic { ", no-reassoc" } else { ", REASSOCIATES" });
        s.push_str(if self.verified { ", canary-ok" } else { ", CANARY-FAILED" });
        s
    }
}

/// Measure what `client`'s device does with the operations bit-exactness
/// depends on.
///
/// A free function taking a client, in the shape the rest of the ecosystem
/// uses. Runs several small kernels, so call it once per device and keep the
/// answer.
pub fn fidelity<R: Runtime>(client: &ComputeClient<R>) -> Fidelity {
    Fidelity::measure(client)
}

/// What a device does with the operations bit-exactness depends on, per
/// precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Fidelity {
    /// Double precision.
    pub f64: Precision,
    /// Single precision.
    pub f32: Precision,
}

impl Fidelity {
    /// Whether `f64` runs here at all. A shorthand for `self.f64.usable`.
    pub fn has_f64(self) -> bool {
        self.f64.usable
    }

    /// Run the probe kernels on `client` and report what they found.
    pub fn measure<R: Runtime>(client: &ComputeClient<R>) -> Self {
        let mut out = Self { f64: measure_f64(client), f32: measure_f32(client) };
        if out.f64.usable {
            let (exact, approx) = canary_f64(client, out.f64.fma_kind());
            out.f64.verified = exact;
            // A backend that cannot get even the *approximate* answer right is
            // not a backend with a precision caveat, it is a backend where
            // `f64` does not work.
            out.f64.usable = approx;
        }
        if out.f32.usable {
            out.f32.verified = canary_f32(client);
        }
        out
    }
}

/// `[a*b+c, fma(a,b,c), tiny*4, sentinel]`.
///
/// The sentinel is not a formality. A backend that cannot compile the kernel
/// does not always say so — `wgpu` on the WGSL path leaves the output buffer
/// untouched and returns success — so the probe has to *exercise* every
/// instruction the real kernels need and then report that it got that far.
/// WGSL is exactly the case that motivates this: it advertises `f64` and does
/// have `f64` arithmetic, but has no `f64` `sqrt`, `floor` or integer
/// conversion, so a kernel that reached only for multiply and add would
/// conclude the precision was usable and then produce silent zeros.
#[cube(launch_unchecked)]
fn probe_k64(inp: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>) {
    if ABSOLUTE_POS < 1usize {
        let a = inp[0];
        let b = inp[1];
        let c = inp[2];
        out[0] = a * b + c;
        out[1] = fma(a, b, c);
        out[2] = inp[3] * 4.0f64;
        // `(a + b) - a`, which is `0` here and `b` on a backend that
        // reassociates. `a` is 1 and `b` is `2^-53`, so the sum ties and
        // rounds back to `a` exactly.
        let s = inp[4] + inp[5];
        out[4] = s - inp[4];

        let bits = u64::reinterpret(a);
        let idx = usize::cast_from(bits >> 62u64);
        let e = u32::cast_from(bits >> 52u64) & 0x7ffu32;
        let k = i32::cast_from(e) - 1023i32;
        let v = f64::reinterpret(bits + 1u64)
            + f64::cast_from(e)
            + f64::cast_from(k)
            + f64::floor(a)
            + f64::sqrt(a)
            + f64::abs(c)
            + inp[idx]
            // The table arena is a `u64` buffer, and reading one is its own
            // capability: WGSL needs an extension for 64-bit storage, and a
            // backend can have `f64` arithmetic without it.
            + f64::reinterpret(tab[idx] | 0x3ff0_0000_0000_0000u64);
        out[3] = select(v == v, 1.0f64, 2.0f64);
    }
}

/// The single-precision counterpart.
#[cube(launch_unchecked)]
fn probe_k32(inp: &Array<f32>, out: &mut Array<f32>, tab: &Array<u64>) {
    if ABSOLUTE_POS < 1usize {
        let a = inp[0];
        let b = inp[1];
        let c = inp[2];
        out[0] = a * b + c;
        out[1] = fma(a, b, c);
        out[2] = inp[3] * 4.0f32;
        let s = inp[4] + inp[5];
        out[4] = s - inp[4];

        let bits = u32::reinterpret(a);
        let idx = usize::cast_from(bits >> 30u32);
        let e = (bits >> 23u32) & 0xffu32;
        let k = i32::cast_from(e) - 127i32;
        let v = f32::reinterpret(bits + 1u32)
            + f32::cast_from(e)
            + f32::cast_from(k)
            + f32::floor(a)
            + f32::sqrt(a)
            + f32::abs(c)
            + inp[idx]
            + f32::cast_from(f64::reinterpret(tab[idx] | 0x3ff0_0000_0000_0000u64));
        out[3] = select(v == v, 1.0f32, 2.0f32);
    }
}

/// `a` and `b` chosen so that their exact product is `1 + 2^-53 - 2^-105`,
/// which rounds to exactly 1 — so `a * b - 1` is zero unfused and
/// `2^-53 - 2^-105` fused. One input pair separates both properties.
fn measure_f64<R: Runtime>(client: &ComputeClient<R>) -> Precision {
    let a = 1.0 + f64::EPSILON;
    let b = 1.0 - f64::EPSILON / 2.0;
    let input = vec![a, b, -1.0, f64::from_bits(1), 1.0, f64::EPSILON / 2.0];
    let n = input.len();
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(input));
    let out_h = client.empty(8 * size_of::<f64>());
    let tab_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u64; 4]));
    unsafe {
        probe_k64::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(in_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), 8),
            ArrayArg::from_raw_parts(tab_h, 4),
        );
    }
    let Ok(bytes) = client.read_one(out_h) else {
        return Precision::default();
    };
    let o = f64::from_bytes(&bytes);
    // A backend that cannot compile the kernel does not always report an
    // error — `wgpu` on the WGSL path simply leaves the buffer untouched — so
    // the only reliable test is whether the kernel wrote its sentinel.
    if o[3] != 1.0 {
        return Precision::default();
    }
    Precision {
        usable: true,
        fused_fma: o[1].to_bits() == a.mul_add(b, -1.0).to_bits(),
        separate_mul_add: o[0] == 0.0,
        subnormals: o[2].to_bits() == f64::from_bits(4).to_bits(),
        stable_arithmetic: o[4] == 0.0,
        // Filled in by `Ctx`, which is the first thing that can run a real
        // kernel.
        verified: false,
    }
}

/// The same test one precision down: `a * b` is `1 + 2^-24 - 2^-47`, which
/// rounds to 1.
fn measure_f32<R: Runtime>(client: &ComputeClient<R>) -> Precision {
    let a = 1.0f32 + f32::EPSILON;
    let b = 1.0f32 - f32::EPSILON / 2.0;
    let input = vec![a, b, -1.0f32, f32::from_bits(1), 1.0, f32::EPSILON / 2.0];
    let n = input.len();
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(input));
    let out_h = client.empty(8 * size_of::<f32>());
    let tab_h = client.create(cubecl::bytes::Bytes::from_elems(vec![0u64; 4]));
    unsafe {
        probe_k32::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(in_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), 8),
            ArrayArg::from_raw_parts(tab_h, 4),
        );
    }
    let Ok(bytes) = client.read_one(out_h) else {
        return Precision::default();
    };
    let o = f32::from_bytes(&bytes);
    if o[3] != 1.0 {
        return Precision::default();
    }
    Precision {
        usable: true,
        fused_fma: o[1].to_bits() == a.mul_add(b, -1.0).to_bits(),
        separate_mul_add: o[0] == 0.0,
        subnormals: o[2].to_bits() == f32::from_bits(4).to_bits(),
        stable_arithmetic: o[4] == 0.0,
        verified: false,
    }
}

/// `exp` at five points, both policies, and a sentinel.
#[cube(launch_unchecked)]
fn canary_k64(inp: &Array<f64>, out: &mut Array<f64>, #[comptime] fk: FmaKind) {
    if ABSOLUTE_POS < inp.len() {
        let x = inp[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = crate::double::exp::exp(x, comptime!(MathConfig::new(Policy::EXACT, fk)));
        out[ABSOLUTE_POS + 8] =
            crate::double::exp::exp(x, comptime!(MathConfig::new(Policy::FAST, fk)));
    }
}

/// A correctly-rounded square root, a tie that rounds to even, and a subnormal
/// that survives being scaled.
#[cube(launch_unchecked)]
fn canary_k32(inp: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS < inp.len() {
        let cfg = comptime!(MathConfig::new(Policy::EXACT, FmaKind::Hardware));
        out[ABSOLUTE_POS] = crate::single::exact::sqrt(inp[ABSOLUTE_POS], cfg);
        out[ABSOLUTE_POS + 8] = crate::single::exact::rint(inp[ABSOLUTE_POS], cfg);
    }
}

/// Evaluate a real kernel on inputs whose correctly-rounded answers are
/// mathematical constants, and check them bit for bit.
///
/// Returns `(bit_exact_ok, approximate_ok)`.
///
/// The mechanical probes above each test one property, and a backend can pass
/// all of them and still be wrong. `wgpu`'s WGSL path is the case in point: it
/// advertises `f64`, has `f64` arithmetic, passes every probe — and then
/// evaluates `exp(1)` to a number that is not `e`. So the last word belongs to
/// a whole kernel.
///
/// The expected values are the correctly rounded `f64` nearest to `e^x`, which
/// is a fact about mathematics rather than about a `libm`, so the canary does
/// not smuggle in a platform assumption. `exp` is the right canary because it
/// exercises everything at once: the constant table, the integer exponent
/// surgery, and a chain of eight multiply-adds whose answer changes if any of
/// them is not fused.
fn canary_f64<R: Runtime>(client: &ComputeClient<R>, fk: FmaKind) -> (bool, bool) {
    let xs = vec![1.0f64, -1.0, 0.5, f64::EPSILON, 20.0];
    let want = [
        0x4005_bf0a_8b14_5769u64, // e
        0x3fd7_8b56_362c_ef38,    // 1/e
        0x3ffa_6129_8e1e_069c,    // sqrt(e)
        0x3ff0_0000_0000_0001,    // exp(2^-52), which rounds to 1 + 2^-52
        0x41bc_eb08_8b68_e804,    // e^20
    ];
    let n = xs.len();
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(xs));
    let out_h = client.empty(16 * size_of::<f64>());
    unsafe {
        canary_k64::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(in_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), 16),
            fk,
        );
    }
    let Ok(bytes) = client.read_one(out_h) else {
        return (false, false);
    };
    let got = f64::from_bytes(&bytes);
    let ok = |base: usize, tol: f64| {
        (0..n).all(|i| {
            let w = f64::from_bits(want[i]);
            (got[base + i] - w).abs() <= tol * w.abs()
        })
    };
    // The approximate arm allows a generous relative error — it is asking
    // "does this backend compute `exp` at all", not "how accurately".
    (ok(0, 0.0), ok(8, 1e-12))
}

/// The single-precision canary. See [`canary_f64`].
///
/// `exp` has no `f32` kernel yet, so this leans on the exact family, whose
/// answers are pinned by IEEE-754 rather than by any `libm`.
fn canary_f32<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let xs = vec![2.0f32, 0.5, 2.5, f32::from_bits(1)];
    let n = xs.len();
    let in_h = client.create(cubecl::bytes::Bytes::from_elems(xs));
    let out_h = client.empty(16 * size_of::<f32>());
    unsafe {
        canary_k32::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(in_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), 16),
        );
    }
    let Ok(bytes) = client.read_one(out_h) else {
        return false;
    };
    let g = f32::from_bytes(&bytes);
    g[0].to_bits() == 2.0f32.sqrt().to_bits()
        && g[3].to_bits() == f32::from_bits(1).sqrt().to_bits()
        && g[9].to_bits() == 0.0f32.to_bits()
        && g[10].to_bits() == 2.0f32.to_bits()
}
