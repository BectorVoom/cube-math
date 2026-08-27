# cube-math

A CubeCL `libm` for `f64` and `f32` — bit-exact to the platform `libm` where
you ask for it, and callable from inside your own kernels.

This is a port of [`rmath`](https://github.com/BectorVoom/rmath), a SIMD `libm`
whose defining property is that its default policy is not "about one ulp" but
*bit-identical to the platform's scalar `libm`*. The port keeps that property
and changes what runs it: instead of a CPU vector register, the parallelism is
a CubeCL launch, so the same functions run on ROCm, CUDA, Vulkan, Metal,
WebGPU and the CPU runtime.

## The device functions are the product

`cube_math::double` and `cube_math::single` hold `#[cube]` functions. Call them
from your own kernel — that is what the crate is for, and it beats launching a
pass of its own over memory:

```rust
use cube_math::{double as m, MathConfig};

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
        let log_norm = m::ln::ln(sigma, cfg) + 0.5 * m::ln::ln(TAU, cfg);
        out[ABSOLUTE_POS] = (-0.5 * z * z - log_norm) * m::exp::exp(-0.25 * z * z, cfg);
    }
}
```

Note what the signature does *not* contain: any table, buffer or context
belonging to `cube-math`. The tables are constant arrays compiled into the
kernel, so a math function is a function and not a resource. `examples/in_kernel.rs`
runs that kernel and checks it against the same expression on the host, bit for
bit.

## The launch side

For when a whole elementwise pass *is* what you want, free functions in the
shape the rest of the ecosystem uses — a client, operands as `TensorBinding`s,
the element type as a runtime `StorageType`, the configuration as a value:

```rust
use cube_math::launch::{unary, Unary, F64};

let out = TensorHandle::<R>::empty(&client, x.shape().to_vec(), F64);
unary(&client, Unary::Exp, x.binding(), out.binding(), F64, MathConfig::exact_for(fid.f64))?;
```

The op is a comptime enum, so `Unary::Exp` compiles a kernel that contains
`exp` and nothing else, and the choice is part of the kernel's identity rather
than a branch inside it.

## What changed in the port, and why it got simpler

`rmath` computes a whole vector on the main path and then *repairs* the lanes
that needed a different branch, because the lanes of a CPU vector register
cannot diverge. A GPU is SIMT: lanes are threads, and a branch is a branch. So
the kernels here are written in the shape of `rmath`'s scalar *reference*
module — the platform algorithm, branches and all — and the parallelism comes
from the launch geometry.

The whole `patch_lanes` / `map_lanes` apparatus disappears with it, and so does
`rmath`'s "delegating" category: the functions whose bit-exact path had to run
one lane at a time on a CPU are ordinary parallel kernels here.

## The two questions, unchanged

`Accuracy` and `Domain` mean what they mean in `rmath`, and are `Policy`'s two
fields. They are *comptime* values, so a kernel is expanded once per policy and
contains only the path you asked for.

## What is ported

Everything. The whole of `libm`'s real-valued double and single precision, in
both precisions, bit-exact to glibc over the equivalence sweep.

| family | functions | `BitExact` | `Fast` |
|---|---|---|---|
| **exact** | `floor` `ceil` `trunc` `round` `rint` `sqrt` `abs` `ilogb` `copysign` `fdim` `fmax` `fmin` `ldexp` `scalbn` `fmod` `remainder` `nextafter` `frexp` `modf` `remquo` | exact by construction, on every device | same code |
| **exponentials** | `exp` `exp2` `exp10` `expm1` | reference schedule | table-free series, ≤ 2 ulp |
| **logarithms** | `ln` `log2` `log10` `log1p` | reference schedule | table-free series, ≤ 3 ulp |
| **algebraic** | `pow` `cbrt` `hypot` | reference schedule | *same code* |
| **trigonometric** | `sin` `cos` `tan` `sincos` | reference schedule, `__branred` included | *same code* |
| **inverse trigonometric** | `asin` `acos` `atan` `atan2` | reference schedule | *same code* |
| **hyperbolic** | `sinh` `cosh` `tanh` | reference schedule | ≤ 4 ulp |
| **inverse hyperbolic** | `asinh` `acosh` `atanh` | `asinh`/`acosh` correctly rounded; `atanh` ≤ 4 ulp | `atanh` ≤ 4 ulp |
| **error** | `erf` `erfc` | correctly rounded | *same code* |
| **gamma** | `lgamma` `lgamma_r` `tgamma` | one algorithm; see below | *same code* |
| **Bessel** | `j0` `j1` `y0` `y1` `jn` `yn` | reference schedule | *same code* |

Several families accept `Fast` and ignore it, and the reason is worth stating
rather than hiding. `pow` multiplies its logarithm by `y`, so a table-free
logarithm accurate to a fifth of an ulp still leaves `x^y` around 200 ulp out
at `|y| = 360` — measured, not assumed; reaching the accuracy `pow` needs means
the double-double table logarithm, and once you have paid for that there is
nothing left for an approximation to save. `cbrt` and `hypot` are already their
own cheap algorithms. The trigonometric families *are* table lookups and short
polynomials, and the reduction — which is the cost, and the part no
approximation can skip without changing which quadrant the answer is in — would
still have to run. `erf` and `erfc` are correctly rounded, which is the whole
point of them.

`lgamma` and `tgamma` are the one family that makes no bit-exactness claim, and
say so: Rust has no `f64::tgamma` or `f64::lgamma`, so there is no call a caller
was already making for `BitExact` to be a claim *about*. Both policies run one
implementation, documented by its measured error. Against `rmath` — which
reaches the same conclusion for the same reason — `tgamma` comes out
bit-identical and `lgamma` within 4 ulp away from its two zeros on the negative
half-line.

### What the port had to add

Three things `rmath` leaves to the platform, because on a CPU there is a
platform to leave them to, and a kernel has none:

* **`__branred`**, the Payne-Hanek reduction `sin`, `cos` and `tan` need past
  `105414350`. `rmath` computes the whole vector by the table algorithm and
  repairs the handful of lanes that landed there by calling the scalar `libm`
  on them one at a time. Ported here, so the band is a real band rather than a
  hole. Its data turned up a trap: `branred.h` declares its own `mp2`, distinct
  from `usncs.h`'s despite the shared name, and using the wrong one leaves the
  reduction a third of an ulp out — a wrong last bit for roughly one argument in
  four across that band.
* **`asinh` and `acosh`'s accurate path.** Both are correctly-rounded
  CORE-MATH routines with a two-tier structure: a double-double evaluation with
  a rounding certificate, and below it a second logarithm to 159 bits plus a
  table of the arguments even that cannot resolve. `rmath` ports the first tier
  and delegates the one input in `2^17` the certificate cannot settle. Both
  tiers are here.
* **`sinf`, `cosf`, `powf`, `atan2f` and the single-precision Bessel family.**
  `rmath`'s single-precision `BitExact` path calls the platform's own `float`
  routine lane by lane, which is exact by construction and — because the
  platform's `float` routines are cheaper than its `double` ones — *faster*
  than widening. Neither is available on a device, so these are schedule ports.

### The one place `BitExact` is not an unconditional claim

The single-precision functions in `cube_math::single::wide` are computed in double
precision and rounded once. That reaches the platform's answer because the
platform computes *those* correctly rounded, and correct rounding is a property
of the answer rather than of the route. Almost always: double rounding fails
where the `f64` result lands within its own error of an `f32` boundary, and
`rmath`'s exhaustive sweep over all `2^32` inputs found three such inputs
across the whole set — one for `log10f`, two for `sinhf`. Rates around one in
four billion, quoted rather than assumed, and stated in that module rather than
buried here.

## Running on ROCm

```text
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off
cargo test --release --features "cpu,hip"
```

**That second line is a correctness requirement, not a tuning knob.** `hiprtc`
defaults to `-ffp-contract=fast`, which fuses `a * b + c` into a single
multiply-add — and several of the reference schedules here are *unfused* on
purpose, because glibc ships no fused variant of them. `exp2`, `exp10`,
`log10`'s final combine and the whole of `hypot` are wrong if the compiler
fuses them. `cubecl-hip` 0.10 hard-codes its `hiprtc` options and offers no way
to add one, so the environment variable is the only lever; `HIPRTC_COMPILE_OPTIONS_APPEND`
is a ROCm feature, not something this crate invents.

You do not have to remember: `fidelity()` measures it, and reports `CONTRACTS`
and `bit_exact_capable() == false` when it is missing.

## Bit-exactness on a device

Every IEEE-754 operation rounds identically on every conforming device, so
replaying a reference schedule on a GPU gives the same bits as replaying it on
a CPU — *provided the backend does not rewrite the arithmetic*. Five things
can, none of them is guaranteed by CubeCL, and every one of them fails on some
backend in wide use. So `cube_math::fidelity(&client)` measures them on the
live device and finishes with a functional canary: a real kernel evaluated on
inputs whose correctly-rounded answers are mathematical constants.

```rust
let fid = cube_math::fidelity(&client);
assert!(fid.f64.bit_exact_capable(), "{}", fid.f64.summary());
let cfg = MathConfig::exact_for(fid.f64);
```

| property | if false | recoverable? |
|---|---|---|
| `usable` | the precision does not work here | no |
| `fused_fma` | `fma` rounds twice | yes — a software multiply-add |
| `separate_mul_add` | `a * b + c` is contracted | no |
| `subnormals` | subnormals are flushed | no |
| `stable_arithmetic` | `(a + b) - a` folds to `b` | no |

### What the probes actually found

These are not hypothetical. On the machine this was developed on — a Radeon
860M (gfx1151) under ROCm 7.1 and Mesa 26.1:

* **ROCm / HIP** passes everything, with `-ffp-contract=off`. This is the
  reference GPU backend.
* **CubeCL's CPU runtime** passes everything.
* **`cubecl-spirv`** (0.10 and 0.11-pre) lowers `Fma` to a separate `OpFMul`
  and `OpFAdd` — two roundings. That is defensible on its own terms:
  GLSL.std.450's `Fma` is only *required* to be fused when it carries the
  `NoContraction` decoration, which cubecl emits neither of. It is the reason
  `cube_math::fma::fma_f64` exists.
* **Mesa RADV** folds `(a + b) - a` to `b` for `f64`. That is an unsafe
  floating-point transform, and it is fatal twice over: it breaks the
  error-free transformations the software multiply-add is built from, *and* it
  breaks the `(x * c + BIG) - BIG` round-to-nearest trick that every Cody-Waite
  argument reduction here opens with. The SPIR-V reaches the driver through
  `create_shader_module_passthrough` and a dump confirms the `OpFAdd`/`OpFSub`
  pair is intact, so the rewrite is the driver's. `RADV_DEBUG=llvm`, `noopt`
  and `nocompute` do not change it.
* **`wgpu`'s WGSL path** advertises `f64`, has `f64` arithmetic, passes every
  mechanical probe — and then evaluates `exp(1)` to something that is not `e`.
  This is why the canary exists: a property probe tests what you thought to
  ask, and a kernel tests what actually happens.

`Fidelity` reports `f64` unusable on both `wgpu` paths, and the suite skips
them rather than claiming a bit-exactness it cannot deliver.

### Three things the C++ backends cannot print

Worth knowing if you write your own `#[cube]` code, because the failure is a
compile error inside a generated kernel rather than anything your source shows:

* A reinterpretation compiles to `reinterpret_cast<T const&>(x)`, which needs
  an **lvalue** — so `f64::reinterpret(SOME_CONSTANT)` does not compile, and
  neither does calling any function that reinterprets its argument with a
  literal. Every public entry point here launders its arguments through
  `bits::opaque64` first, which is why `m::ln::ln(2.0, cfg)` works.
* An **infinity has no literal spelling**, so any expression the optimiser
  folds to one (`f64::INFINITY`, `MAX * 2.0`, `-TWO54 / 0.0`) fails. The
  infinities here are read out of one-element constant arrays.
* `!=` on floats is **ordered** on the CPU runtime and **unordered** on HIP, so
  `x != x` is not a NaN test. Every NaN test here reads the bits.

## Speed

`RUSTFLAGS="-C target-cpu=native" cargo run --release --features "cpu,hip" --example bench`,
one million `f64`, data already resident, AMD Ryzen AI 7 350 with a Radeon
860M, best of four runs — an integrated GPU shares its power budget with the
cores it sits next to, so a single run measures the thermal state as much as
the kernel. The flag matters for the CPU baselines and not for this crate:
`rmath` and `libm` are compiled ahead of time and will not use the wide
instructions unless told to, while a CubeCL kernel is compiled for the machine
it is about to run on either way.

| Melem/s | scalar `libm` | `rmath` (CPU SIMD) | **cube-math, CPU runtime** | **cube-math, ROCm** |
|---|---|---|---|---|
| `exp` | 399 | 635 | **709** | **1857** |
| `ln` | 412 | 514 | **755** | **1410** |
| `pow` | 140 | — | **353** | **746** |
| `sqrt` | — | — | **711** | **2048** |
| `rint` | — | — | **714** | **3611** |
| `fmod` | — | — | **177** | **1861** |

All bit-exact — these are the `BitExact` policy's numbers, not `Fast`'s. The
rest of the set as it stood then lands between 765 (`log1p`) and 1959 (`exp2`)
Melem/s on ROCm.

The families added since — trigonometric, inverse trigonometric, hyperbolic,
error, gamma and Bessel, in both precisions — have **not** been re-measured on
ROCm. The environment they were written in has the same GPU and not the HIP
userspace to reach it with — `/opt/rocm` is absent, so `cargo test --features
hip` cannot link, let alone launch — and a number nobody took is not a number
to print. On the CubeCL CPU runtime, where
the whole suite does run and is bit-exact, they land where their shapes
suggest: the trigonometric family and the inverse hyperbolics around 580-680
Melem/s alongside `exp`'s 729, `erf` at 568, `lgamma` and `tgamma` around 440,
and the two genuinely branchy ones — `erfc` at 266 and the Bessel functions
around 210 — a third of that. `sin` on arguments past `105414350`, where every
thread runs `double::branred`, costs 186.

Two of those numbers are worth reading as design outcomes rather than
measurements. `erfc`'s accurate path runs on roughly one input in thirty
thousand, so a warp pays for it only when one of its threads lands there; the
266 above is a benchmark whose inputs sweep the whole domain uniformly, which
is close to the worst case for that. And the Bessel functions' near-a-zero
repair is exactly the data-dependent branch that stops `rmath` vectorising
them at all — here it costs the thread that takes it and nothing else.

Two caveats worth stating. `fmod`'s shift-and-subtract loop still runs one
iteration per binary digit of the quotient — the same work glibc does — and its
trip count is set by the ratio of the arguments rather than by anything the
kernel controls, so neighbouring threads diverge as badly as their data makes
them. And these are *kernel* numbers, on data already on the device: a
single call that uploads and reads back measures 263 Melem/s, and the
difference is entirely the two transfers. Reach for the GPU when the data is
already there, or when enough work happens per element to pay for the trip.
That is also the argument for calling the device functions from inside your own
kernel rather than launching one of these.

An integrated GPU runs `f64` at a small fraction of its `f32` rate, so the
ROCm column is a floor, not a ceiling.

## Testing

```text
export HIPRTC_COMPILE_OPTIONS_APPEND=-ffp-contract=off
cargo test --release --features "cpu,hip"
```

`tests/equivalence.rs` compares every kernel against `rmath`'s bit-exact
objects — whose own suite pins them to the platform `libm`, so agreement is
established transitively — over a sweep built to land on the values that break
these algorithms: every special value, every power of two across the exponent
range and its two neighbours, a dense walk of the interesting interval, and
random bit patterns. Two-argument functions get a cross product, because what
breaks them is the *relationship* between the arguments; `jn` and `yn` get a
cross product of *orders* against arguments, because what selects their
recurrence is `n` against `x` rather than either alone; and `sin`, `cos` and
`tan` are swept five times over, once per band, because a sweep to `1e300` puts
almost every input in `branred`'s and leaves the three cheaper reductions all
but untested.

Two kinds of comparison, and the suite distinguishes them. Almost everything is
held to the *bits*. `lgamma` is held to a mixed criterion — relative ulp away
from its zeros, absolute at them — because it reaches those zeros by
subtracting two quantities near `1.5`, so the last ulp of either is already
thousands of ulp of the answer and a relative bound there is not a statement
about accuracy. The `Fast` policies are held to the ulp bound each kernel
documents, and to the bits on every special value, in both directions: an
earlier version compared only when the reference was non-finite, and an
infinity where a subnormal belonged sailed through.

`tests/fma.rs` checks the software multiply-add against the hardware
instruction on 1.6 million triples, including the cancellation, subnormal and
tie cases that separate a correct emulation from a plausible one.

Everything runs on a real runtime rather than by calling the kernels as
ordinary Rust functions — a `#[cube]` function is not callable that way, and
more to the point, what could be wrong is what the *backend* does with the
arithmetic, which only a compiled kernel exercises.

## Adding a function

1. Write the kernel under `src/double/` (or `src/single/`), in the shape of
   `rmath`'s `src/reference/` implementation for it — the scalar algorithm,
   branches and all. `fma64(a, b, c, fk)` for a fused multiply-add, plain `*`
   and `+` where the reference schedule is unfused.
2. If it needs a table, add an accessor in `src/tables/consts.rs`.
3. Add a variant to the enum in `src/launch/unary.rs` or `binary.rs`, and an
   arm to each precision's `match`.
4. Add a line to `tests/equivalence.rs`.

Two things the sweep catches that a spot check will not, and both happened
here: a `Fast` series can be an order of magnitude short and still look
plausible (`exp2` was 55 ulp out, `ln` 79), and a subnormal correction applied
after the fact rather than folded into the exponent field costs exactly one
last bit on exactly one input in a hundred thousand.

## Licence

MIT, as `rmath` is. The tables are generated from ARM's optimized-routines,
`MIT OR Apache-2.0 WITH LLVM-exception`.
