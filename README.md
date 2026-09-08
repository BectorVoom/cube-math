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

## The vector entry points

`exp`, `exp2`, `exp10`, `ln`, `log2` and `log10` also come in a `_vec` form
that takes N elements at once:

```rust
#[cube]
fn my_kernel<N: Size>(x: &Array<Vector<f64, N>>, out: &mut Array<Vector<f64, N>>,
                      #[comptime] cfg: MathConfig) {
    out[ABSOLUTE_POS] = m::exp::exp_vec::<N>(x[ABSOLUTE_POS], cfg);
}
```

Each is **bit-identical to its scalar twin on every element**, on every policy,
at every width. That is not a bound, it is an equality, and `tests/vector.rs`
holds all six to it at `to_bits()` at widths 1/2/4/8 over the equivalence
sweep plus every branch boundary each function has.

It is exact by construction rather than by luck. The scalar schedule is
rewritten on `Vector<f64, N>` — every operation elementwise, the fused
multiply-adds through `fma64_vec`, the table gather one element at a time — and
IEEE-754 rounds identically whether an operand sits in a scalar or in a lane,
so the main-path elements are the scalar routine's bits by definition. The
elements the scalar routine would send down a *different* branch are then
overwritten by the scalar routine itself, on that element alone. So the main
path is exact by construction and the rest is the scalar function, full stop.

What each function has to get right is where its main path ends:

| | main path | repaired per element |
|---|---|---|
| `exp` | `0x3c9 <= abstop < 0x408` | tiny, `|x| >= 512`, non-finite |
| `exp2` | `abstop >= 0x3c9` and `|x| <= 928` | tiny, `specialcase`, the range answers |
| `exp10` | `SMALL_TOP <= abstop < SMALL_TOP + THRESH` (`|x| < 256`) | tiny, `specialcase`, the range answers |
| `ln` | positive normal outside the near-one window | the near-one window, the degenerate inputs |
| `log2` | not degenerate, outside the near-one window | the near-one window, the degenerate inputs |
| `log10` | not degenerate | the degenerate inputs |

Subnormals are *not* a repair case for the two `logx` functions: their
`normalised_bits` is a `select` rather than a branch, so it vectorises as it
stands. `log10` is the one that gains least — its main path calls the whole of
`ln`, near-one arm included, and its reduced argument sits within one exponent
step of 1, so a good fraction of any input vector lands in that window.

Why they exist: a kernel that already holds N points in a vector — a grid
collocation, an unrolled stencil — would otherwise extract each element, call
the scalar routine and insert the result, which is N dependent chains where one
vectorised chain will do. What that saves is measured below.

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
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm   # wherever yours lives
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
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

### The kernel cache is not optional either

This crate compiles one kernel per function, per policy, per multiply-add kind,
per precision. The equivalence suite alone asks for over a hundred, and hipRTC
is slow enough that compiling them dominates the run. CubeCL can persist the
compiled binaries and does not by default; the `cubecl.toml` in the repository
root turns it on:

```toml
[compilation]
cache = "target"
```

That takes the suite's ROCm run from **25.1 s to 2.5 s**, and changes nothing
about the arithmetic — the cache is keyed on the kernel's `KernelId`, which is
exactly what the comptime policy and multiply-add already vary, and which is
why two policies were never able to share a cache entry in the first place.

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

The first of those has a corollary that cost this crate two rounds of `hipRTC`
diagnostics: `f64::from_bits(0x…)` written *inside* a `#[cube]` body is a
reinterpretation of a literal, and a literal is an rvalue. It compiles on the
CPU runtime, which lowers through MLIR and never asks the question, and fails
on every C++ backend with an error pointing at generated source the author has
never seen. Every bit pattern in this crate is therefore a module-level
`const`, evaluated on the host, and none is spelled at a call site — including
the ones that only appear once.

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

The families added since, measured the same way — best of six runs, same
machine, same `-ffp-contract=off`:

| Melem/s | ROCm | | Melem/s | ROCm |
|---|---|---|---|---|
| `sin` | 1058 | | `erf` | 590 |
| `cos` | 1108 | | `erfc` | 237 |
| `tan` | 902 | | `lgamma` | 428 |
| `sin`, past `105414350` | 187 | | `tgamma` | 352 |
| `asin` | 1484 | | `j0` | 126 |
| `atan` | 1151 | | `y1` | 134 |
| `atan2` | 544 | | `pow` | 666 |
| `sinh` | 2070 | | `hypot` | 707 |
| `tanh` | 568 | | `asinh` | 326 |
| `acosh` | 877 | | | |

Two of those are worth reading as design outcomes rather than measurements.
`erfc`'s accurate path runs on roughly one input in thirty thousand, and a warp
pays for it only when one of its threads lands there — the benchmark sweeps the
whole domain uniformly, which is close to the worst case. And the Bessel
functions' near-a-zero repair is exactly the data-dependent branch that stops
`rmath` vectorising them at all; here it costs the thread that takes it and
nothing else.

### The vector entry points

`cargo run --release --features "cpu,hip" --example bench_vec`, four million
`f64` already resident, same machine, best of three. Every row is the
`BitExact` policy, and every vector row is bit-identical to the scalar row it
sits under — this is a table of what the identity costs, and it costs less than
nothing.

| Melem/s, CPU runtime | scalar | width 2 | width 4 | width 8 | width 8 vs scalar |
|---|---|---|---|---|---|
| `exp` | 295 | 456 | 600 | **791** | 2.68x |
| `exp2` | 273 | 457 | 603 | **792** | 2.90x |
| `exp10` | 279 | 453 | 625 | **804** | 2.88x |
| `ln` | 279 | 449 | 599 | **800** | 2.87x |
| `log2` | 269 | 420 | 599 | **776** | 2.88x |
| `log10` | 256 | 389 | 559 | **746** | 2.92x |

Close to linear to width 4 and then tapering, which is what a machine with
four-wide `f64` vector registers should do.

On ROCm the same table is **flat** — every ratio lands between 0.95x and 1.6x,
and moving between those two is within the run-to-run spread the next section
describes, not a measurement. That is the correct outcome rather than a
disappointing one: a GPU is already SIMT, the scalar kernel's threads *are* the
lanes, and widening buys no parallelism that was not already there while
costing registers and occupancy. The vector entry points are not there to make
an elementwise pass faster on a GPU — `launch::unary` is already the right tool
for that, and it stays the right tool. They are there so that a kernel which
already holds a `Vector<f64, N>` does not have to take it apart.

### Why "best of six" and not "the number"

Because on this device the two words are not interchangeable, and the spread
says something. Over six runs of the same binary on the same data:

| | best | worst | spread |
|---|---|---|---|
| `rint` | 4646 | 1108 | 4.2x |
| `fmod` | 2152 | 440 | 4.9x |
| `sqrt` | 1889 | 478 | 4.0x |
| `cos` | 1108 | 269 | 4.1x |
| `j0` | 126 | 117 | **1.1x** |
| `y1` | 134 | 110 | **1.2x** |
| `erfc` | 237 | 193 | **1.2x** |
| `asinh` | 326 | 276 | **1.2x** |
| `sin`, past `105414350` | 187 | 171 | **1.1x** |

The functions that swing by four and five times are the *cheap* ones. A kernel
that does eight operations per element is bound by the two memory transactions
either side of them, and on an integrated GPU sharing a power budget with the
cores next to it, what that measures is the thermal state. The functions that
reproduce to within a tenth are the expensive ones — `j0`, `erfc`, `asinh`,
`branred` — which are compute-bound and therefore actually measurable. So the
slow rows in the table above are the trustworthy ones, and the fast rows are an
upper bound that a busy machine will not reach.

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

`tests/software_fma.rs` checks the *kernels* with that emulation substituted —
`FmaKind::Software` forced on a device that has a real fused multiply-add, held
to `rmath`'s bits. Neither backend reachable here needs the software path, so
`tests/equivalence.rs` never takes it: it reads the `FmaKind` off the device
and both devices fuse. Forcing it is what makes the path testable at all, and
it is testable *because* `fma_f64` is correctly rounded — a correct emulation
has to land on the hardware instruction's answer, so agreement is evidence.

It is also how `ln`, `log2` and `log10` were found reaching for the raw `fma`
intrinsic instead of `fma64(.., fk)`, which meant their bit-exact paths ignored
`FmaKind` and would have rounded twice on SPIR-V. Note what that implies about
the limits of this check: on a fused device the two spellings *are* the same
number, so no test that runs here can see the difference. What the file pins is
that the software path runs and reproduces glibc's schedule; that every site
was converted is a static fact, checked by there being no bare `fma(` left in
those modules.

`tests/vector.rs` holds each `_vec` entry point to `to_bits()` equality with
its scalar twin, on all four policies, at widths 1/2/4/8, over the equivalence
sweep plus every branch boundary the function has. The boundaries are the
point: a vector routine computes its main path on *every* element and then
overwrites the ones belonging to another branch, so the failure mode is an
off-by-one in that repair predicate, and it is visible nowhere except on inputs
that sit astride a boundary.

All three suites pass in full on the CubeCL CPU runtime and on ROCm (7.1.1,
gfx1151), which are the two backends that report `bit_exact_capable()`. `wgpu`
reports `f64` unusable and skips itself, as it should — see below.

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
5. Optionally, add a `_vec` twin: the same schedule on `Vector<f64, N>`, with
   the elements the scalar routine would send down another branch overwritten
   afterwards by the scalar routine on that element. Add a line to
   `tests/vector.rs` and one to `examples/bench_vec.rs`; both are macro-driven,
   so a line is all it is. Worth doing where the main path is long and
   straight, not worth it where the function is mostly classification.

   Two things bite when writing one. The C++ backends have **no unary minus on
   a vector type** — `hipcc` rejects `-double_2` and the CPU runtime accepts
   it, so this compiles and passes locally and then fails on ROCm. Fold the
   sign into the constant instead (`kd * -C` for `-kd * C`), which is the same
   number because a product's sign is the exclusive or of its operands'. And
   scalar-typed `reinterpret`/`cast_from` do not lift to vectors implicitly:
   write `Vector::<u64, N>::reinterpret(v)`, not `u64::reinterpret(v)`.

Two things the sweep catches that a spot check will not, and both happened
here: a `Fast` series can be an order of magnitude short and still look
plausible (`exp2` was 55 ulp out, `ln` 79), and a subnormal correction applied
after the fact rather than folded into the exponent field costs exactly one
last bit on exactly one input in a hundred thousand.

## Licence

MIT, as `rmath` is. The tables are generated from ARM's optimized-routines,
`MIT OR Apache-2.0 WITH LLVM-exception`.
