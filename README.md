# cube-math

A CubeCL `libm` for `f64` and `f32` — one kernel source, CPU and GPU, bit-exact
to the platform `libm` where you ask for it.

This is a port of [`rmath`](https://github.com/BectorVoom/rmath), a SIMD `libm`
whose defining property is that its default policy is not "about one ulp" but
*bit-identical to the platform's scalar `libm`*. The port keeps that property
and changes what runs it: instead of a CPU vector register, the parallelism is
a CubeCL launch, so the same functions run on Vulkan, CUDA, HIP, Metal, WebGPU
and the CPU runtime.

```rust
use cube_math::prelude::*;
use cubecl::cpu::CpuRuntime;

let ctx = Ctx::<CpuRuntime>::new(&Default::default());
let xs: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.001).collect();

let exact = Exp::new().eval_f64(&ctx, &xs);   // bit-identical to libm's exp
let quick = Exp::fast().eval_f64(&ctx, &xs);  // below 1 ulp, no table
```

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
the generated shader contains only the path you asked for. Two policies are two
kernels with two `KernelId`s, which is what keeps them out of each other's
compilation cache.

## What is ported

| family | functions | `BitExact` | `Fast` |
|---|---|---|---|
| **exact** (`f64` **and** `f32`) | `floor` `ceil` `trunc` `round` `rint` `sqrt` `abs` `ilogb` `copysign` `fdim` `fmax` `fmin` `ldexp` `scalbn` `fmod` `remainder` `nextafter` `frexp` `modf` `remquo` | exact by construction, on every device | same code |
| **exponentials** (`f64`) | `exp` `exp2` `exp10` `expm1` | reference schedule | table-free series, ≤ 1.5 ulp |
| **logarithms** (`f64`) | `ln` `log2` `log10` `log1p` | reference schedule | table-free series, ≤ 3 ulp |
| **algebraic** (`f64`) | `pow` `cbrt` `hypot` | reference schedule | *same code* — see below |

**Not yet ported**: the trigonometric family (`sin` `cos` `tan` `sincos`), the
inverse trigonometric family (`asin` `acos` `atan` `atan2`), the hyperbolics
(`sinh` `cosh` `tanh` `asinh` `acosh` `atanh`), `erf` / `erfc`, the gamma
functions (`lgamma` `lgamma_r` `tgamma`), the Bessel family (`j0` `j1` `y0`
`y1` `jn` `yn`), and the single-precision transcendentals. Their tables are
already in the arena; what is missing is the kernels. See *Adding a function*.

The trigonometric family carries one extra obligation the others do not:
`rmath` leaves `|x| >= 105414350` to the platform's `__branred` (a Payne-Hanek
reduction) rather than porting it, which a GPU kernel cannot do — there is no
platform to fall back to. Porting `sin` here therefore means porting
`__branred` too.

Three functions accept `Fast` and ignore it, and the reason is worth stating
rather than hiding: `pow` multiplies its logarithm by `y`, so a table-free
logarithm accurate to a fifth of an ulp still leaves `x^y` around 200 ulp out
at `|y| = 360` (measured, not assumed) — reaching the accuracy `pow` needs
means the double-double table logarithm, and once you have paid for that there
is nothing left for an approximation to save. `cbrt` and `hypot` are already
their own cheap algorithms.

## Bit-exactness on a device

Every IEEE-754 operation rounds identically on every conforming device, so
replaying a reference schedule on a GPU gives the same bits as replaying it on
a CPU — *provided the backend does not rewrite the arithmetic*. Four things
can, none of them is guaranteed by CubeCL, and every one of them fails on some
backend in wide use. So `cube_math::probe::Fidelity` measures them on the live
device, and `Ctx` finishes with a functional canary: a real kernel evaluated on
inputs whose correctly-rounded answers are mathematical constants.

```rust
# use cube_math::prelude::*;
# use cubecl::cpu::CpuRuntime;
let ctx = Ctx::<CpuRuntime>::new(&Default::default());
assert!(ctx.fidelity.f64.bit_exact_capable(), "{}", ctx.fidelity.f64.summary());
```

| property | if false | recoverable? |
|---|---|---|
| `usable` | the precision does not work here | no |
| `fused_fma` | `fma` rounds twice | yes — a software multiply-add |
| `separate_mul_add` | `a * b + c` is contracted | no |
| `subnormals` | subnormals are flushed | no |
| `stable_arithmetic` | `(a + b) - a` folds to `b` | no |

### What the probes actually found

These are not hypothetical. On the machine this was developed on:

* **CubeCL's CPU runtime** passes everything. It is the reference backend here.
* **`cubecl-spirv`** (0.10 and 0.11-pre) lowers `Fma` to a separate `OpFMul`
  and `OpFAdd` — two roundings. That is defensible on its own terms:
  GLSL.std.450's `Fma` is only *required* to be fused when it carries the
  `NoContraction` decoration, which cubecl emits neither of. It is the reason
  `cube::fma::fma_f64` exists.
* **Mesa RADV** (26.1, gfx11 APU) folds `(a + b) - a` to `b` for `f64`. That is
  an unsafe floating-point transform, and it is fatal twice over: it breaks the
  error-free transformations the software multiply-add is built from, *and* it
  breaks the `(x * c + BIG) - BIG` round-to-nearest trick that every
  Cody-Waite argument reduction in this crate opens with. The SPIR-V reaches
  the driver through `create_shader_module_passthrough`, and a dump confirms
  the `OpFAdd`/`OpFSub` pair is intact, so the rewrite is the driver's.
  `RADV_DEBUG=llvm`, `noopt` and `nocompute` do not change it. `Fidelity`
  reports `f64` unusable there, and the suite skips it rather than claiming a
  bit-exactness it cannot deliver.
* **`wgpu`'s WGSL path** advertises `f64`, has `f64` arithmetic, passes every
  mechanical probe — and then evaluates `exp(1)` to something that is not `e`.
  This is why the canary exists: a property probe tests what you thought to
  ask, and a kernel tests what actually happens. Its `f32` also flushes
  subnormals and contracts `a * b + c`.

The upshot for this machine is that the CPU runtime is bit-exact and the GPU is
not — not because of anything in these kernels, but because of what those two
backends do with the arithmetic. On a backend with a fused `fma` and no
reassociation (CUDA and HIP emit a real `fma`), the same kernels are bit-exact
without changing a line; `Fidelity` is how you find out, per device, before you
rely on it.

## Speed

`RUSTFLAGS="-C target-cpu=native" cargo run --release --features cpu --example bench`,
one million `f64`, data already resident, AMD Krackan (Zen 5 mobile). The flag
matters for the baselines and not for this crate: `rmath` and `libm` are
compiled ahead of time and will not use the wide instructions unless told to,
while a CubeCL kernel is compiled for the machine it is about to run on either
way.

| | Melem/s |
|---|---|
| scalar `libm` `exp` | 384 |
| `rmath` `exp` (SIMD, bit-exact) | 651 |
| **`cube-math` `exp` (CPU runtime, bit-exact)** | **769** |
| scalar `libm` `ln` | 419 |
| `rmath` `ln` (SIMD, bit-exact) | 475 |
| **`cube-math` `ln` (CPU runtime, bit-exact)** | **754** |
| scalar `libm` `pow` | 138 |
| **`cube-math` `pow` (CPU runtime, bit-exact)** | **351** |

The rest of the ported set lands between 455 (`cbrt`) and 748 (`exp2`)
Melem/s, except `fmod` at 107 — its shift-and-subtract loop runs one iteration
per binary digit of the quotient, the same work glibc does, and it is the one
place in the crate where neighbouring threads can diverge badly.

Those are *kernel* numbers, on data already on the device. A single call over a
host `Vec` measures 112 Melem/s, and the difference is entirely the two
transfers. That is the honest shape of the trade: reach for this when the data
is already where the kernel is, or when enough work happens per element to pay
for the trip.

## Testing

```text
cargo test --release --features cpu
```

`tests/equivalence.rs` compares every kernel against `rmath`'s bit-exact
objects — whose own suite pins them to the platform `libm`, so agreement is
established transitively — over a sweep built to land on the values that break
these algorithms: every special value, every power of two across the exponent
range and its two neighbours, a dense walk of the interesting interval, and
random bit patterns. Two-argument functions get a cross product, because what
breaks them is the *relationship* between the arguments.

`tests/fma.rs` checks the software multiply-add against the hardware
instruction on 1.6 million triples, including the cancellation, subnormal and
tie cases that separate a correct emulation from a plausible one.

Everything runs on a real runtime rather than by calling the kernels as
ordinary Rust functions — a `#[cube]` function is not callable that way, and
more to the point, what could be wrong is what the *backend* does with the
arithmetic, which only a compiled kernel exercises.

## Adding a function

1. Write the kernel under `src/cube/double/` (or `single/`), in the shape of
   `rmath`'s `src/reference/` implementation for it — the scalar algorithm,
   branches and all. `fma64(a, b, c, fk)` for a fused multiply-add, plain `*`
   and `+` where the reference schedule is unfused.
2. If it needs a table, add an `OFF_`/`LEN_` pair in `src/tables/arena.rs` and
   push the data in `build()` in the same order.
3. Add one `math_fn*!` invocation in `src/function.rs`.
4. Add a line to `tests/equivalence.rs`.

Two things the sweep catches that a spot check will not, and both showed up
here: a `Fast` series can be an order of magnitude short and still look
plausible (`exp2` was 55 ulp out, `ln` 79), and a subnormal correction applied
after the fact rather than folded into the exponent field costs exactly one
last bit on exactly one input in a hundred thousand.

## Licence

MIT, as `rmath` is. The tables are generated from ARM's optimized-routines,
`MIT OR Apache-2.0 WITH LLVM-exception`.
