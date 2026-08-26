//! The function catalogue: one macro invocation per function.
//!
//! Each entry generates a launch kernel per precision and a zero-sized host
//! object that drives them. The object carries the [`Policy`] and nothing
//! else, so configuring a function costs no space and no runtime dispatch —
//! the policy reaches the kernel as a comptime value and is gone before any
//! device code exists.
//!
//! Adding a function means adding a module under [`crate::cube`] and one
//! invocation here.

use cubecl::prelude::*;

use crate::config::Config;
use crate::host::{Ctx, MathFn, arg};
use crate::policy::{Accuracy, Domain, Policy};

/// One argument in, one out, both precisions.
macro_rules! math_fn1 {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path,
        f32: $k32:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(inp: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    out[ABSOLUTE_POS] = $k64(inp[ABSOLUTE_POS], tab, cfg);
                }
            }

            #[doc = concat!("`", stringify!($name), "`, single precision.")]
            #[cube(launch_unchecked)]
            pub fn k32(inp: &Array<f32>, out: &mut Array<f32>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    out[ABSOLUTE_POS] = $k32(inp[ABSOLUTE_POS], tab, cfg);
                }
            }
        }
        math_fn1!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
        math_fn1!(@eval $name, $module, f32, k32, eval_f32, eval_f32_into, config_f32);
    };

    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(inp: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    out[ABSOLUTE_POS] = $k64(inp[ABSOLUTE_POS], tab, cfg);
                }
            }
        }
        math_fn1!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
    };

    (@object $(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
        pub struct $name {
            /// Accuracy and domain.
            pub policy: Policy,
        }

        impl MathFn for $name {
            fn policy(&self) -> Policy { self.policy }
            fn with_policy(mut self, policy: Policy) -> Self { self.policy = policy; self }
        }

        impl $name {
            /// Bit-exact, and safe on any input. The default.
            pub fn new() -> Self { <Self as MathFn>::new() }
            /// The cheap approximation, still safe on any input.
            pub fn fast() -> Self { <Self as MathFn>::fast() }
            /// Take the cheap approximation.
            pub fn accuracy(self, a: Accuracy) -> Self { <Self as MathFn>::accuracy(self, a) }
            /// Promise the inputs are in range.
            pub fn domain(self, d: Domain) -> Self { <Self as MathFn>::domain(self, d) }
        }
    };

    (@eval $name:ident, $module:ident, $ty:ty, $kern:ident, $eval:ident, $into:ident, $cfg:ident) => {
        impl $name {
            #[doc = concat!("Evaluate over a device buffer of `", stringify!($ty), "`, writing into another.")]
            ///
            /// # Safety
            /// `inp` and `out` must each hold at least `n` elements.
            pub unsafe fn $into<R: Runtime>(
                &self,
                ctx: &Ctx<R>,
                inp: &cubecl::server::Handle,
                out: &cubecl::server::Handle,
                n: usize,
            ) {
                let (count, dim) = ctx.geometry(n);
                unsafe {
                    $module::$kern::launch_unchecked::<R>(
                        &ctx.client,
                        count,
                        dim,
                        arg::<R>(inp, n),
                        arg::<R>(out, n),
                        ctx.tables_arg(),
                        ctx.$cfg(self.policy),
                    );
                }
            }

            #[doc = concat!("Evaluate over a slice of `", stringify!($ty), "`.")]
            ///
            /// Uploads, launches and reads back. Convenient, and dominated by
            /// the two transfers for anything smaller than a few hundred
            /// thousand elements — use the `_into` form when the data is
            /// already on the device.
            pub fn $eval<R: Runtime>(&self, ctx: &Ctx<R>, xs: &[$ty]) -> Vec<$ty> {
                let n = xs.len();
                if n == 0 {
                    return Vec::new();
                }
                let inp = ctx.upload(xs);
                let out = ctx.alloc::<$ty>(n);
                unsafe { self.$into(ctx, &inp, &out, n) };
                ctx.download(out, n)
            }
        }
    };
}


/// One argument in, one out, for a kernel that reads no table.
///
/// The exact functions need no table, and binding one they never index would
/// put a buffer in every launch for nothing. The launch signature still has
/// it — one signature keeps the host side uniform — but the kernel does not
/// hand it on.
macro_rules! math_fn1_plain {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path,
        f32: $k32:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(inp: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let _ = tab[0];
                    out[ABSOLUTE_POS] = $k64(inp[ABSOLUTE_POS], cfg);
                }
            }

            #[doc = concat!("`", stringify!($name), "`, single precision.")]
            #[cube(launch_unchecked)]
            pub fn k32(inp: &Array<f32>, out: &mut Array<f32>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let _ = tab[0];
                    out[ABSOLUTE_POS] = $k32(inp[ABSOLUTE_POS], cfg);
                }
            }
        }
        math_fn1!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
        math_fn1!(@eval $name, $module, f32, k32, eval_f32, eval_f32_into, config_f32);
    };
}

/// One argument in, two out, for a kernel that reads no table.
macro_rules! math_fn1_pair_plain {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path,
        f32: $k32:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(inp: &Array<f64>, o1: &mut Array<f64>, o2: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let _ = tab[0];
                    let (a, b) = $k64(inp[ABSOLUTE_POS], cfg);
                    o1[ABSOLUTE_POS] = a;
                    o2[ABSOLUTE_POS] = b;
                }
            }

            #[doc = concat!("`", stringify!($name), "`, single precision.")]
            #[cube(launch_unchecked)]
            pub fn k32(inp: &Array<f32>, o1: &mut Array<f32>, o2: &mut Array<f32>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let _ = tab[0];
                    let (a, b) = $k32(inp[ABSOLUTE_POS], cfg);
                    o1[ABSOLUTE_POS] = a;
                    o2[ABSOLUTE_POS] = b;
                }
            }
        }
        math_fn1_pair!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
        math_fn1_pair!(@eval $name, $module, f32, k32, eval_f32, eval_f32_into, config_f32);
    };
}

/// Two arguments in, one out.
macro_rules! math_fn2 {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path,
        f32: $k32:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < out.len() {
                    let _ = tab[0];
                    out[ABSOLUTE_POS] = $k64(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg);
                }
            }

            #[doc = concat!("`", stringify!($name), "`, single precision.")]
            #[cube(launch_unchecked)]
            pub fn k32(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < out.len() {
                    let _ = tab[0];
                    out[ABSOLUTE_POS] = $k32(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg);
                }
            }
        }
        math_fn2!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
        math_fn2!(@eval $name, $module, f32, k32, eval_f32, eval_f32_into, config_f32);
    };

    (@eval $name:ident, $module:ident, $ty:ty, $kern:ident, $eval:ident, $into:ident, $cfg:ident) => {
        impl $name {
            #[doc = concat!("Evaluate over two device buffers of `", stringify!($ty), "`.")]
            ///
            /// # Safety
            /// `a`, `b` and `out` must each hold at least `n` elements.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn $into<R: Runtime>(
                &self,
                ctx: &Ctx<R>,
                a: &cubecl::server::Handle,
                b: &cubecl::server::Handle,
                out: &cubecl::server::Handle,
                n: usize,
            ) {
                let (count, dim) = ctx.geometry(n);
                unsafe {
                    $module::$kern::launch_unchecked::<R>(
                        &ctx.client,
                        count,
                        dim,
                        arg::<R>(a, n),
                        arg::<R>(b, n),
                        arg::<R>(out, n),
                        ctx.tables_arg(),
                        ctx.$cfg(self.policy),
                    );
                }
            }

            #[doc = concat!("Evaluate over two slices of `", stringify!($ty), "`.")]
            ///
            /// # Panics
            /// If the two slices have different lengths.
            pub fn $eval<R: Runtime>(&self, ctx: &Ctx<R>, a: &[$ty], b: &[$ty]) -> Vec<$ty> {
                assert_eq!(a.len(), b.len(), "arguments must have the same length");
                let n = a.len();
                if n == 0 {
                    return Vec::new();
                }
                let ah = ctx.upload(a);
                let bh = ctx.upload(b);
                let out = ctx.alloc::<$ty>(n);
                unsafe { self.$into(ctx, &ah, &bh, &out, n) };
                ctx.download(out, n)
            }
        }
    };
}

/// One argument in, one out, double precision only, no table.
macro_rules! math_fn1_plain_64 {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(inp: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let _ = tab[0];
                    out[ABSOLUTE_POS] = $k64(inp[ABSOLUTE_POS], cfg);
                }
            }
        }
        math_fn1!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
    };
}

/// Two arguments in, one out, double precision only.
macro_rules! math_fn2_64 {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < out.len() {
                    let _ = tab[0];
                    out[ABSOLUTE_POS] = $k64(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg);
                }
            }
        }
        math_fn2!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
    };
}

/// Two arguments in, one out, for a kernel that reads a table.
macro_rules! math_fn2_tab {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < out.len() {
                    out[ABSOLUTE_POS] = $k64(a[ABSOLUTE_POS], b[ABSOLUTE_POS], tab, cfg);
                }
            }
        }
        math_fn2!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
    };
}

/// One argument in, two out — `frexp`, `modf`, `sincos`.
macro_rules! math_fn1_pair {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path,
        f32: $k32:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(inp: &Array<f64>, o1: &mut Array<f64>, o2: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let (a, b) = $k64(inp[ABSOLUTE_POS], tab, cfg);
                    o1[ABSOLUTE_POS] = a;
                    o2[ABSOLUTE_POS] = b;
                }
            }

            #[doc = concat!("`", stringify!($name), "`, single precision.")]
            #[cube(launch_unchecked)]
            pub fn k32(inp: &Array<f32>, o1: &mut Array<f32>, o2: &mut Array<f32>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < inp.len() {
                    let (a, b) = $k32(inp[ABSOLUTE_POS], tab, cfg);
                    o1[ABSOLUTE_POS] = a;
                    o2[ABSOLUTE_POS] = b;
                }
            }
        }
        math_fn1_pair!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
        math_fn1_pair!(@eval $name, $module, f32, k32, eval_f32, eval_f32_into, config_f32);
    };

    (@eval $name:ident, $module:ident, $ty:ty, $kern:ident, $eval:ident, $into:ident, $cfg:ident) => {
        impl $name {
            #[doc = concat!("Evaluate over a device buffer of `", stringify!($ty), "`, into two outputs.")]
            ///
            /// # Safety
            /// `inp`, `o1` and `o2` must each hold at least `n` elements.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn $into<R: Runtime>(
                &self,
                ctx: &Ctx<R>,
                inp: &cubecl::server::Handle,
                o1: &cubecl::server::Handle,
                o2: &cubecl::server::Handle,
                n: usize,
            ) {
                let (count, dim) = ctx.geometry(n);
                unsafe {
                    $module::$kern::launch_unchecked::<R>(
                        &ctx.client,
                        count,
                        dim,
                        arg::<R>(inp, n),
                        arg::<R>(o1, n),
                        arg::<R>(o2, n),
                        ctx.tables_arg(),
                        ctx.$cfg(self.policy),
                    );
                }
            }

            #[doc = concat!("Evaluate over a slice of `", stringify!($ty), "`, returning both outputs.")]
            pub fn $eval<R: Runtime>(&self, ctx: &Ctx<R>, xs: &[$ty]) -> (Vec<$ty>, Vec<$ty>) {
                let n = xs.len();
                if n == 0 {
                    return (Vec::new(), Vec::new());
                }
                let inp = ctx.upload(xs);
                let o1 = ctx.alloc::<$ty>(n);
                let o2 = ctx.alloc::<$ty>(n);
                unsafe { self.$into(ctx, &inp, &o1, &o2, n) };
                (ctx.download(o1, n), ctx.download(o2, n))
            }
        }
    };
}

/// Two arguments in, two out — `remquo`.
macro_rules! math_fn2_pair {
    (
        $(#[$doc:meta])*
        name: $name:ident,
        module: $module:ident,
        f64: $k64:path,
        f32: $k32:path $(,)?
    ) => {
        math_fn1!(@object $(#[$doc])* $name);
        #[doc = concat!("Launch kernels for [`", stringify!($name), "`].")]
        pub mod $module {
            use super::*;

            #[doc = concat!("`", stringify!($name), "`, double precision.")]
            #[cube(launch_unchecked)]
            pub fn k64(a: &Array<f64>, b: &Array<f64>, o1: &mut Array<f64>, o2: &mut Array<f64>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < a.len() {
                    let _ = tab[0];
                    let (p, q) = $k64(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg);
                    o1[ABSOLUTE_POS] = p;
                    o2[ABSOLUTE_POS] = q;
                }
            }

            #[doc = concat!("`", stringify!($name), "`, single precision.")]
            #[cube(launch_unchecked)]
            pub fn k32(a: &Array<f32>, b: &Array<f32>, o1: &mut Array<f32>, o2: &mut Array<f32>, tab: &Array<u64>, #[comptime] cfg: Config) {
                if ABSOLUTE_POS < a.len() {
                    let _ = tab[0];
                    let (p, q) = $k32(a[ABSOLUTE_POS], b[ABSOLUTE_POS], cfg);
                    o1[ABSOLUTE_POS] = p;
                    o2[ABSOLUTE_POS] = q;
                }
            }
        }
        math_fn2_pair!(@eval $name, $module, f64, k64, eval_f64, eval_f64_into, config_f64);
        math_fn2_pair!(@eval $name, $module, f32, k32, eval_f32, eval_f32_into, config_f32);
    };

    (@eval $name:ident, $module:ident, $ty:ty, $kern:ident, $eval:ident, $into:ident, $cfg:ident) => {
        impl $name {
            #[doc = concat!("Evaluate over two device buffers of `", stringify!($ty), "`, into two outputs.")]
            ///
            /// # Safety
            /// Every handle must hold at least `n` elements.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn $into<R: Runtime>(
                &self,
                ctx: &Ctx<R>,
                a: &cubecl::server::Handle,
                b: &cubecl::server::Handle,
                o1: &cubecl::server::Handle,
                o2: &cubecl::server::Handle,
                n: usize,
            ) {
                let (count, dim) = ctx.geometry(n);
                unsafe {
                    $module::$kern::launch_unchecked::<R>(
                        &ctx.client,
                        count,
                        dim,
                        arg::<R>(a, n),
                        arg::<R>(b, n),
                        arg::<R>(o1, n),
                        arg::<R>(o2, n),
                        ctx.tables_arg(),
                        ctx.$cfg(self.policy),
                    );
                }
            }

            #[doc = concat!("Evaluate over two slices of `", stringify!($ty), "`, returning both outputs.")]
            ///
            /// # Panics
            /// If the two slices have different lengths.
            pub fn $eval<R: Runtime>(&self, ctx: &Ctx<R>, a: &[$ty], b: &[$ty]) -> (Vec<$ty>, Vec<$ty>) {
                assert_eq!(a.len(), b.len(), "arguments must have the same length");
                let n = a.len();
                if n == 0 {
                    return (Vec::new(), Vec::new());
                }
                let ah = ctx.upload(a);
                let bh = ctx.upload(b);
                let o1 = ctx.alloc::<$ty>(n);
                let o2 = ctx.alloc::<$ty>(n);
                unsafe { self.$into(ctx, &ah, &bh, &o1, &o2, n) };
                (ctx.download(o1, n), ctx.download(o2, n))
            }
        }
    };
}

math_fn1! {
    /// `10^x`.
    ///
    /// `BitExact` reproduces glibc 2.39's `__exp10`, which shares `exp`'s
    /// table but, like `exp2`, ships with no fused-multiply-add variant.
    name: Exp10,
    module: exp10,
    f64: crate::cube::double::exp10::exp10,
}

math_fn1! {
    /// Natural logarithm.
    ///
    /// `BitExact` reproduces glibc's `__ieee754_log_fma`, including its
    /// separate near-one path.
    name: Ln,
    module: ln,
    f64: crate::cube::double::ln::ln,
}

math_fn1! {
    /// Base-2 logarithm.
    ///
    /// `BitExact` reproduces glibc's `__ieee754_log2_fma`, near-one path
    /// included.
    name: Log2,
    module: log2,
    f64: crate::cube::double::logx::log2,
}

math_fn1! {
    /// Base-10 logarithm.
    ///
    /// `BitExact` reproduces glibc's `__log10_finite`, which is a wrapper
    /// around `__ieee754_log_fma` rather than a table algorithm of its own —
    /// so this is `Ln`'s table walk, reduced and rescaled the way the
    /// disassembly does it.
    name: Log10,
    module: log10,
    f64: crate::cube::double::logx::log10,
}

math_fn1! {
    /// `e^x - 1`, accurate for small `x`.
    ///
    /// `BitExact` reproduces glibc's `__expm1_fma`, Estrin's scheme and all
    /// three of its fused operations included.
    name: Expm1,
    module: expm1,
    f64: crate::cube::double::expm1::expm1,
}

math_fn1! {
    /// `ln(1 + x)`, accurate for small `x`.
    ///
    /// `BitExact` reproduces glibc's `__log1p_fma`.
    name: Log1p,
    module: log1p,
    f64: crate::cube::double::log1p::log1p,
}

math_fn1_plain_64! {
    /// Cube root.
    ///
    /// `BitExact` matches `f64::cbrt`, which is Rust's own correctly-rounded
    /// CORE-MATH port rather than glibc's — see the kernel's module docs for
    /// why that is the right reference for a Rust crate. Both policy axes are
    /// accepted and have no effect.
    name: Cbrt,
    module: cbrt,
    f64: crate::cube::double::cbrt::cbrt,
}

math_fn2_64! {
    /// `sqrt(x^2 + y^2)`, without the intermediate overflow.
    ///
    /// `BitExact` reproduces glibc's `__ieee754_hypot`, which uses no fused
    /// multiply-add anywhere — its error-free transformations depend on
    /// separate roundings. Both policy axes are accepted and have no effect.
    name: Hypot,
    module: hypot,
    f64: crate::cube::double::hypot::hypot,
}

math_fn2_tab! {
    /// `x^y`.
    ///
    /// `BitExact` reproduces glibc's `__pow_fma`, which computes the logarithm
    /// to better than double precision because an error of one ulp there
    /// becomes an error of `y` ulp in the result. `Fast` keeps that
    /// double-double product and drops only the tables.
    name: Pow,
    module: pow,
    f64: crate::cube::double::pow::pow,
}

math_fn1! {
    /// `2^x`.
    ///
    /// `BitExact` reproduces glibc's `__ieee754_exp2`, which — unlike `exp` —
    /// ships with no fused-multiply-add variant, so the schedule is separate
    /// multiplies and adds throughout.
    name: Exp2,
    module: exp2,
    f64: crate::cube::double::exp2::exp2,
}

math_fn1! {
    /// `e^x`.
    ///
    /// `BitExact` reproduces glibc's `__ieee754_exp_fma`; `Fast` takes a
    /// table-free degree-13 series, below 1 ulp. `Finite` means `|x| < 512`.
    name: Exp,
    module: exp,
    f64: crate::cube::double::exp::exp,
}

// ---------------------------------------------------------------------------
// The functions IEEE-754 pins down exactly.
//
// Both policy axes are no-ops for every entry below: there is no approximation
// to make cheaper and no special case to repair, because the special cases are
// on the main path at no cost. They are listed with the same names and the
// same shapes as the rest so that callers do not have to know which category a
// function falls into.
// ---------------------------------------------------------------------------

math_fn1_plain! {
    /// Largest integer not greater than `x`.
    name: Floor,
    module: floor,
    f64: crate::cube::exact::double::floor,
    f32: crate::cube::exact::single::floor,
}

math_fn1_plain! {
    /// Smallest integer not less than `x`.
    name: Ceil,
    module: ceil,
    f64: crate::cube::exact::double::ceil,
    f32: crate::cube::exact::single::ceil,
}

math_fn1_plain! {
    /// `x` truncated towards zero.
    name: Trunc,
    module: trunc,
    f64: crate::cube::exact::double::trunc,
    f32: crate::cube::exact::single::trunc,
}

math_fn1_plain! {
    /// `x` rounded to the nearest integer, ties away from zero.
    name: Round,
    module: round,
    f64: crate::cube::exact::double::round,
    f32: crate::cube::exact::single::round,
}

math_fn1_plain! {
    /// `x` rounded to the nearest integer, ties to even.
    name: Rint,
    module: rint,
    f64: crate::cube::exact::double::rint,
    f32: crate::cube::exact::single::rint,
}

math_fn1_plain! {
    /// `sqrt(x)`, correctly rounded.
    name: Sqrt,
    module: sqrt,
    f64: crate::cube::exact::double::sqrt,
    f32: crate::cube::exact::single::sqrt,
}

math_fn1_plain! {
    /// `|x|`.
    name: Abs,
    module: abs,
    f64: crate::cube::exact::double::abs,
    f32: crate::cube::exact::single::abs,
}

math_fn1_plain! {
    /// The binary exponent of `x`, as an integer in a float lane.
    name: Ilogb,
    module: ilogb,
    f64: crate::cube::exact::double::ilogb,
    f32: crate::cube::exact::single::ilogb,
}

math_fn2! {
    /// The magnitude of `x` with the sign of `y`.
    name: CopySign,
    module: copysign,
    f64: crate::cube::exact::double::copysign_fn,
    f32: crate::cube::exact::single::copysign_fn,
}

math_fn2! {
    /// The positive difference: `x - y` if `x > y`, and `+0` otherwise.
    name: Fdim,
    module: fdim,
    f64: crate::cube::exact::double::fdim,
    f32: crate::cube::exact::single::fdim,
}

math_fn2! {
    /// The larger of `x` and `y`, ignoring NaN.
    name: Fmax,
    module: fmax,
    f64: crate::cube::exact::double::fmax,
    f32: crate::cube::exact::single::fmax,
}

math_fn2! {
    /// The smaller of `x` and `y`, ignoring NaN.
    name: Fmin,
    module: fmin,
    f64: crate::cube::exact::double::fmin,
    f32: crate::cube::exact::single::fmin,
}

math_fn2! {
    /// `x * 2^n`, with `n` carried in a float lane.
    name: Ldexp,
    module: ldexp,
    f64: crate::cube::exact::double::ldexp,
    f32: crate::cube::exact::single::ldexp,
}

math_fn2! {
    /// `x * 2^n`, under its other name. The same code as [`Ldexp`].
    name: Scalbn,
    module: scalbn,
    f64: crate::cube::exact::double::scalbn,
    f32: crate::cube::exact::single::scalbn,
}

math_fn2! {
    /// `x` reduced modulo `y`, with the sign of `x`.
    name: Fmod,
    module: fmod,
    f64: crate::cube::exact::double::fmod,
    f32: crate::cube::exact::single::fmod,
}

math_fn2! {
    /// The IEEE-754 remainder: `x - y * n`, with `n` the nearest integer to `x / y`.
    name: Remainder,
    module: remainder,
    f64: crate::cube::exact::double::remainder,
    f32: crate::cube::exact::single::remainder,
}

math_fn2! {
    /// The next representable value after `x` in the direction of `y`.
    name: NextAfter,
    module: nextafter,
    f64: crate::cube::exact::double::nextafter,
    f32: crate::cube::exact::single::nextafter,
}

math_fn1_pair_plain! {
    /// Split `x` into a significand in `[0.5, 1)` and a power of two.
    name: Frexp,
    module: frexp,
    f64: crate::cube::exact::double::frexp,
    f32: crate::cube::exact::single::frexp,
}

math_fn1_pair_plain! {
    /// Split `x` into its fractional and integral parts, both with `x`'s sign.
    name: Modf,
    module: modf,
    f64: crate::cube::exact::double::modf,
    f32: crate::cube::exact::single::modf,
}

math_fn2_pair! {
    /// `x` reduced modulo `y`, together with the low bits of the quotient.
    name: Remquo,
    module: remquo,
    f64: crate::cube::exact::double::remquo,
    f32: crate::cube::exact::single::remquo,
}
