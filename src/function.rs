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

math_fn1! {
    /// `e^x`.
    ///
    /// `BitExact` reproduces glibc's `__ieee754_exp_fma`; `Fast` takes a
    /// table-free degree-13 series, below 1 ulp. `Finite` means `|x| < 512`.
    name: Exp,
    module: exp,
    f64: crate::cube::double::exp::exp,
}
