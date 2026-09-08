//! Double-precision kernels.
//!
//! These are `#[cube]` functions, which is the point: call them from inside
//! your own kernel rather than launching a pass of your own over memory.
//!
//! ```ignore
//! use cube_math::{double as m, MathConfig};
//!
//! #[cube]
//! fn my_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] cfg: MathConfig) {
//!     out[ABSOLUTE_POS] = m::exp::exp(x[ABSOLUTE_POS], cfg);
//! }
//! ```
//!
//! Nothing has to be bound or uploaded to make that work: the tables are
//! constant arrays compiled into the kernel. See [`crate::tables::consts`].
//!
//! # The vector entry points
//!
//! [`exp`], [`exp2`], [`exp10`], [`ln`] and [`logx`]'s two logarithms each
//! carry a `_vec` twin taking `Vector<f64, N>`, bit-identical to the scalar
//! one on every element. See [`exp`]'s module documentation for how that is
//! arranged and [`crate`]'s for why. They are for a caller who already holds a
//! vector — the launch side never uses them, because a pass over an array
//! already has all the parallelism there is.

pub mod acosh;
pub mod asinh;
pub mod bessel;
pub mod branred;
pub mod cbrt;
pub mod dd;
pub mod erf;
pub mod erfc;
pub mod exact;
pub mod exp;
pub mod exp10;
pub mod exp2;
pub mod expm1;
pub mod gamma;
pub mod hyper;
pub mod hypot;
pub mod invtrig;
pub mod ln;
pub mod log1p;
pub mod logtab;
pub mod logx;
pub mod pow;
pub mod trig;
